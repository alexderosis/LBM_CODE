//==============================================================================
//  Cubic box with no-slip walls -- THROUGHPUT.
//
//  The simplest configuration this code can be asked to run, and the one meant
//  for comparing MLUPS against other GPU lattice Boltzmann codes: a cube, a
//  one-cell solid shell on all six faces, no forcing, no coupled fields.
//
//  THE FLUID STARTS AT REST (-u0 0, the default), which is what to use for a
//  throughput number. The rate is not affected by it: every node reads 27
//  populations, does the same arithmetic on them and writes 27 back, whatever
//  the values are. There is no branch on a population and no data-dependent
//  path anywhere in the kernel, and at equilibrium the values are O(1e-2), so
//  no denormals either. A quiescent box and a stirred one time the same.
//
//  What a quiescent box does NOT do is prove the solver works: mass is
//  conserved trivially because nothing moves. `-u0 0.05` starts a decaying
//  vortex instead, and then the mass-drift column is a real check on the same
//  kernel at the same speed. Worth one run before quoting the rest of them.
//
//  At rest in FP32 the drift is not zero -- it is about 7e-7 and it is CONSTANT:
//  measured 7.252e-07, 7.240e-07, 7.240e-07 at 40, 400 and 4000 steps on 48^3,
//  and exactly 0.000e+00 in an FP64 build. The initial populations are w_i, and
//  their FP32 sum is 1 + O(eps) rather than 1, so the first collision moves the
//  field onto the nearby FP32 fixed point and it then stays there. A leak would
//  grow with step count; this does not, which is how the two are told apart.
//
//  WALLS. A solid cell is skipped, and under Esoteric Pull that IS halfway
//  bounce-back -- see the long note in solver.cuh. The wall plane sits midway
//  between the last fluid node and the first solid one; the parent code's
//  Poiseuille check confirms the placement, err x H^2 holding at 0.333 across
//  H = 16/32/64. No-slip, second-order for a grid-aligned flat wall, one byte
//  per node and no extra kernel.
//
//  READ THIS BEFORE COMPARING THE NUMBER TO ANOTHER CODE.
//
//  MLUPS is nodes x steps / second. It says nothing about how many bytes a node
//  costs, and a lattice Boltzmann step is bandwidth-bound almost everywhere, so
//  MLUPS is very nearly (achieved bandwidth) / (bytes per node). Two codes can
//  differ by a factor of two in MLUPS while moving memory equally well:
//
//      this code        D3Q27, FP32 storage      108 B/node   (+1 for geometry)
//      D3Q19, FP32                                76 B/node
//      D3Q19, FP16 storage                        38 B/node
//
//  D3Q27 is 42% more traffic per node than D3Q19 and 2.8x an FP16 D3Q19 code.
//  Reporting MLUPS alone against such a code understates this one by exactly
//  that factor, and reporting it in this code's favour against a D3Q39 would
//  overstate it. So the table below prints GB/s and the percentage of the
//  device's theoretical pin bandwidth beside every MLUPS figure. THAT is the
//  portable number: it is what fraction of the machine the implementation
//  actually gets, and it is comparable across velocity sets and precisions.
//
//  Traffic counts 27 reads + 27 writes per node per step. Esoteric Pull's
//  advantage is FOOTPRINT -- one lattice in memory instead of two -- not
//  traffic, and counting it as half the bytes flatters it.
//
//    usage: box [-n N] [-steps S] [-op bgk|cm|both] [-tau T] [-u0 U] [-periodic]
//           -u0 defaults to 0 (at rest). -u0 0.05 starts the vortex.
//
//  With no -n the sweep is 520, 720, 1390. A size whose lattice does not fit
//  the device is reported and skipped rather than dying inside cudaMalloc.
//==============================================================================
#include "lbm/backend.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
// Initial condition: a divergence-free vortex that vanishes on all six walls.
//
// Built as u = curl A with A = (phi, phi, phi) and
//
//     phi = C sin^2(k dx) sin^2(k dy) sin^2(k dz),   k = pi/(n-2)
//
// where d = coordinate measured from the wall PLANE at 0.5, not from the solid
// node at 0. Halfway bounce-back places the wall midway between the last fluid
// node and the first solid one, so this is where the no-slip condition actually
// lives; anchoring the sine at the solid node instead would start the flow with
// a half-cell velocity jump at every face.
//
// Two properties, both wanted and neither accidental:
//   * div u = div curl A = 0 identically, so the run does not open with a
//     spurious compression wave on top of the acoustic one it already has;
//   * the SQUARE of the sine makes phi and all its first derivatives vanish on
//     each face, so all THREE velocity components are zero there, not just the
//     normal one. A plain sin^1 potential satisfies no-penetration but leaves a
//     tangential slip the walls would then have to kill in the first few steps.
//
// A struct rather than a lambda: nvcc will not take a function-local type as a
// template argument for a kernel. See src/tgv3d.cu.
//------------------------------------------------------------------------------
struct BoxVortexInit {
  Real k, amp;
  LBM_HD Macro operator()(int x, int y, int z) const {
    const Real X = k * (Real(x) - Real(0.5));
    const Real Y = k * (Real(y) - Real(0.5));
    const Real Z = k * (Real(z) - Real(0.5));
    const Real sx = sin(X), cx = cos(X);
    const Real sy = sin(Y), cy = cos(Y);
    const Real sz = sin(Z), cz = cos(Z);
    Macro m;
    // Uniform density. There is no closed-form pressure compatible with this
    // field, so the run opens with a small acoustic transient; at Ma ~ 0.03 it
    // is far below anything the timing or the mass check can see.
    m.rho = Real(1);
    m.ux = amp * sx * sx * (sy * cy * sz * sz - sy * sy * sz * cz);
    m.uy = amp * sy * sy * (sz * cz * sx * sx - sz * sz * sx * cx);
    m.uz = amp * sz * sz * (sx * cx * sy * sy - sx * sx * sy * cy);
    return m;
  }
};

//------------------------------------------------------------------------------
// A one-cell solid shell on all six faces. Everything else is fluid.
//------------------------------------------------------------------------------
static std::vector<std::uint8_t> box_walls(int n) {
  std::vector<std::uint8_t> fl(std::size_t(n) * n * n, Fluid);
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        if (x == 0 || x == n - 1 || y == 0 || y == n - 1 || z == 0 || z == n - 1)
          fl[std::size_t(node_id(x, y, z, n, n))] = Solid;
  return fl;
}

struct Result { double mlups, gbs, drift; bool ok; };

static Result run(int n, int steps, Op op, Real tau, Real u0, bool walls) {
  const Real nu = Real((double(tau) - 0.5) / 3.0);
  backend::Fluid s(n, n, n, op, nu);

  if (walls) s.set_geometry(box_walls(n));       // before initialise_with, so
                                                 // solid cells seed at rest
  s.initialise_with(BoxVortexInit{
      Real(M_PI) / Real(n - 2), u0});

  const double m0 = s.total_mass();

  s.step();                                      // warm-up, outside the timing
  backend::sync();

  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < steps; ++t) s.step();
  backend::sync();
  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();

  const double lups  = double(s.nodes()) * double(steps);
  const double bytes = lups * 27.0 * 2.0 * double(sizeof(Real));
  const double m1    = s.total_mass();
  const double drift = std::abs(m1 - m0) / m0;

  return {lups / sec / 1e6, bytes / sec / 1e9, drift,
          std::isfinite(m1) && drift < (sizeof(Real) == 4 ? 1e-4 : 1e-11)};
}

int main(int argc, char** argv) {
  int n = 0, steps = 100;
  Real tau = Real(0.8), u0 = Real(0);      // at rest: throughput measurement
  std::string op = "both";
  bool walls = true;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-n"        && i + 1 < argc) n     = std::atoi(argv[++i]);
    if (a == "-steps"    && i + 1 < argc) steps = std::atoi(argv[++i]);
    if (a == "-op"       && i + 1 < argc) op    = argv[++i];
    if (a == "-tau"      && i + 1 < argc) tau   = Real(std::atof(argv[++i]));
    if (a == "-u0"       && i + 1 < argc) u0    = Real(std::atof(argv[++i]));
    if (a == "-periodic")                 walls = false;
  }

  const backend::DeviceInfo dev = backend::device_info();
  const double per_node = 27.0 * double(sizeof(Real)) + (walls ? 1.0 : 0.0);

  std::printf("cubic box%s   D3Q27   %s   %s\n",
              walls ? " with no-slip walls" : " (periodic, no geometry)",
              sizeof(Real) == 4 ? "FP32" : "FP64",
              backend::on_device ? "CUDA native" : "HOST reference");
  std::printf("  device      %s\n", dev.name.c_str());
  if (dev.on_gpu)
    std::printf("  memory      %.1f GB total, %.1f GB free   peak %.0f GB/s\n",
                dev.total_gb, dev.free_gb, dev.peak_gbs);
  std::printf("  per node    %.0f B  (27 populations%s)\n",
              per_node, walls ? " + 1 geometry byte" : "");
  if (double(u0) == 0.0)
    std::printf("  tau = %.3f   nu = %.6f   AT REST   %d steps\n"
                "              (throughput only -- mass is conserved trivially. Use\n"
                "               -u0 0.05 for a decaying vortex that actually checks it.)\n\n",
                double(tau), (double(tau) - 0.5) / 3.0, steps);
  else
    std::printf("  tau = %.3f   nu = %.6f   decaying vortex, |u| <= %.4f, "
                "Ma <= %.4f   %d steps\n\n",
                double(tau), (double(tau) - 0.5) / 3.0, double(u0),
                double(u0) * std::sqrt(3.0), steps);

  // The device sweep is 520/720/1390. A host build has no cudaMalloc to fail
  // gracefully and no way to ask how much RAM it may have, and the smallest of
  // those three would ask for 15 GB, so it gets a sweep it can actually finish.
  // An explicit -n always wins, on either backend.
  if (!n && !dev.on_gpu)
    std::printf("  (host build: using a small sweep. The device sweep is "
                "520/720/1390 -- pass -n to force one.)\n\n");
  const std::vector<int> sizes =
      n            ? std::vector<int>{n}
    : dev.on_gpu   ? std::vector<int>{520, 720, 1390}
                   : std::vector<int>{32, 48, 64};

  std::printf("  %-16s %7s %10s %12s %11s %8s %13s\n",
              "operator", "grid", "GB needed", "MLUPS", "GB/s", "% peak", "mass drift");
  std::printf("  %s\n", std::string(85, '-').c_str());

  std::vector<std::pair<Op, const char*>> ops;
  if (op != "cm")  ops.emplace_back(Op::BGK, "BGK");
  if (op != "bgk") ops.emplace_back(Op::CentralMoments, "central moments");

  bool any_skipped = false;
  for (auto [which, name] : ops) {
    for (int sz : sizes) {
      const double need = double(sz) * sz * sz * per_node / 1e9;

      // A margin, not a bare comparison: the driver keeps a working set of its
      // own, and total_mass allocates a small partial-sum buffer.
      if (dev.on_gpu && need > 0.97 * dev.free_gb) {
        std::printf("  %-16s %5d^3 %10.1f   DOES NOT FIT -- needs %.0f GB, %.0f GB free\n",
                    name, sz, need, need, dev.free_gb);
        any_skipped = true;
        continue;
      }
      const Result r = run(sz, steps, which, tau, u0, walls);
      char pk[16];
      if (dev.peak_gbs > 0) std::snprintf(pk, sizeof pk, "%.1f", 100.0 * r.gbs / dev.peak_gbs);
      else                  std::snprintf(pk, sizeof pk, "%s", "n/a");
      std::printf("  %-16s %5d^3 %10.1f %12.1f %11.1f %8s %13.3e%s\n",
                  name, sz, need, r.mlups, r.gbs, pk,
                  r.drift, r.ok ? "" : "   <-- NOT CONSERVED");
      std::fflush(stdout);
    }
    std::printf("\n");
  }

  if (any_skipped) {
    std::printf("  A skipped size needs more memory than this device has. The lattice is\n"
                "  %.0f B/node and there is exactly one of it; nothing in this code streams\n"
                "  from host memory or splits across devices, so the largest cube that runs\n"
                "  is fixed by capacity alone:  n_max = (0.97 * free / %.0f B)^(1/3).\n",
                per_node, per_node);
    if (dev.on_gpu)
      std::printf("  On this device that is %d^3.\n",
                  int(std::cbrt(0.97 * dev.free_gb * 1e9 / per_node)));
  }

  if (walls)
    std::printf("\n  MLUPS counts every node the kernel launches, n^3, the one-cell solid\n"
                "  shell included -- 6/n of the volume, 1.2%% at 520^3. A solid node returns\n"
                "  immediately and so is cheaper than a fluid one. Counting only fluid nodes\n"
                "  would move the figure by less than that; -periodic removes the question.\n");

  std::printf("\n  MLUPS alone is not comparable across velocity sets: it is very nearly\n"
              "  achieved bandwidth divided by bytes per node, and D3Q27 FP32 moves 108 B\n"
              "  where D3Q19 FP32 moves 76 and a D3Q19 FP16 code moves 38. Compare the\n"
              "  '%% peak' column, which is the fraction of the machine actually used.\n");
  return 0;
}
