//==============================================================================
//  De Rosis & Enan, Phys. Fluids 33, 043315 (2021), Sec. III H and I -- the
//  Rayleigh-Taylor instability in two and three dimensions, against their
//  Tables VIII and IX and their Fig. 11.
//
//  This is the second group of their Sec. III: two immiscible fluids, both the
//  velocity and the order parameter solved. Unlike the interface-capture tests
//  of validation/enan_interface.cpp, a fluid collision operator IS exercised
//  here, and it is the central-moment one on D3Q27 throughout -- for the
//  two-dimensional cases too, run with a single periodic cell in z. The paper
//  uses D3Q19; M3LB's multiphase central-moment operator does not run on D3Q19
//  at all (it is D3Q27 minus its corners, so the moment basis is not a product
//  form), which is stated rather than worked around.
//
//  SETUP, theirs exactly. A box W wide and 4W tall, heavy fluid above light,
//  periodic across and no-slip top and bottom, with the interface given a
//  single-mode perturbation:
//
//      2-D, Eq. (81):  y_i = 2W + 0.10 W cos(2 pi x / W)
//      3-D, Eq. (83):  y_i = 2W + 0.05 W [cos(2 pi x / W) + cos(2 pi z / W)]
//
//  so the spike starts at y = 1.9 W in both, which is what their tables report
//  at t = 0 (1.900 and 1.898). Gravity is set by sqrt(g W) = 0.04, the
//  Reynolds number is Re = W sqrt(g W) / nu, the Atwood number At = 0.5 with
//  rho_L = 1, and time is reported in units of t0 = sqrt(W / (g At)).
//
//  The reported quantity is the paper's y-dagger: the vertical position of the
//  SPIKE -- the lowest point the heavy fluid has reached -- divided by W, so it
//  runs downward from 1.9.
//
//  WHAT THE PAPER DOES NOT STATE, and it matters. The interface width is given
//  nowhere for these cases. Sec. III's "unless otherwise stated we adopt
//  M = 0.001, n = 3" is scoped to the first group, where the mobility is set by
//  a Peclet number anyway; here Pe is given (500, 1000, 500 and 1024) but the
//  width that closes Pe = U xi / M is not. xi = 3 is used, and -iw sweeps it,
//  because a number quoted from an unstated parameter should come with the
//  sensitivity to it.
//
//  The rest follows from their groups: nu = W U / Re with U = sqrt(gW), the
//  surface tension from Ca = mu_H U / sigma, and the mobility from Pe.
//
//  HYDROSTATIC INITIALISATION, THROUGH THE DIFFUSE INTERFACE. The pressure is
//  seeded by integrating the ACTUAL density profile away from the interface
//  rather than assuming a sharp one; demonstrator/rayleigh_taylor.cpp's banner
//  records what assuming sharp costs at a high density ratio. At At = 0.5 the
//  imbalance would be small, but the closed form is free and the same code runs
//  at any At.
//
//  WHAT THIS TEST CAN AND CANNOT SETTLE. A nonlinear instability has no closed
//  form, so this is a comparison against other people's numbers and not a proof
//  of anything. The paper's Table IX puts eight models beside each other and
//  they spread by 30 % at t/t0 = 3 -- 0.648 to 0.863 -- so agreement inside that
//  spread says the scheme belongs in the same family, and nothing sharper. The
//  spread is printed alongside for exactly that reason.
//
//  RESOLUTION IS THE FIRST THING TO CHECK, AND W = 64 IS NOT ENOUGH HERE. On
//  their Table VIII (2-D, Re = 30000, Ca = 0.26) with only W varied, the worst
//  deviation over the seven tabulated instants is
//
//      W = 64    15.8 %
//      W = 128    4.9 %
//      W = 256    2.7 %
//
//  and in every case the error is LATE: the first five instants agree to about
//  1 % at all three resolutions and the whole deficit appears at t/t0 = 2.5 and
//  3, once the spike has rolled up and the structure that has to be resolved is
//  the filament rather than the interface. So the scheme converges, and a
//  disagreement at t/t0 = 3 on a coarse grid is the expected behaviour rather
//  than a defect to hunt.
//
//  SURFACE TENSION DOES NOTHING ON ITS OWN -- it only sharpens the resolution
//  requirement, and the first version of this note got that wrong. At W = 64,
//  Re = 30000, changing ONLY Ca from 0.26 to 960 -- sigma from 1.58e-4 to
//  1.25e-6 -- takes the worst deviation from 15.8 % to 29.3 %, which reads like
//  an independent penalty. It is not. The SAME change at W = 256 gives 1.9 %
//  against 2.7 %, i.e. very slightly BETTER. A nearly surface-tension-free
//  interface simply needs more cells to carry the filament; given them, it
//  costs nothing. Their 3-D
//  cases run at Ca = 960 AND W = 64, so they sit at the bad end of both knobs
//  at once, and their 92.8 % is not a single cause. It is also not fully
//  accounted for: the 2-D equivalent of those settings gives 29.3 %, so
//  something 3-D-specific remains, and it is left open rather than guessed at.
//
//  WHAT IT IS NOT is the spike measure, and that was worth ruling out rather
//  than assuming. "Position of the spike" is the tip of the coherent finger,
//  while a global minimum over phi > 0.5 is a different quantity as soon as
//  anything detaches -- and Ca = 960 is a surface tension of 1.25e-6, exactly
//  the regime where an interface fragments freely. In 3-D the global scan also
//  covers 64x more nodes per y-level than in 2-D, so one stray cell is far
//  likelier to capture it. All three measures below therefore run side by side,
//  and on the 3-D Re = 256 case they agree TO THE DIGIT at every one of the
//  seven tabulated instants, 0.0469 at t/t0 = 3 included. No fragmentation, no
//  stray cells: the finger really does reach the floor, and the deficit is
//  physics. The measures are kept because the hypothesis was reasonable and
//  the next person will have it too.
//==============================================================================
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/PhaseFieldCentralMoments.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/PhaseFieldSolver.hpp"
#include "solver/ViscousInterfaceForce.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

using FL = D3Q27;
using PL = D3Q27;
using FColl = MultiphaseCentralMoments<FL>;
// Central moments on BOTH lattices, which is what the paper uses and what was
// asked for. The BGK phase field is not offered as a switch here: it cannot
// carry the Peclet numbers this case runs at (validation/enan_interface
// measures where it stops), so a -op bgk row would be a blank rather than a
// comparison.
using PColl = PhaseFieldCentralMoments<PL>;

constexpr double PI = 3.14159265358979323846;

// Their four tabulated Rayleigh-Taylor cases. Table VIII is 2-D at Re = 30000;
// Table IX is 3-D at Re = 256, and is the one that carries the eight-model
// spread the text is measured against; Table X is 3-D at Re = 30000, which the
// paper's own bullet list of cases does not mention but which it tabulates.
// Re = 256 and Re = 3000 in 2-D have no table -- their Fig. 11 is a plot.
const double T_STAR[7]   = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
const double RT2D_30K[7] = {1.900, 1.829, 1.620, 1.365, 1.118, 0.863, 0.575};
const double RT3D_FD[7]  = {1.898, 1.858, 1.741, 1.553, 1.304, 1.001, 0.648};
const double RT3D_LO[7]  = {1.887, 1.839, 1.711, 1.504, 1.256, 0.988, 0.648};
const double RT3D_HI[7]  = {1.904, 1.897, 1.776, 1.618, 1.396, 1.149, 0.863};
const double RT3D_30K[7] = {1.898, 1.848, 1.680, 1.384, 0.964, 0.436, 0.000};

const char* arg_str(int argc, char** argv, const char* k, const char* d) {
  for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], k)) return argv[i + 1];
  return d;
}
double arg_num(int argc, char** argv, const char* k, double d) {
  const char* s = arg_str(argc, argv, k, nullptr);  return s ? std::atof(s) : d;
}
bool arg_flag(int argc, char** argv, const char* k) {
  for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], k)) return true;
  return false;
}

}  // namespace

//------------------------------------------------------------------------------
struct RT {
  std::vector<double> t_star, y_spike, y_bubble, umax;
  double nu, sigma, M, tau, rho_h, t_ref;
  std::size_t steps;
  bool finite;
};

static RT run(Index W, bool three_d, double At, double Re, double Ca, double Pe,
              double U, double iw, double tmax, const char* dump,
              const char* field) {
  const Index nx = W, ny = 4 * W, nz = three_d ? W : Index(1);
  const double g     = U * U / double(W);           // so sqrt(gW) = U
  const double nu    = double(W) * U / Re;
  const double rho_l = 1.0, rho_h = (1.0 + At) / (1.0 - At);
  // THREE THINGS HERE COME FROM THEIR DRIVER RATHER THAN FROM THE PAPER'S
  // PROSE, AND ALL THREE DIFFER FROM IT.
  //
  // 1. sigma = nu_H U / Ca, with NO rho_H, although Ca = mu_H U / sigma would
  //    put one there. At At = 0.5 that is a factor of three.
  // 2. M = U d / Pe with d the DOMAIN, not U xi / Pe. Same reading error as
  //    the interface cases; here it is a factor of W/xi = 13 to 51.
  // 3. THE REFERENCE TIME DIFFERS BETWEEN TWO AND THREE DIMENSIONS. Their 2-D
  //    drivers use sqrt(W / (g At)) and their 3-D ones use sqrt(W / g) with no
  //    Atwood factor at all. Using the 2-D form in 3-D stretches the clock by
  //    1/sqrt(At) = 1.41, so a spike reported at t/t0 = 3 has actually been
  //    run 41 % further than theirs -- which is most of why the 3-D case here
  //    was falling through the floor.
  const double sigma = nu * U / Ca;
  const double M     = U * double(W) / Pe;
  const double t_ref = three_d ? std::sqrt(double(W) / g)
                               : std::sqrt(double(W) / (g * At));
  const std::size_t nsteps = std::size_t(tmax * t_ref);

  Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  PColl pc;
  pc.omega = PColl::omega_from_mobility(Real(M));
  pc.width = Real(iw);
  PhaseFieldSolver<PL, EsotericPull<PL>, PColl> pf(d, pc);

  const Real half = Real(0.5) * Real(ny);
  const Real amp  = Real(three_d ? 0.05 : 0.10) * Real(W);
  const Real k    = Real(2.0 * PI) / Real(W), iwr = Real(iw);
  const Index hx = d.hx, hy = d.hy, hz = d.hz;
  const bool td = three_d;
  auto yi_of = KOKKOS_LAMBDA(Real x, Real z) {
    return half + amp * (Kokkos::cos(k * x) + (td ? Kokkos::cos(k * z) : Real(0)));
  };
  pf.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real x = Real(px - hx), y = Real(py - hy), z = Real(pz - hz);
    return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (y - yi_of(x, z)) / iwr));
  });

  ViscousInterfaceForce<FL> vf(d);
  FColl fc;
  fc.phi = pf.phi();
  fc.Gx = pf.grad_x(); fc.Gy = pf.grad_y(); fc.Gz = pf.grad_z();
  fc.Lap = pf.laplacian();
  fc.Vx = vf.x(); fc.Vy = vf.y(); fc.Vz = vf.z();
  fc.rho_L = Real(rho_l);      fc.rho_H = Real(rho_h);
  fc.mu_L  = Real(rho_l * nu); fc.mu_H  = Real(rho_h * nu);
  fc.kappa = FColl::kappa_from_sigma(Real(sigma), Real(iw));
  fc.beta  = FColl::beta_from_sigma(Real(sigma), Real(iw));
  fc.by    = Real(-g);
  FluidSolver<FL, EsotericPull<FL>, FColl> fl(d, fc);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });

  auto phiv = pf.phi();
  const Real rl = Real(rho_l), rh = Real(rho_h), gr = Real(g);
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real x = Real(px - hx), y = Real(py - hy), z = Real(pz - hz);
    const Real dz = y - yi_of(x, z);
    const Real az = (dz < Real(0) ? -dz : dz) * Real(2) / iwr;
    const Real lnch = az + Kokkos::log(Real(1) + Kokkos::exp(Real(-2) * az))
                    - Real(0.6931471805599453);
    const Real I = rl * dz + (rh - rl) * Real(0.5) * (dz + Real(0.5) * iwr * lnch);
    const Real r = rl + phiv(n) * (rh - rl);
    return FlowState{(-gr * I) / (r / Real(3)), Real(0), Real(0), Real(0)};
  });

  pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());

  RT r;
  r.nu = nu; r.sigma = sigma; r.M = M; r.rho_h = rho_h; r.t_ref = t_ref;
  r.steps = nsteps; r.finite = true;
  r.tau = nu * 3.0 + 0.5;

  // Sampled at the paper's own instants, so the table is a table of theirs.
  std::size_t next = 0;
  for (std::size_t step = 0; step <= nsteps; ++step) {
    if (next < 7 && step >= std::size_t(T_STAR[next] * t_ref)) {
      pf.compute_field();
      fl.compute_macroscopic();
      auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
      auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      // THREE spike measures, because the obvious one is not the published
      // one. "Position of the spike" in their Table IX is the tip of the
      // coherent falling finger; the global lowest node with phi > 0.5 is a
      // different quantity the moment anything detaches or a stray cell
      // overshoots, and it can only ever read LOWER. Measuring all three is
      // what distinguishes a physics error from a measurement artefact, and
      // that distinction cost a wrong conclusion once already.
      //
      //   spike      global minimum -- what this case reported before
      //   spike_ax   on the column where the initial perturbation is lowest,
      //              (W/2, W/2), which is where the finger forms
      //   spike_pop  lowest y whose cross-section holds at least 0.1 % of its
      //              nodes above phi = 0.5, so one stray cell cannot set it
      Index spike = ny - 1, bubble = 0, spike_ax = ny - 1, spike_pop = ny - 1;
      double um = 0;
      const Index z0 = three_d ? 0 : 0, z1 = three_d ? nz - 1 : 0;
      const Index xc = nx / 2, zc = three_d ? nz / 2 : 0;
      const Index plane = (z1 - z0 + 1) * nx;
      const Index need = std::max(Index(1), Index(plane / 1000));
      for (Index y = 0; y < ny; ++y) {
        Index cnt = 0;
        for (Index z = z0; z <= z1; ++z)
          for (Index x = 0; x < nx; ++x) {
            const Index n = d.id(x, y, z);
            const double p = double(hp(n));
            if (!std::isfinite(p)) r.finite = false;
            if (p > 0.5) { ++cnt; if (y < spike) spike = y; }
            if (p < 0.5 && y > bubble) bubble = y;
            um = std::max(um, std::hypot(double(hu(n)), double(hv(n))));
          }
        if (cnt >= need && y < spike_pop) spike_pop = y;
        if (double(hp(d.id(xc, y, zc))) > 0.5 && y < spike_ax) spike_ax = y;
      }
      // The order parameter itself, for the figures their Figs. 12, 13 and 15
      // show. Written to doc/fig as raw float32 with the two dimensions in
      // front, because doc/fig/*.bin is gitignored: a snapshot is regenerable
      // output, not tracked reference data, and seven of them on a 256 x 1024
      // grid would be 7 MB of it. In 3-D the plane at z = nz/2 is written --
      // the finger forms on the diagonal of the box, and that plane cuts
      // through it.
      if (field && *field) {
        char fp[256];
        std::snprintf(fp, sizeof fp, "doc/fig/rtphi_%s_t%d.bin", field, int(next));
        if (std::FILE* bf = std::fopen(fp, "wb")) {
          const std::int32_t dims[2] = {std::int32_t(nx), std::int32_t(ny)};
          std::fwrite(dims, sizeof(std::int32_t), 2, bf);
          std::vector<float> plane(std::size_t(nx) * std::size_t(ny));
          const Index zc = three_d ? nz / 2 : 0;
          for (Index yy = 0; yy < ny; ++yy)
            for (Index xx = 0; xx < nx; ++xx)
              plane[std::size_t(yy) * std::size_t(nx) + std::size_t(xx)] =
                  float(hp(d.id(xx, yy, zc)));
          std::fwrite(plane.data(), sizeof(float), plane.size(), bf);
          std::fclose(bf);
        }
      }
      r.t_star.push_back(T_STAR[next]);
      r.y_spike.push_back(double(spike) / double(W));
      r.y_bubble.push_back(double(bubble) / double(W));
      r.umax.push_back(um);
      std::printf("    t/t0 = %-5.2f  step %-8zu  y+ spike %.4f (axis %.4f, "
                  "pop %.4f)  bubble %.4f  |u|max %.4e\n", T_STAR[next], step,
                  double(spike) / double(W), double(spike_ax) / double(W),
                  double(spike_pop) / double(W),
                  double(bubble) / double(W), um);
      std::fflush(stdout);
      ++next;
      if (!r.finite) break;
    }
    fl.step(true);
    pf.refresh();
    pf.step();
  }

  if (dump && *dump) {
    pf.compute_field();
    auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
    const std::string p = std::string("results/L_enan/") + dump + ".dat";
    std::FILE* f = std::fopen(p.c_str(), "w");
    if (f) {
      std::fprintf(f, "# RT %s W=%d Re=%g At=%g Ca=%g Pe=%g xi=%g nu=%.4e "
                      "sigma=%.4e M=%.4e t_ref=%.1f\n",
                   three_d ? "3D" : "2D", int(W), Re, At, Ca, Pe, iw, nu, sigma,
                   M, t_ref);
      std::fprintf(f, "# phi on the mid-z plane: x y phi\n");
      const Index zc = nz / 2;
      for (Index y = 0; y < ny; ++y)
        for (Index x = 0; x < nx; ++x)
          std::fprintf(f, "%d %d %.5f\n", int(x), int(y),
                       double(hp(d.id(x, y, zc))));
      std::fclose(f);
    }
  }
  return r;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  const bool   td   = arg_flag(argc, argv, "-3d");
  const Index  W    = Index(arg_num(argc, argv, "-w", td ? 64 : 256));
  const double Re   = arg_num(argc, argv, "-re", 256.0);
  const double At   = arg_num(argc, argv, "-at", 0.5);
  const double Ca   = arg_num(argc, argv, "-ca", td ? 960.0 : 0.26);
  const double Pe   = arg_num(argc, argv, "-pe", td ? 1024.0 : 500.0);
  const double U    = arg_num(argc, argv, "-u", 0.04);
  const double iw   = arg_num(argc, argv, "-iw", 5.0);   // THEIR value
  const double tmax = arg_num(argc, argv, "-tmax", 3.0);
  const bool   dump = arg_flag(argc, argv, "-dump");
  // Order-parameter snapshots for the figures, into doc/fig as raw float32.
  const bool   fieldflag = arg_flag(argc, argv, "-field");

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::printf("Rayleigh-Taylor instability vs De Rosis & Enan (2021) "
                "Sec. III %s\n", td ? "I (3-D, their Table IX)"
                                    : "H (2-D, their Table VIII / Fig. 11)");
    std::printf("D3Q27 multiphase central moments + D3Q27 conservative "
                "Allen-Cahn phase field\n");
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("  %d x %d%s   Re = %g   At = %g   Ca = %g   Pe = %g   xi = %g\n",
                int(W), int(4 * W), td ? (" x " + std::to_string(int(W))).c_str() : "",
                Re, At, Ca, Pe, iw);
    std::printf("  the interface width is NOT given by the paper; xi = %g is a "
                "choice, -iw sweeps it\n\n", iw);

    char tag[128];
    std::snprintf(tag, sizeof tag, "rt%s_w%d_re%.0f_iw%g",
                  td ? "3d" : "2d", int(W), Re, iw);
    const RT r = run(W, td, At, Re, Ca, Pe, U, iw, tmax, dump ? tag : "",
                     fieldflag ? tag : "");

    std::printf("\n  nu = %.4e   tau = %.5f   sigma = %.4e   M = %.4e   "
                "rho_H = %.2f\n", r.nu, r.tau, r.sigma, r.M, r.rho_h);
    std::printf("  t0 = %.1f steps, %zu steps to t/t0 = %g%s\n\n",
                r.t_ref, r.steps, tmax, r.finite ? "" : "   DIVERGED");

    // Tabulated: 2-D at Re = 30000 (VIII), 3-D at Re = 256 (IX) and 3-D at
    // Re = 30000 (X). The eight-model spread exists only for Table IX.
    const bool is256 = std::abs(Re - 256.0)   < 1e-9;
    const bool is30k = std::abs(Re - 30000.0) < 1e-9;
    const bool have_ref = (td && (is256 || is30k)) || (!td && is30k);
    const bool spread   = td && is256;
    if (have_ref) {
      const double* ref = td ? (is256 ? RT3D_FD : RT3D_30K) : RT2D_30K;
      std::printf("  %-8s %-11s %-13s %-9s%s\n", "t/t0", "y+ (here)",
                  td ? (is256 ? "y+ (Table IX)" : "y+ (Table X)")
                     : "y+ (Table VIII)", "dev",
                  spread ? "   eight-model spread" : "");
      std::printf("  %s\n", std::string(spread ? 74 : 46, '-').c_str());
      double worst = 0;
      for (std::size_t i = 0; i < r.t_star.size(); ++i) {
        // Table X ends at exactly 0.000, so a relative deviation there is
        // undefined rather than infinite. Report the absolute gap in that one
        // slot and keep it out of the worst-case, which is a relative measure.
        const bool rel = std::abs(ref[i]) > 1e-9;
        const double dev = rel ? 100.0 * (r.y_spike[i] / ref[i] - 1.0)
                               : (r.y_spike[i] - ref[i]);
        if (rel) worst = std::max(worst, std::abs(dev));
        if (rel)
          std::printf("  %-8.1f %-11.4f %-13.3f %+8.1f%%", r.t_star[i],
                      r.y_spike[i], ref[i], dev);
        else
          std::printf("  %-8.1f %-11.4f %-13.3f %+7.4f ", r.t_star[i],
                      r.y_spike[i], ref[i], dev);
        if (spread) {
          const bool in = r.y_spike[i] >= RT3D_LO[i] - 1e-9 &&
                          r.y_spike[i] <= RT3D_HI[i] + 1e-9;
          std::printf("   [%.3f, %.3f]  %s", RT3D_LO[i], RT3D_HI[i],
                      in ? "inside" : "OUTSIDE");
        }
        std::printf("\n");
      }
      std::printf("\n  worst deviation %.1f%%.%s\n", worst,
                  spread ? "  The spread column is the range across the eight models\n"
                       "  their Table IX tabulates; landing inside it is the "
                       "honest claim here,\n  because a nonlinear instability has "
                       "no closed form to be right against."
                     : "");
      if (!r.finite) status = 1;
    } else {
      std::printf("  Re = %g has no published table in the paper -- its Fig. 11\n"
                  "  plots the spike history for Re = 256 and 3000 against four\n"
                  "  other studies, so this run is for that figure and is\n"
                  "  reported as a trajectory rather than scored.\n", Re);
      std::printf("\n  %-8s %-11s %-11s %-11s\n", "t/t0", "y+ spike", "y+ bubble",
                  "|u|max");
      std::printf("  %s\n", std::string(46, '-').c_str());
      for (std::size_t i = 0; i < r.t_star.size(); ++i)
        std::printf("  %-8.1f %-11.4f %-11.4f %-11.4e\n", r.t_star[i],
                    r.y_spike[i], r.y_bubble[i], r.umax[i]);
      if (!r.finite) status = 1;
    }

    // Tracked row, one per (dim, Re, xi).
    {
      const std::string path = "results/L_enan/enan_rt.dat";
      bool fresh = true;
      if (std::FILE* t = std::fopen(path.c_str(), "r")) { fresh = false; std::fclose(t); }
      std::FILE* f = std::fopen(path.c_str(), "a");
      if (f) {
        if (fresh)
          std::fputs("# Rayleigh-Taylor vs De Rosis & Enan (2021) Sec. III H and I.\n"
                     "# D3Q27 multiphase central moments + D3Q27 Allen-Cahn phase\n"
                     "# field, Esoteric Pull, FP64. 2-D runs use one periodic cell\n"
                     "# in z. y+ is the spike position over W, from 1.9 downward.\n"
                     "# The interface width xi is NOT specified by the paper.\n"
                     "# Their Table VIII (2-D, Re=30000): "
                     "1.900 1.829 1.620 1.365 1.118 0.863 0.575\n"
                     "# Their Table IX  (3-D, Re=256, FD): "
                     "1.898 1.858 1.741 1.553 1.304 1.001 0.648\n"
                     "# dim W Re At Ca Pe xi nu tau sigma M t_ref steps finite "
                     "y0 y0.5 y1 y1.5 y2 y2.5 y3\n", f);
        std::fprintf(f, "%d %d %g %g %g %g %g %.6e %.6f %.6e %.6e %.1f %zu %d",
                     td ? 3 : 2, int(W), Re, At, Ca, Pe, iw, r.nu, r.tau,
                     r.sigma, r.M, r.t_ref, r.steps, int(r.finite));
        for (std::size_t i = 0; i < 7; ++i)
          std::fprintf(f, " %.4f", i < r.y_spike.size() ? r.y_spike[i] : NAN);
        std::fputc('\n', f);
        std::fclose(f);
      }
    }
  }
  Kokkos::finalize();
  return status;
}
