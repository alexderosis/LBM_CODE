//==============================================================================
//  THE NEW PATHS, ON THE DEVICE, AGAINST WHAT THE HOST ALREADY SAID.
//
//  TRT, shifted storage, the D3Q27 central-moment phase field, the penalised
//  body and the free surface were all written and measured with no GPU, using
//  the reference driver in hostsim.hpp. That verifies the arithmetic, which is
//  most of it. What it CANNOT verify is the launch: grid configuration, register
//  pressure, whether a kernel boundary really fences, and -- for the body -- a
//  seven-accumulator block reduction that has no serial counterpart at all.
//
//  So this driver runs one measurement per new path and prints it beside the
//  number the host predicted. Anything that disagrees is device-side by
//  elimination, which is the whole point of writing it this way round.
//
//  It is deliberately NOT a benchmark and not a validation case: each block is
//  the cheapest thing that would catch a kernel that did not launch, a
//  reduction that summed the wrong thing, or a fence that was not there.
//
//  Build: it is in LBM_APPS, so `cmake --build build` makes it.
//  Run:   ./newpaths
//==============================================================================
#include "lbm/backend.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int bad = 0;

//------------------------------------------------------------------------------
// INITIALISERS AS STRUCTS, NOT LAMBDAS, and that is not a style preference: an
// LBM_HD lambda is a __host__ __device__ lambda under nvcc and needs
// --extended-lambda, which this project does not pass. Every driver here does
// the same. It compiles either way on the host, so the host build will NOT
// catch the omission -- which is exactly the class of thing this driver exists
// to find, arriving one level up.
//------------------------------------------------------------------------------
struct RestInit {
  LBM_HD Macro operator()(int, int, int) const {
    return Macro{Real(1), Real(0), Real(0), Real(0), Real(1)};
  }
};

struct ShearInit {
  Real amp;  Real k;
  LBM_HD Macro operator()(int, int y, int) const {
    Macro m;
    m.rho = Real(1);
    m.ux = amp * Real(sinf(float(k) * float(y)));
    return m;
  }
};

struct SlabInit {
  Real W, xa, xb;
  LBM_HD void operator()(int x, int, int, Real& ph, Real& pt) const {
    ph = Real(0.5f * (tanhf(2.0f * (float(x) - float(xa)) / float(W))
                    - tanhf(2.0f * (float(x) - float(xb)) / float(W))));
    pt = Real(0);
  }
};

struct PoolInit {
  Real surface, g, cs2;
  LBM_HD FsSeed operator()(int, int y, int) const {
    FsSeed s;
    const float f = float(surface) + 0.5f - float(y);
    s.fill = Real(f > 1.0f ? 1.0f : (f < 0.0f ? 0.0f : f));
    const float h = float(surface) - float(y);
    s.rho = Real(1.0f + (h > 0.0f ? float(g) * h / float(cs2) : 0.0f));
    return s;
  }
};

static void row(const char* what, double device, double host, double tol) {
  const double d = std::fabs(device - host);
  const double rel = (host != 0.0) ? d / std::fabs(host) : d;
  const bool ok = rel <= tol;
  std::printf("  %-46s %14.6g %14.6g   %8.1e  %s\n",
              what, device, host, rel, ok ? "ok" : "DIFFERS");
  if (!ok) ++bad;
}

//------------------------------------------------------------------------------
// 1. TRT. The wall position at tau = 2, where BGK puts it at 0.33 and TRT at
//    Lambda = 3/16 puts it at 0.5. Fit a quadratic through the profile and take
//    its lower root -- the same measurement host_physics case 1b makes.
//------------------------------------------------------------------------------
static double wall_root(const std::vector<double>& u, int H) {
  double S[5] = {0, 0, 0, 0, 0}, T[3] = {0, 0, 0};
  for (int y = 1; y <= H; ++y) {
    double pw = 1.0;
    for (int k = 0; k < 5; ++k) { S[k] += pw; pw *= double(y); }
    pw = 1.0;
    for (int k = 0; k < 3; ++k) { T[k] += u[std::size_t(y - 1)] * pw; pw *= double(y); }
  }
  double M[3][4] = {{S[0], S[1], S[2], T[0]},
                    {S[1], S[2], S[3], T[1]},
                    {S[2], S[3], S[4], T[2]}};
  for (int i = 0; i < 3; ++i) {
    int piv = i;
    for (int r = i + 1; r < 3; ++r)
      if (std::fabs(M[r][i]) > std::fabs(M[piv][i])) piv = r;
    for (int k = 0; k < 4; ++k) { const double t = M[i][k]; M[i][k] = M[piv][k]; M[piv][k] = t; }
    for (int r = 0; r < 3; ++r)
      if (r != i) {
        const double f = M[r][i] / M[i][i];
        for (int k = i; k < 4; ++k) M[r][k] -= f * M[i][k];
      }
  }
  const double a = M[0][3] / M[0][0], b = M[1][3] / M[1][1], c = M[2][3] / M[2][2];
  return (-b + std::sqrt(b * b - 4.0 * c * a)) / (2.0 * c);
}

static double trt_wall(Op op) {
  const int H = 16, nx = 4, nz = 4, ny = H + 2;
  const double nu = 0.5, umax = 0.002;
  const double G = 8.0 * nu * umax / (double(H) * double(H));

  backend::Fluid fl(nx, ny, nz, op, Real(nu));
  if (op == Op::TRT) fl.set_magic(Real(3.0 / 16.0));
  std::vector<std::uint8_t> flags(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      flags[std::size_t(node_id(x, 0, z, nx, ny))] = Solid;
      flags[std::size_t(node_id(x, ny - 1, z, nx, ny))] = Solid;
    }
  fl.set_geometry(flags);
  BodyForce b;  b.fx = Real(G);
  fl.set_force(b, ForceUniform);
  fl.initialise_with(RestInit{});
  for (int t = 0; t < 40000; ++t) fl.step();

  std::vector<Real> rho, ux, uy, uz;
  fl.macroscopic_to_host(rho, ux, uy, uz);
  std::vector<double> prof(std::size_t(H), 0.0);
  for (int y = 1; y <= H; ++y) {
    double u = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) u += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
    prof[std::size_t(y - 1)] = u / (double(nx) * nz);
  }
  return wall_root(prof, H);
}

//------------------------------------------------------------------------------
// 2. Shifted storage. A shear wave at A = 1e-5, where raw FP32 loses the decay
//    entirely and shifted does not.
//------------------------------------------------------------------------------
static double shear_decay_error(bool shifted, double A) {
  const int nx = 4, ny = 32, nz = 4;
  const double nu = 0.02, k = 2.0 * M_PI / ny;
  const std::size_t T = 2000;

  backend::Fluid fl(nx, ny, nz, Op::BGK, Real(nu));
  fl.set_shifted(shifted);
  fl.initialise_with(ShearInit{Real(A), Real(k)});

  auto fit = [&]() {
    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    double S = 0;
    for (int y = 0; y < ny; ++y) {
      double p = 0;
      for (int z = 0; z < nz; ++z)
        for (int x = 0; x < nx; ++x) p += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
      S += (p / (double(nx) * nz)) * std::sin(k * y);
    }
    return 2.0 * S / ny;
  };

  const double a0 = fit();
  for (std::size_t t = 0; t < T; ++t) fl.step();
  const double a1 = fit();
  return (-std::log(a1 / a0) / double(T) - nu * k * k) / (nu * k * k);
}

//------------------------------------------------------------------------------
// 3. The D3Q27 central-moment phase field. A flat slab must hold its width --
//    host_phasefield section 8, which measures W 4.33 -> 4.45 over 600 steps for
//    both operators.
//------------------------------------------------------------------------------
static double flat_width(PhaseOp op) {
  const int nx = 64, ny = 6, nz = 6;
  const double W = 4.0, xa = 20.0, xb = 44.0;

  backend::PhaseFieldOn<D3Q27> pf(nx, ny, nz);
  pf.phase.width = Real(W);
  pf.set_mobility(Real(0.05));
  pf.set_phase_op(op);
  pf.fluid.rho_L = Real(1);   pf.fluid.rho_H = Real(1);
  pf.fluid.mu_L  = Real(0.1); pf.fluid.mu_H  = Real(0.1);
  pf.fluid.beta  = Real(0);   pf.fluid.kappa = Real(0);
  pf.initialise_with(SlabInit{Real(W), Real(xa), Real(xb)});
  for (int t = 0; t < 600; ++t) pf.step();

  std::vector<Real> phi;
  pf.field_to_host(pf.phi_device(), phi);
  double dmax = 0;
  for (int x = 10; x < 32; ++x) {
    const long a = node_id(x - 1, ny / 2, nz / 2, nx, ny);
    const long b = node_id(x + 1, ny / 2, nz / 2, nx, ny);
    dmax = std::fmax(dmax, std::fabs(0.5 * (double(phi[std::size_t(b)])
                                          - double(phi[std::size_t(a)]))));
  }
  return 1.0 / dmax;
}

int main() {
  const backend::DeviceInfo dev = backend::device_info();
  std::printf("The new paths, on the device.  %s, %s\n",
              dev.name.c_str(), sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("%-48s %14s %14s   %8s\n\n", "  measurement", "device", "host said", "rel");

  //---- 1. TRT -----------------------------------------------------------------
  row("TRT: bounce-back wall at tau = 2", trt_wall(Op::TRT), 0.500175, 5e-3);
  row("  ... and BGK's, for contrast",    trt_wall(Op::BGK), 0.329796, 5e-3);

  //---- 2. shifted storage -----------------------------------------------------
  row("shifted: shear decay error at A = 1e-2", shear_decay_error(true, 1e-2),
      5.217e-3, 5e-2);
  row("shifted: and at A = 1e-5 (raw loses it)", shear_decay_error(true, 1e-5),
      5.213e-3, 5e-2);
  row("raw: the same, at A = 1e-5",             shear_decay_error(false, 1e-5),
      -0.967929, 5e-2);

  //---- 3. the D3Q27 central-moment phase field --------------------------------
  row("phase field D3Q27 BGK: slab width", flat_width(PhaseOp::BGK), 4.45, 2e-2);
  row("phase field D3Q27 CM:  slab width", flat_width(PhaseOp::CentralMoments),
      4.45, 2e-2);

  //---- 4. the penalised body --------------------------------------------------
  //
  // THE ONE PIECE WITH NO SERIAL COUNTERPART: seven per-block partial sums,
  // added on the host. A reduction that dropped a block, or summed a slot into
  // the wrong accumulator, would show here and nowhere else -- the closed-form
  // reaction below is derived from those seven numbers, so it is exactly what
  // the reduction produced.
  {
    const int n = 96;
    const double h = 12.0, g = 1e-4;
    backend::Body<Rect> body(n, n, 1);
    body.shape.hx = Real(h);  body.shape.hy = Real(h);
    body.shape.smooth = Real(1.5);
    body.shape.cx = Real(n / 2);  body.shape.cy = Real(n / 2);
    body.shape.set_angle(Real(0));
    body.set_uniform_density(Real(10));
    body.props.by = Real(-g);
    body.props.free_rotation = false;

    // A fluid at rest supplies the (zero) velocity field. That is a roundabout
    // way to get a zeroed array, and it is the RIGHT one here: cudaMalloc would
    // make this driver CUDA-only, and the whole point of writing against
    // backend:: is that one source builds and runs both ways -- so the host can
    // predict what the device is about to print.
    backend::Fluid still(n, n, 1, Op::BGK, Real(0.1));
    still.enable_velocity_output();
    still.initialise_with(RestInit{});
    body.couple_velocity(still.ux_device(), still.uy_device());
    const BodyReaction r = body.refresh(UniformDensity{Real(1)});

    // dU = (m_b - m_f) g / (m_b + m_f) = -(10-1)/(10+1) g.
    row("body: dU from the 7-accumulator reduction", double(body.vy),
        -(10.0 - 1.0) / (10.0 + 1.0) * g, 1e-4);
    // m_f is the integral of chi rho: the indicator's own area at rho = 1.
    row("body: fictitious mass = area of chi", r.fluid_mass,
        double(body.indicator_moments().area), 1e-4);
  }

  //---- 5. the free surface ----------------------------------------------------
  //
  // Two-lattice, five kernels, and the mass ledger is the property that can be
  // silently lost. host_freesurface measures a quiescent pool holding its mass
  // to 1.3e-6 over 4000 steps with the interface unchanged.
  {
    const int nx = 48, ny = 48, nz = 4;
    const double nu = 0.01, g = 1e-5, surface = 24.0, cs2 = 1.0 / 3.0;
    backend::FreeSurface fs(nx, ny, nz, Real(nu));
    fs.set_gravity(Real(0), Real(-g));

    std::vector<std::uint8_t> fl(std::size_t(nx) * ny * nz, std::uint8_t(FsGas));
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
          fl[std::size_t(node_id(x, y, z, nx, ny))] =
              (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) ? FsSolid : FsGas;
    fs.set_geometry(fl);
    fs.initialise_with(PoolInit{Real(surface), Real(g), Real(cs2)});

    const double m0 = fs.total_mass();
    for (int t = 0; t < 4000; ++t) fs.step();
    const double m1 = fs.total_mass();

    std::vector<std::uint8_t> f;
    fs.flags_to_host(f);
    long open = 0, interf = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          const long id = node_id(x, y, z, nx, ny);
          if (f[std::size_t(id)] == FsInterface) ++interf;
          if (f[std::size_t(id)] != FsFluid) continue;
          for (int i = 1; i < 27; ++i)
            if (f[std::size_t(neighbour<D3Q27>(x, y, z, i, nx, ny, nz))] == FsGas) ++open;
        }
    std::printf("  %-46s %14.6g %14s\n", "free surface: mass drift over 4000 steps",
                (m1 - m0) / m0, "~1e-6");
    if (std::fabs((m1 - m0) / m0) > 1e-4) ++bad;
    row("free surface: Fluid cells touching Gas", double(open), 0.0, 0.0);
    row("free surface: interface cells", double(interf), 184.0, 1e-9);
  }

  std::printf("\n%s  (%d line%s differ)\n", bad ? "DEVICE AND HOST DISAGREE" : "DEVICE MATCHES HOST",
              bad, bad == 1 ? "" : "s");
  return bad ? 1 : 0;
}
