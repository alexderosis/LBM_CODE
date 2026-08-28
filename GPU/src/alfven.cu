//==============================================================================
//  The shear Alfven wave -- the MHD check that has an exact answer.
//
//      B = (B0, b sin k(x - v_A t), 0),   u = (0, -b sin k(x - v_A t)/sqrt(rho), 0)
//
//  This is an EXACT solution of the full NONLINEAR incompressible MHD equations,
//  not merely the linearised ones. u is perpendicular to B0 and everything
//  depends only on x, so (u.grad)u vanishes identically while (B.grad)B does
//  not: the Lorentz coupling and the induction equation are both driven, and
//  both must be right. It propagates at v_A = B0/sqrt(rho) and damps at
//  (nu + eta) k^2 / 2.
//
//  WHY BOTH SPEED AND DAMPING ARE REPORTED. They fail differently, and that is
//  the diagnostic value of this case:
//
//    * an error in the LORENTZ COUPLING -- the Maxwell stress in the fluid
//      equilibrium -- shows up as the wrong wave SPEED, immediately and at any
//      resolution;
//    * an error in the COUPLING ORDER shows up only in the DAMPING, and only
//      under refinement. Stepping the fluid against a magnetic field from the
//      previous step is a first-order splitting error, and under diffusive
//      scaling the ratio omega^2 dt / (nu k^2) is independent of L, so it appears
//      as a damping offset that SURVIVES every grid refinement. The parent
//      implementation found exactly this: 1.55e-2 -> 2.79e-2 -> 3.16e-2 as it
//      refined, while the phase speed converged cleanly at second order.
//
//  So `-sweep` is not a nicety. A non-converging error sitting beside a
//  converging one is the signature, and one resolution cannot show it.
//
//  THE SWEEP USES DIFFUSIVE SCALING: v_A goes like 1/L with nu and eta held
//  fixed in lattice units, so the Reynolds and Lundquist numbers stay constant
//  and the number of steps to a given damping goes like L^2. Refining any other
//  way changes the physical problem as well as the grid, and then nothing can be
//  concluded from the comparison.
//
//  DIV B IS NOT TESTED HERE, and cannot be: B_y depends only on x, so div B is
//  structurally zero and would report round-off whatever the scheme did. Run
//  orszag_tang for that.
//
//    usage: alfven [-l L] [-nu NU] [-b0 B0] [-amp A] [-op bgk|cm] [-sweep]
//==============================================================================
#include "lbm/backend.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct AlfvenFluidInit {
  Real k, amp;
  LBM_HD Macro operator()(int x, int, int) const {
    Macro m;
    m.rho = Real(1);
    m.ux  = Real(0);
    m.uy  = -amp * Real(sin(double(k) * double(x)));
    m.uz  = Real(0);
    return m;
  }
};

struct AlfvenBInit {
  Real k, amp, B0;
  LBM_HD void operator()(int x, int, int, Real B[3]) const {
    B[0] = B0;
    B[1] = amp * Real(sin(double(k) * double(x)));
    B[2] = Real(0);
  }
};

struct AlfvenUInit {
  Real k, amp;
  LBM_HD void operator()(int x, int, int, Real u[3]) const {
    u[0] = Real(0);
    u[1] = -amp * Real(sin(double(k) * double(x)));
    u[2] = Real(0);
  }
};

//------------------------------------------------------------------------------
// Amplitude and phase of the fundamental x-mode of B_y, averaged over y and z.
//------------------------------------------------------------------------------
static void project(const std::vector<Real>& by, int nx, int ny, int nz,
                    double& amp, double& phase) {
  const double k = 2.0 * M_PI / nx;
  const double inv = 1.0 / (double(ny) * nz);
  double S = 0, C = 0;
  for (int x = 0; x < nx; ++x) {
    double p = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y) p += double(by[std::size_t(node_id(x, y, z, nx, ny))]);
    p *= inv;
    S += p * std::sin(k * x);
    C += p * std::cos(k * x);
  }
  amp   = 2.0 * std::hypot(S, C) / nx;
  phase = std::atan2(C, S);
}

static double slope(const std::vector<double>& x, const std::vector<double>& y) {
  const double n = double(x.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

struct Result { double speed, speed_err, damp, damp_err; };

static Result run(int L, Op op, double nu, double eta, double B0, double rel_amp,
                  bool verbose) {
  const int ny = 4, nz = 4;
  const double rho = 1.0;
  const double k = 2.0 * M_PI / L;
  const double vA = B0 / std::sqrt(rho);
  const double b = rel_amp * B0;

  backend::Magnetic mag(L, ny, nz, Real(eta));
  backend::Fluid    fl (L, ny, nz, op, Real(nu));

  fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  fl.initialise_with(AlfvenFluidInit{Real(k), Real(b / std::sqrt(rho))});
  mag.initialise_with(AlfvenBInit{Real(k), Real(b), Real(B0)},
                      AlfvenUInit{Real(k), Real(b / std::sqrt(rho))});

  // Long enough for a couple of e-folds of damping, which also carries the wave
  // several radians in phase.
  const double damp_exact = 0.5 * (nu + eta) * k * k;
  const std::size_t T = std::size_t(2.0 / damp_exact);
  const std::size_t probe = T / 24 ? T / 24 : 1;
  const std::size_t settle = T / 8;            // let the seeding transient leave

  std::vector<Real> bx, by, bz;
  std::vector<double> ts, ph, la;
  double prev = 0;
  bool first = true;

  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      mag.field_to_host(bx, by, bz);
      double a, p;
      project(by, L, ny, nz, a, p);
      if (!first) { while (p - prev >  M_PI) p -= 2.0 * M_PI;
                    while (p - prev < -M_PI) p += 2.0 * M_PI; }
      prev = p; first = false;
      if (t >= settle) { ts.push_back(double(t)); ph.push_back(p); la.push_back(std::log(a)); }
      if (verbose && t % (probe * 4) == 0)
        std::printf("  %10zu %14.6e %14.4f\n", t, a, p);
    }
    if (t < T) {
      // Refresh B first, so the fluid collides against the field at its own time
      // level. On the device these are launches on the default stream and are
      // already ordered.
      mag.compute_field();
      fl.step();
      mag.step();
    }
  }

  Result r;
  r.speed = -slope(ts, ph) / k;
  r.damp  = -slope(ts, la);
  r.speed_err = (r.speed - vA) / vA;
  r.damp_err  = (r.damp - damp_exact) / damp_exact;
  return r;
}

int main(int argc, char** argv) {
  int L = 64;
  double nu = 0.01, B0 = 0.02, rel_amp = 0.1;
  std::string op = "cm";
  bool sweep = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-l"     && i + 1 < argc) L       = std::atoi(argv[++i]);
    if (a == "-nu"    && i + 1 < argc) nu      = std::atof(argv[++i]);
    if (a == "-b0"    && i + 1 < argc) B0      = std::atof(argv[++i]);
    if (a == "-amp"   && i + 1 < argc) rel_amp = std::atof(argv[++i]);
    if (a == "-op"    && i + 1 < argc) op      = argv[++i];
    if (a == "-sweep") sweep = true;
  }
  const double eta = nu;                       // magnetic Prandtl number 1
  const Op which = (op == "bgk") ? Op::BGK : Op::CentralMoments;

  std::printf("Shear Alfven wave   %s   D3Q27 fluid / D3Q7 field   operator %s   %s\n",
              backend::on_device ? "CUDA native" : "HOST reference",
              which == Op::BGK ? "bgk" : "cm", sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  nu = eta = %.4f (Pr_m = 1)   perturbation %.0f%% of B0\n\n",
              nu, 100.0 * rel_amp);

  const auto wall0 = std::chrono::steady_clock::now();

  if (!sweep) {
    std::printf("  L = %d   B0 = %.4f   v_A = %.6f\n", L, B0, B0);
    std::printf("  %10s %14s %14s\n", "step", "|B_y|", "phase");
    const Result r = run(L, which, nu, eta, B0, rel_amp, true);
    std::printf("\n  wave speed    %.8f   exact %.8f   rel %+.3e\n",
                r.speed, B0, r.speed_err);
    std::printf("  damping rate  %.6e   exact %.6e   rel %+.3e\n",
                r.damp, 0.5 * (nu + eta) * (2.0 * M_PI / L) * (2.0 * M_PI / L),
                r.damp_err);
  } else {
    // DIFFUSIVE SCALING: v_A goes like 1/L, nu and eta fixed, so the Reynolds and
    // Lundquist numbers are the same at every resolution and the three rows are
    // the same physical problem on three grids.
    std::printf("  diffusive scaling: v_A ~ 1/L at fixed nu, eta\n\n");
    std::printf("  %6s %10s %14s %14s %14s\n", "L", "v_A", "speed err", "damping err", "steps");
    for (int l : {32, 64, 128}) {
      const double b0 = B0 * 32.0 / double(l);
      const double k = 2.0 * M_PI / l;
      const std::size_t steps = std::size_t(2.0 / (0.5 * (nu + eta) * k * k));
      const Result r = run(l, which, nu, eta, b0, rel_amp, false);
      std::printf("  %6d %10.6f %+14.3e %+14.3e %14zu\n",
                  l, b0, r.speed_err, r.damp_err, steps);
    }
    std::printf("\n      Both columns must SHRINK. A damping error that stays put --\n");
    std::printf("      or grows -- while the speed converges is the signature of a\n");
    std::printf("      coupling evaluated one step out of date, and it is the reason\n");
    std::printf("      the drivers refresh B before stepping the fluid.\n");
  }

  std::printf("\n  %.2f s\n", std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall0).count());
  return 0;
}
