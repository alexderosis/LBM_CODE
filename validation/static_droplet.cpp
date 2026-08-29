//==============================================================================
//  A static droplet, for the colour-gradient model: Laplace's law at a density
//  ratio of up to 1000.
//
//  THE CASE THE SOURCE PAPER USES TO ESTABLISH ITS OWN MODEL. Saito, De Rosis,
//  Festuccia, Kaneko, Abe & Koyama, Phys. Rev. E 98, 013305 (2018), Sec. III and
//  Table I: a red droplet of radius R at rest in blue fluid, on a periodic
//  lattice, with the densities seeded as
//
//      rho_r = (rho_r^0 / 2) [1 - tanh(2(r - R)/W)],      Eq. (41)
//      rho_b = (rho_b^0 / 2) [1 + tanh(2(r - R)/W)],      Eq. (42)
//
//  and W = 4. Three-dimensional Laplace gives dp = 2 sigma / R, so the measured
//  tension is sigma_Lap = dp R / 2, against the closed form of Eq. (32),
//
//      sigma_th = 4 A tau / 9,
//
//  which contains no fitted constant. The paper reports a maximum error of 0.40%
//  across gamma = 1, 10, 100 and 1000, with spurious velocities of order 1e-4.
//
//  WHY THIS IS WORTH A REGISTERED TEST AND NOT JUST A DEMONSTRATION. Every part
//  of the model is load-bearing here and each fails differently:
//
//    phi_i          sets cs^2 = 9(1-alpha)/19 and therefore the density ratio.
//                   Wrong, and the two phases do not balance in pressure, so the
//                   droplet either collapses or inflates -- visible immediately.
//    A, B_i         set sigma through the capillary stress. Wrong by a constant,
//                   and dp R / 2 comes back wrong by that constant while
//                   everything else looks healthy. This is the failure the unit
//                   test's Eq. (39) derivation was written to catch first, and
//                   it did: Eq. (30) reads A/2 per colour, and the colour-blind
//                   operator carries A.
//    recolouring    holds the interface together. Wrong, and the droplet
//                   dissolves over thousands of steps rather than failing.
//    Phi_i          is invisible in a STATIC test -- u = 0 makes it identically
//                   zero. Nothing here exercises it, and that is stated rather
//                   than hidden: this case cannot vouch for the Galilean term.
//
//  WHAT IS MEASURED. dp between the droplet core and the far field, both read
//  from p = rho cs^2(alpha) with the local alpha, since the sound speed is a
//  property of the phase in this model and a single cs^2 would be wrong by the
//  density ratio. The spurious velocity is the maximum |u| anywhere, which for
//  a static droplet should be zero and is not -- it is the standard measure of
//  how much the discrete capillary stress fails to balance the discrete pressure
//  gradient.
//==============================================================================
#include "collision/ColourGradient.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ColourGradientSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace lbm;

using L    = D3Q27;
using CG   = ColourGradient<L>;
using Slv  = ColourGradientSolver<L, EsotericPull<L>, CG>;

struct Result {
  double sigma = 0, sigma_th = 0, err = 0, umax = 0;
  double p_in = 0, p_out = 0, drift = 0;
  // Diagnostics that separate the ways this can go wrong. A droplet that has
  // SHRUNK reports the wrong sigma through R_eff while its pressure jump is
  // right; a recolouring that has driven a colour NEGATIVE reports phi outside
  // [-1,1], which inflates the integral of |grad phi| through the interface and
  // therefore sigma itself. The two are indistinguishable from sigma alone.
  double r_eff = 0, rho_min = 0, phi_max = 0;
  bool ok = false;
};

static Result run(int N, double R, double gamma, double A, double tau,
                  double W, std::size_t steps, bool verbose) {
  const double ab = 8.0 / 27.0;
  const double ar = 1.0 - (1.0 - ab) / gamma;
  const double rho_b0 = 1.0, rho_r0 = gamma;
  const double nu_b = (1.0 / 3.0) * (tau - 0.5);
  const double nu_r = nu_b;

  // tau is the shear relaxation time and nu = (1/3)(tau - 1/2), Eq. (16), with
  // the LATTICE's 1/3 rather than the phase's 9(1-alpha)/19 -- see the note on
  // omega_at in ColourGradient.hpp. A matched tau is therefore a matched
  // viscosity, which is what the paper's static tests use.

  Domain d(N, N, N, true, true, true);
  CG cg;
  cg.alpha_r = Real(ar);   cg.alpha_b = Real(ab);
  cg.nu_r    = Real(nu_r); cg.nu_b    = Real(nu_b);
  cg.A       = Real(A);
  cg.beta    = Real(0.7);
  cg.omega_bulk = Real(1);
  cg.rho_r0  = Real(rho_r0);  cg.rho_b0 = Real(rho_b0);
  Slv s(d, cg);

  const Real Rr = Real(R), Wr = Real(W);
  const Real c0 = Real(0.5 * double(N));
  const Real rr0 = Real(rho_r0), rb0 = Real(rho_b0);
  const Index hx = d.hx, hy = d.hy, hz = d.hz;
  const Domain dd = d;
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real x = Real(px - hx) - c0, y = Real(py - hy) - c0, z = Real(pz - hz) - c0;
    const Real r = Kokkos::sqrt(x * x + y * y + z * z);
    const Real t = Kokkos::tanh(Real(2) * (r - Rr) / Wr);
    return Slv::Colours{Real(0.5) * rr0 * (Real(1) - t),
                        Real(0.5) * rb0 * (Real(1) + t)};
  });

  const double m_r0 = double(s.total_red()), m_b0 = double(s.total_blue());

  Result out;
  out.sigma_th = 4.0 * A * tau / 9.0;
  for (std::size_t k = 0; k < steps; ++k) {
    s.refresh();
    s.step();
  }
  s.refresh();

  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho_red());
  auto hb = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho_blue());
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hw = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());

  // p = rho cs^2(alpha) with the LOCAL alpha; a single cs^2 would be wrong by
  // the density ratio in one of the two phases.
  // phi comes from the solver rather than being recomputed here: one definition
  // of the order parameter, in one place.
  auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.phi());
  (void)0;
  // p = sum_k rho_k cs_k^2, Eq. (26) with each phase's own alpha -- the same
  // combination the equilibrium's rest term carries, and the one that makes the
  // pure-phase balance of Eq. (25) hold.
  const double cs2r = 9.0 * (1.0 - ar) / 19.0, cs2b = 9.0 * (1.0 - ab) / 19.0;
  auto pressure = [&](Index n) {
    return double(hr(n)) * cs2r + double(hb(n)) * cs2b;
  };


  double pin = 0, pout = 0, umax = 0;
  double rmin = 1e30, pmax = 0, mred = 0, minor_in = 0;
  int nin = 0, nout = 0;
  bool bad = false;
  const double c = 0.5 * double(N);
  for (Index z = 0; z < N; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y, z);
        const double dx = double(x) - c, dy = double(y) - c, dz = double(z) - c;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double p = pressure(n);
        if (!std::isfinite(p)) bad = true;
        // THE AVERAGING WINDOW HAS TO EXCLUDE THE INTERFACE TAIL, and at a
        // large density ratio "the tail" reaches much further than it looks.
        // The pressure is sum_k rho_k cs_k^2, and cs_b^2 is gamma times cs_r^2,
        // so a trace of the LIGHT fluid inside the droplet is weighted a
        // thousandfold against the heavy fluid it sits in. At R = 12, W = 4 the
        // seeded profile still holds rho_b = 0.018 at r = R - W, which
        // contributes 6e-3 to the pressure against a Laplace jump of 5.9e-5 --
        // a hundred times the quantity being measured. Three widths, not one.
        if (r < R - 3 * W)      { pin  += p; ++nin;  minor_in += double(hb(n)); }
        else if (r > R + 3 * W) { pout += p; ++nout; }
        const double a = double(hu(n)), b2 = double(hv(n)), e = double(hw(n));
        umax = std::max(umax, std::sqrt(a * a + b2 * b2 + e * e));
        rmin = std::min(rmin, std::min(double(hr(n)), double(hb(n))));
        pmax = std::max(pmax, std::fabs(double(hp(n))));
        mred += double(hr(n));
      }
  // Effective radius from the red mass, so a droplet that has shrunk says so.
  out.r_eff = std::cbrt(3.0 * mred / (4.0 * M_PI * rho_r0));
  out.rho_min = nin ? minor_in / nin : 0;   // mean LIGHT density in the core
  out.phi_max = pmax;
  out.p_in  = nin  ? pin  / nin  : 0;
  out.p_out = nout ? pout / nout : 0;
  out.sigma = 0.5 * (out.p_in - out.p_out) * R;      // 3D: dp = 2 sigma / R
  out.err   = std::fabs(out.sigma / out.sigma_th - 1.0);
  out.umax  = umax;
  const double m_r1 = double(s.total_red()), m_b1 = double(s.total_blue());
  out.drift = std::max(std::fabs(m_r1 / m_r0 - 1.0), std::fabs(m_b1 / m_b0 - 1.0));
  out.ok = !bad && std::isfinite(out.sigma);
  if (verbose)
    std::printf("%-8.0f %-13.5e %-13.5e %-9.2f %-11.3e %-8.2f %-11.3e %-9.5f %-10.2e\n",
                gamma, out.sigma_th, out.sigma, 100.0 * out.err, out.umax,
                out.r_eff, out.rho_min, out.phi_max, out.drift);
  return out;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    int N = 64; double R = 16, W = 4, A = 8e-4, tau = 1.0;
    std::size_t steps = 8000; double only = 0;
    for (int i = 1; i < argc; ++i) {
      auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-n"))     { if (i+1<argc) N = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-r"))     num(R);
      else if (!std::strcmp(argv[i], "-w"))     num(W);
      else if (!std::strcmp(argv[i], "-a"))     num(A);
      else if (!std::strcmp(argv[i], "-tau"))   num(tau);
      else if (!std::strcmp(argv[i], "-gamma")) num(only);   // one ratio only
      else if (!std::strcmp(argv[i], "-steps")) { if (i+1<argc) steps = std::size_t(std::atol(argv[++i])); }
    }

    std::printf("Static droplet   D3Q27 colour gradient, nonorthogonal central moments\n");
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("%dx%dx%d   R = %.0f   W = %.0f   A = %.2e   tau = %.2f   %zu steps\n",
                N, N, N, R, W, A, tau, steps);
    std::printf("sigma_th = 4 A tau / 9 = %.6e\n\n", 4.0 * A * tau / 9.0);

    std::printf("%-8s %-13s %-13s %-9s %-11s %-8s %-11s %-9s %-10s\n",
                "gamma", "sigma_th", "sigma_Lap", "err (%)", "max |u|",
                "R_eff", "core rho_b", "max |phi|", "drift");
    std::printf("%s\n", std::string(100, '-').c_str());

    double ratios[4] = {1.0, 10.0, 100.0, 1000.0};
    int nr = 4;
    if (only > 0) { ratios[0] = only; nr = 1; }
    Result res[4];
    for (int k = 0; k < nr; ++k)
      res[k] = run(N, R, ratios[k], A, tau, W, steps, true);
    for (int k = nr; k < 4; ++k) res[k] = res[0];

    double worst = 0, worst_u = 0, worst_d = 0;
    bool finite = true;
    for (int k = 0; k < 4; ++k) {
      worst = std::max(worst, res[k].err);
      worst_u = std::max(worst_u, res[k].umax);
      worst_d = std::max(worst_d, res[k].drift);
      finite = finite && res[k].ok;
    }

    const bool p_sig = finite && worst   < 0.05;
    const bool p_u   = finite && worst_u < 1e-2;
    const bool p_m   = finite && worst_d < 1e-9;

    std::printf("\nacceptance:\n");
    std::printf("  Laplace tension within 5%% at every ratio   worst %.2f%%   %s\n",
                100.0 * worst, p_sig ? "PASS" : "FAIL");
    std::printf("  spurious velocity bounded                  worst %.2e   %s\n",
                worst_u, p_u ? "PASS" : "FAIL");
    std::printf("  each colour's mass is conserved            worst %.2e   %s\n",
                worst_d, p_m ? "PASS" : "FAIL");
    std::printf("\n  The paper's Table I reports 0.40%% worst error and spurious\n"
                "  velocities of 1e-4 at 100^3 with R = 25; this runs smaller by\n"
                "  default, so the tolerance is looser than its result, not than\n"
                "  its method. Pass -n 100 -r 25 to reproduce the published grid.\n");
    if (!(p_sig && p_u && p_m)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
