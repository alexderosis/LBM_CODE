//==============================================================================
//  A static droplet on the PHASE-FIELD engine: Laplace's law at a density ratio.
//
//  THE SAME MEASUREMENT src/droplet.cu MAKES ON THE COLOUR GRADIENT, on the same
//  lattice, at the same radius and the same asked surface tension -- so the two
//  multiphase engines can be compared directly rather than by reputation. In
//  three dimensions
//
//      dp = 2 sigma / R,
//
//  so the measured tension is 0.5 (p_in - p_out) R.
//
//  THE PRESSURE IS p = rho(phi) cs2 p~ in the default local-rho normalisation:
//  the populations carry p~, not p, and rho is interpolated from phi. That is
//  the whole reason this operator reaches a density ratio at all -- see the
//  banner in phasefield.cuh.
//
//  THE GAUGE IS NOT FREE, AND IT IS THE FIRST THING TO GET WRONG. Seeding
//  p~ = 0 is right for a problem whose pressure is near zero at the interface,
//  which a static droplet is. Seeding p = rho_L cs2 instead -- p~ running from 1
//  to 1/ratio -- changes no physics in the continuum and wrecks the scheme on
//  the lattice: the two large terms that must cancel across the interface,
//  rho cs2 grad p~ and F_p = -p~ cs2 grad rho, are both PROPORTIONAL to the
//  gauge, so their difference is not. -gauge exposes it as a flag precisely so
//  that it can be measured rather than argued about.
//
//  THE AVERAGING WINDOW EXCLUDES TWO INTERFACE WIDTHS, NOT THREE, and the
//  difference from src/droplet.cu is reasoned rather than copied. There the
//  pressure is sum_k rho_k cs_k^2 with cs_b^2 a factor gamma above cs_r^2, so a
//  trace of the LIGHT fluid inside the droplet is weighted gamma-fold and three
//  widths are needed. Here the pressure is rho(phi) cs2 p~ with a single
//  interpolated rho, so the error at the window edge is just
//  (drho/dphi)(1 - phi)/rho. At two widths the tanh gives 1 - phi = 3.4e-4, so
//  at a ratio of 10 that is 3.0e-4 of the pressure -- negligible against a
//  measurement at the per-cent level, and it buys back a core large enough to
//  average over.
//
//  -window overrides that gap, in units of W, and the default stays 2. It is
//  there so this driver and src/droplet.cu can be measured IDENTICALLY when the
//  two models are compared -- otherwise the measurement differs along with the
//  model and the comparison says nothing.
//
//  -nu MATCHES KINEMATIC viscosity across the phases, setting mu_L = nu and
//  mu_H = nu gamma. That is the flag a comparison against src/droplet.cu wants,
//  because that driver sets nu_r = nu_b directly. Matching mu instead (the -mul
//  / -muh default) leaves the heavy phase at nu/gamma; see main().
//
//  -w NOW MEANS THE SAME THING IN src/droplet.cu, and for a while it did not:
//  that driver was missing the factor of 2 in its seeded tanh, so its -w 4 was
//  twice this interface. Comparing the two engines is what found it. Here W is
//  a model parameter -- beta_from_sigma and kappa_from_sigma both read it -- so
//  this file was the one that had to be right, and it was. See that file's
//  header for what the wrong width cost.
//
//  Runs on the host with no GPU, at a smaller grid:
//     c++ -std=c++17 -O2 -Iinclude -x c++ src/bubble.cu -o bubble
//==============================================================================
#include "lbm/backend.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace lbm;

struct DropInit {
  int n;
  Real R, W, gauge;
  LBM_HD void operator()(int x, int y, int z, Real& phi, Real& pt) const {
    const double c = 0.5 * double(n);
    const double dx = double(x) - c, dy = double(y) - c, dz = double(z) - c;
    const double r = sqrt(dx * dx + dy * dy + dz * dz);
    phi = Real(0.5 * (1.0 - tanh(2.0 * (r - double(R)) / double(W))));
    pt  = gauge;
  }
};

int main(int argc, char** argv) {
  int n = 64, steps = 6000, report = 0;
  double R = 16.0, W = 4.0, gamma = 1.0, sigma = 1e-3;
  double mobility = 0.05, mu_L = 0.05, mu_H = 0.05, gauge = 0.0;
  double window = 2.0;   // averaging half-gap, in interface widths
  double nu_matched = -1.0;        // -nu: set mu from rho, see below
  int viscous = 1;
  for (int i = 1; i < argc; ++i) {
    auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-n"))      { if (i+1<argc) n = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-steps"))  { if (i+1<argc) steps = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-report")) { if (i+1<argc) report = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-novisc")) { viscous = 0; }
    else if (!std::strcmp(argv[i], "-r"))      num(R);
    else if (!std::strcmp(argv[i], "-w"))      num(W);
    else if (!std::strcmp(argv[i], "-gamma"))  num(gamma);
    else if (!std::strcmp(argv[i], "-sigma"))  num(sigma);
    else if (!std::strcmp(argv[i], "-mob"))    num(mobility);
    else if (!std::strcmp(argv[i], "-mul"))    num(mu_L);
    else if (!std::strcmp(argv[i], "-muh"))    num(mu_H);
    else if (!std::strcmp(argv[i], "-gauge"))  num(gauge);
    else if (!std::strcmp(argv[i], "-window")) num(window);
    else if (!std::strcmp(argv[i], "-nu"))     num(nu_matched);
  }

  // MATCHED KINEMATIC viscosity, applied AFTER the parse loop so that -nu and
  // -gamma may appear in either order. Doing it inside the loop reads whatever
  // gamma happened to be at that point, which is the default when -nu comes
  // first -- a silent wrong run, not an error.
  //
  // mu_L = nu rho_L and mu_H = nu rho_H. Both mu and rho interpolate linearly
  // in phi, so mu(phi) = nu rho(phi) EXACTLY and nu is the same constant
  // everywhere; omega is then 1/(nu/cs2 + 1/2), independent of gamma. Matching
  // mu instead leaves the heavy phase at nu/gamma, and at gamma = 100 that is
  // omega = 1.994 against a limit of 2 -- which is what diverged.
  if (nu_matched > 0) { mu_L = nu_matched * 1.0; mu_H = nu_matched * gamma; }

  const auto dev = backend::device_info();
  std::printf("Static droplet, phase field   D3Q27 fluid + D3Q7 phase\n");
  std::printf("device %s   precision %s\n", dev.name.c_str(),
              sizeof(Real) == 4 ? "FP32" : "FP64");

  backend::PhaseField pf(n, n, n);
  pf.phase.width = Real(W);
  pf.phase.omega = Real(PhaseModel::omega_from_mobility(Real(mobility)));
  pf.fluid.rho_L = Real(1);      pf.fluid.rho_H = Real(gamma);
  pf.fluid.mu_L  = Real(mu_L);   pf.fluid.mu_H  = Real(mu_H);
  pf.fluid.beta  = Real(MultiphaseModel::beta_from_sigma(Real(sigma), Real(W)));
  pf.fluid.kappa = Real(MultiphaseModel::kappa_from_sigma(Real(sigma), Real(W)));
  pf.enable_viscous_force(viscous != 0);

  // OMEGA IS PRINTED, NOT LEFT TO BE INFERRED. omega = 1/(mu/(rho cs2) + 1/2),
  // so matching mu across a large ratio drives the HEAVY phase towards 2 while
  // every input still looks reasonable. A gamma = 100 run at mu = 0.05 gives
  // omega = 1.994 and diverges; the number that says so belongs on screen
  // before the run, not in a post-mortem.
  const double cs2f = 1.0 / 3.0;
  const double omL = 1.0 / (mu_L / (1.0     * cs2f) + 0.5);
  const double omH = 1.0 / (mu_H / (gamma   * cs2f) + 0.5);
  std::printf("%d^3   R = %.1f   W = %.1f   gamma = %.0f   M = %.3f   "
              "window = %.1fW%s\n",
              n, R, W, gamma, mobility, window,
              viscous ? "" : "   (F_nu off)");
  std::printf("mu = %.4f / %.4f   nu = %.4f / %.4f   omega = %.3f / %.3f%s\n",
              mu_L, mu_H, mu_L / 1.0, mu_H / gamma, omL, omH,
              (omL > 1.9 || omH > 1.9) ? "   <-- OVER 1.9, EXPECT DIVERGENCE" : "");
  std::printf("sigma asked = %.6e   (beta = %.4e, kappa = %.4e)   "
              "p~ gauge = %.3f   %d steps\n\n",
              sigma, double(pf.fluid.beta), double(pf.fluid.kappa), gauge, steps);

  pf.initialise_with(DropInit{n, Real(R), Real(W), Real(gauge)});

  std::vector<Real> hphi, hpt, hu, hv, hw;
  const double cs2 = 1.0 / 3.0;
  const double c = 0.5 * double(n);

  struct Out { double p_in, p_out, sigma, umax, r_eff, phi_min, phi_max;
               bool bad; long nin, nout; };
  auto measure = [&]() {
    Out o{0,0,0,0,0,1e30,-1e30,false,0,0};
    pf.field_to_host(pf.phi_device(), hphi);
    pf.field_to_host(pf.pt_device(),  hpt);
    pf.field_to_host(pf.ux_device(),  hu);
    pf.field_to_host(pf.uy_device(),  hv);
    pf.field_to_host(pf.uz_device(),  hw);
    double pin = 0, pout = 0, mheavy = 0;
    long nin = 0, nout = 0;
    for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          const long id = node_id(x, y, z, n, n);
          const double dx = x - c, dy = y - c, dz = z - c;
          const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
          const double ph = double(hphi[id]);
          // p = rho(phi) cs2 p~, the local-rho normalisation.
          const double rho = double(pf.fluid.local(Real(ph)).rho);
          const double p = rho * cs2 * double(hpt[id]);
          if (!std::isfinite(p) || !std::isfinite(ph)) o.bad = true;
          if      (r < R - window * W) { pin  += p; ++nin; }
          else if (r > R + window * W) { pout += p; ++nout; }
          const double a = hu[id], b = hv[id], e = hw[id];
          o.umax = std::max(o.umax, std::sqrt(a*a + b*b + e*e));
          o.phi_min = std::min(o.phi_min, ph);
          o.phi_max = std::max(o.phi_max, ph);
          mheavy += ph;
        }
    o.nin = nin;  o.nout = nout;
    o.p_in  = nin  ? pin  / double(nin)  : 0;
    o.p_out = nout ? pout / double(nout) : 0;
    o.sigma = 0.5 * (o.p_in - o.p_out) * R;        // 3D: dp = 2 sigma / R
    o.r_eff = std::cbrt(3.0 * mheavy / (4.0 * M_PI));
    return o;
  };

  {
    const Out probe = measure();
    if (probe.nin == 0 || probe.nout == 0) {
      std::printf("  NO AVERAGING WINDOW: %ld inside (needs r < %.1f), "
                  "%ld outside (needs r > %.1f).\n",
                  probe.nin, R - window * W, probe.nout, R + window * W);
      std::printf("  The core needs R > %.1fW, and the box half-diagonal > R + %.1fW,\n"
                  "  i.e. n > %.0f at this R and W.\n",
                  window, window, 2.0 * (R + window * W) / std::sqrt(3.0));
      return 1;
    }
  }

  std::printf("  step      p_in         p_out        sigma        err      "
              "max|u|      R_eff   phi range\n");
  std::printf("  --------------------------------------------------------"
              "-------------------------------------\n");
  const int every = report > 0 ? report : steps;
  for (int t = 0; t <= steps; ++t) {
    if (t % every == 0 || t == steps) {
      backend::sync();
      const Out o = measure();
      std::printf("  %-9d %-12.6e %-12.6e %-12.6e %+7.2f%%  %-11.3e %.3f   "
                  "%.3f..%.3f\n",
                  t, o.p_in, o.p_out, o.sigma,
                  100.0 * (o.sigma - sigma) / sigma, o.umax, o.r_eff,
                  o.phi_min, o.phi_max);
      if (o.bad) { std::printf("  DIVERGED\n"); break; }
    }
    if (t == steps) break;
    pf.step();
  }

  backend::sync();
  const Out o = measure();
  std::printf("\n  Laplace tension  %.6e measured against %.6e asked   %+.2f%%\n",
              o.sigma, sigma, 100.0 * (o.sigma - sigma) / sigma);
  std::printf("  effective radius %.3f against %.1f seeded\n", o.r_eff, R);
  std::printf("  spurious current %.3e\n", o.umax);
  std::printf("  phi range %.4f .. %.4f  (the clamp in the equation of state is\n"
              "            what keeps rho inside the two phases when this leaves [0,1])\n",
              o.phi_min, o.phi_max);
  return 0;
}
