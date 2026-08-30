//==============================================================================
//  A static droplet: Laplace's law for the colour-gradient model.
//
//  THE CANONICAL TEST OF A SURFACE-TENSION MODEL, and the one that decides
//  whether the perturbation coefficient is right. A sphere of red in blue,
//  seeded at rest with a tanh profile and left alone. In three dimensions
//
//      dp = 2 sigma / R,
//
//  so the measured tension is 0.5 (p_in - p_out) R, and it is compared against
//  the sigma the model was ASKED for through sigma = 4 A tau / 9, Eq. (32).
//  Getting the perturbation coefficient wrong -- A/2 instead of A, which is what
//  Eq. (30) literally reads -- halves the answer here and changes nothing else
//  in the model, which is exactly why this case exists.
//
//  THE PRESSURE IS p = sum_k rho_k cs_k^2, Eq. (26), each phase with its own
//  alpha. That is the same combination the equilibrium's rest term carries, and
//  the one that makes the pure-phase balance of Eq. (25) hold.
//
//  THE AVERAGING WINDOW EXCLUDES THREE INTERFACE WIDTHS, NOT ONE, and at a large
//  density ratio that is not conservatism. cs_b^2 is gamma times cs_r^2, so a
//  trace of the LIGHT fluid inside the droplet is weighted gamma-fold against
//  the heavy fluid it sits in. At R = 12, W = 4 the seeded profile still holds
//  rho_b = 0.018 at r = R - W, contributing 6e-3 to the pressure against a
//  Laplace jump of 5.9e-5 -- a hundred times the quantity being measured.
//
//  -window changes that gap, in units of W, and the default stays 3. It exists
//  so this driver and bubble.cu can be measured THE SAME WAY when the two
//  models are compared: bubble.cu justifies 2 on its own error structure, and
//  a comparison run must not let the measurement differ along with the model.
//
//  THE SEEDED INTERFACE WAS TWICE TOO WIDE UNTIL THIS COMMIT, and it is worth
//  recording because everything else about the driver was right. The paper's
//  Eqs. (41)-(42) seed tanh[2(r - R)/W]; this file had tanh[(r - R)/W]. The
//  parent's validation/static_droplet.cpp has the 2, and so does bubble.cu next
//  door, so -w 4 here bought an interface no other part of the tree agreed with.
//
//  It hid because a wrong interface width is still a CONSISTENT simulation. The
//  device reproduced the host to every digit, the equilibrium and perturbation
//  identities all held, sigma converged to a steady value -- nothing was
//  inconsistent, only wrong, and the one number that could have shown it was
//  read as a property of the model:
//
//      gamma     parent      this file, 2x wide     this file, fixed
//        1       0.87%           -2.48%                 +0.94%
//       10       0.77%           -3.02%                 +0.67%
//      100         --           -13.61%                 -3.36%
//
//  The GPU port was quietly 3x worse than the code it was ported from at
//  gamma = 1, and that gap sat in the README as an open question about the
//  model rather than being chased. It was the seed.
//
//  WHAT THE PARENT MEASURED, and what this is expected to reproduce: gamma = 1
//  within 0.87% and gamma = 10 within 0.77%. AT gamma = 100 AND 1000 THE PARENT
//  OVERESTIMATES sigma and the cause is not settled -- four hypotheses were
//  tested and falsified there (shrinkage, negative densities, phi overshoot,
//  core contamination). Do not read a large-ratio number off this driver as a
//  property of the GPU port; it is a property of the model as implemented, and
//  the two codes should agree on being wrong together.
//
//  Runs on the host with no GPU, at a smaller grid:
//     c++ -std=c++17 -O2 -Iinclude -x c++ src/droplet.cu -o droplet
//==============================================================================
#include "lbm/backend.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
// A tanh sphere, Eqs. (41)-(42). Each phase carries its own bulk density, so the
// order parameter is +-1 in the pure phases whatever the ratio.
//------------------------------------------------------------------------------
struct DropletInit {
  int n;
  Real R, W, rho_r0, rho_b0;
  LBM_HD void operator()(int x, int y, int z, Real& rr, Real& rb) const {
    const double c = 0.5 * double(n);
    const double dx = double(x) - c, dy = double(y) - c, dz = double(z) - c;
    const double r = sqrt(dx * dx + dy * dy + dz * dz);
    // Eq. (41)-(42). THE FACTOR OF 2 IS THE PAPER-S, and it was missing here
    // until it was caught by comparing against bubble.cu: without it -w is a
    // twice-wider interface than the same flag buys in every other place this
    // profile is seeded -- validation/static_droplet.cpp in the parent, and
    // src/bubble.cu next door. See the header for what that cost.
    const double t = 0.5 * (1.0 - tanh(2.0 * (r - double(R)) / double(W)));
    rr = Real(double(rho_r0) * t);
    rb = Real(double(rho_b0) * (1.0 - t));
  }
};

int main(int argc, char** argv) {
  int n = 64, steps = 4000, report = 0;
  double R = 16.0, W = 4.0, gamma = 1.0, sigma = 1e-3, tau = 0.8;
  double window = 3.0;   // averaging half-gap, in interface widths
  for (int i = 1; i < argc; ++i) {
    auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-n"))     { if (i + 1 < argc) n = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-steps")) { if (i + 1 < argc) steps = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-report")){ if (i + 1 < argc) report = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-r"))     num(R);
    else if (!std::strcmp(argv[i], "-w"))     num(W);
    else if (!std::strcmp(argv[i], "-gamma")) num(gamma);
    else if (!std::strcmp(argv[i], "-sigma")) num(sigma);
    else if (!std::strcmp(argv[i], "-tau"))   num(tau);
    else if (!std::strcmp(argv[i], "-window")) num(window);
  }

  const auto dev = backend::device_info();
  std::printf("Static droplet, colour gradient   D3Q27, central moments\n");
  std::printf("device %s   precision %s\n", dev.name.c_str(),
              sizeof(Real) == 4 ? "FP32" : "FP64");

  // alpha_b = 8/27 makes the blue phase an ordinary lattice with cs^2 = 1/3
  // exactly; alpha_r then carries the density ratio through Eq. (25).
  const double ab = 8.0 / 27.0;
  const double ar = ColourModel::alpha_r_from_ratio(Real(gamma), Real(ab));
  const double nu = ColourModel::viscosity_from_tau(Real(tau));
  const double A  = ColourModel::A_from_sigma(Real(sigma), Real(tau));

  backend::Colour cg(n, n, n);
  cg.model.alpha_r = Real(ar);
  cg.model.alpha_b = Real(ab);
  cg.model.rho_r0  = Real(gamma);
  cg.model.rho_b0  = Real(1);
  cg.model.nu_r    = Real(nu);
  cg.model.nu_b    = Real(nu);
  cg.model.A       = Real(A);
  cg.model.beta    = Real(0.7);

  std::printf("%d^3   R = %.1f   W = %.1f   gamma = %.0f   tau = %.2f   nu = %.4f"
              "   window = %.1fW\n",
              n, R, W, gamma, tau, nu, window);
  std::printf("sigma asked = %.6e   (A = %.6e)   %d steps\n\n", sigma, A, steps);

  cg.initialise_with(DropletInit{n, Real(R), Real(W), Real(gamma), Real(1)});

  std::vector<Real> hr, hb, hu, hv, hw;
  const double cs2r = 9.0 * (1.0 - ar) / 19.0;
  const double cs2b = 9.0 * (1.0 - ab) / 19.0;
  const double c = 0.5 * double(n);

  struct Out { double p_in, p_out, sigma, umax, r_eff, minor; bool bad; long nin, nout; };
  auto measure = [&]() {
    Out o{0, 0, 0, 0, 0, 0, false, 0, 0};
    cg.field_to_host(cg.rho_red_device(), hr);
    cg.field_to_host(cg.rho_blue_device(), hb);
    cg.field_to_host(cg.ux_device(), hu);
    cg.field_to_host(cg.uy_device(), hv);
    cg.field_to_host(cg.uz_device(), hw);
    double pin = 0, pout = 0, mred = 0, minor = 0;
    long nin = 0, nout = 0;
    for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          const long id = node_id(x, y, z, n, n);
          const double dx = x - c, dy = y - c, dz = z - c;
          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          const double p = double(hr[id]) * cs2r + double(hb[id]) * cs2b;
          if (!std::isfinite(p)) o.bad = true;
          if      (r < R - window * W) { pin  += p; ++nin; minor += double(hb[id]); }
          else if (r > R + window * W) { pout += p; ++nout; }
          const double a = hu[id], b = hv[id], e = hw[id];
          o.umax = std::max(o.umax, std::sqrt(a * a + b * b + e * e));
          mred += double(hr[id]);
        }
    o.nin = nin;  o.nout = nout;
    o.p_in  = nin  ? pin  / double(nin)  : 0;
    o.p_out = nout ? pout / double(nout) : 0;
    o.sigma = 0.5 * (o.p_in - o.p_out) * R;         // 3D: dp = 2 sigma / R
    o.r_eff = std::cbrt(3.0 * mred / (4.0 * M_PI * gamma));
    o.minor = nin ? minor / double(nin) : 0;
    return o;
  };

  // THE WINDOW MUST EXIST. Averaging over no cells gives zero, and zero minus
  // the outside pressure is a large negative "surface tension" that looks like a
  // physics failure and is a geometry mistake. R > kW is required for the core to
  // be non-empty at all, and the box must reach kW beyond the interface, where k
  // is -window (default 3, see the header).
  {
    const Out probe = measure();
    if (probe.nin == 0 || probe.nout == 0) {
      std::printf("  NO AVERAGING WINDOW: %ld cells inside (needs r < %.1f), "
                  "%ld outside (needs r > %.1f).\n",
                  probe.nin, R - window * W, probe.nout, R + window * W);
      std::printf("  The core needs R > %.1fW, and the box half-diagonal > R + %.1fW.\n",
                  window, window);
      std::printf("  With R = %.1f and W = %.1f that means n > %.0f.\n",
                  R, W, 2.0 * (R + window * W) / std::sqrt(3.0));
      return 1;
    }
  }

  std::printf("  step      p_in         p_out        sigma        err      max|u|      R_eff\n");
  std::printf("  ---------------------------------------------------------------------------\n");
  const int every = report > 0 ? report : steps;
  for (int t = 0; t <= steps; ++t) {
    if (t % every == 0 || t == steps) {
      backend::sync();
      const Out o = measure();
      std::printf("  %-9d %-12.6e %-12.6e %-12.6e %+7.2f%%  %-11.3e %.3f\n",
                  t, o.p_in, o.p_out, o.sigma,
                  100.0 * (o.sigma - sigma) / sigma, o.umax, o.r_eff);
      if (o.bad) { std::printf("  DIVERGED\n"); break; }
    }
    if (t == steps) break;
    cg.refresh();
    cg.step();
  }

  backend::sync();
  const Out o = measure();
  const double err = 100.0 * (o.sigma - sigma) / sigma;
  std::printf("\n  Laplace tension  %.6e measured against %.6e asked   %+.2f%%\n",
              o.sigma, sigma, err);
  std::printf("  effective radius %.3f against %.1f seeded\n", o.r_eff, R);
  std::printf("  spurious current %.3e\n", o.umax);
  std::printf("  mean blue density in the core %.3e\n", o.minor);
  return 0;
}
