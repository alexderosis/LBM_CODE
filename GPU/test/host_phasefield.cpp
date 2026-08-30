//==============================================================================
//  Phase-field checks for the CUDA port, compiled and run WITHOUT a GPU.
//
//  THE SHARPEST CHECK ON THIS MODEL IS THAT THE INTERFACE DOES NOT MOVE OR
//  SPREAD. The conservative Allen-Cahn form exists so that the tanh profile of
//  width W is an exact equilibrium: for it, |grad phi| = (4/W) phi (1 - phi)
//  identically, so the diffusive and anti-diffusive fluxes cancel term for
//  term. Section 1 asserts that identity on the profile, section 4 runs it and
//  measures the width after 600 steps.
//
//  That is not a formality. The parent recorded a bug here that is loud in its
//  effect and silent in its cause: with a Guo prefactor and a 1/cs2 the source
//  comes out a factor M/cs2 too small, the anti-diffusion cannot balance the
//  diffusion, and W = 4 became 14.1 in 600 steps -- against the sqrt(4 M t) = 11
//  of pure diffusion. Nothing else in the model looks wrong while that happens.
//
//  Build:  c++ -std=c++17 -O2 -I../include host_phasefield.cpp -o host_phasefield
//==============================================================================
#include "lbm/phasefield.cuh"
#include "lbm/hostsim.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace lbm;

static int failures = 0;
static void check(bool ok, const char* what, double detail = 0.0) {
  if (ok) std::printf("  PASS  %s\n", what);
  else  { std::printf("  FAIL  %s   (%.3e)\n", what, detail); ++failures; }
}
static double worst(double a, double b) { return std::fabs(a) > std::fabs(b) ? a : b; }

int main() {
  std::printf("Phase-field checks, host build, Real = %s\n",
              sizeof(Real) == 4 ? "float" : "double");
  const double eps = (sizeof(Real) == 4) ? 2e-5 : 1e-12;

  //===========================================================================
  std::printf("\n1. the tanh profile is an exact equilibrium\n\n");
  //===========================================================================
  {
    // For phi = 1/2 [1 + tanh(2x/W)],  dphi/dx = (2/W) phi (1 - phi) * 2 = ...
    // precisely |grad phi| = (4/W) phi (1 - phi). Check it analytically first,
    // then check that anti_diffusion() returns grad phi itself on that profile,
    // which is the cheapest possible test of the function.
    PhaseModel pm;  pm.width = Real(4);  pm.omega = Real(0.8);
    double wid = 0, wa = 0;
    for (double x = -6; x <= 6; x += 0.5) {
      const double ph = 0.5 * (1.0 + std::tanh(2.0 * x / 4.0));
      const double dp = (1.0 / 4.0) / std::cosh(2.0 * x / 4.0)
                                   / std::cosh(2.0 * x / 4.0);
      const double theta = (4.0 / 4.0) * ph * (1.0 - ph);
      wid = worst(wid, dp - theta);
      const Real G[3] = {Real(dp), Real(0), Real(0)};
      Real A[3];
      pm.anti_diffusion(Real(ph), G, A);
      wa = worst(wa, double(A[0]) - dp);
    }
    check(std::fabs(wid) < 1e-12, "|grad phi| == (4/W) phi (1-phi) on the profile", wid);
    check(std::fabs(wa) < 1e-6, "so A = theta n IS grad phi there", wa);
  }

  //===========================================================================
  std::printf("\n2. the source conserves phi, and the equilibrium carries it\n\n");
  //===========================================================================
  {
    PhaseModel pm;  pm.width = Real(4);  pm.omega = Real(0.7);
    // sum_i S_i = 0 with S_i = pref w_i (c_i . A). That is the "conservative"
    // in conservative Allen-Cahn: phi survives the source exactly.
    const Real A[3] = {Real(0.3), Real(-0.2), Real(0.1)};
    double s = 0;
    for (int i = 0; i < PhaseLattice::Q; ++i) {
      const Real cA = Real(PhaseLattice::cx(i)) * A[0]
                    + Real(PhaseLattice::cy(i)) * A[1]
                    + Real(PhaseLattice::cz(i)) * A[2];
      s += PhaseLattice::w(i) * double(cA);
    }
    check(std::fabs(s) < eps, "sum_i S_i = 0, so phi is conserved exactly", s);

    // and its first moment is A itself, which is what the Chapman-Enskog
    // matching in the banner assumed.
    double m[3] = {0, 0, 0};
    for (int i = 0; i < PhaseLattice::Q; ++i) {
      const Real cA = Real(PhaseLattice::cx(i)) * A[0]
                    + Real(PhaseLattice::cy(i)) * A[1]
                    + Real(PhaseLattice::cz(i)) * A[2];
      const double si = PhaseLattice::w(i) * double(cA) * PhaseLattice::inv_cs2();
      m[0] += si * PhaseLattice::cx(i);
      m[1] += si * PhaseLattice::cy(i);
      m[2] += si * PhaseLattice::cz(i);
    }
    double wm = 0;
    for (int a = 0; a < 3; ++a) wm = worst(wm, m[a] - double(A[a]));
    check(std::fabs(wm) < 1e-6, "sum_i c_i S_i / cs2 = A", wm);

    // g^eq: zeroth moment phi, first moment phi u.
    const Real ph = Real(0.37);
    const Real u[3] = {Real(0.03), Real(-0.02), Real(0.015)};
    double z = 0, p1[3] = {0, 0, 0};
    for (int i = 0; i < PhaseLattice::Q; ++i) {
      const double e = PhaseModel::eq(i, ph, u[0], u[1], u[2]);
      z += e;
      p1[0] += e * PhaseLattice::cx(i);
      p1[1] += e * PhaseLattice::cy(i);
      p1[2] += e * PhaseLattice::cz(i);
    }
    check(std::fabs(z - double(ph)) < 1e-6, "sum g^eq = phi", z - double(ph));
    double wp = 0;
    for (int a = 0; a < 3; ++a) wp = worst(wp, p1[a] - double(ph) * double(u[a]));
    check(std::fabs(wp) < 1e-6, "sum c g^eq = phi u", wp);

    // mobility round trip, on D3Q7's cs2 = 1/4 and not the fluid's 1/3.
    const Real om = PhaseModel::omega_from_mobility(Real(0.05));
    check(std::fabs(double(PhaseModel::mobility_from_omega(om)) - 0.05) < 1e-6,
          "M = cs2 (1/omega - 1/2) inverts, with cs2 = 1/4",
          double(PhaseModel::mobility_from_omega(om)) - 0.05);
  }

  //===========================================================================
  std::printf("\n3. the fluid equilibrium carries p~ and u, not rho u\n\n");
  //===========================================================================
  {
    const Real pt = Real(0.37);
    const Real u[3] = {Real(0.05), Real(-0.03), Real(0.02)};
    double z = 0, m1[3] = {0, 0, 0};
    for (int i = 0; i < 27; ++i) {
      const double e = FluidLattice::w(i) * (double(pt) + pf_phi_eq(i, u[0], u[1], u[2]));
      z += e;
      m1[0] += e * FluidLattice::cx(i);
      m1[1] += e * FluidLattice::cy(i);
      m1[2] += e * FluidLattice::cz(i);
    }
    check(std::fabs(z - double(pt)) < 1e-6, "sum f^eq = p~ exactly (Phi sums to zero)",
          z - double(pt));
    double wm = 0;
    for (int a = 0; a < 3; ++a) wm = worst(wm, m1[a] - double(u[a]));
    check(std::fabs(wm) < 1e-6, "sum c f^eq = u, NOT rho u", wm);

    // The interpolant is clamped, so rho stays between the two phases even when
    // phi overshoots -- a large enough undershoot would otherwise make rho
    // negative and 1/rho unbounded.
    MultiphaseModel fm;
    fm.rho_L = Real(1);  fm.rho_H = Real(100);
    fm.mu_L = Real(0.01);  fm.mu_H = Real(0.05);
    bool inrange = true;
    for (double p : {-0.7, -0.05, 0.0, 0.5, 1.0, 1.05, 1.9}) {
      const double r = double(fm.local(Real(p)).rho);
      if (r < 1.0 - 1e-9 || r > 100.0 + 1e-9) inrange = false;
    }
    check(inrange, "rho stays inside [rho_L, rho_H] for phi outside [0,1]");
    check(std::fabs(double(fm.drho_dphi()) - 99.0) < 1e-6, "drho/dphi = (rho_H-rho_L)/(phi_H-phi_L)",
          double(fm.drho_dphi()) - 99.0);
    check(!fm.constant_reference(), "local-rho normalisation is the default");
    fm.rho_0 = Real(1);
    check(fm.constant_reference(), "a positive rho_0 selects the constant one");
  }

  //===========================================================================
  std::printf("\n4. THE ONE THAT MATTERS: a flat interface must hold its width\n\n");
  //===========================================================================
  {
    const int nx = 64, ny = 6, nz = 6;
    const double W = 4.0;
    host::PhaseField pf(nx, ny, nz);
    pf.phase.width = Real(W);
    pf.phase.omega = Real(PhaseModel::omega_from_mobility(Real(0.05)));
    // No surface tension and no density ratio: the fluid must stay quiescent so
    // this measures the phase field alone.
    pf.fluid.rho_L = Real(1);  pf.fluid.rho_H = Real(1);
    pf.fluid.mu_L = Real(0.1); pf.fluid.mu_H = Real(0.1);
    pf.fluid.beta = Real(0);   pf.fluid.kappa = Real(0);

    const double xa = 20.0, xb = 44.0;
    pf.initialise_with([&](int x, int, int, Real& ph, Real& pt) {
      const double s = 0.5 * (std::tanh(2.0 * (x - xa) / W)
                            - std::tanh(2.0 * (x - xb) / W));
      ph = Real(s);  pt = Real(0);
    });

    auto width_now = [&]() {
      std::vector<Real> phi;
      pf.field_to_host(pf.phi_device(), phi);
      // W = 1 / max|dphi/dx| for the tanh profile, measured near the left face.
      double dmax = 0;
      for (int x = 10; x < 32; ++x) {
        const long a = node_id(x - 1, ny / 2, nz / 2, nx, ny);
        const long b = node_id(x + 1, ny / 2, nz / 2, nx, ny);
        dmax = std::max(dmax, std::fabs(0.5 * (double(phi[b]) - double(phi[a]))));
      }
      return 1.0 / dmax;
    };

    const double w0 = width_now();
    const double m0 = pf.total_phase();
    for (int t = 0; t < 600; ++t) pf.step();
    const double w1 = width_now();
    const double m1 = pf.total_phase();
    for (int t = 0; t < 600; ++t) pf.step();
    const double m2 = pf.total_phase();

    std::printf("        (W seeded %.2f, measured %.2f -> %.2f after 600 steps)\n",
                W, w0, w1);
    check(std::fabs(w1 - w0) / w0 < 0.05,
          "the interface neither spreads nor sharpens", (w1 - w0) / w0);
    // Pure diffusion at this mobility would reach sqrt(4 M t) = 11, so this is
    // a wide margin against the failure mode, not a tight one against noise.
    check(w1 < 6.0, "and is nowhere near the sqrt(4 M t) = 11 of pure diffusion", w1);
    // PHI CONSERVATION IS EXACT IN THE ALGORITHM -- sum_i S_i = 0 and both
    // streaming and collision are conservative -- and FP32 cannot represent
    // that. So the test bounds the drift PER STEP against one ulp of the
    // working precision: round-off cannot do better and a real conservation bug
    // would exceed it by orders of magnitude. Two windows are measured rather
    // than one so that a leak, which would grow, is separated from round-off,
    // which does not have to.
    //
    // Measured in FP32 over 150 / 600 / 2400 / 9600 steps, the TOTAL drift runs
    // -2.9e-06, -3.4e-05, -1.0e-04, -1.4e-04: it grows fast while the seeded
    // profile relaxes onto the discrete equilibrium and then nearly stops, so
    // quadrupling the run from 2400 to 9600 costs only 30% more. In FP64 it
    // stays at 1e-13 for all four. Neither is a leak; the FP32 figure is the
    // price of the precision and is worth knowing before a long run.
    const double d1 = (m1 - m0) / m0, d2 = (m2 - m1) / m0;
    const double ulp = (sizeof(Real) == 4) ? 1.19e-7 : 2.2e-16;
    check(std::fabs(d1) / 600.0 < ulp, "phi conserved to under one ulp per step",
          std::fabs(d1) / 600.0 / ulp);
    check(std::fabs(d2) / 600.0 < ulp, "  ... and still so over the next 600",
          std::fabs(d2) / 600.0 / ulp);
    std::printf("        (total phi %.10f -> %.10f -> %.10f;"
                " %.2f then %.2f ulp/step)\n", m0, m1, m2,
                std::fabs(d1) / 600.0 / ulp, std::fabs(d2) / 600.0 / ulp);

    double umax = 0;
    std::vector<Real> ux, uy, uz;
    pf.field_to_host(pf.ux_device(), ux);
    pf.field_to_host(pf.uy_device(), uy);
    pf.field_to_host(pf.uz_device(), uz);
    for (long n = 0; n < pf.nodes(); ++n)
      umax = std::max(umax, std::sqrt(double(ux[n])*double(ux[n])
                                    + double(uy[n])*double(uy[n])
                                    + double(uz[n])*double(uz[n])));
    check(umax < 1e-6, "and the fluid stayed quiescent", umax);
  }

  //===========================================================================
  std::printf("\n5. a droplet at a density ratio stays a droplet\n\n");
  //===========================================================================
  {
    const int n = 32;
    const double R = 8.0, W = 4.0, sigma = 1e-3, gamma = 10.0;
    host::PhaseField pf(n, n, n);
    pf.phase.width = Real(W);
    pf.phase.omega = Real(PhaseModel::omega_from_mobility(Real(0.05)));
    pf.fluid.rho_L = Real(1);  pf.fluid.rho_H = Real(gamma);
    pf.fluid.mu_L = Real(0.01); pf.fluid.mu_H = Real(0.1);
    pf.fluid.beta  = Real(MultiphaseModel::beta_from_sigma(Real(sigma), Real(W)));
    pf.fluid.kappa = Real(MultiphaseModel::kappa_from_sigma(Real(sigma), Real(W)));
    pf.enable_viscous_force(true);

    const double c = 0.5 * n;
    pf.initialise_with([&](int x, int y, int z, Real& ph, Real& pt) {
      const double dx = x - c, dy = y - c, dz = z - c;
      const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
      ph = Real(0.5 * (1.0 - std::tanh(2.0 * (r - R) / W)));
      pt = Real(0);              // the gauge; see the banner
    });

    const double m0 = pf.total_phase();
    bool finite = true;
    for (int t = 0; t < 400 && finite; ++t) {
      pf.step();
      if (t % 100 == 99) {
        std::vector<Real> phi;
        pf.field_to_host(pf.phi_device(), phi);
        for (long k = 0; k < pf.nodes(); ++k)
          if (!std::isfinite(double(phi[k]))) { finite = false; break; }
      }
    }
    const double m1 = pf.total_phase();
    std::vector<Real> phi, ux, uy, uz;
    pf.field_to_host(pf.phi_device(), phi);
    pf.field_to_host(pf.ux_device(), ux);
    pf.field_to_host(pf.uy_device(), uy);
    pf.field_to_host(pf.uz_device(), uz);
    const double pc = double(phi[node_id(n/2, n/2, n/2, n, n)]);
    const double pe = double(phi[node_id(0, 0, 0, n, n)]);
    double umax = 0;
    for (long k = 0; k < pf.nodes(); ++k)
      umax = std::max(umax, std::sqrt(double(ux[k])*double(ux[k])
                                    + double(uy[k])*double(uy[k])
                                    + double(uz[k])*double(uz[k])));

    check(finite, "400 steps at a density ratio of 10 stay finite");
    const double tol = (sizeof(Real) == 4) ? 2e-5 : 1e-12;
    check(std::fabs(m1 - m0) < tol * std::fabs(m0),
          "phi conserved through the coupled step", (m1 - m0) / m0);
    check(pc > 0.9, "heavy phase at the centre", pc);
    check(pe < 0.1, "light phase in the corner", pe);
    std::printf("        (phi centre %.4f, corner %.4f)\n", pc, pe);
    check(umax < 0.05, "spurious current stays small", umax);
    std::printf("        (max |u| = %.4e)\n", umax);
  }

  std::printf("\n[phasefield] %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
