//==============================================================================
//  The colour-gradient operator, checked against its own algebra.
//
//  Saito et al., Phys. Rev. E 98, 013305 (2018). No flow, no time stepping: this
//  file establishes the identities the model rests on, each of which can be
//  wrong in a way a running simulation would absorb rather than report.
//
//   1. phi_i is a weight set. It must sum to one for ANY alpha, and its second
//      moment must be 9(1-alpha)/19 -- that number IS the sound speed, and the
//      density ratio is obtained from it through p = rho cs^2 and nothing else.
//      Get it wrong and the ratio is wrong by exactly the same factor, silently,
//      because every other test still passes.
//   2. The equilibrium carries rho and rho u and nothing else in its first two
//      moments, at third order in u and with the Galilean term Phi_i present.
//      Phi_i is the term most easily dropped, and dropping it changes nothing
//      that is checked anywhere else in a static test.
//   3. The perturbation conserves mass and momentum, and acts only on the second
//      moment. sum_i B_i = 1/3 against sum_i w_i (c.n)^2 = 1/3 is the identity
//      that makes it so, and it holds for EVERY direction of n.
//   4. Recolouring is a partition: f^r + f^b = f identically, for any beta and
//      any gradient. Mass and momentum cannot be lost there whatever else is
//      wrong, and this says so.
//   5. Equilibrium is a fixed point. With no perturbation, no force and no
//      density gradient, collision must return the equilibrium unchanged at any
//      relaxation rate.
//   6. The capillary stress of Eq. (39) matches sigma = 4 A tau / 9. This is the
//      only closed form the model offers for the interfacial tension, and
//      validation/static_droplet.cpp measures the same number the hard way.
//==============================================================================
#include "Check.hpp"
#include "collision/ColourGradient.hpp"
#include "grid/Domain.hpp"
#include "core/Types.hpp"

#include <cmath>

using namespace lbm;
using L  = D3Q27;
using CG = ColourGradient<L>;

static Real TOL() { return sizeof(Real) == 4 ? Real(2e-5) : Real(1e-12); }

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const Real tol = TOL();

    //--------------------------------------------------------------------------
    std::printf("\n1. phi_i is a weight set, and its second moment is the sound speed\n");
    {
      const double alphas[4] = {8.0 / 27.0, 0.2, 0.5, 0.9};
      for (int k = 0; k < 4; ++k) {
        const Real a = Real(alphas[k]);
        double s = 0, m2[3] = {0, 0, 0}, off = 0, odd = 0;
        for (int i = 0; i < L::Q; ++i) {
          const double p = double(CG::phi_i(i, a));
          s += p;
          for (int d = 0; d < 3; ++d) m2[d] += p * cvel<L>(i, d) * cvel<L>(i, d);
          off += p * cvel<L>(i, 0) * cvel<L>(i, 1);
          odd += p * cvel<L>(i, 0);
        }
        const double cs2 = double(CG::cs2_of_alpha(a));
        char b[128];
        std::snprintf(b, sizeof b, "sum phi_i = 1            (alpha = %.4f)", alphas[k]);
        check::near(s, 1.0, double(tol), b);
        std::snprintf(b, sizeof b, "second moment = 9(1-a)/19 (alpha = %.4f)", alphas[k]);
        check::near(m2[0], cs2, double(tol), b);
        check::near(m2[1], cs2, double(tol), "  ... isotropic in y");
        check::near(m2[2], cs2, double(tol), "  ... isotropic in z");
        check::near(off, 0.0, double(tol), "  ... no off-diagonal part");
        check::near(odd, 0.0, double(tol), "  ... odd moments vanish");
      }
      // alpha_b = 8/27 is the paper's choice because it makes the blue fluid an
      // ordinary lattice; if that stops being true the density ratio moves.
      check::near(double(CG::cs2_of_alpha(Real(8) / Real(27))), 1.0 / 3.0,
                  double(tol), "alpha = 8/27 gives cs^2 = 1/3 exactly");
      // and the ratio relation of Eq. (25) inverts consistently
      for (double g : {1.0, 10.0, 100.0, 1000.0}) {
        const Real ab = Real(8) / Real(27);
        const Real ar = CG::alpha_r_from_ratio(Real(g), ab);
        const double back = double((Real(1) - ab) / (Real(1) - ar));
        char b[128];
        std::snprintf(b, sizeof b, "gamma = (1-ab)/(1-ar) inverts   (gamma = %.0f)", g);
        check::near(back, g, g * 1e-12, b);
      }
    }

    //--------------------------------------------------------------------------
    std::printf("\n2. the equilibrium carries rho and rho u, Phi_i included\n");
    {
      CG cg;
      cg.alpha_r = Real(0.1);  cg.alpha_b = Real(8) / Real(27);
      cg.rho_r0 = Real(1);     cg.rho_b0 = Real(1);
      const Real rho = Real(1.7);
      const Real rr_ = Real(1.1), rb_ = rho - rr_;
      const Real u[3] = {Real(0.11), Real(-0.07), Real(0.05)};
      const Real dr[3] = {Real(0.3), Real(-0.2), Real(0.13)};   // grad rho
      Real G[3][3];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) G[i][j] = (u[i] * dr[j] + u[j] * dr[i]) / Real(48);
      const Real udr = u[0] * dr[0] + u[1] * dr[1] + u[2] * dr[2];
      const Real nub = Real(0.13);

      Real fe[L::Q];
      cg.equilibrium(fe, rr_, rb_, u, G, nub, udr);
      double s = 0, mx = 0, my = 0, mz = 0;
      for (int i = 0; i < L::Q; ++i) {
        s += double(fe[i]);
        mx += double(fe[i]) * cvel<L>(i, 0);
        my += double(fe[i]) * cvel<L>(i, 1);
        mz += double(fe[i]) * cvel<L>(i, 2);
      }
      check::near(s, double(rho), double(tol) * 10, "sum f^eq = rho");
      check::near(mx, double(rho * u[0]), double(tol) * 10, "sum c_x f^eq = rho u_x");
      check::near(my, double(rho * u[1]), double(tol) * 10, "sum c_y f^eq = rho u_y");
      check::near(mz, double(rho * u[2]), double(tol) * 10, "sum c_z f^eq = rho u_z");

      // Phi_i on its own must be invisible to both conserved moments, which is
      // what lets it correct the STRESS without disturbing anything else.
      Real fz[L::Q];
      const Real zeroG[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      cg.equilibrium(fz, rr_, rb_, u, zeroG, Real(0), Real(0));
      double ds = 0, dm = 0, dstress = 0;
      for (int i = 0; i < L::Q; ++i) {
        const double d = double(fe[i]) - double(fz[i]);
        ds += d;  dm += d * cvel<L>(i, 0);
        dstress += d * cvel<L>(i, 0) * cvel<L>(i, 1);
      }
      check::near(ds, 0.0, double(tol) * 10, "Phi_i changes no density");
      check::near(dm, 0.0, double(tol) * 10, "Phi_i changes no momentum");
      check::ok(std::fabs(dstress) > 1e-6,
                "Phi_i DOES change the stress (it would be dead code otherwise)");
      std::printf("        (its xy stress contribution is %.4e)\n", dstress);

      // The pressure is rho cs^2(alpha), Eq. (26): the trace of the second
      // moment at rest, which is where the density ratio actually comes from.
      Real fr[L::Q];
      const Real u0[3] = {0, 0, 0};
      cg.equilibrium(fr, rr_, rb_, u0, zeroG, Real(0), Real(0));
      double tr = 0;
      for (int i = 0; i < L::Q; ++i)
        tr += double(fr[i]) * cvel<L>(i, 0) * cvel<L>(i, 0);
      // p = sum_k rho_k cs_k^2 -- per colour, which is what keeps the pressure
      // continuous through an interface at a density ratio.
      check::near(tr, double(rr_ * CG::cs2_of_alpha(cg.alpha_r)
                           + rb_ * CG::cs2_of_alpha(cg.alpha_b)),
                  double(tol) * 10, "p = sum_k rho_k cs_k^2 at rest");
    }

    //--------------------------------------------------------------------------
    std::printf("\n3. the perturbation is a pure second-moment source\n");
    {
      double sb = 0;
      for (int i = 0; i < L::Q; ++i) sb += double(CG::B_i(i));
      check::near(sb, 1.0 / 3.0, double(tol), "sum B_i = 1/3");

      // For EVERY direction of n, not just the axes: the cancellation is
      // sum w_i (c.n)^2 = cs^2 = 1/3 and it is direction independent.
      const double dirs[5][3] = {{1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                                 {1, 1, 1}, {0.37, -0.62, 0.19}};
      double worst_m = 0, worst_p = 0, worst_n = 0, least_s = 1e30;
      for (int k = 0; k < 5; ++k) {
        const double nm = std::sqrt(dirs[k][0] * dirs[k][0] + dirs[k][1] * dirs[k][1]
                                  + dirs[k][2] * dirs[k][2]);
        const double nh[3] = {dirs[k][0] / nm, dirs[k][1] / nm, dirs[k][2] / nm};
        // a tangent to n, for the stress component that must NOT vanish
        double t[3] = {-nh[1], nh[0], 0};
        double tm = std::sqrt(t[0] * t[0] + t[1] * t[1]);
        if (tm < 1e-8) { t[0] = 0; t[1] = -nh[2]; t[2] = nh[1]; tm = std::sqrt(t[1]*t[1]+t[2]*t[2]); }
        for (int d = 0; d < 3; ++d) t[d] /= tm;
        double s = 0, p[3] = {0, 0, 0}, snn = 0, stt = 0;
        for (int i = 0; i < L::Q; ++i) {
          const double cn = cvel<L>(i, 0) * nh[0] + cvel<L>(i, 1) * nh[1]
                          + cvel<L>(i, 2) * nh[2];
          const double ct = cvel<L>(i, 0) * t[0] + cvel<L>(i, 1) * t[1]
                          + cvel<L>(i, 2) * t[2];
          const double o = double(weight<L, Real>(i)) * cn * cn - double(CG::B_i(i));
          s += o;
          for (int d = 0; d < 3; ++d) p[d] += o * cvel<L>(i, d);
          snn += o * cn * cn;
          stt += o * ct * ct;
        }
        worst_m = std::max(worst_m, std::fabs(s));
        for (int d = 0; d < 3; ++d) worst_p = std::max(worst_p, std::fabs(p[d]));
        worst_n = std::max(worst_n, std::fabs(snn));
        least_s = std::min(least_s, std::fabs(stt));
      }
      check::near(worst_m, 0.0, double(tol), "conserves mass, every direction of n");
      check::near(worst_p, 0.0, double(tol), "conserves momentum, every direction");
      check::near(worst_n, 0.0, double(tol),
                  "normal capillary stress vanishes, every direction of n");
      check::ok(least_s > 1e-2, "but the tangential one does not");

      // sigma = 4 A tau / 9, Eq. (32), against the capillary stress of Eq. (39),
      // S = -tau sum_i Omega^(2) c_i c_i. For a flat interface with normal n the
      // tension is the difference between the stress along n and across it.
      const double A = 8e-4, tau = 1.0, gm = 1.0;
      double snn = 0, stt = 0;
      for (int i = 0; i < L::Q; ++i) {
        const double cn = cvel<L>(i, 2);
        const double o = double(CG::perturbation_coefficient(Real(A))) * gm
                       * (double(weight<L, Real>(i)) * cn * cn - double(CG::B_i(i)));
        snn += -tau * o * cvel<L>(i, 2) * cvel<L>(i, 2);
        stt += -tau * o * cvel<L>(i, 0) * cvel<L>(i, 0);
      }
      // sigma is the integral of (S_tt - S_nn) THROUGH the interface. The
      // integrand is proportional to |grad phi|, whose integral across the
      // interface is the jump in phi -- and phi runs -1 to +1, so that is 2.
      const double dphi = 2.0;
      const double sigma = (stt - snn) / gm * dphi;
      check::near(sigma, 4.0 * A * tau / 9.0, 1e-15,
                  "capillary stress gives sigma = 4 A tau / 9, Eq. (32)");
      std::printf("        (sigma = %.8e, 4 A tau / 9 = %.8e)\n",
                  sigma, 4.0 * A * tau / 9.0);
    }

    //--------------------------------------------------------------------------
    std::printf("\n4. recolouring is a partition\n");
    {
      Domain d(4, 4, 4, true, true, true);
      CG cg;
      cg.alpha_r = Real(0.1);  cg.alpha_b = Real(8) / Real(27);
      cg.rho_r0 = Real(1);     cg.rho_b0 = Real(1);
      cg.beta = Real(0.7);
      View1D<Real> gx("gx", d.n_padded), gy("gy", d.n_padded), gz("gz", d.n_padded);
      auto hx = Kokkos::create_mirror_view(gx);
      auto hy = Kokkos::create_mirror_view(gy);
      auto hz = Kokkos::create_mirror_view(gz);
      for (Index n = 0; n < d.n_padded; ++n) {
        hx(n) = Real(0.31); hy(n) = Real(-0.12); hz(n) = Real(0.44);
      }
      Kokkos::deep_copy(gx, hx); Kokkos::deep_copy(gy, hy); Kokkos::deep_copy(gz, hz);
      cg.Gx = gx; cg.Gy = gy; cg.Gz = gz;

      Real f[L::Q], fr[L::Q], fb[L::Q];
      for (int i = 0; i < L::Q; ++i)
        f[i] = Real(0.03) * Real((i * 7) % 11 - 5) + weight<L, Real>(i);
      const Real rr = Real(1.3), rb = Real(0.4);
      cg.recolour(f, rr, rb, Real(0.2), 0, fr, fb);
      double worst = 0, ms = 0, px = 0;
      for (int i = 0; i < L::Q; ++i) {
        worst = std::max(worst, std::fabs(double(fr[i]) + double(fb[i]) - double(f[i])));
        ms += double(fr[i]) + double(fb[i]);
        px += (double(fr[i]) + double(fb[i])) * cvel<L>(i, 0);
      }
      check::near(worst, 0.0, double(tol), "f^r + f^b = f, population by population");
      double fs = 0, fx = 0;
      for (int i = 0; i < L::Q; ++i) { fs += double(f[i]); fx += double(f[i]) * cvel<L>(i, 0); }
      check::near(ms, fs, double(tol), "so mass survives it");
      check::near(px, fx, double(tol), "and momentum survives it");
      // and it does something: colour is pushed along the gradient
      double split = 0;
      for (int i = 0; i < L::Q; ++i)
        split = std::max(split, std::fabs(double(fr[i]) - rr / (rr + rb) * double(f[i])));
      check::ok(split > 1e-4, "but it is not the identity: colour moves up grad phi");
    }

    //--------------------------------------------------------------------------
    std::printf("\n5. equilibrium is a fixed point\n");
    {
      Domain d(4, 4, 4, true, true, true);
      CG cg;
      cg.alpha_r = Real(0.15);  cg.alpha_b = Real(8) / Real(27);
      cg.nu_r = Real(0.05);     cg.nu_b = Real(0.05);
      cg.rho_r0 = Real(1);      cg.rho_b0 = Real(1);
      cg.A = Real(0);           // no perturbation
      View1D<Real> z("z", d.n_padded);
      cg.Gx = z; cg.Gy = z; cg.Gz = z;   // no colour gradient
      cg.Rx = z; cg.Ry = z; cg.Rz = z;   // no density gradient

      const Real rho = Real(1.4);
      const Real u[3] = {Real(0.09), Real(-0.06), Real(0.03)};
      for (double p : {-1.0, -0.3, 0.0, 0.5, 1.0}) {
        const Real pp = Real(p);
        const Real rr2 = rho * Real(0.5) * (Real(1) + pp);
        const Real rb2 = rho - rr2;
        const Real zeroG[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        Real fe[L::Q], f[L::Q];
        cg.equilibrium(fe, rr2, rb2, u, zeroG, Real(0), Real(0));
        for (int i = 0; i < L::Q; ++i) f[i] = fe[i];
        cg.collide(f, rr2, rb2, u, pp, 0);
        double worst = 0;
        for (int i = 0; i < L::Q; ++i)
          worst = std::max(worst, std::fabs(double(f[i]) - double(fe[i])));
        char b[128];
        std::snprintf(b, sizeof b, "collision leaves f^eq alone   (phi = %+.1f)", p);
        check::near(worst, 0.0, double(tol) * 100, b);
      }
      // ... and the collision conserves mass and momentum away from equilibrium
      Real f[L::Q];
      const Real zeroG[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      const Real rr3 = rho * Real(0.6), rb3 = rho - rr3;
      cg.equilibrium(f, rr3, rb3, u, zeroG, Real(0), Real(0));
      for (int i = 0; i < L::Q; ++i) f[i] *= Real(1) + Real(0.05) * Real((i * 5) % 7 - 3);
      double s0 = 0, p0[3] = {0, 0, 0};
      for (int i = 0; i < L::Q; ++i) {
        s0 += double(f[i]);
        for (int dd = 0; dd < 3; ++dd) p0[dd] += double(f[i]) * cvel<L>(i, dd);
      }
      cg.collide(f, rr3, rb3, u, Real(0.2), 0);
      double s1 = 0, p1[3] = {0, 0, 0};
      for (int i = 0; i < L::Q; ++i) {
        s1 += double(f[i]);
        for (int dd = 0; dd < 3; ++dd) p1[dd] += double(f[i]) * cvel<L>(i, dd);
      }
      check::near(s1, s0, double(tol) * 100, "collision conserves mass off equilibrium");
      check::near(p1[0], p0[0], double(tol) * 100, "  ... and x momentum");
      check::near(p1[1], p0[1], double(tol) * 100, "  ... and y momentum");
      check::near(p1[2], p0[2], double(tol) * 100, "  ... and z momentum");
    }
  }
  const int rc = check::report("colour_gradient");
  Kokkos::finalize();
  return rc;
}
