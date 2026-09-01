//==============================================================================
//  The phase-field collision operators, checked against their own algebra.
//
//  PhaseFieldCentralMoments is De Rosis & Enan (2021) Sec. II.D, and its whole
//  content is a claim about moments: keep phi, relax the three first-order
//  central moments at omega_phi, send everything else to equilibrium. Every
//  block below pins one part of that claim down by direct contraction over the
//  velocity set, because a wrong factor there is silent -- it changes the
//  mobility, the interface then spreads or sharpens over thousands of steps,
//  and the first symptom is a validation number that is merely worse.
//
//  What each block establishes, and how it fails:
//    1. The equilibrium's own moments: sum_i g^eq = phi and sum_i c_i g^eq =
//       phi u. A wrong inverse transform breaks the second and not the first.
//    2. The equilibrium central moments are (phi, 0, ..., 0) in the SHIFTED
//       product basis. This is the step that makes the scheme short, and it is
//       the paper's Eq. (54) re-expressed -- their five nonzero monomial
//       moments collapse to one here. If this fails the banner's derivation is
//       wrong, not the code.
//    3. Equilibrium is a fixed point with no source, at every omega.
//    4. phi is conserved by collision -- with a source and without. This is the
//       "conservative" in conservative Allen-Cahn and it must hold exactly,
//       not to truncation.
//    5. The source delivers exactly (1 - omega/2) A to the first moment, which
//       is the prefactor PhaseFieldBGK derives and the one thing the two
//       operators must share if a difference between them is to mean anything.
//    6. Against PhaseFieldBGK at matched omega: the two agree to the O(u^3)
//       gap between the product-form and second-order equilibria and no more.
//       That is what says they are the same scheme with different ghost
//       handling rather than two different models.
//    7. The mobility round-trip, omega <-> M, on both operators.
//==============================================================================
#include "Check.hpp"
#include "collision/PhaseFieldBGK.hpp"
#include "collision/PhaseFieldCentralMoments.hpp"
#include "core/Types.hpp"

#include <algorithm>
#include <cmath>

using namespace lbm;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using L  = D3Q27;
    using CM = PhaseFieldCentralMoments<L>;
    using BG = PhaseFieldBGK<L>;
    using B  = ProductBasis<L>;
    const Real tol = Real(1e-12);

    const Real phi = Real(0.37);
    const Real u[3] = {Real(0.05), Real(-0.03), Real(0.02)};

    //-------------------------------------------------------------- 1
    std::printf("\n-- equilibrium moments --\n");
    Real ge[L::Q];
    CM::eq_populations(phi, u, ge);
    {
      Real s = Real(0), m[3] = {Real(0), Real(0), Real(0)};
      for (int i = 0; i < L::Q; ++i) {
        s += ge[i];
        for (int a = 0; a < 3; ++a) m[a] += ge[i] * Real(cvel<L>(i, a));
      }
      check::near(s, phi, tol, "sum_i g^eq = phi");
      check::near(m[0], phi * u[0], tol, "sum_i c_x g^eq = phi u_x");
      check::near(m[1], phi * u[1], tol, "sum_i c_y g^eq = phi u_y");
      check::near(m[2], phi * u[2], tol, "sum_i c_z g^eq = phi u_z");
    }

    //-------------------------------------------------------------- 2
    std::printf("\n-- equilibrium central moments are (phi, 0, ..., 0) --\n");
    {
      Real k[CM::NM];
      B::to_moments(ge, u, k);
      check::near(k[B::index_of(0, 0, 0)], phi, tol, "k^eq_000 = phi");
      double worst = 0;
      for (int n = 0; n < CM::NM; ++n)
        if (n != B::index_of(0, 0, 0)) worst = std::max(worst, std::abs(double(k[n])));
      check::near(Real(worst), Real(0), tol,
                  "every other equilibrium central moment vanishes");
    }

    //-------------------------------------------------------------- 3
    std::printf("\n-- equilibrium is a fixed point --\n");
    for (double w : {0.2, 1.0, 1.6, 1.9982}) {
      CM cm; cm.omega = Real(w); cm.width = Real(4);
      Real g[L::Q];
      for (int i = 0; i < L::Q; ++i) g[i] = ge[i];
      const Real A[3] = {Real(0), Real(0), Real(0)};
      cm.collide(g, phi, u, A);
      double worst = 0;
      for (int i = 0; i < L::Q; ++i) worst = std::max(worst, std::abs(double(g[i] - ge[i])));
      check::near(Real(worst), Real(0), tol,
                  "omega = " + std::to_string(w) + ": g^eq unchanged");
    }

    //-------------------------------------------------------------- 4 and 5
    std::printf("\n-- conservation, and what the source delivers --\n");
    {
      CM cm; cm.omega = Real(1.4); cm.width = Real(4);
      Real g[L::Q];
      for (int i = 0; i < L::Q; ++i)          // a deliberately off-equilibrium state
        g[i] = ge[i] * (Real(1) + Real(0.05) * Real((i * 7) % 11 - 5));
      Real p0 = Real(0);
      for (int i = 0; i < L::Q; ++i) p0 += g[i];

      Real before[3] = {Real(0), Real(0), Real(0)};
      for (int i = 0; i < L::Q; ++i)
        for (int a = 0; a < 3; ++a) before[a] += g[i] * Real(cvel<L>(i, a));

      const Real A[3] = {Real(0.011), Real(-0.007), Real(0.004)};
      Real g2[L::Q];
      for (int i = 0; i < L::Q; ++i) g2[i] = g[i];
      cm.collide(g2, p0, u, A);

      Real p1 = Real(0), after[3] = {Real(0), Real(0), Real(0)};
      for (int i = 0; i < L::Q; ++i) {
        p1 += g2[i];
        for (int a = 0; a < 3; ++a) after[a] += g2[i] * Real(cvel<L>(i, a));
      }
      check::near(p1, p0, tol, "phi conserved by collision WITH a source");

      // The first RAW moment after collision, predicted from the scheme:
      //   k*_a = (1-omega) k_a + (1-omega/2) A_a,   r*_a = k*_a + phi u_a.
      const Real pref = (Real(1) - Real(0.5) * cm.omega) * cs2<L, Real>();
      for (int a = 0; a < 3; ++a) {
        const Real k_a = before[a] - p0 * u[a];
        const Real want = (Real(1) - cm.omega) * k_a + pref * A[a] + p0 * u[a];
        check::near(after[a], want, tol,
                    "first moment " + std::string(1, char('x' + a)) +
                    " is (1-omega) k + (1-omega/2) cs2 A");
      }

      // and without a source phi is still conserved
      Real g3[L::Q];
      for (int i = 0; i < L::Q; ++i) g3[i] = g[i];
      const Real Z[3] = {Real(0), Real(0), Real(0)};
      cm.collide(g3, p0, u, Z);
      Real p2 = Real(0);
      for (int i = 0; i < L::Q; ++i) p2 += g3[i];
      check::near(p2, p0, tol, "phi conserved by collision with NO source");
    }

    //-------------------------------------------------------------- 6
    std::printf("\n-- against PhaseFieldBGK at matched omega --\n");
    {
      const Real w = Real(1.0);        // at omega = 1 both send everything to eq
      CM cm; cm.omega = w; cm.width = Real(4);
      BG bg; bg.omega = w; bg.width = Real(4);
      Real ga[L::Q], gb[L::Q];
      for (int i = 0; i < L::Q; ++i) {
        ga[i] = ge[i] * (Real(1) + Real(0.03) * Real((i * 5) % 7 - 3));
        gb[i] = ga[i];
      }
      Real p = Real(0);
      for (int i = 0; i < L::Q; ++i) p += ga[i];
      const Real A[3] = {Real(0.01), Real(-0.006), Real(0.003)};
      cm.collide(ga, p, u, A);
      bg.collide(gb, p, u, A);
      double worst = 0;
      for (int i = 0; i < L::Q; ++i)
        worst = std::max(worst, std::abs(double(ga[i] - gb[i])));
      // The tolerance is |u|^3 itself, not a round number: the two equilibria
      // are the product form and its second-order truncation, which differ at
      // third order in u, so the post-collision gap MUST be O(|u|^3) and must
      // not exceed it. A fixed tolerance would either pass a real defect at
      // small |u| or fail this at large.
      const double u3 = std::pow(std::sqrt(double(u[0] * u[0] + u[1] * u[1] +
                                                  u[2] * u[2])), 3.0);
      check::near(Real(worst), Real(0), Real(u3),
                  "post-collision states agree to within |u|^3, the equilibrium gap");
      std::printf("        (gap %.3e, |u|^3 = %.3e)\n", worst, u3);

      // The agreement must be in the MOMENTS that matter, exactly, not just
      // approximately: both must conserve phi and both must put the same
      // source into the first moment.
      Real pa = Real(0), pb = Real(0), ma = Real(0), mb = Real(0);
      for (int i = 0; i < L::Q; ++i) {
        pa += ga[i]; pb += gb[i];
        ma += ga[i] * Real(cvel<L>(i, 0));
        mb += gb[i] * Real(cvel<L>(i, 0));
      }
      check::near(pa, pb, tol, "both conserve phi to the same value");
      check::near(ma, mb, Real(1e-13), "both give the same first moment");
    }

    //-------------------------------------------------------------- 7
    std::printf("\n-- mobility round trip --\n");
    for (double m : {1e-5, 1e-3, 1e-1}) {
      check::near(double(CM::mobility_from_omega(CM::omega_from_mobility(Real(m)))),
                  m, 1e-14, "CM  M -> omega -> M at M = " + std::to_string(m));
      check::near(double(BG::mobility_from_omega(BG::omega_from_mobility(Real(m)))),
                  m, 1e-14, "BGK M -> omega -> M at M = " + std::to_string(m));
    }

    //-------------------------------------------------------------- 8
    // Their Eq. (61) lists NINE nonzero source entries: R_{1,2,3} = F and the
    // six third-order R_{aab} = F_b cs2. That is a statement about their
    // MONOMIAL basis, and reading it as a statement about this operator is a
    // trap worth pinning shut, because the two bases disagree about where the
    // same source lives. In the shifted basis phi_2 = C^2 - cs2 the (a,a,b)
    // slot is (C_a^2 - cs2) C_b, whose source moment -- AT u = 0, which is
    // where both this operator and theirs evaluate it -- is
    //
    //     cs4 A_b  -  cs2 * cs2 A_b  =  0,
    //
    // so all six vanish identically and the source really is confined to the
    // three first-order slots here. Both halves are checked: the raw moments
    // reproduce their Eq. (61), and the shifted ones outside the first three
    // are zero to round-off. Get this backwards in either direction and the
    // anti-diffusion is wrong by a factor with no crash to announce it.
    std::printf("\n-- where the source lives, in two bases --\n");
    {
      using B = CM::Basis;
      const Real A[3] = {Real(0.037), Real(-0.021), Real(0.014)};
      const Real u0[3] = {Real(0), Real(0), Real(0)};
      const Real om = Real(1.7);
      const double sp = 1.0 - 0.5 * double(om);
      Real S[L::Q];
      for (int i = 0; i < L::Q; ++i) {
        const Real cA = Real(cvel<L>(i, 0)) * A[0] + Real(cvel<L>(i, 1)) * A[1] +
                        Real(cvel<L>(i, 2)) * A[2];
        S[i] = Real(sp) * weight<L, Real>(i) * cA;
      }
      constexpr double cs2d = double(cs2<L, double>());
      const double cs4 = cs2d * cs2d;
      const char* nm[6] = {"xxy", "xyy", "xxz", "xzz", "yyz", "yzz"};
      const int   e[6][3] = {{2,1,0},{1,2,0},{2,0,1},{1,0,2},{0,2,1},{0,1,2}};
      const int   b[6]    = {1, 0, 2, 0, 2, 1};
      for (int t = 0; t < 6; ++t) {
        double raw = 0;
        for (int i = 0; i < L::Q; ++i) {
          double m = double(S[i]);
          for (int d = 0; d < 3; ++d)
            for (int q = 0; q < e[t][d]; ++q) m *= double(cvel<L>(i, d));
          raw += m;
        }
        check::near(raw, sp * cs4 * double(A[b[t]]), 1e-15,
                    std::string("raw R_") + nm[t] + " = (1-w/2) cs4 A_" + "xyz"[b[t]]);
      }
      Real r[CM::NM];
      B::to_moments(S, u0, r);
      for (int d = 0; d < 3; ++d) {
        const int idx = B::index_of(d == 0, d == 1, d == 2);
        check::near(double(r[idx]), sp * cs2d * double(A[d]), 1e-15,
                    std::string("shifted R_") + "xyz"[d] + " = (1-w/2) cs2 A");
      }
      double above = 0;
      for (int n = 0; n < CM::NM; ++n) {
        if (n == B::index_of(0,0,0) || n == B::index_of(1,0,0) ||
            n == B::index_of(0,1,0) || n == B::index_of(0,0,1)) continue;
        above = std::max(above, std::abs(double(r[n])));
      }
      check::near(above, 0.0, 1e-15,
                  "AT u=0 the shift absorbs their six third-order entries");
      std::printf("        (worst shifted slot outside the first three: %.3e)\n", above);
    }
  }
  const int rc = check::report("phase_field");
  Kokkos::finalize();
  return rc;
}
