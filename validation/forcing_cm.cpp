//==============================================================================
//  Forcing central moments -- D2Q9 and D3Q27 central-moment operators.
//
//  RESULT: the forcing this code already had is EXACTLY the high-order Hermite
//  forcing, on both lattices. No extra term is needed, and adding one double
//  counts.
//
//  Expanding a body force in Hermite polynomials -- to fourth order on D2Q9, to
//  sixth on D3Q27 -- and transforming to MONOMIAL central moments gives a single
//  rule. Write a moment as k_pqr with exponents in {0,1,2}. Then
//
//      k = cs^(2m) * F_a   if exactly one axis a has exponent 1 and m other
//                          axes have exponent 2   (m = 0, 1, 2)
//      k = 0               otherwise.
//
//  On D2Q9 that is [0, Fx, Fy, 0, 0, 0, cs^2 Fy, cs^2 Fx, 0] in the ordering
//  (00,10,01,20,02,11,21,12,22). On D3Q27 it is F_a at first order, six terms at
//  cs^2 F_a, three at cs^4 F_a, and fifteen exact zeros.
//
//  Two features are easy to get wrong and are checked rather than assumed:
//
//    * Moments with no axis at exponent 1 -- k_20, k_02, k_11, k_22 and their 3D
//      analogues -- are EXACTLY zero. In the co-moving frame the force touches
//      only the odd-in-one-axis moments. This holds only if Hermite components
//      with an exponent of 3 or more on a single axis are left OUT of the sum.
//      They are not independent on these lattices: H4(xxxx) reduces to -3 H2(xx),
//      so including it injects a spurious O(F u^3) term into the second moment.
//      H3(xxx) vanishes identically and may be dropped freely.
//    * The cs^2 and cs^4 entries are not truncation error; they are what the
//      lattice can represent.
//
//  THE TRAP is concluding that an operator which puts the force only in the
//  first-order moments therefore misses all of them. It does not, when the basis
//  is Hermite. ProductBasis stores phi_2(C) = C^2 - cs^2 per axis, so expanding
//  C^2 = phi_2 + cs^2 gives, for example,
//
//      k_122(monomial) = k_122 + cs^2 k_120 + cs^2 k_102 + cs^4 k_100
//
//  in that basis -- and with only k_100 = Fx populated, the monomial value is
//  cs^4 Fx, exactly as required. Every cs^(2m) entry is delivered for free by the
//  cs^2 carried inside the basis functions. This test therefore checks BOTH
//  representations of the same populations: monomial must show the full pattern,
//  Hermite must show the force in the first-order slots and nowhere else.
//
//  METHOD. The force contribution is isolated by colliding the SAME macroscopic
//  state twice, once with forcing and once without, and differencing the central
//  moments. Passing Macro explicitly keeps u identical between the two calls, so
//  the equilibrium cancels and only the force term remains. Going through
//  macroscopic() instead would also pick up the F/(2 rho) velocity shift and
//  measure the wrong thing.
//==============================================================================
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace lbm;

// Monomial (basis = 0) or Hermite (basis = 1) central moments about u.
template <class L>
static void cmoments(const Real f[], const Real u[3], int basis,
                     const int (*E)[3], int NM, double k[]) {
  const double cs2v = double(cs2<L, Real>());
  auto phi = [&](int p, double C) {
    return p == 0 ? 1.0 : (p == 1 ? C : C * C - (basis ? cs2v : 0.0));
  };
  for (int m = 0; m < NM; ++m) {
    double s = 0;
    for (int i = 0; i < L::Q; ++i) {
      const double C[3] = {double(cvel<L>(i, 0)) - double(u[0]),
                           double(cvel<L>(i, 1)) - double(u[1]),
                           double(cvel<L>(i, 2)) - double(u[2])};
      s += double(f[i]) * phi(E[m][0], C[0]) * phi(E[m][1], C[1]) * phi(E[m][2], C[2]);
    }
    k[m] = s;
  }
}

template <class L>
static int check(const char* name, const int (*E)[3], int NM) {
  using Forced   = CentralMoments<L, Guo, RawPopulations>;
  using Unforced = CentralMoments<L, NoForcing, RawPopulations>;
  constexpr int D = L::D;
  const double cs2v = double(cs2<L, Real>());

  std::mt19937 rng(2024);
  std::uniform_real_distribution<double> U(-0.10, 0.10), Fd(-2e-3, 2e-3);
  int bad = 0;

  for (int basis = 0; basis < 2; ++basis) {
    double worst = 0; int wj = 0;
    for (int t = 0; t < 200; ++t) {
      const Real rho = Real(1);
      const Real u[3] = {Real(U(rng)), Real(U(rng)), D == 3 ? Real(U(rng)) : Real(0)};
      const double F[3] = {Fd(rng), Fd(rng), D == 3 ? Fd(rng) : 0.0};

      Forced fc;  fc.omega = Real(1.2);
      fc.forcing = Guo{Real(F[0]), Real(F[1]), Real(F[2])};
      Unforced uc; uc.omega = Real(1.2);

      std::vector<Real> f1(L::Q), f2(L::Q), d(L::Q);
      for (int i = 0; i < L::Q; ++i)
        f1[i] = f2[i] = Unforced::seed_value(i, rho, u[0], u[1], u[2]);
      const Macro m{rho, u[0], u[1], u[2]};
      fc.collide(f1.data(), m, 0);
      uc.collide(f2.data(), m, 0);
      for (int i = 0; i < L::Q; ++i) d[i] = f1[i] - f2[i];

      std::vector<double> k(NM);
      cmoments<L>(d.data(), u, basis, E, NM, k.data());

      const double sc = std::max({std::abs(F[0]), std::abs(F[1]), std::abs(F[2])});
      for (int j = 0; j < NM; ++j) {
        int ones = 0, twos = 0, axis = -1;
        for (int a = 0; a < D; ++a) {
          if (E[j][a] == 1) { ++ones; axis = a; } else if (E[j][a] == 2) ++twos;
        }
        // monomial: cs^(2m) F_a.   Hermite: F_a only where m = 0.
        double want = 0.0;
        if (ones == 1) {
          if (basis == 0)              want = std::pow(cs2v, twos) * F[axis];
          else if (twos == 0)          want = F[axis];
        }
        const double e = std::abs(k[j] - want) / sc;
        if (e > worst) { worst = e; wj = j; }
      }
    }
    const bool ok = worst < 1e-11;
    if (!ok) ++bad;
    std::printf("  %-6s %-9s worst |operator - K_force| / |F| = %.2e   at k_%d%d%d   %s\n",
                name, basis ? "Hermite" : "monomial", worst,
                E[wj][0], E[wj][1], E[wj][2], ok ? "OK" : "MISMATCH");
  }
  return bad;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    static int E2[9][3]; int n2 = 0;
    for (int p = 0; p < 3; ++p)
      for (int q = 0; q < 3; ++q) { E2[n2][0] = p; E2[n2][1] = q; E2[n2][2] = 0; ++n2; }
    static int E3[27][3]; int n3 = 0;
    for (int p = 0; p < 3; ++p)
      for (int q = 0; q < 3; ++q)
        for (int r = 0; r < 3; ++r) { E3[n3][0] = p; E3[n3][1] = q; E3[n3][2] = r; ++n3; }

    std::printf("Forcing central moments vs the Hermite closed form\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());
    std::printf("  K_force(monomial) = cs^(2m) F_a   exponent 1 on axis a, 2 on m others\n");
    std::printf("  K_force(Hermite)  = F_a on the first-order slots, zero elsewhere\n\n");
    bad += check<D2Q9>("D2Q9", E2, n2);
    bad += check<D3Q27>("D3Q27", E3, n3);
    std::printf("\n  %s\n", bad == 0
      ? "CONFIRMED on both lattices: the existing forcing already equals the Hermite K_force"
      : "*** MISMATCH ***");
  }
  Kokkos::finalize();
  return bad == 0 ? 0 : 1;
}
