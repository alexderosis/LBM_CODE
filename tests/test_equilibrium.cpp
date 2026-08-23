//==============================================================================
//  Equilibrium moment consistency, Galilean invariance, and the shifted form.
//
//  The central-moment check on the D2Q9 product form is the sharp one: its
//  central moments must equal the Maxwellian values (which do not depend on u)
//  to machine precision. That is the property the central-moment collision
//  operator will rely on.
//
//  The shifted check verifies the identity  eq_shifted(rho-1) == eq(rho) - w_i
//  in FP64, where cancellation is harmless -- confirming the algebra is right
//  before FP32 relies on it being computed the other way round.
//==============================================================================
#include "Check.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"

using namespace lbm;

static Real TOL() { return sizeof(Real) == 4 ? Real(2e-6) : Real(5e-14); }

template <class L, class Eq>
void hydrodynamic_moments(Real rho, Real ux, Real uy, Real uz) {
  const std::string n = std::string(L::name) + "/" + Eq::name;
  Real f[L::Q];
  for (int i = 0; i < L::Q; ++i) f[i] = Eq::eq(i, rho, ux, uy, uz);

  Real m0 = 0, m1[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m0 += f[i];
    for (int a = 0; a < 3; ++a) m1[a] += f[i] * Real(cvel<L>(i, a));
  }
  check::near(m0, rho, TOL(), n + ": sum f_eq == rho");
  const Real u[3] = {ux, uy, uz};
  for (int a = 0; a < L::D; ++a)
    check::near(m1[a], rho * u[a], TOL(), n + ": sum c f_eq == rho u");

  for (int a = 0; a < L::D; ++a)
    for (int b = 0; b < L::D; ++b) {
      Real m = 0;
      for (int i = 0; i < L::Q; ++i)
        m += f[i] * Real(cvel<L>(i, a)) * Real(cvel<L>(i, b));
      const Real want = rho * (cs2<L, Real>() * Real(a == b) + u[a] * u[b]);
      check::near(m, want, TOL(), n + ": sum cc f_eq == rho(cs2 I + uu)");
    }
}

template <class L, class Eq>
void shifted_identity(Real rho, Real ux, Real uy, Real uz) {
  const std::string n = std::string(L::name) + "/" + Eq::name;
  double worst = 0;
  for (int i = 0; i < L::Q; ++i) {
    const Real direct  = Eq::eq(i, rho, ux, uy, uz) - weight<L, Real>(i);
    const Real shifted = Eq::eq_shifted(i, rho - Real(1), ux, uy, uz);
    worst = Kokkos::max(worst, std::abs(double(direct - shifted)));
  }
  char buf[64];
  std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            n + ": eq_shifted(rho-1) == eq(rho) - w_i" + buf);
}

// Continuous Maxwellian central moments factorise: k_pqr = rho M(p)M(q)M(r)
static Real Mm(int k, Real cs2v) {
  switch (k) {
    case 0: return Real(1);
    case 1: return Real(0);
    case 2: return cs2v;
    case 3: return Real(0);
    case 4: return Real(3) * cs2v * cs2v;
    default: return Real(0);
  }
}

template <class L, class Eq>
void galilean_invariance(Real rho, Real ux, Real uy) {
  const std::string n = std::string(L::name) + "/" + Eq::name;
  constexpr Real cs2v = cs2<L, Real>();
  Real f[L::Q];
  for (int i = 0; i < L::Q; ++i) f[i] = Eq::eq(i, rho, ux, uy, Real(0));

  int deviating = 0;
  double worst = 0;
  for (int p = 0; p <= 2; ++p)
    for (int q = 0; q <= 2; ++q) {
      Real k = 0;
      for (int i = 0; i < L::Q; ++i) {
        const Real cxu = Real(cvel<L>(i, 0)) - ux;
        const Real cyu = Real(cvel<L>(i, 1)) - uy;
        Real t = f[i];
        for (int e = 0; e < p; ++e) t *= cxu;
        for (int e = 0; e < q; ++e) t *= cyu;
        k += t;
      }
      const Real want = rho * Mm(p, cs2v) * Mm(q, cs2v);
      const double err = std::abs(double(k - want));
      if (err > double(TOL())) ++deviating;
      worst = std::max(worst, err);
    }
  char buf[64];
  std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(deviating == 0,
            n + ": all 9 central moments equal the Maxwellian values" + buf);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const Real r = Real(1.03), ux = Real(0.031), uy = Real(-0.017), uz = Real(0.022);
    hydrodynamic_moments<D2Q9,  SecondOrderEquilibrium<D2Q9>>(r, ux, uy, 0);
    hydrodynamic_moments<D3Q19, SecondOrderEquilibrium<D3Q19>>(r, ux, uy, uz);
    hydrodynamic_moments<D3Q27, SecondOrderEquilibrium<D3Q27>>(r, ux, uy, uz);
    hydrodynamic_moments<D2Q9,  ProductFormEquilibrium<D2Q9>>(r, ux, uy, 0);

    shifted_identity<D2Q9,  SecondOrderEquilibrium<D2Q9>>(r, ux, uy, 0);
    shifted_identity<D3Q19, SecondOrderEquilibrium<D3Q19>>(r, ux, uy, uz);
    shifted_identity<D3Q27, SecondOrderEquilibrium<D3Q27>>(r, ux, uy, uz);
    shifted_identity<D2Q9,  ProductFormEquilibrium<D2Q9>>(r, ux, uy, 0);
    shifted_identity<D3Q19, SecondOrderEquilibrium<D3Q19>>(Real(1), 0, 0, 0);

    // The second-order equilibrium is NOT Galilean invariant -- it is correct
    // only to O(u^2) -- so only the product form is asserted here.
    galilean_invariance<D2Q9, ProductFormEquilibrium<D2Q9>>(r, ux, uy);
    galilean_invariance<D2Q9, ProductFormEquilibrium<D2Q9>>(Real(0.97), Real(-0.05), Real(0.04));
  }
  const int r = check::report("equilibrium");
  Kokkos::finalize();
  return r;
}
