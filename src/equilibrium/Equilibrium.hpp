#pragma once
//==============================================================================
//  Equilibrium distributions.
//
//  Each policy supplies only  phi(i, u) , defined by
//
//      f_i^eq = w_i rho (1 + phi_i)
//
//  from which both storage forms follow, once, in `Equilibrium`:
//
//      raw      f_i^eq       = w_i rho (1 + phi_i)
//      shifted  f_i^eq - w_i = w_i (drho + (1 + drho) phi_i)
//
//  The shifted form is written that way on purpose: every term is small, so it
//  never forms the difference of two O(w_i) quantities. Writing it as
//  `eq(i,rho,u) - w_i` would be algebraically identical and numerically useless.
//
//  Policies are stateless and static; they inline completely into the collision.
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Turns a phi-provider into the two storage forms.
//------------------------------------------------------------------------------
template <class L, class Phi>
struct Equilibrium {
  using Lattice = L;
  static constexpr const char* name = Phi::name;

  KOKKOS_INLINE_FUNCTION
  static Real eq(int i, Real rho, Real ux, Real uy, Real uz) {
    return weight<L, Real>(i) * rho * (Real(1) + Phi::phi(i, ux, uy, uz));
  }

  // drho = rho - 1
  KOKKOS_INLINE_FUNCTION
  static Real eq_shifted(int i, Real drho, Real ux, Real uy, Real uz) {
    return weight<L, Real>(i) * (drho + (Real(1) + drho) * Phi::phi(i, ux, uy, uz));
  }
};

//------------------------------------------------------------------------------
// Standard second-order (truncated Hermite) expansion.
//   phi = (c.u)/cs2 + (c.u)^2/(2 cs4) - u.u/(2 cs2)
//------------------------------------------------------------------------------
template <class L>
struct SecondOrderPhi {
  static constexpr const char* name = "SecondOrder";
  KOKKOS_INLINE_FUNCTION
  static Real phi(int i, Real ux, Real uy, Real uz) {
    constexpr Real ics2 = inv_cs2<L, Real>();
    const Real cu = Real(cvel<L>(i, 0)) * ux + Real(cvel<L>(i, 1)) * uy +
                    Real(cvel<L>(i, 2)) * uz;
    const Real uu = ux * ux + uy * uy + uz * uz;
    return ics2 * cu + Real(0.5) * ics2 * ics2 * cu * cu - Real(0.5) * ics2 * uu;
  }
};

//------------------------------------------------------------------------------
// Product-form (complete) expansion for D2Q9 -- `index_equilibrium = 4` from
// MATLAB/D2Q9_CM.m. Verified: its central moments are EXACTLY Maxwellian for all
// nine representable k_pq, i.e. it is fully Galilean invariant on this lattice.
// That is the property the central-moment operator will rely on.
//------------------------------------------------------------------------------
template <class L>
struct ProductFormPhi;

template <>
struct ProductFormPhi<D2Q9> {
  static constexpr const char* name = "ProductForm";
  KOKKOS_INLINE_FUNCTION
  static Real phi(int i, Real ux, Real uy, Real /*uz*/) {
    constexpr Real cs2v = cs2<D2Q9, Real>();
    const Real cx = Real(D2Q9::cx[i]);
    const Real cy = Real(D2Q9::cy[i]);
    const Real ax = cx * cx - cs2v;
    const Real ay = cy * cy - cs2v;
    const Real o1 = (cx * ux + cy * uy) / cs2v;
    const Real o2 = (ax * ux * ux + ay * uy * uy + Real(2) * cx * cy * ux * uy) /
                    (Real(2) * cs2v * cs2v);
    const Real o3 = (ax * cy * ux * ux * uy + ay * cx * ux * uy * uy) /
                    (Real(2) * cs2v * cs2v * cs2v);
    const Real o4 = (ax * ay * ux * ux * uy * uy) /
                    (Real(4) * cs2v * cs2v * cs2v * cs2v);
    return o1 + o2 + o3 + o4;
  }
};

//------------------------------------------------------------------------------
// Product-form (complete) expansion for D3Q27 -- the Hermite series carried to
// SIXTH order in MATLAB/D3Q27_CM.m. That series was verified, in exact rational
// arithmetic, to be identically equal to the factorised product
//
//     f_i^eq = rho * psi(c_ix, u) psi(c_iy, v) psi(c_iz, w),
//     psi(0,  u) = 1 - cs2 - u^2,   psi(+-1, u) = (cs2 + u^2 +- u) / 2,
//
// so the factorised form is what is evaluated here: three multiplies instead of
// six Hermite tensors, bit-for-bit the same polynomial. Dividing each factor by
// its 1D weight {1/6, 2/3, 1/6} puts it in the (1 + phi) shape the interface
// wants, since w_i = w1d(c_ix) w1d(c_iy) w1d(c_iz) on a product lattice.
//
// Its central moments are EXACTLY Maxwellian in all 27 moments:
//     k_pqr = rho cs2^n  if p, q, r are all even (n of them equal to 2),
//           = 0          otherwise,
// i.e. every Galilean-invariance defect of the second-order equilibrium is gone.
//------------------------------------------------------------------------------
template <>
struct ProductFormPhi<D3Q27> {
  static constexpr const char* name = "ProductForm";
  // psi(c, u) / w1d(c), exact for the 1D moments {1, u, cs2 + u^2}
  KOKKOS_INLINE_FUNCTION
  static Real chi(int c, Real u) {
    return (c == 0) ? Real(1) - Real(1.5) * u * u
                    : Real(1) + Real(3) * u * (Real(c) + u);
  }
  KOKKOS_INLINE_FUNCTION
  static Real phi(int i, Real ux, Real uy, Real uz) {
    return chi(D3Q27::cx[i], ux) * chi(D3Q27::cy[i], uy) * chi(D3Q27::cz[i], uz) -
           Real(1);
  }
};

//------------------------------------------------------------------------------
// Fourth-order equilibrium for D3Q19, transcribed from MATLAB/D3Q19_CM.m.
//
// D3Q19 is not a product lattice, so there is no factorised form and the series
// stops at fourth order -- the highest the 19 velocities support. The expression
// is written per velocity group (rest / axis / edge), exactly as in the script.
//
// Within the 19 REPRESENTABLE central moments -- {1, 3 first, 6 second, 6 third
// of type k_aab, 3 fourth of type k_aabb} -- its central moments are likewise
// exactly Maxwellian. The nonzero k_111, k_112, ... that a naive contraction
// returns are outside that set and are aliases of it, not independent modes.
//------------------------------------------------------------------------------
template <class L>
struct FourthOrderPhi;

template <>
struct FourthOrderPhi<D3Q19> {
  static constexpr const char* name = "FourthOrder";
  KOKKOS_INLINE_FUNCTION
  static Real phi(int i, Real ux, Real uy, Real uz) {
    const Real c[3] = {Real(D3Q19::cx[i]), Real(D3Q19::cy[i]), Real(D3Q19::cz[i])};
    const Real u[3] = {ux, uy, uz};
    const Real s[3] = {ux * ux, uy * uy, uz * uz};
    int a = -1, b = -1, n = 0;
    for (int d = 0; d < 3; ++d)
      if (c[d] != Real(0)) { (n == 0 ? a : b) = d; ++n; }

    if (n == 0)                                            // rest
      return -(s[0] + s[1] + s[2]) +
             Real(3) * (s[0] * s[1] + s[0] * s[2] + s[1] * s[2]);

    if (n == 1) {                                          // axis
      const int p = (a + 1) % 3, q = (a + 2) % 3;
      const Real rest = s[p] + s[q];
      return Real(3) * c[a] * u[a] + Real(3) * (s[a] - rest)
           - Real(9) * c[a] * u[a] * rest - Real(9) * s[a] * rest;
    }
    // edge: only the two in-plane components appear
    return Real(3) * (c[a] * u[a] + c[b] * u[b]) + Real(3) * (s[a] + s[b])
         + Real(9) * c[a] * c[b] * u[a] * u[b]
         + Real(9) * (c[b] * s[a] * u[b] + c[a] * u[a] * s[b])
         + Real(9) * s[a] * s[b];
  }
};

template <class L> using SecondOrderEquilibrium = Equilibrium<L, SecondOrderPhi<L>>;
template <class L> using ProductFormEquilibrium = Equilibrium<L, ProductFormPhi<L>>;
template <class L> using FourthOrderEquilibrium = Equilibrium<L, FourthOrderPhi<L>>;

//------------------------------------------------------------------------------
// The highest-order equilibrium each lattice admits. This is what the MHD and
// central-moment paths use by default.
//------------------------------------------------------------------------------
template <class L> struct BestEquilibriumOf            { using type = SecondOrderEquilibrium<L>; };
template <> struct BestEquilibriumOf<D2Q9>             { using type = ProductFormEquilibrium<D2Q9>; };
template <> struct BestEquilibriumOf<D3Q27>            { using type = ProductFormEquilibrium<D3Q27>; };
template <> struct BestEquilibriumOf<D3Q19>            { using type = FourthOrderEquilibrium<D3Q19>; };
template <class L> using HighOrderEquilibrium = typename BestEquilibriumOf<L>::type;

}  // namespace lbm
