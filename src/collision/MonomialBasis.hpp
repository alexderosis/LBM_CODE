#pragma once
//==============================================================================
//  Monomial moment basis for D3Q19 -- GENERATED, do not hand-edit.
//
//  D3Q19 is D3Q27 minus its eight corners, so it is not a product lattice and
//  the factorised transform in ProductBasis.hpp does not apply. What does work
//  is a 19-monomial basis, which is exactly the non-ortho basis of
//  MATLAB/D3Q19_CM.m written as plain monomials:
//
//      (000) (100)(010)(001) (200)(020)(002)(110)(101)(011)
//      (210)(120)(201)(102)(021)(012) (220)(202)(022)
//
//  Verified in exact rational arithmetic before generating this file:
//    * M is invertible on D3Q19 (4*M^-1 is integral),
//    * the set is DOWNWARD CLOSED, so the binomial central-moment shift needs no
//      moment outside the basis and is exact,
//    * the shift agrees with direct central summation and round-trips exactly,
//    * the equilibrium raw moments are EXACTLY the product form
//         m_eq(p,q,r) = rho P(p,ux) P(q,uy) P(r,uz),  P = {1, u, cs2 + u^2}
//      so the equilibrium CENTRAL moments are the Maxwellian ones on all 19
//      representable monomials -- the same structure as D3Q27. (The known D3Q19
//      equilibrium defects live at monomials such as (300) and (111), which this
//      lattice cannot represent at all and which are not in the basis.)
//
//  ORTHO vs NON-ORTHO IS IMMATERIAL HERE. D3Q19_CM.m relaxes positions 6..10 at
//  omega and everything else at 1 in both branches; in either basis those five
//  span the deviatoric second-order subspace and position 5 spans the trace, and
//  sending a set of independent combinations to equilibrium is the same as
//  sending the subspace to equilibrium. Both bases define the identical operator.
//
//  Sparsity: M has 127/361 nonzeros, M^-1 has 91/361, the shift adds 72 terms --
//  about 290 operations, which is cheaper than D3Q27's factorised transform.
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

template <class L>
struct MonomialBasis;

template <>
struct MonomialBasis<D3Q19> {
  using Lattice = D3Q19;
  static constexpr bool enabled = true;
  static constexpr int  D  = 3;
  static constexpr int  NM = 19;

  // The exponents live INSIDE the accessors rather than as static constexpr
  // array members. A static constexpr array indexed with a runtime index in
  // device code has to exist in device memory and nvcc will not put it there
  // ("identifier is undefined in device code"); a constexpr array declared in a
  // function body is materialised normally. Same reason the lattice velocity
  // sets are accessors -- see Lattices.hpp.
  KOKKOS_INLINE_FUNCTION static constexpr int p_of(int n) noexcept {
    constexpr int v[NM] = {0,1,0,0,2,0,0,1,1,0,2,1,2,1,0,0,2,2,0};
    return v[n];
  }
  KOKKOS_INLINE_FUNCTION static constexpr int q_of(int n) noexcept {
    constexpr int v[NM] = {0,0,1,0,0,2,0,1,0,1,1,2,0,0,2,1,2,0,2};
    return v[n];
  }
  KOKKOS_INLINE_FUNCTION static constexpr int r_of(int n) noexcept {
    constexpr int v[NM] = {0,0,0,1,0,0,2,0,1,1,0,0,1,2,1,2,0,2,2};
    return v[n];
  }
  KOKKOS_INLINE_FUNCTION static constexpr int order(int n) noexcept {
    return p_of(n) + q_of(n) + r_of(n);
  }
  static constexpr int index_of(int p, int q, int r) {
    for (int n = 0; n < NM; ++n)
      if (p_of(n) == p && q_of(n) == q && r_of(n) == r) return n;
    return -1;
  }

  // 1D equilibrium factors in this basis (plain monomials, so cs2 is present):
  //   Q = moments of a Maxwellian at u, measured about u_b   -> {1, du, cs2+du^2}
  //   A = moments of the rest weights,  measured about u_b   -> {1, -ub, cs2+ub^2}
  // The 1D basis function itself, for tests that contract directly.
  KOKKOS_INLINE_FUNCTION
  static Real phi(int p, int c, Real u) {
    const Real C = Real(c) - u;
    return p == 0 ? Real(1) : (p == 1 ? C : C * C);
  }

  KOKKOS_INLINE_FUNCTION
  static void eq_1d(Real du, Real ub, Real Qf[3], Real Aw[3]) {
    constexpr Real cs2v = cs2<D3Q19, Real>();
    Qf[0] = Real(1); Qf[1] = du;  Qf[2] = cs2v + du * du;
    Aw[0] = Real(1); Aw[1] = -ub; Aw[2] = cs2v + ub * ub;
  }

  KOKKOS_INLINE_FUNCTION
  static void raw_moments(const Real f[19], Real m[19]) {
    m[0] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8] + f[9] + f[10] + f[11] 
           + f[12] + f[13] + f[14] + f[15] + f[16] + f[17] + f[18];
    m[1] = f[1] + f[7] + f[9] + f[11] + f[13] - f[2] - f[8] - f[10] - f[12] - f[14];
    m[2] = f[3] + f[7] + f[10] + f[15] + f[17] - f[4] - f[8] - f[9] - f[16] - f[18];
    m[3] = f[5] + f[11] + f[14] + f[15] + f[18] - f[6] - f[12] - f[13] - f[16] - f[17];
    m[4] = f[1] + f[2] + f[7] + f[8] + f[9] + f[10] + f[11] + f[12] + f[13] + f[14];
    m[5] = f[3] + f[4] + f[7] + f[8] + f[9] + f[10] + f[15] + f[16] + f[17] + f[18];
    m[6] = f[5] + f[6] + f[11] + f[12] + f[13] + f[14] + f[15] + f[16] + f[17] + f[18];
    m[7] = f[7] + f[8] - f[9] - f[10];
    m[8] = f[11] + f[12] - f[13] - f[14];
    m[9] = f[15] + f[16] - f[17] - f[18];
    m[10] = f[7] + f[10] - f[8] - f[9];
    m[11] = f[7] + f[9] - f[8] - f[10];
    m[12] = f[11] + f[14] - f[12] - f[13];
    m[13] = f[11] + f[13] - f[12] - f[14];
    m[14] = f[15] + f[18] - f[16] - f[17];
    m[15] = f[15] + f[17] - f[16] - f[18];
    m[16] = f[7] + f[8] + f[9] + f[10];
    m[17] = f[11] + f[12] + f[13] + f[14];
    m[18] = f[15] + f[16] + f[17] + f[18];
  }

  KOKKOS_INLINE_FUNCTION
  static void populations(const Real m[19], Real f[19]) {
    f[0] = m[0] + m[16] + m[17] + m[18] - m[4] - m[5] - m[6];
    f[1] = Real(1)/Real(2)*m[1] + Real(1)/Real(2)*m[4] - Real(1)/Real(2)*m[11] - 
           Real(1)/Real(2)*m[13] - Real(1)/Real(2)*m[16] - Real(1)/Real(2)*m[17];
    f[2] = Real(1)/Real(2)*m[4] + Real(1)/Real(2)*m[11] + Real(1)/Real(2)*m[13] - 
           Real(1)/Real(2)*m[1] - Real(1)/Real(2)*m[16] - Real(1)/Real(2)*m[17];
    f[3] = Real(1)/Real(2)*m[2] + Real(1)/Real(2)*m[5] - Real(1)/Real(2)*m[10] - 
           Real(1)/Real(2)*m[15] - Real(1)/Real(2)*m[16] - Real(1)/Real(2)*m[18];
    f[4] = Real(1)/Real(2)*m[5] + Real(1)/Real(2)*m[10] + Real(1)/Real(2)*m[15] - 
           Real(1)/Real(2)*m[2] - Real(1)/Real(2)*m[16] - Real(1)/Real(2)*m[18];
    f[5] = Real(1)/Real(2)*m[3] + Real(1)/Real(2)*m[6] - Real(1)/Real(2)*m[12] - 
           Real(1)/Real(2)*m[14] - Real(1)/Real(2)*m[17] - Real(1)/Real(2)*m[18];
    f[6] = Real(1)/Real(2)*m[6] + Real(1)/Real(2)*m[12] + Real(1)/Real(2)*m[14] - 
           Real(1)/Real(2)*m[3] - Real(1)/Real(2)*m[17] - Real(1)/Real(2)*m[18];
    f[7] = Real(1)/Real(4)*m[7] + Real(1)/Real(4)*m[10] + Real(1)/Real(4)*m[11] + 
           Real(1)/Real(4)*m[16];
    f[8] = Real(1)/Real(4)*m[7] + Real(1)/Real(4)*m[16] - Real(1)/Real(4)*m[10] - 
           Real(1)/Real(4)*m[11];
    f[9] = Real(1)/Real(4)*m[11] + Real(1)/Real(4)*m[16] - Real(1)/Real(4)*m[7] - 
           Real(1)/Real(4)*m[10];
    f[10] = Real(1)/Real(4)*m[10] + Real(1)/Real(4)*m[16] - Real(1)/Real(4)*m[7] - 
           Real(1)/Real(4)*m[11];
    f[11] = Real(1)/Real(4)*m[8] + Real(1)/Real(4)*m[12] + Real(1)/Real(4)*m[13] + 
           Real(1)/Real(4)*m[17];
    f[12] = Real(1)/Real(4)*m[8] + Real(1)/Real(4)*m[17] - Real(1)/Real(4)*m[12] - 
           Real(1)/Real(4)*m[13];
    f[13] = Real(1)/Real(4)*m[13] + Real(1)/Real(4)*m[17] - Real(1)/Real(4)*m[8] - 
           Real(1)/Real(4)*m[12];
    f[14] = Real(1)/Real(4)*m[12] + Real(1)/Real(4)*m[17] - Real(1)/Real(4)*m[8] - 
           Real(1)/Real(4)*m[13];
    f[15] = Real(1)/Real(4)*m[9] + Real(1)/Real(4)*m[14] + Real(1)/Real(4)*m[15] + 
           Real(1)/Real(4)*m[18];
    f[16] = Real(1)/Real(4)*m[9] + Real(1)/Real(4)*m[18] - Real(1)/Real(4)*m[14] - 
           Real(1)/Real(4)*m[15];
    f[17] = Real(1)/Real(4)*m[15] + Real(1)/Real(4)*m[18] - Real(1)/Real(4)*m[9] - 
           Real(1)/Real(4)*m[14];
    f[18] = Real(1)/Real(4)*m[14] + Real(1)/Real(4)*m[18] - Real(1)/Real(4)*m[9] - 
           Real(1)/Real(4)*m[15];
  }

  // Binomial shift. su/sv/sw = -u_b going to central moments, +u_b coming back.
  KOKKOS_INLINE_FUNCTION
  static void shift(const Real m[19], Real su, Real sv, Real sw, Real o[19]) {
    o[0] = m[0];
    o[1] = m[1] + su*m[0];
    o[2] = m[2] + sv*m[0];
    o[3] = m[3] + sw*m[0];
    o[4] = m[4] + su*su*m[0] + Real(2)*su*m[1];
    o[5] = m[5] + sv*sv*m[0] + Real(2)*sv*m[2];
    o[6] = m[6] + sw*sw*m[0] + Real(2)*sw*m[3];
    o[7] = m[7] + su*sv*m[0] + su*m[2] + sv*m[1];
    o[8] = m[8] + su*sw*m[0] + su*m[3] + sw*m[1];
    o[9] = m[9] + sv*sw*m[0] + sv*m[3] + sw*m[2];
    o[10] = m[10] + su*su*sv*m[0] + su*su*m[2] + Real(2)*su*sv*m[1] + Real(2)*su*m[7] + 
           sv*m[4];
    o[11] = m[11] + su*sv*sv*m[0] + Real(2)*su*sv*m[2] + su*m[5] + sv*sv*m[1] + 
           Real(2)*sv*m[7];
    o[12] = m[12] + su*su*sw*m[0] + su*su*m[3] + Real(2)*su*sw*m[1] + Real(2)*su*m[8] + 
           sw*m[4];
    o[13] = m[13] + su*sw*sw*m[0] + Real(2)*su*sw*m[3] + su*m[6] + sw*sw*m[1] + 
           Real(2)*sw*m[8];
    o[14] = m[14] + sv*sv*sw*m[0] + sv*sv*m[3] + Real(2)*sv*sw*m[2] + Real(2)*sv*m[9] + 
           sw*m[5];
    o[15] = m[15] + sv*sw*sw*m[0] + Real(2)*sv*sw*m[3] + sv*m[6] + sw*sw*m[2] + 
           Real(2)*sw*m[9];
    o[16] = m[16] + su*su*sv*sv*m[0] + Real(2)*su*su*sv*m[2] + su*su*m[5] + 
           Real(2)*su*sv*sv*m[1] + Real(4)*su*sv*m[7] + Real(2)*su*m[11] + sv*sv*m[4] + 
           Real(2)*sv*m[10];
    o[17] = m[17] + su*su*sw*sw*m[0] + Real(2)*su*su*sw*m[3] + su*su*m[6] + 
           Real(2)*su*sw*sw*m[1] + Real(4)*su*sw*m[8] + Real(2)*su*m[13] + sw*sw*m[4] + 
           Real(2)*sw*m[12];
    o[18] = m[18] + sv*sv*sw*sw*m[0] + Real(2)*sv*sv*sw*m[3] + sv*sv*m[6] + 
           Real(2)*sv*sw*sw*m[2] + Real(4)*sv*sw*m[9] + Real(2)*sv*m[15] + sw*sw*m[5] + 
           Real(2)*sw*m[14];
  }

  template <bool Shift>
  KOKKOS_INLINE_FUNCTION
  static void to_moments(const Real f[19], const Real ub[3], Real k[19]) {
    Real m[19];
    raw_moments(f, m);
    if constexpr (Shift) shift(m, -ub[0], -ub[1], -ub[2], k);
    else                 for (int n = 0; n < 19; ++n) k[n] = m[n];
  }

  template <bool Shift>
  KOKKOS_INLINE_FUNCTION
  static void to_populations(const Real k[19], const Real ub[3], Real f[19]) {
    Real m[19];
    if constexpr (Shift) shift(k, ub[0], ub[1], ub[2], m);
    else                 for (int n = 0; n < 19; ++n) m[n] = k[n];
    populations(m, f);
  }
};

}  // namespace lbm
