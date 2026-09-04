#pragma once
//==============================================================================
//  Factorised moment transform for PRODUCT lattices (D2Q9, D3Q27).
//
//  A product lattice contains every velocity in {-1,0,1}^D exactly once, and the
//  moment basis used here is a product of the same three 1D functions on each
//  axis:
//
//      phi_0(C) = 1        phi_1(C) = C        phi_2(C) = C^2 - cs2
//
//  with C = c - u_b for basis velocity u_b (u_b = u gives central moments, u_b = 0
//  gives raw moments). The transform therefore factorises into D successive 1D
//  passes instead of one Q x Q contraction:
//
//      D3Q27:  1458 mul-add   ->  ~324      (4.5x)
//      D2Q9:    162 mul-add   ->   ~72
//
//  Verified against the dense contraction in exact rational arithmetic.
//
//  Why this basis. The product-form equilibrium has EXACTLY ONE nonzero moment
//  in it -- k_{0..0} = rho, every other equilibrium moment is identically zero.
//  (Checked symbolically for D2Q9 and D3Q27.) That is what collapses the
//  collision to a handful of lines in MomentCollision.hpp.
//
//  D3Q19 is not a product lattice -- it is D3Q27 minus the eight corners -- so
//  none of this applies to it and `enabled` is false. Its central-moment
//  operator needs the 19-moment d'Humieres basis and a genuine 19x19 transform.
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

// Index of the population with the given integer velocity, or -1.
template <class L>
LBM_FN constexpr int dir_index(int a, int b, int c) {
  for (int i = 0; i < L::Q; ++i)
    if (cvel<L>(i, 0) == a && cvel<L>(i, 1) == b && cvel<L>(i, 2) == c) return i;
  return -1;
}

template <class L>
constexpr bool is_product_lattice() {
  int seen = 0;
  const int lo = (L::D == 3) ? -1 : 0, hi = (L::D == 3) ? 1 : 0;
  for (int a = -1; a <= 1; ++a)
    for (int b = -1; b <= 1; ++b)
      for (int c = lo; c <= hi; ++c) {
        if (dir_index<L>(a, b, c) < 0) return false;
        ++seen;
      }
  return seen == L::Q;
}

template <class L>
struct ProductBasis {
  static constexpr bool enabled = is_product_lattice<L>();
  static constexpr int D  = L::D;
  static constexpr int NM = (D == 2) ? 9 : 27;    // number of moments

  // moment index for orders (p, q, r)
  static constexpr int mi(int p, int q, int r = 0) {
    return (D == 2) ? (p * 3 + q) : ((p * 3 + q) * 3 + r);
  }
  // ---- interface shared with MonomialBasis, so MomentCollision is basis-generic ----
  //
  // Table lookups, not arithmetic: these are called with a runtime moment index
  // inside the collision, and the obvious `n / 9`, `(n / 3) % 3` form costs three
  // integer divisions per moment. That measured as a 15% regression on D3Q27.
  //
  // THEY ARE ALSO `constexpr`, WHICH IS NOT DECORATION. MomentCollision calls
  // them with a COMPILE-TIME moment index (see its `eq_moment<N>`), and that is
  // the whole reason the moment array can live in registers: a table lookup on a
  // runtime index forces the 432-byte table into memory, and the value it
  // returns then indexes Qf/Aw, so those go to memory too. On a CPU that is a
  // 464-byte stack frame in L1 and costs 2.18x; in device code the same frame is
  // per-thread LOCAL memory, i.e. off-chip DRAM, with every index uncoalesced.
  // Marking these constexpr costs nothing and keeps the runtime form available
  // for any caller that genuinely has a runtime index.
  struct Ord { int p[27], q[27], r[27], o[27]; };
  static constexpr Ord make_ord() {
    Ord t{};
    for (int n = 0; n < NM; ++n) {
      t.p[n] = (D == 2) ? (n / 3) : (n / 9);
      t.q[n] = (D == 2) ? (n % 3) : ((n / 3) % 3);
      t.r[n] = (D == 2) ? 0 : (n % 3);
      t.o[n] = t.p[n] + t.q[n] + t.r[n];
    }
    return t;
  }
  KOKKOS_INLINE_FUNCTION static constexpr int p_of(int n)  { constexpr Ord t = make_ord(); return t.p[n]; }
  KOKKOS_INLINE_FUNCTION static constexpr int q_of(int n)  { constexpr Ord t = make_ord(); return t.q[n]; }
  KOKKOS_INLINE_FUNCTION static constexpr int r_of(int n)  { constexpr Ord t = make_ord(); return t.r[n]; }
  KOKKOS_INLINE_FUNCTION static constexpr int order(int n) { constexpr Ord t = make_ord(); return t.o[n]; }
  static constexpr int index_of(int p, int q, int r) { return mi(p, q, r); }

  // 1D equilibrium factors in this basis. phi_2 = C^2 - cs2 already has the cs2
  // subtracted, which is why the Maxwellian is diagonal here and these come out
  // as pure powers rather than carrying a cs2 term (contrast MonomialBasis).
  // The 1D basis function itself, for tests that contract directly.
  KOKKOS_INLINE_FUNCTION
  static Real phi(int p, int c, Real u) {
    const Real C = Real(c) - u;
    return p == 0 ? Real(1) : (p == 1 ? C : C * C - cs2<L, Real>());
  }

  KOKKOS_INLINE_FUNCTION
  static void eq_1d(Real du, Real ub, Real Qf[3], Real Aw[3]) {
    Qf[0] = Real(1); Qf[1] = du;  Qf[2] = du * du;
    Aw[0] = Real(1); Aw[1] = -ub; Aw[2] = ub * ub;
  }
  // Population index for component indices a, b, c in {0,1,2} meaning c = -1,0,+1.
  //
  // This MUST come from a precomputed table. dir_index() is a linear search over
  // the velocity set; it folds away only when its arguments are compile-time
  // constants, and here they are loop variables. Calling it directly cost 2*Q*Q
  // comparisons per node and made the operator 7.7x slower than BGK.
  struct Table { int v[27]; };
  static constexpr Table make_table() {
    Table t{};
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < ((D == 3) ? 3 : 1); ++c)
          t.v[(D == 2) ? (a * 3 + b) : ((a * 3 + b) * 3 + c)] =
              dir_index<L>(a - 1, b - 1, (D == 3) ? c - 1 : 0);
    return t;
  }
  KOKKOS_INLINE_FUNCTION static int pi(int a, int b, int c = 1) {
    constexpr Table t = make_table();
    return t.v[(D == 2) ? (a * 3 + b) : ((a * 3 + b) * 3 + c)];
  }

  //----------------------------------------------------------------------------
  // One 1D pass. Values live at c = -1, 0, +1; moments come back in the same
  // three slots, in order (m0, m1, m2).
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static void fwd1d(Real& a, Real& b, Real& c, Real u) {
    constexpr Real cs2v = cs2<L, Real>();
    const Real s0 = a + b + c;      // sum
    const Real s1 = c - a;          // first raw moment
    const Real s2 = c + a;          // second raw moment
    a = s0;
    b = s1 - u * s0;
    c = s2 - Real(2) * u * s1 + (u * u - cs2v) * s0;
  }
  KOKKOS_INLINE_FUNCTION
  static void inv1d(Real& a, Real& b, Real& c, Real u) {
    constexpr Real cs2v = cs2<L, Real>();
    const Real m0 = a, m1 = b, m2 = c;
    const Real s1 = m1 + u * m0;
    const Real s2 = m2 + Real(2) * u * s1 - (u * u - cs2v) * m0;
    a = Real(0.5) * (s2 - s1);
    b = m0 - s2;
    c = Real(0.5) * (s2 + s1);
  }

  //----------------------------------------------------------------------------
  template <bool Shift = true>
  KOKKOS_INLINE_FUNCTION
  static void to_moments(const Real f[L::Q], const Real ub[3], Real k[NM]) {
    if constexpr (D == 2) {
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) k[a * 3 + b] = f[pi(a, b)];
      for (int a = 0; a < 3; ++a) fwd1d(k[a * 3 + 0], k[a * 3 + 1], k[a * 3 + 2], ub[1]);
      for (int q = 0; q < 3; ++q) fwd1d(k[0 * 3 + q], k[1 * 3 + q], k[2 * 3 + q], ub[0]);
    } else {
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          for (int c = 0; c < 3; ++c) k[mi(a, b, c)] = f[pi(a, b, c)];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          fwd1d(k[mi(a, b, 0)], k[mi(a, b, 1)], k[mi(a, b, 2)], ub[2]);
      for (int a = 0; a < 3; ++a)
        for (int r = 0; r < 3; ++r)
          fwd1d(k[mi(a, 0, r)], k[mi(a, 1, r)], k[mi(a, 2, r)], ub[1]);
      for (int q = 0; q < 3; ++q)
        for (int r = 0; r < 3; ++r)
          fwd1d(k[mi(0, q, r)], k[mi(1, q, r)], k[mi(2, q, r)], ub[0]);
    }
  }

  template <bool Shift = true>
  KOKKOS_INLINE_FUNCTION
  static void to_populations(Real k[NM], const Real ub[3], Real f[L::Q]) {
    if constexpr (D == 2) {
      for (int q = 0; q < 3; ++q) inv1d(k[0 * 3 + q], k[1 * 3 + q], k[2 * 3 + q], ub[0]);
      for (int a = 0; a < 3; ++a) inv1d(k[a * 3 + 0], k[a * 3 + 1], k[a * 3 + 2], ub[1]);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) f[pi(a, b)] = k[a * 3 + b];
    } else {
      for (int q = 0; q < 3; ++q)
        for (int r = 0; r < 3; ++r)
          inv1d(k[mi(0, q, r)], k[mi(1, q, r)], k[mi(2, q, r)], ub[0]);
      for (int a = 0; a < 3; ++a)
        for (int r = 0; r < 3; ++r)
          inv1d(k[mi(a, 0, r)], k[mi(a, 1, r)], k[mi(a, 2, r)], ub[1]);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          inv1d(k[mi(a, b, 0)], k[mi(a, b, 1)], k[mi(a, b, 2)], ub[2]);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          for (int c = 0; c < 3; ++c) f[pi(a, b, c)] = k[mi(a, b, c)];
    }
  }
};

}  // namespace lbm
