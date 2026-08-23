#pragma once
//==============================================================================
//  Lattice descriptors -- Esoteric Pull ordering
//
//  ORDERING CONTRACT (do not change without regenerating every moment matrix):
//      index 0            : rest velocity
//      indices 1,2 / 3,4 / 5,6 / ... : opposite pairs, ADJACENT
//  so that
//      opp(0) = 0,  opp(i) = i+1 if i is odd, i-1 if i is even.
//
//  Esoteric Pull (Lehmann 2022) walks the pairs as `for (i = 1; i < Q; i += 2)`
//  and relies on (i, i+1) being opposites.  Adjacent pairing keeps opp() pure
//  integer arithmetic -- no lookup table, no register pressure in the kernel.
//
//  Weights are stored as exact rationals (w_num[i] / w_den) and the speed of
//  sound as cs2_num / cs2_den, so the lattice identities below are checked at
//  compile time in exact integer arithmetic rather than in floating point.
//==============================================================================

#include <cstdint>

// Device-callable when Kokkos is on the include path; a plain inline function
// otherwise, so this header stays compilable standalone (the compile-time
// lattice checks below can then be run without a Kokkos build).
#if defined(KOKKOS_INLINE_FUNCTION)
  #define LBM_FN KOKKOS_INLINE_FUNCTION
#elif defined(__has_include)
  #if __has_include(<Kokkos_Core.hpp>)
    #include <Kokkos_Core.hpp>
    #define LBM_FN KOKKOS_INLINE_FUNCTION
  #else
    #define LBM_FN inline
  #endif
#else
  #define LBM_FN inline
#endif

namespace lbm {

//------------------------------------------------------------------------------
// opp(): the Esoteric Pull pairing rule. Valid for every lattice below.
//------------------------------------------------------------------------------
LBM_FN constexpr int opp(int i) noexcept {
  return i == 0 ? 0 : (i & 1 ? i + 1 : i - 1);
}

//==============================================================================
//  D2Q5  -- advection/diffusion only (no Navier-Stokes: 4th-order moments are
//           not isotropic). Intended for the future scalar-transport module.
//==============================================================================
struct D2Q5 {
  static constexpr const char* name = "D2Q5";
  static constexpr int D = 2, Q = 5;
  static constexpr int cs2_num = 1, cs2_den = 3;
  static constexpr int w_den = 6;
  static constexpr bool supports_navier_stokes = false;

  static constexpr int cx[Q]    = {  0,  1, -1,  0,  0 };
  static constexpr int cy[Q]    = {  0,  0,  0,  1, -1 };
  static constexpr int w_num[Q] = {  2,  1,  1,  1,  1 };
};

//==============================================================================
//  D2Q9
//==============================================================================
struct D2Q9 {
  static constexpr const char* name = "D2Q9";
  static constexpr int D = 2, Q = 9;
  static constexpr int cs2_num = 1, cs2_den = 3;
  static constexpr int w_den = 36;
  static constexpr bool supports_navier_stokes = true;

  //                             0   1   2   3   4   5   6   7   8
  static constexpr int cx[Q]    = { 0,  1, -1,  0,  0,  1, -1,  1, -1 };
  static constexpr int cy[Q]    = { 0,  0,  0,  1, -1,  1, -1, -1,  1 };
  static constexpr int w_num[Q] = {16,  4,  4,  4,  4,  1,  1,  1,  1 };
};

//==============================================================================
//  D3Q7  -- advection/diffusion only.  NOTE cs2 = 1/4, not 1/3.
//==============================================================================
struct D3Q7 {
  static constexpr const char* name = "D3Q7";
  static constexpr int D = 3, Q = 7;
  static constexpr int cs2_num = 1, cs2_den = 4;
  static constexpr int w_den = 8;
  static constexpr bool supports_navier_stokes = false;

  static constexpr int cx[Q]    = { 0,  1, -1,  0,  0,  0,  0 };
  static constexpr int cy[Q]    = { 0,  0,  0,  1, -1,  0,  0 };
  static constexpr int cz[Q]    = { 0,  0,  0,  0,  0,  1, -1 };
  static constexpr int w_num[Q] = { 2,  1,  1,  1,  1,  1,  1 };
};

//==============================================================================
//  D3Q19
//==============================================================================
struct D3Q19 {
  static constexpr const char* name = "D3Q19";
  static constexpr int D = 3, Q = 19;
  static constexpr int cs2_num = 1, cs2_den = 3;
  static constexpr int w_den = 36;
  static constexpr bool supports_navier_stokes = true;

  //                          0  1   2  3   4  5   6  7   8  9  10  11 12  13 14  15 16  17 18
  static constexpr int cx[Q]    = { 0, 1, -1, 0,  0, 0,  0, 1, -1, 1, -1, 1, -1, 1, -1, 0,  0, 0,  0 };
  static constexpr int cy[Q]    = { 0, 0,  0, 1, -1, 0,  0, 1, -1,-1,  1, 0,  0, 0,  0, 1, -1, 1, -1 };
  static constexpr int cz[Q]    = { 0, 0,  0, 0,  0, 1, -1, 0,  0, 0,  0, 1, -1,-1,  1, 1, -1,-1,  1 };
  static constexpr int w_num[Q] = {12, 2,  2, 2,  2, 2,  2, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1 };
};

//==============================================================================
//  D3Q27
//==============================================================================
struct D3Q27 {
  static constexpr const char* name = "D3Q27";
  static constexpr int D = 3, Q = 27;
  static constexpr int cs2_num = 1, cs2_den = 3;
  static constexpr int w_den = 216;
  static constexpr bool supports_navier_stokes = true;

  static constexpr int cx[Q] = { 0, 1, -1, 0,  0, 0,  0,
                                 1,-1,  1,-1,  1,-1,  1,-1,  0, 0,  0, 0,
                                 1,-1,  1,-1,  1,-1, -1, 1 };
  static constexpr int cy[Q] = { 0, 0,  0, 1, -1, 0,  0,
                                 1,-1, -1, 1,  0, 0,  0, 0,  1,-1,  1,-1,
                                 1,-1,  1,-1, -1, 1,  1,-1 };
  static constexpr int cz[Q] = { 0, 0,  0, 0,  0, 1, -1,
                                 0, 0,  0, 0,  1,-1, -1, 1,  1,-1, -1, 1,
                                 1,-1, -1, 1,  1,-1,  1,-1 };
  static constexpr int w_num[Q] = {64, 16,16,16,16,16,16,
                                    4,  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                    1,  1, 1, 1, 1, 1, 1, 1 };
};

//------------------------------------------------------------------------------
// Typed accessors. Weights and cs2 are stored as exact rationals above; these
// turn them into whatever scalar type the caller is using. All constexpr, so
// they fold to immediates.
//------------------------------------------------------------------------------
template <class L, class T>
LBM_FN constexpr T weight(int i) noexcept {
  return static_cast<T>(L::w_num[i]) / static_cast<T>(L::w_den);
}
template <class L, class T>
LBM_FN constexpr T cs2() noexcept {
  return static_cast<T>(L::cs2_num) / static_cast<T>(L::cs2_den);
}
template <class L, class T>
LBM_FN constexpr T inv_cs2() noexcept {
  return static_cast<T>(L::cs2_den) / static_cast<T>(L::cs2_num);
}
// component a of velocity i; returns 0 for a == 2 on a 2D lattice
template <class L>
LBM_FN constexpr int cvel(int i, int a) noexcept {
  if (a == 0) return L::cx[i];
  if (a == 1) return L::cy[i];
  if constexpr (L::D == 3) return L::cz[i];
  else return 0;
}

//==============================================================================
//  Compile-time verification of every lattice.
//  All arithmetic below is exact integer arithmetic.
//==============================================================================
namespace detail {

template <class L> constexpr int cx_(int i) { return L::cx[i]; }
template <class L> constexpr int cy_(int i) { return L::cy[i]; }
template <class L> constexpr int cz_(int i) { if constexpr (L::D == 3) return L::cz[i]; else return 0; }
template <class L> constexpr int c_(int i, int a) {
  return a == 0 ? cx_<L>(i) : (a == 1 ? cy_<L>(i) : cz_<L>(i));
}

// opposite directions really are adjacent pairs
template <class L> constexpr bool check_pairing() {
  for (int i = 0; i < L::Q; ++i) {
    const int j = opp(i);
    if (j < 0 || j >= L::Q) return false;
    for (int a = 0; a < L::D; ++a)
      if (c_<L>(j, a) != -c_<L>(i, a)) return false;
  }
  return true;
}

// no duplicated velocities
template <class L> constexpr bool check_unique() {
  for (int i = 0; i < L::Q; ++i)
    for (int j = i + 1; j < L::Q; ++j) {
      bool same = true;
      for (int a = 0; a < L::D; ++a) same = same && (c_<L>(i, a) == c_<L>(j, a));
      if (same) return false;
    }
  return true;
}

// sum_i w_i == 1
template <class L> constexpr bool check_w_sum() {
  int s = 0;
  for (int i = 0; i < L::Q; ++i) s += L::w_num[i];
  return s == L::w_den;
}

// sum_i w_i c_ia c_ib == cs2 delta_ab   (cleared of denominators)
template <class L> constexpr bool check_second_moment() {
  for (int a = 0; a < L::D; ++a)
    for (int b = 0; b < L::D; ++b) {
      int s = 0;
      for (int i = 0; i < L::Q; ++i) s += L::w_num[i] * c_<L>(i, a) * c_<L>(i, b);
      const int lhs = s * L::cs2_den;
      const int rhs = (a == b) ? L::w_den * L::cs2_num : 0;
      if (lhs != rhs) return false;
    }
  return true;
}

// sum_i w_i c_ia c_ib c_ic == 0
template <class L> constexpr bool check_third_moment() {
  for (int a = 0; a < L::D; ++a)
    for (int b = 0; b < L::D; ++b)
      for (int d = 0; d < L::D; ++d) {
        int s = 0;
        for (int i = 0; i < L::Q; ++i) s += L::w_num[i] * c_<L>(i, a) * c_<L>(i, b) * c_<L>(i, d);
        if (s != 0) return false;
      }
  return true;
}

// sum_i w_i c_ia c_ib c_ic c_id == cs4 (d_ab d_cd + d_ac d_bd + d_ad d_bc)
// Required for Navier-Stokes; deliberately false for D2Q5 / D3Q7.
template <class L> constexpr bool check_fourth_moment() {
  for (int a = 0; a < L::D; ++a)
    for (int b = 0; b < L::D; ++b)
      for (int d = 0; d < L::D; ++d)
        for (int e = 0; e < L::D; ++e) {
          int s = 0;
          for (int i = 0; i < L::Q; ++i)
            s += L::w_num[i] * c_<L>(i, a) * c_<L>(i, b) * c_<L>(i, d) * c_<L>(i, e);
          const int iso = (a == b) * (d == e) + (a == d) * (b == e) + (a == e) * (b == d);
          if (s * L::cs2_den * L::cs2_den != L::w_den * L::cs2_num * L::cs2_num * iso) return false;
        }
  return true;
}

}  // namespace detail

#define LBM_VERIFY_LATTICE(L)                                                             \
  static_assert(detail::check_pairing<L>(),                                               \
                #L ": opposite directions are not adjacent pairs (Esoteric Pull needs "   \
                   "opp(i) = i+1 for odd i, i-1 for even i).");                           \
  static_assert(detail::check_unique<L>(),        #L ": duplicated velocity.");            \
  static_assert(detail::check_w_sum<L>(),         #L ": weights do not sum to 1.");        \
  static_assert(detail::check_second_moment<L>(), #L ": <c c> != cs2 I.");                 \
  static_assert(detail::check_third_moment<L>(),  #L ": <c c c> != 0.");                   \
  static_assert(detail::check_fourth_moment<L>() == L::supports_navier_stokes,             \
                #L ": 4th-order isotropy disagrees with supports_navier_stokes.");

LBM_VERIFY_LATTICE(D2Q5)
LBM_VERIFY_LATTICE(D2Q9)
LBM_VERIFY_LATTICE(D3Q7)
LBM_VERIFY_LATTICE(D3Q19)
LBM_VERIFY_LATTICE(D3Q27)

#undef LBM_VERIFY_LATTICE

}  // namespace lbm
