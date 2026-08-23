#pragma once
//==============================================================================
//  What the population arrays actually hold.
//
//  RawPopulations      stores f_i
//  ShiftedPopulations  stores g_i = f_i - w_i
//
//  Why the shift. Populations are O(w_i) ~ 1e-1 but the collision operates on
//  differences O(1e-6), so `f - f^eq` in FP32 throws away most of the mantissa;
//  the momentum sum is worse still, since sum_i c_i f_i cancels ~1e-1 terms down
//  to ~1e-2. Storing g_i removes both cancellations: sum_i c_i w_i = 0 exactly,
//  so sum_i c_i g_i == sum_i c_i f_i identically while every summand is small.
//
//  Almost nothing else changes:
//    - streaming is a pure copy, so it is untouched;
//    - halfway bounce-back is g_i <- g_opp(i), untouched, because w_i == w_opp(i);
//    - the Guo source term is already an O(F) quantity, untouched.
//  Only the equilibrium and the density reconstruction differ, and both are
//  written in a cancellation-free form (see Equilibrium.hpp).
//
//  The reference density is exactly 1: g_i = f_i - w_i, rho = 1 + sum_i g_i.
//==============================================================================
#include "core/Types.hpp"

namespace lbm {

struct RawPopulations {
  static constexpr const char* name = "raw";
  static constexpr bool shifted = false;
};

struct ShiftedPopulations {
  static constexpr const char* name = "shifted";
  static constexpr bool shifted = true;
};

}  // namespace lbm
