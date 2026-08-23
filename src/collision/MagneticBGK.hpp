#pragma once
//==============================================================================
//  Magnetic induction, Dellar's vector-valued distribution.
//
//  The magnetic field is carried by a distribution that is a VECTOR at every
//  lattice link, g_i^alpha, with
//
//      B_alpha = sum_i g_i^alpha
//
//  and equilibrium
//
//      g_i^{alpha,eq} = w_i [ B_alpha + (1/cs2) c_{i,beta} (u_beta B_alpha - B_beta u_alpha) ]
//
//  Taking moments:  sum_i g^eq = B_alpha, and
//  sum_i c_beta g^eq = u_beta B_alpha - B_beta u_alpha, which is exactly the
//  antisymmetric flux of the induction equation
//
//      d_t B_alpha + d_beta (u_beta B_alpha - B_beta u_alpha) = eta lap B_alpha.
//
//  Only that first moment is needed, so the magnetic lattice can be far smaller
//  than the fluid's -- D2Q5 or D3Q7 suffices, since all that is required of it is
//  sum_i w_i c_beta c_gamma = cs2 delta. Running the field on a different lattice
//  from the flow is the point, not an economy.
//
//  Resistivity: eta = cs^2 (1/omega - 1/2).
//
//  Storage is unshifted. Unlike density there is no natural nonzero reference for
//  B -- it oscillates about zero in every case here -- so a shift would subtract
//  nothing. See ScalarBGK for the same argument made the other way.
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

template <class L>
struct MagneticBGK {
  using Lattice = L;
  static constexpr const char* name = "MagneticBGK";
  static constexpr int NC = L::D;          // components of B carried

  Real omega = Real(1);

  static Real omega_from_resistivity(Real eta) {
    return Real(1) / (eta * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real resistivity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
  }

  // B_alpha = sum_i g_i^alpha
  KOKKOS_INLINE_FUNCTION
  static Real field(const Real g[L::Q]) {
    Real b = Real(0);
    for (int i = 0; i < L::Q; ++i) b += g[i];
    return b;
  }

  KOKKOS_INLINE_FUNCTION
  static Real eq(int i, int a, const Real B[3], const Real u[3]) {
    constexpr Real ics2 = inv_cs2<L, Real>();
    Real flux = Real(0);                   // c_beta (u_beta B_a - B_beta u_a)
    for (int b = 0; b < L::D; ++b)
      flux += Real(cvel<L>(i, b)) * (u[b] * B[a] - B[b] * u[a]);
    return weight<L, Real>(i) * (B[a] + ics2 * flux);
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real g[L::Q], int a, const Real B[3], const Real u[3]) const {
    for (int i = 0; i < L::Q; ++i) g[i] += omega * (eq(i, a, B, u) - g[i]);
  }
};

}  // namespace lbm
