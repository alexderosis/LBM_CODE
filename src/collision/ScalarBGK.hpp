#pragma once
//==============================================================================
//  Advection-diffusion collision for a passive scalar (temperature).
//
//  Differs from the fluid operators in one structural way: the velocity is an
//  INPUT, not something recovered from its own populations. The scalar carries
//  only its own zeroth moment,
//
//      T = sum_i g_i,
//
//  and is advected by whatever velocity field the fluid solver hands it. That is
//  why this has its own collision concept rather than reusing `Macro`.
//
//  Equilibrium: g_i^eq = w_i T (1 + c_i.u / cs2), first order in u.
//
//  That truncation is deliberate. D2Q5 and D3Q7 have an isotropic second moment
//  but NOT an isotropic fourth-order one, and on D3Q7 every velocity has a single
//  nonzero component, so (c.u)^2 reduces to c_a^2 u_a^2 and the cross terms of
//  the uu tensor cannot be represented at all. Adding a second-order term would
//  therefore add an anisotropic error rather than accuracy. The cost is an O(u^2)
//  defect in the advection term, which is why these lattices suit low-Mach
//  transport -- exactly the regime Boussinesq convection lives in.
//
//  Diffusivity: D = cs^2 (1/omega - 1/2), with cs2 = 1/4 on D3Q7 and 1/3 on D2Q5.
//
//  STORAGE REFERENCE. The arrays hold h_i = g_i - w_i T_ref, for the same reason
//  the fluid stores f_i - w_i: the collision works on differences far smaller
//  than the populations themselves, and in FP32 that cancellation costs most of
//  the mantissa. Unlike the fluid, though, there is no universal reference --
//  rho is always near 1, but a temperature scale is whatever the problem says it
//  is. So T_ref is a runtime parameter, and leaving it at its default of 0
//  reproduces the unshifted scheme exactly. Set it to the mean temperature.
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

template <class L>
struct ScalarBGK {
  using Lattice = L;
  using Storage = RawPopulations;      // the scalar is O(1); no shift needed yet
  static constexpr const char* name = "ScalarBGK";

  Real omega = Real(1);
  Real T_ref = Real(0);      // storage reference; 0 reproduces raw storage

  static Real omega_from_diffusivity(Real d) {
    return Real(1) / (d * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real diffusivity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
  }

  // Deviation from the reference: dT = sum_i h_i, so T = T_ref + dT.
  KOKKOS_INLINE_FUNCTION
  static Real deviation(const Real h[L::Q]) {
    Real t = Real(0);
    for (int i = 0; i < L::Q; ++i) t += h[i];
    return t;
  }
  KOKKOS_INLINE_FUNCTION Real temperature(const Real h[L::Q]) const {
    return T_ref + deviation(h);
  }

  // h_i^eq = w_i [ dT + T (c_i.u)/cs2 ], every term small when dT and u are.
  // Writing it as `eq_raw - w_i T_ref` would be algebraically identical and
  // numerically pointless.
  KOKKOS_INLINE_FUNCTION
  Real eq(int i, Real dT, Real ux, Real uy, Real uz) const {
    constexpr Real ics2 = inv_cs2<L, Real>();
    const Real cu = Real(cvel<L>(i, 0)) * ux + Real(cvel<L>(i, 1)) * uy +
                    Real(cvel<L>(i, 2)) * uz;
    return weight<L, Real>(i) * (dT + (T_ref + dT) * ics2 * cu);
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dT, Real ux, Real uy, Real uz) const {
    for (int i = 0; i < L::Q; ++i) h[i] += omega * (eq(i, dT, ux, uy, uz) - h[i]);
  }
};

//------------------------------------------------------------------------------
// Cell roles for a scalar field. The geometry is shared with the fluid, but the
// boundary CONDITIONS are not: a wall that is no-slip for momentum may be either
// insulating or held at a fixed temperature.
//------------------------------------------------------------------------------
enum ScalarCell : std::uint8_t {
  ScalarBulk      = 0,   // transport
  ScalarAdiabatic = 1,   // zero flux   -- bounce-back
  ScalarDirichlet = 2,   // fixed value -- anti-bounce-back toward T_wall
  ScalarExcluded  = 3,   // not part of the simulation
  ScalarMoment    = 4,   // fixed value AT the node, Dellar's moment condition
};

}  // namespace lbm
