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
//
//  CONJUGATE HEAT TRANSFER: a per-node relaxation rate.
//
//  `omega_of` is an optional field of relaxation rates. Leave it empty (the
//  default) and every node uses the scalar `omega`; fill it and the diffusivity
//  becomes a property of the node, which is what a solid inclusion in a fluid
//  needs. The two are never mixed: omega_of, once set, is authoritative
//  everywhere, so a partially filled field is a silent zero-diffusivity bug and
//  not a supported state.
//
//  WHAT THIS MODELS, AND WHAT IT DOES NOT. Varying omega alone varies the
//  diffusivity alpha = lambda / (rho c_p). It therefore reproduces a
//  conductivity ratio ONLY where rho c_p is uniform, because the scheme
//  transports sum_i g_i and that sum is the temperature, not the enthalpy. A
//  jump in volumetric heat capacity needs a different variable and is not
//  implemented; validation/zhou_thermal.cpp's conduction case is the uniform-
//  rho-c_p kind, which is also what Zhou et al. (2026) Sec. 3.1 specifies.
//
//  The interface condition is not imposed anywhere -- continuity of T and of
//  alpha dT/dn emerges from the streaming, with the effective interface sitting
//  midway between the last node of one material and the first of the other.
//  That is second-order accurate and NOT exact: validation/zhou_thermal.cpp
//  measures what it costs (0.11% at kappa = 100 and N = 400, falling as N^-2).
//==============================================================================
#include "grid/Domain.hpp"
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
  View1D<Real> omega_of;     // optional per-node omega; empty means uniform

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

  // The rate this node relaxes at. One branch on a pointer that is uniform
  // across the whole kernel launch, so the uniform-omega path is unaffected.
  KOKKOS_INLINE_FUNCTION Real omega_at(Index n) const {
    return omega_of.data() ? omega_of(n) : omega;
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dT, Real ux, Real uy, Real uz, Real w) const {
    for (int i = 0; i < L::Q; ++i) h[i] += w * (eq(i, dT, ux, uy, uz) - h[i]);
  }
  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dT, Real ux, Real uy, Real uz) const {
    collide(h, dT, ux, uy, uz, omega);
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
  ScalarOutflow   = 5,   // open boundary -- equilibrium at the donor's value
};

}  // namespace lbm
