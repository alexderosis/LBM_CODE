#pragma once
//==============================================================================
//  REGULARISED advection-diffusion collision for a passive scalar.
//
//  Same physics as ScalarBGK -- same equilibrium, same diffusivity, same
//  interface -- and one difference that only matters near omega = 2: it relaxes
//  the FLUX moments at omega and sends the ghost moments straight to
//  equilibrium, instead of relaxing everything at omega.
//
//  ================== WHY BGK IS NOT ENOUGH NEAR omega = 2 ===================
//  D3Q7 carries seven moments: the scalar T, the three fluxes j_a, and three
//  ghosts (the axial second moments). Only the first four have physical content
//  -- T is conserved and j_a sets the diffusive flux, so it must relax at the
//  rate that fixes alpha. The ghosts carry nothing, and what happens to them is
//  a free choice.
//
//  BGK relaxes them at omega too. At omega = 1 that is projection onto
//  equilibrium and the two operators are IDENTICAL. As omega -> 2 it becomes a
//  REFLECTION: the post-collision ghost is (1 - omega) times the pre-collision
//  one, so at omega = 1.997 it is multiplied by -0.997 every step. It inverts
//  and never damps.
//
//  MEASURED, and this is why the operator exists. At Ra = 1e14
//  (omega_T = 1.99999905) the near-wall temperature under BGK RINGS rather than
//  relaxing: GPU/'s twin of this driver saw Nu_bot swing 34.9 <-> 93.6 over
//  sixty free-fall times against an analytic ~100 -- bounded rather than
//  diverging, which is the dangerous kind of wrong, because it averages into a
//  plausible number. The same mechanism sets how far a cold-start initial
//  condition undershoots its own lower bound: worst T_min, cold start,
//
//      ScalarBGK       Ra = 1e10  -0.552     Ra = 1e14  -0.813
//      this operator   Ra = 1e14  -0.112, and it RECOVERS
//
//  against a physical floor of T_cold = 0. The maximum principle is not a
//  detail here: a scheme that leaves [T_cold, T_hot] is not solving the problem
//  that was posed, so a fifth of the excursion is the difference between a run
//  that can be quoted and one that cannot.
//
//  THE INVERSE IS CLOSED-FORM, not a transform. On these lattices every
//  velocity has one nonzero component and the axes decouple, so with
//
//      M_a = sum_i h_i c_ia^2 = cs2 dT   (the ghost, at equilibrium)
//      p_a = j_a + omega (T u_a - j_a)   (the flux, relaxed)
//
//  the pair along axis a is just (M_a +/- p_a)/2 and the rest population is
//  whatever is left of dT. No matrix, no basis, seven multiplies. That is also
//  why this is NOT a moment operator in the sense of MomentCollision.hpp and
//  needs none of its machinery -- and, per the warning in that file's
//  eq_moment banner, no runtime-indexed moment array either.
//
//  WHAT IT IS NOT. It is not more accurate in the hydrodynamic limit: the
//  ghosts do not appear in the advection-diffusion equation the scheme
//  recovers, so at moderate omega the two operators agree to the same order and
//  ScalarBGK remains the simpler default for those cases. It changes the
//  STABILITY and the boundedness near omega = 2, nothing else. Below about
//  omega = 1.9 there is no reason to prefer it.
//
//  It also does not fix the underlying resolution problem. omega -> 2 means
//  alpha -> 0, which means the thermal boundary layer is thinner than the cell
//  that has to represent it; annihilating the ghosts stops that from RINGING
//  but does not resolve the layer. See the Nu ceiling of H/2 in
//  demonstrator/rb_high_ra.cpp.
//
//  Interface, storage reference, per-node omega, omega_from_diffusivity: all
//  inherited from ScalarBGK unchanged. Only `collide` differs.
//==============================================================================
#include "collision/ScalarBGK.hpp"

namespace lbm {

template <class L>
struct ScalarRegularised : ScalarBGK<L> {
  using Base    = ScalarBGK<L>;
  using Lattice = L;
  using Storage = RawPopulations;
  static constexpr const char* name = "ScalarRegularised";

  // The closed form below assumes ONE REST VELOCITY AND D AXIAL PAIRS, with
  // pair a occupying slots (2a+1, 2a+2). That is D2Q5 and D3Q7, and it is the
  // same adjacency contract Esoteric Pull relies on -- so it is asserted rather
  // than trusted. A lattice with diagonal velocities (D2Q9, D3Q27) has coupled
  // axes and a genuine 9- or 27-moment transform, which this is not.
  static_assert(L::Q == 2 * L::D + 1,
                "ScalarRegularised needs a rest-plus-axial-pairs lattice "
                "(D2Q5 or D3Q7); D2Q9/D3Q19/D3Q27 have coupled axes.");
  static_assert(cvel<L>(1, 0) == 1 && cvel<L>(2, 0) == -1,
                "slots 1,2 must be the +/-x pair.");
  static_assert(cvel<L>(3, 1) == 1 && cvel<L>(4, 1) == -1,
                "slots 3,4 must be the +/-y pair.");
  static_assert(L::D < 3 || (cvel<L>(5, 2) == 1 && cvel<L>(6, 2) == -1),
                "slots 5,6 must be the +/-z pair.");

  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dT, Real ux, Real uy, Real uz, Real w) const {
    constexpr Real cs2v = cs2<L, Real>();
    const Real T    = this->T_ref + dT;
    const Real u[3] = {ux, uy, uz};
    const Real d    = cs2v * dT;              // every ghost, at equilibrium

    // This loop DOES unroll -- D is 2 or 3 with a constexpr bound -- so the
    // subscripts fold to constants and h stays in registers. Verified in the
    // assembly with tests/frame_check.sh; the failure mode it is being kept
    // away from is written up in MomentCollision.hpp's eq_moment banner.
    for (int a = 0; a < L::D; ++a) {
      const Real j = h[2 * a + 1] - h[2 * a + 2];
      const Real p = j + w * (T * u[a] - j);
      h[2 * a + 1] = Real(0.5) * (d + p);
      h[2 * a + 2] = Real(0.5) * (d - p);
    }
    // sum_i h_i = dT is exact by construction, which is what conserves the
    // scalar: the D pairs contribute D*d between them and the rest slot takes
    // the remainder. Writing h[0] as w_0 * dT would be algebraically the same
    // and would stop being the same the moment a weight changed.
    h[0] = dT - Real(L::D) * d;
  }
  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dT, Real ux, Real uy, Real uz) const {
    collide(h, dT, ux, uy, uz, this->omega);
  }
};

}  // namespace lbm
