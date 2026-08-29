#pragma once
//==============================================================================
//  Conservative Allen-Cahn collision for the phase field.
//
//  Tracks an order parameter phi in [0, 1] -- 1 in the heavy phase, 0 in the
//  light one -- obeying
//
//      d_t phi + div(phi u) = div[ M ( grad phi - theta n ) ],
//      theta = (4/W) phi (1 - phi),      n = grad phi / |grad phi|,
//
//  whose one-dimensional equilibrium is EXACTLY the hyperbolic tangent
//
//      phi(x) = 1/2 [ 1 + tanh(2x/W) ],
//
//  because for that profile |grad phi| = (4/W) phi (1 - phi) identically, so the
//  diffusive and anti-diffusive fluxes cancel term for term. That is the whole
//  point of the conservative form: the interface neither spreads nor sharpens,
//  and W is a prescribed number of lattice units rather than something that
//  emerges from a balance and has to be measured afterwards.
//
//  FOLLOWS De Rosis & Enan, Phys. Fluids 33, 043315 (2021), Sec. II.B: Eq. (9)
//  for the relaxation, Eq. (11) for the equilibrium, Eq. (13) for tau_phi = M/cs2,
//  Eqs. (15)-(16) for the source and Eq. (20) for the gradient. The source below
//  was derived independently and then checked against Eqs. (15)-(16); they agree
//  term for term, which is worth recording because getting it wrong is silent
//  (see THE SOURCE, below).
//
//  RELATION TO ScalarBGK. Structurally this is that operator plus one source
//  term. Velocity is an INPUT and the only moment carried is the zeroth,
//  phi = sum_i h_i.
//
//  THE EQUILIBRIUM IS A POLICY, because the right truncation depends on the
//  lattice and this operator is meant to run on both kinds:
//
//    * On a Navier-Stokes lattice (D2Q9, D3Q19, D3Q27) the default is the
//      SECOND-ORDER form, which is Eq. (11) of the reference exactly -- and it
//      is the fluid's own equilibrium with phi substituted for rho, so
//      SecondOrderEquilibrium is reused rather than retyped. The paper runs the
//      phase field on the full D3Q19 alongside the flow.
//    * On D2Q5 / D3Q7 the default drops to FIRST order, because those lattices
//      have no isotropic fourth-order moment; on D3Q7 every velocity has a
//      single nonzero component, so (c.u)^2 collapses to c_a^2 u_a^2 and the
//      cross terms of uu cannot be represented at all. A second-order term
//      there adds anisotropy, not accuracy -- the same argument ScalarBGK makes.
//
//  The trade is cost against advection accuracy: the reduced lattice moves 5 or
//  7 populations against 9 or 19, and pays an O(u^2) defect in the advection
//  term. A static interface cannot see that difference; a sheared one can.
//
//  THE SOURCE, AND WHY ITS PREFACTOR IS NOT GUO'S.
//
//  Chapman-Enskog on the advection-diffusion LBE with a source S_i obeying
//  sum_i S_i = 0 and sum_i c_i S_i = A_S gives
//
//      d_t phi + div(phi u) = M lap(phi) - (1/omega) div(A_S),
//
//  so matching the anti-diffusion term needs
//
//      A_S = omega M theta n = cs2 (1 - omega/2) theta n,
//
//  using omega M = cs2 (1 - omega/2). With S_i = w_i (c_i . A_S) / cs2 the cs2
//  cancels and the source is simply
//
//      S_i = (1 - omega/2) theta w_i (c_i . n),
//
//  carrying no mobility and no speed of sound at all. sum_i S_i = 0, so phi is
//  conserved exactly -- that is the "conservative" in conservative Allen-Cahn.
//
//  The coefficient is 1/omega, NOT the 1/(1 - omega/2) that Guo's force term
//  compensates, and the difference is structural rather than a convention. A
//  body force enters the FIRST-moment equation directly, where the half-step
//  correction is absorbed by redefining the velocity. This enters the ZEROTH-
//  moment equation through the divergence of the first moment, so the source
//  contributes twice -- once through h^(1) and once through the second-order
//  Taylor term -- and (1/omega - 1/2) + 1/2 = 1/omega is where that lands.
//
//  Getting this wrong is not subtle in its effect but is silent in its cause:
//  with the Guo prefactor and a 1/cs2 the source comes out a factor
//  M/cs2 = 1/omega - 1/2 too small, the anti-diffusion cannot balance the
//  diffusion, and the interface simply spreads as though the term were absent.
//  Measured before the fix: W = 4 became 14.1 in 600 steps, against the
//  sqrt(4 M t) = 11 of pure diffusion.
//
//  CHECK. At a static flat interface the total flux M grad phi - M theta n must
//  vanish pointwise, which is exactly the tanh identity |grad phi| = theta.
//
//  MOBILITY. M = cs^2 (1/omega - 1/2), exactly as the diffusivity in ScalarBGK,
//  with cs2 = 1/3 on D2Q5 and 1/4 on D3Q7. It is a numerical parameter, not a
//  physical one: too small and the interface cannot heal after being advected
//  through the grid, too large and it diffuses faster than the flow deforms it.
//  M in [0.01, 0.1] is the usual working range.
//
//  NO STORAGE SHIFT. phi is O(1) and its useful range is exactly [0, 1], so
//  there is no cancellation to protect against -- unlike the fluid, where the
//  populations are O(w_i) and the collision works on differences many orders
//  smaller. RawPopulations is not a placeholder here, it is the right answer.
//==============================================================================
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// g_i^eq = w_i phi (1 + c_i.u / cs2). The truncation D2Q5 and D3Q7 admit.
//------------------------------------------------------------------------------
template <class L>
struct FirstOrderPhaseEq {
  static constexpr const char* name = "FirstOrder";
  KOKKOS_INLINE_FUNCTION
  static Real eq(int i, Real phi, Real ux, Real uy, Real uz) {
    constexpr Real ics2 = inv_cs2<L, Real>();
    const Real cu = Real(cvel<L>(i, 0)) * ux + Real(cvel<L>(i, 1)) * uy +
                    Real(cvel<L>(i, 2)) * uz;
    return weight<L, Real>(i) * phi * (Real(1) + ics2 * cu);
  }
};

// Second order where the lattice supports it, first order where it does not.
template <class L, bool Ns = L::supports_navier_stokes> struct DefaultPhaseEqOf;
template <class L> struct DefaultPhaseEqOf<L, false> { using type = FirstOrderPhaseEq<L>; };
template <class L> struct DefaultPhaseEqOf<L, true>  { using type = SecondOrderEquilibrium<L>; };

template <class L, class Eq = typename DefaultPhaseEqOf<L>::type>
struct PhaseFieldBGK {
  using Lattice     = L;
  using Equilibrium = Eq;
  using Storage     = RawPopulations;
  static constexpr const char* name = "PhaseFieldBGK";

  Real omega = Real(1);      // sets the mobility
  Real width = Real(4);      // interface width W, in lattice units

  // Eq. (13): tau_phi = M / cs2, and omega = 1 / (tau_phi + 1/2).
  static Real omega_from_mobility(Real m) {
    return Real(1) / (m * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real mobility_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
  }

  KOKKOS_INLINE_FUNCTION Real mobility() const {
    return (Real(1) / omega - Real(0.5)) * cs2<L, Real>();
  }

  // phi = sum_i h_i. The only moment this field carries.
  KOKKOS_INLINE_FUNCTION
  static Real order_parameter(const Real h[L::Q]) {
    Real s = Real(0);
    for (int i = 0; i < L::Q; ++i) s += h[i];
    return s;
  }

  // Eq. (11) of the reference on a full lattice, its first-order truncation on
  // a reduced one. Note the fluid's own equilibrium serves unchanged: it is
  // w_i rho (1 + phi_Hermite), and phi takes the place of rho.
  KOKKOS_INLINE_FUNCTION
  Real eq(int i, Real phi, Real ux, Real uy, Real uz) const {
    return Eq::eq(i, phi, ux, uy, uz);
  }

  //----------------------------------------------------------------------------
  // A = theta n = (4/W) phi (1 - phi) G / |G|,  G = grad phi.
  //
  // Purely geometric: no mobility and no cs2, both of which cancelled in the
  // prefactor above. For the equilibrium profile A is grad phi itself, which is
  // the cheapest possible check on this function.
  //
  // Computed once per node rather than per direction: it costs a square root and
  // a divide, a real expense to repeat Q times for something with no i in it.
  // The same class of defect as ProductBasis::pi() -- see the README.
  //
  // Away from the interface phi (1 - phi) is zero anyway, so the |G| guard is a
  // second line of defence rather than the only one: it stops 0/0 in the bulk,
  // where G is round-off noise about zero and n is meaningless.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void anti_diffusion(Real phi, const Real G[3], Real A[3]) const {
    const Real g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    const Real gn = Kokkos::sqrt(g2);
    if (!(gn > Real(1e-12))) { A[0] = A[1] = A[2] = Real(0); return; }
    const Real s = (Real(4) / width) * phi * (Real(1) - phi) / gn;
    A[0] = s * G[0];  A[1] = s * G[1];  A[2] = s * G[2];
  }

  //----------------------------------------------------------------------------
  // Relax toward equilibrium and add the anti-diffusion source. A is passed in
  // rather than recomputed so the caller pays for it once per node.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real phi, const Real u[3], const Real A[3]) const {
    const Real pref = Real(1) - Real(0.5) * omega;
    for (int i = 0; i < L::Q; ++i) {
      const Real cA = Real(cvel<L>(i, 0)) * A[0] + Real(cvel<L>(i, 1)) * A[1] +
                      Real(cvel<L>(i, 2)) * A[2];
      h[i] += omega * (eq(i, phi, u[0], u[1], u[2]) - h[i])
            + pref * weight<L, Real>(i) * cA;
    }
  }
};

//------------------------------------------------------------------------------
// Cell roles for the phase field. As with the scalar, the geometry is shared
// with the fluid but the CONDITIONS are not.
//
// PhaseWall is zero-flux on the populations (bounce-back), which is the right
// condition for the transport but does NOT by itself set a contact angle: that
// needs a wetting condition on grad phi at the wall, which is not implemented.
// See the "Not implemented" list in PhaseFieldSolver.hpp before putting an
// interface against a wall.
//------------------------------------------------------------------------------
enum PhaseCell : std::uint8_t {
  PhaseBulk     = 0,   // transport
  PhaseWall     = 1,   // zero flux -- bounce-back, no wetting condition
  PhaseExcluded = 2,   // not part of the simulation
};

}  // namespace lbm
