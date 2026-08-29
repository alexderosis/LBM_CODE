#pragma once
//==============================================================================
//  Fluid collision for two-phase flow at a DENSITY RATIO -- the pressure-based
//  formulation with surface tension as a chemical-potential body force.
//
//  De Rosis & Enan, Phys. Fluids 33, 043315 (2021): Eq. (8) for the relaxation,
//  (10) for the equilibrium, (12) for the viscosity, (14) for the source,
//  (17)-(19) and (22)-(23) for the force, (24) for the moments, (25) for rho.
//
//  ============================ SCOPE, READ FIRST ============================
//  THIS OPERATOR IS FOR MULTIPHASE AND MULTICOMPONENT FLOW ONLY. It redefines
//  what the zeroth moment means -- sum_i f_i is a normalised PRESSURE here, not
//  a density -- so it is not a drop-in replacement for anything and must not be
//  reached for on a single-phase problem. For single-phase and single-component
//  work the operators to use remain BGK, TRT and MomentCollision, unchanged and
//  with rho = sum_i f_i as they have always had it. Nothing in FluidSolver, the
//  lattices, the streaming schemes or the existing operators is modified by this
//  file; it is one more collision policy beside them.
//  ===========================================================================
//
//  WHY THE ZEROTH MOMENT CARRIES PRESSURE. At a density ratio, rho is no longer
//  a small perturbation about 1 and a density-carrying distribution stops being
//  well conditioned -- ShiftedPopulations assumes rho_ref is exactly 1, and at a
//  ratio of 10 the stored g_i are not small any more. The incompressible model
//  moves the density out of the distribution entirely: rho is a function of phi
//  (Eq. 25, a linear interpolation) and the populations carry
//
//      p~ = sum_i f_i,          u = sum_i c_i f_i + F / (2 rho),      (24)
//
//  with the equilibrium of Eq. (10),
//
//      f_i^eq = w_i [ p~ + Phi_i(u) ],
//      Phi_i  = c_i.u/cs2 + (c_i.u)^2/(2 cs4) - u.u/(2 cs2),
//
//  whose zeroth moment is p~ exactly (the Phi terms sum to zero) and whose first
//  moment is u, NOT rho u. The physical pressure is p = rho cs2 p~.
//
//  THE SHIFT STILL WORKS, and this is the one place it is easier than usual: the
//  equilibrium is AFFINE in p~, so storing f_i - w_i simply replaces p~ by
//  p~ - 1 in the same expression. One line serves both storage conventions. It
//  is only worth using when p~ stays near 1, which -- see the next paragraph --
//  is exactly the gauge one should NOT choose at a density ratio.
//
//  ============ TWO PRESSURE NORMALISATIONS, AND NEITHER DOMINATES ============
//  rho_0 <= 0  (DEFAULT):  p = rho(phi) cs2 p~,  F_p = -p~ cs2 grad rho
//  rho_0 >  0           :  p = rho_0    cs2 p~,  F_p =  cs2 (rho - rho_0) grad p~
//
//  The first is De Rosis & Enan's. Because p is continuous while rho is not, p~
//  jumps by the whole density ratio across an interface, so F_p scales with the
//  pressure LEVEL and is amplified by the ratio. validation/layered_poiseuille
//  measures the cost against an exact solution: F_p reaches 220 times the driving
//  force, the residual does not converge (order -0.21), and a pure gauge shift in
//  p~ -- which changes no physics -- moves the error seventeenfold.
//
//  The second normalises by a CONSTANT, so p~ is uniform wherever p is and F_p
//  scales with the pressure GRADIENT instead. On that same case:
//
//      normalisation    order (offset 0)   order (offset 0.2)   gauge sensitivity
//      local rho        -0.21              -0.37                17.3x / 19.4x
//      constant rho_0    0.95               0.86                 1.93x / 2.04x
//
//  It converges where the other does not, and is an order of magnitude less
//  sensitive to a shift that changes no physics. Note what does NOT improve:
//  F_p/G is LARGER, 379 against 220 at H = 32. The gain is not that the term
//  shrinks but that it stops tracking the pressure level. The residual 1.9x is
//  the discrete equilibrium's own p~ dependence -- f^eq = w_i[p~ + Phi] carries
//  (p~ - 1) into its higher moments, so a p~ offset is not a pure gauge on the
//  lattice even when it is one in the continuum.
//
//  IT IS NOT A FREE WIN. The recovered pressure term becomes
//  -(rho_0/rho) cs2 grad p~, so one phase always pays:
//
//    * rho_0 = rho_L: the heavy phase runs at an effective sound speed
//      cs sqrt(rho_L/rho_H) -- stable, but the pressure term there is a
//      cancellation of ratio size. Fine where grad p is diffuse; at a Laplace
//      jump, where grad p is concentrated at the interface, it diverges at
//      ratio 10 and above, on a case the local-rho form handles at ratio 100.
//    * rho_0 = rho_H: the LIGHT phase gets sound speed cs sqrt(rho_H/rho_L),
//      which at a ratio of 10 is 1.83 lattice units per step -- past the lattice
//      speed. Tried first, diverged inside 500 steps.
//
//  So the local-rho form suits problems whose pressure is near zero at the
//  interface (a static droplet with the gauge at zero), and the constant-rho_0
//  form suits problems with a real pressure field but diffuse gradients (a driven
//  channel). The default stays local-rho because that is what every validated
//  result in this tree was obtained with; rho_0 makes the alternative available
//  where it is the better choice, and layered_poiseuille exercises both.
//  ===========================================================================
//
//  SURFACE TENSION AS A FORCE, not as a stress. Where MultiphaseBGK puts the
//  capillary tensor into the equilibrium's second moment, this uses the
//  potential form
//
//      F_s = mu_phi grad phi,                                          (18)
//      mu_phi = 4 beta (phi-phi_L)(phi-phi_H)(phi-phi_0) - kappa lap(phi),  (19)
//
//  with beta = 12 sigma / W and kappa = 3 sigma W / 2. It needs a LAPLACIAN,
//  which the stress form does not -- that is the price of the exchange, and it
//  is paid for by a density ratio the stress form cannot reach. PhaseFieldSolver
//  computes lap(phi) from the same neighbour gather as the gradient, so the cost
//  is one accumulator rather than one pass.
//
//  THE OTHER TWO FORCES ARE NOT OPTIONAL EXTRAS. The LBE recovers
//
//      rho d_t u + ... = -rho cs2 grad p~ + mu lap(u) + F,
//
//  while the equation wanted is -grad p + div(mu(grad u + grad u^T)) + F_s.
//  Since p = rho cs2 p~, grad p = rho cs2 grad p~ + p~ cs2 grad rho, so the
//  first mismatch is exactly
//
//      F_p = -p~ cs2 grad rho,                                         (22)
//
//  and the second, from the difference between mu lap(u) and the full divergence
//  of the stress at variable mu, is
//
//      F_nu = nu (grad u + grad u^T) . grad rho.                       (22)
//
//  Both vanish identically at a matched density, because grad rho does. Neither
//  is a correction one may drop at a ratio. grad rho follows from grad phi by
//  Eq. (23), so no second gradient field is needed for it.
//
//  F_s AND F_p SHARE A DIRECTION. Both are multiples of grad phi -- one through
//  mu_phi, the other through grad rho = (drho/dphi) grad phi -- so they are
//  assembled as a single coefficient times one vector, not as two vectors added.
//
//  F_nu IS IMPLEMENTED BUT UNEXERCISED HERE. It is identically zero in a static
//  droplet, which is the only case in the tree that runs this operator. A
//  layered Poiseuille flow at a viscosity and density ratio is what would
//  exercise it, and there is not one yet. Stated rather than glossed.
//
//  ONE COST OF THE FIXED Macro SHAPE. `Macro` carries four Reals and all four
//  are spoken for, so the force -- needed once to correct the velocity in
//  macroscopic() and once to source the collision -- is assembled twice per
//  node instead of being carried between them. That is about eight extra view
//  reads. Widening `Macro` by three defaulted fields would remove it, at the
//  cost of three Reals in every other operator's hot loop; not done here,
//  because that trade should be measured before it is taken.
//==============================================================================
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

template <class L, class Phi = SecondOrderPhi<L>, class Store = RawPopulations>
struct MultiphasePotentialBGK {
  using Lattice     = L;
  using Storage     = Store;
  using ForcingPolicy = NoForcing;      // the force is internal; see the header
  static constexpr const char* name = "MultiphasePotentialBGK";
  static_assert(L::supports_navier_stokes,
                "the multiphase fluid operator needs a Navier-Stokes lattice.");

  // Phase field and its derivatives, owned by PhaseFieldSolver.
  View1D<Real> phi, Gx, Gy, Gz, Lap;
  // Viscous interface force, owned by ViscousInterfaceForce. Empty is allowed
  // and means "matched density", where the term is zero anyway.
  View1D<Real> Vx, Vy, Vz;
  // An arbitrary EXTERNAL force field, per node. Empty means none. This is the
  // slot an immersed or penalised solid writes into -- kept separate from Vx so
  // that the two owners never have to agree about who clears the array.
  View1D<Real> Ex, Ey, Ez;

  // Gradient of p~, owned by a ScalarGradient. Empty switches the pressure force
  // off, which is only correct at a matched density.
  View1D<Real> Px, Py, Pz;

  Real phi_L = Real(0), phi_H = Real(1);
  Real rho_L = Real(1), rho_H = Real(1);
  // Reference density for the pressure normalisation. ZERO (the default) keeps
  // the local-rho form of the reference; a positive value selects the constant
  // form, and must be the LIGHT phase or the acoustic CFL is violated.
  Real rho_0 = Real(0);
  Real mu_L  = Real(0.1), mu_H = Real(0.1);   // DYNAMIC viscosities
  Real beta = Real(0), kappa = Real(0);
  Real bx = Real(0), by = Real(0), bz = Real(0);   // body acceleration, e.g. gravity

  // kappa = 3 sigma W / 2 and beta = 12 sigma / W, the pair consistent with the
  // tanh profile of width W that the conservative Allen-Cahn equation maintains.
  static Real kappa_from_sigma(Real sigma, Real width) { return Real(1.5) * sigma * width; }
  static Real beta_from_sigma (Real sigma, Real width) { return Real(12) * sigma / width; }

  //----------------------------------------------------------------------------
  // Everything the interpolations need, from ONE read of phi(n).
  //----------------------------------------------------------------------------
  struct Local { Real p, rho, mu, omega; };

  KOKKOS_INLINE_FUNCTION Local local(Index n) const {
    constexpr Real cs2v = cs2<L, Real>();
    const Real p = phi(n);
    // CLAMPED. The conservative Allen-Cahn form keeps phi close to [0,1] but does
    // not guarantee it: shear against a solid, or any under-resolved feature,
    // overshoots it locally. Interpolating rho off an unclamped phi then puts the
    // density outside the two phases it is meant to lie between -- and a large
    // enough undershoot makes rho NEGATIVE, at which point 1/rho in the velocity
    // and the force is unbounded. Clamping the INTERPOLANT rather than phi itself
    // leaves the transported field alone; only the equation of state is bounded.
    Real s = (p - phi_L) / (phi_H - phi_L);
    s = s < Real(0) ? Real(0) : (s > Real(1) ? Real(1) : s);
    const Real r = rho_L + s * (rho_H - rho_L);          // Eq. (25)
    const Real m = mu_L + s * (mu_H - mu_L);
    const Real tau = m / (r * cs2v);                     // Eq. (12)
    return Local{p, r, m, Real(1) / (tau + Real(0.5))};
  }
  KOKKOS_INLINE_FUNCTION Real density_at(Index n) const { return local(n).rho; }
  KOKKOS_INLINE_FUNCTION Real omega_at(Index n) const { return local(n).omega; }
  // nu = mu / rho, needed by ViscousInterfaceForce.
  KOKKOS_INLINE_FUNCTION Real viscosity_at(Index n) const {
    const Local l = local(n);
    return l.mu / l.rho;
  }
  // > 0 selects the constant-reference normalisation; see the banner.
  KOKKOS_INLINE_FUNCTION bool constant_reference() const { return rho_0 > Real(0); }
  // The density that converts p~ to a physical pressure, for diagnostics.
  KOKKOS_INLINE_FUNCTION Real pressure_scale(Real rho_local) const {
    return constant_reference() ? rho_0 : rho_local;
  }
  KOKKOS_INLINE_FUNCTION Real drho_dphi() const {
    return (rho_H - rho_L) / (phi_H - phi_L);            // Eq. (23)
  }

  //----------------------------------------------------------------------------
  // F = F_s + F_p + F_nu + F_b, with the first two folded into one coefficient
  // on grad phi.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void force(const Local& l, Index n, Real p_tilde, Real F[3]) const {
    constexpr Real cs2v = cs2<L, Real>();
    const Real phi0 = Real(0.5) * (phi_L + phi_H);
    const Real mu_phi = Real(4) * beta * (l.p - phi_L) * (l.p - phi_H) * (l.p - phi0)
                      - kappa * Lap(n);                              // Eq. (19)
    // F_s is always a multiple of grad phi. F_p is a multiple of grad phi in the
    // local-rho form and of grad p~ in the constant-rho_0 one, so it is carried
    // separately and only shares a direction with F_s in the first.
    const bool cref = constant_reference();
    const Real coef = cref ? mu_phi : (mu_phi - p_tilde * cs2v * drho_dphi());
    const Real dr = cref ? cs2v * (l.rho - rho_0) : Real(0);
    const bool have_p = cref && Px.data() != nullptr;
    const bool have_v = Vx.data() != nullptr;
    const bool have_e = Ex.data() != nullptr;
    F[0] = coef * Gx(n) + (have_p ? dr * Px(n) : Real(0))
         + (have_v ? Vx(n) : Real(0)) + (have_e ? Ex(n) : Real(0)) + l.rho * bx;
    F[1] = coef * Gy(n) + (have_p ? dr * Py(n) : Real(0))
         + (have_v ? Vy(n) : Real(0)) + (have_e ? Ey(n) : Real(0)) + l.rho * by;
    F[2] = (L::D == 3) ? coef * Gz(n) + (have_p ? dr * Pz(n) : Real(0))
                           + (have_v ? Vz(n) : Real(0)) + (have_e ? Ez(n) : Real(0))
                           + l.rho * bz
                       : Real(0);
  }

  // p~ from whatever the storage holds.
  KOKKOS_INLINE_FUNCTION static Real density(const Macro& m) {
    if constexpr (Store::shifted) return Real(1) + m.dens;
    else                          return m.dens;
  }

  //----------------------------------------------------------------------------
  // Eq. (24). Note the first moment is u, not rho u, so there is no division by
  // rho here -- only the half-force correction carries one. Getting that wrong
  // is a viscosity error that looks like a bad boundary condition.
  //
  // With shifted storage sum_i c_i (f_i - w_i) == sum_i c_i f_i exactly, because
  // sum_i c_i w_i = 0, so the momentum sum is untouched by the shift.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  Macro macroscopic(const Real f[L::Q], Index n = 0) const {
    Real s = Real(0), mx = Real(0), my = Real(0), mz = Real(0);
    for (int i = 0; i < L::Q; ++i) {
      s  += f[i];
      mx += f[i] * Real(cvel<L>(i, 0));
      my += f[i] * Real(cvel<L>(i, 1));
      mz += f[i] * Real(cvel<L>(i, 2));
    }
    Macro m{s, Real(0), Real(0), Real(0)};
    const Local l = local(n);
    Real F[3];
    force(l, n, density(m), F);
    const Real h = Real(0.5) / l.rho;
    m.ux = mx + h * F[0];  m.uy = my + h * F[1];  m.uz = mz + h * F[2];
    return m;
  }

  //----------------------------------------------------------------------------
  // Eq. (8) with the source of Eq. (14), F_i = w_i (c_i . F) / (rho cs2), and
  // the (1 - omega/2) prefactor of Eq. (27).
  //
  // The one equilibrium expression below serves both storage conventions,
  // because f_i^eq is affine in p~: m.dens is p~ raw and p~ - 1 shifted, and
  // w_i(p~ + Phi) - w_i == w_i((p~ - 1) + Phi).
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], const Macro& m, Index n = 0) const {
    constexpr Real ics2 = inv_cs2<L, Real>();
    const Local l = local(n);
    Real F[3];
    force(l, n, density(m), F);
    const Real w = l.omega;
    const Real pref = (Real(1) - Real(0.5) * w) * ics2 / l.rho;
    for (int i = 0; i < L::Q; ++i) {
      const Real feq = weight<L, Real>(i) * (m.dens + Phi::phi(i, m.ux, m.uy, m.uz));
      const Real cF = Real(cvel<L>(i, 0)) * F[0] + Real(cvel<L>(i, 1)) * F[1] +
                      Real(cvel<L>(i, 2)) * F[2];
      f[i] += w * (feq - f[i]) + pref * weight<L, Real>(i) * cF;
    }
  }

  // FlowState::rho is the NORMALISED PRESSURE p~ for this operator, not a
  // density. FluidSolver::initialize(r) therefore seeds p~ = r.
  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real p_tilde, Real ux, Real uy, Real uz) {
    const Real d = Store::shifted ? p_tilde - Real(1) : p_tilde;
    return weight<L, Real>(i) * (d + Phi::phi(i, ux, uy, uz));
  }
};

}  // namespace lbm
