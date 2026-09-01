#pragma once
//==============================================================================
//  Central-moment collision for the conservative Allen-Cahn phase field.
//
//  De Rosis & Enan, Phys. Fluids 33, 043315 (2021), Sec. II.D -- "3D GMRT-LBM
//  for g_i", Eqs. (50)-(67). PhaseFieldBGK implements that paper's PHYSICS
//  (Sec. II.B: the equilibrium, tau_phi = M/cs2, the anti-diffusion source);
//  this implements its COLLISION, which is not BGK, and the difference is a
//  stability limit rather than a refinement.
//
//  WHY IT EXISTS. The mobility fixes the relaxation rate,
//
//      M = cs2 (1/omega_phi - 1/2)   =>   omega_phi = 1 / (M/cs2 + 1/2),
//
//  and the Peclet number of an interface-capture test fixes the mobility,
//  Pe = U0 xi / M. So a high Peclet number is a low mobility is omega_phi -> 2,
//  which is the BGK stability edge, and there is nothing to trade against it:
//  BGK relaxes EVERY moment at that same rate, ghost modes included. Measured:
//  the Zalesak disk of the paper's Sec. III B runs at Pe = 80 (omega = 1.9910)
//  and reaches 1e300 at Pe = 400 (omega = 1.9982). The paper's own sweep goes to
//  Pe = 4000, and it gets there because only its FIRST-order moments carry
//  omega_phi -- everything else is sent straight to equilibrium.
//
//  THE SCHEME, and why it comes out so short. The relaxation matrix is
//
//      K_phi = diag[1, omega_phi, omega_phi, omega_phi, 1, 1, ... 1]         (55)
//
//  in central moments: the zeroth is conserved, the three first-order ones
//  relax at omega_phi and set the mobility, and every higher moment is
//  overwritten with its equilibrium value each step. The paper lists those
//  equilibrium central moments in the MONOMIAL basis and finds five nonzero
//  ones -- k0 = phi, k4 = phi, k16 = k17 = k18 = phi cs^4 (Eq. 54). ProductBasis
//  here uses the SHIFTED basis, phi_2 = C^2 - cs2, in which those five collapse:
//
//      k4'  = k4 - 3 cs2 k0                     = phi - phi        = 0,
//      k16' = k16 - cs2 k_xx - cs2 k_yy + cs4 k0
//           = phi cs4 - phi cs4 - phi cs4 + phi cs4                 = 0,
//
//  so in this basis the equilibrium is simply
//
//      k^eq = (phi, 0, 0, ..., 0),
//
//  which is the same statement the fluid operator's banner makes about the
//  product-form Maxwellian being diagonal here. That is worth pausing on: it
//  means the whole collision is "keep phi, damp the three first moments, zero
//  everything else", and no equilibrium table is needed at all.
//
//  THE SOURCE. The anti-diffusion term has sum_i S_i = 0 and
//  sum_i c_i S_i = cs2 (1 - omega/2) theta n, so in central moments it touches
//  only the three first-order slots -- its zeroth moment vanishes and, being
//  first order in c, its central and raw first moments differ by u times its
//  zeroth, which is zero. It is therefore added to k1..k3 directly, with the
//  same (1 - omega/2) prefactor PhaseFieldBGK derives and for the same reason:
//  this source enters the ZEROTH-moment equation through the divergence of the
//  first moment, so it contributes twice and the coefficient is 1/omega, not
//  Guo's 1/(1 - omega/2). That derivation is in PhaseFieldBGK's banner and is
//  not repeated; what matters here is that the two operators use the identical
//  source, so any difference between them is the collision and nothing else.
//
//  WHAT IT COSTS, AND WHERE IT LOSES. Two three-pass transforms per node
//  against BGK's single loop. It does NOT dominate BGK, and the crossover is
//  sharp. On their Table II translation (L0 = 100, R = 25, xi = 3, U0 = 0.02,
//  10T) with only the mobility varied -- which is the only honest way to
//  compare two collisions, since M fixes omega and omega is the whole question:
//
//      M       omega     BGK        this
//      0.2     0.909     0.0609     0.0112      CM 5.4x better
//      0.05    1.539     0.0097     0.0066      CM 1.5x better
//      0.001   1.988     0.0082     0.0397      CM 4.8x WORSE
//
//  Pe = 60 is the paper's own operating point and is the last row, so on that
//  case as published this operator is the worse of the two on D3Q27 -- and
//  also 3x worse than the paper's own D3Q19 CM result of 0.0134, which is not
//  explained here and should not be written up as though it were.
//
//  The reversal is not a defect in either operator. Both put IDENTICAL zeroth
//  and first moments into the post-collision state -- tests/test_phase_field
//  block 6 checks that to 3e-18 -- so every difference between them is the
//  ghost modes, which BGK relaxes at omega and this sends to equilibrium. At
//  omega near 2 BGK's ghosts are nearly reflected rather than damped, and on a
//  pure-advection test that evidently helps; resetting them each step does not.
//
//  What this operator is FOR is therefore the stability ceiling, not accuracy
//  at a shared omega: the reachable Peclet number is in the table written by
//  validation/enan_interface (the operator is named in its lat column), against
//  BGK's NaN rows at Pe = 400, 800 and 4000 on the Zalesak disk.
//
//  PRODUCT LATTICES ONLY. The transform is three one-dimensional passes, which
//  needs the velocity set to be a tensor product: D2Q9 and D3Q27, not D3Q19 and
//  not D2Q5 or D3Q7. The paper's own scheme is D3Q19 and reaches the same place
//  by writing out a 19x19 transform; that is a different piece of work and is
//  not done here. Pair this with PhaseFieldBGK on the reduced lattices.
//==============================================================================
#include "collision/PhaseFieldBGK.hpp"
#include "collision/ProductBasis.hpp"
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

template <class L>
struct PhaseFieldCentralMoments {
  using Lattice = L;
  using Basis   = ProductBasis<L>;
  using Storage = RawPopulations;
  static constexpr const char* name = "PhaseFieldCM";
  static constexpr int NM = Basis::NM;

  static_assert(Basis::enabled,
                "the phase-field central-moment collision needs a product "
                "lattice: D2Q9 or D3Q27. Use PhaseFieldBGK on D2Q5, D3Q7 or "
                "D3Q19.");

  Real omega = Real(1);      // sets the mobility, exactly as in PhaseFieldBGK
  Real width = Real(4);      // interface width W, in lattice units

  static Real omega_from_mobility(Real m) {
    return Real(1) / (m * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real mobility_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
  }
  KOKKOS_INLINE_FUNCTION Real mobility() const {
    return (Real(1) / omega - Real(0.5)) * cs2<L, Real>();
  }

  KOKKOS_INLINE_FUNCTION
  static Real order_parameter(const Real h[L::Q]) {
    Real s = Real(0);
    for (int i = 0; i < L::Q; ++i) s += h[i];
    return s;
  }

  // The product-form equilibrium, so that the central moments are diagonal.
  // Reached by inverse-transforming k^eq = (phi, 0, ..., 0) rather than by
  // evaluating a Hermite series, which is both cheaper and exactly consistent
  // with what the collision below relaxes toward.
  KOKKOS_INLINE_FUNCTION
  static void eq_populations(Real phi, const Real u[3], Real g[L::Q]) {
    Real k[NM];
    for (int n = 0; n < NM; ++n) k[n] = Real(0);
    k[Basis::index_of(0, 0, 0)] = phi;
    Basis::to_populations(k, u, g);
  }

  // Single-population equilibrium, for the solver's seeding path. Only ever
  // called at u = 0 there, but written for a general u so it cannot silently
  // become wrong if that changes.
  KOKKOS_INLINE_FUNCTION
  Real eq(int i, Real phi, Real ux, Real uy, Real uz) const {
    Real g[L::Q];
    const Real u[3] = {ux, uy, uz};
    eq_populations(phi, u, g);
    return g[i];
  }

  // Identical to PhaseFieldBGK's, deliberately: see the banner.
  KOKKOS_INLINE_FUNCTION
  void anti_diffusion(Real phi, const Real G[3], Real A[3]) const {
    const Real g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    const Real gn = Kokkos::sqrt(g2);
    if (!(gn > Real(1e-12))) { A[0] = A[1] = A[2] = Real(0); return; }
    const Real s = (Real(4) / width) * phi * (Real(1) - phi) / gn;
    A[0] = s * G[0];  A[1] = s * G[1];  A[2] = s * G[2];
  }

  //----------------------------------------------------------------------------
  // Eq. (51) with the relaxation matrix of Eq. (55) and the source of Eq. (61).
  //
  //   k*_0        = phi                                        conserved
  //   k*_{1,2,3}  = (1 - omega) k_{1,2,3} + (1 - omega/2) cs2 A
  //   k*_rest     = 0                                straight to equilibrium
  //
  // WHY THE SOURCE IS THREE TERMS HERE AND NINE IN THE PAPER. Their Eq. (61)
  // lists nine nonzero entries -- R_{1,2,3} = F and, at third order,
  // R10 = R15 = F_y cs2, R11 = R13 = F_x cs2, R12 = R14 = F_z cs2 -- and it is
  // tempting to read that as six slots missing from the three lines above.
  // They are not missing; they are zero in THIS basis. Eq. (61) is written in
  // the monomial CMs, whereas ProductBasis is shifted, phi_2 = C^2 - cs2, so
  // the (a,a,b) slot is (C_a^2 - cs2) C_b and the same source contributes
  //
  //     cs4 A_b  -  cs2 * cs2 A_b  =  0
  //
  // to every one of the six. tests/test_phase_field.cpp block 8 checks both
  // halves of that -- the raw moments reproduce their Eq. (61) exactly, and
  // the shifted ones outside the first three are 4e-20 -- because the trap is
  // live in both directions: adding the six here would double-count them, and
  // dropping them from a monomial implementation would lose them.
  //
  // THE cs2 IS NOT DECORATION. PhaseFieldBGK adds S_i = (1 - omega/2) w_i
  // (c_i . A) to the populations, whose first moment is (1 - omega/2) cs2 A
  // because sum_i w_i c_ia c_ib = cs2 delta_ab. Writing A here without the cs2
  // makes the anti-diffusion three times too weak on a cs2 = 1/3 lattice, which
  // does not blow up and does not look wrong: the interface simply spreads, the
  // way PhaseFieldBGK's banner records W = 4 becoming 14.1 in 600 steps when
  // the same term was got wrong the other way. Block 5 pins it by predicting
  // the post-collision first moment and contracting for it.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real phi, const Real u[3], const Real A[3]) const {
    Real k[NM];
    Basis::to_moments(h, u, k);

    const int i0 = Basis::index_of(0, 0, 0);
    const int ix = Basis::index_of(1, 0, 0);
    const int iy = Basis::index_of(0, 1, 0);
    const int iz = (L::D == 3) ? Basis::index_of(0, 0, 1) : -1;
    constexpr Real cs2v = cs2<L, Real>();
    const Real keep = Real(1) - omega;
    const Real pref = (Real(1) - Real(0.5) * omega) * cs2v;

    const Real k1 = k[ix] * keep + pref * A[0];
    const Real k2 = k[iy] * keep + pref * A[1];
    const Real k3 = (iz >= 0) ? (k[iz] * keep + pref * A[2]) : Real(0);

    for (int n = 0; n < NM; ++n) k[n] = Real(0);
    k[i0] = phi;                      // k*_0: exactly conserved
    k[ix] = k1;  k[iy] = k2;
    if (iz >= 0) k[iz] = k3;

    Basis::to_populations(k, u, h);
  }
};

}  // namespace lbm
