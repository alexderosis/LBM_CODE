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
//  WHAT IT COSTS. Two three-pass transforms per node against BGK's single loop.
//
//  A RETRACTION, AND IT MATTERS BECAUSE IT WAS THE CLAIM THIS OPERATOR WAS
//  JUSTIFIED BY. An earlier version of this banner reported that the operator
//  completes De Rosis & Enan's Table III at Pe = 80, 400, 800 and 4000 where
//  BGK reaches 1e300 on the last three, and read that as a stability ceiling
//  the central-moment form lifts. The Peclet number had been read as
//  Pe = U0 xi / M, which is what their paper's text says. Their drivers all
//  compute
//
//      M = U_ref * d / Pe
//
//  with d the DOMAIN SIZE. On Table III at Pe = 80 that is M = 0.0497 and
//  omega = 1.54; the interface-width reading gives M = 7.5e-4 and omega =
//  1.991. So BGK was being run 66 times short of mobility, hard against the
//  stability edge, and it diverged for that reason and not because of anything
//  intrinsic to BGK at their operating point. ACROSS EVERY CASE IN THE PAPER
//  THEIR OMEGA NEVER EXCEEDS 1.988.
//
//  The comparison is being rerun at their mobilities. Until it lands, this
//  operator has NO measured advantage over PhaseFieldBGK recorded here, and
//  the numbers previously tabulated in this banner -- the mobility ladder and
//  the cycle scan -- describe a regime the paper does not visit. They were
//  real measurements of the wrong thing.
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
  // Transform the source at the actual velocity instead of truncating it at
  // u = 0. Their drivers truncate, and so does the default here, because
  // MEASURED IT MAKES NO DIFFERENCE. On the advected flat slab of
  // validation/phase_flat -- 50 000 steps, W = 3, the case that can see this
  // if anything can -- the width drift is
  //
  //     omega      truncated     transformed
  //     1.9881     -4.58e-2      -4.52e-2
  //     1.9940     +7.43e-1      +7.45e-1
  //
  // 1.3 % and 0.3 % apart. The switch is kept because the reasoning that says
  // it SHOULD matter is sound -- the discarded moments are O(|u| A), which is
  // zero at rest and not under advection -- and the next person will have it
  // too. It is simply not what limits this operator.
  bool full_source = false;

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
  // THAT CANCELLATION IS AT u = 0, and the qualifier matters. Transform S_i at
  // a general velocity and 26 of the 27 shifted moments are nonzero, not 3;
  // the sparsity is a property of the rest frame. Their drivers add a
  // u-INDEPENDENT source and so does this, so both truncate the source's
  // central moments at u = 0 -- a shared approximation rather than an
  // identity, and MATLAB/D3Q27_CM_phase.m prints both so the difference stays
  // visible. The cost is O(|u| A) in the higher slots, which is the same order
  // as the O(|u|^3) equilibrium gap against BGK that block 6 measures.
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

    if (!full_source) {
      // The u = 0 truncation, which is what their drivers use.
      const Real k1 = k[ix] * keep + pref * A[0];
      const Real k2 = k[iy] * keep + pref * A[1];
      const Real k3 = (iz >= 0) ? (k[iz] * keep + pref * A[2]) : Real(0);
      for (int n = 0; n < NM; ++n) k[n] = Real(0);
      k[i0] = phi;
      k[ix] = k1;  k[iy] = k2;
      if (iz >= 0) k[iz] = k3;
    } else {
      // The source transformed at the ACTUAL velocity, so every slot it
      // occupies is filled. Identical to the branch above at u = 0 and not
      // otherwise: see the banner.
      Real S[L::Q];
      const Real sp = Real(1) - Real(0.5) * omega;
      for (int i = 0; i < L::Q; ++i) {
        const Real cA = Real(cvel<L>(i, 0)) * A[0] + Real(cvel<L>(i, 1)) * A[1] +
                        Real(cvel<L>(i, 2)) * A[2];
        S[i] = sp * weight<L, Real>(i) * cA;
      }
      Real r[NM];
      Basis::to_moments(S, u, r);
      const Real k1 = k[ix] * keep;
      const Real k2 = k[iy] * keep;
      const Real k3 = (iz >= 0) ? (k[iz] * keep) : Real(0);
      for (int n = 0; n < NM; ++n) k[n] = r[n];
      k[i0] = phi;                    // R_0 = 0, so phi is still conserved
      k[ix] = k1 + r[ix];  k[iy] = k2 + r[iy];
      if (iz >= 0) k[iz] = k3 + r[iz];
    }

    Basis::to_populations(k, u, h);
  }
};

}  // namespace lbm
