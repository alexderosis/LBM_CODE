#pragma once
//==============================================================================
//  Central-moment collision for the pressure-based multiphase operator.
//
//  De Rosis & Enan, Phys. Fluids 33, 043315 (2021), Sec. II.C -- the GMRT-LBM
//  for f_i: Eq. (31) for the basis, (34) for the equilibrium central moments,
//  (35) for the relaxation and (36) for the post-collision moments.
//
//  ============================ WHY THIS EXISTS ==============================
//  BGK relaxes EVERY mode at omega, so at high Reynolds number, where omega
//  approaches 2, the non-hydrodynamic modes are over-relaxed and ring instead of
//  decaying. That is not a stability margin one can tune around: at Re = 30000
//  on a 128-wide box, omega = 1.99795. The whole point of a moment operator is
//  that the modes which carry no physics can be sent straight to equilibrium at
//  rate 1 while the shear modes alone carry the viscosity.
//
//  Measured symptom before this existed: a Rayleigh-Taylor run at a density
//  ratio of 999 completed under BGK but its velocity field was dominated by a
//  one-cell alternating mode, 70x stronger than the same case at a ratio of 3.
//  ===========================================================================
//
//  THE EQUILIBRIUM MOMENTS FACTORISE, and that is the one derivation worth
//  writing down. The pressure-form equilibrium splits exactly:
//
//      f_i^eq = w_i [ p~ + Phi_i(u) ] = [ w_i (1 + Phi_i) ] + (p~ - 1) w_i
//             = (Maxwellian at rho = 1) + (p~ - 1) (weights).
//
//  The first bracket has central moments {1, 0, 0, ...} -- exactly, for the
//  product-form equilibrium this code base already verified symbolically. The
//  second is (p~ - 1) times the central moments of the WEIGHTS about u, and on a
//  product lattice those factorise into the 1D triple {1, -u, u^2}, which is
//  precisely ProductBasis::eq_1d's `Aw`. Hence
//
//      k_eq(p,q,r) = [pqr == 000] + (p~ - 1) A(p,ux) A(q,uy) A(r,uz),
//      A = {1, -u, u^2}.
//
//  No dense Q x Q matrix, no stored equilibrium vector, and the same factorised
//  transform the single-phase operator uses. Checked against Eq. (34) of the
//  reference on four independent moments: k_5 = (ux^2 - uy^2)(p~ - 1),
//  k_7 = ux uy (p~ - 1), and -- after converting this Hermite-like basis to the
//  paper's monomial one via k_mono(2,1) = k(2,1) + cs2 k(0,1) -- their k_10 and
//  k_16 come out identical too. The paper's odd-order moments are printed
//  without the leading minus that the derivation gives; the even ones agree
//  outright, which is what identifies it as a typography rather than a scheme
//  difference. tests/test_multiphase.cpp checks the whole vector against a
//  direct contraction rather than against the paper.
//
//  ONE DELIBERATE DEPARTURE. The reference builds its equilibrium central
//  moments from the SECOND-ORDER truncated equilibrium of its Eq. (10). This
//  uses the product-form one instead, whose central moments are exactly
//  Maxwellian on D2Q9 and D3Q27, so the O(u^3) Galilean defects of the
//  truncation are absent rather than reproduced. That is the same choice
//  MomentCollision makes for single-phase flow, for the same reason.
//
//  RELAXATION, following Eq. (35) K = diag[1,1,1,1,1, w,w,w,w,w, 1,...,1]:
//      order 0     conserved -- p~ is untouched
//      order 1     exact, not relaxed: see THE FORCE below
//      order 2     trace at omega_bulk (the paper's 1), deviatoric and shear at w
//      order >= 3  straight to equilibrium
//
//  THE FORCE IS EXACT IN THE FIRST MOMENT, not relaxed into it. With
//  u = sum_i c_i f_i + F/(2 rho), the pre-collision first moment is
//
//      k_1 = sum_i f_i (c_x - u_x) = k_eq(1,0) - F_x / (2 rho),
//
//  and the post-collision one has to be k_eq(1,0) + F_x / (2 rho) for the next
//  step to recover the right velocity. So the update is an assignment, not a
//  relaxation, and the rate-1 entry in the paper's K is that assignment.
//
//  Note the source of Eq. (14), F_i = w_i (c_i.F) / (rho cs2), has NO second
//  moment -- it is odd in c_i -- so unlike Guo's there is no second-order force
//  term to add, in any basis. The elegance of collision in the co-moving frame
//  is not even needed here.
//
//  RAW STORAGE ONLY. ShiftedPopulations centres the stored variable on p~ = 1,
//  which is exactly the pressure gauge that must be avoided at a density ratio
//  -- see the banner in MultiphasePotentialBGK.hpp.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/ProductBasis.hpp"
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

template <class L>
struct MultiphaseCentralMoments {
  using Lattice = L;
  using Storage = RawPopulations;
  using Basis   = ProductBasis<L>;
  using ForcingPolicy = NoForcing;              // the force is internal
  static constexpr const char* name = "MultiphaseCentralMoments";
  static constexpr int D  = L::D;
  static constexpr int NM = Basis::NM;

  static_assert(L::supports_navier_stokes,
                "the multiphase fluid operator needs a Navier-Stokes lattice.");
  static_assert(Basis::enabled,
                "this operator needs a product lattice (D2Q9 or D3Q27); D3Q19 "
                "would need the monomial basis and its own equilibrium moments.");

  // Phase field and its derivatives, owned by PhaseFieldSolver.
  View1D<Real> phi, Gx, Gy, Gz, Lap;
  // Viscous interface force; empty is allowed and means matched density.
  View1D<Real> Vx, Vy, Vz;

  Real phi_L = Real(0), phi_H = Real(1);
  Real rho_L = Real(1), rho_H = Real(1);
  Real mu_L  = Real(0.1), mu_H = Real(0.1);     // DYNAMIC viscosities
  Real beta = Real(0), kappa = Real(0);
  Real bx = Real(0), by = Real(0), bz = Real(0);
  // The paper sends the trace to equilibrium at 1. Exposed because it is the
  // bulk viscosity and nothing else in the scheme pins it.
  Real omega_bulk = Real(1);

  static Real kappa_from_sigma(Real sigma, Real width) { return Real(1.5) * sigma * width; }
  static Real beta_from_sigma (Real sigma, Real width) { return Real(12) * sigma / width; }

  //---- identical to MultiphasePotentialBGK; the difference is the collision ----
  struct Local { Real p, rho, mu, omega; };

  KOKKOS_INLINE_FUNCTION Local local(Index n) const {
    constexpr Real cs2v = cs2<L, Real>();
    const Real p = phi(n);
    const Real s = (p - phi_L) / (phi_H - phi_L);
    const Real r = rho_L + s * (rho_H - rho_L);
    const Real m = mu_L + s * (mu_H - mu_L);
    const Real tau = m / (r * cs2v);
    return Local{p, r, m, Real(1) / (tau + Real(0.5))};
  }
  KOKKOS_INLINE_FUNCTION Real density_at(Index n) const { return local(n).rho; }
  KOKKOS_INLINE_FUNCTION Real omega_at(Index n) const { return local(n).omega; }
  KOKKOS_INLINE_FUNCTION Real viscosity_at(Index n) const {
    const Local l = local(n);  return l.mu / l.rho;
  }
  KOKKOS_INLINE_FUNCTION Real drho_dphi() const {
    return (rho_H - rho_L) / (phi_H - phi_L);
  }

  KOKKOS_INLINE_FUNCTION
  void force(const Local& l, Index n, Real p_tilde, Real F[3]) const {
    constexpr Real cs2v = cs2<L, Real>();
    const Real phi0 = Real(0.5) * (phi_L + phi_H);
    const Real mu_phi = Real(4) * beta * (l.p - phi_L) * (l.p - phi_H) * (l.p - phi0)
                      - kappa * Lap(n);
    const Real coef = mu_phi - p_tilde * cs2v * drho_dphi();
    const bool have_v = Vx.data() != nullptr;
    F[0] = coef * Gx(n) + (have_v ? Vx(n) : Real(0)) + l.rho * bx;
    F[1] = coef * Gy(n) + (have_v ? Vy(n) : Real(0)) + l.rho * by;
    F[2] = (L::D == 3) ? coef * Gz(n) + (have_v ? Vz(n) : Real(0)) + l.rho * bz
                       : Real(0);
  }

  KOKKOS_INLINE_FUNCTION static Real density(const Macro& m) { return m.dens; }

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
    force(l, n, s, F);
    const Real h = Real(0.5) / l.rho;
    m.ux = mx + h * F[0];  m.uy = my + h * F[1];  m.uz = mz + h * F[2];
    return m;
  }

  //----------------------------------------------------------------------------
  // The weight factors A = {1, -u, u^2}, per axis. `Aw[2]` is {1, 0, 0} in two
  // dimensions, so the r index is always 0 there and the third factor is 1.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static void weight_factors(const Real u[3], Real Aw[3][3]) {
    for (int a = 0; a < 3; ++a) {
      const Real ua = (a < D) ? u[a] : Real(0);
      Aw[a][0] = Real(1);  Aw[a][1] = -ua;  Aw[a][2] = ua * ua;
    }
  }

  // k_eq(n) = [n == 0] + (p~ - 1) * prod_a A(order_a, u_a)
  KOKKOS_INLINE_FUNCTION
  static Real eq_moment(int n, Real p_tilde, const Real Aw[3][3]) {
    const int p = Basis::p_of(n), q = Basis::q_of(n), r = Basis::r_of(n);
    Real w = Aw[0][p] * Aw[1][q];
    if constexpr (D == 3) w *= Aw[2][r];
    return ((n == 0) ? Real(1) : Real(0)) + (p_tilde - Real(1)) * w;
  }

  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], const Macro& m, Index n = 0) const {
    const Local l = local(n);
    const Real pt = m.dens;
    const Real u[3] = {m.ux, m.uy, m.uz};
    Real F[3];
    force(l, n, pt, F);

    Real Aw[3][3];
    weight_factors(u, Aw);

    Real k[NM];
    Basis::template to_moments<true>(f, u, k);

    const Real w  = l.omega;
    const Real wb = omega_bulk;
    const Real hr = Real(0.5) / l.rho;

    // ---- order 1: exact assignment, the force included ----
    for (int a = 0; a < D; ++a)
      k[i1(a)] = eq_moment(i1(a), pt, Aw) + hr * F[a];

    // ---- order 2: trace at omega_bulk, deviatoric and shear at omega ----
    {
      Real d[3], e[3];
      Real tr = Real(0), tre = Real(0);
      for (int a = 0; a < D; ++a) {
        d[a] = k[i2d(a)];
        e[a] = eq_moment(i2d(a), pt, Aw);
        tr += d[a];  tre += e[a];
      }
      const Real invD = Real(1) / Real(D);
      const Real tr_post = (Real(1) - wb) * tr + wb * tre;
      for (int a = 0; a < D; ++a)
        k[i2d(a)] = (Real(1) - w) * (d[a] - tr * invD)
                  + w * (e[a] - tre * invD) + tr_post * invD;
      for (int a = 0; a < D; ++a)
        for (int b = a + 1; b < D; ++b) {
          const int id = i2s(a, b);
          k[id] = (Real(1) - w) * k[id] + w * eq_moment(id, pt, Aw);
        }
    }

    // ---- order >= 3: straight to equilibrium, which is the whole point ----
    for (int j = 0; j < NM; ++j)
      if (Basis::order(j) >= 3) k[j] = eq_moment(j, pt, Aw);

    Basis::template to_populations<true>(k, u, f);
  }

  //----------------------------------------------------------------------------
  // Equilibrium populations, by inverse-transforming the equilibrium moments in
  // the co-moving basis. FlowState::rho is the NORMALISED PRESSURE here.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real p_tilde, Real ux, Real uy, Real uz) {
    const Real u[3] = {ux, uy, uz};
    Real Aw[3][3];
    weight_factors(u, Aw);
    Real k[NM], f[L::Q];
    for (int j = 0; j < NM; ++j) k[j] = eq_moment(j, p_tilde, Aw);
    Basis::template to_populations<true>(k, u, f);
    return f[i];
  }

  // Moment slots, located through the basis rather than hardcoded.
  static constexpr int i1(int a) {
    return Basis::index_of(a == 0, a == 1, (D == 3) && a == 2);
  }
  static constexpr int i2d(int a) {
    return Basis::index_of(2 * (a == 0), 2 * (a == 1), (D == 3) ? 2 * (a == 2) : 0);
  }
  static constexpr int i2s(int a, int b) {
    return Basis::index_of((a == 0 || b == 0), (a == 1 || b == 1),
                           (D == 3) && (a == 2 || b == 2));
  }
};

}  // namespace lbm
