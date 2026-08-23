#pragma once
//==============================================================================
//  Moment-space collision: raw MRT and CENTRAL MOMENTS from one implementation.
//
//  The only difference between them is the velocity the basis is built at:
//
//      Central = true   ->  basis velocity u_b = u   (central moments)
//      Central = false  ->  basis velocity u_b = 0   (raw moments)
//
//  which is exactly the `central_moments` toggle in MATLAB/*_CM.m.
//
//  EQUILIBRIUM IN CLOSED FORM. The product basis {1, C, C^2-cs2} diagonalises
//  the Maxwellian: writing du = u - u_b, the equilibrium moments factorise as
//
//      k_eq{pqr} = rho * Q(p,du_x) Q(q,du_y) Q(r,du_z),   Q = {1, du, du^2}
//
//  For central moments du = 0, so k_eq is rho in the zeroth moment and EXACTLY
//  ZERO everywhere else. (Verified symbolically against T*feq for both D2Q9 and
//  D3Q27.) No equilibrium populations are ever formed and no K_eq vector is
//  stored -- which is also why this never needs the dense Q x Q matrices.
//
//  SHIFTED STORAGE. When the arrays hold g = f - w, the moments of w must be
//  subtracted. The weights of a product lattice factorise too (w = W(cx)W(cy)W(cz)
//  with W(0) = 2/3, W(+-1) = 1/6), and their 1D moments in this basis are
//  (1, -u_b, u_b^2), so
//
//      W{pqr} = A(p,u_bx) A(q,u_by) A(r,u_bz),   A = {1, -u_b, u_b^2}
//
//  and the whole scheme goes through with k_eq -> k_eq - W. Nothing is formed as
//  a difference of two O(w) quantities.
//
//  RELAXATION (matching the corrected MATLAB scripts):
//      order 0    conserved
//      order 1    k* = k + F        (the force enters here and only here when
//                                    Central = true -- see below)
//      order 2    trace at omega_bulk, deviatoric and shear at omega
//      order >= 3 set to equilibrium
//
//  FORCING. The Guo source has moments sum(c_a F_i) = F_a and
//  sum(c_a c_b F_i) = u_a F_b + u_b F_a. Transformed to the basis at u_b these
//  become F_a and (F_b du_a + F_a du_b). For CENTRAL moments du = 0, so the
//  second-order force contribution vanishes identically and the force enters
//  only through the first moments -- the well-known elegance of collision in the
//  co-moving frame. For raw MRT it does not vanish and is applied with the usual
//  (1 - omega/2) prefactor.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MonomialBasis.hpp"
#include "collision/ProductBasis.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Which moment basis a lattice uses. Product lattices get the factorised
// transform; D3Q19 is not one (it is D3Q27 minus its eight corners) and uses the
// generated 19-monomial basis instead. Both expose the same interface, so the
// operator below does not care which it got.
//------------------------------------------------------------------------------
template <class L> struct SelectBasis { using type = ProductBasis<L>; };
template <> struct SelectBasis<D3Q19> { using type = MonomialBasis<D3Q19>; };

template <class L, class Forcing = NoForcing, class Store = RawPopulations,
          bool Central = true>
struct MomentCollision {
  using Lattice = L;
  using Storage = Store;
  using Basis   = typename SelectBasis<L>::type;
  static constexpr const char* name = Central ? "CentralMoments" : "MRT";
  static constexpr int D  = L::D;
  static constexpr int NM = Basis::NM;

  static_assert(L::supports_navier_stokes,
                "moment collision as a Navier-Stokes operator needs a lattice "
                "with isotropic 4th-order moments.");
  static_assert(Basis::enabled, "no moment basis is available for this lattice.");

  Real omega      = Real(1);   // shear: sets the viscosity
  Real omega_bulk = Real(-1);  // trace; < 0 means "follow omega"
  Forcing forcing{};

  static Real omega_from_viscosity(Real nu) {
    return Real(1) / (nu * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real viscosity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
  }
  KOKKOS_INLINE_FUNCTION Real bulk() const {
    return omega_bulk < Real(0) ? omega : omega_bulk;
  }

  KOKKOS_INLINE_FUNCTION static Real density(const Macro& m) {
    if constexpr (Store::shifted) return Real(1) + m.dens;
    else                          return m.dens;
  }

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
    const Real rho = density(m);
    const Real ir  = Real(1) / rho;
    m.ux = mx * ir;  m.uy = my * ir;  m.uz = mz * ir;
    forcing.shift_velocity(n, rho, m.ux, m.uy, m.uz);
    return m;
  }

  //----------------------------------------------------------------------------
  // Equilibrium moments in the STORED variable, as three 1D factor triples.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static void eq_factors(const Real du[3], const Real ub[3],
                         Real Qf[3][3], Real Aw[3][3]) {
    for (int a = 0; a < 3; ++a) Basis::eq_1d(du[a], ub[a], Qf[a], Aw[a]);
  }

  // Equilibrium of moment slot n, expressed in the STORED variable.
  KOKKOS_INLINE_FUNCTION
  static Real eq_moment(Real rho, const Real Qf[3][3], const Real Aw[3][3], int n) {
    const int p = Basis::p_of(n), q = Basis::q_of(n), r = Basis::r_of(n);
    Real e = rho * Qf[0][p] * Qf[1][q];
    Real w = Aw[0][p] * Aw[1][q];
    if constexpr (D == 3) { e *= Qf[2][r]; w *= Aw[2][r]; }
    if constexpr (Store::shifted) return e - w;
    else                          return e;
  }

  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], const Macro& m, Index n = 0) const {
    const Real rho = density(m);
    const Real u[3] = {m.ux, m.uy, m.uz};
    Real ub[3], du[3];
    for (int a = 0; a < 3; ++a) {
      ub[a] = Central ? u[a] : Real(0);
      du[a] = u[a] - ub[a];
    }
    Real Qf[3][3], Aw[3][3];
    eq_factors(du, ub, Qf, Aw);

    Real k[NM];
    Basis::template to_moments<Central>(f, ub, k);

    Real F[3]; forcing.at(n, F);
    const Real w2 = Real(1) - Real(0.5) * omega;
    const Real wb = bulk();

    // ---- order 1: the force enters here (and, when Central, only here) ----
    if constexpr (Forcing::active)
      for (int a = 0; a < D; ++a) k[i1(a)] += F[a];

    // ---- order 2: trace at omega_bulk, deviatoric and shear at omega ----
    {
      Real d[3], e[3], g[3];
      Real tr = Real(0), tre = Real(0), trg = Real(0);
      for (int a = 0; a < D; ++a) {
        d[a] = k[i2d(a)];
        e[a] = eq_moment(rho, Qf, Aw, i2d(a));
        g[a] = Forcing::active ? Real(2) * F[a] * du[a] : Real(0);
        tr += d[a]; tre += e[a]; trg += g[a];
      }
      const Real invD = Real(1) / Real(D);
      const Real tr_post = (Real(1) - wb) * tr + wb * tre +
                           (Real(1) - Real(0.5) * wb) * trg;
      for (int a = 0; a < D; ++a)
        k[i2d(a)] = (Real(1) - omega) * (d[a] - tr * invD)
                  + omega * (e[a] - tre * invD)
                  + w2 * (g[a] - trg * invD) + tr_post * invD;
      for (int a = 0; a < D; ++a)
        for (int b = a + 1; b < D; ++b) {
          const int id = i2s(a, b);
          const Real ge = Forcing::active ? (F[b] * du[a] + F[a] * du[b]) : Real(0);
          k[id] = (Real(1) - omega) * k[id] + omega * eq_moment(rho, Qf, Aw, id) + w2 * ge;
        }
    }

    // ---- order >= 3: straight to equilibrium ----
    for (int n = 0; n < NM; ++n)
      if (Basis::order(n) >= 3) k[n] = eq_moment(rho, Qf, Aw, n);

    // ---- order 3: NO extra force term is needed, and adding one is wrong ----
    //
    // Expanding the body force in Hermite polynomials to fourth order and
    // transforming to MONOMIAL central moments gives, on D2Q9,
    //
    //     K_force = [0, Fx, Fy, 0, 0, 0, cs^2 Fy, cs^2 Fx, 0]
    //
    // in the ordering (00,10,01,20,02,11,21,12,22) -- a pair of nonzero THIRD
    // order moments alongside the momentum source. It is tempting to conclude
    // that putting the force only in k_1, as above, misses them.
    //
    // It does not. This basis is Hermite, phi_2(C) = C^2 - cs^2, not monomial,
    // and the two are related by
    //
    //     k_21(monomial) = k_21(this basis) + cs^2 * k_01(this basis).
    //
    // So adding F to the first-order slot alone already delivers
    // k_21 = cs^2 Fy and k_12 = cs^2 Fx in monomial central moments, for free,
    // from the cs^2 the basis function carries. Adding them explicitly on top
    // double counts: measured, it gives 1.5 cs^2 Fy where cs^2 Fy is correct.
    //
    // validation/forcing_cm.cpp checks both representations against the Hermite
    // expansion and confirms the equality to machine precision.
    Basis::template to_populations<Central>(k, ub, f);
  }

  //----------------------------------------------------------------------------
  // Equilibrium populations, by inverse-transforming the equilibrium moments.
  // Initialisation only.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real rho, Real ux, Real uy, Real uz) {
    const Real ub[3] = {Real(0), Real(0), Real(0)};   // raw basis: k_eq is explicit
    const Real du[3] = {ux, uy, uz};
    Real Qf[3][3], Aw[3][3];
    eq_factors(du, ub, Qf, Aw);
    Real k[NM], f[L::Q];
    for (int n = 0; n < NM; ++n) k[n] = eq_moment(rho, Qf, Aw, n);
    Basis::template to_populations<false>(k, ub, f);
    return f[i];
  }

 private:
  // Moment slots, located through the basis rather than hardcoded, so the same
  // relaxation works for the product basis and for D3Q19's monomial basis.
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
  // Third-order slot carrying exponent 2 on axis a and 1 on axis b (a != b),
  // i.e. k_{aab}. On D2Q9 these are the only third-order moments the force
  // reaches.
  static constexpr int i3(int a, int b) {
    return Basis::index_of(2 * (a == 0) + (b == 0),
                           2 * (a == 1) + (b == 1),
                           (D == 3) ? (2 * (a == 2) + (b == 2)) : 0);
  }
};

template <class L, class F = NoForcing, class S = RawPopulations>
using CentralMoments = MomentCollision<L, F, S, true>;
template <class L, class F = NoForcing, class S = RawPopulations>
using MRT = MomentCollision<L, F, S, false>;

}  // namespace lbm
