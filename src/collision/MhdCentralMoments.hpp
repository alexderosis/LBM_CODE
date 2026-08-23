#pragma once
#include <type_traits>
//==============================================================================
//  Hybrid central-moment MHD collision for D2Q9, after
//
//    A. De Rosis, E. Leveque, R. Chahine, "Advanced lattice Boltzmann scheme for
//    high-Reynolds-number magneto-hydrodynamic flows", J. Turbulence 19(6) 2018,
//    Equations (7)-(13).
//
//  Central moments are taken in the paper's non-orthogonal basis (its Eq. 8),
//
//    t0 = 1        t1 = Cx       t2 = Cy       t3 = Cx^2 + Cy^2
//    t4 = Cx^2 - Cy^2            t5 = Cx Cy    t6 = Cx^2 Cy
//    t7 = Cx Cy^2                t8 = Cx^2 Cy^2,        C = c - u,
//
//  which is the monomial central-moment set with the two normal stresses written
//  as their sum and difference. Note this is NOT the shifted basis used by
//  MomentCollision: there phi_2 = C^2 - cs^2, chosen because it diagonalises the
//  product-form equilibrium. Here the equilibrium is the paper's, so the paper's
//  basis is used and the transform is written out directly.
//
//  EQUILIBRIUM (Eq. 11). These are the central moments of the SECOND-ORDER
//  truncated equilibrium plus the Maxwell term of Eq. (1) -- not of a
//  product-form equilibrium. That is why the hydrodynamic parts are nonzero at
//  third and fourth order: -rho ux^2 uy in k6 and 3 rho ux^2 uy^2 in k8 are
//  exactly the Galilean defects of the truncation, and reproducing the paper
//  means reproducing them rather than "improving" them away.
//
//  RELAXATION (Eq. 12): k*_i = k_i + omega_i (k_i^eq - k_i) for i = 3..8, with
//  omega_4 = omega_5 = omega_v setting the viscosity, omega_3 the bulk
//  viscosity, and omega_6 = omega_7 = omega_8 = 1 sending the ghost moments
//  straight to equilibrium. k0, k1, k2 are collision invariants.
//
//  Storage is raw. The paper's scheme is defined on f itself, and the shifted
//  variant would need the weight moments carried through Eq. (11); that is a
//  separate exercise from reproducing the published scheme.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/ProductBasis.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

// `HighOrder` selects the hydrodynamic part of the equilibrium:
//   true  -- the highest order the lattice admits (product form on D2Q9/D3Q27),
//            whose central moments are exactly Maxwellian, so every rho-dependent
//            term above second order drops out of the equilibrium moments;
//   false -- the second-order truncation, which is what the published Eq. (11)
//            is built on. Kept so Table 1 of the paper remains reproducible.
// The Maxwell-stress part is identical either way.
template <class L, bool HighOrder = true>
struct MhdCentralMoments;

//------------------------------------------------------------------------------
//  D2Q9 -- the published two-dimensional scheme, Eq. (11) written out directly.
//------------------------------------------------------------------------------
template <bool HighOrder>
struct MhdCentralMoments<D2Q9, HighOrder> {
  using Lattice     = D2Q9;
  using Equilibrium = std::conditional_t<HighOrder, ProductFormEquilibrium<D2Q9>,
                                                    SecondOrderEquilibrium<D2Q9>>;
  using Storage     = RawPopulations;
  static constexpr const char* name = "MhdCM";

  Real omega      = Real(1);    // omega_4 = omega_5, sets the viscosity
  Real omega_bulk = Real(1);    // omega_3
  View1D<Real> Bx, By, Bz;
  NoForcing forcing{};

  static Real omega_from_viscosity(Real nu) {
    return Real(1) / (nu * inv_cs2<D2Q9, Real>() + Real(0.5));
  }
  static Real viscosity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<D2Q9, Real>();
  }
  KOKKOS_INLINE_FUNCTION static Real density(const Macro& m) { return m.dens; }

  KOKKOS_INLINE_FUNCTION
  Macro macroscopic(const Real f[9], Index = 0) const {
    Real s = Real(0), mx = Real(0), my = Real(0);
    for (int i = 0; i < 9; ++i) {
      s  += f[i];
      mx += f[i] * Real(D2Q9::cx(i));
      my += f[i] * Real(D2Q9::cy(i));
    }
    const Real ir = Real(1) / s;
    return Macro{s, mx * ir, my * ir, Real(0)};
  }

  //----------------------------------------------------------------------------
  // 1D monomial central-moment transform: values at c = -1,0,+1 -> (m0, m1, m2)
  // with m_p = sum_c (c-u)^p g_c. Same factorisation as ProductBasis, without the
  // cs^2 subtraction, because this basis is the plain monomial one.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static void fwd(Real& a, Real& b, Real& c, Real u) {
    const Real s0 = a + b + c, s1 = c - a, s2 = c + a;
    a = s0;
    b = s1 - u * s0;
    c = s2 - Real(2) * u * s1 + u * u * s0;
  }
  KOKKOS_INLINE_FUNCTION
  static void inv(Real& a, Real& b, Real& c, Real u) {
    const Real m0 = a, m1 = b, m2 = c;
    const Real s1 = m1 + u * m0;
    const Real s2 = m2 + Real(2) * u * s1 - u * u * m0;
    a = Real(0.5) * (s2 - s1);
    b = m0 - s2;
    c = Real(0.5) * (s2 + s1);
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real f[9], const Macro& mac, Index n = 0) const {
    using B = ProductBasis<D2Q9>;
    const Real rho = mac.dens, ux = mac.ux, uy = mac.uy;
    const Real bx = Bx(n), by = By(n);

    // populations -> monomial central moments m[p][q]
    Real m[3][3];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) m[a][b] = f[B::pi(a, b)];
    for (int a = 0; a < 3; ++a) fwd(m[a][0], m[a][1], m[a][2], uy);
    for (int q = 0; q < 3; ++q) fwd(m[0][q], m[1][q], m[2][q], ux);

    // paper's basis (Eq. 8)
    const Real k3 = m[2][0] + m[0][2];
    const Real k4 = m[2][0] - m[0][2];
    const Real k5 = m[1][1];
    const Real k6 = m[2][1];
    const Real k7 = m[1][2];
    const Real k8 = m[2][2];

    // equilibria (Eq. 11)
    const Real bx2 = bx * bx, by2 = by * by, bxy = bx * by;
    const Real k3e = Real(2) / Real(3) * rho;
    const Real k4e = by2 - bx2;
    const Real k5e = -bxy;
    // The magnetic contributions are common to both equilibria; only the
    // hydrodynamic ones differ, and with the product form they vanish outright
    // because its central moments are Maxwellian (k210 = k120 = 0, k220 = rho/9).
    const Real hyd6 = HighOrder ? Real(0) : -rho * ux * ux * uy;
    const Real hyd7 = HighOrder ? Real(0) : -rho * ux * uy * uy;
    const Real hyd8 = HighOrder ? rho / Real(9)
                                : rho / Real(9) * (Real(27) * ux * ux * uy * uy + Real(1));
    const Real k6e = hyd6 + Real(0.5) * uy * (bx2 - by2) + Real(2) * ux * bxy;
    const Real k7e = hyd7 + Real(0.5) * ux * (by2 - bx2) + Real(2) * uy * bxy;
    const Real k8e = hyd8 + Real(0.5) * (ux * ux - uy * uy) * (bx2 - by2)
                   - Real(4) * ux * uy * bxy;

    // relaxation (Eq. 12); ghost moments 6,7,8 go straight to equilibrium
    const Real k3s = k3 + omega_bulk * (k3e - k3);
    const Real k4s = k4 + omega * (k4e - k4);
    const Real k5s = k5 + omega * (k5e - k5);

    // back to monomial central moments; k0 = rho, k1 = k2 = 0 are invariants
    m[0][0] = rho;
    m[1][0] = Real(0);
    m[0][1] = Real(0);
    m[2][0] = Real(0.5) * (k3s + k4s);
    m[0][2] = Real(0.5) * (k3s - k4s);
    m[1][1] = k5s;
    m[2][1] = k6e;
    m[1][2] = k7e;
    m[2][2] = k8e;

    for (int q = 0; q < 3; ++q) inv(m[0][q], m[1][q], m[2][q], ux);
    for (int a = 0; a < 3; ++a) inv(m[a][0], m[a][1], m[a][2], uy);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) f[B::pi(a, b)] = m[a][b];
    (void)k6; (void)k7; (void)k8;
  }

  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real rho, Real ux, Real uy, Real uz) {
    return SecondOrderEquilibrium<D2Q9>::eq(i, rho, ux, uy, uz);
  }
};

//------------------------------------------------------------------------------
//  D3Q27 -- the three-dimensional extension outlined in \S"extension to three
//  dimensions" of De Rosis, Leveque & Chahine (2018), which prescribes D3Q27 for
//  the velocity field and D3Q7 (theta^2 = 1/4) for the magnetic field.
//
//  Rather than hard-coding 27 equilibrium expressions -- orders 4 to 6 do not
//  factorise compactly and k222 alone runs to two dozen terms -- the equilibrium
//  CENTRAL MOMENTS are obtained by transforming the equilibrium populations,
//  which are the same second-order-plus-Maxwell form as in 2D. That costs one
//  extra population evaluation and one extra forward transform per node; the
//  closed forms are in the document if that ever needs optimising.
//
//  The result was checked symbolically against the closed forms derived from
//  this equilibrium: orders 2 and 3 satisfy
//      k_ab   = rho cs2 delta_ab + M_ab,
//      k_abc  = -rho u_a u_b u_c - (u_a M_bc + u_b M_ac + u_c M_ab),
//  with M_ab = |b|^2 delta_ab / 2 - b_a b_b, and setting w = b_z = 0 reproduces
//  every moment of Eq. (11).
//
//  Relaxation follows the 2D scheme: the trace of the second-order block at
//  omega_bulk, its five deviatoric components at omega, everything of order
//  three and above straight to equilibrium.
//------------------------------------------------------------------------------
template <bool HighOrder>
struct MhdCentralMoments<D3Q27, HighOrder> {
  using Lattice     = D3Q27;
  using Equilibrium = std::conditional_t<HighOrder, ProductFormEquilibrium<D3Q27>,
                                                    SecondOrderEquilibrium<D3Q27>>;
  using Storage     = RawPopulations;
  static constexpr const char* name = "MhdCM3D";

  Real omega      = Real(1);
  Real omega_bulk = Real(1);
  View1D<Real> Bx, By, Bz;
  NoForcing forcing{};

  static Real omega_from_viscosity(Real nu) {
    return Real(1) / (nu * inv_cs2<D3Q27, Real>() + Real(0.5));
  }
  static Real viscosity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<D3Q27, Real>();
  }
  KOKKOS_INLINE_FUNCTION static Real density(const Macro& m) { return m.dens; }

  KOKKOS_INLINE_FUNCTION
  Macro macroscopic(const Real f[27], Index = 0) const {
    Real s = Real(0), mx = Real(0), my = Real(0), mz = Real(0);
    for (int i = 0; i < 27; ++i) {
      s  += f[i];
      mx += f[i] * Real(D3Q27::cx(i));
      my += f[i] * Real(D3Q27::cy(i));
      mz += f[i] * Real(D3Q27::cz(i));
    }
    const Real ir = Real(1) / s;
    return Macro{s, mx * ir, my * ir, mz * ir};
  }

  // Equilibrium populations: the hydrodynamic part at the order selected by
  // `HighOrder` -- product form (sixth order, MATLAB/D3Q27_CM.m) by default --
  // plus the Maxwell stress, which is the same at either order.
  KOKKOS_INLINE_FUNCTION
  void equilibrium(Real fe[27], Real rho, const Real u[3], const Real b[3]) const {
    constexpr Real cs2v = cs2<D3Q27, Real>();
    constexpr Real cs4v = cs2v * cs2v;
    const Real b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    for (int i = 0; i < 27; ++i) {
      const Real c[3] = {Real(D3Q27::cx(i)), Real(D3Q27::cy(i)), Real(D3Q27::cz(i))};
      // M_ab (c_a c_b - cs2 delta_ab) in closed form: with cs2 M_aa = |b|^2 / 6
      // in three dimensions this is 0.5|b|^2|c|^2 - (c.b)^2 - |b|^2/6.
      const Real c2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
      const Real cb = c[0] * b[0] + c[1] * b[1] + c[2] * b[2];
      const Real acc = Real(0.5) * b2 * c2 - cb * cb - b2 / Real(6);
      fe[i] = Equilibrium::eq(i, rho, u[0], u[1], u[2]) +
              weight<D3Q27, Real>(i) * acc / (Real(2) * cs4v);
    }
  }

  // 1D monomial central-moment transform, as in the D2Q9 specialisation
  KOKKOS_INLINE_FUNCTION
  static void fwd(Real& a, Real& b, Real& c, Real u) {
    const Real s0 = a + b + c, s1 = c - a, s2 = c + a;
    a = s0;  b = s1 - u * s0;  c = s2 - Real(2) * u * s1 + u * u * s0;
  }
  KOKKOS_INLINE_FUNCTION
  static void inv(Real& a, Real& b, Real& c, Real u) {
    const Real m0 = a, m1 = b, m2 = c;
    const Real s1 = m1 + u * m0;
    const Real s2 = m2 + Real(2) * u * s1 - u * u * m0;
    a = Real(0.5) * (s2 - s1);  b = m0 - s2;  c = Real(0.5) * (s2 + s1);
  }

  KOKKOS_INLINE_FUNCTION
  static void to_moments(const Real f[27], const Real u[3], Real k[27]) {
    using B = ProductBasis<D3Q27>;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c) k[B::mi(a, b, c)] = f[B::pi(a, b, c)];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        fwd(k[B::mi(a, b, 0)], k[B::mi(a, b, 1)], k[B::mi(a, b, 2)], u[2]);
    for (int a = 0; a < 3; ++a)
      for (int r = 0; r < 3; ++r)
        fwd(k[B::mi(a, 0, r)], k[B::mi(a, 1, r)], k[B::mi(a, 2, r)], u[1]);
    for (int q = 0; q < 3; ++q)
      for (int r = 0; r < 3; ++r)
        fwd(k[B::mi(0, q, r)], k[B::mi(1, q, r)], k[B::mi(2, q, r)], u[0]);
  }
  KOKKOS_INLINE_FUNCTION
  static void to_populations(Real k[27], const Real u[3], Real f[27]) {
    using B = ProductBasis<D3Q27>;
    for (int q = 0; q < 3; ++q)
      for (int r = 0; r < 3; ++r)
        inv(k[B::mi(0, q, r)], k[B::mi(1, q, r)], k[B::mi(2, q, r)], u[0]);
    for (int a = 0; a < 3; ++a)
      for (int r = 0; r < 3; ++r)
        inv(k[B::mi(a, 0, r)], k[B::mi(a, 1, r)], k[B::mi(a, 2, r)], u[1]);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        inv(k[B::mi(a, b, 0)], k[B::mi(a, b, 1)], k[B::mi(a, b, 2)], u[2]);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c) f[B::pi(a, b, c)] = k[B::mi(a, b, c)];
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real f[27], const Macro& mac, Index n = 0) const {
    using B = ProductBasis<D3Q27>;
    const Real rho = mac.dens;
    const Real u[3] = {mac.ux, mac.uy, mac.uz};
    const Real b[3] = {Bx(n), By(n), Bz(n)};

    Real k[27], ke[27], fe[27];
    to_moments(f, u, k);
    equilibrium(fe, rho, u, b);
    to_moments(fe, u, ke);

    // order 0 and 1 are collision invariants
    k[B::mi(0, 0, 0)] = rho;
    k[B::mi(1, 0, 0)] = Real(0);
    k[B::mi(0, 1, 0)] = Real(0);
    k[B::mi(0, 0, 1)] = Real(0);

    // order 2: trace at omega_bulk, the five deviatoric components at omega
    {
      const int d[3] = {B::mi(2, 0, 0), B::mi(0, 2, 0), B::mi(0, 0, 2)};
      Real tr = Real(0), tre = Real(0);
      for (int a = 0; a < 3; ++a) { tr += k[d[a]]; tre += ke[d[a]]; }
      const Real third = Real(1) / Real(3);
      const Real tr_post = (Real(1) - omega_bulk) * tr + omega_bulk * tre;
      for (int a = 0; a < 3; ++a)
        k[d[a]] = (Real(1) - omega) * (k[d[a]] - tr * third)
                + omega * (ke[d[a]] - tre * third) + tr_post * third;
      const int sh[3] = {B::mi(1, 1, 0), B::mi(1, 0, 1), B::mi(0, 1, 1)};
      for (int a = 0; a < 3; ++a)
        k[sh[a]] = (Real(1) - omega) * k[sh[a]] + omega * ke[sh[a]];
    }

    // order >= 3 straight to equilibrium
    for (int p = 0; p < 3; ++p)
      for (int q = 0; q < 3; ++q)
        for (int r = 0; r < 3; ++r)
          if (p + q + r >= 3) k[B::mi(p, q, r)] = ke[B::mi(p, q, r)];

    to_populations(k, u, f);
  }

  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real rho, Real ux, Real uy, Real uz) {
    return Equilibrium::eq(i, rho, ux, uy, uz);
  }
};

}  // namespace lbm
