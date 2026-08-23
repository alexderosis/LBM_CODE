#pragma once
//==============================================================================
//  Fluid collision for MHD: BGK plus the Maxwell stress.
//
//  The momentum equation gains -div(B B - |B|^2/2 I), and the clean way to get
//  it is NOT to compute that divergence and apply it as a body force -- that
//  would need derivatives of B and lose an order. Instead the fluid equilibrium
//  is given the right second moment directly (Dellar 2002):
//
//      sum_i c_a c_b f^eq = rho u_a u_b + (p + |B|^2/2) delta_ab - B_a B_b
//
//  which is achieved by adding, to the hydrodynamic equilibrium,
//
//      df_i = (w_i / 2 cs4) M_ab (c_ia c_ib - cs2 delta_ab),
//      M_ab = (|B|^2/2) delta_ab - B_a B_b.
//
//  That term contributes nothing to mass or momentum -- sum df = 0 and
//  sum c df = 0 -- so it perturbs only the stress, exactly as intended, and it
//  leaves the shifted-storage bookkeeping untouched.
//==============================================================================
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

template <class L, class Eq = SecondOrderEquilibrium<L>, class Store = RawPopulations,
          class Forcing = NoForcing>
struct MhdBGK {
  using Lattice     = L;
  using Equilibrium = Eq;
  using Storage     = Store;
  using ForcingPolicy = Forcing;
  static constexpr const char* name = "MhdBGK";
  static_assert(L::supports_navier_stokes,
                "the MHD fluid operator needs a Navier-Stokes lattice.");

  Real omega = Real(1);
  View1D<Real> Bx, By, Bz;                 // magnetic field, owned by MagneticSolver
  Forcing forcing{};

  static Real omega_from_viscosity(Real nu) {
    return Real(1) / (nu * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real viscosity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
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
    const Real ir = Real(1) / rho;
    m.ux = mx * ir;  m.uy = my * ir;  m.uz = mz * ir;
    // Guo's half-force velocity shift, exactly as in BGK. Omitting it biases
    // the measured velocity by F / (2 rho), which on a forced channel is a
    // systematic offset in the profile rather than a small error.
    forcing.shift_velocity(n, rho, m.ux, m.uy, m.uz);
    return m;
  }

  // The Maxwell-stress addition to f_i^eq at node n.
  KOKKOS_INLINE_FUNCTION
  Real maxwell(int i, Index n) const {
    constexpr Real cs2v = cs2<L, Real>();
    const Real B[3] = {Bx(n), By(n), (L::D == 3) ? Bz(n) : Real(0)};
    const Real b2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
    // M_ab (c_a c_b - cs2 delta_ab) written out, which removes the D x D loop:
    //   M_ab c_a c_b = 0.5 |b|^2 |c|^2 - (c.b)^2,
    //   cs2 M_aa     = cs2 (D/2 - 1) |b|^2  ->  |b|^2 / 6 in 3D, 0 in 2D.
    Real c2 = Real(0), cb = Real(0);
    for (int a = 0; a < L::D; ++a) {
      const Real c = Real(cvel<L>(i, a));
      c2 += c * c;
      cb += c * B[a];
    }
    const Real trace = cs2v * (Real(L::D) * Real(0.5) - Real(1)) * b2;
    const Real acc = Real(0.5) * b2 * c2 - cb * cb - trace;
    return weight<L, Real>(i) * acc / (Real(2) * cs2v * cs2v);
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], const Macro& m, Index n = 0) const {
    for (int i = 0; i < L::Q; ++i) {
      const Real feq = (Store::shifted ? Eq::eq_shifted(i, m.dens, m.ux, m.uy, m.uz)
                                       : Eq::eq(i, m.dens, m.ux, m.uy, m.uz))
                     + maxwell(i, n);
      f[i] += omega * (feq - f[i]);
      if constexpr (Forcing::active)
        f[i] += forcing.template source<L>(n, i, omega, m.ux, m.uy, m.uz);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real rho, Real ux, Real uy, Real uz) {
    if constexpr (Store::shifted) return Eq::eq_shifted(i, rho - Real(1), ux, uy, uz);
    else                          return Eq::eq(i, rho, ux, uy, uz);
  }
};

}  // namespace lbm
