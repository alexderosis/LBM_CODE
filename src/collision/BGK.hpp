#pragma once
//==============================================================================
//  BGK (single relaxation time).
//
//  Collision policy contract, shared by BGK / TRT / MRT / CentralMoments:
//
//      Macro m = macroscopic(f);          // reduce the incoming populations
//      collide(f, m);                     // relax in place
//
//  The policy owns its equilibrium, its forcing scheme and its storage
//  convention, and knows nothing about streaming or geometry.
//
//  `Macro::dens` is rho for raw storage and rho-1 for shifted storage. It is
//  deliberately NOT normalised to rho in the shifted case: forming rho and then
//  subtracting 1 again would throw away exactly the precision the shift was
//  introduced to protect.
//==============================================================================
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

struct Macro {
  Real dens;              // rho (raw) or rho - 1 (shifted)
  Real ux, uy, uz;
};

template <class L, class Eq = SecondOrderEquilibrium<L>, class Forcing = NoForcing,
          class Store = RawPopulations>
struct BGK {
  using Lattice     = L;
  using Equilibrium = Eq;
  using Storage     = Store;
  using ForcingPolicy = Forcing;
  static constexpr const char* name = "BGK";
  static_assert(L::supports_navier_stokes,
                "BGK as a Navier-Stokes operator needs a lattice with isotropic "
                "4th-order moments; D2Q5/D3Q7 are advection-diffusion lattices.");

  Real omega = Real(1);
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

  //----------------------------------------------------------------------------
  // Note sum_i c_i g_i == sum_i c_i f_i exactly, because sum_i c_i w_i = 0. The
  // momentum expression is therefore identical for both storage conventions --
  // only its conditioning differs, and that is the whole point of the shift.
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
    const Real rho = density(m);
    const Real ir  = Real(1) / rho;
    m.ux = mx * ir;  m.uy = my * ir;  m.uz = mz * ir;
    forcing.shift_velocity(n, rho, m.ux, m.uy, m.uz);
    return m;
  }

  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], const Macro& m, Index n = 0) const {
    for (int i = 0; i < L::Q; ++i) {
      const Real feq = Store::shifted
                     ? Eq::eq_shifted(i, m.dens, m.ux, m.uy, m.uz)
                     : Eq::eq        (i, m.dens, m.ux, m.uy, m.uz);
      f[i] += omega * (feq - f[i]);
      if constexpr (Forcing::active) {
        f[i] += forcing.template source<L>(n, i, omega, m.ux, m.uy, m.uz);
      }
    }
  }

  // Value to seed population i with at initialisation, in the stored variable.
  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real rho, Real ux, Real uy, Real uz) {
    if constexpr (Store::shifted) return Eq::eq_shifted(i, rho - Real(1), ux, uy, uz);
    else                          return Eq::eq(i, rho, ux, uy, uz);
  }
};

}  // namespace lbm
