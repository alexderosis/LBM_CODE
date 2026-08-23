#pragma once
//==============================================================================
//  TRT -- two relaxation times (Ginzburg, d'Humieres).
//
//  Split every population into its symmetric and antisymmetric parts about the
//  opposite direction and relax them at different rates:
//
//      f_i^+ = (f_i + f_opp(i))/2      relaxed at omega_plus   -> viscosity
//      f_i^- = (f_i - f_opp(i))/2      relaxed at omega_minus  -> free
//
//  Five lines more than BGK, and it buys the magic parameter
//
//      Lambda = (1/omega_plus - 1/2)(1/omega_minus - 1/2)
//
//  At Lambda = 3/16 the halfway bounce-back wall sits exactly midway between the
//  fluid and solid nodes at ANY viscosity -- BGK only manages that at the single
//  tau = 0.5 + sqrt(3)/4, because it is TRT with omega_minus == omega_plus.
//  That makes TRT the honest baseline for judging whether MRT or the
//  central-moment operator is actually buying anything.
//
//  The Esoteric Pull direction ordering pays off here: opp(i) = i+1 for odd i,
//  so the pair loop is `for (i = 1; i < Q; i += 2)` with no lookup table.
//==============================================================================
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

template <class L, class Eq = SecondOrderEquilibrium<L>, class Forcing = NoForcing,
          class Store = RawPopulations>
struct TRT {
  using Lattice     = L;
  using Equilibrium = Eq;
  using Storage     = Store;
  static constexpr const char* name = "TRT";
  static_assert(L::supports_navier_stokes,
                "TRT as a Navier-Stokes operator needs a lattice with isotropic "
                "4th-order moments.");

  static constexpr Real magic_3_16 = Real(3) / Real(16);

  Real omega_p = Real(1);      // symmetric: sets the viscosity
  Real omega_m = Real(1);      // antisymmetric: free, set via the magic parameter
  Forcing forcing{};

  static Real omega_from_viscosity(Real nu) {
    return Real(1) / (nu * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real viscosity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
  }
  // omega_minus that realises a given magic parameter for this omega_plus.
  static Real omega_minus_for(Real omega_plus, Real lambda) {
    return Real(1) / (lambda / (Real(1) / omega_plus - Real(0.5)) + Real(0.5));
  }
  static Real magic_parameter(Real omega_plus, Real omega_minus) {
    return (Real(1) / omega_plus - Real(0.5)) * (Real(1) / omega_minus - Real(0.5));
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

  KOKKOS_INLINE_FUNCTION
  Real eq_of(int i, const Macro& m) const {
    if constexpr (Store::shifted) return Eq::eq_shifted(i, m.dens, m.ux, m.uy, m.uz);
    else                          return Eq::eq(i, m.dens, m.ux, m.uy, m.uz);
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], const Macro& m, Index n = 0) const {
    // The equilibrium (and source) are evaluated in a flat loop first: the pair
    // loop below reads f[i] and f[i+1] together and does not vectorise, so
    // leaving the equilibrium evaluation inside it costs more than the extra
    // array does.
    Real e[L::Q];
    for (int i = 0; i < L::Q; ++i) e[i] = eq_of(i, m);
    Real sr[L::Q];
    if constexpr (Forcing::active)
      for (int i = 0; i < L::Q; ++i)
        sr[i] = forcing.template source_raw<L>(n, i, m.ux, m.uy, m.uz);

    const Real cp = Real(1) - Real(0.5) * omega_p;
    const Real cm = Real(1) - Real(0.5) * omega_m;

    // rest population: opp(0) == 0, so it is purely symmetric
    f[0] += omega_p * (e[0] - f[0]);
    if constexpr (Forcing::active) f[0] += cp * sr[0];

    for (int i = 1; i < L::Q; i += 2) {
      const int j = i + 1;                       // opp(i), by the ordering contract
      const Real fp = Real(0.5) * (f[i] + f[j]);
      const Real fm = Real(0.5) * (f[i] - f[j]);
      const Real ep = Real(0.5) * (e[i] + e[j]);
      const Real em = Real(0.5) * (e[i] - e[j]);
      Real dp = omega_p * (ep - fp);
      Real dm = omega_m * (em - fm);
      if constexpr (Forcing::active) {
        dp += cp * Real(0.5) * (sr[i] + sr[j]);
        dm += cm * Real(0.5) * (sr[i] - sr[j]);
      }
      f[i] += dp + dm;
      f[j] += dp - dm;
    }
  }

  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real rho, Real ux, Real uy, Real uz) {
    if constexpr (Store::shifted) return Eq::eq_shifted(i, rho - Real(1), ux, uy, uz);
    else                          return Eq::eq(i, rho, ux, uy, uz);
  }
};

}  // namespace lbm
