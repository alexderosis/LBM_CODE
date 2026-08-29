#pragma once
//==============================================================================
//  Fluid collision for two-phase flow: BGK plus the capillary stress.
//
//  Surface tension is NOT applied as a body force. The fluid equilibrium is
//  given the right second moment directly, exactly as MhdBGK does for the
//  Maxwell stress:
//
//      sum_i c_a c_b f^eq = rho u_a u_b + (p - kappa |G|^2 / 2) delta_ab
//                           + kappa G_a G_b,          G = grad phi,
//
//  achieved by adding, to the hydrodynamic equilibrium,
//
//      df_i = (w_i / 2 cs4) M_ab (c_ia c_ib - cs2 delta_ab),
//      M_ab = kappa ( G_a G_b - |G|^2/2 delta_ab ).
//
//  THIS IS THE MAXWELL STRESS WITH B -> sqrt(kappa) grad phi AND THE OPPOSITE
//  SIGN. Same tensor structure, same three invariants -- sum_i df_i = 0 and
//  sum_i c_i df_i = 0, so it perturbs only the stress and leaves mass, momentum
//  and the shifted-storage bookkeeping untouched -- but the sign is reversed,
//  and that is physics rather than convention. The Maxwell stress is a tension
//  ALONG B, so it pulls field lines together; surface tension is a tension along
//  the INTERFACE, which is perpendicular to grad phi. Relative to the vector
//  that defines the tensor, the two therefore differ by a sign.
//
//  It is not a subtle error to make. With the Maxwell sign the mechanical
//  surface tension integral over the profile comes out negative and the droplet
//  develops a pressure DEFICIT: measured -2.5e-04 where +8.3e-04 was expected.
//
//  `capillary()` below is thus -kappa times `maxwell()`. If a third stress of
//  this shape ever appears the two should be factored into one helper, but
//  duplicating twelve lines is cheaper than disturbing a validated MHD path.
//
//  WHY NOT THE POTENTIAL FORM. F = mu grad phi with the chemical potential
//  mu = 4 beta phi (phi-1) (phi-1/2) - kappa lap(phi) is algebraically
//  equivalent, and needs a LAPLACIAN of phi. The stress form needs only the
//  gradient, which PhaseFieldSolver has to compute anyway for the interface
//  normal -- so it is one field cheaper and one derivative order safer, the
//  same argument Dellar's makes for the Lorentz force.
//
//  SURFACE TENSION. Integrating kappa (dphi/dx)^2 across the tanh profile of
//  width W gives
//
//      sigma = 2 kappa / (3 W)      <=>      kappa = 3 sigma W / 2,
//
//  for phi in [0, 1]. That is what `kappa_from_sigma` returns. Note the
//  double-well strength beta never appears: the conservative Allen-Cahn
//  equation maintains the profile geometrically, from W and the normal, so the
//  free energy is implicit. (In the potential form it would be beta = 12 sigma / W,
//  the value consistent with the same sigma and W.)
//
//  SCOPE. This is the MATCHED-DENSITY operator: rho is a single field near 1 and
//  phi enters only through the stress and, optionally, through the relaxation
//  rate. That isolates surface tension completely and keeps shifted storage
//  meaningful, which a density ratio would not -- see the note on
//  ShiftedPopulations in the header of PhaseFieldSolver.hpp.
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
struct MultiphaseBGK {
  using Lattice     = L;
  using Equilibrium = Eq;
  using Storage     = Store;
  using ForcingPolicy = Forcing;
  static constexpr const char* name = "MultiphaseBGK";
  static_assert(L::supports_navier_stokes,
                "the multiphase fluid operator needs a Navier-Stokes lattice.");

  Real omega = Real(1);
  Real kappa = Real(0);                  // surface-tension coefficient
  View1D<Real> Gx, Gy, Gz;               // grad phi, owned by PhaseFieldSolver

  //----------------------------------------------------------------------------
  // OPTIONAL per-node relaxation rate, for a viscosity contrast between the
  // phases. Left empty, every node uses the scalar `omega` above.
  //
  // This is here because it costs almost nothing: the collision interface
  // already carries a node index -- the addition the thermal module forced, so
  // that a body force could vary in space -- and a relaxation rate that varies
  // in space is the same kind of thing. Not exercised by validation/laplace,
  // which is deliberately at a matched viscosity.
  //----------------------------------------------------------------------------
  View1D<Real> omega_n;

  Forcing forcing{};

  static Real omega_from_viscosity(Real nu) {
    return Real(1) / (nu * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real viscosity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
  }
  // sigma = 2 kappa / (3 W), for phi in [0, 1] on the tanh profile.
  static Real kappa_from_sigma(Real sigma, Real width) {
    return Real(1.5) * sigma * width;
  }

  KOKKOS_INLINE_FUNCTION static Real density(const Macro& m) {
    if constexpr (Store::shifted) return Real(1) + m.dens;
    else                          return m.dens;
  }
  KOKKOS_INLINE_FUNCTION Real rate(Index n) const {
    return omega_n.data() ? omega_n(n) : omega;
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
  // The capillary addition to f_i^eq at node n. Written out rather than looped
  // over a and b, as in MhdBGK::maxwell:
  //     M_ab c_a c_b = kappa [ (c.G)^2 - |G|^2 |c|^2 / 2 ],
  //     cs2 M_aa     = kappa cs2 (1 - D/2) |G|^2.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  Real capillary(int i, Index n) const {
    constexpr Real cs2v = cs2<L, Real>();
    const Real G[3] = {Gx(n), Gy(n), (L::D == 3) ? Gz(n) : Real(0)};
    const Real g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    Real c2 = Real(0), cg = Real(0);
    for (int a = 0; a < L::D; ++a) {
      const Real c = Real(cvel<L>(i, a));
      c2 += c * c;
      cg += c * G[a];
    }
    const Real trace = cs2v * (Real(1) - Real(L::D) * Real(0.5)) * g2;
    const Real acc = cg * cg - Real(0.5) * g2 * c2 - trace;
    return kappa * weight<L, Real>(i) * acc / (Real(2) * cs2v * cs2v);
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], const Macro& m, Index n = 0) const {
    const Real w = rate(n);
    for (int i = 0; i < L::Q; ++i) {
      const Real feq = (Store::shifted ? Eq::eq_shifted(i, m.dens, m.ux, m.uy, m.uz)
                                       : Eq::eq(i, m.dens, m.ux, m.uy, m.uz))
                     + capillary(i, n);
      f[i] += w * (feq - f[i]);
      if constexpr (Forcing::active)
        f[i] += forcing.template source<L>(n, i, w, m.ux, m.uy, m.uz);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real rho, Real ux, Real uy, Real uz) {
    if constexpr (Store::shifted) return Eq::eq_shifted(i, rho - Real(1), ux, uy, uz);
    else                          return Eq::eq(i, rho, ux, uy, uz);
  }
};

}  // namespace lbm
