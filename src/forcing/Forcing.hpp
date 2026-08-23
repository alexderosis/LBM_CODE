#pragma once
//==============================================================================
//  Body-force schemes.
//
//  Forcing is NOT orthogonal to collision: for the central-moment operator the
//  force enters at the moment level, not as a post-collision addend. So the
//  forcing policy is a member of the collision policy and the collision operator
//  decides how to consume it -- never applied as a separate pass.
//
//  Every policy answers `at(n, F)` for the force at node n. Uniform forces
//  ignore n; a Boussinesq buoyancy reads the temperature field there. That node
//  index is why the collision interface carries one: a body force that varies in
//  space is the normal case, not the exception -- buoyancy needs it now and the
//  Lorentz force will need it for MHD.
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Shared Guo source term, without the (1 - omega/2) prefactor. TRT and the
// moment operators relax different modes at different rates and each applies its
// own prefactor per mode; only BGK can fold a single one in.
//------------------------------------------------------------------------------
template <class L>
KOKKOS_INLINE_FUNCTION
Real guo_source_raw(int i, const Real F[3], Real ux, Real uy, Real uz) {
  constexpr Real ics2 = inv_cs2<L, Real>();
  const Real cx = Real(cvel<L>(i, 0));
  const Real cy = Real(cvel<L>(i, 1));
  const Real cz = Real(cvel<L>(i, 2));
  const Real cu = cx * ux + cy * uy + cz * uz;
  const Real bx = (cx - ux) * ics2 + cu * ics2 * ics2 * cx;
  const Real by = (cy - uy) * ics2 + cu * ics2 * ics2 * cy;
  const Real bz = (cz - uz) * ics2 + cu * ics2 * ics2 * cz;
  return weight<L, Real>(i) * (bx * F[0] + by * F[1] + bz * F[2]);
}

//------------------------------------------------------------------------------
struct NoForcing {
  static constexpr const char* name = "None";
  static constexpr bool active = false;

  KOKKOS_INLINE_FUNCTION void at(Index, Real F[3]) const { F[0] = F[1] = F[2] = Real(0); }
  KOKKOS_INLINE_FUNCTION void shift_velocity(Index, Real, Real&, Real&, Real&) const {}
  template <class L>
  KOKKOS_INLINE_FUNCTION Real source_raw(Index, int, Real, Real, Real) const { return Real(0); }
  template <class L>
  KOKKOS_INLINE_FUNCTION Real source(Index, int, Real, Real, Real, Real) const { return Real(0); }
};

//------------------------------------------------------------------------------
// Guo et al. (2002), uniform force.
//   u   = ( sum_i c_i f_i + F/2 ) / rho
//   S_i = (1 - omega/2) w_i [ (c_i - u)/cs2 + (c_i.u) c_i / cs4 ] . F
//------------------------------------------------------------------------------
struct Guo {
  static constexpr const char* name = "Guo";
  static constexpr bool active = true;

  Real fx = Real(0), fy = Real(0), fz = Real(0);

  KOKKOS_INLINE_FUNCTION void at(Index, Real F[3]) const { F[0] = fx; F[1] = fy; F[2] = fz; }

  KOKKOS_INLINE_FUNCTION
  void shift_velocity(Index, Real rho, Real& ux, Real& uy, Real& uz) const {
    const Real h = Real(0.5) / rho;
    ux += h * fx;  uy += h * fy;  uz += h * fz;
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source_raw(Index n, int i, Real ux, Real uy, Real uz) const {
    Real F[3]; at(n, F);
    return guo_source_raw<L>(i, F, ux, uy, uz);
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source(Index n, int i, Real omega, Real ux, Real uy, Real uz) const {
    return (Real(1) - Real(0.5) * omega) * source_raw<L>(n, i, ux, uy, uz);
  }
};

//------------------------------------------------------------------------------
// Boussinesq buoyancy, delivered through the same Guo machinery.
//
//   F(n) = rho0 * g * beta * (T(n) - T0)
//
// The density is held constant everywhere except in the buoyancy term itself --
// that is the Boussinesq approximation, and it is why the thermal field couples
// back into the flow through a force rather than through the equation of state.
//------------------------------------------------------------------------------
struct BoussinesqGuo {
  static constexpr const char* name = "Boussinesq";
  static constexpr bool active = true;

  View1D<Real> T;                                   // temperature field
  Real gx = Real(0), gy = Real(-1), gz = Real(0);   // gravity direction
  Real rho0 = Real(1), beta = Real(1), T0 = Real(0);
  Real fx = Real(0), fy = Real(0), fz = Real(0);    // optional uniform part

  KOKKOS_INLINE_FUNCTION
  void at(Index n, Real F[3]) const {
    const Real b = rho0 * beta * (T(n) - T0);
    F[0] = fx + gx * b;  F[1] = fy + gy * b;  F[2] = fz + gz * b;
  }
  KOKKOS_INLINE_FUNCTION
  void shift_velocity(Index n, Real rho, Real& ux, Real& uy, Real& uz) const {
    Real F[3]; at(n, F);
    const Real h = Real(0.5) / rho;
    ux += h * F[0];  uy += h * F[1];  uz += h * F[2];
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source_raw(Index n, int i, Real ux, Real uy, Real uz) const {
    Real F[3]; at(n, F);
    return guo_source_raw<L>(i, F, ux, uy, uz);
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source(Index n, int i, Real omega, Real ux, Real uy, Real uz) const {
    return (Real(1) - Real(0.5) * omega) * source_raw<L>(n, i, ux, uy, uz);
  }
};

}  // namespace lbm
