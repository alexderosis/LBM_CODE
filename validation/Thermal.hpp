#pragma once
//==============================================================================
//  Milestone 7 validation: scalar transport, and natural convection.
//
//  1. PURE DIFFUSION      T = T0 + A sin(kx) exp(-D k^2 t)
//     Exact solution of the diffusion equation. Recovers the effective
//     diffusivity from the amplitude decay, on D2Q5 and D3Q7.
//
//  2. ADVECTION-DIFFUSION T = T0 + A sin(k(x - U t)) exp(-D k^2 t)
//     Same, carried by a uniform velocity. Checks the coupling term: an error
//     there shows up as the wave arriving in the wrong place, which the amplitude
//     check alone would not see, so the phase is measured too.
//
//  3. NATURAL CONVECTION in a differentially heated square cavity, against the
//     de Vahl Davis (1983) benchmark. This is the one that exercises the whole
//     module composition at once: two distribution sets on two different
//     lattices, coupled both ways -- buoyancy from the scalar into the fluid,
//     advection from the fluid into the scalar.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/TRT.hpp"
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

using namespace lbm;
namespace { constexpr double PI = 3.14159265358979323846; }

//------------------------------------------------------------------------------
// Cases 1 and 2: a decaying (and optionally advected) sinusoid.
//------------------------------------------------------------------------------
struct Decay { double d_eff, d_err, phase_err, l2; std::size_t steps; };

template <class L>
Decay sinusoid(Index N, Real tau, Real amp, Real U) {
  const Real D = ScalarBGK<L>::diffusivity_from_omega(Real(1) / tau);
  const double k = 2.0 * PI / double(N);
  const double rate = double(D) * k * k;
  const auto steps = std::size_t(0.5 / rate);

  const Index Ny = (L::D == 3) ? 8 : 8;
  const Index Nz = (L::D == 3) ? 8 : 1;
  Domain d(N, Ny, Nz, true, true, true);

  ScalarBGK<L> coll; coll.omega = Real(1) / tau;
  coll.T_ref = Real(1);        // the sinusoid oscillates about T = 1
  ScalarSolver<L, EsotericPull<L>, ScalarBGK<L>> s(d, coll);

  // uniform advecting velocity, supplied exactly as the fluid solver would
  View1D<Real> vx("vx", d.n_padded), vy("vy", d.n_padded), vz("vz", d.n_padded);
  Kokkos::deep_copy(vx, U);
  s.set_velocity(vx, vy, vz);

  const Real kk = Real(k), A = amp;
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    return Real(1) + A * Kokkos::sin(kk * Real(x));
  });

  for (std::size_t t = 0; t < steps; ++t) s.step();
  s.compute_field();
  auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());

  // Project onto sin/cos to get amplitude and phase without assuming either.
  double cs = 0, sn = 0;
  for (Index x = 0; x < N; ++x) {
    double row = 0;
    for (Index z = 0; z < Nz; ++z)
      for (Index y = 0; y < Ny; ++y) row += double(hT(d.id(x, y, z)));
    row /= double(Ny * Nz);
    sn += (row - 1.0) * std::sin(k * x);
    cs += (row - 1.0) * std::cos(k * x);
  }
  sn *= 2.0 / double(N);  cs *= 2.0 / double(N);
  const double got_amp = std::hypot(sn, cs);
  const double got_ph  = std::atan2(cs, sn);

  const double d_eff = -std::log(got_amp / double(amp)) / (k * k * double(steps));
  const double want_ph = -k * double(U) * double(steps);
  double ph_err = got_ph - want_ph;
  while (ph_err >  PI) ph_err -= 2 * PI;
  while (ph_err < -PI) ph_err += 2 * PI;

  // field error against the analytic solution
  const double E = std::exp(-rate * double(steps));
  double num = 0, den = 0;
  for (Index x = 0; x < N; ++x) {
    const double ana = double(amp) * std::sin(k * (double(x) - double(U) * double(steps))) * E;
    double row = 0;
    for (Index z = 0; z < Nz; ++z)
      for (Index y = 0; y < Ny; ++y) row += double(hT(d.id(x, y, z))) - 1.0;
    row /= double(Ny * Nz);
    num += (row - ana) * (row - ana);
    den += ana * ana;
  }
  return {d_eff, std::abs(d_eff / double(D) - 1.0),
          std::abs(ph_err), std::sqrt(num / den), steps};
}

//------------------------------------------------------------------------------
// Case 3: natural convection in a differentially heated square cavity.
//
//   left wall hot (T = +1/2), right wall cold (T = -1/2), top and bottom
//   adiabatic, no-slip everywhere, buoyancy F_y = rho0 g beta (T - T_ref).
//
//   Ra = g beta dT H^3 / (nu D),  Pr = nu / D
//
// Fixing the characteristic velocity u_c = sqrt(g beta dT H) sets the Mach
// number, and then nu = H u_c sqrt(Pr/Ra) follows. That is the standard way to
// pin a Boussinesq case in lattice units: Ra and Pr are what the benchmark
// specifies, u_c is what keeps the scheme in its low-Mach regime.
//------------------------------------------------------------------------------
struct Cavity { double nu_avg, umax, vmax; std::size_t steps; };

template <class FluidColl, class Setup>
Cavity cavity(Index N, double Ra, double Pr, Real uc, Setup setup) {
  using FL = D2Q9;
  using SL = D2Q5;
  const Index H = N - 2;                       // fluid nodes across the cavity
  const Real nu = Real(double(H) * double(uc) * std::sqrt(Pr / Ra));
  const Real D  = Real(double(nu) / Pr);
  const Real gb = Real(double(uc) * double(uc) / double(H));   // g*beta, with dT = 1

  Domain d(N, N, 1, false, false, true);

  ScalarBGK<SL> scoll; scoll.omega = ScalarBGK<SL>::omega_from_diffusivity(D);
  scoll.T_ref = Real(0);       // the cavity is symmetric about T = 0
  ScalarSolver<SL, EsotericPull<SL>, ScalarBGK<SL>> th(d, scoll);
  th.set_geometry([&](Index x, Index y, Index) -> ScalarCell {
    if (x == 0 || x == N - 1) return ScalarDirichlet;
    if (y == 0 || y == N - 1) return ScalarAdiabatic;
    return ScalarBulk;
  });
  th.set_wall_values([&](Index x, Index, Index) -> Real {
    return (x == 0) ? Real(0.5) : Real(-0.5);
  });
  th.initialize(Real(0));
  th.compute_field();

  BoussinesqGuo force;
  force.T = th.temperature();
  force.gx = Real(0); force.gy = Real(1); force.gz = Real(0);
  force.rho0 = Real(1); force.beta = gb; force.T0 = Real(0);

  FluidColl fcoll;
  setup(fcoll, BGK<FL>::omega_from_viscosity(nu));
  fcoll.forcing = force;

  FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fcoll);
  fl.set_geometry([&](Index x, Index y, Index) -> CellType {
    return (x == 0 || x == N - 1 || y == 0 || y == N - 1) ? Solid : Fluid;
  });
  fl.initialize(Real(1));
  th.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // Nusselt at the hot wall. The wall plane sits midway between the Dirichlet
  // cell at x=0 and the first fluid cell at x=1, i.e. at x = 0.5, so the
  // one-sided derivative uses unequal spacing (0.5 and 1.5).
  auto nusselt = [&]() {
    th.compute_field();
    auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    double acc = 0;
    for (Index y = 1; y <= H; ++y) {
      const double Tw = 0.5;
      const double T1 = double(hT(d.id(1, y)));
      const double T2 = double(hT(d.id(2, y)));
      const double dTdx = -(2.0 / 0.75) * Tw + 3.0 * T1 - (1.0 / 3.0) * T2;
      acc += -dTdx * double(H);                // dT = 1
    }
    return acc / double(H);
  };

  const std::size_t probe = 500, cap = 2000000;
  double prev = 0, nu_avg = 0;
  std::size_t taken = 0;
  for (std::size_t t = 0; t < cap; t += probe) {
    for (std::size_t k = 0; k < probe; ++k) { fl.step(true); th.step(); }
    taken += probe;
    nu_avg = nusselt();
    if (t > 0 && std::abs(nu_avg - prev) < 1e-9 * std::abs(nu_avg)) break;
    prev = nu_avg;
  }

  fl.compute_macroscopic();
  auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  // benchmark maxima, nondimensionalised by D/H
  double umax = 0, vmax = 0;
  const double scale = double(H) / double(D);
  for (Index y = 1; y <= H; ++y) umax = std::max(umax, double(hx(d.id(N / 2, y))) * scale);
  for (Index x = 1; x <= H; ++x) vmax = std::max(vmax, double(hy(d.id(x, N / 2))) * scale);
  return {nu_avg, umax, vmax, taken};
}

