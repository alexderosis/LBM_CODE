#pragma once
//==============================================================================
//  Milestone 3 validation: decaying flows with exact Navier-Stokes solutions.
//
//  Both cases are periodic, so no boundary condition is involved: what is being
//  measured is the bulk collision operator and the streaming, in isolation from
//  the wall treatment that poiseuille.cpp covers.
//
//  1. TAYLOR-GREEN (nonlinear, exact solution of incompressible NS)
//         u_x = -U cos(kx) sin(ky) E(t)
//         u_y =  U sin(kx) cos(ky) E(t)
//         p   = -(rho0 U^2 / 4) [cos(2kx) + cos(2ky)] E(t)^2
//         E(t) = exp(-2 nu k^2 t),   k = 2 pi / L
//     The field is z-independent, so it is an exact solution in 3D as well and
//     runs unchanged on D3Q19/D3Q27. Nonlinear, so it exercises the parts of the
//     collision operator a linear test cannot reach.
//
//  2. ABC / BELTRAMI FLOW (nonlinear, exact, and genuinely three-dimensional)
//         u = U ( sin kz + cos ky,  sin kx + cos kz,  sin ky + cos kx )
//     curl u = k u, so u x (curl u) = 0 and the nonlinear term collapses to
//     grad(|u|^2/2), which the pressure absorbs: u(t) = u_0 exp(-nu k^2 t) with
//     p = -|u|^2/2 is an exact solution. Unlike Taylor-Green above -- whose
//     field is z-independent and therefore still a 2D problem no matter which
//     lattice runs it -- every velocity component here varies in every
//     direction. This is the nonlinear 3D convergence test.
//
//  3. DIAGONAL SHEAR WAVE (linear, exact, and deliberately off-axis)
//         u = U e sin(K.x) exp(-nu |K|^2 t),  K = (2pi/L)(1,1,1),  e = (1,-1,0)/sqrt2
//     e.K = 0 makes both (u.grad)u and div u vanish identically, so this is an
//     exact solution for any K. Because K is along the body diagonal it is
//     sensitive to lattice anisotropy in a way an axis-aligned wave is not.
//
//  The headline number is the EFFECTIVE VISCOSITY recovered from the decay rate.
//  For BGK nu = cs^2 (1/omega - 1/2) exactly, so any error here is a defect in
//  the implementation rather than in the model, which makes it a sharp check --
//  and the baseline against which MRT and the central-moment operator will be
//  judged at Milestone 4.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/TRT.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "memory/TwoLattice.hpp"
#include "solver/FluidSolver.hpp"

#include <array>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;

namespace {
constexpr double PI = 3.14159265358979323846;

struct Result {

  double nu_eff;      // viscosity recovered from the decay rate
  double nu_err;      // relative error against the input viscosity
  double l2_u;        // relative L2 error of the velocity field at t_end
  double mass_drift;
  std::size_t steps;
};

// L2 norm of the velocity field over the fluid cells.
template <class Solver>
double velocity_norm(Solver& s) {
  const Domain& d = s.domain();
  s.compute_macroscopic();
  auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hz = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
  double sum = 0;
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) {
        const Index n = d.id(x, y, z);
        sum += double(hx(n)) * double(hx(n)) + double(hy(n)) * double(hy(n)) +
               double(hz(n)) * double(hz(n));
      }
  return std::sqrt(sum);
}

// March to t_end, recover the effective viscosity from the decay of ||u||, and
// measure the field error against the analytic solution.
//   decay_coef: the velocity decays as exp(-nu * decay_coef * t)
//   ana(x,y,z) -> the analytic velocity at t_end, already including the decay
template <class Solver, class Ana>
Result run_and_measure(Solver& s, std::size_t steps, double decay_coef,
                       double nu, Ana ana) {
  const Domain& d = s.domain();
  const double m0 = double(s.total_mass());
  const double n0 = velocity_norm(s);
  for (std::size_t t = 0; t < steps; ++t) s.step();
  const double n1 = velocity_norm(s);
  const double m1 = double(s.total_mass());
  const double nu_eff = -std::log(n1 / n0) / (decay_coef * double(steps));

  s.compute_macroscopic();
  auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hz = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
  double num = 0, den = 0;
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) {
        const Index n = d.id(x, y, z);
        const auto a = ana(int(x), int(y), int(z));
        const double e0 = double(hx(n)) - a[0];
        const double e1 = double(hy(n)) - a[1];
        const double e2 = double(hz(n)) - a[2];
        num += e0 * e0 + e1 * e1 + e2 * e2;
        den += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
      }
  return {nu_eff, std::abs(nu_eff / nu - 1.0), std::sqrt(num / den),
          std::abs(m1 - m0) / m0, steps};
}
}  // namespace

//------------------------------------------------------------------------------
// Case 1: Taylor-Green. The field is z-independent, so this is the SAME physical
// problem on every lattice -- useful as a cross-lattice consistency check, but
// not a 3D test. Use the ABC flow below for that.
//------------------------------------------------------------------------------
template <class Coll, class L, template <class> class Streaming, class Setup>
Result taylor_green(Index N, Index Nz, Real tau, Real U, double frac, Setup setup) {
  const Real nu = (tau - Real(0.5)) * cs2<L, Real>();
  const double k = 2.0 * PI / double(N);
  const double coef = 2.0 * k * k;
  const auto steps = std::size_t(frac / (double(nu) * coef));

  Domain d(N, N, Nz, true, true, true);
  Coll coll;
  setup(coll, Real(1) / tau);
  FluidSolver<L, Streaming<L>, Coll> s(d, coll);

  const Real Uk = U, kk = Real(k), ics2 = inv_cs2<L, Real>();
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    const Real X = kk * Real(x), Y = kk * Real(y);
    FlowState st;
    st.rho = Real(1) - Real(0.25) * Uk * Uk *
             (Kokkos::cos(Real(2) * X) + Kokkos::cos(Real(2) * Y)) * ics2;
    st.ux = -Uk * Kokkos::cos(X) * Kokkos::sin(Y);
    st.uy =  Uk * Kokkos::sin(X) * Kokkos::cos(Y);
    st.uz =  Real(0);
    return st;
  });

  const double E = std::exp(-double(nu) * coef * double(steps));
  const double u0 = double(U);
  return run_and_measure(s, steps, coef, double(nu),
    [k, E, u0](int x, int y, int z) -> std::array<double, 3> {
      (void)z;
      return {-u0 * std::cos(k * x) * std::sin(k * y) * E,
               u0 * std::sin(k * x) * std::cos(k * y) * E, 0.0};
    });
}

//------------------------------------------------------------------------------
// Case 2: ABC / Beltrami flow -- nonlinear and genuinely 3D.
//------------------------------------------------------------------------------
template <class Coll, class L, template <class> class Streaming, class Setup>
Result abc_flow(Index N, Real tau, Real U, double frac, Setup setup) {
  static_assert(L::D == 3, "the ABC flow is a 3D case");
  const Real nu = (tau - Real(0.5)) * cs2<L, Real>();
  const double k = 2.0 * PI / double(N);
  const double coef = k * k;                        // curl u = k u  =>  lap u = -k^2 u
  const auto steps = std::size_t(frac / (double(nu) * coef));

  Domain d(N, N, N, true, true, true);
  Coll coll;
  setup(coll, Real(1) / tau);
  FluidSolver<L, Streaming<L>, Coll> s(d, coll);

  const Real Uk = U, kk = Real(k), ics2 = inv_cs2<L, Real>();
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    const Real X = kk * Real(x), Y = kk * Real(y), Z = kk * Real(z);
    FlowState st;
    st.ux = Uk * (Kokkos::sin(Z) + Kokkos::cos(Y));
    st.uy = Uk * (Kokkos::sin(X) + Kokkos::cos(Z));
    st.uz = Uk * (Kokkos::sin(Y) + Kokkos::cos(X));
    // the nonlinear term is grad(|u|^2/2), so the balancing pressure is -|u|^2/2
    const Real u2 = st.ux * st.ux + st.uy * st.uy + st.uz * st.uz;
    st.rho = Real(1) - Real(0.5) * u2 * ics2;
    return st;
  });

  const double E = std::exp(-double(nu) * coef * double(steps));
  const double u0 = double(U);
  return run_and_measure(s, steps, coef, double(nu),
    [k, E, u0](int x, int y, int z) -> std::array<double, 3> {
      return {u0 * (std::sin(k * z) + std::cos(k * y)) * E,
              u0 * (std::sin(k * x) + std::cos(k * z)) * E,
              u0 * (std::sin(k * y) + std::cos(k * x)) * E};
    });
}

//------------------------------------------------------------------------------
// Case 3: diagonal shear wave -- linear, off-axis, probes lattice isotropy.
//------------------------------------------------------------------------------
template <class Coll, class L, template <class> class Streaming, class Setup>
Result shear_wave(Index N, Real tau, Real U, double frac, Setup setup) {
  static_assert(L::D == 3, "the diagonal shear wave is a 3D case");
  const Real nu = (tau - Real(0.5)) * cs2<L, Real>();
  const double k = 2.0 * PI / double(N);
  const double coef = 3.0 * k * k;                  // |K|^2 for K = k(1,1,1)
  const auto steps = std::size_t(frac / (double(nu) * coef));

  Domain d(N, N, N, true, true, true);
  Coll coll;
  setup(coll, Real(1) / tau);
  FluidSolver<L, Streaming<L>, Coll> s(d, coll);

  const Real Uk = U, kk = Real(k);
  const Real inv_sqrt2 = Real(0.70710678118654752440);
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    const Real a = Uk * Kokkos::sin(kk * (Real(x) + Real(y) + Real(z))) * inv_sqrt2;
    FlowState st;
    st.rho = Real(1); st.ux = a; st.uy = -a; st.uz = Real(0);
    return st;
  });

  const double E = std::exp(-double(nu) * coef * double(steps));
  const double u0 = double(U) / std::sqrt(2.0);
  return run_and_measure(s, steps, coef, double(nu),
    [k, E, u0](int x, int y, int z) -> std::array<double, 3> {
      const double a = u0 * std::sin(k * (x + y + z)) * E;
      return {a, -a, 0.0};
    });
}


//------------------------------------------------------------------------------
// Operator zoo. Every operator uses Esoteric Pull + shifted storage.
//------------------------------------------------------------------------------
namespace ops {
auto plain = [](auto& c, Real w) { c.omega = w; };
auto trt   = [](auto& c, Real w) {
  using T = std::decay_t<decltype(c)>;
  c.omega_p = w; c.omega_m = T::omega_minus_for(w, T::magic_3_16);
};
template <class L> using Bgk = BGK<L, SecondOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;
template <class L> using Trt = TRT<L, SecondOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;
template <class L> using Mrt = MomentCollision<L, NoForcing, ShiftedPopulations, false>;
template <class L> using Cm  = MomentCollision<L, NoForcing, ShiftedPopulations, true>;
}  // namespace ops

