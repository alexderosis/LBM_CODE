//==============================================================================
//  D3Q27 forcing: plane channels on every axis, and a fully three-dimensional
//  forced flow with an exact solution.
//
//  The force central moments on D3Q27 are
//
//      k = cs^(2m) F_a   for exponent 1 on axis a and 2 on m other axes,
//      k = 0             otherwise,
//
//  so there are three first-order entries, six at cs^2 F_a, and three at
//  cs^4 F_a. The cs^4 group has no D2Q9 analogue. Two tests here:
//
//  PART 1 -- PLANE CHANNEL, ALL THREE AXES. Halfway bounce-back walls normal to
//  one axis, uniform force along another, periodic in the third. Running the
//  three cyclic combinations puts every axis in the wall role and every axis in
//  the force role. With H fluid nodes the no-slip planes are at 0.5 and H + 0.5,
//
//      u(s) = (G / 2 rho nu) (s - 0.5)(H + 0.5 - s),
//
//  and at the central-moment magic point tau = 7/8 this is exact to round-off.
//  D3Q27 is invariant under the cyclic permutation of axes, so the three cases
//  must also agree with each other to round-off -- a check no single orientation
//  can make.
//
//  PART 2 -- FULLY THREE-DIMENSIONAL. The channel is still quasi-one
//  dimensional: u varies along one axis only, so it never excites a moment with
//  two squared axes at once and leaves the cs^4 group barely exercised. This
//  case fixes that. Periodic cube, no walls, body force
//
//      Fx = F0 sin(k y) sin(k z),      k = 2 pi / N,
//
//  whose steady solution is EXACT for the full Navier-Stokes equations:
//
//      ux = F0 sin(k y) sin(k z) / (2 rho nu k^2),   uy = uz = 0.
//
//  It is exact because ux does not depend on x, so (u.grad)u vanishes
//  identically and the pressure stays uniform -- the nonlinear term is not
//  small, it is zero. The velocity varies in TWO directions at once, which is
//  what makes it three-dimensional in the sense that matters here, and there are
//  no walls, so nothing about the wall treatment can contaminate the measurement.
//
//  The discrete Laplacian of a sinusoid is -(2 - 2 cos k) rather than -k^2, so
//  the amplitude carries an O(k^2) = O(1/N^2) error and the expected order is 2.
//==============================================================================
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "solver/FluidSolver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;
using L = D3Q27;
using CM = CentralMoments<L, Guo, ShiftedPopulations>;

static constexpr double PI = 3.14159265358979323846;

//------------------------------------------------------------------------------
// Fx = F0 sin(k y) sin(k z), zero elsewhere. Same interface as Guo; kept local
// to this test rather than added to the library, since nothing else needs it.
//------------------------------------------------------------------------------
struct SinForce {
  static constexpr const char* name = "SinForce";
  static constexpr bool active = true;
  Real F0 = Real(0), k = Real(0);
  Domain dom{1, 1, 1, true, true, true};

  KOKKOS_INLINE_FUNCTION void at(Index n, Real F[3]) const {
    Index px, py, pz; dom.coords(n, px, py, pz);
    const Real y = Real(py - dom.hy), z = Real(pz - dom.hz);
    F[0] = F0 * Kokkos::sin(k * y) * Kokkos::sin(k * z);
    F[1] = Real(0); F[2] = Real(0);
  }
  KOKKOS_INLINE_FUNCTION
  void shift_velocity(Index n, Real rho, Real& ux, Real& uy, Real& uz) const {
    Real F[3]; at(n, F);
    const Real h = Real(0.5) / rho;
    ux += h * F[0]; uy += h * F[1]; uz += h * F[2];
  }
  template <class LL>
  KOKKOS_INLINE_FUNCTION Real source_raw(Index n, int i, Real ux, Real uy, Real uz) const {
    Real F[3]; at(n, F);
    return guo_source_raw<LL>(i, F, ux, uy, uz);
  }
  template <class LL>
  KOKKOS_INLINE_FUNCTION Real source(Index n, int i, Real om, Real ux, Real uy, Real uz) const {
    return (Real(1) - Real(0.5) * om) * source_raw<LL>(n, i, ux, uy, uz);
  }
};
using CMS = CentralMoments<L, SinForce, ShiftedPopulations>;

//------------------------------------------------------------------------------
// PART 1: plane channel. wall = axis normal to the walls, fdir = force axis.
//------------------------------------------------------------------------------
struct Chan { double l2, slip; std::size_t steps; };

static Chan channel(int wall, int fdir, Real tau, Index H, Real umax, std::size_t cap) {
  Index n[3] = {8, 8, 8};
  n[wall] = H + 2;
  const Real rho0 = Real(1);
  const Real nu = (tau - Real(0.5)) * cs2<L, Real>();
  const Real G = Real(8) * rho0 * nu * umax / (Real(H) * Real(H));

  bool per[3] = {true, true, true};
  per[wall] = false;
  Domain d(n[0], n[1], n[2], per[0], per[1], per[2]);

  CM coll;
  coll.omega = Real(1) / tau;
  Real Fv[3] = {Real(0), Real(0), Real(0)};
  Fv[fdir] = G;
  coll.forcing = Guo{Fv[0], Fv[1], Fv[2]};

  FluidSolver<L, EsotericPull<L>, CM> s(d, coll);
  s.set_geometry([&](Index x, Index y, Index z) -> CellType {
    const Index c = (wall == 0) ? x : (wall == 1) ? y : z;
    return (c == 0 || c == n[wall] - 1) ? Solid : Fluid;
  });
  s.initialize(rho0);

  auto vel = [&]() { return fdir == 0 ? s.ux() : (fdir == 1 ? s.uy() : s.uz()); };
  auto at = [&](Index j) {
    Index c[3] = {n[0] / 2, n[1] / 2, n[2] / 2};
    c[wall] = j;
    return d.id(c[0], c[1], c[2]);
  };

  const std::size_t probe = 200;
  Real prev = 0; std::size_t taken = 0;
  for (std::size_t t = 0; t < cap; t += probe) {
    for (std::size_t k = 0; k < probe; ++k) s.step();
    taken += probe;
    s.compute_macroscopic();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, vel());
    const Real cur = h(at(n[wall] / 2));
    if (t > 0 && std::abs(double(cur - prev)) < 1e-15 * std::abs(double(cur))) break;
    prev = cur;
  }

  s.compute_macroscopic();
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, vel());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
  double rsum = 0;
  for (Index j = 1; j <= H; ++j) rsum += double(hr(at(j)));
  const double rho_mean = rsum / double(H);
  const double A = double(G) / (2.0 * rho_mean * double(nu));

  double num = 0, den = 0;
  for (Index j = 1; j <= H; ++j) {
    const double y = double(j);
    const double ana = A * (y - 0.5) * (double(H) + 0.5 - y);
    const double e = double(hv(at(j))) - ana;
    num += e * e; den += ana * ana;
  }
  // The wall-slip behaviour is characterised thoroughly on D2Q9 in
  // validation/forcing_transverse.cpp; what this test adds is the third axis
  // and the cross-orientation comparison, so only the L2 norm is reported.
  return {std::sqrt(num / den), 0.0, taken};
}

//------------------------------------------------------------------------------
// PART 2: fully 3D, Fx = F0 sin(k y) sin(k z), exact steady solution.
//------------------------------------------------------------------------------
struct Sine { double l2, amp_err; std::size_t steps; };

static Sine sine3d(Index N, Real tau, Real umax, std::size_t cap) {
  const Real rho0 = Real(1);
  const Real nu = (tau - Real(0.5)) * cs2<L, Real>();
  const double k = 2.0 * PI / double(N);
  // umax = F0 / (2 rho nu k^2)
  const double F0 = double(umax) * 2.0 * double(rho0) * double(nu) * k * k;

  Domain d(N, N, N, true, true, true);
  CMS coll;
  coll.omega = Real(1) / tau;
  coll.forcing = SinForce{Real(F0), Real(k), d};

  FluidSolver<L, EsotericPull<L>, CMS> s(d, coll);
  s.initialize(rho0);

  const std::size_t probe = 100;
  Real prev = 0; std::size_t taken = 0;
  for (std::size_t t = 0; t < cap; t += probe) {
    for (std::size_t j = 0; j < probe; ++j) s.step();
    taken += probe;
    s.compute_macroscopic();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
    const Real cur = h(d.id(0, N / 4, N / 4));
    if (t > 0 && std::abs(double(cur - prev)) < 1e-14 * (std::abs(double(cur)) + 1e-30)) break;
    prev = cur;
  }

  s.compute_macroscopic();
  auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hz = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());

  double num = 0, den = 0, peak = 0;
  for (Index z = 0; z < N; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index id = d.id(x, y, z);
        const double ana = double(umax) * std::sin(k * double(y)) * std::sin(k * double(z));
        const double ex = double(hx(id)) - ana;
        num += ex * ex + double(hy(id)) * double(hy(id)) + double(hz(id)) * double(hz(id));
        den += ana * ana;
        peak = std::max(peak, std::abs(double(hx(id))));
      }
  return {std::sqrt(num / den), peak / double(umax) - 1.0, taken};
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    std::size_t cap = 4000000;
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == "-cap" && i + 1 < argc) cap = std::size_t(std::atol(argv[i + 1]));

    const Real umax = Real(0.02);
    std::printf("D3Q27 forcing   central moments, Guo, EsotericPull + shifted\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    // ---- PART 1 ----
    std::printf("PART 1  plane channel, every axis in both roles\n");
    std::printf("  %-18s %7s %5s %14s %11s\n", "walls / force", "tau", "H", "L2 rel err", "steps");
    std::printf("  %s\n", std::string(62, '-').c_str());
    const char* ax = "xyz";
    const Real taus[3] = {Real(0.875), Real(0.6), Real(1.2)};
    for (Real tau : taus) {
      double ref = -1;
      for (int c = 0; c < 3; ++c) {
        const int wall = c, fdir = (c + 1) % 3;
        const Chan r = channel(wall, fdir, tau, 16, umax, cap);
        char lbl[32];
        std::snprintf(lbl, sizeof lbl, "wall %c, force %c", ax[wall], ax[fdir]);
        std::printf("  %-18s %7.4f %5d %14.4e %11zu\n", lbl, double(tau), 16, r.l2, r.steps);
        if (ref < 0) ref = r.l2;
        else if (std::abs(r.l2 - ref) > 1e-12) {
          std::printf("      ^ differs from the first orientation by %.2e   ANISOTROPIC\n",
                      std::abs(r.l2 - ref));
          ++bad;
        }
        if (std::abs(double(tau) - 0.875) < 1e-12 && r.l2 > 1e-11) {
          std::printf("      ^ magic point NOT exact\n"); ++bad;
        }
      }
      std::printf("\n");
    }

    // ---- PART 2 ----
    std::printf("PART 2  fully 3D: Fx = F0 sin(k y) sin(k z), exact NS solution\n");
    std::printf("  %5s %7s %14s %8s %13s %11s\n",
                "N", "tau", "L2 rel err", "order", "peak err", "steps");
    std::printf("  %s\n", std::string(64, '-').c_str());
    for (Real tau : {Real(0.8), Real(1.2)}) {
      double prev = 0; Index pN = 0;
      for (Index N : {Index(8), Index(16), Index(32), Index(64)}) {
        const Sine r = sine3d(N, tau, umax, cap);
        const double ord = pN ? std::log(prev / r.l2) / std::log(2.0) : NAN;
        std::printf("  %5d %7.4f %14.4e", int(N), double(tau), r.l2);
        if (pN) std::printf(" %8.3f", ord); else std::printf(" %8s", "--");
        std::printf(" %13.4e %11zu\n", r.amp_err, r.steps);
        std::fflush(stdout);
        prev = r.l2; pN = N;
      }
      std::printf("\n");
    }
    // ---- PART 3: where does the amplitude error vanish? ----
    // The error in PART 2 is pure amplitude and changes sign with tau, so some
    // tau makes it zero. That is the bulk analogue of the wall magic point: it
    // is the viscosity at which the operator's O(k^2) dispersion error cancels.
    std::printf("PART 3  amplitude error vs tau   (N = 32, fully 3D case)\n");
    std::printf("  %8s %15s\n", "tau", "peak err");
    std::printf("  %s\n", std::string(26, '-').c_str());
    double prevt = 0, preve = 0, root = 0;
    for (double t = 0.6; t <= 1.45; t += 0.05) {
      const Sine r = sine3d(32, Real(t), umax, cap);
      std::printf("  %8.4f %15.4e%s\n", t, r.amp_err,
                  (preve != 0 && preve * r.amp_err < 0) ? "   <-- sign change" : "");
      if (preve != 0 && preve * r.amp_err < 0)
        root = prevt + (t - prevt) * (-preve) / (r.amp_err - preve);
      prevt = t; preve = r.amp_err;
      std::fflush(stdout);
    }
    if (root > 0) std::printf("\n  amplitude error vanishes near tau = %.4f\n", root);
    std::printf("\n  %s\n", bad == 0 ? "PART 1 PASS" : "*** PART 1 FAIL ***");
  }
  Kokkos::finalize();
  return bad == 0 ? 0 : 1;
}
