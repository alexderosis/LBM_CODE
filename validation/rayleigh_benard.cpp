//==============================================================================
//  Rayleigh-Benard convection in two dimensions.
//
//  A layer of depth H heated from below and cooled from above, periodic in the
//  horizontal, no-slip and isothermal on both plates. The controlling groups are
//
//      Ra = g beta dT H^3 / (nu D),      Pr = nu / D.
//
//  BOUNDARY CONDITIONS. The scalar uses the Dirichlet (anti-bounce-back)
//  condition and the fluid uses halfway bounce-back, so BOTH wall planes sit
//  midway between the last fluid node and the wall node -- at y = 0.5 and
//  y = H + 0.5, giving a layer of exactly H lattice units. Pairing an on-node
//  condition for one field with a midway condition for the other would put the
//  thermal and viscous boundaries half a spacing apart and corrupt Ra itself.
//
//  WHAT IS BEING CHECKED. Two things, of very different sharpness:
//
//   1. The onset. Linear stability of a rigid-rigid layer gives Ra_c = 1707.762
//      at critical wavenumber k_c H = 3.117 -- a classical result known to
//      seven figures, and the single most demanding scalar this configuration
//      can be asked for. It tests the buoyancy coupling, both boundary
//      conditions and the diffusivity calibration simultaneously: an error in
//      any one of them moves Ra_c. The box width is set to the critical
//      wavelength 2 pi / k_c = 2.0158 H so that the marginal mode fits exactly.
//
//   2. Nu(Ra) above onset. Below Ra_c the state is pure conduction and Nu = 1
//      identically, which is exact. Above it, Nu is reported from two
//      independent estimators -- the wall gradient and the volume-averaged
//      convective flux -- which must agree; disagreement means the run is not
//      converged rather than that the physics is interesting.
//==============================================================================
#include "boundary/MomentDirichlet.hpp"
#include "boundary/Regularized.hpp"
#include "collision/BGK.hpp"
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "FieldDump.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace lbm;

using FL = D2Q9;
using SL = D2Q5;
using FluidColl = BGK<FL, SecondOrderEquilibrium<FL>, BoussinesqGuo, ShiftedPopulations>;

struct RB {
  double nu_wall, nu_vol;   // two independent Nusselt estimators
  double growth;            // d ln(E_kin) / dt, in units of 1 / (H^2 / D)
  double umax;
  std::size_t steps;
  bool ok;
};

// `on_node` selects which family of wall conditions to use. Both give a layer
// of depth exactly H, so Ra means the same thing in each:
//   false -- bounce-back + anti-bounce-back, planes midway at 0.5 and H+0.5;
//   true  -- regularised + moment, planes ON nodes 0 and H.
static RB run(Index H, double Ra, double Pr, Real uc, std::size_t nsteps,
              bool measure_growth, bool on_node = false) {
  const Index nx = Index(std::lround(2.0158 * double(H)));   // one critical wavelength
  const Index ny = on_node ? (H + 1) : (H + 2);

  const Real nu = Real(double(H) * double(uc) * std::sqrt(Pr / Ra));
  const Real D  = Real(double(nu) / Pr);
  const Real gb = Real(double(uc) * double(uc) / double(H)); // g*beta, with dT = 1

  Domain d(nx, ny, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  ScalarBGK<SL> scoll;
  scoll.omega = ScalarBGK<SL>::omega_from_diffusivity(D);
  scoll.T_ref = Real(0);
  ScalarSolver<SL, EsotericPull<SL>, ScalarBGK<SL>> th(d, scoll);
  th.set_geometry([&](Index, Index y, Index) -> ScalarCell {
    if (y == 0 || y == ny - 1) return on_node ? ScalarMoment : ScalarDirichlet;
    return ScalarBulk;
  });
  th.set_wall_values([&](Index, Index y, Index) -> Real {
    return (y == 0) ? Real(0.5) : Real(-0.5);      // hot below, cold above
  });

  // Conduction profile plus a small perturbation at the critical wavelength.
  // Starting from the exact conductive state and nudging it is what makes the
  // growth rate a clean measurement of the linear mode.
  const double eps = 1e-3, kx = 2.0 * M_PI / double(nx);
  const Index Hc = H, nyc = ny; const bool onn = on_node;
  th.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Index x = px - d.hx, y = py - d.hy;
    const double y0 = onn ? 0.0 : 0.5;                        // hot plane position
    if (!onn && (y <= 0 || y >= nyc - 1)) return Real(0);
    const double yy = (double(y) - y0) / double(Hc);
    const double cond = 0.5 - yy;
    const double pert = eps * std::sin(kx * double(x)) * std::sin(M_PI * yy);
    return Real(cond + pert);
  });
  th.finalize_geometry();
  th.compute_field();

  BoussinesqGuo force;
  force.T = th.temperature();
  force.gx = Real(0); force.gy = Real(1); force.gz = Real(0);
  force.rho0 = Real(1); force.beta = gb; force.T0 = Real(0);

  FluidColl fcoll;
  fcoll.omega = BGK<FL>::omega_from_viscosity(nu);
  fcoll.forcing = force;
  FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fcoll);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    if (on_node) return Fluid;
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });
  if (on_node) {
    using WS = decltype(fl)::WallSpec;
    fl.set_regularized_walls([&](Index, Index y, Index) -> WS {
      if (y == 0)      return WS{NrmYm, Real(0), Real(0), Real(0)};
      if (y == ny - 1) return WS{NrmYp, Real(0), Real(0), Real(0)};
      return WS{};
    });
  }
  fl.initialize(Real(1));
  th.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // Amplitude of the CRITICAL MODE, not the total kinetic energy. Projecting
  // v onto sin(k x) sin(pi y) rejects everything else -- acoustic transients,
  // round-off, the spurious currents of the initial state -- which a bulk
  // energy sum happily counts as growth. With a 1e-6 seed the energy sits near
  // round-off and its "growth rate" is not even monotonic in Ra.
  const Index ylo = on_node ? Index(0) : Index(1);
  const Index yhi = on_node ? Index(ny - 1) : H;
  const double y0p = on_node ? 0.0 : 0.5;
  auto amplitude = [&]() {
    fl.compute_macroscopic();
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    double c = 0, sN = 0;
    for (Index y = ylo; y <= yhi; ++y)
      for (Index x = 0; x < nx; ++x) {
        const double w = std::sin(kx * double(x)) *
                         std::sin(M_PI * (double(y) - y0p) / double(H));
        c  += double(hy(d.id(x, y))) * w;
        sN += w * w;
      }
    return std::abs(c) / std::max(sN, 1e-300);
  };

  // Growth rate of the linear mode, sampled after a short transient so the
  // initial condition's non-modal content has decayed.
  const double t_diff = double(H) * double(H) / double(D);   // diffusive time
  double growth = 0;
  std::size_t taken = 0;
  if (measure_growth) {
    const std::size_t warm = std::size_t(0.25 * t_diff);
    const std::size_t span = std::size_t(0.50 * t_diff);
    for (std::size_t k = 0; k < warm; ++k) { fl.step(true); th.step(); }
    const double e0 = amplitude();
    for (std::size_t k = 0; k < span; ++k) { fl.step(true); th.step(); }
    const double e1 = amplitude();
    taken = warm + span;
    if (!(e0 > 0) || !std::isfinite(e1)) return {0, 0, 0, 0, taken, false};
    growth = std::log(e1 / e0) / (double(span) / t_diff);   // per diffusive time
    return {0, 0, growth, 0, taken, true};
  }

  for (std::size_t k = 0; k < nsteps; ++k) { fl.step(true); th.step(); }
  taken = nsteps;

  th.compute_field(); fl.compute_macroscopic();
  auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
  if (std::getenv("FIGDUMP")) {
    using namespace lbm::figdump;
    auto gx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    auto gy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    // tag by Rayleigh number, or every case in a sweep overwrites the last
    char rb[32]; std::snprintf(rb, sizeof rb, "ra%d", int(std::lround(Ra)));
    const std::string tag = rb;
    scalar_slice("rb_T_" + tag + ".bin", nx, ny,
                 [&](Index x, Index y) { return double(hT(d.id(x, y))); });
    scalar_slice("rb_speed_" + tag + ".bin", nx, ny, [&](Index x, Index y) {
      const double a = double(gx(d.id(x, y))), b = double(gy(d.id(x, y)));
      return std::sqrt(a * a + b * b);
    });
  }
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  for (Index y = ylo; y <= yhi; ++y)
    for (Index x = 0; x < nx; ++x)
      if (!std::isfinite(double(hT(d.id(x, y))))) return {0, 0, 0, 0, taken, false};

  // (a) wall gradient at the hot plate. The plane is at y = 0.5, so the
  // one-sided stencil uses spacings 0.5 and 1.5 -- the same unequal-spacing
  // formula the cavity case uses.
  double accw = 0;
  for (Index x = 0; x < nx; ++x) {
    const double Tw = 0.5;
    double dTdy;
    if (on_node) {   // wall AT node 0: uniform spacing, second-order one-sided
      const double T1 = double(hT(d.id(x, 1))), T2 = double(hT(d.id(x, 2)));
      dTdy = -1.5 * Tw + 2.0 * T1 - 0.5 * T2;
    } else {         // wall at y = 0.5: spacings 0.5 and 1.5
      const double T1 = double(hT(d.id(x, 1))), T2 = double(hT(d.id(x, 2)));
      dTdy = -(2.0 / 0.75) * Tw + 3.0 * T1 - (1.0 / 3.0) * T2;
    }
    accw += -dTdy * double(H);
  }
  const double nu_wall = accw / double(nx);

  // (b) Nu = 1 + <v' T'> H / (D dT), the volume relation -- on the
  // FLUCTUATIONS. The textbook form uses <v T> and is equivalent only where
  // <v> = 0. That is true of an incompressible closed layer but NOT of the
  // velocity this scheme reports: Guo's half shift adds F/(2 rho), so a layer
  // whose mean temperature differs from T0 carries a uniform vertical offset
  // belonging to the forcing scheme rather than to any mass flux.
  //
  // It is zero for every case in THIS file, because they all start from the
  // conduction profile and its mean is exactly zero in the symmetric +/- 1/2
  // gauge -- so none of the numbers below move. It is written correctly anyway
  // because the term is not small when the mean is not zero: a cold-start
  // initial condition at Ra = 1e14 made the textbook form read 53.77 at t = 0,
  // with the fluid at rest, where the answer is exactly 1
  // (demonstrator/rb_high_ra.cpp records the arithmetic).
  double accv = 0, umax = 0, accu = 0, acct = 0;
  Index ncell = 0;
  for (Index y = ylo; y <= yhi; ++y)
    for (Index x = 0; x < nx; ++x) {
      accv += double(hv(d.id(x, y))) * double(hT(d.id(x, y)));
      accu += double(hv(d.id(x, y)));
      acct += double(hT(d.id(x, y)));
      umax = std::max(umax, std::hypot(double(hu(d.id(x, y))), double(hv(d.id(x, y)))));
      ++ncell;
    }
  const double nu_vol = 1.0 + (accv / double(ncell)
                               - (accu / double(ncell)) * (acct / double(ncell)))
                              * double(H) / double(D);

  return {nu_wall, nu_vol, 0, umax, taken, true};
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index H = 48;
    double Pr = 0.71;
    Real uc = Real(0.02);
    bool onset = true, sweep = true, on_node = false, conv = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-h"  && i + 1 < argc) H = std::atoi(argv[++i]);
      if (a == "-pr" && i + 1 < argc) Pr = std::atof(argv[++i]);
      if (a == "-uc" && i + 1 < argc) uc = Real(std::atof(argv[++i]));
      if (a == "-noonset") onset = false;
      if (a == "-nosweep") sweep = false;
      if (a == "-onnode") on_node = true;
      if (a == "-conv") conv = true;
    }
    const Index nxp = Index(std::lround(2.0158 * double(H)));

    std::printf("Rayleigh-Benard convection, 2D\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());
    std::printf("  H = %d   box = %d x %d   Pr = %.2f   u_c = %.3f\n",
                int(H), int(nxp), int(H + 2), Pr, double(uc));
    std::printf("  walls: %s\n",
                on_node ? "regularised + moment  (planes ON nodes 0 and H)"
                        : "bounce-back + anti-bounce-back  (planes midway)");
    std::printf("  layer depth exactly H either way, so Ra means the same thing\n\n");

    auto find_rac = [&](Index Hh, bool onn) {
      double lo = 1400, hi = 2100;
      for (int it = 0; it < 10; ++it) {
        const double mid = 0.5 * (lo + hi);
        const RB r = run(Hh, mid, Pr, uc, 0, true, onn);
        if (r.ok && r.growth > 0) hi = mid; else lo = mid;
      }
      return 0.5 * (lo + hi);
    };

    if (conv) {
      std::printf("  --- Ra_c vs resolution, both wall families ---\n");
      std::printf("  %5s %13s %9s %13s %9s\n",
                  "H", "midway", "dev %", "on-node", "dev %");
      std::printf("  %s\n", std::string(56, '-').c_str());
      for (Index Hh : {Index(16), Index(24), Index(32), Index(48)}) {
        const double a = find_rac(Hh, false), b = find_rac(Hh, true);
        std::printf("  %5d %13.1f %+8.2f %13.1f %+8.2f\n", int(Hh),
                    a, 100.0 * (a - 1707.762) / 1707.762,
                    b, 100.0 * (b - 1707.762) / 1707.762);
        std::fflush(stdout);
      }
      std::printf("\n  reference Ra_c = 1707.762 (rigid-rigid, k_c H = 3.117)\n\n");
      Kokkos::finalize();
      return 0;
    }

    if (onset) {
      std::printf("  --- onset: bisection on the sign of the growth rate ---\n");
      std::printf("  %10s %14s %10s\n", "Ra", "d lnE / dt", "verdict");
      std::printf("  %s\n", std::string(40, '-').c_str());
      double lo = 1400, hi = 2100;
      for (int it = 0; it < 9; ++it) {
        const double mid = 0.5 * (lo + hi);
        const RB r = run(H, mid, Pr, uc, 0, true, on_node);
        const bool grows = r.ok && r.growth > 0;
        std::printf("  %10.2f %14.5f %10s\n", mid, r.growth, grows ? "grows" : "decays");
        std::fflush(stdout);
        if (grows) hi = mid; else lo = mid;
      }
      const double rac = 0.5 * (lo + hi);
      std::printf("\n  Ra_c (measured) = %.1f    reference 1707.76    deviation %+.2f%%\n\n",
                  rac, 100.0 * (rac - 1707.762) / 1707.762);
    }

    if (sweep) {
      std::printf("  --- Nu(Ra) ---\n");
      std::printf("  %10s %8s %12s %12s %10s %10s\n",
                  "Ra", "Ra/Ra_c", "Nu (wall)", "Nu (volume)", "spread", "|u|max");
      std::printf("  %s\n", std::string(68, '-').c_str());
      for (double Ra : {1500.0, 2500.0, 5000.0, 1e4, 3e4, 5e4, 1e5}) {
        const double t_diff = double(H) * double(H) /
                              (double(H) * double(uc) * std::sqrt(Pr / Ra) / Pr);
        const std::size_t ns = std::size_t(12.0 * t_diff);
        const RB r = run(H, Ra, Pr, uc, ns, false, on_node);
        if (!r.ok) { std::printf("  %10.0f %8.2f   DIVERGED\n", Ra, Ra / 1707.762); continue; }
        const double spread = std::abs(r.nu_wall - r.nu_vol) /
                              std::max(1.0, std::abs(r.nu_vol));
        std::printf("  %10.0f %8.2f %12.4f %12.4f %9.2e %10.4f\n",
                    Ra, Ra / 1707.762, r.nu_wall, r.nu_vol, spread, r.umax);
        std::fflush(stdout);
      }
      std::printf("\n  below Ra_c the state is pure conduction and Nu = 1 exactly;\n");
      std::printf("  the two estimators must agree, or the run is simply not converged.\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
