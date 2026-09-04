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
#include <utility>
#include <vector>

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
// `aspect` is the box width in units of H, and since the box is periodic it IS
// the imposed horizontal wavelength: k H = 2 pi / aspect. The default 2.0158 is
// the critical wavelength, 2 pi / 3.117. Making it a parameter is what turns
// this from a test of Ra_c into a test of the whole MARGINAL CURVE Ra(k), whose
// minimum is the second seven-figure number in the problem.
static RB run(Index H, double Ra, double Pr, Real uc, std::size_t nsteps,
              bool measure_growth, bool on_node = false,
              double aspect = 2.0158) {
  const Index nx = Index(std::lround(aspect * double(H)));
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
    bool marginal = false, rate = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-h"  && i + 1 < argc) H = std::atoi(argv[++i]);
      if (a == "-pr" && i + 1 < argc) Pr = std::atof(argv[++i]);
      if (a == "-uc" && i + 1 < argc) uc = Real(std::atof(argv[++i]));
      if (a == "-noonset") onset = false;
      if (a == "-nosweep") sweep = false;
      if (a == "-onnode") on_node = true;
      if (a == "-conv") conv = true;
      if (a == "-marginal") marginal = true;
      if (a == "-rate") rate = true;
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

    // The bracket has to hold the whole marginal curve, not just its minimum:
    // away from k_c the neutral Rayleigh number rises steeply, so [1400, 2100]
    // is enough for the critical wavelength and not for the wings. 13 halvings
    // of [1500, 5000] leave 0.43, well inside the O(1/H^2) discretisation.
    auto find_rac_k = [&](Index Hh, bool onn, double aspect) {
      double lo = 1500, hi = 5000;
      for (int it = 0; it < 13; ++it) {
        const double mid = 0.5 * (lo + hi);
        const RB r = run(Hh, mid, Pr, uc, 0, true, onn, aspect);
        if (r.ok && r.growth > 0) hi = mid; else lo = mid;
      }
      return 0.5 * (lo + hi);
    };
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

    //=========================================================================
    // THE MARGINAL CURVE, and its minimum. Linear stability of a rigid-rigid
    // layer gives a neutral curve Ra(k) with a single minimum,
    //
    //     min Ra = 1707.762   at   k_c H = 3.117,
    //
    // and BOTH numbers are references. Ra_c alone can be hit by a code that has
    // the wrong preferred wavelength, because this box only admits k = 2 pi / L
    // and the case picks L = 2 pi / k_c by hand -- so measuring Ra_c at the
    // assumed k_c tests one number and assumes the other. Sweeping L measures
    // the curve, and the LOCATION of its minimum is then an independent check
    // on k_c that no choice of box width can fake.
    //
    // k H is quantised because nx is an integer, so the requested and the
    // realised k H both print; the fit uses the realised one.
    //=========================================================================
    if (marginal) {
      std::printf("  --- marginal curve Ra(k), and where its minimum lies ---\n");
      std::printf("  %8s %10s %6s %12s\n", "k H req", "k H", "nx", "Ra_c");
      std::printf("  %s\n", std::string(40, '-').c_str());
      std::vector<double> kk, rr;
      for (double khr : {2.2, 2.6, 3.0, 3.117, 3.4, 3.8, 4.2}) {
        const double aspect = 2.0 * M_PI / khr;
        const Index nxk = Index(std::lround(aspect * double(H)));
        const double kh = 2.0 * M_PI * double(H) / double(nxk);
        const double rac = find_rac_k(H, on_node, double(nxk) / double(H));
        std::printf("  %8.3f %10.4f %6d %12.1f\n", khr, kh, int(nxk), rac);
        std::fflush(stdout);
        kk.push_back(kh); rr.push_back(rac);
      }
      // THE VERTEX MUST BE FITTED LOCALLY, and getting this wrong cost a
      // factor of thirty in accuracy. A parabola across the whole sampled range
      // is the WRONG model: the neutral curve is markedly asymmetric -- measured
      // here, the low-k side is 2.13x steeper in d2Ra/dk2 than the high-k side --
      // so a global fit trades vertical error against horizontal position and
      // places the vertex toward the shallower side. On this data it returned
      // k_c H = 3.2045 (+2.81%) where a three-point fit around the minimum gives
      // 3.1196 (+0.08%) and a quartic through all seven gives 3.1085 (-0.27%).
      //
      // So the answer is the three-point parabola through the lowest sample and
      // its neighbours -- textbook parabolic interpolation for a minimum -- and
      // the global fit is printed BESIDE it rather than deleted, because the
      // size of its bias is the evidence for the asymmetry and someone will
      // otherwise reach for it again.
      //
      // The residual uncertainty is the estimator, not the simulation: the two
      // sound estimators bracket 3.117 at +0.08% and -0.27%. Sampling the vertex
      // more finely needs a larger H, because nx is an integer and k H = 2 pi
      // H / nx is quantised at about 0.05 near the minimum at H = 32.
      double S[5] = {0, 0, 0, 0, 0}, T[3] = {0, 0, 0};
      for (std::size_t i = 0; i < kk.size(); ++i) {
        double x = kk[i], p = 1.0;
        for (int j = 0; j < 5; ++j) { S[j] += p; p *= x; }
        T[0] += rr[i]; T[1] += rr[i] * x; T[2] += rr[i] * x * x;
      }
      // solve the 3x3 normal equations by Cramer
      const double M[3][3] = {{S[0], S[1], S[2]}, {S[1], S[2], S[3]}, {S[2], S[3], S[4]}};
      auto det3 = [](const double m[3][3]) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
      };
      const double D0 = det3(M);
      double c[3];
      for (int j = 0; j < 3; ++j) {
        double Mj[3][3];
        for (int r2 = 0; r2 < 3; ++r2)
          for (int c2 = 0; c2 < 3; ++c2) Mj[r2][c2] = (c2 == j) ? T[r2] : M[r2][c2];
        c[j] = det3(Mj) / D0;
      }
      const double kglob = -c[1] / (2.0 * c[2]);

      // Three-point parabola about the lowest sample: the local, unbiased one.
      std::size_t im = 0;
      for (std::size_t i = 1; i < rr.size(); ++i) if (rr[i] < rr[im]) im = i;
      if (im == 0) im = 1;
      if (im + 1 >= rr.size()) im = rr.size() - 2;
      auto vertex3 = [&](std::size_t i) {
        const double x1 = kk[i - 1], x2 = kk[i], x3 = kk[i + 1];
        const double y1 = rr[i - 1], y2 = rr[i], y3 = rr[i + 1];
        const double den = (x1 - x2) * (x1 - x3) * (x2 - x3);
        const double A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / den;
        const double B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1)
                        + x1 * x1 * (y2 - y3)) / den;
        const double C = y2 - A * x2 * x2 - B * x2;
        const double xv = -B / (2.0 * A);
        return std::pair<double, double>{xv, A * xv * xv + B * xv + C};
      };
      const auto [kmin, ramin] = vertex3(im);
      // Curvature of each wing, which is the evidence that a global fit is wrong.
      const double cl = (kk.size() >= 3) ? vertex3(1).first : 0.0;
      double dlo = 0, dhi = 0;
      {
        const double x1 = kk[0], x2 = kk[1], x3 = kk[2];
        const double y1 = rr[0], y2 = rr[1], y3 = rr[2];
        dlo = 2.0 * ((y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)) / (x3 - x1);
        const std::size_t n2 = kk.size();
        const double u1 = kk[n2 - 3], u2 = kk[n2 - 2], u3 = kk[n2 - 1];
        const double v1 = rr[n2 - 3], v2 = rr[n2 - 2], v3 = rr[n2 - 1];
        dhi = 2.0 * ((v3 - v2) / (u3 - u2) - (v2 - v1) / (u2 - u1)) / (u3 - u1);
      }
      (void)cl;
      std::printf("\n  vertex, 3 points about the minimum (THE answer):\n");
      std::printf("    k_c H = %.4f   (reference 3.117, %+.2f%%)\n",
                  kmin, 100.0 * (kmin - 3.117) / 3.117);
      std::printf("    Ra_c  = %.1f    (reference 1707.762, %+.2f%%)\n",
                  ramin, 100.0 * (ramin - 1707.762) / 1707.762);
      std::printf("\n  for the record, a parabola fitted to ALL points gives\n");
      std::printf("    k_c H = %.4f   (%+.2f%%)  -- BIASED, see the banner\n",
                  kglob, 100.0 * (kglob - 3.117) / 3.117);
      std::printf("    because the curve is asymmetric: d2Ra/dk2 = %.0f on the low-k\n"
                  "    side against %.0f on the high-k side, a ratio of %.2f\n",
                  dlo, dhi, dhi != 0.0 ? dlo / dhi : 0.0);
      std::printf("\n  NOTE the two references are contaminated DIFFERENTLY. The O(1/H^2)\n"
                  "  error shifts the whole curve up in Ra, and a vertical shift does not\n"
                  "  move a vertex sideways -- so k_c is good to ~0.1%% here where Ra_c is\n"
                  "  only good to %+.2f%%. The preferred wavelength is the sharper test of\n"
                  "  the two at modest resolution, which is the opposite of how this file\n"
                  "  used them before: Ra_c as the headline and k_c baked into the box.\n\n",
                  100.0 * (ramin - 1707.762) / 1707.762);
      Kokkos::finalize();
      return 0;
    }

    //=========================================================================
    // THE GROWTH RATE AGAINST LINEAR THEORY, in the one way that needs no
    // remembered coefficient.
    //
    // Near onset the linear growth rate is proportional to the supercriticality,
    // sigma = A (Ra/Ra_c - 1) + O(...)^2, and it must pass through ZERO exactly
    // at Ra_c. Two parameter-free predictions follow: sigma is linear in Ra, and
    // its zero crossing is the SAME Ra_c the bisection finds independently. The
    // slope A is a genuine reference number too, but not one this file asserts
    // from memory -- it is reported as a measurement, for checking against a
    // source rather than against itself. Bracketing the sign of sigma, which is
    // all -onset does, is much weaker: it uses one bit of each run.
    //=========================================================================
    if (rate) {
      std::printf("  --- growth rate vs supercriticality, at k H = 3.117 ---\n");
      std::printf("  %10s %10s %16s\n", "Ra", "Ra/Ra_c", "d ln E / dt");
      std::printf("  %s\n", std::string(40, '-').c_str());
      std::vector<double> xr, yr;
      for (double Ra : {1650.0, 1700.0, 1750.0, 1800.0, 1900.0, 2000.0, 2200.0}) {
        const RB r = run(H, Ra, Pr, uc, 0, true, on_node);
        if (!r.ok) { std::printf("  %10.0f %10.3f %16s\n", Ra, Ra / 1707.762, "unusable"); continue; }
        std::printf("  %10.0f %10.3f %16.5f\n", Ra, Ra / 1707.762, r.growth);
        std::fflush(stdout);
        xr.push_back(Ra); yr.push_back(r.growth);
      }
      const std::size_t n = xr.size();
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (std::size_t i = 0; i < n; ++i)
        { sx += xr[i]; sy += yr[i]; sxx += xr[i] * xr[i]; sxy += xr[i] * yr[i]; }
      const double m = (double(n) * sxy - sx * sy) / (double(n) * sxx - sx * sx);
      const double b0 = (sy - m * sx) / double(n);
      double ss = 0, rs = 0; const double ybar = sy / double(n);
      for (std::size_t i = 0; i < n; ++i)
        { ss += (yr[i] - ybar) * (yr[i] - ybar);
          rs += (yr[i] - (m * xr[i] + b0)) * (yr[i] - (m * xr[i] + b0)); }
      const double zero = -b0 / m;
      const double rac_bis = find_rac(H, on_node);
      std::printf("\n  least squares:    R^2 = %.5f   (linear in Ra, as theory requires)\n",
                  1.0 - rs / ss);
      std::printf("  zero crossing:    Ra = %.1f\n", zero);
      std::printf("  bisected Ra_c:    Ra = %.1f    the two agree to %+.2f%%\n",
                  rac_bis, 100.0 * (zero - rac_bis) / rac_bis);
      std::printf("  slope d sigma/d(Ra/Ra_c) = %.4f per diffusive time"
                  "  (MEASURED, not checked against a source)\n\n",
                  m * 1707.762);
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
