//==============================================================================
//  Hartmann flow: the MHD analogue of Poiseuille flow.
//
//  Reference: P. J. Dellar, "Moment-Based Boundary Conditions for Lattice
//  Boltzmann Magnetohydrodynamics", Sec. 3-4, Eqs. (13)-(15).
//
//  A uniform body force F drives flow along a channel that a uniform field
//  B0 spans crosswise. The flow stretches that field into a streamwise
//  component b, whose Lorentz force resists the motion. Axes follow the paper:
//  x across the channel with walls at x = +-L, y along it and periodic.
//
//  BOUNDARY CONDITIONS. Two different mechanisms meet here, which is the point
//  of the test:
//    u  -- regularised velocity walls (Latt et al.), u = 0 at the wall NODE;
//    B  -- moment-based walls (Dellar, Eqs. 13a-13b), B = (B0, 0, 0) at the
//          wall node. Maxwell's equations make both components of B continuous
//          across the wall, so B simply takes its external applied value.
//  Both place the boundary ON the grid point, so they agree about where the
//  wall is; mixing one of these with a bounce-back condition would not.
//
//  EXACT SOLUTION, Eq. (14), with xi = x/L and H = B0 L / sqrt(nu eta):
//
//     b(xi) = (F L / B0) [ sinh(H xi)/sinh(H) - xi ]
//     u(xi) = (F L / B0) sqrt(eta/nu) coth(H) [ 1 - cosh(H xi)/cosh(H) ]
//
//  Both vanish at xi = +-1, and both should do so here to round-off rather
//  than to O(h^2) -- that is what "boundary at the node" buys.
//==============================================================================
#include "boundary/MomentDirichlet.hpp"
#include "boundary/Regularized.hpp"
#include "collision/MagneticBGK.hpp"
#include "collision/MhdBGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/MagneticSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;

using FL = D2Q9;
using ML = D2Q5;
using FluidColl = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations, Guo>;

struct Result {
  double eu, eb;      // l2 norms, Eq. (15)
  double wall_u, wall_b;
  double umax;
  std::size_t steps;
};

static Result run(Index nx, double Ha, Real nu, Real umax_target, bool dump) {
  const Index ny = 8;                       // periodic along the channel
  const Real  L  = Real(nx - 1) / Real(2);  // walls sit ON nodes 0 and nx-1
  const Real  eta = nu;                     // Pr_m = 1, so sqrt(nu eta) = nu
  const Real  B0  = Real(Ha) * nu / L;      // from H = B0 L / sqrt(nu eta)
  // u_max ~ (F L / B0) coth(H) (1 - sech H); invert for the target
  const double H = Ha;
  const double shape = (1.0 / std::tanh(H)) * (1.0 - 1.0 / std::cosh(H));
  const Real F = Real(double(umax_target) * double(B0) / (double(L) * shape));

  Domain d(nx, ny, 1, /*periodic x*/ false, /*y*/ true, /*z*/ true);

  MagneticBGK<ML> mc;
  mc.omega = MagneticBGK<ML>::omega_from_resistivity(eta);
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);

  FluidColl fc;
  fc.omega   = FluidColl::omega_from_viscosity(nu);
  fc.forcing = Guo{Real(0), F, Real(0)};
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fc);

  fl.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  using WS = decltype(fl)::WallSpec;
  fl.set_regularized_walls([&](Index x, Index, Index) -> WS {
    if (x == 0)      return WS{NrmXm, Real(0), Real(0), Real(0)};
    if (x == nx - 1) return WS{NrmXp, Real(0), Real(0), Real(0)};
    return WS{};
  });

  using WB = decltype(mag)::WallB;
  mag.set_moment_walls([&](Index x, Index, Index) -> WB {
    if (x == 0 || x == nx - 1) return WB{true, B0, Real(0), Real(0)};
    return WB{};
  });

  const Real B0c = B0;
  mag.initialize_field(KOKKOS_LAMBDA(Index) {
    Kokkos::Array<Real, 3> b; b[0] = B0c; b[1] = Real(0); b[2] = Real(0);
    return b;
  });
  fl.initialize(Real(1));
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // March to steady state on the centreline velocity.
  const std::size_t probe = 500, max_steps = 4000000;
  Real prev = 0; std::size_t taken = 0;
  for (std::size_t t = 0; t < max_steps; t += probe) {
    for (std::size_t k = 0; k < probe; ++k) {
      mag.compute_field(); fl.step(true); mag.step(true);
    }
    taken += probe;
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    const Real cur = h(d.id(nx / 2, ny / 2));
    if (!std::isfinite(double(cur))) return {NAN, NAN, NAN, NAN, NAN, taken};
    if (t > 0 && std::abs(double(cur - prev)) < 1e-13 * (std::abs(double(cur)) + 1e-30)) break;
    prev = cur;
  }

  fl.compute_macroscopic(); mag.compute_field();
  auto huy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto hby = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());

  const double A = double(F) * double(L) / double(B0);
  double su = 0, sb = 0, umax = 0;
  std::vector<double> xs, ub, bb, ua, ba;
  for (Index i = 0; i < nx; ++i) {
    const double xi = (double(i) - double(L)) / double(L);
    const double ana_u = A * std::sqrt(double(eta) / double(nu)) / std::tanh(H) *
                         (1.0 - std::cosh(H * xi) / std::cosh(H));
    const double ana_b = A * (std::sinh(H * xi) / std::sinh(H) - xi);
    const double num_u = double(huy(d.id(i, ny / 2)));
    const double num_b = double(hby(d.id(i, ny / 2)));
    su += (num_u - ana_u) * (num_u - ana_u);
    sb += (num_b - ana_b) * (num_b - ana_b);
    umax = std::max(umax, std::abs(num_u));
    xs.push_back(xi); ub.push_back(num_u); bb.push_back(num_b);
    ua.push_back(ana_u); ba.push_back(ana_b);
  }
  if (dump) {
    std::printf("\n  %8s %13s %13s %13s %13s\n", "x/L", "u (LB)", "u (exact)", "b (LB)", "b (exact)");
    for (std::size_t k = 0; k < xs.size(); k += (xs.size() > 20 ? xs.size() / 16 : 1))
      std::printf("  %8.4f %13.6e %13.6e %13.6e %13.6e\n", xs[k], ub[k], ua[k], bb[k], ba[k]);
    std::printf("  %8.4f %13.6e %13.6e %13.6e %13.6e\n",
                xs.back(), ub.back(), ua.back(), bb.back(), ba.back());
  }
  return {std::sqrt(su / double(nx)), std::sqrt(sb / double(nx)),
          std::abs(double(huy(d.id(0, ny / 2)))),
          std::abs(double(hby(d.id(0, ny / 2)))), umax, taken};
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    double Ha = 10.0;
    Real nu = Real(0.1), umax = Real(0.02);
    bool dump = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-ha" && i + 1 < argc) Ha = std::atof(argv[++i]);
      if (a == "-nu" && i + 1 < argc) nu = Real(std::atof(argv[++i]));
      if (a == "-dump") dump = true;
    }
    std::printf("Hartmann flow (Dellar, moment-based magnetic boundary conditions)\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());
    std::printf("  Ha = %.1f   nu = eta = %.4f   u_max target = %.3f\n", Ha, double(nu), double(umax));
    std::printf("  u: regularised walls    B: moment walls, Eqs. (13a)-(13b)\n\n");
    std::printf("  %5s %13s %8s %13s %8s %11s %11s\n",
                "n", "l2(u)", "order", "l2(b)", "order", "|u| wall", "|b| wall");
    std::printf("  %s\n", std::string(80, '-').c_str());

    double pu = 0, pb = 0; Index pn = 0;
    for (Index n : {Index(17), Index(33), Index(65), Index(129)}) {
      const Result r = run(n, Ha, nu, umax, dump && n == 65);
      const double ou = pn ? std::log(pu / r.eu) / std::log(double(n - 1) / double(pn - 1)) : NAN;
      const double ob = pn ? std::log(pb / r.eb) / std::log(double(n - 1) / double(pn - 1)) : NAN;
      if (pn) std::printf("  %5d %13.5e %8.3f %13.5e %8.3f %11.2e %11.2e\n",
                          int(n), r.eu, ou, r.eb, ob, r.wall_u, r.wall_b);
      else    std::printf("  %5d %13.5e %8s %13.5e %8s %11.2e %11.2e\n",
                          int(n), r.eu, "--", r.eb, "--", r.wall_u, r.wall_b);
      pu = r.eu; pb = r.eb; pn = n;
    }
    std::printf("\n  the wall columns should be at round-off: both conditions put the\n");
    std::printf("  boundary ON the node, so u and b vanish there exactly.\n");
  }
  Kokkos::finalize();
  return 0;
}
