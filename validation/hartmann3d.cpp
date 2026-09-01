//==============================================================================
//  Hartmann flow in three dimensions: Poiseuille with a wall-normal magnetic
//  field, on D3Q27 for the fluid and D3Q7 for the magnetic field.
//
//  Reference: P. J. Dellar, "Moment-Based Boundary Conditions for Lattice
//  Boltzmann Magnetohydrodynamics", Sec. 3-4, Eqs. (13)-(15).
//
//  GEOMETRY. A channel with walls at the BOTTOM and TOP (y = 0 and y = ny-1),
//  periodic along the flow (x) and across it (z). A uniform body force F drives
//  the flow along x; a uniform field B0 is applied along y, normal to the walls
//  and imposed at them. The flow stretches that field into a streamwise
//  component b_x whose Lorentz force resists the motion, and the balance is
//
//      0 = F + nu u_x'' + B0 b_x' / rho,     0 = B0 u_x' + eta b_x''.
//
//  This differs from validation/hartmann.cpp in more than an axis label. That
//  case is D2Q9 + D2Q5 with nz = 1; this one is D3Q27 + D3Q7 on a genuinely
//  three-dimensional domain, and D3Q7 has cs2 = 1/4 rather than D2Q5's 1/3, so
//  the resistivity calibration takes a different arithmetic path -- getting eta
//  wrong by 4/3 would show up here and nowhere in the 2-D case.
//
//  EXACT SOLUTION, Eq. (14), with xi = y/L measured from mid-channel, L the
//  half-width and Ha = B0 L / sqrt(nu eta):
//
//     b_x(xi) = (F L / B0) [ sinh(Ha xi)/sinh(Ha) - xi ]
//     u_x(xi) = (F L / B0) sqrt(eta/nu) coth(Ha) [ 1 - cosh(Ha xi)/cosh(Ha) ]
//
//  Both vanish at xi = +-1. As Ha -> 0 the second reduces to the parabola
//  F (L^2 - y^2) / 2 nu, which is the sense in which this is Poiseuille flow
//  with a field; -ha 0 runs exactly that, with B0 = 0 and the parabola as the
//  reference. It is worth having as its own row because it separates the
//  plumbing from the physics -- but only if what it should give is stated
//  correctly, and the first version of this banner did not.
//
//  The field-free case is NOT exact at a general omega. The regularised wall
//  carries a curvature-induced slip, delta u = -(2/3)((omega-1)/omega^2) d^2u/dn^2
//  (doc/m3lb.tex, known limitations), and a parabola is all curvature, so the
//  error is second order and not round-off: 4.4e-3 -> 1.1e-3 -> 2.8e-4 at
//  nu = 0.1, order 1.98. That slip vanishes identically at omega = 1, so the
//  table carries a nu = 1/6 row as well, where the same case comes back at
//  round-off. Those two rows together are what pins the error down: the driver,
//  the forcing and the geometry are exact, and the O(h^2) at nu = 0.1 is the
//  documented wall slip rather than anything here.
//
//  BOUNDARY CONDITIONS, and why they are the point. Two different mechanisms
//  meet at the same plane:
//    u  -- regularised velocity walls (Latt et al.), u = 0 AT the wall node;
//    B  -- Dellar's moment condition (Eqs. 13a-13b), B = (0, B0, 0) at the node.
//  On a straight wall the magnetic lattice leaves exactly one unknown direction,
//  so that single moment equation closes the system exactly -- no closure
//  assumption and no free parameter. Both families put the boundary ON the grid
//  point, so they agree about where the wall is; pairing either with bounce-back
//  would put the two boundaries half a cell apart and corrupt Ha itself, which
//  is the same trap the thermal cases document for the wall families there.
//
//  THE TWO WALL COLUMNS ARE NOT AN ACCURACY MEASUREMENT, and it is worth saying
//  so because the 2-D case's banner reads as though they were. Both conditions
//  IMPOSE their value at the node -- the regularised wall sets u = 0, the moment
//  condition sets B_x = 0 -- and the collision preserves the zeroth moment, so
//  those columns are zero by construction. What they do establish is that the
//  boundary sits ON the node: a bounce-back pairing would put the zero half a
//  cell away and the column would read O(u_max) instead. Read them as a wiring
//  check, not as a convergence result.
//
//  THE TRANSVERSE CHECK NEEDS A PERTURBATION, for the same reason. The exact
//  solution depends on y alone, so it is tempting to read "max spread over
//  (x, z) at fixed y" as a three-dimensionality test. It is not: the initial
//  state, the forcing and both walls are uniform in x and z, the lattice is
//  translation-invariant along them and the scheme is deterministic, so the
//  solution stays exactly x,z-invariant and the column reads 0.000e+00 whatever
//  the scheme does. That is the same vacuous-diagnostic trap as the
//  centro-symmetric Nusselt pair in validation/zhou_thermal.cpp.
//
//  -pert makes it a measurement. It seeds a transverse perturbation of relative
//  amplitude eps on u_x, cos(2 pi x/nx) cos(2 pi z/nz), which the Hartmann
//  solution should absorb: the run is then asked whether the spread DECAYS, and
//  by how much, rather than whether it is zero. The seeded and final spreads are
//  both reported so the ratio is visible.
//
//  RESOLUTION IS THE HARTMANN LAYER, NOT THE CHANNEL. The field confines the
//  velocity adjustment to a layer of thickness L/Ha at each wall, so the cell
//  count that matters is L/Ha and not L: at Ha = 10 and ny = 17 that is 0.8
//  cells. The column is printed beside every row so that a shortfall in the
//  fitted order can be attributed rather than explained away -- and so that the
//  absence of one is visible too, which is what happened here. The measured
//  orders at Ha = 10 are 2.12, 2.01 and 1.98 for u and 2.06, 2.00, 1.96 for b,
//  so the ladder is second order end to end even across the rows where the
//  layer has fewer than two cells in it. The overshoot above 2 at the coarse
//  end is the usual sign of a point outside the asymptotic range; it is not
//  evidence of anything better than second order.
//
//  COUPLING ORDER. B is refreshed BEFORE the fluid collides, every step. The
//  first coupled driver in this tree did it the other way and paid a
//  first-order splitting error whose ratio to the physical damping is
//  independent of N -- so it did not refine away, and the Alfven damping error
//  GREW under refinement while the phase speed converged cleanly. Any two-way
//  coupling wants that check.
//
//  BGK ONLY. MhdCentralMoments carries NoForcing on both its specialisations,
//  so the central-moment operator cannot be driven by a body force and this
//  case cannot use it. That is a gap in the operator, not in the case; the
//  Orszag-Tang vortex, which needs no force, runs both.
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using FL = D3Q27;                 // fluid lattice
using ML = D3Q7;                  // magnetic lattice, cs2 = 1/4

namespace {

//------------------------------------------------------------------------------
// Steady-state detector: the whole-field relative change over a probe interval,
// with the change still to come extrapolated geometrically. Thresholding the
// residual itself is not the same thing -- near the fixed point the interval
// change decays as res_{k+1} ~ q res_k, so the remaining drift is
// res q / (1 - q), and for a diffusive relaxation q sits within a percent of 1.
// Same argument, and the same struct, as validation/zhou_thermal.cpp.
//------------------------------------------------------------------------------
struct Settle {
  double tol = 1e-10;
  double res = 1, prev = 1, drift = 1, ratio = 1;
  int seen = 0;
  bool update(double r) {
    prev = res; res = r; ++seen;
    ratio = (prev > 0) ? res / prev : 1.0;
    drift = (ratio < 1.0 && ratio > 0.0) ? res * ratio / (1.0 - ratio) : res;
    return seen >= 3 && drift < tol;
  }
};

double interval_residual(const std::vector<double>& now,
                         const std::vector<double>& before) {
  double a = 0, b = 0;
  for (std::size_t k = 0; k < now.size(); ++k) {
    const double d = now[k] - before[k];
    a += d * d; b += now[k] * now[k];
  }
  return (b > 0) ? std::sqrt(a / b) : 0.0;
}

const char* arg_str(int argc, char** argv, const char* key, const char* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return dflt;
}
double arg_num(int argc, char** argv, const char* key, double dflt) {
  const char* s = arg_str(argc, argv, key, nullptr);
  return s ? std::atof(s) : dflt;
}
bool arg_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], key) == 0) return true;
  return false;
}

}  // namespace

//------------------------------------------------------------------------------
struct Result {
  double eu, eb;            // relative l2 over the whole domain, Eq. (15)
  double wall_u, wall_b;    // |u| and |b| at the wall node -- should be round-off
  double spread_u, spread_b;// max spread over (x,z) at fixed y, /peak
  double spread_seed;       // the same, right after seeding -- 0 without -pert
  double umax_num, umax_ana;
  double B0, F, nu, eta, tau_f, tau_m, delta;   // delta = L/Ha, in cells
  double res, drift;
  std::size_t steps;
  bool converged, finite;
};

template <class Eq>
Result run(Index ny, Index nx, Index nz, double Ha, double nu_in, double prm,
           double umax_target, double tol, std::size_t cap, std::size_t probe,
           const char* dump, double pert = 0.0) {
  using FluidColl = MhdBGK<FL, Eq, ShiftedPopulations, Guo>;

  const double L   = 0.5 * double(ny - 1);      // walls sit ON nodes 0 and ny-1
  const Real   nu  = Real(nu_in);
  const Real   eta = Real(nu_in / prm);
  const bool   mhd = Ha > 0.0;

  // Ha = B0 L / sqrt(nu eta)  ->  B0 = Ha sqrt(nu eta) / L.
  const double sq  = std::sqrt(double(nu) * double(eta));
  const Real   B0  = mhd ? Real(Ha * sq / L) : Real(0);

  // Invert the peak for the driving force. The MHD peak is
  // A sqrt(eta/nu) coth(Ha) (1 - sech Ha) with A = F L / B0; the field-free one
  // is F L^2 / 2 nu.
  const double shape = mhd ? (1.0 / std::tanh(Ha)) * (1.0 - 1.0 / std::cosh(Ha)) : 0.0;
  const Real F = mhd
      ? Real(umax_target * double(B0) / (L * shape * std::sqrt(double(eta) / double(nu))))
      : Real(2.0 * double(nu) * umax_target / (L * L));

  Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  MagneticBGK<ML> mc;
  mc.omega = MagneticBGK<ML>::omega_from_resistivity(eta);
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);

  FluidColl fc;
  fc.omega   = FluidColl::omega_from_viscosity(nu);
  fc.forcing = Guo{F, Real(0), Real(0)};        // along the channel
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fc);

  fl.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  using WS = typename decltype(fl)::WallSpec;
  fl.set_regularized_walls([&](Index, Index y, Index) -> WS {
    if (y == 0)      return WS{NrmYm, Real(0), Real(0), Real(0)};
    if (y == ny - 1) return WS{NrmYp, Real(0), Real(0), Real(0)};
    return WS{};
  });

  // The applied field is normal to the walls and continuous across them, so the
  // wall node simply takes its external value -- that is what Maxwell gives and
  // what the moment condition imposes, exactly, in one equation.
  using WB = typename decltype(mag)::WallB;
  mag.set_moment_walls([&](Index, Index y, Index) -> WB {
    if (y == 0 || y == ny - 1) return WB{true, Real(0), B0, Real(0)};
    return WB{};
  });

  const Real B0c = B0;
  mag.initialize_field(KOKKOS_LAMBDA(Index) {
    Kokkos::Array<Real, 3> b; b[0] = Real(0); b[1] = B0c; b[2] = Real(0);
    return b;
  });
  // The transverse perturbation, if asked for. It carries no net momentum in x
  // (a cosine over a whole period) so it does not shift the flow rate, and it
  // is divergence-free by being a single velocity component varying only in the
  // OTHER two directions.
  const Real eps = Real(pert * umax_target);
  const Index nxc = nx, nzc = nz;
  if (pert > 0.0) {
    fl.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real X = Real(px - d.hx) / Real(nxc), Z = Real(pz - d.hz) / Real(nzc);
      FlowState s;
      s.rho = Real(1);
      s.ux = eps * Kokkos::cos(Real(2) * Real(M_PI) * X) *
                   Kokkos::cos(Real(2) * Real(M_PI) * Z);
      s.uy = Real(0); s.uz = Real(0);
      return s;
    });
  } else {
    fl.initialize(Real(1));
  }
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // The seeded spread, measured before any stepping, so the decay ratio below
  // is against something real rather than against the nominal amplitude.
  double spread_seed = 0;
  {
    fl.compute_macroscopic();
    auto h0 = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    for (Index y = 0; y < ny; ++y) {
      double lo = 1e300, hi = -1e300;
      for (Index z = 0; z < nz; ++z)
        for (Index x = 0; x < nx; ++x) {
          const double v = double(h0(d.id(x, y, z)));
          lo = std::min(lo, v); hi = std::max(hi, v);
        }
      spread_seed = std::max(spread_seed, hi - lo);
    }
  }

  // Exact solution, as a function of the wall-normal index.
  const double A = mhd ? double(F) * L / double(B0) : 0.0;
  auto ana_u = [&](double xi) {
    return mhd ? A * std::sqrt(double(eta) / double(nu)) / std::tanh(Ha) *
                     (1.0 - std::cosh(Ha * xi) / std::cosh(Ha))
               : double(F) * L * L * (1.0 - xi * xi) / (2.0 * double(nu));
  };
  auto ana_b = [&](double xi) {
    return mhd ? A * (std::sinh(Ha * xi) / std::sinh(Ha) - xi) : 0.0;
  };

  auto snapshot = [&]() {
    fl.compute_macroscopic();
    mag.compute_field();
    auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    auto hb = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
    std::vector<double> v;
    for (Index y = 0; y < ny; ++y) {
      v.push_back(double(hu(d.id(nx / 2, y, nz / 2))) / umax_target);
      v.push_back(double(hb(d.id(nx / 2, y, nz / 2))) / umax_target);
    }
    return v;
  };

  std::vector<double> now, before = snapshot();
  Settle st{tol};
  std::size_t taken = 0;
  bool converged = false, finite = true;
  while (taken < cap) {
    for (std::size_t k = 0; k < probe; ++k) {
      mag.compute_field(); fl.step(true); mag.step(true);
    }
    taken += probe;
    now = snapshot();
    for (double v : now) if (!std::isfinite(v)) finite = false;
    if (!finite) break;
    converged = st.update(interval_residual(now, before));
    before = now;
    if (converged) break;
  }

  fl.compute_macroscopic();
  mag.compute_field();
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hb = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());

  // Relative l2 over EVERY node, not one line: the exact solution depends on y
  // alone, so any x or z structure is error and belongs in the norm. The spread
  // columns then say how much of it that is.
  double su = 0, sb = 0, du = 0, db = 0, umax_num = 0, umax_ana = 0;
  double sp_u = 0, sp_b = 0, peak_b = 0;
  for (Index y = 0; y < ny; ++y) {
    const double xi = (double(y) - L) / L;
    const double au = ana_u(xi), ab = ana_b(xi);
    umax_ana = std::max(umax_ana, std::abs(au));
    peak_b   = std::max(peak_b, std::abs(ab));
    double lo_u = 1e300, hi_u = -1e300, lo_b = 1e300, hi_b = -1e300;
    for (Index z = 0; z < nz; ++z)
      for (Index x = 0; x < nx; ++x) {
        const Index n = d.id(x, y, z);
        const double nu_v = double(hu(n)), nb_v = double(hb(n));
        su += (nu_v - au) * (nu_v - au);  du += au * au;
        sb += (nb_v - ab) * (nb_v - ab);  db += ab * ab;
        umax_num = std::max(umax_num, std::abs(nu_v));
        lo_u = std::min(lo_u, nu_v); hi_u = std::max(hi_u, nu_v);
        lo_b = std::min(lo_b, nb_v); hi_b = std::max(hi_b, nb_v);
      }
    sp_u = std::max(sp_u, hi_u - lo_u);
    sp_b = std::max(sp_b, hi_b - lo_b);
  }

  if (dump && *dump) {
    const std::string p = std::string("results/K_hartmann3d/") + dump + ".dat";
    std::FILE* f = std::fopen(p.c_str(), "w");
    if (f) {
      std::fprintf(f, "# 3-D Hartmann: ny=%d nx=%d nz=%d Ha=%g nu=%g Pr_m=%g\n",
                   int(ny), int(nx), int(nz), Ha, double(nu), prm);
      std::fprintf(f, "# B0=%.8e F=%.8e  delta=L/Ha=%.4f cells\n",
                   double(B0), double(F), mhd ? L / Ha : 0.0);
      std::fprintf(f, "# y/L  u_num  u_exact  b_num  b_exact\n");
      for (Index y = 0; y < ny; ++y) {
        const double xi = (double(y) - L) / L;
        std::fprintf(f, "%.8f %.10e %.10e %.10e %.10e\n", xi,
                     double(hu(d.id(nx / 2, y, nz / 2))), ana_u(xi),
                     double(hb(d.id(nx / 2, y, nz / 2))), ana_b(xi));
      }
      std::fclose(f);
    }
  }

  return {std::sqrt(su / std::max(du, 1e-300)),
          mhd ? std::sqrt(sb / std::max(db, 1e-300)) : 0.0,
          std::abs(double(hu(d.id(nx / 2, 0, nz / 2)))),
          mhd ? std::abs(double(hb(d.id(nx / 2, 0, nz / 2)))) : 0.0,
          sp_u / std::max(umax_ana, 1e-300),
          mhd ? sp_b / std::max(peak_b, 1e-300) : 0.0,
          spread_seed / std::max(umax_ana, 1e-300),
          umax_num, umax_ana,
          double(B0), double(F), double(nu), double(eta),
          1.0 / double(fc.omega), 1.0 / double(mc.omega),
          mhd ? L / Ha : 0.0,
          st.res, st.drift, taken, converged, finite};
}

//------------------------------------------------------------------------------
namespace {

void header(const char* what) {
  std::printf("\n%s\n%s\n", what, std::string(std::strlen(what), '=').c_str());
}

void row_header() {
  std::printf("  %5s %8s %10s %8s %10s %8s %10s %10s %9s %9s %8s\n",
              "ny", "L/Ha", "l2(u)", "order", "l2(b)", "order",
              "|u| wall", "|b| wall", "spread u", "spread b", "steps");
  std::printf("  %s\n", std::string(110, '-').c_str());
}

void table_row(const std::string& name, const char* header_text,
               const std::string& row) {
  const std::string path = "results/K_hartmann3d/" + name;
  bool fresh = true;
  if (std::FILE* t = std::fopen(path.c_str(), "r")) { fresh = false; std::fclose(t); }
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (!f) return;
  if (fresh) std::fputs(header_text, f);
  std::fputs(row.c_str(), f);
  std::fputc('\n', f);
  std::fclose(f);
}

const char* HDR =
    "# 3-D Hartmann flow (Dellar, moment-based magnetic boundary conditions).\n"
    "# D3Q27 MhdBGK fluid + D3Q7 MagneticBGK, Esoteric Pull, FP64.\n"
    "# Walls at y = 0 and y = ny-1, ON the node for BOTH families: regularised\n"
    "# velocity walls and Dellar's moment condition. Periodic in x and z.\n"
    "# B0 is normal to the walls; the force drives x. Ha = B0 L / sqrt(nu eta),\n"
    "# L = (ny-1)/2. Ha = 0 is the field-free parabola, B0 = 0.\n"
    "# Exact (Eq. 14): b = A[sinh(Ha xi)/sinh(Ha) - xi],\n"
    "#   u = A sqrt(eta/nu) coth(Ha)[1 - cosh(Ha xi)/cosh(Ha)], A = F L / B0.\n"
    "# l2 is relative and taken over EVERY node, so x or z structure is in it;\n"
    "# spread is the max range over (x,z) at fixed y, over the peak.\n"
    "# ha ny nx nz nu prm B0 F tau_f tau_m delta l2u l2b wall_u wall_b "
    "spread_u spread_b umax_num umax_ana spread_seed drift steps conv\n";

}  // namespace

int main(int argc, char** argv) {
  const double Ha_one = arg_num(argc, argv, "-ha", 10.0);
  const double nu_in  = arg_num(argc, argv, "-nu", 0.1);
  const double prm    = arg_num(argc, argv, "-prm", 1.0);
  const double umax   = arg_num(argc, argv, "-umax", 0.02);
  const Index  nx     = Index(arg_num(argc, argv, "-nx", 8));
  const Index  nz     = Index(arg_num(argc, argv, "-nz", 8));
  const double tol    = arg_num(argc, argv, "-tol", 1e-10);
  const std::size_t cap = std::size_t(arg_num(argc, argv, "-cap", 8000000));
  const bool eq2      = arg_flag(argc, argv, "-eq2");
  const bool dump     = arg_flag(argc, argv, "-dump");
  const bool hasweep  = arg_flag(argc, argv, "-hasweep");
  const Index nyfix   = Index(arg_num(argc, argv, "-ny", 0));
  const double pert   = arg_num(argc, argv, "-pert", 0.0);

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::printf("Hartmann flow in 3-D: D3Q27 fluid + D3Q7 magnetic field\n");
    std::printf("Dellar, moment-based magnetic boundary conditions, Eqs. (13)-(15)\n");
    std::printf("backend %s   precision %s   equilibrium %s\n",
                ExecSpace::name(), precision_name(),
                eq2 ? "second order" : "product form (highest D3Q27 admits)");
    std::printf("  nx = %d, nz = %d (both periodic); walls at y = 0 and y = ny-1\n",
                int(nx), int(nz));
    std::printf("  nu = %g, Pr_m = %g, u_max target = %g\n", nu_in, prm, umax);
    std::printf("  steady state on the EXTRAPOLATED DRIFT, tol %.0e\n", tol);

    auto once = [&](double Ha, Index ny, const char* tag) {
      return eq2 ? run<SecondOrderEquilibrium<FL>>(ny, nx, nz, Ha, nu_in, prm,
                                                   umax, tol, cap, 500, tag, pert)
                 : run<HighOrderEquilibrium<FL>>(ny, nx, nz, Ha, nu_in, prm,
                                                 umax, tol, cap, 500, tag, pert);
    };

    // ONE emitter for the table, taking nu as an argument. There were two --
    // this one and a copy inside the Ha = 0 block, which runs at two
    // viscosities -- and they drifted apart: the copy wrote spread_seed and
    // this one did not, so the file carried 23-field rows and 22-field rows
    // under a single 23-field header and every column past spread_seed was
    // shifted by one for half the data. A tracked table with two writers is
    // the defect; the missing column was only how it showed up.
    auto emit = [&](double Ha, Index ny, double nuv, const Result& r) {
      char row[640];
      std::snprintf(row, sizeof row,
          "%g %d %d %d %g %g %.8e %.8e %.6f %.6f %.4f %.6e %.6e %.3e %.3e "
          "%.3e %.3e %.8e %.8e %.3e %.3e %zu %d",
          Ha, int(ny), int(nx), int(nz), nuv, prm, r.B0, r.F, r.tau_f, r.tau_m,
          r.delta, r.eu, r.eb, r.wall_u, r.wall_b, r.spread_u, r.spread_b,
          r.umax_num, r.umax_ana, r.spread_seed, r.drift, r.steps,
          int(r.converged));
      table_row("hartmann3d.dat", HDR, row);
    };

    //--------------------------------------------------------------------------
    // A single resolution, when asked for one.
    if (nyfix > 0) {
      char tag[128];
      std::snprintf(tag, sizeof tag, "prof_ha%g_ny%d", Ha_one, int(nyfix));
      header("Single run");
      const Result r = once(Ha_one, nyfix, dump ? tag : "");
      std::printf("  Ha = %g   ny = %d   L/Ha = %.3f cells   B0 = %.4e   F = %.4e\n",
                  Ha_one, int(nyfix), r.delta, r.B0, r.F);
      std::printf("  tau_f = %.5f   tau_m = %.5f   (D3Q7 cs2 = 1/4)\n",
                  r.tau_f, r.tau_m);
      std::printf("  l2(u) = %.5e   l2(b) = %.5e\n", r.eu, r.eb);
      std::printf("  |u| wall = %.3e   |b| wall = %.3e   (imposed, so zero by "
                  "construction: this says the boundary is ON the node)\n",
                  r.wall_u, r.wall_b);
      if (pert > 0.0)
        std::printf("  transverse spread: seeded %.3e -> final %.3e   (decayed %.1fx)\n",
                    r.spread_seed, r.spread_u,
                    r.spread_u > 0 ? r.spread_seed / r.spread_u : INFINITY);
      else
        std::printf("  transverse spread: u %.3e   b %.3e   (zero by symmetry -- "
                    "use -pert to make this a measurement)\n",
                    r.spread_u, r.spread_b);
      std::printf("  u_max: %.8e numerical vs %.8e exact   (%+.3f%%)\n",
                  r.umax_num, r.umax_ana, 100.0 * (r.umax_num / r.umax_ana - 1.0));
      std::printf("  %zu steps, drift %.2e, %s%s\n", r.steps, r.drift,
                  r.converged ? "converged" : "NOT CONVERGED",
                  r.finite ? "" : ", DIVERGED");
      emit(Ha_one, nyfix, nu_in, r);
      Kokkos::finalize();
      return r.finite ? 0 : 1;
    }

    //--------------------------------------------------------------------------
    // The field-free row first: it separates the plumbing from the physics.
    header("Ha = 0: the field-free limit, against the parabola");
    std::printf("  With B0 = 0 the induction equation decouples and the exact answer\n");
    std::printf("  is F (L^2 - y^2) / 2 nu. This is NOT round-off at a general omega:\n");
    std::printf("  the regularised wall's curvature-induced slip is proportional to\n");
    std::printf("  (omega-1)/omega^2 and a parabola is all curvature. It vanishes at\n");
    std::printf("  omega = 1, i.e. nu = 1/6, which is the second block.\n\n");
    std::printf("  %5s %10s %10s %8s %10s %9s %8s\n",
                "ny", "nu", "l2(u)", "order", "|u| wall", "tau_f", "steps");
    std::printf("  %s\n", std::string(70, '-').c_str());
    for (double nuv : {nu_in, 1.0 / 6.0}) {
      double pu = 0; Index pn = 0;
      for (Index ny : {Index(17), Index(33), Index(65)}) {
        char tag[128];
        std::snprintf(tag, sizeof tag, "prof_ha0_nu%.4f_ny%d", nuv, int(ny));
        const Result r = eq2
            ? run<SecondOrderEquilibrium<FL>>(ny, nx, nz, 0.0, nuv, prm, umax,
                                              tol, cap, 500, dump ? tag : "", pert)
            : run<HighOrderEquilibrium<FL>>(ny, nx, nz, 0.0, nuv, prm, umax,
                                            tol, cap, 500, dump ? tag : "", pert);
        char ou[16] = "--";
        if (pn) std::snprintf(ou, sizeof ou, "%.3f",
                              std::log(pu / r.eu) /
                              std::log(double(ny - 1) / double(pn - 1)));
        std::printf("  %5d %10.5f %10.3e %8s %10.2e %9.5f %8zu\n",
                    int(ny), nuv, r.eu, ou, r.wall_u, r.tau_f, r.steps);
        pu = r.eu; pn = ny;
        emit(0.0, ny, nuv, r);
        if (!r.finite) status = 1;
      }
      if (nuv == nu_in) std::printf("  %s\n", std::string(70, '-').c_str());
    }
    std::printf("\n  the nu = 1/6 block is omega = 1 exactly, where the slip term\n"
                "  (omega-1)/omega^2 is identically zero.\n");

    //--------------------------------------------------------------------------
    header("Grid convergence at fixed Ha");
    std::printf("  Ha = %g. nu and the peak velocity are held fixed while L grows,\n"
                "  so B0 = Ha sqrt(nu eta)/L and F both shrink and the exact\n"
                "  profile keeps the same amplitude -- which is what makes the l2\n"
                "  norms comparable across the ladder.\n\n", Ha_one);
    row_header();
    {
      double pu = 0, pb = 0; Index pn = 0;
      for (Index ny : {Index(17), Index(33), Index(65), Index(129)}) {
        char tag[128];
        std::snprintf(tag, sizeof tag, "prof_ha%g_ny%d", Ha_one, int(ny));
        const Result r = once(Ha_one, ny, dump ? tag : "");
        char ou[16] = "--", ob[16] = "--";
        if (pn) {
          const double h = std::log(double(ny - 1) / double(pn - 1));
          std::snprintf(ou, sizeof ou, "%.3f", std::log(pu / r.eu) / h);
          std::snprintf(ob, sizeof ob, "%.3f", std::log(pb / r.eb) / h);
        }
        std::printf("  %5d %8.2f %10.3e %8s %10.3e %8s %10.2e %10.2e %9.2e %9.2e %8zu\n",
                    int(ny), r.delta, r.eu, ou, r.eb, ob,
                    r.wall_u, r.wall_b, r.spread_u, r.spread_b, r.steps);
        pu = r.eu; pb = r.eb; pn = ny;
        emit(Ha_one, ny, nu_in, r);
        if (!r.finite) status = 1;
      }
    }
    std::printf("\n  L/Ha is the Hartmann layer in cells. It is printed so that a\n"
                "  shortfall in the fitted order can be attributed to the ladder\n"
                "  rather than to the scheme -- and so that the absence of one is\n"
                "  visible too.\n");

    //--------------------------------------------------------------------------
    if (hasweep) {
      header("Hartmann number sweep at fixed resolution");
      std::printf("  ny = 129 throughout, so L = 64 and the layer thins as 64/Ha.\n\n");
      std::printf("  %6s %8s %10s %10s %10s %10s %9s %9s %8s\n",
                  "Ha", "L/Ha", "l2(u)", "l2(b)", "|u| wall", "|b| wall",
                  "spread u", "tau_m", "steps");
      std::printf("  %s\n", std::string(96, '-').c_str());
      for (double Ha : {1.0, 3.0, 10.0, 30.0, 100.0}) {
        char tag[128]; std::snprintf(tag, sizeof tag, "prof_ha%g_ny129", Ha);
        const Result r = once(Ha, 129, dump ? tag : "");
        std::printf("  %6.0f %8.2f %10.3e %10.3e %10.2e %10.2e %9.2e %9.5f %8zu\n",
                    Ha, r.delta, r.eu, r.eb, r.wall_u, r.wall_b, r.spread_u,
                    r.tau_m, r.steps);
        emit(Ha, 129, nu_in, r);
        if (!r.finite) status = 1;
      }
    }
  }
  Kokkos::finalize();
  return status;
}
