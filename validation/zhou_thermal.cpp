//==============================================================================
//  Zhou, De Rosis & Revell, Engineering with Computers 42:56 (2026),
//  "Coupling finite volume-lattice Boltzmann methods for advanced heat transfer
//  simulations" -- the benchmarks of its Section 3, run here on M3LB alone.
//
//  WHAT IS AND IS NOT BEING REPRODUCED. The paper's subject is a COUPLED
//  FVM-LBM framework: each domain is split into an LBM part, an FVM part and an
//  overlap, and most of its tables measure the coupling (overlap width, number
//  of LBM sub-iterations per FVM step, mass-flux continuity across an
//  interface). M3LB has no FVM side and no coupling, so none of that is
//  testable here and none of it is attempted. What IS testable is the physics
//  each case is posed to check, against the same external references the paper
//  uses -- an analytic solution where one exists, de Vahl Davis (1983),
//  Cheng & Liu (2010), Bettaibi et al. (2014), Fusegi et al. (1991).
//
//  So the comparison is three-cornered: this solver, the paper's coupled
//  result, and the reference the paper itself is measured against. Where the
//  paper quotes its own coupled number in a table, that number is printed for
//  context, NOT as the target -- a monolithic LBM has no coupling error to pay,
//  so beating it is expected and means nothing on its own. The reference column
//  is the one that matters.
//
//  Sections 3.3 (side-open cavity with porous blocks) and 3.7 (Rayleigh-Benard
//  with a melting boundary) are excluded by request. 3.3 needs a porous-media
//  drag model, 3.7 an enthalpy/phase-change formulation; neither exists here.
//
//  DISCRETISATION, fixed for every case below:
//
//    fluid       D3Q27, central moments (or BGK/MRT via -op), shifted storage,
//                Esoteric Pull, Guo forcing in the Boussinesq form
//    temperature D3Q7, BGK, Esoteric Pull                       [cs2 = 1/4]
//
//  D3Q27 is a product lattice, so a z-invariant field on nz = 1 with periodic z
//  reduces to D2Q9 exactly. D3Q7 does the same, and it is worth spelling out
//  because it is less obvious: its +z and -z populations then stream onto their
//  own node, so they never transport, and since c_z.u = 0 they relax to w_z dT
//  exactly as the rest population does. The seven-speed set therefore collapses
//  to a five-speed one with w0 = 1/4 + 2(1/8) = 1/2 and w_{x,y} = 1/8 -- whose
//  second moment is still 2(1/8) = 1/4 = cs2. A legitimate D2Q5, just not the
//  usual one, and with the SAME diffusivity calibration -- which case 3.4
//  then confirms rather than assumes: it recovers the exact advection-
//  diffusion profile at Pe = 1 to 3.6e-5 %, and Pe = u_y H / alpha would be
//  wrong by a constant factor if diffusivity_from_omega were reading the
//  wrong cs2 on a degenerate z. Nothing special is done for the 2-D cases;
//  they are 3-D runs one cell deep.
//
//  WALL CONVENTIONS ARE THE TRAP IN EVERY COUPLED CASE. Two families:
//
//    midway   fluid halfway bounce-back  + scalar anti-bounce-back
//             both planes at 0.5 and N-1.5, so the cavity is N-2 units wide
//    on-node  fluid regularised          + scalar Dellar moment
//             both planes on nodes 0 and N-1, so the cavity is N-1 wide
//
//  Mixing them puts the viscous and thermal boundaries half a spacing apart and
//  corrupts Ra or Gr silently -- the same argument as in
//  validation/rayleigh_benard.cpp. Cases 3.2 and 3.6 use the midway family
//  throughout. Case 3.5 has a MOVING lid, which needs the on-node family for
//  the fluid, so it uses the on-node family for the isothermal walls too; its
//  two adiabatic side walls are the one unavoidable exception, because there is
//  no on-node zero-flux condition in the scalar module and bounce-back puts
//  them at 0.5 and N-1.5. That makes the cavity N-1 tall and N-2 wide -- an
//  aspect-ratio error of 1/(N-1), so that case is run at two resolutions and
//  the gap between them carries this along with everything else that refines.
//
//  LATTICE UNITS. Every case is dimensionless and is pinned the way CLAUDE.md
//  prescribes: choose the resolution, choose one velocity, let the transport
//  coefficient follow, then read tau back. For the buoyant cases that velocity
//  is u_c = sqrt(g beta dT H), which gives nu = u_c H sqrt(Pr/Ra) = u_c H /
//  sqrt(Gr) -- so u_c is the knob that trades compressibility error against
//  step count, and it is printed with tau and the measured peak Mach number
//  every run.
//
//  CONVERGENCE, and why the obvious criterion is not enough. Steady state is
//  judged on the whole-field relative change over an INTERVAL, not per step:
//
//      res = || q(t+dt_probe) - q(t) ||_2 / || q(t+dt_probe) ||_2
//
//  over both u and T. A per-step residual measures the step size as much as the
//  drift and can be made small by shrinking dt.
//
//  But res is not the distance to the fixed point, and mistaking it for one is
//  a real error with a measured cost. Near steady state the interval change
//  decays geometrically, res_{k+1} ~ q res_k, so the change still to come is
//
//      drift = res q + res q^2 + ... = res q / (1 - q).
//
//  For a diffusive relaxation with a probe interval much shorter than the
//  diffusive time, q is within a percent of 1 and drift is a HUNDRED times res.
//  Stopping at res < 1e-6 in case 3.1 left the kappa = 1 profile -- whose exact
//  answer is a single straight line -- wrong in the fourth digit and reported as
//  converged. Declaring steady state on drift instead of res is what makes that
//  case return round-off. Every case below thresholds drift, both numbers are
//  printed, and q >= 1 falls back to drift = res so a run that is still growing
//  can never be called converged.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "FieldDump.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using FL = D3Q27;                  // fluid lattice, fixed
using SL = D3Q7;                   // temperature lattice, fixed
using SColl = ScalarBGK<SL>;

template <class L> using FBGK = BGK<L, HighOrderEquilibrium<L>, BoussinesqGuo, ShiftedPopulations>;
template <class L> using FMRT = MRT<L, BoussinesqGuo, ShiftedPopulations>;
template <class L> using FCM  = CentralMoments<L, BoussinesqGuo, ShiftedPopulations>;

namespace {

constexpr double PI = 3.14159265358979323846;

//------------------------------------------------------------------------------
// Least-squares straight line through (x_k, y_k). Returns slope and intercept.
//------------------------------------------------------------------------------
struct Line { double a, b; };
Line fit_line(const std::vector<double>& x, const std::vector<double>& y) {
  const double n = double(x.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t k = 0; k < x.size(); ++k) {
    sx += x[k]; sy += y[k]; sxx += x[k] * x[k]; sxy += x[k] * y[k];
  }
  const double det = n * sxx - sx * sx;
  return {(n * sxy - sx * sy) / det, (sxx * sy - sx * sxy) / det};
}

// Relative l2 error in percent, Eq. (38) of the paper read as a norm ratio.
double rel_l2_pct(const std::vector<double>& num, const std::vector<double>& ana) {
  double a = 0, b = 0;
  for (std::size_t k = 0; k < num.size(); ++k) {
    const double d = num[k] - ana[k];
    a += d * d; b += ana[k] * ana[k];
  }
  return 100.0 * std::sqrt(a / b);
}

// Interval residual of a host-side snapshot against the previous one.
double interval_residual(const std::vector<double>& now,
                         const std::vector<double>& before) {
  double a = 0, b = 0;
  for (std::size_t k = 0; k < now.size(); ++k) {
    const double d = now[k] - before[k];
    a += d * d; b += now[k] * now[k];
  }
  return (b > 0) ? std::sqrt(a / b) : 0.0;
}

//------------------------------------------------------------------------------
// Steady-state detector: the interval residual, plus the change still to come
// under a geometric extrapolation. See the CONVERGENCE note in the banner --
// the second is the one that is thresholded, and the difference between them is
// a factor of a hundred on a diffusive problem.
//------------------------------------------------------------------------------
struct Settle {
  double tol = 1e-6;
  double res = 1, prev = 1, drift = 1, ratio = 1;
  int seen = 0;
  bool update(double r) {
    prev = res; res = r; ++seen;
    ratio = (prev > 0) ? res / prev : 1.0;
    drift = (ratio < 1.0 && ratio > 0.0) ? res * ratio / (1.0 - ratio) : res;
    return seen >= 3 && drift < tol;      // three intervals before ratio means anything
  }
};

//------------------------------------------------------------------------------
// One-sided quadratic wall derivative. Nodes sit at s0 (the wall, where the
// value is known), s1 and s2; returns dT/ds at s0.
//------------------------------------------------------------------------------
double wall_gradient(double Tw, double T1, double T2, double h1, double h2) {
  return -((h1 + h2) / (h1 * h2)) * Tw + (h2 / (h1 * (h2 - h1))) * T1
         - (h1 / (h2 * (h2 - h1))) * T2;
}

//------------------------------------------------------------------------------
// Append one row to a tracked table under results/, writing the header first if
// the file is new. A sweep is several invocations of this executable, so the
// table has to be built by appending rather than by parsing stdout afterwards --
// which is what the first attempt did, and what a changed printf silently
// breaks.
//------------------------------------------------------------------------------
void table_row(const std::string& name, const char* header, const std::string& row) {
  const std::string path = "results/J_zhou_thermal/" + name;
  bool fresh = true;
  if (std::FILE* t = std::fopen(path.c_str(), "r")) { fresh = false; std::fclose(t); }
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (!f) return;
  if (fresh) std::fputs(header, f);
  std::fputs(row.c_str(), f);
  std::fputc('\n', f);
  std::fclose(f);
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

//==============================================================================
//  3.1  ONE-DIMENSIONAL CONJUGATE HEAT CONDUCTION
//
//  Domain of length L = 1, solid over 0 <= x <= 0.2, fluid over 0.2 < x <= 1,
//  theta = 1 at x = 0 and theta = 0 at x = 1, periodic elsewhere. The
//  conductivity ratio kappa = lambda_s / lambda_f is varied over 0.1, 1, 10,
//  100. With rho c_p uniform -- which is what the paper's Eq. (36) assumes,
//  writing the balance in terms of alpha -- kappa is also the diffusivity ratio,
//  so a per-node relaxation rate is the whole model. Steady state:
//
//      theta = a_s x + 1              (0 <= x <= Ls)
//      theta = kappa a_s x + b_f      (Ls < x <= 1)
//      a_s = -1 / (Ls + (1 - Ls) kappa),  b_f = 1 + a_s Ls (1 - kappa)
//
//  WHY THIS IS A SHARPER TEST THAN THE PERCENTAGE SUGGESTS. At kappa = 1 the
//  answer is a single straight line, the relaxation rate is uniform, and both
//  anti-bounce-back walls land their value exactly (validation/scalar_walls.cpp
//  measures where: 0.0 and N, to fourteen digits). So kappa = 1 must come back
//  at round-off, and any error there is a bug rather than a discretisation
//  cost. Every nonzero error below therefore belongs to the INTERFACE, and is
//  reported as such: each branch is fitted, the two slopes are checked against
//  a_s and kappa a_s, and the x where the fitted lines cross is compared with
//  0.2. That offset -- in lattice units, and how it scales with N -- is the
//  actual property of the scheme. The percentage is reported too because it is
//  what the paper's Table 3 quotes.
//
//  TAU SPLIT. kappa is imposed as D_s = D0 sqrt(kappa), D_f = D0 / sqrt(kappa),
//  a geometric split about D0. Putting the ratio entirely on one side would
//  drive one tau either to the 1/2 floor (making the run cost L^2 / D_min
//  steps, 100x more at kappa = 100) or to tens (where the wall and interface
//  closures, whose error grows as (1/omega - 1/2)^2, stop being accurate). The
//  split keeps both taus inside a factor sqrt(kappa) of D0 and is stated here
//  because it is a choice, not a property of the problem: -d0 changes it and
//  the tau pair is printed every run.
//==============================================================================
namespace zt {

struct Cond {
  double eps_pct;                  // Eq. (38), percent
  double slope_s, slope_f;         // fitted branch slopes
  double x_int;                    // fitted interface position
  double tau_s, tau_f;
  double res, drift;
  std::size_t steps;
  bool converged;
};

Cond conduction(Index L, double kappa, double D0, double tol,
                std::size_t cap, std::size_t probe, const char* dump) {
  const Index Ls = Index(std::lround(0.2 * double(L)));   // solid cells
  const Index nx = L + 2, ny = 5, nz = 1;

  const Real Df = Real(D0 / std::sqrt(kappa));
  const Real Ds = Real(D0 * std::sqrt(kappa));
  const Real wf = SColl::omega_from_diffusivity(Df);
  const Real ws = SColl::omega_from_diffusivity(Ds);

  Domain d(nx, ny, nz, /*periodic x*/ false, /*y*/ true, /*z*/ true);

  SColl coll;
  coll.omega = wf;
  coll.T_ref = Real(0.5);
  // Per-node relaxation. Filled everywhere, halo included: a partially filled
  // omega field is a zero-diffusivity bug, not a supported state.
  View1D<Real> wof("omega_of", d.n_padded);
  {
    auto h = Kokkos::create_mirror_view(wof);
    for (Index n = 0; n < d.n_padded; ++n) h(n) = wf;
    for (Index z = 0; z < nz; ++z)
      for (Index y = 0; y < ny; ++y)
        for (Index x = 0; x <= Ls; ++x) h(d.id(x, y, z)) = ws;
    Kokkos::deep_copy(wof, h);
  }
  coll.omega_of = wof;

  ScalarSolver<SL, EsotericPull<SL>, SColl> th(d, coll);
  th.set_geometry([&](Index x, Index, Index) -> ScalarCell {
    if (x == 0 || x == nx - 1) return ScalarDirichlet;
    return ScalarBulk;
  });
  th.set_wall_values([&](Index x, Index, Index) -> Real {
    return (x == 0) ? Real(1) : Real(0);
  });
  th.finalize_geometry();
  // Uniform mid-temperature start. The fixed point is unique, so the seed only
  // buys steps; a warm start from the kappa = 1 line would buy more and is
  // deliberately not used, so the reported step count is the honest one.
  th.initialize(Real(0.5));

  const double Lc = double(L);
  auto Xof = [&](Index x) { return (double(x) - 0.5) / Lc; };

  const double a_s = -1.0 / (0.2 + 0.8 * kappa);
  const double a_f = kappa * a_s;
  const double b_f = 1.0 + a_s * 0.2 * (1.0 - kappa);
  auto ana = [&](double X) { return (X <= 0.2) ? a_s * X + 1.0 : a_f * X + b_f; };

  std::vector<double> now, before;
  auto snapshot = [&]() {
    th.compute_field();
    auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    std::vector<double> v(std::size_t(nx - 2));
    for (Index x = 1; x <= nx - 2; ++x) v[std::size_t(x - 1)] = double(hT(d.id(x, 2, 0)));
    return v;
  };

  Settle st{tol};
  std::size_t taken = 0;
  bool converged = false;
  before = snapshot();
  while (taken < cap) {
    for (std::size_t k = 0; k < probe; ++k) th.step();
    taken += probe;
    now = snapshot();
    converged = st.update(interval_residual(now, before));
    before = now;
    if (converged) break;
  }

  // Branch fits. The two nodes flanking the interface are excluded: whatever the
  // interface closure does wrong, it does within one cell of the interface, and
  // including those points would fold that error into the slope instead of
  // leaving it in the crossing point where it can be read.
  std::vector<double> xs, ys, xf, yf, num, ref;
  for (Index x = 1; x <= nx - 2; ++x) {
    const double X = Xof(x), T = now[std::size_t(x - 1)];
    num.push_back(T); ref.push_back(ana(X));
    if (x <= Ls - 1)      { xs.push_back(X); ys.push_back(T); }
    else if (x >= Ls + 2) { xf.push_back(X); yf.push_back(T); }
  }
  const Line fs = fit_line(xs, ys), ff = fit_line(xf, yf);
  const double x_int = (ff.b - fs.b) / (fs.a - ff.a);

  if (dump && *dump) {
    std::string p = std::string("results/J_zhou_thermal/") + dump + ".dat";
    std::FILE* f = std::fopen(p.c_str(), "w");
    if (f) {
      std::fprintf(f, "# 3.1 conjugate conduction: L=%d kappa=%g D0=%g\n",
                   int(L), kappa, D0);
      std::fprintf(f, "# x/L  theta_num  theta_ana\n");
      for (Index x = 1; x <= nx - 2; ++x)
        std::fprintf(f, "%.8f %.10e %.10e\n", Xof(x),
                     num[std::size_t(x - 1)], ref[std::size_t(x - 1)]);
      std::fclose(f);
    }
  }

  return {rel_l2_pct(num, ref), fs.a, ff.a, x_int,
          1.0 / double(ws), 1.0 / double(wf), st.res, st.drift, taken, converged};
}

}  // namespace zt

//==============================================================================
//  3.4  NORMAL PLATE VELOCITY  (one-dimensional advection-diffusion)
//
//  A uniform vertical stream u_y crosses the layer, injected at the bottom and
//  withdrawn at the top; the bottom is cold (theta = 0) and the top hot
//  (theta = 1), and the flanks are periodic. Steady state is
//
//      theta(Y) = [exp(Pe Y) - 1] / [exp(Pe) - 1],     Pe = Re Pr = u_y H / alpha
//
//  the paper's Eq. (44). The name comes from the companion fact that u_x obeys
//  the same equation with Pe replaced by Re, so the moving lid's momentum
//  profile has the identical analytic form -- checked separately by -coupled.
//
//  ONLY Pe MATTERS for the temperature, which is why the three cases the paper
//  labels Pr = 0.1, 1, 10 at Re = 10 are run here as Pe = 1, 10, 100. Nothing
//  in the scalar module can see Re or Pr apart from their product.
//
//  WHAT IS HELD FIXED, AND WHY IT MATTERS. The pair (D, u_y) is not free: Pe
//  ties them to H. Fixing u_y across the three cases would sweep tau from 8.5
//  to 0.51 and the comparison would be a tau study wearing a Pe label, so tau
//  is fixed instead and u_y follows.
//
//  That was expected to cost something. This equilibrium is only first order in
//  u, so a defect growing as u^2 -- and therefore as Pe -- looked like the
//  obvious suspect for the error at Pe = 100. It is not there: -uscan reruns
//  Pe = 100 at tau = 0.8, 0.65, 0.575, a factor of four in u_y and in D at
//  fixed u_y/D, and the error does not move in five significant figures
//  (2.3313 % three times over). At fixed Pe the steady discrete solution is a
//  function of u_y/D alone, so the tau choice above costs nothing and every bit
//  of the remaining error is grid resolution of the boundary layer -- which is
//  what the -n ladder then confirms, at order 2.00 from H = 200 to H = 1600.
//
//  BOTH WALL FAMILIES ARE RUN. Anti-bounce-back is second-order for pure
//  diffusion but its derivation assumes no normal velocity through the wall,
//  which is exactly what this case has; Dellar's moment condition puts the
//  value on the node and makes no such assumption. Whether that shows up at
//  Pe = 100 is a measurement, not an argument, so the table carries both.
//==============================================================================
namespace zt {

struct Plate {
  double eps_pct, tau, U, D;
  double res, drift;
  std::size_t steps;
  bool converged;
};

Plate plate(Index H, double Pe, double tau, bool on_node, double tol,
            std::size_t cap, std::size_t probe, const char* dump) {
  const Index nx = 5, ny = on_node ? (H + 1) : (H + 2), nz = 1;
  const Real D = SColl::diffusivity_from_omega(Real(1.0 / tau));
  const Real U = Real(Pe * double(D) / double(H));

  Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  SColl coll;
  coll.omega = Real(1.0 / tau);
  coll.T_ref = Real(0.5);
  ScalarSolver<SL, EsotericPull<SL>, SColl> th(d, coll);
  th.set_geometry([&](Index, Index y, Index) -> ScalarCell {
    if (y == 0 || y == ny - 1) return on_node ? ScalarMoment : ScalarDirichlet;
    return ScalarBulk;
  });
  th.set_wall_values([&](Index, Index y, Index) -> Real {
    return (y == 0) ? Real(0) : Real(1);          // cold below, hot above
  });
  th.finalize_geometry();
  th.initialize(Real(0.5));

  View1D<Real> vx("vx", d.n_padded), vy("vy", d.n_padded), vz("vz", d.n_padded);
  Kokkos::deep_copy(vy, U);                       // uniform upward stream
  th.set_velocity(vx, vy, vz);

  const double y0 = on_node ? 0.0 : 0.5;
  const Index yf0 = 1, yf1 = ny - 2;
  auto Yof = [&](Index y) { return (double(y) - y0) / double(H); };
  auto ana = [&](double Y) { return (std::exp(Pe * Y) - 1.0) / (std::exp(Pe) - 1.0); };

  auto snapshot = [&]() {
    th.compute_field();
    auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    std::vector<double> v;
    for (Index y = yf0; y <= yf1; ++y) v.push_back(double(hT(d.id(2, y, 0))));
    return v;
  };

  std::vector<double> now, before = snapshot();
  Settle st{tol}; std::size_t taken = 0; bool converged = false;
  while (taken < cap) {
    for (std::size_t k = 0; k < probe; ++k) th.step();
    taken += probe;
    now = snapshot();
    converged = st.update(interval_residual(now, before));
    before = now;
    if (converged) break;
  }

  std::vector<double> ref;
  for (Index y = yf0; y <= yf1; ++y) ref.push_back(ana(Yof(y)));

  if (dump && *dump) {
    std::string p = std::string("results/J_zhou_thermal/") + dump + ".dat";
    std::FILE* f = std::fopen(p.c_str(), "w");
    if (f) {
      std::fprintf(f, "# 3.4 normal plate velocity: H=%d Pe=%g tau=%g walls=%s\n",
                   int(H), Pe, tau, on_node ? "moment" : "abb");
      std::fprintf(f, "# y/H  theta_num  theta_ana\n");
      for (std::size_t k = 0; k < now.size(); ++k)
        std::fprintf(f, "%.8f %.10e %.10e\n", Yof(Index(k) + yf0), now[k], ref[k]);
      std::fclose(f);
    }
  }
  return {rel_l2_pct(now, ref), tau, double(U), double(D), st.res, st.drift,
          taken, converged};
}

}  // namespace zt

//==============================================================================
//  3.2 / 3.6  NATURAL CONVECTION IN A DIFFERENTIALLY HEATED CAVITY
//
//  One routine for both, because they are the same problem one cell deep and N
//  cells deep: hot at x = 0, cold at x = L, every other wall adiabatic and
//  no-slip, gravity along -y. The controlling groups are
//
//      Ra = g beta dT H^3 / (nu alpha),      Pr = nu / alpha = 0.71.
//
//  Fixing u_c = sqrt(g beta dT H) gives nu = u_c H sqrt(Pr/Ra) and alpha = nu/Pr,
//  and the buoyancy coefficient follows as g beta = u_c^2 / H with dT = 1. The
//  reported peak velocities are non-dimensionalised by alpha/H, which is de Vahl
//  Davis's convention and NOT the u_c used to set the run up -- so the printed
//  Mach number is the only place the lattice velocity is visible.
//
//  THE HOT/COLD NUSSELT DIFFERENCE IS NOT A CONVERGENCE CHECK HERE, and the
//  first version of this file used it as one. This configuration is invariant
//  under (x,y) -> (L-x, L-y) with T -> -T and u -> -u: the boundary conditions,
//  D3Q27, D3Q7, bounce-back and anti-bounce-back all respect that map, and the
//  initial state T = 0, u = 0 is a fixed point of it. So the discrete solution
//  is centro-symmetric at EVERY step, the two wall Nusselt numbers are equal
//  identically, and their difference reads 0.000 % on a run that has barely
//  started. It is still printed -- as a check that the symmetry is intact, which
//  is worth something -- but it says nothing about convergence.
//
//  The independent estimator is the volume one. At steady state the horizontal
//  heat flux through every vertical plane is the same, and averaging it over x
//  gives an identity that involves no wall gradient at all:
//
//      Nu_vol = 1 + <u_x theta> H / alpha,
//
//  the average over the fluid volume, with <u_x> = 0 in a closed cavity so the
//  temperature gauge drops out. It shares no arithmetic with the wall stencil,
//  and the two agreeing to three digits is a real statement about the run.
//  Case 3.5, by contrast, has a moving lid that breaks the symmetry, so there
//  the two wall values ARE independent -- and they differ by 30 % mid-transient.
//
//  The wall plane sits at x = 0.5, midway between the anti-bounce-back node at
//  x = 0 and the first fluid node at x = 1, so the one-sided derivative uses
//  unequal spacings 0.5 and 1.5 and the cavity is H = N - 2 units wide. The
//  extremum locations are reported as (y - 0.5)/H, which cannot reach 0 or 1 --
//  the nearest sample to a corner is half a cell away. Both the paper and
//  Luan et al. report (y/L)_min = 0.997 rather than 1.000 for the same reason;
//  de Vahl Davis's 1.000 comes from a grid with a node ON the wall.
//==============================================================================
namespace zt {

struct Conv {
  double nu_hot, nu_cold;                 // averaged Nusselt, both walls
  double nu_vol, ux_mean;                 // volume estimator, and its premise
  double nu_max, y_max, nu_min, y_min;    // on the hot wall
  double umax, y_umax, vmax, x_vmax;      // scaled by alpha/H
  double nu_lat, tau_f, tau_t, ma;
  double res, drift;
  std::size_t steps;
  bool converged, finite;
};

template <class FColl>
Conv natural(Index N, Index Nz, double Ra, double Pr, double uc,
             double tol, std::size_t cap, std::size_t probe,
             const char* dump, FColl fcoll_proto, bool verbose = false) {
  const bool three_d = Nz > 1;
  const Index H = N - 2;                              // fluid nodes across
  const Real nu = Real(double(H) * uc * std::sqrt(Pr / Ra));
  const Real D  = Real(double(nu) / Pr);
  const Real gb = Real(uc * uc / double(H));          // g*beta, dT = 1

  Domain d(N, N, Nz, /*periodic x*/ false, /*y*/ false, /*z*/ !three_d);

  SColl scoll;
  scoll.omega = SColl::omega_from_diffusivity(D);
  scoll.T_ref = Real(0);                              // symmetric about T = 0
  ScalarSolver<SL, EsotericPull<SL>, SColl> th(d, scoll);
  th.set_geometry([&](Index x, Index y, Index z) -> ScalarCell {
    if (x == 0 || x == N - 1) return ScalarDirichlet;
    if (y == 0 || y == N - 1) return ScalarAdiabatic;
    if (three_d && (z == 0 || z == Nz - 1)) return ScalarAdiabatic;
    return ScalarBulk;
  });
  th.set_wall_values([&](Index x, Index, Index) -> Real {
    return (x == 0) ? Real(0.5) : Real(-0.5);
  });
  th.finalize_geometry();
  th.initialize(Real(0));
  th.compute_field();

  BoussinesqGuo force;
  force.T = th.temperature();
  force.gx = Real(0); force.gy = Real(1); force.gz = Real(0);
  force.rho0 = Real(1); force.beta = gb; force.T0 = Real(0);

  FColl fcoll = fcoll_proto;
  fcoll.omega = FColl::omega_from_viscosity(nu);
  fcoll.forcing = force;

  FluidSolver<FL, EsotericPull<FL>, FColl> fl(d, fcoll);
  fl.set_geometry([&](Index x, Index y, Index z) -> CellType {
    if (x == 0 || x == N - 1 || y == 0 || y == N - 1) return Solid;
    if (three_d && (z == 0 || z == Nz - 1)) return Solid;
    return Fluid;
  });
  fl.initialize(Real(1));
  th.set_velocity(fl.ux(), fl.uy(), fl.uz());

  const Index zm = three_d ? Nz / 2 : 0;
  const double scale = double(H) / double(D);

  // Nusselt on a wall: quadratic one-sided gradient, averaged over the wall.
  auto nusselt = [&](bool hot, std::vector<double>* local) {
    auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    const double Tw = hot ? 0.5 : -0.5;
    const Index i1 = hot ? 1 : N - 2, i2 = hot ? 2 : N - 3;
    const double sgn = hot ? 1.0 : -1.0;      // outward-x for the cold wall
    double acc = 0; Index cnt = 0;
    const Index z0 = three_d ? 1 : 0, z1 = three_d ? Nz - 2 : 0;
    if (local) local->clear();
    for (Index z = z0; z <= z1; ++z)
      for (Index y = 1; y <= H; ++y) {
        const double g = wall_gradient(Tw, double(hT(d.id(i1, y, z))),
                                       double(hT(d.id(i2, y, z))), 0.5, 1.5);
        const double lv = -sgn * g * double(H);
        acc += lv; ++cnt;
        if (local && z == zm) local->push_back(lv);
      }
    return acc / double(cnt);
  };

  // Interval residual on u and T together, sampled on the mid-z plane.
  auto snapshot = [&]() {
    fl.compute_macroscopic();
    th.compute_field();
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    std::vector<double> v;
    for (Index y = 1; y <= H; ++y)
      for (Index x = 1; x <= H; ++x) {
        const Index n = d.id(x, y, zm);
        v.push_back(double(hx(n)) * scale);
        v.push_back(double(hy(n)) * scale);
        v.push_back(double(hT(n)));
      }
    return v;
  };

  std::vector<double> now, before = snapshot();
  Settle st{tol}; std::size_t taken = 0; bool converged = false, finite = true;
  while (taken < cap) {
    for (std::size_t k = 0; k < probe; ++k) { fl.step(true); th.step(); }
    taken += probe;
    now = snapshot();
    for (double v : now) if (!std::isfinite(v)) finite = false;
    if (!finite) break;
    converged = st.update(interval_residual(now, before));
    before = now;
    if (verbose) {
      std::printf("    t = %-9zu res %.3e  drift %.3e  Nu_hot %.5f  Nu_cold %.5f\n",
                  taken, st.res, st.drift, nusselt(true, nullptr),
                  nusselt(false, nullptr));
      std::fflush(stdout);           // a long run is watched from another shell
    }
    if (converged) break;
  }

  std::vector<double> nu_local;
  const double nu_hot  = nusselt(true, &nu_local);
  const double nu_cold = nusselt(false, nullptr);

  // Volume estimator: Nu = 1 + <u_x theta> H / alpha. <u_x> is reported with it
  // because the identity needs <u_x> = 0 -- if the cavity is not closed to
  // machine precision the number is meaningless rather than merely inaccurate.
  double nu_vol = 0, ux_mean = 0;
  {
    fl.compute_macroscopic();
    th.compute_field();
    auto ax = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    auto aT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    const Index z0 = three_d ? 1 : 0, z1 = three_d ? Nz - 2 : 0;
    double acc = 0, accu = 0; Index cnt = 0;
    for (Index z = z0; z <= z1; ++z)
      for (Index y = 1; y <= H; ++y)
        for (Index x = 1; x <= H; ++x) {
          const Index n = d.id(x, y, z);
          acc  += double(ax(n)) * double(aT(n));
          accu += double(ax(n));
          ++cnt;
        }
    nu_vol  = 1.0 + (acc / double(cnt)) * double(H) / double(D);
    ux_mean = (accu / double(cnt)) * double(H) / double(D);
  }

  double nmax = -1e300, nmin = 1e300, ymax = 0, ymin = 0;
  for (std::size_t k = 0; k < nu_local.size(); ++k) {
    const double yy = (double(k) + 1.0 - 0.5) / double(H);
    if (nu_local[k] > nmax) { nmax = nu_local[k]; ymax = yy; }
    if (nu_local[k] < nmin) { nmin = nu_local[k]; ymin = yy; }
  }

  fl.compute_macroscopic();
  auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto hz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uz());
  auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());

  // The mid-plane falls between nodes when H is even; interpolate rather than
  // rounding, so that H even and H odd are the same measurement.
  const double mc = 0.5 * double(H) + 0.5;
  const Index m0 = Index(std::floor(mc)); const double mw = mc - double(m0);
  auto lerp_x = [&](const auto& f, Index y) {
    return (1.0 - mw) * double(f(d.id(m0, y, zm))) + mw * double(f(d.id(m0 + 1, y, zm)));
  };
  auto lerp_y = [&](const auto& f, Index x) {
    return (1.0 - mw) * double(f(d.id(x, m0, zm))) + mw * double(f(d.id(x, m0 + 1, zm)));
  };

  // The PLAIN maximum, not the largest magnitude. The two peaks on a centreline
  // are nearly equal and opposite, so an absolute-value search flips sign
  // between resolutions -- it did, between N = 66 and N = 98 -- and then the
  // location no longer matches de Vahl Davis, who tabulates the positive one.
  double umax = -1e300, yu = 0, vmax = -1e300, xv = 0, ma = 0;
  for (Index y = 1; y <= H; ++y) {
    const double u = lerp_x(hx, y) * scale;
    if (u > umax) { umax = u; yu = (double(y) - 0.5) / double(H); }
  }
  for (Index x = 1; x <= H; ++x) {
    const double v = lerp_y(hy, x) * scale;
    if (v > vmax) { vmax = v; xv = (double(x) - 0.5) / double(H); }
  }
  {
    const Index z0 = three_d ? 1 : 0, z1 = three_d ? Nz - 2 : 0;
    for (Index z = z0; z <= z1; ++z)
      for (Index y = 1; y <= H; ++y)
        for (Index x = 1; x <= H; ++x) {
          const Index n = d.id(x, y, z);
          ma = std::max(ma, std::sqrt(double(hx(n)) * double(hx(n)) +
                                      double(hy(n)) * double(hy(n)) +
                                      double(hz(n)) * double(hz(n))));
        }
    ma /= std::sqrt(1.0 / 3.0);
  }

  if (dump && *dump) {
    const std::string base = std::string("results/J_zhou_thermal/") + dump;
    std::FILE* f = std::fopen((base + "_centrelines.dat").c_str(), "w");
    if (f) {
      std::fprintf(f, "# natural convection Ra=%g Pr=%g N=%d Nz=%d op=%s\n",
                   Ra, Pr, int(N), int(Nz), FColl::name);
      std::fprintf(f, "# s  theta(x=s,y=0.5)  theta(x=0.5,y=s)  "
                      "u_x(x=0.5,y=s)*H/a  u_y(x=s,y=0.5)*H/a  Nu_local(y=s)\n");
      for (Index k = 1; k <= H; ++k) {
        const double s = (double(k) - 0.5) / double(H);
        std::fprintf(f, "%.8f %.8e %.8e %.8e %.8e %.8e\n", s,
                     lerp_y(hT, k), lerp_x(hT, k),
                     lerp_x(hx, k) * scale, lerp_y(hy, k) * scale,
                     nu_local[std::size_t(k - 1)]);
      }
      std::fclose(f);
    }
    figdump::scalar_slice(base + "_temp.dat", N, N,
        [&](Index x, Index y) { return float(hT(d.id(x, y, zm))); });
    // u and v separately, not just the speed: the reference figures are
    // streamline plots and a speed field cannot be integrated into one.
    figdump::scalar_slice(base + "_ux.dat", N, N,
        [&](Index x, Index y) { return float(double(hx(d.id(x, y, zm))) * scale); });
    figdump::scalar_slice(base + "_uy.dat", N, N,
        [&](Index x, Index y) { return float(double(hy(d.id(x, y, zm))) * scale); });
  }

  return {nu_hot, nu_cold, nu_vol, ux_mean,
          nmax, ymax, nmin, ymin, umax, yu, vmax, xv,
          double(nu), 1.0 / double(FColl::omega_from_viscosity(nu)),
          1.0 / double(scoll.omega), ma, st.res, st.drift, taken, converged, finite};
}

}  // namespace zt

//==============================================================================
//  3.5  THERMAL LID-DRIVEN CAVITY  (mixed convection)
//
//  A square cavity, lid sliding right at U, the other three walls no-slip; the
//  BOTTOM is hot (theta = 1) and the TOP -- the moving lid -- is cold
//  (theta = 0), so the temperature gradient opposes gravity and the layer is
//  unstably stratified. The flanks are adiabatic. Four groups, three
//  independent:
//
//      Pr = nu / alpha = 0.71,   Gr = g beta dT L^3 / nu^2 = 1e6,
//      Re = U L / nu,            Ri = Gr / Re^2.
//
//  Ri is the knob: Ri = 10, 1, 0.1 give Re = 316.2, 1000, 3162.3. Setting
//  u_c = sqrt(g beta dT L) fixes nu = u_c L / sqrt(Gr), and then the lid speed
//  is not free -- U = u_c / sqrt(Ri). That is the whole difficulty of this case:
//  at Ri = 0.1 the lid must run 3.16x the buoyancy scale, so u_c has to come
//  DOWN to keep the lid subsonic, and tau = 1/2 + 3 u_c L/1000 comes down with
//  it. The two constraints pull opposite ways and the largest u_c each Ri
//  tolerates is also the cheapest, since <theta> relaxes on L^2/(alpha Nu) and
//  alpha grows with u_c -- so u_c is chosen per Ri (0.15, 0.08, 0.03) rather
//  than held fixed, and the pair (u_c, tau, Ma) is printed for every run.
//
//  WALL FAMILY, AND A MISMATCH THAT DID MUCH WORSE THAN SPOIL AN ASPECT RATIO.
//  A moving wall needs the regularised velocity condition, which sits ON the
//  node, so the isothermal walls use Dellar's moment condition and the cavity is
//  L = N - 1 tall. The obvious choice for the two adiabatic flanks was
//  ScalarAdiabatic -- bounce-back -- and it was wrong for a reason that took a
//  measurement to find.
//
//  Bounce-back puts the insulated plane at 0.5, which makes the boundary node a
//  GHOST outside the fluid, and ScalarSolver's field_kernel says so: it writes
//  field(n) = 0 at an adiabatic node, because a node outside the fluid has no
//  temperature. In cases 3.2 and 3.6 that costs nothing, since those boundary
//  nodes are Solid for the fluid and never collide. Here they are RegWall nodes
//  -- fluid nodes, on the boundary, which DO collide and DO get forced. The
//  Boussinesq force read T = 0 against T0 = 0.5 and applied -0.5 g beta down
//  both side walls, a spurious body force on two whole columns.
//
//  The symptom was the hot/cold Nusselt imbalance, which the moving lid makes an
//  honest check here (unlike case 3.2, where centro-symmetry makes it vacuous):
//  Nu = 7.53 at the hot wall against 7.82 at the cold one, a 3.9 % gap that
//  conservation forbids at steady state, sitting there while the fluctuation was
//  only 0.3 %.
//
//  Two things were changed. The gauge is now symmetric about zero, so a node
//  reporting field = 0 is neutrally buoyant instead of maximally cold, and the
//  spurious force is gone whatever the flank condition is. And -flank selects
//  the flank treatment, because neither candidate is right and the choice has to
//  be measured rather than argued:
//
//    abb      ScalarAdiabatic, bounce-back. Conservative and second order, but
//             the node is a ghost at 0.5, so its buoyancy force is zero rather
//             than the local value, and the cavity is N-1 by N-2.
//    outflow  ScalarOutflow. On-node, zero gradient, field(n) is the real
//             temperature so the force is right, and the cavity is N-1 square --
//             but the node is FULLY PRESCRIBED, so whatever streams in is
//             discarded and the condition does not conserve heat.
//
//  The hot/cold Nusselt imbalance and the domain-mean temperature separate them,
//  and the answer was that neither flank leaks. What looked like a leak was the
//  mean temperature still falling:
//
//      d<theta>/dt = (Nu_hot - Nu_cold) alpha / L^2,
//
//  which at N = 65 predicted -0.128 against a measured imbalance of -0.111 --
//  agreement to 15 %, so the imbalance IS the drift and nothing is being
//  destroyed at the walls. -flank stays because the two treatments are not
//  equivalent, but the choice is second order beside what follows.
//
//  WHICH MEANS THE CONVERGENCE CRITERION HAS TO CHANGE FOR THIS CASE. The
//  whole-field residual cannot see the slow drift: it is dominated by the
//  velocity field, which has already settled, so it floors while <theta> is
//  still walking. After 190000 steps at N = 65 it read O(1e-2) with <theta> at
//  0.386 and moving.
//
//  So the lid case declares steady state on the heat budget itself:
//  |Nu_hot - Nu_cold| / Nu below -imbtol for three consecutive probes. That is a
//  statement about the heat budget rather than about the fields, it is immune to
//  the oscillation, and by the identity above it is exactly the statement that
//  <theta> has stopped moving.
//
//  NUSSELT is taken on the hot bottom wall, where the value is ON the node, so
//  the one-sided gradient uses equal spacings 1 and 2 -- a different stencil
//  from the natural-convection case above, and one that is exact for a quadratic.
//
//  THIS CASE DOES REACH A STEADY STATE -- eventually. The converged runs have a
//  Nusselt fluctuation of 0.05 % to 0.5 % of the mean over the averaging window
//  and streamline patterns that match the paper's Fig. 23 feature by feature.
//  But getting there is slow for a reason that has nothing to do with the flow,
//  and every wrong answer this case produced came from stopping early.
//
//  Two intermediate claims were written into this banner and are recorded here
//  because both were WRONG, and neither the field residual nor the eye caught
//  them. The first was that the case has no steady state, argued from a residual
//  that floored at O(1e-2) with a Nusselt number near 9.7 at Ri = 1. The second
//  was that the buoyancy-dominated cases disagree with Cheng & Liu (2010) by
//  some 44 %, argued from Nu = 7.0 at Ri = 10 against their 4.860. Both came
//  from runs whose DOMAIN-MEAN TEMPERATURE had not equilibrated. Once it has,
//  Ri = 1 gives 5.749 against their 5.750, and Ri = 10 gives 4.689 against
//  4.860.
//
//  The mechanism is worth stating plainly because it is general. <theta> relaxes
//  on L^2/(alpha Nu) ~ t_diff/Nu -- 1e5 steps at N = 65, 4e5 at N = 129 -- while
//  the velocity field reaches its own quasi-equilibrium in a few thousand. In
//  between, the flow looks converged and IS converged, conditionally on a core
//  temperature that is still wrong; and a core 0.08 too cold inflates the
//  hot-wall gradient, which is exactly how Nu = 7 survived 130000 steps at
//  Ri = 10 before falling back to 4.7 as <theta> climbed from 0.38 to 0.46.
//==============================================================================
namespace zt {

struct Lid {
  double nu_hot, nu_cold;                 // time-averaged over the last navg probes
  double nu_rms, nu_span;                 // fluctuation, and peak-to-peak
  double tbar, tbar_drift;                // domain-mean theta, and its change
  double imb;                             // |Nu_hot - Nu_cold| / Nu, the budget
  double nu_lat, tau_f, tau_t, U, uc, ma;
  double res, drift;
  std::size_t steps;
  bool converged, finite;
};

template <class FColl>
Lid lid(Index N, double Ri, double Gr, double Pr, double uc,
        double tol, std::size_t cap, std::size_t probe,
        const char* dump, FColl fcoll_proto, bool verbose = false,
        std::size_t navg = 40, bool lid_off = false,
        const std::string& flank = "abb", double imbtol = 2e-3,
        double pert = 1e-3, double tseed = -1.0, double nwave = 2.0) {
  const Index Lc = N - 1;                             // on-node cavity height
  const double Re = std::sqrt(Gr / Ri);
  const Real nu = Real(uc * double(Lc) / std::sqrt(Gr));
  const Real D  = Real(double(nu) / Pr);
  const Real U  = lid_off ? Real(0) : Real(Re * double(nu) / double(Lc));
  const Real gb = Real(uc * uc / double(Lc));

  Domain d(N, N, 1, false, false, true);

  SColl scoll;
  scoll.omega = SColl::omega_from_diffusivity(D);
  // Symmetric about zero rather than about 0.5, so that any node which reports
  // the mean temperature is neutrally buoyant instead of maximally cold. Belt
  // and braces beside the ScalarOutflow fix above, not a substitute for it.
  scoll.T_ref = Real(0);
  ScalarSolver<SL, EsotericPull<SL>, SColl> th(d, scoll);
  th.set_geometry([&](Index x, Index y, Index) -> ScalarCell {
    if (y == 0 || y == N - 1) return ScalarMoment;     // hot below, cold above
    if (x == 0 || x == N - 1)
      return (flank == "abb") ? ScalarAdiabatic : ScalarOutflow;
    return ScalarBulk;
  });
  th.set_wall_values([&](Index, Index y, Index) -> Real {
    return (y == 0) ? Real(0.5) : Real(-0.5);
  });
  th.finalize_geometry();
  // Conduction profile PLUS a perturbation at the box wavelength.
  //
  // The perturbation is not decoration. The conduction state is exactly
  // x-invariant, the lattice and the boundary conditions are exactly
  // x-translation-invariant, and the scheme is deterministic -- so with no
  // x-dependent seed the Rayleigh-Benard mode has nothing to grow from, not even
  // round-off, and the layer sits in the unstable conduction state for ever. The
  // -u0 control did exactly that: Nu = 0.98991 and <theta> = 0.50000000 at 400
  // times the critical Rayleigh number. The lid breaks the symmetry on its own
  // when it is moving, but a seed whose symmetry decides the answer is a defect
  // whether or not this particular run happens to survive it.
  // validation/rayleigh_benard.cpp makes the same argument.
  // -tseed replaces the conduction profile with a well-mixed core at a given
  // mean. It buys steps and nothing else: the fixed point is unique, so the seed
  // cannot change the answer, only how long the mean temperature takes to walk
  // to it -- and that walk is the expensive part of this case (see below). The
  // value used comes from a cheap coarse run, not from a guess, and the
  // converged <theta> is printed so the seed can be checked against it.
  const Index Lci = Lc, Nxi = N;
  const Real eps = Real(pert);
  const Real ts = Real(tseed);
  const bool mixed = tseed >= 0.0;
  // Number of half-wavelengths across the box in the seed: 1 seeds a single
  // roll, 2 a counter-rotating pair. It is exposed because it MATTERS -- 2-D
  // convection in a closed box is multi-stable, and which branch the run lands
  // on is a property of the seed, not of the Rayleigh number.
  const double kw = nwave;
  th.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const double Y = double(py - d.hy) / double(Lci);
    const double X = double(px - d.hx) / double(Nxi - 1);
    const double base = mixed ? (double(ts) - 0.5) : (0.5 - Y);
    return Real(base + double(eps) * Kokkos::sin(kw * PI * X) *
                       Kokkos::sin(PI * Y));
  });
  th.compute_field();

  BoussinesqGuo force;
  force.T = th.temperature();
  force.gx = Real(0); force.gy = Real(1); force.gz = Real(0);
  force.rho0 = Real(1); force.beta = gb; force.T0 = Real(0);

  FColl fcoll = fcoll_proto;
  fcoll.omega = FColl::omega_from_viscosity(nu);
  fcoll.forcing = force;

  FluidSolver<FL, EsotericPull<FL>, FColl> fl(d, fcoll);
  fl.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  using WS = typename decltype(fl)::WallSpec;
  fl.set_regularized_walls([&](Index x, Index y, Index) -> WS {
    const bool xl = (x == 0), xr = (x == N - 1), yb = (y == 0), yt = (y == N - 1);
    if ((xl || xr) && (yb || yt)) return WS{NrmCorner, Real(0), Real(0), Real(0)};
    if (yt) return WS{NrmYp, U, Real(0), Real(0)};
    if (yb) return WS{NrmYm, Real(0), Real(0), Real(0)};
    if (xl) return WS{NrmXm, Real(0), Real(0), Real(0)};
    if (xr) return WS{NrmXp, Real(0), Real(0), Real(0)};
    return WS{};
  });
  fl.initialize(Real(1));
  th.set_velocity(fl.ux(), fl.uy(), fl.uz());

  const double scale = double(Lc) / double(D);

  // Domain-mean theta over the interior. Constant at statistical steady state
  // if and only if the flank condition conserves heat, so this is the leak test.
  auto mean_theta = [&]() {
    auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    double acc = 0; Index cnt = 0;
    for (Index y = 1; y <= N - 2; ++y)
      for (Index x = 1; x <= N - 2; ++x) { acc += double(hT(d.id(x, y, 0))); ++cnt; }
    return acc / double(cnt) + 0.5;          // report on the 0..1 scale
  };

  // Nusselt on a horizontal isothermal wall whose value is ON the node.
  auto nusselt = [&](bool hot) {
    auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    const double Tw = hot ? 0.5 : -0.5;
    const Index j1 = hot ? 1 : N - 2, j2 = hot ? 2 : N - 3;
    const double sgn = hot ? 1.0 : -1.0;
    double acc = 0; Index cnt = 0;
    for (Index x = 1; x <= N - 2; ++x) {
      const double g = wall_gradient(Tw, double(hT(d.id(x, j1, 0))),
                                     double(hT(d.id(x, j2, 0))), 1.0, 2.0);
      acc += -sgn * g * double(Lc); ++cnt;
    }
    return acc / double(cnt);
  };

  // The velocity scale for the residual is max(U, u_c), NOT U. With the lid
  // stopped -- the -u0 control -- U is zero, and dividing by it filled the
  // snapshot with inf, tripped the finiteness check on the first probe, and
  // reported a perfectly healthy Rayleigh-Benard run as DIVERGED.
  const double vscale = std::max(double(U), uc);
  auto snapshot = [&]() {
    fl.compute_macroscopic();
    th.compute_field();
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    std::vector<double> v;
    for (Index y = 1; y <= N - 2; ++y)
      for (Index x = 1; x <= N - 2; ++x) {
        const Index n = d.id(x, y, 0);
        v.push_back(double(hx(n)) / vscale);
        v.push_back(double(hy(n)) / vscale);
        v.push_back(double(hT(n)));
      }
    return v;
  };

  std::vector<double> now, before = snapshot();
  Settle st{tol}; std::size_t taken = 0; bool converged = false, finite = true;
  int budget_ok = 0;
  double imb = 1;
  while (taken < cap) {
    for (std::size_t k = 0; k < probe; ++k) { fl.step(true); th.step(); }
    taken += probe;
    now = snapshot();
    for (double v : now) if (!std::isfinite(v)) finite = false;
    if (!finite) break;
    const bool field_ok = st.update(interval_residual(now, before));
    before = now;
    const double nh = nusselt(true), nc = nusselt(false);
    imb = std::abs(nh - nc) / std::max(1e-30, 0.5 * (nh + nc));
    budget_ok = (imb < imbtol) ? budget_ok + 1 : 0;
    // The budget criterion needs a floor on the elapsed time as well as three
    // consecutive quiet probes. A state that is still SYMMETRIC balances the
    // budget trivially -- equal and opposite wall gradients -- so a run seeded
    // symmetrically satisfies it within a few thousand steps, long before the
    // flow exists. Twenty probes is not a guess: it is longer than the
    // convective transient at every (N, u_c) used here, and short compared with
    // the mean-temperature relaxation the criterion is there to catch.
    const bool budget = budget_ok >= 3 && taken >= 20 * probe;
    // Either route counts: a genuinely steady run trips the field residual, an
    // oscillatory one trips the heat budget.
    converged = field_ok || budget;
    if (verbose) {
      std::printf("    t = %-9zu res %.3e  drift %.3e  Nu %.5f / %.5f  "
                  "imb %.3e  <theta> %.6f\n",
                  taken, st.res, st.drift, nh, nc, imb, mean_theta());
      std::fflush(stdout);
    }
    if (converged) break;
  }

  // Time average, always: a converged run's average equals its instantaneous
  // value and the RMS comes back at round-off, so this costs a fixed navg
  // intervals and tells the two situations apart without being asked.
  double s1 = 0, s2 = 0, c1 = 0, lo = 1e300, hi = -1e300;
  const double tbar0 = mean_theta();
  for (std::size_t k = 0; k < navg; ++k) {
    for (std::size_t j = 0; j < probe; ++j) { fl.step(true); th.step(); }
    taken += probe;
    th.compute_field();
    const double nh = nusselt(true), nc = nusselt(false);
    s1 += nh; s2 += nh * nh; c1 += nc;
    lo = std::min(lo, nh); hi = std::max(hi, nh);
    if (verbose) {
      std::printf("    [avg %2zu/%zu] t = %-9zu Nu_hot %.5f  Nu_cold %.5f  "
                  "<theta> %.8f\n", k + 1, navg, taken, nh, nc, mean_theta());
      std::fflush(stdout);
    }
  }
  const double tbar1 = mean_theta();
  const double nh_bar = s1 / double(navg), nc_bar = c1 / double(navg);
  const double nh_rms = std::sqrt(std::max(0.0, s2 / double(navg) - nh_bar * nh_bar));

  fl.compute_macroscopic();
  auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
  double ma = 0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const Index n = d.id(x, y, 0);
      ma = std::max(ma, std::hypot(double(hx(n)), double(hy(n))));
    }
  ma /= std::sqrt(1.0 / 3.0);

  if (dump && *dump) {
    const std::string base = std::string("results/J_zhou_thermal/") + dump;
    std::FILE* f = std::fopen((base + "_centrelines.dat").c_str(), "w");
    if (f) {
      std::fprintf(f, "# thermal lid cavity Ri=%g Gr=%g Pr=%g N=%d op=%s\n",
                   Ri, Gr, Pr, int(N), FColl::name);
      // theta is reported on the paper's 0..1 scale; internally the field is
      // carried symmetric about zero (see the gauge note above).
      std::fprintf(f, "# velocity scale is max(U, u_c) = %.6f\n", vscale);
      std::fprintf(f, "# s  u_x(0.5,s)  u_y(s,0.5)  theta(0.5,s)  theta(s,0.5)\n");
      for (Index k = 0; k < N; ++k) {
        const double s = double(k) / double(Lc);
        std::fprintf(f, "%.8f %.8e %.8e %.8e %.8e\n", s,
                     double(hx(d.id(N / 2, k, 0))) / vscale,
                     double(hy(d.id(k, N / 2, 0))) / vscale,
                     double(hT(d.id(N / 2, k, 0))) + 0.5,
                     double(hT(d.id(k, N / 2, 0))) + 0.5);
      }
      std::fclose(f);
    }
    figdump::scalar_slice(base + "_temp.dat", N, N,
        [&](Index x, Index y) { return float(double(hT(d.id(x, y, 0))) + 0.5); });
    figdump::scalar_slice(base + "_ux.dat", N, N,
        [&](Index x, Index y) { return float(double(hx(d.id(x, y, 0))) / vscale); });
    figdump::scalar_slice(base + "_uy.dat", N, N,
        [&](Index x, Index y) { return float(double(hy(d.id(x, y, 0))) / vscale); });
  }

  return {nh_bar, nc_bar, nh_rms, hi - lo, tbar1, tbar1 - tbar0,
          std::abs(nh_bar - nc_bar) / (0.5 * (nh_bar + nc_bar)), double(nu),
          1.0 / double(FColl::omega_from_viscosity(nu)),
          1.0 / double(scoll.omega), double(U), uc, ma,
          st.res, st.drift, taken, converged, finite};
}

}  // namespace zt

//==============================================================================
//  REFERENCE DATA, transcribed from the paper and from the works it cites.
//
//  Two kinds of number live here and they are never mixed in a comparison:
//
//    ref_*     the external reference the paper itself is measured against --
//              analytic, de Vahl Davis (1983), Cheng & Liu (2010),
//              Bettaibi et al. (2014). This is the target.
//    zhou_*    the paper's own coupled FVM-LBM result, printed for context.
//              A monolithic solver pays no coupling error, so being closer to
//              the reference than this column is expected and is not a claim.
//
//  Table 4/5 quote de Vahl Davis via the paper (4.510 and 8.928 for the averaged
//  Nusselt number). His widely tabulated Richardson-extrapolated values are
//  4.519 and 8.800 -- the figure validation/natural_convection.cpp already uses
//  at Ra = 1e5. Both are printed; where they disagree by more than our own
//  discretisation error, that spread IS the uncertainty of the benchmark.
//==============================================================================
namespace ref {

struct Cond1D { double kappa, zhou_pct; };
const Cond1D cond[4] = {{0.1, 1.23}, {1.0, 0.48}, {10.0, 0.37}, {100.0, 0.35}};

struct Plate1D { double Pr, Pe, zhou_pct; };
const Plate1D plate[3] = {{0.1, 1.0, 0.41}, {1.0, 10.0, 2.00}, {10.0, 100.0, 3.65}};

struct NatRef {
  double Ra;
  double nu_zhou, nu_davis_paper, nu_luan, nu_davis_extrap;
  double numax_zhou, numax_davis, numax_luan;
  double ymax_zhou, ymax_davis, ymax_luan;
  double numin_zhou, numin_davis, numin_luan;
  double umax_davis, yu_davis, vmax_davis, xv_davis;
};
const NatRef nat[2] = {
  {1e5, 4.505, 4.510, 4.507, 4.519,  7.742, 7.761, 7.738,
        0.082, 0.085, 0.083,  0.700, 0.736, 0.746,
        34.73, 0.855,  68.59, 0.066},
  {1e6, 8.803, 8.928, 8.807, 8.800, 17.686, 18.076, 17.71,
        0.0375, 0.0456, 0.0375, 0.996, 1.005, 0.978,
        64.63, 0.850, 219.36, 0.0379},
};

// Case 3.6 has no table in the paper -- it compares centreline profiles against
// Fusegi et al. (1991), plotted as symbols in its Figs. 27 and 29. These are the
// peaks READ OFF THOSE PLOTS, so they carry the precision of a plot: roughly the
// last digit, call it 5 %. They are still worth having, because they come from
// the source the user supplied rather than from a remembered table, and because
// the paper's own normalisation u_ref = sqrt(g beta h dT) is exactly the u_c this
// driver is set up with -- so the comparison needs no conversion at all.
struct Nat3DRef { double Ra, ux_ref, y_ux, uy_ref, x_uy; };
const Nat3DRef nat3d[2] = {{1e5, 0.14, 0.85, 0.22, 0.075},
                           {1e6, 0.11, 0.88, 0.26, 0.055}};

struct LidRef { double Ri, cheng, bettaibi, fvm, zhou; };
const LidRef lid[3] = {{10.0, 4.860, 4.848, 4.862, 4.588},
                       {1.0,  5.750, 5.739, 5.758, 5.402},
                       {0.1, 12.161, 12.138, 12.580, 12.610}};

}  // namespace ref

static const char* NATHDR_TEXT =
    "# Zhou et al. (2026) Sec. 3.2 (2-D) and 3.6 (3-D): natural convection in a\n"
    "# differentially heated cavity, Pr = 0.71, hot x=0 / cold x=L, every other\n"
    "# wall adiabatic and no-slip. M3LB D3Q27 fluid + D3Q7 BGK temperature,\n"
    "# midway wall family, Esoteric Pull, FP64.\n"
    "# Velocities scaled by alpha/H; extrema taken on the z = L/2 plane.\n"
    "# de Vahl Davis (1983), Richardson-extrapolated:\n"
    "#   Ra=1e5  Nu=4.519  u*max=34.73@0.855  v*max=68.59@0.066\n"
    "#   Ra=1e6  Nu=8.800  u*max=64.63@0.850  v*max=219.36@0.0379\n"
    "# Nu_hot and Nu_cold are equal identically -- the configuration is\n"
    "# centro-symmetric -- so Nu_vol = 1 + <u_x theta> H/alpha is the\n"
    "# independent estimator, and <u_x>H/alpha must be ~0 for it to hold.\n"
    "# dim Ra N H op uc Nu_hot Nu_cold Nu_vol ux_mean Nu_max y_max Nu_min y_min "
    "u*max y_u v*max x_v tau_f tau_T Ma res drift steps conv\n";

template <class Fn>
bool with_op(const std::string& op, Fn&& fn) {
  if (op == "bgk") { fn(FBGK<FL>{}); return true; }
  if (op == "mrt") { fn(FMRT<FL>{}); return true; }
  if (op == "cm")  { fn(FCM<FL>{});  return true; }
  return false;
}

static void banner(const char* title) {
  std::printf("\n%s\n%s\n", title, std::string(std::strlen(title), '=').c_str());
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  const std::string which = arg_str(argc, argv, "-case", "all");
  const std::string op    = arg_str(argc, argv, "-op", "cm");
  const bool dump    = arg_flag(argc, argv, "-dump");
  // The right steady-state threshold is not the same for a 1010-node conduction
  // problem and a 40000-node convection one. The two diffusive cases must reach
  // round-off to say anything -- their exact answers are polynomials -- while
  // asking 1e-11 of a Ra = 1e6 cavity would cost hours for digits nobody reads.
  // Per CASE, not per invocation: -case all must not hand the conduction
  // problem the tolerance a Ra = 1e6 cavity needs, or its exact answer -- a
  // straight line -- comes back wrong in the fourth digit (see CONVERGENCE).
  auto tol_for = [&](const char* c) {
    const double d = (std::strcmp(c, "conduction") == 0) ? 1e-11 :
                     (std::strcmp(c, "plate")      == 0) ? 1e-9  :
                     (std::strcmp(c, "lid")        == 0) ? 1e-7  : 1e-8;
    return arg_num(argc, argv, "-tol", d);
  };
  const bool quick   = arg_flag(argc, argv, "-quick");
  const bool verbose = arg_flag(argc, argv, "-v");
  const std::size_t cap_arg = std::size_t(arg_num(argc, argv, "-cap", 0));

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::printf("Zhou, De Rosis & Revell, Eng. Comput. 42:56 (2026), Section 3\n");
    std::printf("M3LB alone -- no FVM side, no coupling.  fluid D3Q27/%s + "
                "temperature D3Q7/ScalarBGK\n", op.c_str());
    std::printf("backend %s   precision %s\n", ExecSpace::name(),
                precision_name());
    std::printf("steady state on the EXTRAPOLATED DRIFT, not the residual; the "
                "threshold is per case\n");

    //--------------------------------------------------------------------------
    if (which == "all" || which == "conduction") {
      banner("3.1  One-dimensional conjugate heat conduction");
      const double tol = tol_for("conduction");
      const double D0 = arg_num(argc, argv, "-d0", 0.05);
      const Index L   = Index(arg_num(argc, argv, "-n", quick ? 100 : 200));
      std::printf("  L = %d cells, solid over 0 <= x <= 0.2, D0 = %g, "
                  "geometric tau split\n", int(L), D0);
      std::printf("  eps is Eq. (38) as a norm ratio; x_int is where the two "
                  "fitted branches cross (exact 0.2)\n\n");
      std::printf("%-8s %-8s %-8s %-11s %-9s %-11s %-11s %-11s %-10s %-9s %-6s\n",
                  "kappa", "tau_s", "tau_f", "eps [%]", "Zhou [%]",
                  "slope_s", "slope_f", "x_int", "drift", "steps", "conv");
      std::printf("%s\n", std::string(122, '-').c_str());
      double worst = 0, kappa1 = 0;
      for (const auto& r : ref::cond) {
        char tag[128];
        std::snprintf(tag, sizeof tag, "cond_L%d_k%g", int(L), r.kappa);
        const zt::Cond c = zt::conduction(L, r.kappa, D0, tol,
                                          quick ? 2000000 : 40000000, 2000,
                                          dump ? tag : "");
        const double a_s = -1.0 / (0.2 + 0.8 * r.kappa);
        char xi[16];
        if (r.kappa == 1.0) std::snprintf(xi, sizeof xi, "%-11s", "n/a");
        else                std::snprintf(xi, sizeof xi, "%-11.7f", c.x_int);
        std::printf("%-8g %-8.4f %-8.4f %-11.3e %-9.2f %-11.6f %-11.6f "
                    "%s %-10.2e %-9zu %-6s\n",
                    r.kappa, c.tau_s, c.tau_f, c.eps_pct, r.zhou_pct,
                    c.slope_s, c.slope_f, xi, c.drift, c.steps,
                    c.converged ? "yes" : "NO");
        if (r.kappa == 1.0) kappa1 = c.eps_pct;
        else worst = std::max(worst, c.eps_pct);
        // nan, not the fitted value: at kappa = 1 the two branches are parallel
        // and their crossing point is the ratio of two round-off residuals. The
        // console prints n/a; a tracked table has to say the same thing.
        char xicol[32];
        if (r.kappa == 1.0) std::snprintf(xicol, sizeof xicol, "nan");
        else                std::snprintf(xicol, sizeof xicol, "%.9f", c.x_int);
        char row[512];
        std::snprintf(row, sizeof row,
            "%d %g %g %.6f %.6f %.6e %.9f %.9f %s %.6e %.6e %zu",
            int(L), r.kappa, D0, c.tau_s, c.tau_f, c.eps_pct,
            c.slope_s, c.slope_f, xicol, c.res, c.drift, c.steps);
        table_row("conduction.dat",
            "# Zhou et al. (2026) Sec. 3.1: 1-D conjugate conduction, solid over\n"
            "# 0 <= x <= 0.2L, kappa = alpha_s/alpha_f. M3LB D3Q7 BGK with a\n"
            "# per-node relaxation rate, anti-bounce-back walls, FP64.\n"
            "# Exact: theta = a_s x + 1 (x<=Ls), kappa a_s x + b_f (x>Ls),\n"
            "#   a_s = -1/(0.2 + 0.8 kappa); interface exactly at x/L = 0.2.\n"
            "# Zhou et al. Table 3 (coupled FVM-LBM at 1/500): 1.23 0.48 0.37 0.35 %\n"
            "# L kappa D0 tau_s tau_f eps[%] slope_s slope_f x_int res drift steps\n",
            row);
        (void)a_s;
      }
      std::printf("\n  exact slopes: a_s = -1/(0.2 + 0.8 kappa), a_f = kappa a_s;"
                  " x_int is undefined at kappa = 1 (the branches are parallel)\n");
      const bool p1 = kappa1 < 1e-8;
      const bool p2 = worst < 0.35;
      std::printf("  kappa = 1 is a single line and must be round-off: "
                  "%.2e                %s\n", kappa1, p1 ? "PASS" : "FAIL");
      std::printf("  kappa != 1 below the coupled framework's 0.35%% floor: "
                  "worst %.3e%%  %s\n", worst, p2 ? "PASS" : "FAIL");
      if (!(p1 && p2)) status = 1;
    }

    //--------------------------------------------------------------------------
    if (which == "all" || which == "plate") {
      banner("3.4  Normal plate velocity (1-D advection-diffusion, Eq. 44)");
      const double tol = tol_for("plate");
      const Index H  = Index(arg_num(argc, argv, "-n", 200));
      const double tf = arg_num(argc, argv, "-tau", 0.8);
      std::printf("  H = %d, tau fixed at %.3f so U follows Pe; "
                  "only Pe = Re Pr enters theta\n\n", int(H), tf);
      std::printf("%-6s %-8s %-9s %-11s %-11s %-11s %-11s %-9s %-8s\n",
                  "Pr", "Pe", "U", "eps abb [%]", "eps mom [%]", "Zhou [%]",
                  "drift", "steps", "conv");
      std::printf("%s\n", std::string(100, '-').c_str());
      const double pe_only = arg_num(argc, argv, "-pe", 0);
      bool beats_all = true;
      double worst = 0;
      for (const auto& r : ref::plate) {
        if (pe_only > 0 && std::abs(r.Pe / pe_only - 1.0) > 1e-9) continue;
        char t1[128], t2[128];
        std::snprintf(t1, sizeof t1, "plate_H%d_Pe%g_abb", int(H), r.Pe);
        std::snprintf(t2, sizeof t2, "plate_H%d_Pe%g_mom", int(H), r.Pe);
        const zt::Plate a = zt::plate(H, r.Pe, tf, false, tol,
                                      20000000, 2000, dump ? t1 : "");
        const zt::Plate b = zt::plate(H, r.Pe, tf, true, tol,
                                      20000000, 2000, dump ? t2 : "");
        std::printf("%-6g %-8g %-9.3e %-11.4f %-11.4f %-11.2f %-11.2e "
                    "%-9zu %-8s\n",
                    r.Pr, r.Pe, a.U, a.eps_pct, b.eps_pct, r.zhou_pct,
                    std::max(a.drift, b.drift), a.steps,
                    (a.converged && b.converged) ? "yes" : "NO");
        const double best = std::min(a.eps_pct, b.eps_pct);
        worst = std::max(worst, best);
        if (best >= r.zhou_pct) beats_all = false;
        char row[512];
        std::snprintf(row, sizeof row,
            "%d %g %.4f %.6e %.6e %.6e %.6e %.6e %zu %zu",
            int(H), r.Pe, tf, a.U, a.D, a.eps_pct, b.eps_pct,
            std::max(a.drift, b.drift), a.steps, b.steps);
        table_row("plate.dat",
            "# Zhou et al. (2026) Sec. 3.4: normal plate velocity. Uniform u_y\n"
            "# through an isothermal layer, theta = (exp(Pe y/L)-1)/(exp(Pe)-1),\n"
            "# Pe = Re Pr. M3LB D3Q7 BGK, prescribed velocity, FP64. Two wall\n"
            "# families: abb = anti-bounce-back (midway), mom = Dellar moment\n"
            "# (on-node). tau is held fixed so u_y follows Pe.\n"
            "# Zhou et al. Table 9 (coupled, H=200): 0.41 2.00 3.65 % at Pe = 1, 10, 100\n"
            "# H Pe tau U D eps_abb[%] eps_mom[%] drift steps_abb steps_mom\n",
            row);
      }
      if (arg_flag(argc, argv, "-uscan")) {
        std::printf("\n  O(u^2) separation at Pe = 100: the equilibrium is first "
                    "order in u, so\n  halving U at fixed Pe (by halving tau-1/2) "
                    "should quarter that part of the error.\n");
        for (double f : {1.0, 0.5, 0.25}) {
          const double tt = 0.5 + f * (tf - 0.5);
          const zt::Plate a = zt::plate(H, 100.0, tt, false, tol, 20000000, 2000, "");
          const zt::Plate b = zt::plate(H, 100.0, tt, true,  tol, 20000000, 2000, "");
          std::printf("    tau = %.4f  U = %.4e   eps abb = %-9.4f  eps mom = %-9.4f\n",
                      tt, a.U, a.eps_pct, b.eps_pct);
        }
      }
      std::printf("\n  compared PER Pe against that Pe's entry in Table 9, not "
                  "against the worst of them:\n");
      std::printf("  better than the coupled framework at every Pe "
                  "(worst here %.4f%%)          %s\n",
                  worst, beats_all ? "PASS" : "FAIL");
      if (!beats_all) status = 1;
    }

    //--------------------------------------------------------------------------
    if (which == "all" || which == "natural2d") {
      banner("3.2  Two-dimensional natural convection in a cavity (Pr = 0.71)");
      const double tol = tol_for("natural2d");
      const Index N   = Index(arg_num(argc, argv, "-n", quick ? 66 : 130));
      const double uc = arg_num(argc, argv, "-uc", 0.1);
      const double ra = arg_num(argc, argv, "-ra", 0);
      std::printf("  N = %d (H = %d fluid nodes), u_c = %g, midway wall family "
                  "throughout\n", int(N), int(N - 2), uc);
      std::printf("  velocities are non-dimensionalised by alpha/H "
                  "(de Vahl Davis's convention)\n\n");
      for (const auto& r : ref::nat) {
        if (ra > 0 && std::abs(r.Ra / ra - 1.0) > 1e-9) continue;
        char tag[128];
        std::snprintf(tag, sizeof tag, "nat2d_N%d_ra%.0e_%s", int(N), r.Ra, op.c_str());
        zt::Conv c{};
        with_op(op, [&](auto proto) {
          c = zt::natural(N, 1, r.Ra, 0.71, uc, tol,
                          cap_arg ? cap_arg : (quick ? 400000 : 4000000), 2000,
                          dump ? tag : "", proto, verbose);
        });
        std::printf("  Ra = %.0e   nu = %.5e  tau_f = %.4f  tau_T = %.4f  "
                    "Ma_max = %.4f\n", r.Ra, c.nu_lat, c.tau_f, c.tau_t, c.ma);
        std::printf("  %zu steps, residual %.2e, remaining drift %.2e, %s%s\n",
                    c.steps, c.res, c.drift,
                    c.converged ? "converged" : "NOT CONVERGED",
                    c.finite ? "" : ", DIVERGED");
        std::printf("  %-14s %-12s %-12s %-12s %-12s %-12s\n", "quantity",
                    "this", "Davis(paper)", "Davis(extrap)", "Luan", "Zhou(coupled)");
        std::printf("  %s\n", std::string(78, '-').c_str());
        std::printf("  %-14s %-12.4f %-12.3f %-12.3f %-12.3f %-12.3f\n",
                    "Nu_avg", c.nu_hot, r.nu_davis_paper, r.nu_davis_extrap,
                    r.nu_luan, r.nu_zhou);
        std::printf("  %-14s %-12.4f %-12s %-12s %-12s %-12s\n",
                    "Nu_avg (cold)", c.nu_cold, "-", "-", "-", "-");
        std::printf("  %-14s %-12.4f %-12s %-12s %-12s %-12s\n",
                    "Nu_vol", c.nu_vol, "-", "-", "-", "-");
        std::printf("  %-14s %-12.4f %-12.3f %-12s %-12.3f %-12.3f\n",
                    "Nu_max", c.nu_max, r.numax_davis, "-", r.numax_luan,
                    r.numax_zhou);
        std::printf("  %-14s %-12.4f %-12.4f %-12s %-12.4f %-12.4f\n",
                    "(y/L)_max", c.y_max, r.ymax_davis, "-", r.ymax_luan,
                    r.ymax_zhou);
        std::printf("  %-14s %-12.4f %-12.3f %-12s %-12.3f %-12.3f\n",
                    "Nu_min", c.nu_min, r.numin_davis, "-", r.numin_luan,
                    r.numin_zhou);
        std::printf("  %-14s %-12.4f %-12s %-12s %-12s %-12s\n",
                    "(y/L)_min", c.y_min, "1.000", "-", "0.997", "1.000");
        std::printf("  %-14s %-12.3f %-12.3f %-12s %-12s %-12s\n",
                    "u*_max", c.umax, r.umax_davis, "-", "-", "-");
        std::printf("  %-14s %-12.4f %-12.4f %-12s %-12s %-12s\n",
                    "  at y/L", c.y_umax, r.yu_davis, "-", "-", "-");
        std::printf("  %-14s %-12.3f %-12.3f %-12s %-12s %-12s\n",
                    "v*_max", c.vmax, r.vmax_davis, "-", "-", "-");
        std::printf("  %-14s %-12.4f %-12.4f %-12s %-12s %-12s\n",
                    "  at x/L", c.x_vmax, r.xv_davis, "-", "-", "-");
        std::printf("  hot/cold imbalance %+.2e%% (zero by centro-symmetry, not a "
                    "convergence check)\n"
                    "  wall vs volume estimator %+.3f%%   <u_x>H/alpha = %.2e "
                    "(must be 0 for Nu_vol to mean anything)\n\n",
                    100.0 * (c.nu_hot / c.nu_cold - 1.0),
                    100.0 * (c.nu_hot / c.nu_vol - 1.0), c.ux_mean);
        char row[640];
        std::snprintf(row, sizeof row,
            "2 %.0e %d %d %s %g %.6f %.6f %.6f %.2e %.6f %.6f %.6f %.6f %.4f %.6f "
            "%.4f %.6f %.6f %.6f %.6f %.3e %.3e %zu %d",
            r.Ra, int(N), int(N - 2), op.c_str(), uc, c.nu_hot, c.nu_cold,
            c.nu_vol, c.ux_mean,
            c.nu_max, c.y_max, c.nu_min, c.y_min, c.umax, c.y_umax,
            c.vmax, c.x_vmax, c.tau_f, c.tau_t, c.ma, c.res, c.drift,
            c.steps, int(c.converged));
        table_row("natural.dat", NATHDR_TEXT, row);
        const double dev = std::abs(c.nu_hot / r.nu_davis_extrap - 1.0);
        const double spread = std::abs(c.nu_hot / c.nu_vol - 1.0);
        std::printf("  Nu within 1%% of de Vahl Davis (%.3f): %+.3f%%          "
                    "%s\n", r.nu_davis_extrap, 100.0 * (c.nu_hot / r.nu_davis_extrap - 1.0),
                    dev < 0.01 ? "PASS" : "FAIL");
        std::printf("  wall and volume estimators within 1%% of each other:      "
                    "     %s\n\n", spread < 0.01 ? "PASS" : "FAIL");
        if (!c.finite || dev >= 0.01 || spread >= 0.01) status = 1;
      }
    }

    //--------------------------------------------------------------------------
    if (which == "lid") {
      banner("3.5  Two-dimensional thermal lid-driven cavity (Gr = 1e6, Pr = 0.71)");
      const double tol = tol_for("lid");
      const Index N   = Index(arg_num(argc, argv, "-n", 129));
      const double uc = arg_num(argc, argv, "-uc", 0.03);
      const double ri = arg_num(argc, argv, "-ri", 0);
      std::printf("  N = %d (L = %d on-node), u_c = %g so U = u_c/sqrt(Ri)\n",
                  int(N), int(N - 1), uc);
      std::printf("  Nusselt on the hot BOTTOM wall; the two adiabatic flanks "
                  "sit half a cell in (see banner)\n\n");
      const std::size_t navg = std::size_t(arg_num(argc, argv, "-navg", 40));
      const bool lid_off = arg_flag(argc, argv, "-u0");
      const std::string flank = arg_str(argc, argv, "-flank", "abb");
      const double imbtol = arg_num(argc, argv, "-imbtol", 2e-3);
      const double pert = arg_num(argc, argv, "-pert", 1e-3);
      const double tseed = arg_num(argc, argv, "-tseed", -1.0);
      const double nwave = arg_num(argc, argv, "-nwave", 2.0);
      std::printf("  adiabatic flanks: %s\n", flank == "abb"
                  ? "ScalarAdiabatic (bounce-back, conservative, plane at 0.5)"
                  : "ScalarOutflow (on-node zero gradient, NOT conservative)");
      std::printf("  seed: %s + %.0e sin(%g pi x/L) sin(pi y/L) -- without the "
                  "perturbation the layer\n        cannot convect at all, and "
                  "the half-wavelength count selects the branch\n",
                  tseed >= 0.0 ? "well-mixed core" : "conduction profile",
                  pert, nwave);
      if (tseed >= 0.0)
        std::printf("        core seeded at theta = %.3f; compare it with the "
                    "converged <theta> below\n", tseed);
      if (lid_off)
        std::printf("  -u0: LID STOPPED. This is the control -- pure "
                    "Rayleigh-Benard at the same Ra, Ri is only a label.\n");
      std::printf("  Nu is averaged over the last %zu probe intervals; "
                  "rms/mean says whether the state is steady\n", navg);
      std::printf("  statistical steady state is declared on the HEAT BUDGET, "
                  "|Nu_hot-Nu_cold|/Nu < %.0e for three\n  consecutive probes -- "
                  "see the banner: the field residual floors on the oscillation "
                  "and cannot\n  see the slow drift of <theta> underneath it.\n\n",
                  imbtol);
      std::printf("%-6s %-8s %-9s %-8s %-8s %-7s %-9s %-9s %-8s %-8s %-8s %-8s %-8s %-6s\n",
                  "Ri", "Re", "U", "tau_f", "tau_T", "Ma", "Nu(hot)",
                  "Nu(cold)", "rms/Nu", "pk-pk", "Cheng", "Bettaibi", "Zhou",
                  "steady");
      // fall through: the imbalance and <theta> drift are printed per row below
      std::printf("%s\n", std::string(132, '-').c_str());
      for (const auto& r : ref::lid) {
        if (ri > 0 && std::abs(r.Ri / ri - 1.0) > 1e-9) continue;
        char tag[128];
        std::snprintf(tag, sizeof tag, "lid_N%d_ri%g_%s", int(N), r.Ri, op.c_str());
        zt::Lid c{};
        with_op(op, [&](auto proto) {
          c = zt::lid(N, r.Ri, 1e6, 0.71, uc, tol,
                      cap_arg ? cap_arg : 8000000, 2000,
                      dump ? tag : "", proto, verbose, navg, lid_off,
                      flank, imbtol, pert, tseed, nwave);
        });
        std::printf("%-6g %-8.1f %-9.4f %-8.4f %-8.4f %-7.4f %-9.4f %-9.4f "
                    "%-8.4f %-8.4f %-8.3f %-8.3f %-8.3f %-6s\n",
                    r.Ri, std::sqrt(1e6 / r.Ri), c.U, c.tau_f, c.tau_t, c.ma,
                    c.nu_hot, c.nu_cold, c.nu_rms / c.nu_hot, c.nu_span,
                    r.cheng, r.bettaibi, r.zhou,
                    !c.finite ? "DIVERG" : (c.nu_rms / c.nu_hot < 1e-3 ? "yes" : "NO"));
        std::printf("       heat budget |dNu|/Nu %.2e   <theta> %.6f, drifting "
                    "%+.2e over the averaging window\n",
                    c.imb, c.tbar, c.tbar_drift);
        char row[640];
        std::snprintf(row, sizeof row,
            "%d %g %.1f %s %s %.6f %g %.6f %.6f %.6f %.6f %.6f %.6f %.6f "
            "%.8f %.3e %.3e %.3e %.0e %g %.3e %.3e %zu %d",
            int(N), r.Ri, std::sqrt(1e6 / r.Ri), lid_off ? "off" : "on",
            flank.c_str(), c.U, uc, c.tau_f, c.tau_t, c.ma, c.nu_hot, c.nu_cold,
            c.nu_rms, c.nu_span, c.tbar, c.tbar_drift, c.imb, pert, nwave,
            c.res, c.drift, c.steps, int(c.converged));
        table_row("lid.dat",
            "# Zhou et al. (2026) Sec. 3.5: thermal lid-driven cavity, heated from\n"
            "# below, Gr = 1e6, Pr = 0.71, Ri = Gr/Re^2. M3LB D3Q27 fluid + D3Q7 BGK\n"
            "# temperature, ON-NODE wall family (regularised fluid + Dellar moment),\n"
            "# adiabatic flanks by bounce-back. FP64.\n"
            "# The state is TIME DEPENDENT: Nu is a mean over the last navg probe\n"
            "# intervals, with its RMS and peak-to-peak alongside. lid=off is the\n"
            "# control -- the same configuration with U = 0, i.e. Rayleigh-Benard.\n"
            "# References (steady-state solutions): Cheng & Liu (2010) 4.860 5.750\n"
            "#   12.161 and Bettaibi et al. (2014) 4.848 5.739 12.138 at Ri = 10, 1, 0.1\n"
            "# N Ri Re lid flank U uc tau_f tau_T Ma Nu_hot Nu_cold Nu_rms "
            "Nu_pkpk theta_mean theta_drift imbalance pert nwave res drift "
            "steps conv\n",
            row);
        if (!c.finite) status = 1;
      }
      std::printf("\n  Cheng & Liu (2010) and Bettaibi et al. (2014) are the "
                  "references; Zhou is the coupled result.\n");
      std::printf("  A run marked steady = NO is oscillatory: read Nu as the "
                  "mean, and pk-pk as the honest error bar.\n");
    }

    //--------------------------------------------------------------------------
    if (which == "natural3d") {
      banner("3.6  Three-dimensional natural convection in a cubic cavity (Pr = 0.71)");
      const double tol = tol_for("natural3d");
      const Index N   = Index(arg_num(argc, argv, "-n", 50));
      const double uc = arg_num(argc, argv, "-uc", 0.2);
      const double ra = arg_num(argc, argv, "-ra", 1e5);
      std::printf("  %d^3 nodes (H = %d), u_c = %g, hot x=0 / cold x=L, "
                  "every other wall adiabatic\n", int(N), int(N - 2), uc);
      std::printf("  the paper gives no table for this case -- it compares "
                  "centreline profiles against\n  Fusegi et al. (1991). Those "
                  "profiles are dumped; the numbers below are ours, with the\n"
                  "  hot/cold Nusselt imbalance as the internal check.\n\n");
      char tag[128];
      std::snprintf(tag, sizeof tag, "nat3d_N%d_ra%.0e_%s", int(N), ra, op.c_str());
      zt::Conv c{};
      with_op(op, [&](auto proto) {
        c = zt::natural(N, N, ra, 0.71, uc, tol,
                        cap_arg ? cap_arg : 8000000, 1000,
                        dump ? tag : "", proto, verbose);
      });
      std::printf("  Ra = %.0e   nu = %.5e  tau_f = %.4f  tau_T = %.4f  "
                  "Ma_max = %.4f\n", ra, c.nu_lat, c.tau_f, c.tau_t, c.ma);
      std::printf("  %zu steps, residual %.2e, remaining drift %.2e, %s%s\n",
                  c.steps, c.res, c.drift,
                  c.converged ? "converged" : "NOT CONVERGED",
                  c.finite ? "" : ", DIVERGED");
      std::printf("  Nu_avg (hot)  %.4f\n", c.nu_hot);
      std::printf("  Nu_avg (cold) %.4f     imbalance %+.2e%% "
                  "(zero by centro-symmetry)\n", c.nu_cold,
                  100.0 * (c.nu_hot / c.nu_cold - 1.0));
      std::printf("  Nu_vol        %.4f     wall vs volume %+.3f%%   "
                  "<u_x>H/alpha = %.2e\n", c.nu_vol,
                  100.0 * (c.nu_hot / c.nu_vol - 1.0), c.ux_mean);
      std::printf("  Nu_max %.4f at (y/L) = %.4f;  Nu_min %.4f at %.4f\n",
                  c.nu_max, c.y_max, c.nu_min, c.y_min);
      std::printf("  on the z = L/2 plane: u*_max %.3f at y/L = %.4f, "
                  "v*_max %.3f at x/L = %.4f\n", c.umax, c.y_umax, c.vmax, c.x_vmax);
      // The paper's own normalisation. u/u_ref = (u H/alpha)/sqrt(Ra Pr), since
      // alpha/(H u_c) = 1/sqrt(Ra Pr) once nu = u_c H sqrt(Pr/Ra).
      const double uref = std::sqrt(ra * 0.71);
      const ref::Nat3DRef* fr = nullptr;
      for (const auto& q : ref::nat3d) if (std::abs(q.Ra / ra - 1.0) < 1e-9) fr = &q;
      std::printf("  same peaks in the paper's own scaling, u_ref = "
                  "sqrt(g beta h dT) = u_c:\n");
      std::printf("    u_x/u_ref %.4f at y/L %.4f", c.umax / uref, c.y_umax);
      if (fr) std::printf("   Fusegi via Fig. 27/29: ~%.2f at ~%.2f",
                          fr->ux_ref, fr->y_ux);
      std::printf("\n    u_y/u_ref %.4f at x/L %.4f", c.vmax / uref, c.x_vmax);
      if (fr) std::printf("   Fusegi via Fig. 27/29: ~%.2f at ~%.3f",
                          fr->uy_ref, fr->x_uy);
      std::printf("\n  those two reference numbers are READ OFF A PLOT -- treat "
                  "them as good to about 5%%.\n");
      char row[640];
      std::snprintf(row, sizeof row,
          "3 %.0e %d %d %s %g %.6f %.6f %.6f %.2e %.6f %.6f %.6f %.6f %.4f %.6f "
          "%.4f %.6f %.6f %.6f %.6f %.3e %.3e %zu %d",
          ra, int(N), int(N - 2), op.c_str(), uc, c.nu_hot, c.nu_cold,
          c.nu_vol, c.ux_mean,
          c.nu_max, c.y_max, c.nu_min, c.y_min, c.umax, c.y_umax,
          c.vmax, c.x_vmax, c.tau_f, c.tau_t, c.ma, c.res, c.drift,
          c.steps, int(c.converged));
      table_row("natural.dat", NATHDR_TEXT, row);
      if (!c.finite) status = 1;
    }
  }
  Kokkos::finalize();
  return status;
}
