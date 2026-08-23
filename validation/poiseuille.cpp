//==============================================================================
//  Force-driven plane Poiseuille flow -- Milestone 1 validation.
//
//  Periodic in x, halfway bounce-back walls in y, uniform body force along x.
//  H fluid nodes sit at interior y = 1..H; the walls are the solid layers at
//  y = 0 and y = H+1, so the no-slip planes are at y = 0.5 and y = H + 0.5:
//
//      u(y) = (G / 2 rho nu) (y - 0.5)(H + 0.5 - y),   u_max = G H^2 / (8 rho nu)
//
//  Why this case. With halfway bounce-back the effective wall position is
//  viscosity-dependent for BGK, and the slip vanishes exactly when
//      Lambda = (1/omega - 1/2)^2 = 3/16   ->   tau = 0.5 + sqrt(3)/4
//  At that "magic" tau the discrete solution is exact to round-off. So this one
//  case checks streaming, bounce-back, the equilibrium and the Guo forcing
//  simultaneously, against machine precision rather than a hand-picked
//  tolerance -- and away from the magic tau it measures the wall slip, which is
//  the number to watch when TRT lands at Milestone 4.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/TRT.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "io/VtiWriter.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "memory/TwoLattice.hpp"
#include "solver/FluidSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

using namespace lbm;
using L = D2Q9;

//------------------------------------------------------------------------------
// Least-squares fit u(y) = a + b y + c y^2; the roots are the effective walls.
//------------------------------------------------------------------------------
static void parabola_roots(const std::vector<double>& y, const std::vector<double>& u,
                           double& lo, double& hi) {
  double S[3][4] = {};
  for (std::size_t k = 0; k < y.size(); ++k) {
    const double p[3] = {1.0, y[k], y[k] * y[k]};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) S[i][j] += p[i] * p[j];
      S[i][3] += p[i] * u[k];
    }
  }
  for (int c = 0; c < 3; ++c) {
    int piv = c;
    for (int r = c; r < 3; ++r) if (std::abs(S[r][c]) > std::abs(S[piv][c])) piv = r;
    for (int j = 0; j < 4; ++j) std::swap(S[c][j], S[piv][j]);
    const double d = S[c][c];
    for (int j = 0; j < 4; ++j) S[c][j] /= d;
    for (int r = 0; r < 3; ++r) if (r != c) {
      const double f = S[r][c];
      for (int j = 0; j < 4; ++j) S[r][j] -= f * S[c][j];
    }
  }
  const double a = S[0][3], b = S[1][3], c2 = S[2][3];
  const double disc = std::sqrt(b * b - 4 * c2 * a);
  lo = (-b + disc) / (2 * c2);
  hi = (-b - disc) / (2 * c2);
  if (lo > hi) std::swap(lo, hi);
}

struct Result {
  double l2_rel;     // relative L2 error against the analytic profile
  double rho_mean;   // mean fluid density (must stay at rho0)
  double slip;       // effective wall offset: 0 means the wall is exactly at 0.5
  std::size_t steps;
};

template <class Coll, template <class> class Streaming, class Setup>
static Result run(Real tau, Index H, Real umax_target, const char* vti, Setup setup) {
  const Index nx = 8, ny = H + 2;
  const Real rho0  = Real(1);
  const Real nu    = (tau - Real(0.5)) * cs2<L, Real>();
  const Real omega = Real(1) / tau;
  const Real G     = Real(8) * rho0 * nu * umax_target / (Real(H) * Real(H));

  Domain d(nx, ny, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);
  Coll coll;
  setup(coll, omega, G);

  FluidSolver<L, Streaming<L>, Coll> s(d, coll);
  s.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });
  s.initialize(rho0);

  // March to steady state: stop when the centreline velocity stops moving.
  const std::size_t max_steps = 600000, probe_every = 200;
  Real prev = 0;
  std::size_t taken = 0;
  for (std::size_t t = 0; t < max_steps; t += probe_every) {
    for (std::size_t k = 0; k < probe_every; ++k) s.step();
    taken += probe_every;
    s.compute_macroscopic();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
    const Real cur = h(d.id(nx / 2, ny / 2));
    if (t > 0 && std::abs(double(cur - prev)) < 1e-15 * std::abs(double(cur))) break;
    prev = cur;
  }

  s.compute_macroscopic();
  auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hrh = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());

  std::vector<double> yy, uu;
  double rsum = 0;
  for (Index j = 1; j <= H; ++j) {
    yy.push_back(double(j));
    uu.push_back(double(hux(d.id(nx / 2, j))));
    rsum += double(hrh(d.id(nx / 2, j)));
  }
  const double rho_mean = rsum / double(H);

  // The analytic amplitude uses the ACTUAL mean density: the momentum equation
  // carries mu = rho nu, so assuming rho == rho0 would silently absorb any
  // density drift into the "error".
  const double A = double(G) / (2.0 * rho_mean * double(nu));
  double num = 0, den = 0;
  for (Index j = 1; j <= H; ++j) {
    const double y   = double(j);
    const double ana = A * (y - 0.5) * (double(H) + 0.5 - y);
    const double e   = uu[j - 1] - ana;
    num += e * e;
    den += ana * ana;
  }

  double lo, hi;
  parabola_roots(yy, uu, lo, hi);

  if (vti) write_vti(vti, s);
  return {std::sqrt(num / den), rho_mean, 0.5 - lo, taken};
}

//------------------------------------------------------------------------------
// Operator configurators. Each receives the omega implied by tau and the body
// force needed to reach the target peak velocity.
//------------------------------------------------------------------------------
namespace setups {
using EqL = SecondOrderEquilibrium<L>;

auto bgk = [](auto& c, Real omega, Real G) {
  c.omega = omega;
  c.forcing = Guo{G, Real(0), Real(0)};
};
// TRT pinned to the magic parameter: omega_minus is chosen so that
// Lambda = (1/omega_plus - 1/2)(1/omega_minus - 1/2) = 3/16 at every viscosity.
auto trt_magic = [](auto& c, Real omega, Real G) {
  using T = std::decay_t<decltype(c)>;
  c.omega_p = omega;
  c.omega_m = T::omega_minus_for(omega, T::magic_3_16);
  c.forcing = Guo{G, Real(0), Real(0)};
};
// TRT with omega_minus == omega_plus is exactly BGK -- used to show that the
// magic parameter, not the operator, is what removes the slip.
auto trt_bgk_like = [](auto& c, Real omega, Real G) {
  c.omega_p = omega; c.omega_m = omega;
  c.forcing = Guo{G, Real(0), Real(0)};
};
auto cm = [](auto& c, Real omega, Real G) {
  c.omega = omega;
  c.forcing = Guo{G, Real(0), Real(0)};
};
}  // namespace setups

using BGKf = BGK<L, SecondOrderEquilibrium<L>, Guo, ShiftedPopulations>;
using TRTf = TRT<L, SecondOrderEquilibrium<L>, Guo, ShiftedPopulations>;
using CMf  = MomentCollision<L, Guo, ShiftedPopulations, true>;
using MRTf = MomentCollision<L, Guo, ShiftedPopulations, false>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    // Predicted magic parameters. Lambda = (1/w_e - 1/2)(1/w_o - 1/2) = 3/16
    // removes the bounce-back wall slip. Which rates w_e, w_o are is set by the
    // operator:
    //   BGK  both modes at omega          -> (1/tau - 1/2)^2 = 3/16, tau = 0.5+sqrt3/4
    //   TRT  omega_minus chosen for it    -> ANY tau
    //   CM   2nd order at omega, 3rd at 1 -> (1/tau-1/2)(1/2) = 3/16, tau = 7/8
    const Real tau_bgk = Real(0.5) + Real(std::sqrt(3.0) / 4.0);
    const Real tau_mom = Real(0.875);

    std::printf("Poiseuille  (D2Q9, Guo forcing, halfway bounce-back, EsotericPull + shifted)\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    const double tol_first = sizeof(Real) == 4 ? 2e-4 : 1e-11;
    double worst_rho = 0;
    auto track = [&](const Result& r) { worst_rho = std::max(worst_rho, std::abs(r.rho_mean - 1.0)); return r; };

    //--------------------------------------------------------------------------
    std::printf("WALL SLIP vs TAU   (H = 32; slip is the fitted wall position minus 0.5)\n");
    std::printf("%-12s %-13s %-13s %-13s %-13s\n", "tau", "BGK", "TRT(3/16)", "MRT", "CentralMoments");
    std::printf("%s\n", std::string(68, '-').c_str());
    const std::vector<Real> taus = {Real(0.6), Real(0.8), tau_mom, tau_bgk, Real(1.2)};
    double worst_trt_slip = 0, bgk_at_its_magic = 1, cm_at_its_magic = 1;
    for (Real t : taus) {
      const Result b = track(run<BGKf, EsotericPull>(t, 32, Real(0.01), nullptr, setups::bgk));
      const Result r = track(run<TRTf, EsotericPull>(t, 32, Real(0.01), nullptr, setups::trt_magic));
      const Result q = track(run<MRTf, EsotericPull>(t, 32, Real(0.01), nullptr, setups::cm));
      const Result c = track(run<CMf,  EsotericPull>(t, 32, Real(0.01), nullptr, setups::cm));
      std::printf("%-12.7f %-+13.2e %-+13.2e %-+13.2e %-+13.2e\n",
                  double(t), b.slip, r.slip, q.slip, c.slip);
      worst_trt_slip = std::max(worst_trt_slip, std::abs(r.slip));
      if (t == tau_bgk) bgk_at_its_magic = std::abs(b.slip);
      if (t == tau_mom) cm_at_its_magic  = std::abs(c.slip);
    }
    std::printf("\nTRT holds the wall at 0.5 for EVERY viscosity -- that is the whole point of\n"
                "the magic parameter. BGK manages it only at tau = %.7f, and the moment\n"
                "operators only at tau = 7/8, because they relax the third-order modes at 1.\n",
                double(tau_bgk));

    //--------------------------------------------------------------------------
    std::printf("\nEXACTNESS AT EACH OPERATOR'S OWN MAGIC POINT   (H = 32)\n");
    std::printf("%-18s %-12s %-13s %-13s %-8s\n", "operator", "tau", "L2 rel err", "wall slip", "steps");
    std::printf("%s\n", std::string(68, '-').c_str());
    struct Row { const char* nm; double l2, slip; std::size_t st; double tau; };
    std::vector<Row> rows;
    { auto r = track(run<BGKf, EsotericPull>(tau_bgk, 32, Real(0.01), nullptr, setups::bgk));
      rows.push_back({"BGK", r.l2_rel, r.slip, r.steps, double(tau_bgk)}); }
    { auto r = track(run<TRTf, EsotericPull>(Real(0.6), 32, Real(0.01), nullptr, setups::trt_magic));
      rows.push_back({"TRT(3/16)", r.l2_rel, r.slip, r.steps, 0.6}); }
    { auto r = track(run<TRTf, EsotericPull>(Real(1.4), 32, Real(0.01), nullptr, setups::trt_magic));
      rows.push_back({"TRT(3/16)", r.l2_rel, r.slip, r.steps, 1.4}); }
    { auto r = track(run<MRTf, EsotericPull>(tau_mom, 32, Real(0.01), nullptr, setups::cm));
      rows.push_back({"MRT", r.l2_rel, r.slip, r.steps, 0.875}); }
    { auto r = track(run<CMf, EsotericPull>(tau_mom, 32, Real(0.01), "poiseuille.vti", setups::cm));
      rows.push_back({"CentralMoments", r.l2_rel, r.slip, r.steps, 0.875}); }
    double worst_magic = 0;
    for (const auto& r : rows) {
      std::printf("%-18s %-12.7f %-13.3e %-+13.3e %-8zu\n", r.nm, r.tau, r.l2, r.slip, r.st);
      worst_magic = std::max(worst_magic, r.l2);
    }

    //--------------------------------------------------------------------------
    // Resolution behaviour. At its magic point an operator is exact at EVERY H,
    // so there is nothing to fit -- the assertion is that the error stays at
    // round-off as the grid refines.
    //
    // Away from the magic point the discrete defect is a CONSTANT velocity
    // offset C proportional to (Lambda - 3/16) G / nu. With the peak velocity
    // held fixed, G ~ nu u_max / H^2, so C ~ 1/H^2 and the relative error is
    // SECOND order -- a lattice-aligned bounce-back wall does not degrade the
    // scheme. What does fall only as 1/H is the fitted wall POSITION, because
    // the parabola's curvature A ~ 1/H^2 shrinks with it and the root shift goes
    // as C/(A H). Both orders are fitted below, and seeing 2 and 1 together is
    // what identifies the defect as an offset rather than a wall displacement.
    std::printf("\nRESOLUTION BEHAVIOUR   (vs H)\n");
    std::printf("%-22s %-11s %-11s %-11s %-9s %-9s %-9s\n",
                "operator @ tau", "L2 H=16", "L2 H=32", "L2 H=64",
                "ord(L2)", "ord(slip)", "verdict");
    std::printf("%s\n", std::string(88, '-').c_str());

    const std::vector<Index> Hs = {16, 32, 64};
    auto ladder = [&](const char* nm, auto run_at) {
      double e[3], sl[3];
      for (int i = 0; i < 3; ++i) {
        const Result r = track(run_at(Hs[i]));
        e[i] = r.l2_rel; sl[i] = std::abs(r.slip);
      }
      const double ord  = std::log(e[1] / e[2]) / std::log(2.0);
      const double ords = std::log(sl[1] / sl[2]) / std::log(2.0);
      const bool exact = e[0] < tol_first && e[1] < tol_first && e[2] < tol_first;
      char o1[8] = "-", o2[8] = "-";
      if (!exact) { std::snprintf(o1, 8, "%.2f", ord); std::snprintf(o2, 8, "%.2f", ords); }
      std::printf("%-22s %-11.3e %-11.3e %-11.3e %-9s %-9s %-9s\n",
                  nm, e[0], e[1], e[2], o1, o2, exact ? "exact" : "converging");
      return std::pair<bool, double>{exact, ord};
    };

    const auto r_bgk = ladder("BGK @ 0.9330127",
      [&](Index H) { return run<BGKf, EsotericPull>(tau_bgk, H, Real(0.01), nullptr, setups::bgk); });
    const auto r_trt = ladder("TRT(3/16) @ 0.6",
      [&](Index H) { return run<TRTf, EsotericPull>(Real(0.6), H, Real(0.01), nullptr, setups::trt_magic); });
    const auto r_mrt = ladder("MRT @ 0.875",
      [&](Index H) { return run<MRTf, EsotericPull>(tau_mom, H, Real(0.01), nullptr, setups::cm); });
    const auto r_cm  = ladder("CentralMoments @ 0.875",
      [&](Index H) { return run<CMf, EsotericPull>(tau_mom, H, Real(0.01), nullptr, setups::cm); });
    const auto r_off = ladder("BGK @ 0.6 (off-magic)",
      [&](Index H) { return run<BGKf, EsotericPull>(Real(0.6), H, Real(0.01), nullptr, setups::bgk); });

    // These two assertions are FP64-only, and deliberately so. Refining to H=64
    // multiplies the step count by ~4 and drives the discretisation error down
    // to 3e-4, which in FP32 sits BELOW the accumulated round-off floor of about
    // 1e-3 -- the quantity being measured is no longer the quantity that
    // dominates. Loosening the tolerance would not fix that; it would just stop
    // the test from measuring anything. In FP32 the ladder is printed for
    // information and the magic-point exactness is checked at H=32 above.
    const bool fp64 = (sizeof(Real) == 8);
    const bool pass_exact = !fp64 || (r_bgk.first && r_trt.first && r_mrt.first && r_cm.first);
    const bool pass_first = !fp64 || (!r_off.first && std::abs(r_off.second - 2.0) < 0.25);
    if (!fp64)
      std::printf("\n  [FP32: the H=64 column is round-off dominated (floor ~1e-3 against a\n"
                  "   3e-4 signal), so the two resolution assertions below are FP64-only.]\n");

    //--------------------------------------------------------------------------
    std::printf("\nSTREAMING / STORAGE EQUIVALENCE   (CentralMoments, tau = 7/8)\n");
    const Result m[4] = {
      run<MomentCollision<L, Guo, RawPopulations,     true>, TwoLattice>  (tau_mom, 32, Real(0.01), nullptr, setups::cm),
      run<MomentCollision<L, Guo, ShiftedPopulations, true>, TwoLattice>  (tau_mom, 32, Real(0.01), nullptr, setups::cm),
      run<MomentCollision<L, Guo, RawPopulations,     true>, EsotericPull>(tau_mom, 32, Real(0.01), nullptr, setups::cm),
      run<MomentCollision<L, Guo, ShiftedPopulations, true>, EsotericPull>(tau_mom, 32, Real(0.01), nullptr, setups::cm),
    };
    const char* qn[4] = {"TwoLattice/raw", "TwoLattice/shifted",
                         "EsotericPull/raw", "EsotericPull/shifted"};
    for (int i = 0; i < 4; ++i)
      std::printf("  %-22s L2 %.3e   slip %+.3e\n", qn[i], m[i].l2_rel, m[i].slip);
    const bool pass_same = (m[0].l2_rel == m[2].l2_rel) && (m[1].l2_rel == m[3].l2_rel);

    //--------------------------------------------------------------------------
    const double tol      = tol_first;
    const double rho_tol  = sizeof(Real) == 4 ? 1e-5 : 1e-9;
    const bool pass_magic = worst_magic < tol;
    const bool pass_trt   = worst_trt_slip < tol;
    const bool pass_bgk   = bgk_at_its_magic < tol;
    const bool pass_cm    = cm_at_its_magic  < tol;
    const bool pass_rho   = worst_rho < rho_tol;

    std::printf("\nacceptance:\n");
    std::printf("  every operator exact at its magic point   worst L2 %.2e < %.1e   %s\n",
                worst_magic, tol, pass_magic ? "PASS" : "FAIL");
    std::printf("  TRT slip < %.1e at EVERY tau tested       worst %.2e          %s\n",
                tol, worst_trt_slip, pass_trt ? "PASS" : "FAIL");
    std::printf("  BGK slip vanishes at tau = 0.5+sqrt3/4    %.2e                %s\n",
                bgk_at_its_magic, pass_bgk ? "PASS" : "FAIL");
    std::printf("  CM  slip vanishes at tau = 7/8            %.2e                %s\n",
                cm_at_its_magic, pass_cm ? "PASS" : "FAIL");
    std::printf("  streaming/storage equivalence, bit for bit                     %s\n",
                pass_same ? "PASS" : "FAIL");
    std::printf("  every operator exact at H = 16, 32 AND 64            %s\n",
                fp64 ? (pass_exact ? "PASS" : "FAIL") : "n/a (FP32)");
    std::printf("  off-magic error still 2nd order (measured %.2f)      %s\n",
                r_off.second, fp64 ? (pass_first ? "PASS" : "FAIL") : "n/a (FP32)");
    std::printf("  rho drift  %.2e  < %.1e                                 %s\n",
                worst_rho, rho_tol, pass_rho ? "PASS" : "FAIL");
    if (!(pass_magic && pass_trt && pass_bgk && pass_cm && pass_same && pass_rho &&
          pass_exact && pass_first)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
