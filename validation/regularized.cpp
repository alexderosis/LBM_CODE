//==============================================================================
//  Regularized velocity boundary condition (Latt et al. 2008, Sec. IV C).
//
//  Poiseuille flow with regularised walls instead of bounce-back.
//
//  GEOMETRY. This is the part that differs from validation/poiseuille.cpp and
//  the part most likely to be got wrong. Halfway bounce-back puts the no-slip
//  plane midway between the last fluid node and the first solid node; the
//  regularised condition imposes the velocity ON the boundary node, so the wall
//  sits exactly on it. With RegWall rows at y = 0 and y = ny-1,
//
//      u(y) = (G / 2 rho nu) y (W - y),   W = ny - 1,   u_max = G W^2 / (8 rho nu)
//
//  and u must vanish IDENTICALLY on the wall rows, not merely to O(h^2). That
//  is the sharpest available check that the condition enforces velocity exactly
//  -- it is what separates BC3 from a bounce-back of the off-equilibrium part.
//
//  The interior error is expected to converge at second order, matching the
//  paper's conclusion that all five reviewed conditions are second order.
//==============================================================================
#include "boundary/Regularized.hpp"
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;


struct Result {
  double l2;        // relative L2 error over the interior
  double slip;      // fitted position of the no-slip plane, want 0
  double umax;
  std::size_t steps;
};

template <class L>
static Result run(Real tau, Index W, Real umax_target) {
  using Coll = BGK<L, SecondOrderEquilibrium<L>, Guo, ShiftedPopulations>;
  // Periodic along the flow AND across the span, so the only walls are the two
  // in y: this isolates straight 3D walls from edges, which the unit tests
  // cover separately.
  const Index nz = (L::D == 3) ? Index(8) : Index(1);
  const Index nx = 8, ny = W + 1;          // wall rows at y = 0 and y = W
  const Real rho0  = Real(1);
  const Real nu    = (tau - Real(0.5)) * cs2<L, Real>();
  const Real omega = Real(1) / tau;
  const Real G     = Real(8) * rho0 * nu * umax_target / (Real(W) * Real(W));

  Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);
  Coll coll;
  coll.omega   = omega;
  coll.forcing = Guo{G, Real(0), Real(0)};

  FluidSolver<L, EsotericPull<L>, Coll> s(d, coll);
  s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  using WS = decltype(s)::WallSpec;
  s.set_regularized_walls([&](Index, Index y, Index) -> WS {
    if (y == 0)      return WS{NrmYm, Real(0), Real(0), Real(0)};
    if (y == ny - 1) return WS{NrmYp, Real(0), Real(0), Real(0)};
    return WS{};
  });
  s.initialize(rho0);

  const std::size_t max_steps = 600000, probe_every = 200;
  Real prev = 0;
  std::size_t taken = 0;
  for (std::size_t t = 0; t < max_steps; t += probe_every) {
    for (std::size_t k = 0; k < probe_every; ++k) s.step();
    taken += probe_every;
    s.compute_macroscopic();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
    const Real cur = h(d.id(nx / 2, ny / 2, nz / 2));
    if (!std::isfinite(double(cur))) return {NAN, NAN, NAN, taken};
    if (t > 0 && std::abs(double(cur - prev)) < 1e-15 * std::abs(double(cur))) break;
    prev = cur;
  }

  s.compute_macroscopic();
  auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hrh = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());

  double rsum = 0;
  for (Index j = 1; j < W; ++j) rsum += double(hrh(d.id(nx / 2, j, nz / 2)));
  const double rho_mean = rsum / double(W - 1);

  const double A = double(G) / (2.0 * rho_mean * double(nu));
  double num = 0, den = 0, umax = 0;
  for (Index j = 1; j < W; ++j) {
    const double y   = double(j);
    const double ana = A * y * (double(W) - y);
    const double e   = double(hux(d.id(nx / 2, j, nz / 2))) - ana;
    num += e * e; den += ana * ana;
    umax = std::max(umax, double(hux(d.id(nx / 2, j, nz / 2))));
  }
  // Where the wall actually is, measured rather than assumed. A parabola is
  // fitted to the three interior nodes nearest the lower wall and its root
  // taken; the offset from y = 0 is the effective slip. Reporting u at the wall
  // node instead would be meaningless -- the solver sets that to the imposed
  // value by construction, so it cannot disagree.
  const double y1 = 1, y2 = 2, y3 = 3;
  const double f1 = double(hux(d.id(nx / 2, 1, nz / 2)));
  const double f2 = double(hux(d.id(nx / 2, 2, nz / 2)));
  const double f3 = double(hux(d.id(nx / 2, 3, nz / 2)));
  const double a = ((f3 - f1) / (y3 - y1) - (f2 - f1) / (y2 - y1)) / (y3 - y2);
  const double b = (f2 - f1) / (y2 - y1) - a * (y1 + y2);
  const double c = f1 - a * y1 * y1 - b * y1;
  double slip = 0;
  if (std::abs(a) > 1e-300) {
    const double disc = b * b - 4 * a * c;
    if (disc >= 0) {
      const double r1 = (-b + std::sqrt(disc)) / (2 * a);
      const double r2 = (-b - std::sqrt(disc)) / (2 * a);
      slip = (std::abs(r1) < std::abs(r2)) ? r1 : r2;
    }
  }
  return {std::sqrt(num / den), slip, umax, taken};
}

//------------------------------------------------------------------------------
// Couette: the same walls with NO body force. The exact profile is linear, so
// fitting it and asking where u = 0 and u = U measures wall placement with the
// force removed -- which is what separates a force-coupling defect from an
// intrinsic tau dependence of the condition itself.
//------------------------------------------------------------------------------
template <class L>
static void couette(Real tau, Index W, Real U) {
  using Coll = BGK<L, SecondOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;
  const Index nx = 8, ny = W + 1;
  Domain d(nx, ny, 1, true, false, true);
  Coll coll; coll.omega = Real(1) / tau;
  FluidSolver<L, EsotericPull<L>, Coll> s(d, coll);
  s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  using WS = typename decltype(s)::WallSpec;
  s.set_regularized_walls([&](Index, Index y, Index) -> WS {
    if (y == 0)      return WS{NrmYm, Real(0), Real(0), Real(0)};
    if (y == ny - 1) return WS{NrmYp, U, Real(0), Real(0)};
    return WS{};
  });
  s.initialize(Real(1));
  for (std::size_t k = 0; k < std::size_t(60.0 * double(W) * double(W) /
                                          ((tau - Real(0.5)) / Real(3))); ++k) s.step();
  s.compute_macroscopic();
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  double sx = 0, sy = 0, sxx = 0, sxy = 0, c = 0;
  for (Index j = 1; j <= W - 1; ++j) {
    const double X = double(j), Y = double(h(d.id(nx / 2, j)));
    sx += X; sy += Y; sxx += X * X; sxy += X * Y; c += 1;
  }
  const double b = (c * sxy - sx * sy) / (c * sxx - sx * sx);
  const double a = (sy - b * sx) / c;
  std::printf("  %5.2f %5d   %12.4e %12.4e\n", double(tau), int(W),
              -a / b, (double(U) - a) / b);
}

//------------------------------------------------------------------------------
// Decompose f^neq at an INTERIOR node of a converged Poiseuille channel into its
// odd Hermite components and compare with the forced Chapman-Enskog prediction:
//
//   first order  (~ w_i c_ix)         a = -F / (2 cs^2)
//   third order  (~ w_i c_iy^2 c_ix)  b = (1 - om/2)/om^2 * rho u'' / cs^2
//
// The probe is checked against two EXACT constraints before its fit is read:
// sum_i f^neq = 0 and sum_i c_i f^neq = -F/2. An earlier version of this probe
// used rho = 1 + rho() -- double counting, since the solver already converts
// shifted density -- which inflated the equilibrium's odd term by u/cs^2 and
// produced a beautifully consistent but entirely fictitious answer.
//------------------------------------------------------------------------------
static void hermite_probe(Real tau, Index W, Real umax_target) {
  using LL = D2Q9;
  using Coll = BGK<LL, SecondOrderEquilibrium<LL>, Guo, ShiftedPopulations>;
  const Index nx = 8, ny = W + 1;
  const Real nu = (tau - Real(0.5)) * cs2<LL, Real>();
  const Real G = Real(8) * nu * umax_target / (Real(W) * Real(W));
  Domain d(nx, ny, 1, true, false, true);
  Coll coll; coll.omega = Real(1) / tau; coll.forcing = Guo{G, Real(0), Real(0)};
  FluidSolver<LL, EsotericPull<LL>, Coll> s(d, coll);
  s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  using WS = decltype(s)::WallSpec;
  s.set_regularized_walls([&](Index, Index y, Index) -> WS {
    if (y == 0)      return WS{NrmYm, Real(0), Real(0), Real(0)};
    if (y == ny - 1) return WS{NrmYp, Real(0), Real(0), Real(0)};
    return WS{};
  });
  s.initialize(Real(1));
  for (std::size_t k = 0; k < 400000; ++k) s.step();
  s.compute_macroscopic();

  auto F2 = s.gather_populations();
  auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, F2);
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());

  const Index j = W / 2, i0 = nx / 2;
  const Index n = d.id(i0, j);
  const double rho = double(hr(n));                 // already the true density
  const double u   = double(hu(n));
  const double upp = double(hu(d.id(i0, j + 1))) - 2.0 * u + double(hu(d.id(i0, j - 1)));

  double fneq[9], odd[9];
  for (int q = 0; q < 9; ++q) {
    const double f = double(hf(n, q)) + double(weight<LL, Real>(q));   // un-shift
    fneq[q] = f - double(SecondOrderEquilibrium<LL>::eq(q, Real(rho), Real(u), Real(0), Real(0)));
  }
  double m0 = 0, m1x = 0;
  for (int q = 0; q < 9; ++q) { m0 += fneq[q]; m1x += fneq[q] * LL::cx(q); }
  const bool ok = std::abs(m0) < 1e-10 && std::abs(m1x + double(G) / 2.0) < 1e-9;
  for (int q = 0; q < 9; ++q) odd[q] = 0.5 * (fneq[q] - fneq[opp(q)]);

  double A11=0, A12=0, A22=0, r1=0, r2=0;
  for (int q = 0; q < 9; ++q) {
    const double w  = double(weight<LL, Real>(q));
    const double p1 = w * LL::cx(q);
    const double p2 = w * LL::cy(q) * LL::cy(q) * LL::cx(q);
    A11 += p1*p1; A12 += p1*p2; A22 += p2*p2; r1 += p1*odd[q]; r2 += p2*odd[q];
  }
  const double det = A11*A22 - A12*A12;
  const double a = ( A22*r1 - A12*r2) / det;
  const double b = (-A12*r1 + A11*r2) / det;
  const double om = 1.0 / double(tau), cs2v = 1.0/3.0;
  const double b_pred = (1.0 - om/2.0)/(om*om) * rho * upp / cs2v;
  std::printf("  %5.2f %s  m0=%+.1e  m1=%+.3e/%+.3e   b=%+.4e  CE=%+.4e  ratio %6.3f\n",
              double(tau), ok ? "OK  " : "BAD ", m0, m1x, -double(G)/2.0, b, b_pred, b/b_pred);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Regularized velocity BC (Latt et al. 2008, Sec. IV C)\n");
    std::printf("Poiseuille flow, walls ON the boundary nodes\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    Real tau = Real(0.8); const Real umax = Real(0.02);
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == "-tau" && i + 1 < argc) tau = Real(std::atof(argv[i+1]));
    Real umax2 = umax;
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == "-umax" && i + 1 < argc) umax2 = Real(std::atof(argv[i+1]));
    std::printf("  tau = %.2f   u_max target = %.3f\n", double(tau), double(umax));

    auto sweep = [&](auto lat, const char* nm) {
      using LL = decltype(lat);
      std::printf("\n  === %s ===\n", nm);
      std::printf("  %4s %14s %8s %14s %10s\n", "W", "L2 error", "order", "wall at y =", "steps");
      std::printf("  %s\n", "-------------------------------------------------------------");
      double prev_e = 0; Index prev_W = 0;
      for (Index W : {Index(8), Index(16), Index(32)}) {
        const Result r = run<LL>(tau, W, umax2);
        double ord = NAN;
        if (prev_W) ord = std::log(prev_e / r.l2) / std::log(double(W) / double(prev_W));
        if (prev_W)
          std::printf("  %4d %14.6e %8.3f %14.2e %10zu\n", int(W), r.l2, ord, r.slip, r.steps);
        else
          std::printf("  %4d %14.6e %8s %14.2e %10zu\n", int(W), r.l2, "--", r.slip, r.steps);
        prev_e = r.l2; prev_W = W;
      }
    };
    std::printf("\n  === bulk Hermite decomposition of f^neq (interior node) ===\n");
    std::printf("  %5s %4s  %9s  %-23s %s\n", "tau", "chk", "sum f^neq", "sum c_x f^neq / want", "third-order term");
    std::printf("  %s\n", std::string(92, '-').c_str());
    for (Real t : {Real(0.6), Real(0.8), Real(1.0), Real(1.5)})
      hermite_probe(t, Index(32), Real(0.02));
    std::printf("\n");

    std::printf("\n  === Couette (NO body force): where the fitted line hits the wall values ===\n");
    std::printf("  %5s %5s   %12s %12s   (want 0 and W)\n", "tau", "W", "u=0 at y=", "u=U at y=");
    std::printf("  %s\n", std::string(52, '-').c_str());
    for (Real t : {Real(0.6), Real(0.8), Real(1.0), Real(2.0)})
      for (Index W : {Index(8), Index(16)}) couette<D2Q9>(t, W, Real(0.02));
    std::printf("\n");

    sweep(D2Q9{},  "D2Q9");
    sweep(D3Q19{}, "D3Q19  (periodic x and z, walls in y)");
    sweep(D3Q27{}, "D3Q27  (periodic x and z, walls in y)");
    std::printf("\n  the wall column is the fitted no-slip plane: it should sit at y = 0,\n");
    std::printf("  since the regularised condition places the wall ON the boundary node.\n");
  }
  Kokkos::finalize();
  return 0;
}
