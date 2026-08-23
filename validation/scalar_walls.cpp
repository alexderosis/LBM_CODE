//==============================================================================
//  Scalar Dirichlet walls: anti-bounce-back against the moment condition.
//
//  Steady one-dimensional conduction between two walls held at T = 0 and T = 1,
//  no flow. The exact solution is linear in the distance between the wall
//  PLANES, which is where the two conditions differ:
//
//    anti-bounce-back  -- the plane sits half-way between the Dirichlet node
//                         and its fluid neighbour, so the node itself is not
//                         at the wall value and the effective width is N - 2;
//    moment condition  -- the inward population is chosen so that sum_i h_i is
//                         the imposed value (Dellar, Eqs. 13a-13b applied to
//                         the scalar's zeroth moment), placing the wall exactly
//                         ON the node; the effective width is N - 1.
//
//  Both are second-order schemes; the test is not which is more accurate but
//  whether each attains ITS OWN wall value, and whether the profile matches the
//  line implied by its own wall placement. Getting the width wrong by one
//  lattice unit is the classic way to mistake a placement error for an
//  accuracy result.
//==============================================================================
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace lbm;

using L    = D2Q5;
using Coll = ScalarBGK<L>;

struct Out { double l2, wall_lo, wall_hi; std::size_t steps; };

static Out run(Index N, Real tau, bool moment) {
  const Index ny = 4;
  Domain d(N, ny, 1, /*periodic x*/ false, /*y*/ true, /*z*/ true);
  Coll coll;
  coll.omega = Real(1) / tau;
  coll.T_ref = Real(0.5);

  ScalarSolver<L, EsotericPull<L>, Coll> s(d, coll);
  s.set_geometry([&](Index x, Index, Index) -> ScalarCell {
    if (x == 0 || x == N - 1) return moment ? ScalarMoment : ScalarDirichlet;
    return ScalarBulk;
  });
  s.set_wall_values([&](Index x, Index, Index) -> Real {
    return (x == 0) ? Real(0) : Real(1);
  });
  s.finalize_geometry();
  s.initialize(Real(0.5));

  // Diffusion needs O(N^2 / D) steps to cross the domain. Testing only that the
  // centre stopped moving declares convergence after a few hundred steps, while
  // the front has not even reached it -- which looks exactly like a boundary
  // condition that fails to converge.
  const Real D = cs2<L, Real>() * (Real(1) / coll.omega - Real(0.5));
  const std::size_t warmup = std::size_t(20.0 * double(N) * double(N) / double(D));
  const std::size_t probe = 200;
  Real prev = 0; std::size_t taken = 0;
  for (std::size_t t = 0; t < 20000000; t += probe) {
    for (std::size_t k = 0; k < probe; ++k) s.step();
    taken += probe;
    s.compute_field();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
    const Real cur = h(d.id(N / 2, ny / 2));
    if (!std::isfinite(double(cur))) return {NAN, NAN, NAN, taken};
    if (taken > warmup && std::abs(double(cur - prev)) < 1e-15) break;
    prev = cur;
  }
  s.compute_field();
  auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());

  // Least-squares line through the INTERIOR nodes only, then ask where it
  // reaches the two wall values. That measures wall placement instead of
  // asserting it: the solver reports the imposed value at a wall node either
  // way, so reading the node back would test nothing.
  double sx = 0, sy = 0, sxx = 0, sxy = 0, cnt = 0;
  for (Index i = 1; i <= N - 2; ++i) {
    const double X = double(i), Y = double(hT(d.id(i, ny / 2)));
    sx += X; sy += Y; sxx += X * X; sxy += X * Y; cnt += 1;
  }
  const double b = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx);
  const double a = (sy - b * sx) / cnt;
  double num = 0;
  for (Index i = 1; i <= N - 2; ++i) {
    const double e = double(hT(d.id(i, ny / 2))) - (a + b * double(i));
    num += e * e;
  }
  return {std::sqrt(num / cnt), -a / b, (1.0 - a) / b, taken};
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Scalar Dirichlet walls: anti-bounce-back vs the moment condition\n");
    std::printf("steady 1D conduction, T = 0 and T = 1, no flow\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    const Real tau = Real(0.8);
    for (int m = 0; m < 2; ++m) {
      const bool moment = (m == 1);
      std::printf("  === %s ===\n", moment ? "moment condition (wall ON the node)"
                                           : "anti-bounce-back (wall at half-spacing)");
      std::printf("  %5s %14s %13s %13s %11s\n", "N", "residual", "T=0 at x =", "T=1 at x =", "steps");
      std::printf("  %s\n", std::string(60, '-').c_str());
      double pe = 0; Index pn = 0;
      for (Index N : {Index(9), Index(17), Index(33), Index(65)}) {
        const Out r = run(N, tau, moment);
        (void)pe; (void)pn;
        std::printf("  %5d %14.5e %13.6f %13.6f %11zu   (want %.1f, %.1f)\n",
                    int(N), r.l2, r.wall_lo, r.wall_hi, r.steps,
                    moment ? 0.0 : 0.5, moment ? double(N - 1) : double(N) - 1.5);
        pe = r.l2; pn = N;
      }
      std::printf("\n");
    }
    std::printf("  the profile is exactly linear, so the residual is round-off and the\n");
    std::printf("  measurement that matters is WHERE the line reaches the wall values.\n");
  }
  Kokkos::finalize();
  return 0;
}
