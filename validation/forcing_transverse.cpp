//==============================================================================
//  Transverse forced Poiseuille -- closing the gap left by validation/poiseuille.
//
//  WHY. The force central moments on D2Q9 are
//
//      K_force = [0, Fx, Fy, 0, 0, 0, cs^2 Fy, cs^2 Fx, 0]   (monomial basis)
//
//  and validation/poiseuille.cpp drives the channel along x with F = (G, 0, 0).
//  That leaves Fy = 0, so it exercises k_12 = cs^2 Fx and never touches
//  k_21 = cs^2 Fy. The same case rotated by ninety degrees -- walls x-normal,
//  force along +y -- swaps the two and covers the other half.
//
//  It also buys a second check for free. D2Q9 is invariant under a ninety-degree
//  rotation, so the two orientations must agree to round-off. Any anisotropy in
//  the forcing, the equilibrium or the streaming shows up as a difference here
//  that neither orientation on its own could reveal.
//
//  Walls are halfway bounce-back, so with H fluid nodes the no-slip planes sit
//  at 0.5 and H + 0.5 and the exact solution is
//
//      u(s) = (G / 2 rho nu) (s - 0.5)(H + 0.5 - s),  u_max = G H^2 / (8 rho nu)
//
//  At the central-moment magic point tau = 7/8 this is exact to round-off, which
//  is what makes it a sharp test rather than a tolerance comparison.
//==============================================================================
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "solver/FluidSolver.hpp"

#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>

using namespace lbm;
using L = D2Q9;
using CM = CentralMoments<L, Guo, ShiftedPopulations>;

// Least-squares parabola through the profile; returns its two roots, whose
// distance from 0.5 and H+0.5 is the wall slip.
static void parabola_roots(const std::vector<double>& s, const std::vector<double>& u,
                           double& lo, double& hi) {
  double S[5] = {0, 0, 0, 0, 0}, T[3] = {0, 0, 0};
  for (std::size_t i = 0; i < s.size(); ++i) {
    double p = 1.0;
    for (int k = 0; k < 5; ++k) { S[k] += p; p *= s[i]; }
    T[0] += u[i]; T[1] += u[i] * s[i]; T[2] += u[i] * s[i] * s[i];
  }
  const double M[3][3] = {{S[0],S[1],S[2]},{S[1],S[2],S[3]},{S[2],S[3],S[4]}};
  double A[3][4];
  for (int i = 0; i < 3; ++i) { for (int j = 0; j < 3; ++j) A[i][j] = M[i][j]; A[i][3] = T[i]; }
  for (int c = 0; c < 3; ++c) {
    int piv = c;
    for (int r = c + 1; r < 3; ++r) if (std::abs(A[r][c]) > std::abs(A[piv][c])) piv = r;
    for (int j = 0; j < 4; ++j) std::swap(A[c][j], A[piv][j]);
    for (int r = 0; r < 3; ++r) {
      if (r == c) continue;
      const double f = A[r][c] / A[c][c];
      for (int j = 0; j < 4; ++j) A[r][j] -= f * A[c][j];
    }
  }
  const double c0 = A[0][3]/A[0][0], c1 = A[1][3]/A[1][1], c2 = A[2][3]/A[2][2];
  const double disc = std::sqrt(c1 * c1 - 4.0 * c2 * c0);
  lo = (-c1 + disc) / (2.0 * c2);
  hi = (-c1 - disc) / (2.0 * c2);
  if (lo > hi) std::swap(lo, hi);
}

struct Result { double l2, slip; std::size_t steps; };

// orient = 0 : walls y-normal, force along +x, profile u_x(y)   [the usual case]
// orient = 1 : walls x-normal, force along +y, profile u_y(x)   [transverse]
static std::size_t g_cap = 600000;
static Result run(int orient, Real tau, Index H, Real umax) {
  const Index across = H + 2, along = 8;
  const Index nx = orient ? across : along;
  const Index ny = orient ? along  : across;
  const Real rho0 = Real(1);
  const Real nu = (tau - Real(0.5)) * cs2<L, Real>();
  const Real G  = Real(8) * rho0 * nu * umax / (Real(H) * Real(H));

  Domain d(nx, ny, 1, /*periodic x*/ orient == 0, /*y*/ orient == 1, /*z*/ true);
  CM coll;
  coll.omega = Real(1) / tau;
  coll.forcing = orient ? Guo{Real(0), G, Real(0)} : Guo{G, Real(0), Real(0)};

  FluidSolver<L, EsotericPull<L>, CM> s(d, coll);
  s.set_geometry([&](Index x, Index y, Index) -> CellType {
    const Index c = orient ? x : y;
    return (c == 0 || c == across - 1) ? Solid : Fluid;
  });
  s.initialize(rho0);

  const std::size_t cap = g_cap, probe = 200;
  Real prev = 0; std::size_t taken = 0;
  for (std::size_t t = 0; t < cap; t += probe) {
    for (std::size_t k = 0; k < probe; ++k) s.step();
    taken += probe;
    s.compute_macroscopic();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{},
                orient ? s.uy() : s.ux());
    const Real cur = h(d.id(nx / 2, ny / 2));
    if (t > 0 && std::abs(double(cur - prev)) < 1e-15 * std::abs(double(cur))) break;
    prev = cur;
  }

  s.compute_macroscopic();
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, orient ? s.uy() : s.ux());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());

  std::vector<double> ss, vv;
  double rsum = 0;
  for (Index j = 1; j <= H; ++j) {
    const Index id = orient ? d.id(j, ny / 2) : d.id(nx / 2, j);
    ss.push_back(double(j));
    vv.push_back(double(hv(id)));
    rsum += double(hr(id));
  }
  const double rho_mean = rsum / double(H);
  const double A = double(G) / (2.0 * rho_mean * double(nu));

  double num = 0, den = 0;
  for (Index j = 1; j <= H; ++j) {
    const double y = double(j);
    const double ana = A * (y - 0.5) * (double(H) + 0.5 - y);
    const double e = vv[j - 1] - ana;
    num += e * e; den += ana * ana;
  }
  double lo, hi;
  parabola_roots(ss, vv, lo, hi);
  return {std::sqrt(num / den), 0.5 - lo, taken};
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    const Real umax = Real(0.02);
    std::printf("Transverse forced Poiseuille   D2Q9, central moments, Guo forcing\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());
    std::printf("  orient 0 = walls y-normal, F along x  -> exercises k_12 = cs^2 Fx\n");
    std::printf("  orient 1 = walls x-normal, F along y  -> exercises k_21 = cs^2 Fy\n\n");

    std::printf("  %-10s %6s %5s %14s %14s %9s\n",
                "case", "tau", "H", "L2 rel err", "wall slip", "steps");
    std::printf("  %s\n", std::string(66, '-').c_str());

    bool conv = false;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "-conv") conv = true;
      if (std::string(argv[i]) == "-cap" && i + 1 < argc) g_cap = std::size_t(std::atol(argv[i+1]));
    }

    if (!conv) {
    const Real taus[3] = {Real(0.875), Real(0.6), Real(1.2)};
    for (Index H : {Index(16), Index(32)}) {
      for (Real tau : taus) {
        Result r0 = run(0, tau, H, umax);
        Result r1 = run(1, tau, H, umax);
        std::printf("  %-10s %6.4f %5d %14.4e %14.4e %9zu\n",
                    "along x", double(tau), int(H), r0.l2, r0.slip, r0.steps);
        std::printf("  %-10s %6.4f %5d %14.4e %14.4e %9zu\n",
                    "along y", double(tau), int(H), r1.l2, r1.slip, r1.steps);
        const double drot = std::abs(r0.l2 - r1.l2) + std::abs(r0.slip - r1.slip);
        const bool iso = drot < 1e-12;
        std::printf("  %-10s %6s %5s %14s %14.2e %9s\n",
                    "  rotation", "", "", "|diff|", drot, iso ? "ok" : "ANISO");
        if (!iso) ++bad;
        if (std::abs(double(tau) - 0.875) < 1e-12)
          if (r0.l2 > 1e-11 || r1.l2 > 1e-11) { ++bad; std::printf("  magic point NOT exact\n"); }
        std::printf("\n");
      }
    }
    } else {
      // ---- convergence: sweep H at fixed tau, both orientations ----
      const Real taus[4] = {Real(0.6), Real(0.875), Real(1.2), Real(2.0)};
      const Index Hs[5] = {Index(8), Index(16), Index(32), Index(64), Index(128)};
      std::FILE* fp = std::fopen("results/H_forcing_convergence/forcing_conv.dat", "w");
      if (fp) std::fprintf(fp, "# tau H l2_x order_x slip_x l2_y order_y slip_y\n");
      for (Real tau : taus) {
        std::printf("\n  tau = %.4f%s\n", double(tau),
                    std::abs(double(tau) - 0.875) < 1e-12 ? "   (magic point: exact, no convergence expected)" : "");
        std::printf("  %5s %13s %8s %13s %8s %13s %8s\n",
                    "H", "L2 (along x)", "order", "L2 (along y)", "order", "slip", "order");
        std::printf("  %s\n", std::string(76, '-').c_str());
        double p0 = 0, p1 = 0, ps = 0; Index pH = 0;
        for (Index H : Hs) {
          Result r0 = run(0, tau, H, umax);
          Result r1 = run(1, tau, H, umax);
          const double lg = std::log(2.0);
          const double o0 = pH ? std::log(p0 / r0.l2) / lg : NAN;
          const double o1 = pH ? std::log(p1 / r1.l2) / lg : NAN;
          const double os = pH ? std::log(std::abs(ps / r0.slip)) / lg : NAN;
          std::printf("  %5d %13.4e", int(H), r0.l2);
          if (pH) std::printf(" %8.3f", o0); else std::printf(" %8s", "--");
          std::printf(" %13.4e", r1.l2);
          if (pH) std::printf(" %8.3f", o1); else std::printf(" %8s", "--");
          std::printf(" %13.4e", r0.slip);
          if (pH) std::printf(" %8.3f", os); else std::printf(" %8s", "--");
          std::printf("  %9zu%s\n", r0.steps, r0.steps >= g_cap ? "  CAPPED" : "");
          std::fflush(stdout);
          if (fp) { std::fprintf(fp, "%.4f %d %.8e %.4f %.8e %.8e %.4f %.8e\n",
                                 double(tau), int(H), r0.l2, o0, r0.slip, r1.l2, o1, r1.slip);
                    std::fflush(fp); }
          p0 = r0.l2; p1 = r1.l2; ps = r0.slip; pH = H;
        }
      }
      if (fp) std::fclose(fp);
      std::printf("\n");
    }
    std::printf("  %s\n", bad == 0
      ? "PASS: both force components exact at tau = 7/8, and the two orientations agree"
      : "*** FAIL ***");
  }
  Kokkos::finalize();
  return bad == 0 ? 0 : 1;
}
