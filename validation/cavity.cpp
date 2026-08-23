//==============================================================================
//  Lid-driven cavity with regularised velocity walls.
//
//  Reference data: U. Ghia, K. N. Ghia & C. T. Shin, J. Comput. Phys. 48, 387
//  (1982), Tables I and II -- the 129x129 solution at Re = 100.
//
//  This is the test that actually exercises corner nodes. Eq. (27) needs a
//  single wall normal, which a corner does not have, so those four nodes take
//  rho by second-order extrapolation along one wall instead (Latt et al. 2008,
//  Sec. V: "the density is extrapolated with second-order accuracy from
//  neighboring cells").
//
//  CORNER VELOCITY. The two upper corners sit on a genuine discontinuity: the
//  lid moves, the side walls do not. There is no correct value there. They are
//  given u = 0 here, the side-wall value, so that no spurious momentum is
//  injected at the singular point. Setting them to the lid velocity instead is
//  also defensible and changes the corner vortices slightly; it does not change
//  the centreline profiles compared below.
//
//  The cavity spans L = N-1 lattice units, not N, because the regularised
//  condition puts the walls ON the boundary nodes.
//==============================================================================
#include "boundary/Regularized.hpp"
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;

using L    = D2Q9;
using Eq   = SecondOrderEquilibrium<L>;
using Coll = BGK<L, Eq, NoForcing, ShiftedPopulations>;

// Ghia et al. (1982), Re = 100. u along the vertical centreline x = 0.5.
static const double GY[17] = {0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                              0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                              0.9531, 0.9609, 0.9688, 0.9766, 1.0000};
static const double GU[17] = {0.00000, -0.03717, -0.04192, -0.04775, -0.06434,
                              -0.10150, -0.15662, -0.21090, -0.20581, -0.13641,
                              0.00332, 0.23151, 0.68717, 0.73722, 0.78871,
                              0.84123, 1.00000};
// v along the horizontal centreline y = 0.5.
static const double GX[17] = {0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                              0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                              0.9453, 0.9531, 0.9609, 0.9688, 1.0000};
static const double GV[17] = {0.00000, 0.09233, 0.10091, 0.10890, 0.12317,
                              0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
                              -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
                              -0.05906, 0.00000};

// Ghia et al. (1982), Re = 1000, at the same station lists.
static const double GU1K[17] = {0.00000, -0.18109, -0.20196, -0.22220, -0.29730,
                                -0.38289, -0.27805, -0.10648, -0.06080, 0.05702,
                                0.18719, 0.33304, 0.46604, 0.51117, 0.57492,
                                0.65928, 1.00000};
static const double GV1K[17] = {0.00000, 0.27485, 0.29012, 0.30353, 0.32627,
                                0.37095, 0.33075, 0.32235, 0.02526, -0.31966,
                                -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
                                -0.21388, 0.00000};

static double lerp_at(const std::vector<double>& v, double s) {
  const double p = s * double(v.size() - 1);
  const int i = int(p);
  if (i >= int(v.size()) - 1) return v.back();
  return v[i] + (p - double(i)) * (v[i + 1] - v[i]);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index N = 129;
    double Re = 100.0;
    int dbg = 0;
    bool couette = false;   // periodic in x: moving top wall, no corners
    bool cstraight = false; // give corners a straight normal instead of NrmCorner
    bool fdcorner = true;   // Pi^(1) at corners from FD gradients (Latt et al., Sec. V)
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-n"  && i + 1 < argc) N  = std::atoi(argv[++i]);
      if (a == "-re" && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-dbg" && i + 1 < argc) dbg = std::atoi(argv[++i]);
      if (a == "-couette") couette = true;
      if (a == "-cornerstraight") cstraight = true;
      if (a == "-localcorner") fdcorner = false;
    }

    const Real U    = Real(0.05);
    const Real Lc   = Real(N - 1);              // walls sit on the boundary nodes
    const Real nu   = Real(double(U) * double(Lc) / Re);
    const Real tau  = nu * inv_cs2<L, Real>() + Real(0.5);

    std::printf("Lid-driven cavity, regularised walls (Latt et al. 2008, Sec. IV C)\n");
    std::printf("reference: Ghia, Ghia & Shin, JCP 48, 387 (1982), Re = %.0f\n", Re);
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());
    std::printf("  N = %d   L = %.0f   U = %.3f   nu = %.6f   tau = %.5f\n",
                int(N), double(Lc), double(U), double(nu), double(tau));
    std::printf("  corners: %s\n\n",
                fdcorner ? "finite-difference gradients (BC4/BC5 form)"
                         : "local populations (same closure as a straight wall)");

    Domain d(N, N, 1, /*periodic x*/ couette, false, true);
    Coll coll; coll.omega = Real(1) / tau;
    FluidSolver<L, EsotericPull<L>, Coll> s(d, coll);
    s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });

    using WS = decltype(s)::WallSpec;
    s.set_regularized_walls([&](Index x, Index y, Index) -> WS {
      const bool xl = (x == 0) && !couette, xr = (x == N - 1) && !couette;
      const bool yb = (y == 0), yt = (y == N - 1);
      if ((xl || xr) && (yb || yt))
        return WS{cstraight ? (yb ? NrmYm : NrmYp) : NrmCorner, Real(0), Real(0), Real(0)};
      if (yt) return WS{NrmYp, U, Real(0), Real(0)};        // the lid
      if (yb) return WS{NrmYm, Real(0), Real(0), Real(0)};
      if (xl) return WS{NrmXm, Real(0), Real(0), Real(0)};
      if (xr) return WS{NrmXp, Real(0), Real(0), Real(0)};
      return WS{};
    });
    s.set_fd_corners(fdcorner);
    s.initialize(Real(1));

    if (dbg) {
      for (int t = 0; t < dbg; ++t) {
        s.step();
        s.compute_macroscopic();
        auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        double rmin = 1e300, rmax = -1e300, umax = 0; Index worst = 0;
        // INTERIOR only: compute_macroscopic reports the streamed populations,
        // which at a wall node still contain the unknown directions the boundary
        // condition has not been applied to. Wall values there are meaningless.
        for (Index y = 1; y < N - 1; ++y)
          for (Index x = 1; x < N - 1; ++x) {
            const Index n = d.id(x, y);
            const double r = double(hr(n));
            const double q = std::hypot(double(hu(n)), double(hv(n)));
            rmin = std::min(rmin, r); rmax = std::max(rmax, r);
            if (q > umax) { umax = q; worst = n; }
          }
        Index wx, wy, wz; d.coords(worst, wx, wy, wz);   // padded coords
        std::printf("  step %3d  interior rho [%.6f, %.6f]  |u|max %.6e at interior (%d,%d)\n",
                    t + 1, rmin, rmax, umax, int(wx) - int(d.hx), int(wy) - int(d.hy));
        if (!std::isfinite(rmax)) break;
      }
      Kokkos::finalize();
      return 0;
    }

    const std::size_t probe = 500;
    Real prev = 0; std::size_t taken = 0;
    for (std::size_t t = 0; t < 4000000; t += probe) {
      for (std::size_t k = 0; k < probe; ++k) s.step();
      taken += probe;
      s.compute_macroscopic();
      auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
      const Real cur = h(d.id(N / 2, N / 2));
      if (!std::isfinite(double(cur))) { std::printf("  DIVERGED at step %zu\n", taken); break; }
      if (t > 0 && std::abs(double(cur - prev)) < 1e-11 * (std::abs(double(cur)) + 1e-12)) break;
      prev = cur;
    }
    std::printf("  steady after %zu steps\n\n", taken);

    s.compute_macroscopic();
    auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
    auto huy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());

    std::vector<double> uc(N), vc(N);
    for (Index j = 0; j < N; ++j) uc[j] = double(hux(d.id(N / 2, j))) / double(U);
    for (Index i = 0; i < N; ++i) vc[i] = double(huy(d.id(i, N / 2))) / double(U);

    std::printf("  u along x = 0.5                         v along y = 0.5\n");
    std::printf("  %6s %10s %10s %8s   %6s %10s %10s %8s\n",
                "y", "Ghia", "this", "diff", "x", "Ghia", "this", "diff");
    std::printf("  %s\n", std::string(76, '-').c_str());
    // Comparing against the wrong Reynolds number produces numbers that look
    // like errors but are not; refuse rather than mislead.
    const bool has_ref = (std::abs(Re - 100.0) < 1e-9) || (std::abs(Re - 1000.0) < 1e-9);
    const double* ru = (std::abs(Re - 1000.0) < 1e-9) ? GU1K : GU;
    const double* rv = (std::abs(Re - 1000.0) < 1e-9) ? GV1K : GV;
    if (!has_ref)
      std::printf("  NOTE: no Ghia table for Re = %.0f; the reference columns below\n"
                  "  are Re = 100 data and the diffs are meaningless.\n", Re);
    double eu = 0, ev = 0;
    for (int k = 0; k < 17; ++k) {
      const double su = lerp_at(uc, GY[k]), sv = lerp_at(vc, GX[k]);
      eu = std::max(eu, std::abs(su - ru[k]));
      ev = std::max(ev, std::abs(sv - rv[k]));
      std::printf("  %6.4f %10.5f %10.5f %8.4f   %6.4f %10.5f %10.5f %8.4f\n",
                  GY[k], ru[k], su, su - ru[k], GX[k], rv[k], sv, sv - rv[k]);
    }
    if (!has_ref) { eu = ev = NAN; }
    std::printf("\n  max |diff|:  u %.4f    v %.4f\n", eu, ev);
    std::printf("  wall check: u(lid)/U = %.6f (want 1), u(bottom) = %.2e (want 0)\n",
                double(hux(d.id(N / 2, N - 1))) / double(U),
                double(hux(d.id(N / 2, 0))));
  }
  Kokkos::finalize();
  return 0;
}
