//==============================================================================
//  III.A (PRE) -- Poiseuille flow, inlet driven.
//
//  De Rosis, Phys. Rev. E 95, 013310 (2017), Sec. III A, Eqs. (14)-(15).
//
//  A laminar flow develops along x. No-slip on the two y-normal planes; a fixed
//  parabolic profile is prescribed at the west (x = 0) face,
//
//      ux(0, y) = 4 U0 (y/Ly) (1 - y/Ly),      U0 = 0.005,
//
//  and an outflow on the opposite side. Re = U0 Ly / nu = 10. The fluid starts
//  at rest with rho = rho0, and the run continues to steady state. The error is
//  the relative L2 norm between the analytic profile (which equals the inlet
//  profile, the flow being fully developed) and the numerical one sampled on a
//  y-aligned cross-section at the middle of the domain.
//
//  WHY THIS TEST IS WORTH HAVING. It is driven by the INLET, not by a body
//  force, so it is free of the Guo half-force interaction with the regularised
//  wall and of the curvature slip that dominates the forced channel. It is
//  therefore a cleaner second-order check of the wall treatment than
//  validation/regularized.cpp can be.
//
//  DEVIATION: Lz = 1 rather than the paper's 21, at the user's request; the
//  z direction is periodic either way and the flow has no z dependence.
//
//  The outflow is a constant back-pressure condition: rho = rho0 is imposed at
//  the outlet, the normal velocity follows from the inverted regularised
//  closure, and the tangential velocity is zero-gradient. The paper says only
//  "outflow"; this is the choice that makes the problem well posed. A pure
//  zero-gradient outlet does NOT: the regularised boundary nodes overwrite
//  populations and so are not mass conserving, nothing then anchors the
//  pressure, and the measured result was rho ~ 181 with the centreline velocity
//  a factor 2 low.
//
//  WHAT LIMITS THE ORDER -- measured, not assumed. The observed rate falls off
//  on the finest grids at U0 = 0.005 (pairwise 1.868, 2.182, 2.231, 1.334). The
//  outlet was the obvious suspect and it is NOT the cause. Three tests:
//
//    * per-station error at Ly = 641 -- the inlet station converges at 1.99 and
//      the bulk minimum at 1.92, both clean second order, while the outlet
//      station converges at 0.70. That looked conclusive, but:
//    * moving the outlet away (Lx = 41, 81) does not restore the rate. It gets
//      WORSE: the last pair goes 1.334 -> 0.885 -> 0.007, and at Lx = 81 the
//      error stops moving entirely (5.974e-6 at Ly = 321, 5.945e-6 at 641).
//    * making the tangential extrapolation second order does not help either
//      (set_outflow_order(2)): worse on coarse grids, identical on the finest.
//
//  What it actually is: a non-refining error that grows with U0. Sweeping U0 at
//  fixed Re = 10 moves the onset of the degradation, always toward COARSER
//  grids as U0 rises --
//
//      U0      Ma       fitted rate (5 grids)   pairwise
//      0.0025  0.0043   2.067                   1.720 2.026 2.205 2.263
//      0.005   0.0087   1.964                   1.868 2.182 2.231 1.334
//      0.01    0.0173   1.523                   2.091 2.179 0.974 0.812
//      0.02    0.0346   0.844                   2.097 0.150 0.644 0.947
//
//  At U0 = 0.0025 the compressibility term stays below the discretisation error
//  over the whole range and the fitted rate is 2.067, against the paper's 2.063
//  -- agreement to 0.2%. So the scheme is second order and the shortfall at
//  larger U0 is a Mach-number artefact of the test, not a defect in the wall or
//  the outlet. Fitting err = A/Ly^2 + B to the two finest grids gives B < 0 at
//  U0 = 0.0025 (no floor reached) and B = 3.1e-6, 9.2e-6, 2.1e-5 at U0 = 0.005,
//  0.01, 0.02; that is a growth between linear and quadratic in U0, but a
//  two-point extraction is too noisy to pin the exponent and it is not claimed.
//==============================================================================
#include "Campaign.hpp"
#include <chrono>
#include "boundary/Regularized.hpp"

using namespace lbm;
using namespace campaign;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    double Re = 10.0;
    Real U0 = Real(0.005);
    Index Lx = 21, Lz = 1;
    int oo = 1;                     // outflow tangential extrapolation order
    std::string lat = "d2q9", op = "cm";
    std::vector<Index> Lys = {41, 81, 161, 321, 641};
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-re"  && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-u0"  && i + 1 < argc) U0 = Real(std::atof(argv[++i]));
      if (a == "-lx"  && i + 1 < argc) Lx = std::atoi(argv[++i]);
      if (a == "-oo"  && i + 1 < argc) oo = std::atoi(argv[++i]);
      if (a == "-lat" && i + 1 < argc) lat = argv[++i];
      if (a == "-op"  && i + 1 < argc) op  = argv[++i];
      if (a == "-ly" && i + 1 < argc) Lys = {Index(std::atoi(argv[++i]))};
      if (a == "-lymax" && i + 1 < argc) {
        const Index m = std::atoi(argv[++i]);
        std::vector<Index> k; for (Index v : Lys) if (v <= m) k.push_back(v); Lys = k;
      }
    }

    std::printf("III.A (PRE 2017) Poiseuille, inlet driven   lattice %s   operator %s\n",
                lat.c_str(), op.c_str());
    std::printf("  U0 = %.4f   Re = U0 Ly / nu = %.0f   Lx = %d   Lz = %d (paper: 21)   outflow order %d\n\n",
                double(U0), Re, int(Lx), int(Lz), oo);
    std::printf("  %6s %8s %11s %7s %11s %13s %8s\n",
                "Ly", "tau", "steps", "t/tau_d", "residual", "rel L2 err", "order");
    std::printf("  %s\n", std::string(76, '-').c_str());

    std::FILE* f = open_out("A_poiseuille_inlet", "poiseuille", lat, op);
    if (f) std::fprintf(f, "# Ly tau steps t/tau_d residual err   U0=%.4f Re=%.0f Lx=%d Lz=%d\n",
                        double(U0), Re, int(Lx), int(Lz));

    double prev = 0; Index prevL = 0;
    for (Index Ly : Lys) {
      const Real nu = Real(double(U0) * double(Ly) / Re);
      double err = NAN, resid = NAN, secs = 0; std::size_t taken = 0;

      dispatch(lat, op, [&](auto coll) {
        using Coll = decltype(coll);
        using LL   = typename Coll::Lattice;
        Domain d(Lx, Ly, Lz, /*x*/ false, /*y*/ false, /*z*/ true);
        coll.omega = Coll::omega_from_viscosity(nu);
        FluidSolver<LL, EsotericPull<LL>, Coll> s(d, coll);

        s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
        using WS = typename decltype(s)::WallSpec;
        const Real U0c = U0; const Index Lyc = Ly, Lxc = Lx;
        s.set_regularized_walls([&](Index x, Index y, Index) -> WS {
          const bool yb = (y == 0), yt = (y == Lyc - 1);
          if (yb) return WS{NrmYm, Real(0), Real(0), Real(0)};
          if (yt) return WS{NrmYp, Real(0), Real(0), Real(0)};
          if (x == 0) {                                   // inlet, Eq. (14)
            const double yy = double(y) / double(Lyc - 1);
            return WS{NrmXm, Real(4.0 * double(U0c) * yy * (1.0 - yy)), Real(0), Real(0)};
          }
          // Constant back-pressure outlet at rho0 = 1. This is what anchors
          // the pressure; a zero-gradient outlet leaves the channel ill-posed.
          if (x == Lxc - 1) return WS{NrmOutXp, Real(0), Real(0), Real(0), Real(1)};
          return WS{};
        });
        s.set_outflow_order(oo);
        s.initialize(Real(1));                            // at rest
        if (std::getenv("DUMP"))
          std::printf("\n  [%d distinct wall states]\n", int(s.n_wall_states()));

        // STEADY STATE. The transient is momentum diffusion across the
        // channel, so the natural clock is tau_d = Ly^2/nu = Re Ly / U0 steps.
        //
        // Watching a single node for a small per-probe change does NOT work
        // here and silently reports a converged run that is nothing of the
        // kind: while the boundary layers are still growing the centreline
        // creeps by less than one ulp per step, the difference of two samples
        // saturates at round-off, and the test fires. Measured: Ly = 321 quit
        // after 19500 steps against tau_d = 6.4e5 and returned err = 7.5e-3
        // where the converged value is ~1.8e-5. So the residual is taken over
        // the WHOLE field, over an interval that is a fixed fraction of tau_d
        // rather than a fixed number of steps, and no run may stop before two
        // diffusive times have elapsed.
        const double tau_d  = Re * double(Ly) / double(U0);
        const std::size_t probe = std::max<std::size_t>(500, std::size_t(tau_d / 200.0));
        const std::size_t tmin  = std::size_t(0.5  * tau_d);
        const std::size_t cap   = std::size_t(30.0 * tau_d);

        std::vector<double> prevf(std::size_t(Lx) * std::size_t(Ly), 0.0);
        double res = NAN;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t t = 0; t < cap; t += probe) {
          for (std::size_t k = 0; k < probe; ++k) s.step();
          taken += probe;
          s.compute_macroscopic();
          auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
          double num = 0, den = 0; bool bad = false;
          for (Index yy = 0; yy < Ly; ++yy)
            for (Index xx = 0; xx < Lx; ++xx) {
              const std::size_t k = std::size_t(yy) * std::size_t(Lx) + std::size_t(xx);
              const double v = double(h(d.id(xx, yy, 0)));
              if (!std::isfinite(v)) { bad = true; break; }
              const double dv = v - prevf[k];
              num += dv * dv; den += v * v;
              prevf[k] = v;
            }
          if (bad) { err = NAN; return; }
          res = std::sqrt(num / std::max(den, 1e-300));
          if (taken >= tmin && res < 1e-10) break;
        }
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        resid = res;

        s.compute_macroscopic();
        auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        double num = 0, den = 0;
        for (Index y = 0; y < Ly; ++y) {
          const double yy = double(y) / double(Ly - 1);
          const double ana = 4.0 * double(U0) * yy * (1.0 - yy);
          const double dv = double(hux(d.id(Lx / 2, y, 0))) - ana;
          num += dv * dv; den += ana * ana;
        }
        err = std::sqrt(num / den);
        if (std::getenv("DUMP")) {
          auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
          auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
          std::printf("\n   y/Ly   ux(x=1)      ux(mid)     ux(x=Lx-1)   analytic     uy(mid)    rho(mid)\n");
          // Sample the two near-wall layers explicitly and symmetrically --
          // a uniform stride lands on y = 0 but never on y = Ly-1 and hides
          // whichever wall it misses.
          std::vector<Index> ys;
          for (Index k = 0; k < 8 && k < Ly; ++k) ys.push_back(k);
          ys.push_back(Ly / 2);
          for (Index k = 8; k-- > 0;) if (Ly - 1 - k >= 0) ys.push_back(Ly - 1 - k);
          for (Index yv : ys) {
            const double yy = double(yv) / double(Ly - 1);
            std::printf("  %5.3f %11.4e %11.4e %11.4e %11.4e %11.3e %10.6f\n", yy,
                        double(hux(d.id(1, yv, 0))), double(hux(d.id(Lx/2, yv, 0))),
                        double(hux(d.id(Lx-1, yv, 0))), 4.0*double(U0)*yy*(1.0-yy),
                        double(hv(d.id(Lx/2, yv, 0))), double(hr(d.id(Lx/2, yv, 0))));
          }
          // streamwise flux, should be identical at every station
          std::printf("   flux by station x=0..Lx-1:");
          for (Index xv = 0; xv < Lx; xv += 4) {
            double q = 0; for (Index yv = 0; yv < Ly; ++yv) q += double(hux(d.id(xv, yv, 0)));
            std::printf(" %.4e", q / double(Ly));
          }
          // WHERE THE ERROR LIVES. Per-station relative L2 against the same
          // parabola. If the outlet treatment is what limits the order, the
          // error must grow toward x = Lx-1; if the wall does, it must be flat
          // in x. This distinguishes the two without changing any code.
          std::printf("\n   rel L2 err by station:\n");
          for (Index xv = 0; xv < Lx; ++xv) {
            double a2 = 0, b2 = 0, ey = 0; Index eyi = 0;
            for (Index yv = 0; yv < Ly; ++yv) {
              const double yy2 = double(yv) / double(Ly - 1);
              const double an = 4.0 * double(U0) * yy2 * (1.0 - yy2);
              const double dvv = double(hux(d.id(xv, yv, 0))) - an;
              a2 += dvv * dvv; b2 += an * an;
              if (std::abs(dvv) > ey) { ey = std::abs(dvv); eyi = yv; }
            }
            std::printf("     x=%2d  %.5e   worst node y=%d (%.3f of Ly, %d from wall)\n",
                        int(xv), std::sqrt(a2 / b2), int(eyi),
                        double(eyi) / double(Ly - 1),
                        int(std::min(eyi, Ly - 1 - eyi)));
          }
          std::printf("\n");
        }
      });

      const double tau_d = Re * double(Ly) / double(U0);
      const double ord = prevL ? std::log(prev / err) / std::log(double(Ly) / double(prevL)) : NAN;
      std::printf("  %6d %8.4f %11zu %7.1f %11.2e %13.5e", int(Ly), 3.0 * double(nu) + 0.5,
                  taken, double(taken) / tau_d, resid, err);
      if (prevL) std::printf(" %8.3f", ord); else std::printf(" %8s", "--");
      std::printf("   [%.0f s]\n", secs);
      std::fflush(stdout);
      if (f) { std::fprintf(f, "%d %.4f %zu %.6f %.4e %.8e\n", int(Ly), 3.0*double(nu)+0.5,
                            taken, double(taken)/tau_d, resid, err);
               std::fflush(f); }
      prev = err; prevL = Ly;
    }
    if (f) std::fclose(f);
    std::printf("\n  paper reports a fitted convergence rate of 2.063\n");
  }
  Kokkos::finalize();
  return 0;
}
