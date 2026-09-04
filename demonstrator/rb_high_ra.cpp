//==============================================================================
//  Rayleigh-Benard at high Rayleigh number -- D3Q27 central moments + D3Q7 BGK.
//
//  The Kokkos twin of GPU/src/rb_high_ra.cu, and a replication of a D3Q19
//  reference driver's setup on this tree's lattices. Layer of depth H heated
//  from below, periodic in x, isothermal no-slip plates, Boussinesq buoyancy.
//
//  =================== THE PARAMETERISATION IS INVERTED ======================
//  validation/rayleigh_benard.cpp fixes nu and derives g. That is right for the
//  onset problem it solves -- bracketing Ra_c = 1707.762 -- and wrong above
//  Ra ~ 1e6, because the free-fall velocity it implies,
//
//      U_f = sqrt(g beta dT H) = (nu / H) sqrt(Ra / Pr),
//
//  then grows without bound: at Ra = 1e14 with nu = 0.02 it is 24.7 at H = 256.
//  So U_f is the INPUT here and the transport coefficients follow,
//
//      g beta = U_f^2 / (dT H),   nu = U_f H sqrt(Pr/Ra),   D = nu / Pr,
//
//  which is what the reference does. Three choices -- H, Ra, U_f -- and tau is
//  not a fourth; it is printed so it can be read back before the run.
//  ===========================================================================
//
//  cs2 IS 1/4 ON D3Q7 AND 1/3 ON D3Q27. The reference carries its temperature
//  on D3Q19 and so writes tauT = 3 D + 1/2; here that line would give D' = 3D/4,
//  a Prandtl number 4/3 too large, stable and converged and wrong.
//  `ScalarBGK<D3Q7>::omega_from_diffusivity` reads its own lattice's cs2, which
//  is why it is handed D and not a relaxation rate. This is the trap CLAUDE.md
//  names first and it is the one that does not announce itself.
//
//  WALL FAMILY. Halfway bounce-back for the momentum and anti-bounce-back for
//  the scalar, so BOTH planes sit at y = 0.5 and y = H + 0.5 and the layer is
//  exactly H deep with ny = H + 2. That is the pairing whose Ra_c this tree
//  reproduces, which is the evidence that H means what Ra says it means. The
//  reference uses the on-node pair instead; half a cell in H is 1% in H and 3%
//  in Ra, far below this run's discretisation error (below).
//
//  THE TEMPERATURE GAUGE IS SYMMETRIC, +/- 1/2, not the reference's [0, 1].
//  Same dT = 1 and the same physics, but T = 0 then means NEUTRALLY BUOYANT, so
//  a scalar node that reports zero is harmless rather than a buoyancy source --
//  the defence CLAUDE.md argues for after that cost a benchmark run. T_ref = 0
//  is then already the mean, so shifted storage needs nothing further.
//
//  nz = 1, AND IT IS GENUINELY 2-D. In a periodic direction one cell deep the
//  +/-z neighbour IS the node, so Esoteric Pull still writes two distinct slots
//  and the population simply stays put -- which is what streaming into yourself
//  means. `-nz` is kept so this can be checked against a slab rather than
//  argued; they agree.
//
//  ======================== WHAT Nu MEANS HERE ===============================
//  READ THIS BEFORE QUOTING A NUMBER. The 2-D correlation Nu ~ 0.14 Ra^0.29
//  gives Nu ~ 1600 at Ra = 1e14, so the thermal boundary layer H/2Nu is under
//  one third of a cell even at H = 1000. The lattice cannot make that layer
//  thinner than about a cell, which CAPS the Nusselt number at roughly H/2 --
//  500 here against the 1600 the physics wants. Whatever Nu this prints is a
//  property of the discretisation.
//
//  So the run prints the evidence rather than the claim: the volume estimator
//  and BOTH plate estimators, which agree only when resolved, and -- the sharpest
//  of the lot -- the RANGE of T. The advection-diffusion equation with Dirichlet
//  data in [-1/2, 1/2] obeys a maximum principle, so T outside those bounds is
//  the scheme failing and nothing else. On the CUDA twin at Ra = 1e14 that
//  column caught the failure a long way before the Nusselt numbers looked wrong:
//  T reached [-2.22, 1.54] at the same instant Nu_vol jumped to 56, while
//  Nu_top sat at exactly 0 -- no heat had reached the top plate at all, so the
//  "convection" was an overshoot rather than a plume. The run aborts when |T|
//  exceeds 1, i.e. twice its physical range.
//  ===========================================================================
//
//  WHAT THIS DOES NOT DO: no MPI, no grid stretching (the boundary layer costs
//  the same as the bulk), no restart -- at these step counts a run that outlives
//  its session is lost, which is the binding constraint rather than memory.
//
//    usage: rb_high_ra [-ny NY] [-nx NX] [-nz NZ] [-ra RA] [-pr PR] [-u U_F]
//                      [-amp A] [-tf N] [-out N] [-dump PREFIX]
//                      [--kokkos-num-threads=4]
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"
#include "../validation/FieldDump.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

using FL = D3Q27;
using SL = D3Q7;
using FluidColl = MomentCollision<FL, BoussinesqGuo, ShiftedPopulations, true>;
using ScalColl  = ScalarBGK<SL>;

int main(int argc, char** argv) {
  Index ny = 1000, nx = 2000, nz = 1;
  double Ra = 1e14, Pr = 0.71, U = 0.05, amp = 0.01;
  double tf = 40.0, out_every = 1.0;
  std::string dump;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-ny"   && i + 1 < argc) ny        = Index(std::atol(argv[++i]));
    if (a == "-nx"   && i + 1 < argc) nx        = Index(std::atol(argv[++i]));
    if (a == "-nz"   && i + 1 < argc) nz        = Index(std::atol(argv[++i]));
    if (a == "-ra"   && i + 1 < argc) Ra        = std::atof(argv[++i]);
    if (a == "-pr"   && i + 1 < argc) Pr        = std::atof(argv[++i]);
    if (a == "-u"    && i + 1 < argc) U         = std::atof(argv[++i]);
    if (a == "-amp"  && i + 1 < argc) amp       = std::atof(argv[++i]);
    if (a == "-tf"   && i + 1 < argc) tf        = std::atof(argv[++i]);
    if (a == "-out"  && i + 1 < argc) out_every = std::atof(argv[++i]);
    if (a == "-dump" && i + 1 < argc) dump      = argv[++i];
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const Index H = ny - 2;                       // fluid rows 1 .. H
    const double dT = 1.0, T_hot = 0.5, T_cold = -0.5;
    const double nu = U * double(H) * std::sqrt(Pr / Ra);
    const double D  = nu / Pr;
    const double gb = U * U / (dT * double(H));   // g*beta

    const double t_ff = double(H) / U;            // one free-fall time, in steps
    const std::size_t T_end = std::size_t(tf * t_ff);
    const std::size_t probe = std::size_t(out_every * t_ff)
                            ? std::size_t(out_every * t_ff) : 1;

    const double Nu_est = 0.14 * std::pow(Ra, 0.29);
    const double bl_cells = double(H) / (2.0 * Nu_est);

    std::printf("Rayleigh-Benard, free-fall scaling   Kokkos   %s fluid / %s scalar\n",
                FL::name, SL::name);
    std::printf("  operator %s + ScalarBGK   %s   %lld x %lld x %lld   H = %lld"
                "   %.3e cells\n", FluidColl::name,
                sizeof(Real) == 4 ? "FP32" : "FP64",
                (long long)nx, (long long)ny, (long long)nz, (long long)H,
                double(nx) * double(ny) * double(nz));
    std::printf("  Ra = %.3e   Pr = %.4f   U_f = %.4g   Ma = %.4f\n",
                Ra, Pr, U, U * std::sqrt(3.0));
    std::printf("  g beta = %.6e   nu = %.6e   D = %.6e\n", gb, nu, D);
    std::printf("  tau_f  = %.10f (omega %.8f)   [%s, cs2 = 1/3]\n",
                nu * 3.0 + 0.5, 1.0 / (nu * 3.0 + 0.5), FL::name);
    std::printf("  tau_g  = %.10f (omega %.8f)   [%s, cs2 = 1/4 -- NOT 3D+1/2]\n",
                D * 4.0 + 0.5, 1.0 / (D * 4.0 + 0.5), SL::name);
    if (sizeof(Real) == 4) {
      const double ulps = nu * 3.0 / 5.96e-8;
      std::printf("  ** FP32: tau - 1/2 = %.3e is %.1f ulp at 0.5, so nu and Ra are\n"
                  "     quantised by about %.2f%%.%s **\n", nu * 3.0, ulps,
                  50.0 / ulps, ulps < 10.0 ? "  REBUILD IN FP64." : "");
    }
    std::printf("  one free-fall time = %.0f steps;  %zu steps = %.0f of them\n",
                t_ff, T_end, tf);
    std::printf("  RESOLUTION: Nu ~ %.0f (2-D, 0.14 Ra^0.29) -> thermal BL = %.4f"
                " cells, Nu ceiling ~ H/2 = %lld.  %s\n\n",
                Nu_est, bl_cells, (long long)(H / 2),
                bl_cells >= 10.0 ? "Resolved."
                                 : "UNDER-RESOLVED: Nu below is the scheme, not Ra.");

    Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);

    // ---- the scalar ---------------------------------------------------------
    ScalColl scoll;
    scoll.omega = ScalColl::omega_from_diffusivity(Real(D));
    scoll.T_ref = Real(0);                        // already the mean, see banner
    ScalarSolver<SL, EsotericPull<SL>, ScalColl> th(d, scoll);
    th.set_geometry([&](Index, Index y, Index) -> ScalarCell {
      return (y == 0 || y == ny - 1) ? ScalarDirichlet : ScalarBulk;
    });
    th.set_wall_values([&](Index, Index y, Index) -> Real {
      return (y == 0) ? Real(T_hot) : Real(T_cold);          // hot below
    });
    // The reference's initial condition: the whole layer COLD, the entire dT
    // dropped across the bottom half cell at t = 0.
    const Real Tc = Real(T_cold);
    const Index nyc = ny;
    th.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Index y = py - d.hy;
      return (y <= 0 || y >= nyc - 1) ? Real(0) : Tc;
    });
    th.finalize_geometry();
    th.compute_field();

    // ---- the fluid ----------------------------------------------------------
    BoussinesqGuo force;
    force.T = th.temperature();
    force.gx = Real(0); force.gy = Real(1); force.gz = Real(0);
    force.rho0 = Real(1); force.beta = Real(gb); force.T0 = Real(0);

    FluidColl fcoll;
    fcoll.omega = FluidColl::omega_from_viscosity(Real(nu));
    fcoll.forcing = force;
    FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fcoll);
    fl.set_geometry([&](Index, Index y, Index) -> CellType {
      return (y == 0 || y == ny - 1) ? Solid : Fluid;
    });
    // At rest, with one single-mode density perturbation on the row nearest
    // mid-depth -- the reference's seed, and deliberately not noise: the
    // question is whether THIS mode grows.
    const Index seed_row = ny / 2;
    const Real ampr = Real(amp);
    const Index nxc = nx;
    fl.initialize_field(KOKKOS_LAMBDA(Index n) -> FlowState {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Index x = px - d.hx, y = py - d.hy;
      Real r = Real(1);
      if (y == seed_row)
        r += ampr * Real(Kokkos::cos(2.0 * M_PI * double(x) / double(nxc)));
      return FlowState{r, Real(0), Real(0), Real(0)};
    });
    th.set_velocity(fl.ux(), fl.uy(), fl.uz());

    std::printf("  %10s %11s %11s %11s %11s %9s %9s %10s %10s\n",
                "t/t_ff", "Nu_vol", "Nu_bot", "Nu_top", "max|u|",
                "T_min", "T_max", "<uy>", "residual");

    std::vector<Real> Tprev;
    int frame = 0;
    bool bad = false;
    std::size_t steps_run = T_end;
    const auto wall0 = std::chrono::steady_clock::now();

    for (std::size_t t = 0; t <= T_end; ++t) {
      if (t % probe == 0) {
        fl.compute_macroscopic();
        th.compute_field();
        auto hT  = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
        auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
        auto huy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
        auto huz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uz());

        double flux = 0, peak = 0, bot = 0, top = 0;
        double tmin = 1e300, tmax = -1e300;
        double mv = 0, mt = 0;                  // <u_y> and <T>, see below
        for (Index z = 0; z < nz; ++z)
          for (Index y = 1; y <= H; ++y)
            for (Index x = 0; x < nx; ++x) {
              const Index n = d.id(x, y, z);
              const double Tv = double(hT(n));
              flux += double(huy(n)) * Tv;
              mv += double(huy(n));  mt += Tv;
              const double s = std::sqrt(double(hux(n)) * double(hux(n)) +
                                         double(huy(n)) * double(huy(n)) +
                                         double(huz(n)) * double(huz(n)));
              if (s > peak) peak = s;
              if (Tv < tmin) tmin = Tv;
              if (Tv > tmax) tmax = Tv;
              if (y == 1) bot += Tv;
              if (y == H) top += Tv;
            }
        const double plate = double(nx) * double(nz);
        // ================== Nu USES THE FLUCTUATIONS ========================
        // Nu = 1 + H <u_y T> / (D dT) is the textbook form and it is only valid
        // where <u_y> = 0. That holds for an incompressible closed layer, and it
        // does NOT hold for the velocity this scheme reports: Guo's half shift
        // adds F/(2 rho) to it, so a layer whose mean temperature differs from
        // T0 carries a uniform vertical offset that is a property of the forcing
        // scheme and not a mass flux.
        //
        // It is not a small effect. At t = 0 here the fluid is at rest and T is
        // uniformly -1/2, so Nu must be exactly 1; the textbook form returned
        // 53.77, because u_y = g beta (T - T0)/2 = -6.26e-07 everywhere and
        // <u_y T> / (nx nz D) came to 52.8. The convective flux is the
        // correlation of the FLUCTUATIONS,
        //
        //     <u_y' T'> = <u_y T> - <u_y><T>,
        //
        // which is identically zero for any uniform state and reduces to the
        // textbook form whenever <u_y> = 0, i.e. whenever the physics is right.
        // <u_y> is printed rather than quietly subtracted: it is the size of the
        // artefact, and if it grows the run has a net vertical mass flux, which
        // is a separate problem from anything Nu can express.
        //
        // The same term is latent in validation/rayleigh_benard.cpp. It never
        // fired there because that case starts from the conduction profile,
        // whose mean is exactly zero in this symmetric gauge; the reference's
        // cold-start initial condition is what exposed it.
        // ====================================================================
        const double ncell = double(nx) * double(H) * double(nz);
        mv /= ncell;  mt /= ncell;
        const double Nu_vol =
            1.0 + (flux / ncell - mv * mt) * double(H) / (D * dT);
        const double Nu_bot = double(H) * (T_hot - bot / plate) / (0.5 * dT);
        const double Nu_top = double(H) * (top / plate - T_cold) / (0.5 * dT);

        // Whole-field residual over the INTERVAL. A per-step change is bounded
        // by the timestep and shrinks under refinement whether or not the flow
        // has settled; over an interval it is a statement about the field.
        double resid = 0;
        std::vector<Real> Tnow(std::size_t(hT.extent(0)));
        for (std::size_t k = 0; k < Tnow.size(); ++k) Tnow[k] = hT(Index(k));
        if (!Tprev.empty()) {
          double num = 0, den = 0;
          for (std::size_t k = 0; k < Tnow.size(); ++k) {
            const double dd = double(Tnow[k]) - double(Tprev[k]);
            num += dd * dd;  den += double(Tnow[k]) * double(Tnow[k]);
          }
          resid = (den > 0) ? std::sqrt(num / den) : 0.0;
        }
        Tprev = Tnow;

        std::printf("  %10.2f %11.4f %11.4f %11.4f %11.3e %9.4f %9.4f %10.2e %10.3e\n",
                    double(t) / t_ff, Nu_vol, Nu_bot, Nu_top, peak,
                    tmin, tmax, mv, resid);
        std::fflush(stdout);

        if (!dump.empty()) {
          char tag[32];
          std::snprintf(tag, sizeof tag, "_%04d.bin", frame);
          figdump::scalar_slice(dump + "_T" + tag, nx, ny,
                       [&](Index x, Index y) { return double(hT(d.id(x, y, nz / 2))); });
          figdump::scalar_slice(dump + "_u" + tag, nx, ny, [&](Index x, Index y) {
            const Index n = d.id(x, y, nz / 2);
            return std::sqrt(double(hux(n)) * double(hux(n)) +
                             double(huy(n)) * double(huy(n)) +
                             double(huz(n)) * double(huz(n)));
          });
          ++frame;
        }

        // The maximum principle, as a stopping rule. See the banner: this
        // catches the failure well before the Nusselt numbers look wrong.
        if (!std::isfinite(Nu_vol) || peak > 1.0 || tmin < -1.0 || tmax > 1.0) {
          std::printf("  STOPPED at t = %zu: %s\n", t,
                      !std::isfinite(Nu_vol) ? "not finite"
                      : peak > 1.0 ? "max|u| > 1"
                      : "T outside [-1, 1], i.e. twice its physical range");
          bad = true; steps_run = t; rc = 1;
          break;
        }
      }
      if (t < T_end) { fl.step(true); th.step(); }
    }

    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall0).count();
    std::printf("\n  %zu steps in %.2f s  ->  %.1f MLUPS%s\n", steps_run, sec,
                double(nx) * double(ny) * double(nz) * double(steps_run) / sec / 1e6,
                bad ? "  (stopped early)" : "");
    if (!dump.empty())
      std::printf("  %d frame(s) as %s_T_*.bin and %s_u_*.bin  (%lld x %lld float32,"
                  " two int32 of header)\n", frame, dump.c_str(), dump.c_str(),
                  (long long)nx, (long long)ny);
  }
  Kokkos::finalize();
  return rc;
}
