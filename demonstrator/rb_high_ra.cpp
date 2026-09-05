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
//  MEASURED, and this is the part to read before choosing Ra. A 200 x 100 layer
//  at U_f = 0.05, conductive initial condition, out to 100 free-fall times:
//
//    Ra      tau_f - 1/2   BL cells   Nu_bot   Nu_top   verdict
//    1e6     1.24e-02       6.37       7.40     7.48    agree to 1%, and to 7%
//                                                       of 0.14 Ra^0.29 = 8.0
//    1e10    1.24e-04       0.44      46.57    46.79    agree to 0.5%, but that
//                                                       is the CEILING H/2 = 49,
//                                                       not the physical 111
//    1e14    1.24e-06       0.031        --       --    DIVERGES by t/t_ff ~ 30
//
//  Three things follow. The scheme is quantitatively right where the boundary
//  layer is resolved -- 7% of a published correlation at Ra = 1e6, which is the
//  positive control this file needed. The Nusselt CEILING is real and measured,
//  not argued: at Ra = 1e10 the two plate estimators agree with each other to
//  0.5% and land on H/2 rather than on the physics. And Ra = 1e14 is not
//  reachable at this H with either initial condition -- the conductive profile
//  removes the immediate undershoot the cold start causes, but the run still
//  dies once convection develops, which is the tau floor rather than the seed.
//
//  So the ceiling sets the usable Ra: Nu <= H/2 with Nu = 0.14 Ra^0.29 gives
//  Ra_max ~ (H/0.28)^(1/0.29), i.e. ~6e8 at H = 98 and ~2e12 at H = 998. Ra =
//  1e14 needs H >= 2 x 1607 = 3200 for a single cell in the layer, and ten
//  times that to resolve it properly.
//
//  ======== Ra_max IS A RESOLUTION RULE, NOT A STABILITY RULE (2026-09-05) ====
//  It does not predict whether a run SURVIVES, and two measurements at
//  Ra = 1e11 say so in opposite directions.
//
//    H = 498, 1000 x 500, conductive start, four threads. Ra/Ra_max = 0.62, so
//    the rule says comfortable. It HALTED at t/t_ff = 25 on the maximum
//    principle, T reaching 1.19 against a physical 0.5, with Nu_bot = 80.4
//    against Nu_top = 57.4. The overshoot was abrupt -- 0.0046, 0.0089, 0.0170,
//    then 0.690 in successive free-fall times -- and the failure was in the
//    BOTTOM boundary layer, 0.69 dT above the hot plate against 0.046 below the
//    cold one. GPU/'s independent CUDA driver reproduced it from a COLD start
//    on an A100, exiting 1 after 97 s. So H = 498 fails at Ra = 1e11 from both
//    initial conditions in both codebases.
//
//    H = 98, 200 x 100, cold start, 64 free-fall times. Ra/Ra_max = 169, so the
//    rule says hopeless -- and it is, but it did NOT halt. T_min reached -0.92
//    at t = 20 (0.42 dT below the floor, within 0.08 of the stop threshold) and
//    stayed out of bounds every row from t = 5 to t = 64, while Nu_bot ~ 95 sat
//    at THIRTY TIMES Nu_top ~ 3.1. The bulk never warmed: <T> at mid-depth was
//    still -0.5000 at t = 19. Heat leaves the bottom at Nu ~ 95 and arrives at
//    the top at Nu ~ 3, which is a layer filling rather than a steady state.
//
//  Neither is a measurement of Ra = 1e11. What is worth noticing is the ORDER:
//  the grid 25x coarser SURVIVED where the finer one halted. A hypothesis with
//  two points behind it and no third: when Nu is far above the H/2 ceiling the
//  scheme clips and behaves as an implicit LES, and when Nu sits NEAR the
//  ceiling it attempts a layer it cannot represent and fails. It predicts that
//  H = 998 (Nu/ceiling = 0.43) should be better behaved than H = 498
//  (0.87), not worse -- which is the point of the CSF3 array in GPU/csf3/.
//  Treat it as something to test, not as a result.
//  ===========================================================================
//
//  ONE CALIBRATION NOTE ON THE STARTUP WARNING. It prints UNDER-RESOLVED below
//  ten cells in the layer, and Ra = 1e6 above got within 7% on 6.37 cells. Ten
//  is a comfort criterion, not a correctness cliff; treat the warning as "check
//  this" rather than "discard this".
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
//                      [-amp A] [-tf N] [-out N] [-ic cond|cold] [-dump PREFIX]
//                      [-sop reg|bgk] [-slack S] [-grace G]
//                      [--kokkos-num-threads=4]
//
//  ============ ALIGNED WITH GPU/src/rb_high_ra.cu (2026-09-04) =============
//  These two drivers solve the same case in two independent codebases, which is
//  only worth anything if they are the same SETUP. Four ways they were not, all
//  found by reading them side by side against the D3Q19 reference:
//
//    * SCALAR OPERATOR. This used ScalarBGK, which relaxes the D3Q7 ghost
//      moments at omega and so REFLECTS them as omega -> 2. The reference
//      relaxes only the flux moments and sends every higher one to
//      equilibrium, and so does GPU/. `ScalarRegularised` (new, in src/) is
//      that operator for this tree; `-sop reg` is now the default and
//      `-sop bgk` reproduces the old behaviour. It is not a small difference:
//      worst cold-start T_min against a floor of T_cold is -0.813 under BGK
//      and -0.112 under the regularised form at the same point.
//    * omega_bulk. This left it at -1, "follow omega", so the trace of the
//      second moment relaxed at 1.9984. The reference sets it to 1 (its
//      k4 = R) and so does GPU/. Now 1.
//    * Nu_ref. GPU/ prints the reference's own Nusselt normalisation beside
//      the exact one so the two can go in a table. Added, with a note on the
//      two terms it cannot reproduce in this gauge.
//    * THE INITIAL CONDITION DEFAULT. This defaulted to `cold`, GPU/ to
//      `cond`. Two drivers of the same case defaulting to different initial
//      conditions is the "comparison of drivers" that CLAUDE.md's measurement
//      discipline warns about. Now `cond` in both.
//
//  DELIBERATELY STILL DIFFERENT: the temperature gauge. This one is symmetric
//  about zero (T_hot = +1/2, T_cold = -1/2) and GPU/'s is [0, 1] with
//  T_ref = 1/2. A gauge is not physics, and the symmetric one is the defence
//  CLAUDE.md recommends -- `field = 0` then means neutrally buoyant, which is
//  what makes an adiabatic scalar node harmless. It does mean T_min/T_max and
//  Nu_ref are not comparable digit-for-digit between the two; Nu_bot, Nu_top
//  and Nu_vol are.
//  ==========================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/ScalarBGK.hpp"
#include "collision/ScalarRegularised.hpp"
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

// Everything the run needs, AT NAMESPACE SCOPE and not inside main. `run` below
// is a template on the scalar operator and its body contains KOKKOS_LAMBDAs,
// and nvcc forbids a FUNCTION-LOCAL type as a template argument of a function
// that does -- so a struct declared in main would compile here and fail the
// moment this file is built for CUDA. validation/cmbench.cpp carries the same
// note for the same restriction.
struct Opts {
  Index nx, ny, nz;
  double Ra, Pr, U, amp, tf, out_every, slack, grace;
  std::string ic, dump;
};

template <class ScalColl>
static int run(const Opts& o) {
  const Index nx = o.nx, ny = o.ny, nz = o.nz;
  const double Ra = o.Ra, Pr = o.Pr, U = o.U, amp = o.amp;
  const double tf = o.tf, out_every = o.out_every, slack = o.slack;
  const std::string& ic = o.ic;
  const std::string& dump = o.dump;
  double grace = o.grace;
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

    // ================== THE GRACE WINDOW, AS IN THE CUDA TWIN ================
    // A cold start drops the whole dT across the bottom half cell and the D3Q7
    // scalar near omega = 2 undershoots rather than smoothing it, so the
    // maximum-principle stop rule can fire on the INITIAL CONDITION instead of
    // on a failure. How far it undershoots is a property of the SCALAR
    // OPERATOR, not of the IC: worst T_min against a floor of T_cold,
    //
    //     ScalarBGK          Ra = 1e10  -0.552      Ra = 1e14  -0.813
    //     ScalarRegularised  Ra = 1e14  -0.112, and it recovers
    //
    // which is why `-sop reg` is now the default here. The window suppresses
    // the HALT (never the report) while the step is still diffusing away:
    // delta = sqrt(D t), so four cells takes 16/D steps, and the window is
    // EXACTLY that -- the halt is suppressed for precisely as long as the step
    // is still sharper than four cells, and not one free-fall time longer.
    //
    // It was 4x that at first, which is padding with no argument behind it, and
    // the padding is what did the damage: at H = 498, Ra = 1e11, tf = 100 the
    // step smooths in 17 t_ff -- 17% of the run, plainly a transient -- yet
    // 4 x 17 = 69 t_ff tripped the refusal, and the message then blamed the
    // physics for a multiplier this file had chosen. A window with a meaning
    // beats a window with a safety factor.
    //
    // It is REFUSED -- set to zero, with the arithmetic printed -- when even
    // that exceeds a quarter of the run, because the undershoot is then not
    // transient on the RUN'S timescale and stopping is the right answer.
    // nan, a non-finite Nu and max|u| > 1 halt at any time regardless.
    // =========================================================================
    const double t_smooth_tff = 16.0 / (D * t_ff);
    bool grace_refused = false;
    if (grace < 0.0) {
      grace = (ic == "cond") ? 0.0 : t_smooth_tff;
      if (grace > 0.25 * tf) { grace = 0.0; grace_refused = true; }
    }

    std::printf("Rayleigh-Benard, free-fall scaling   Kokkos   %s fluid / %s scalar\n",
                FL::name, SL::name);
    std::printf("  operator %s + %s   %s   %lld x %lld x %lld   H = %lld"
                "   %.3e cells\n", FluidColl::name, ScalColl::name,
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
    std::printf("  initial condition: %s\n", (ic == "cond")
                ? "conductive profile (no discontinuity)"
                : "cold layer, whole dT across the bottom half cell (the reference's)");
    std::printf("  one free-fall time = %.0f steps;  %zu steps = %.0f of them\n",
                t_ff, T_end, tf);
    std::printf("  RESOLUTION: Nu ~ %.0f (2-D, 0.14 Ra^0.29) -> thermal BL = %.4f"
                " cells, Nu ceiling ~ H/2 = %lld.  %s\n",
                Nu_est, bl_cells, (long long)(H / 2),
                bl_cells >= 10.0 ? "Resolved."
                                 : "UNDER-RESOLVED: Nu below is the scheme, not Ra.");
    std::printf("  max-principle halt at T outside [%.2f, %.2f] (slack %.3g dT)",
                T_cold - slack * dT, T_hot + slack * dT, slack);
    if (grace_refused)
      std::printf(", enforced from step 0.\n"
                  "     ** GRACE WINDOW REFUSED: the step needs %.3g t_ff to diffuse over "
                  "four cells,\n        which is %.1f%% of this %.0f t_ff run -- more than "
                  "the quarter of it this\n        window is allowed to cover. The "
                  "undershoot is not a transient on THIS run's\n        timescale, so the "
                  "halt stands and the run may stop early. Lengthen -tf,\n        use "
                  "-ic cond, raise H, or force a window with -grace N. **",
                  t_smooth_tff, 100.0 * t_smooth_tff / tf, tf);
    else if (grace > 0.0)
      std::printf(", suppressed for the\n     first %.1f t_ff -- the time the step needs "
                  "to diffuse over four cells.\n     Out-of-bounds lines are marked "
                  "`!`, never hidden; nan and max|u| > 1 still halt.", grace);
    else
      std::printf(", enforced from step 0.");
    std::printf("\n\n");

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
    // ================== THE INITIAL CONDITION IS A FLAG ==================
    // `cold` is the reference's: the whole layer at T_cold, so the ENTIRE dT is
    // dropped across the bottom half cell at t = 0. `cond` is the conductive
    // profile, the same end states with no discontinuity anywhere.
    //
    // ============ AND THE SEED HAS TO GO WHERE THE GRADIENT IS =============
    // MEASURED, and it is the reason this paragraph exists. The fluid's density
    // seed sits at mid-depth, y = ny/2, which is exactly where the critical
    // mode peaks for a CONDUCTIVE start -- and is useless for a COLD one. A
    // cold start has no gradient anywhere except in the diffusing layer at the
    // bottom plate, and that layer is thin:
    //
    //     delta = sqrt(D t) = 2.55 cells at t/t_ff = 7   (H = 498, Ra = 1e11)
    //
    // so a perturbation at y = 250 sits in NEUTRALLY STRATIFIED fluid with no
    // buoyancy to act on it. Measured on exactly that run: the k = 1 mode
    // decayed monotonically, 9.69e-05 -> 6.26e-05 over seven free-fall times,
    // with its peak pinned at y = 250 the whole way -- while the conductive
    // start at the same parameters had it GROWING at 1.51x per free-fall time
    // with the peak locked at y = 364. For the mid-depth seed to matter the
    // layer would have to reach 250 cells, which takes 250^2/D = 6.7e8 steps.
    // It never happens.
    //
    // So `cold` seeds the TEMPERATURE inside the boundary layer instead:
    //
    //     T = T_cold + amp dT * exp(-(y-1)/d0) * (1 + sum_k cos(2 pi k x/nx))/2
    //
    // with d0 = 4 cells and k over a resolvable band. Three properties, each
    // deliberate. It is NON-NEGATIVE -- the (1 + cos)/2 form keeps the whole
    // field inside [T_cold, T_cold + amp dT], so the initial condition does not
    // itself violate the maximum principle that this driver uses as its stop
    // rule. It is BROADBAND rather than single-mode, because a cold start's
    // instability belongs to the layer and selects its own wavelength from
    // delta, not from the box: the critical wavelength for a layer of depth
    // delta is about 2 delta, i.e. FIVE CELLS here, and seeding one mode of
    // wavelength nx would be answering a question the layer is not asking. And
    // it is DETERMINISTIC, so two runs with the same flags agree bit for bit.
    //
    // `cond` is untouched -- same density seed at mid-depth, no temperature
    // perturbation -- so every number already measured with it still stands.
    //
    // AND IT WORKS, measured at 200 x 100, Ra = 1e11, cold, out to 64 free-fall
    // times. The horizontal fluctuation rms of T grows from the seed at
    // 1.47x per free-fall time (sigma = 0.38/t_ff) from t = 2, saturating
    // around t = 12, with its peak row at y = 1 to 2 -- AT THE HOT PLATE, which
    // is where a cold start's instability lives. The old mid-depth seed on the
    // same physics decayed instead. max|u| went 4.9e-04 -> 7.4e-03 between
    // t = 10 and 15, and Nu_top lifted off zero at t ~ 25 as the first plumes
    // crossed. Compare sigma = 0.45/t_ff for the conductive start at H = 498:
    // the same order, which is the check that this is the same instability and
    // not an artefact of the seed.
    // ====================================================================
    const bool cold = (ic != "cond");
    const Real Tc = Real(T_cold), Th = Real(T_hot);
    const Index nyc = ny, Hc = H, nxs = nx;
    const Real ampT = Real(amp) * Real(dT);
    th.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Index y = py - d.hy, x = px - d.hx;
      if (y <= 0 || y >= nyc - 1) return Real(0);
      if (cold) {
        // Four modes at wavelengths nx/8 .. nx/64, all resolvable, with fixed
        // offsets so they do not all peak at the same x.
        double m = 0.0;
        for (int j = 0; j < 4; ++j) {
          const double k = double(8 << j);
          m += Kokkos::cos(2.0 * M_PI * k * double(x) / double(nxs) + 0.7 * j);
        }
        const double env = Kokkos::exp(-(double(y) - 1.0) / 4.0);
        return Real(double(Tc) + double(ampT) * env * (1.0 + m / 4.0) * 0.5);
      }
      // The hot plane sits at y = 0.5, so the conductive profile is linear in
      // (y - 0.5)/H and lands on the plate values at both half-cell planes.
      return Real(double(Th) - (double(y) - 0.5) / double(Hc) * (double(Th) - double(Tc)));
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
    // THE TRACE GOES STRAIGHT TO EQUILIBRIUM, which is what the reference does
    // (its k4 = R, rate 1) and what GPU/'s twin does (its Solver defaults
    // omega_bulk to 1). This driver was leaving omega_bulk at -1, i.e. "follow
    // omega", so the bulk viscosity was relaxed at 1.9984 while both siblings
    // relaxed it at 1 -- a third configuration belonging to neither. The trace
    // carries no shear physics here, so sending it to equilibrium is both the
    // reference's choice and the better-damped one.
    fcoll.omega_bulk = Real(1);
    fcoll.forcing = force;
    FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fcoll);
    fl.set_geometry([&](Index, Index y, Index) -> CellType {
      return (y == 0 || y == ny - 1) ? Solid : Fluid;
    });
    // At rest, with one single-mode density perturbation on the row nearest
    // mid-depth -- the reference's seed, and deliberately not noise: the
    // question is whether THIS mode grows.
    // The density seed is CONDUCTIVE-ONLY now. For a cold start it sits in
    // neutrally stratified fluid and decays -- see the initial-condition banner
    // above -- and the temperature seed there does the work instead. Leaving it
    // on for `cold` would add an acoustic transient that buys nothing.
    const Index seed_row = ny / 2;
    const Real ampr = cold ? Real(0) : Real(amp);
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

    std::printf("  %10s %11s %10s %10s %10s %10s %10s %8s %8s %9s\n",
                "t/t_ff", "Nu_vol", "Nu_floor", "Nu_bot", "Nu_top", "Nu_ref",
                "max|u|", "T_min", "T_max", "residual");

    std::vector<Real> Tprev;
    int frame = 0;
    bool bad = false;
    std::size_t steps_run = T_end;
    // The worst bounds excursion of the whole run, and when. Tracked whether or
    // not it halts, so a run that finishes inside the window still reports it:
    // with a grace window "it completed" no longer implies "it stayed in
    // bounds", and a Nu measured while T was outside them is not a measurement.
    double worst_lo = T_cold, worst_hi = T_hot, worst_lo_t = 0.0, worst_hi_t = 0.0;
    double last_lo = T_cold, last_hi = T_hot;
    bool   ever_out = false;
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
        long nbad = 0;                          // non-finite T, see below
        double mv = 0, mt = 0;                  // <u_y> and <T>, see below
        double qv = 0, qt = 0;                  // and their mean squares
        for (Index z = 0; z < nz; ++z)
          for (Index y = 1; y <= H; ++y)
            for (Index x = 0; x < nx; ++x) {
              const Index n = d.id(x, y, z);
              const double Tv = double(hT(n));
              flux += double(huy(n)) * Tv;
              mv += double(huy(n));  mt += Tv;
              qv += double(huy(n)) * double(huy(n));  qt += Tv * Tv;
              const double s = std::sqrt(double(hux(n)) * double(hux(n)) +
                                         double(huy(n)) * double(huy(n)) +
                                         double(huz(n)) * double(huz(n)));
              if (s > peak) peak = s;
              // A nan compares false BOTH ways, so a nan field left tmin and
              // tmax at their sentinels and the run printed +1e300 and -1e300
              // as the temperature range -- a number that looks like data.
              if (!std::isfinite(Tv)) ++nbad;
              else { if (Tv < tmin) tmin = Tv; if (Tv > tmax) tmax = Tv; }
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

        // ============ Nu_vol HAS A NOISE FLOOR, AND IT IS LARGE =============
        // Nu_vol carries a factor H/D, and at Ra = 1e14 that is 1.7e8. So an
        // ACCIDENTAL correlation between the velocity noise and T is amplified
        // enormously: two uncorrelated fields of scale u' and T' still leave a
        // sample mean of order u' T' / sqrt(N), hence
        //
        //     Nu_floor ~ H u'_rms T'_rms / (D dT sqrt(N)).
        //
        // THIS IS A LOWER BOUND, NOT THE NOISE, and the gap was measured
        // rather than assumed. On a 200 x 98 layer with the conductive initial
        // condition -- a state that is NOT convecting, where both plate
        // estimators correctly sit at 1.0, T stays inside its bounds and
        // max|u| is 1e-03 -- Nu_vol ran 180, +606, +1216, +1719 at Ra = 1e14
        // and 2.79, -1.79, +7.07, -5.33, +13.1, -10.4, +18.2 at Ra = 1e10.
        // Against the floor below that is a factor of 28 and 36 respectively:
        // CONSISTENT across a hundredfold change in D, which says the spurious
        // velocity is spatially correlated over some 900 cells rather than
        // independent per cell, so dividing by sqrt(N) is too generous by
        // sqrt(900).
        //
        // AND IT IS A WEAK INSTRUMENT, weaker than the paragraph above first
        // claimed. On 604 x 302 at Ra = 1e10, in the pure-conduction phase where
        // both plate estimators agree to four decimals at 1.0, Nu_vol ran -8.7,
        // +35.2, -28.0, +52.0, -44.7, +58.1, -55.8 against a floor of 0.10 to
        // 0.18 -- eighty to three hundred times ABOVE its floor and still
        // unambiguously noise. So the floor rules out almost nothing at this
        // resolution. Read it only as a hard lower bound, never as a threshold
        // for belief; the two reliable signals are the ones below.
        //
        // Two signatures are sharper than any noise model, and both are visible
        // in those numbers. Nu_vol ALTERNATES SIGN -- a convective flux is
        // positive in the mean because hot fluid rises, so a sign-flipping
        // Nu_vol is noise by construction. And Nu_bot and Nu_top AGREE WITH
        // EACH OTHER while disagreeing with Nu_vol by three orders. That
        // disagreement is the error bar, and this file's parent already says so:
        // "the two estimators must agree, or the run is simply not converged."
        // When they do and Nu_vol does not, believe them.
        // ====================================================================
        const double vrms = std::sqrt(std::max(qv / ncell - mv * mv, 0.0));
        const double trms = std::sqrt(std::max(qt / ncell - mt * mt, 0.0));
        const double Nu_floor =
            double(H) * vrms * trms / (D * dT * std::sqrt(ncell));
        const double Nu_bot = double(H) * (T_hot - bot / plate) / (0.5 * dT);
        const double Nu_top = double(H) * (top / plate - T_cold) / (0.5 * dT);

        // THE REFERENCE'S OWN NORMALISATION, VERBATIM, so the two can be put in
        // one table. It sums u_y T over EVERY node -- wall rows included -- and
        // divides by (nx - 1) rather than by ncell/H, which for its 101 x 51
        // grid is 100 against 103.02, so its Nu - 1 runs 3.0% high. It is also
        // the RAW correlation, not the fluctuation one, so it inherits the
        // artefact described above.
        //
        // TWO THINGS IT DOES NOT REPRODUCE, and they matter before anyone lines
        // the columns up. The reference's wall nodes are FLUID nodes that
        // collide and get forced, so each carries u_y = gb (T - T0)/2 at rest
        // and its bottom row alone contributes 0.25 gb / D to Nu -- +8.4 at its
        // own parameters, where the answer is 1. Here the wall rows are Solid,
        // never collide, and the sum below runs y = 1..H, so that term is
        // absent. And this gauge is symmetric about zero while the reference's
        // is [0, 1], so <T> differs by T0 between them. Nu_ref is the
        // reference's ARITHMETIC on this code's field, which is the honest
        // half of the comparison; the offset is not portable.
        const double Nu_ref = 1.0 + flux / (D * dT * double(nx - 1) * double(nz));

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

        // Track the excursion before anything decides whether to stop.
        const double now_tff = double(t) / t_ff;
        if (!nbad) {
          if (tmin < worst_lo) { worst_lo = tmin; worst_lo_t = now_tff; }
          if (tmax > worst_hi) { worst_hi = tmax; worst_hi_t = now_tff; }
          last_lo = tmin;  last_hi = tmax;
          if (tmin < T_cold || tmax > T_hot) ever_out = true;
        }
        // `!` marks an out-of-bounds line whether or not the window is letting
        // the run continue, so the window can only make an excursion
        // non-fatal, never invisible.
        const char* mark = (!nbad && (tmin < T_cold || tmax > T_hot)) ? " !" : "";

        if (nbad) {
          std::printf("  %10.2f %11s %10s %10s %10s %10s %10s %8s %8s %9s   "
                      "(%ld of %.0f cells non-finite)\n",
                      double(t) / t_ff, "nan", "-", "-", "-", "-", "-", "nan",
                      "nan", "-", nbad, ncell);
        } else {
          std::printf("  %10.2f %11.4f %10.4f %10.4f %10.4f %10.4f %10.3e %8.4f"
                      " %8.4f %9.2e%s\n",
                      double(t) / t_ff, Nu_vol, Nu_floor, Nu_bot, Nu_top, Nu_ref,
                      peak, tmin, tmax, resid, mark);
        }
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

        // THE MAXIMUM PRINCIPLE, AS A STOPPING RULE. See the banner: it
        // catches the failure well before the Nusselt numbers look wrong.
        // Three failures halt unconditionally because none of them recovers;
        // only the bounds excursion is gated by the window, and only up to
        // `slack`. Beyond four times the physical range it stops regardless --
        // a scalar smoothing a step undershoots by a fraction of dT, not by
        // 4 dT, so that is a different failure wearing the same symptom.
        const bool fatal = nbad || !std::isfinite(Nu_vol) || peak > 1.0;
        const double m = slack * dT;
        const bool out_soft = (tmin < T_cold - m)         || (tmax > T_hot + m);
        const bool out_hard = (tmin < T_cold - 4.0 * dT)  || (tmax > T_hot + 4.0 * dT);
        const bool in_grace = now_tff < grace;
        if (fatal || out_hard || (out_soft && !in_grace)) {
          std::printf("  STOPPED at t = %zu: %s\n", t,
                      nbad ? "T not finite"
                      : !std::isfinite(Nu_vol) ? "Nu not finite"
                      : peak > 1.0 ? "max|u| > 1"
                      : out_hard ? "T outside FOUR times its physical range"
                      : "T outside twice its physical range");
          bad = true; steps_run = t; rc = 1;
          break;
        }
        if (out_soft && in_grace)
          std::printf("     (out of bounds by %.4f dT, inside the %.1f t_ff grace "
                      "window -- continuing)\n",
                      std::max(T_cold - tmin, tmax - T_hot) / dT, grace);
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

    // THE MAXIMUM-PRINCIPLE VERDICT, ALWAYS PRINTED, for the reason given where
    // the tracking is declared.
    if (!ever_out) {
      std::printf("  maximum principle: T stayed inside [%.2f, %.2f] throughout.\n",
                  T_cold, T_hot);
    } else {
      const bool recovered = (last_lo >= T_cold) && (last_hi <= T_hot);
      std::printf("  maximum principle: VIOLATED. worst T_min = %.4f at t/t_ff = %.2f,"
                  "  worst T_max = %.4f at t/t_ff = %.2f\n"
                  "     (%.4f dT below / %.4f dT above the physical range)\n"
                  "     by the end: T in [%.4f, %.4f] -- %s\n",
                  worst_lo, worst_lo_t, worst_hi, worst_hi_t,
                  (T_cold - worst_lo) / dT, (worst_hi - T_hot) / dT,
                  last_lo, last_hi,
                  recovered ? "RECOVERED, so treat only the in-bounds tail as data"
                            : "STILL OUT OF BOUNDS: nothing in this run is a "
                              "measurement of Ra");
    }
  }
  return rc;
}

int main(int argc, char** argv) {
  Opts o;
  o.ny = 1000; o.nx = 2000; o.nz = 1;
  o.Ra = 1e14; o.Pr = 0.71; o.U = 0.05; o.amp = 0.01;
  o.tf = 40.0; o.out_every = 1.0;
  o.slack = 0.5;                 // max-principle halt margin, in units of dT
  o.grace = -1.0;                // < 0 means "choose from the IC"
  o.ic = "cond";                 // matches GPU/rb_high_ra; `cold` is the reference's
  std::string sop = "reg";       // ScalarRegularised, matching GPU/'s default

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-ny"    && i + 1 < argc) o.ny        = Index(std::atol(argv[++i]));
    if (a == "-nx"    && i + 1 < argc) o.nx        = Index(std::atol(argv[++i]));
    if (a == "-nz"    && i + 1 < argc) o.nz        = Index(std::atol(argv[++i]));
    if (a == "-ra"    && i + 1 < argc) o.Ra        = std::atof(argv[++i]);
    if (a == "-pr"    && i + 1 < argc) o.Pr        = std::atof(argv[++i]);
    if (a == "-u"     && i + 1 < argc) o.U         = std::atof(argv[++i]);
    if (a == "-amp"   && i + 1 < argc) o.amp       = std::atof(argv[++i]);
    if (a == "-tf"    && i + 1 < argc) o.tf        = std::atof(argv[++i]);
    if (a == "-out"   && i + 1 < argc) o.out_every = std::atof(argv[++i]);
    if (a == "-slack" && i + 1 < argc) o.slack     = std::atof(argv[++i]);
    if (a == "-grace" && i + 1 < argc) o.grace     = std::atof(argv[++i]);
    if (a == "-dump"  && i + 1 < argc) o.dump      = argv[++i];
    if (a == "-ic"    && i + 1 < argc) o.ic        = argv[++i];
    if (a == "-sop"   && i + 1 < argc) sop         = argv[++i];
  }

  Kokkos::initialize(argc, argv);
  // The scalar operator is a TEMPLATE parameter, so the flag dispatches to two
  // instantiations rather than switching a branch. That is the parent tree's
  // convention -- collision operators are types, not enums -- and it is why
  // `run` is a template and `Opts` sits at namespace scope.
  const int rc = (sop == "bgk") ? run<ScalarBGK<SL>>(o)
                                : run<ScalarRegularised<SL>>(o);
  Kokkos::finalize();
  return rc;
}
