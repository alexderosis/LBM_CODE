//==============================================================================
//  Laplace's law, and the interface the phase field actually maintains.
//
//  The first validation of the two-phase module, and deliberately the first:
//  a static droplet has an exact answer, it needs no reference table, and the
//  two things it measures fail independently.
//
//   1. THE INTERFACE WIDTH. The conservative Allen-Cahn equation has the tanh
//      profile of prescribed width W as its exact one-dimensional equilibrium,
//      so a flat interface must neither spread nor sharpen. W is recovered from
//      an integral rather than from a fit,
//
//          integral phi (1 - phi) dy = W / 4   per interface,
//
//      because for phi = 1/2 [1 + tanh(2y/W)] the integrand is
//      (1/4) sech^2(2y/W), whose integral is W/4 exactly. An integral measure
//      uses every node in the profile instead of the two or three near the
//      steepest point, so it does not degrade as W shrinks toward the grid.
//
//      This is second-order accurate, not exact: the CONTINUUM equilibrium is
//      the tanh profile, and the lattice converges to it. Do not expect
//      round-off here.
//
//   2. THE PRESSURE JUMP. Across a curved interface,
//
//          dp = sigma (D - 1) / R,
//
//      so measuring dp at several R and fitting the slope against 1/R recovers
//      sigma with no free parameter. What makes this a real test of the
//      capillary stress rather than of the initial condition is that sigma is
//      set through kappa = 3 sigma W / 2, a coefficient in a tensor added to
//      the equilibrium's second moment -- nothing in the code ever handles a
//      surface tension as such, so the number coming back out is a statement
//      about the whole chain.
//
//      Far from the interface grad phi is zero and the capillary stress with
//      it, so the total normal stress there is just rho cs^2. The jump is
//      therefore read straight off the density field, deep inside against far
//      outside, with the interface region excluded from both.
//
//   3. SPURIOUS CURRENTS, reported rather than asserted away. A static droplet
//      has an exact solution u = 0, and the residual velocity is the standard
//      figure of merit for a multiphase scheme -- it is what the isotropy of
//      the gradient stencil buys, and it is the number to watch when anything
//      about the stencil or the stress changes.
//
//  CONVERGENCE IS SHOWN, NOT ASSUMED. dp is reported at half the run and at the
//  end. If those two disagree the droplet is still equilibrating and the fitted
//  sigma means nothing, whatever it happens to equal.
//
//  WHAT THE MEASURED SIGMA IS, AND WHY IT IS LOW.
//
//  At the default resolution the recovered sigma sits about 6-7% BELOW the
//  prescribed value, consistently at every radius. That is a discretisation
//  error, not a wrong coefficient, and `-refine` demonstrates it: holding the
//  droplet's shape fixed (R/W = 5, R/N fixed) and refining the lattice gives
//
//      N=96  R=20 W=4   sigma 9.316e-03   err 6.84e-02
//      N=144 R=30 W=6   sigma 9.689e-03   err 3.11e-02      order 1.94
//      N=192 R=40 W=8   sigma 9.837e-03   err 1.63e-02      order 2.25
//
//  i.e. second order, which is what the scheme is. The absolute value at any one
//  resolution is therefore not the assertion; the 1/R LAW is, and it holds to
//  about 1% across the sweep -- sigma = dp R is constant even while its value is
//  6% low, which is exactly the signature of a resolution error rather than a
//  broken stress.
//
//  DO NOT TUNE W TO MAKE THE ERROR SMALL. At FIXED R two errors of opposite
//  sign compete: under-resolution of the profile, O((h/W)^2) and negative, and
//  the finite thickness of the interface against the curvature, O(W/R) and
//  positive. Measured at R = 28: -6.51e-02 (W=4), -2.68e-02 (W=6), -0.97e-02
//  (W=8), +1.00e-02 (W=12). The deficit passes through zero near W = 10, and a
//  W chosen there would look excellent while measuring nothing but the
//  cancellation. Refine both together, as -refine does.
//
//  THE PHASE LATTICE. -plat d2q5 runs the reduced lattice with a first-order
//  equilibrium, d2q9 (the default) the full one with De Rosis & Enan's Eq. (11).
//  A static droplet cannot tell them apart in sigma -- 9.738e-03 against
//  9.722e-03 at R=20, W=6, since u is zero and the equilibrium's u-truncation
//  has nothing to act on -- but the spurious currents halve, 3.63e-05 to
//  1.64e-05. The advection accuracy the second-order form buys needs a moving
//  interface to show, which this case does not have.
//
//  SCOPE. Matched density and matched viscosity -- see MultiphaseBGK. This case
//  isolates surface tension and nothing else.
//==============================================================================
#include "collision/MultiphaseBGK.hpp"
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/MultiphasePotentialBGK.hpp"
#include "collision/PhaseFieldBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/PhaseFieldSolver.hpp"
#include "solver/ViscousInterfaceForce.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace lbm;

using FL = D2Q9;

//------------------------------------------------------------------------------
// The two ways of putting surface tension into the flow, as tags. They are
// independent implementations of the same physics and must agree at a matched
// density; only the second reaches a density ratio.
//
//   StressForm    -- the capillary tensor in the equilibrium's second moment.
//                    Carries density in the zeroth moment, as every single-phase
//                    operator in this code base does.
//   PotentialForm -- De Rosis & Enan: F_s = mu_phi grad phi as a body force, on
//                    a distribution whose zeroth moment is a PRESSURE.
//------------------------------------------------------------------------------
struct StressForm {};
struct PotentialForm {};
struct CentralForm {};        // the potential form, collided in central moments

using StressColl = MultiphaseBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations>;
using PotColl    = MultiphasePotentialBGK<FL, SecondOrderPhi<FL>, RawPopulations>;
using CmColl     = MultiphaseCentralMoments<FL>;

// The phase lattice is a parameter, not a decision baked into the case. D2Q5
// with a first-order equilibrium is the cheap route; D2Q9 with the second-order
// one is what De Rosis & Enan run (their D3Q19, reduced to two dimensions).
// Select with -plat.
template <class PL> using PhaseColl = PhaseFieldBGK<PL>;
template <class PL> using PhaseSlv  = PhaseFieldSolver<PL, EsotericPull<PL>, PhaseColl<PL>>;

struct Case {
  double dp = 0, dp_half = 0;   // pressure jump at the end and at half the run
  double sigma = 0;             // dp R / (D - 1)
  double umax = 0;              // spurious current
  double drift = 0;             // relative change in the conserved sum of phi
  double width = 0;             // measured interface width, flat case only
  // Diagnostics: if the droplet has moved off its prescribed size or phi has
  // drifted off its plateau values, a pressure jump measured against the
  // NOMINAL radius is being asked the wrong question.
  double r_eff = 0;             // sqrt(sum phi / pi), the area-equivalent radius
  double phi_in = 0, phi_out = 0;
  double p_in = 0, p_out = 0;
  bool ok = false;
};

//------------------------------------------------------------------------------
// One run. `radius <= 0` means a flat slab instead of a droplet, which is the
// zero-curvature control: same code path, same coupling, and dp must come out
// zero rather than "small".
//------------------------------------------------------------------------------
template <class PL, class Form = StressForm>
static Case run(Index N, double radius, double sigma, double W, double tau,
                double M, std::size_t nsteps, double ratio = 1.0) {
  // Both potential-family operators expose the same fields, so one setup block
  // serves them; only the collision differs.
  constexpr bool cmf = std::is_same_v<Form, CentralForm>;
  constexpr bool pot = cmf || std::is_same_v<Form, PotentialForm>;
  using FColl    = std::conditional_t<cmf, CmColl,
                     std::conditional_t<pot, PotColl, StressColl>>;
  using FluidSlv = FluidSolver<FL, EsotericPull<FL>, FColl>;

  Domain d(N, N, 1, true, true, true);

  PhaseColl<PL> pc;
  pc.omega = PhaseColl<PL>::omega_from_mobility(Real(M));
  pc.width = Real(W);
  PhaseSlv<PL> pf(d, pc);

  const Real ctr = Real(0.5) * Real(N - 1);
  const Real Rr  = Real(radius);
  const Real Ww  = Real(W);
  const bool drop = radius > 0;
  pf.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real x = Real(px - d.hx) - ctr;
    const Real y = Real(py - d.hy) - ctr;
    // Droplet: phi = 1 inside. Slab: phi = 1 for |y| < N/4, two interfaces.
    const Real s = drop ? (Rr - Kokkos::sqrt(x * x + y * y))
                        : (Real(0.25) * Real(N) - Kokkos::fabs(y));
    return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * s / Ww));
  });

  // Heavy fluid inside the droplet (phi = 1). Equal KINEMATIC viscosity in both
  // phases, so mu scales with rho -- the simplest choice that leaves tau uniform
  // and isolates the density ratio from a viscosity ratio.
  const double nu = (tau - 0.5) / 3.0;
  const double rho_l = 1.0, rho_h = ratio;

  // Constructed before the fluid so its output views exist to bind into the
  // collision; the fluid solver copies the collision by value, so binding after
  // that would not propagate.
  ViscousInterfaceForce<FL> vf(d);

  FColl fc;
  if constexpr (pot) {
    fc.phi = pf.phi();
    fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
    fc.Lap = pf.laplacian();
    fc.Vx = vf.x();  fc.Vy = vf.y();  fc.Vz = vf.z();
    fc.rho_L = Real(rho_l);      fc.rho_H = Real(rho_h);
    fc.mu_L  = Real(rho_l * nu); fc.mu_H  = Real(rho_h * nu);
    fc.kappa = FColl::kappa_from_sigma(Real(sigma), Real(W));
    fc.beta  = FColl::beta_from_sigma(Real(sigma), Real(W));
  } else {
    fc.omega = StressColl::omega_from_viscosity(Real(nu));
    fc.kappa = StressColl::kappa_from_sigma(Real(sigma), Real(W));
    fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
  }
  FluidSlv fl(d, fc);

  // THE PRESSURE GAUGE IS NOT FREE HERE, EVEN THOUGH THE PHYSICS IS.
  //
  // Only grad p is physical, so the absolute level P0 may be chosen at will --
  // but p~ = p / (rho cs2) is what the populations carry, and shifting p by P0
  // shifts p~ by P0/(rho cs2), which is NOT a constant when rho varies. The two
  // large terms that must cancel in the interface, rho cs2 grad p~ and
  // F_p = -p~ cs2 grad rho, are both PROPORTIONAL to the gauge. Their difference
  // is the physical answer at any P0; their individual sizes, and so the size of
  // the discrete mismatch between them, are not.
  //
  // Seeding p = rho_L cs2 (p~ from 1 to 1/ratio) therefore makes the scheme far
  // worse conditioned than it needs to be. Seeding p = 0 makes both terms small
  // and leaves only the Laplace jump in p~. Measured below.
  if constexpr (pot) {
    fl.initialize(Real(0));
  } else {
    fl.initialize(Real(1));
  }
  pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());
  // F_nu is identically zero at a matched density, so the extra macroscopic
  // pass it needs is skipped there rather than run to produce zeros.
  const bool need_vf = pot && ratio != 1.0;

  const double phi0 = double(pf.total_population());

  //----------------------------------------------------------------------------
  // Pressure jump from the density field. Deep inside is r < R - 2W and far
  // outside is r > R + 2W, so the interface contributes to neither.
  //----------------------------------------------------------------------------
  double c_probe_in = 0, c_probe_out = 0;

  // The physical pressure differs by form, and this is the whole reason the
  // zeroth moment has to be read carefully: the stress form stores rho, so
  // p = rho cs2; the potential form stores p~, so p = rho(phi) cs2 p~ with rho
  // interpolated from the order parameter. Reading fl.rho() as a density under
  // the potential form would silently measure the wrong quantity.
  auto pressure_jump = [&]() -> double {
    if (!drop) return 0.0;
    fl.compute_macroscopic();
    auto h  = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
    const double lo = radius - 2.0 * W, hi = radius + 2.0 * W;
    double si = 0, so = 0;
    long ni = 0, no = 0;
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index id = d.id(x, y);
        // p = rho(phi) cs2 p~: this case runs the DEFAULT local-rho
        // normalisation, so the local density is what converts p~ to a
        // pressure. Reconstructing with a constant instead silently reports
        // the wrong pressure jump -- measured, it turned a 1.7% error at a
        // ratio of 100 into 19.5%, with the physics untouched.
        double pv = double(h(id)) / 3.0;
        if constexpr (pot) pv *= rho_l + double(hf(id)) * (rho_h - rho_l);
        const double dx = double(x) - double(ctr), dy = double(y) - double(ctr);
        const double r = std::sqrt(dx * dx + dy * dy);
        if (r < lo)      { si += pv; ++ni; }
        else if (r > hi) { so += pv; ++no; }
      }
    if (ni == 0 || no == 0) return std::nan("");
    c_probe_in = si / double(ni);
    c_probe_out = so / double(no);
    return c_probe_in - c_probe_out;
  };

  //----------------------------------------------------------------------------
  // The coupling order the module requires: phi and grad phi first, then the
  // fluid against them, then the interface with the velocity it just produced.
  // See the header of PhaseFieldSolver.hpp.
  //----------------------------------------------------------------------------
  Case c;
  for (std::size_t k = 0; k < nsteps; ++k) {
    pf.refresh();
    if constexpr (pot) {
      if (need_vf) { fl.compute_macroscopic(); vf.refresh(fc); }
    }
    fl.step(true);
    pf.step();
    if (k + 1 == nsteps / 2) c.dp_half = pressure_jump();
  }
  pf.refresh();
  c.dp = pressure_jump();
  c.sigma = drop ? c.dp * radius : 0.0;         // 2D: dp = sigma / R
  c.drift = std::abs(double(pf.total_population()) - phi0) / std::abs(phi0);

  fl.compute_macroscopic();
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
  (void)0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const double a = double(hu(d.id(x, y))), b = double(hv(d.id(x, y)));
      c.umax = std::max(c.umax, std::sqrt(a * a + b * b));
      if (!std::isfinite(a) || !std::isfinite(b)) return c;
    }

  c.p_in = c_probe_in;  c.p_out = c_probe_out;

  // Where the interface actually ended up. sum phi over the domain is pi R^2 in
  // two dimensions, so an area-equivalent radius costs one reduction and says
  // immediately whether the droplet held its size.
  if (drop) {
    double sphi = 0, ain = 0, aout = 0;
    long nin = 0, nout = 0;
    const double lo = radius - 2.0 * W, hi = radius + 2.0 * W;
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const double v = double(hp(d.id(x, y)));
        sphi += v;
        const double dx = double(x) - double(ctr), dy = double(y) - double(ctr);
        const double r = std::sqrt(dx * dx + dy * dy);
        if (r < lo)      { ain += v; ++nin; }
        else if (r > hi) { aout += v; ++nout; }
      }
    c.r_eff = std::sqrt(std::max(sphi, 0.0) / M_PI);
    c.phi_in = nin ? ain / double(nin) : 0.0;
    c.phi_out = nout ? aout / double(nout) : 0.0;
  }

  // Interface width, from the integral identity in the header. The slab has two
  // interfaces in the column, so the column integral is W/2.
  if (!drop) {
    double acc = 0;
    for (Index y = 0; y < N; ++y) {
      const double p = double(hp(d.id(N / 2, y)));
      acc += p * (1.0 - p);
    }
    c.width = 2.0 * acc;
  }
  c.ok = std::isfinite(c.dp);
  return c;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    Index N = 96;
    double sigma = 0.01, W = 4.0, tau = 0.8, M = 0.05, radius = 0.0;
    bool full_lattice = true;       // -plat d2q5 for the reduced lattice
    bool refine = false;            // -refine: the grid-refinement ladder
    bool pot_main = false;          // -form potential: drive the main sweep with it
    double ratio = 1.0;             // -ratio: density ratio, potential form only
    bool use_cm = false;            // -op cm: collide the potential form in CMs
    std::size_t nsteps = 10000;
    for (int i = 1; i < argc; ++i) {
      auto next = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-n"))     { if (i + 1 < argc) N = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-steps")) { if (i + 1 < argc) nsteps = std::size_t(std::atol(argv[++i])); }
      else if (!std::strcmp(argv[i], "-sigma")) next(sigma);
      else if (!std::strcmp(argv[i], "-w"))     next(W);
      else if (!std::strcmp(argv[i], "-tau"))   next(tau);
      else if (!std::strcmp(argv[i], "-m"))     next(M);
      else if (!std::strcmp(argv[i], "-r"))     next(radius);
      else if (!std::strcmp(argv[i], "-plat")) {
        if (i + 1 < argc) full_lattice = !std::strcmp(argv[++i], "d2q9");
      }
      else if (!std::strcmp(argv[i], "-refine")) refine = true;
      else if (!std::strcmp(argv[i], "-ratio")) next(ratio);
      else if (!std::strcmp(argv[i], "-op")) {
        if (i + 1 < argc) use_cm = !std::strcmp(argv[++i], "cm");
      }
      else if (!std::strcmp(argv[i], "-form")) {
        if (i + 1 < argc) pot_main = !std::strcmp(argv[++i], "potential");
      }
    }

    auto go = [&](double R, std::size_t st) {
      if (pot_main && use_cm)
        return full_lattice ? run<D2Q9, CentralForm>(N, R, sigma, W, tau, M, st, ratio)
                            : run<D2Q5, CentralForm>(N, R, sigma, W, tau, M, st, ratio);
      if (pot_main)
        return full_lattice ? run<D2Q9, PotentialForm>(N, R, sigma, W, tau, M, st, ratio)
                            : run<D2Q5, PotentialForm>(N, R, sigma, W, tau, M, st, ratio);
      return full_lattice ? run<D2Q9, StressForm>(N, R, sigma, W, tau, M, st)
                          : run<D2Q5, StressForm>(N, R, sigma, W, tau, M, st);
    };
    auto go_pot = [&](double R, std::size_t st, double ratio) {
      if (use_cm)
        return full_lattice ? run<D2Q9, CentralForm>(N, R, sigma, W, tau, M, st, ratio)
                            : run<D2Q5, CentralForm>(N, R, sigma, W, tau, M, st, ratio);
      return full_lattice ? run<D2Q9, PotentialForm>(N, R, sigma, W, tau, M, st, ratio)
                          : run<D2Q5, PotentialForm>(N, R, sigma, W, tau, M, st, ratio);
    };
    std::printf("Laplace's law   D2Q9 fluid + %s phase field, conservative Allen-Cahn\n",
                full_lattice ? "D2Q9" : "D2Q5");
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("N = %d   sigma = %g   W = %g   tau = %g   M = %g   %zu steps\n\n",
                int(N), sigma, W, tau, M, nsteps);

    //--------------------------------------------------------------------------
    // Grid refinement with the droplet's SHAPE held fixed -- R/W and R/N both
    // constant, so only the lattice spacing changes. Refining W alone at fixed R
    // does not measure an order, because it moves the interface-thickness error
    // at the same time; see the header. Minutes, not seconds.
    //--------------------------------------------------------------------------
    if (refine) {
      struct Rung { Index n; double r, w; std::size_t steps; };
      const Rung ladder[3] = {{96, 20.0, 4.0, 12000},
                              {144, 30.0, 6.0, 27000},
                              {192, 40.0, 8.0, 48000}};
      std::printf("Grid refinement, R/W = 5 and R/N fixed\n\n");
      std::printf("%-5s %-5s %-4s %-14s %-11s %-8s\n", "N", "R", "W", "sigma", "rel err", "order");
      std::printf("%s\n", std::string(52, '-').c_str());
      double prev_e = 0, prev_n = 0, worst_ord = 1e9;
      bool all_ok = true;
      for (const Rung& g : ladder) {
        const Case c = full_lattice
            ? run<D2Q9, StressForm>(g.n, g.r, sigma, g.w, tau, M, g.steps)
            : run<D2Q5, StressForm>(g.n, g.r, sigma, g.w, tau, M, g.steps);
        const double e = std::abs(c.sigma / sigma - 1.0);
        all_ok = all_ok && c.ok;
        std::printf("%-5d %-5.0f %-4.0f %-14.5e %-11.2e", int(g.n), g.r, g.w, c.sigma, e);
        if (prev_e > 0) {
          const double ord = std::log(prev_e / e) / std::log(double(g.n) / prev_n);
          worst_ord = std::min(worst_ord, ord);
          std::printf(" %-8.2f", ord);
        }
        std::printf("\n");
        prev_e = e; prev_n = double(g.n);
      }
      const bool pass = all_ok && worst_ord > 1.7;
      std::printf("\nacceptance:\n");
      std::printf("  sigma converges at second order      worst %.2f       %s\n",
                  worst_ord, pass ? "PASS" : "FAIL");
      if (!pass) status = 1;
      Kokkos::finalize();
      return status;
    }

    //--------------------------------------------------------------------------
    std::printf("Flat interface (zero curvature): the width must be maintained,\n");
    std::printf("and a flat interface must drive no flow.\n\n");
    const Case flat = go(-1.0, nsteps);
    const double w_err = std::abs(flat.width / W - 1.0);
    std::printf("  W prescribed %.4f   measured %.4f   rel err %.2e\n",
                W, flat.width, w_err);
    std::printf("  max |u| %.3e      phi drift %.3e\n\n", flat.umax, flat.drift);

    //--------------------------------------------------------------------------
    std::printf("Droplet: dp = sigma / R in two dimensions\n\n");
    std::printf("%-6s %-12s %-12s %-12s %-11s %-11s\n",
                "R", "dp (half)", "dp (final)", "sigma = dp R", "rel err", "max |u|");
    std::printf("%s\n", std::string(70, '-').c_str());

    // Radii as fractions of the box, so -n alone rescales the sweep coherently:
    // the pressure sampling needs R - 2W cells of clear interior and the far
    // field needs room beyond R + 2W, and both scale with N.
    std::vector<double> Rs = {N / 8.0, N / 6.0, N / 4.8, N / 4.0};
    if (radius > 0) Rs.assign(1, radius);
    std::vector<Case> cs;
    double worst = 0, worst_u = flat.umax, worst_drift = flat.drift;
    bool finite = flat.ok;
    for (double R : Rs) {
      const Case c = go(R, nsteps);
      cs.push_back(c);
      const double e = std::abs(c.sigma / sigma - 1.0);
      worst = std::max(worst, e);
      worst_u = std::max(worst_u, c.umax);
      worst_drift = std::max(worst_drift, c.drift);
      finite = finite && c.ok;
      std::printf("%-6.1f %-12.5e %-12.5e %-12.5e %-11.2e %-11.2e\n",
                  R, c.dp_half, c.dp, c.sigma, e, c.umax);
      std::printf("       R_eff %-8.3f phi_in %-9.6f phi_out %-9.6f"
                  " p_in %-11.6f p_out %-11.6f\n",
                  c.r_eff, c.phi_in, c.phi_out, c.p_in, c.p_out);
    }

    // Least-squares slope of dp against 1/R through the origin: sigma is the
    // slope, and using every radius at once is less sensitive to the pressure
    // sampling at any one of them than four independent quotients are.
    double num = 0, den = 0;
    for (std::size_t i = 0; i < Rs.size(); ++i) {
      const double k = 1.0 / Rs[i];
      num += k * cs[i].dp;
      den += k * k;
    }
    const double slope = num / den;
    const double slope_err = std::abs(slope / sigma - 1.0);
    std::printf("\n  fitted sigma (slope of dp vs 1/R) = %.6e   prescribed %.6e"
                "   rel err %.2e\n", slope, sigma, slope_err);

    //--------------------------------------------------------------------------
    // The potential form, and the density ratio the stress form cannot reach.
    //
    // The first row is the cross-check that matters: at a matched density the
    // two forms are INDEPENDENT implementations of the same surface tension --
    // one a tensor in the equilibrium's second moment, the other a body force
    // built from a chemical potential and a Laplacian -- sharing no code beyond
    // the phase field that feeds them. They have to agree, and the size of the
    // disagreement is a bound on both.
    //
    // The remaining rows change only rho_H/rho_L. sigma is a property of the
    // interface, so the recovered value must not move with it; if it does, the
    // density is leaking into the surface tension through F_p or the pressure
    // reconstruction. It did, until the pressure gauge was moved to zero -- see
    // the banner in MultiphasePotentialBGK.hpp.
    //
    // The tolerance on the spread is looser than one might like for a reason
    // worth reading off the table: the recovered sigma gets BETTER as the ratio
    // rises, 6.8% low at a matched density against 1.7% at a hundredfold one, so
    // most of the spread is the matched-density row's ordinary resolution
    // deficit rather than any dependence on the ratio.
    //--------------------------------------------------------------------------
    const double R_ref = N / 4.8;
    std::printf("\n\nPotential form (De Rosis & Enan) and the density ratio, at R = %.1f\n\n",
                R_ref);
    std::printf("%-12s %-8s %-13s %-13s %-11s %-11s\n",
                "form", "rho_H", "dp", "sigma", "rel err", "max |u|");
    std::printf("%s\n", std::string(72, '-').c_str());

    const Case ref_stress = cs[2];      // stress form, same radius
    std::printf("%-12s %-8.0f %-13.5e %-13.5e %-11.2e %-11.2e\n",
                "stress", 1.0, ref_stress.dp, ref_stress.sigma,
                std::abs(ref_stress.sigma / sigma - 1.0), ref_stress.umax);

    const std::vector<double> ratios = {1.0, 10.0, 100.0};
    std::vector<Case> pcs;
    double worst_ratio = 0;
    for (double rr : ratios) {
      const Case c = go_pot(R_ref, nsteps, rr);
      pcs.push_back(c);
      finite = finite && c.ok;
      worst_u = std::max(worst_u, c.umax);
      worst_drift = std::max(worst_drift, c.drift);
      const double e = std::abs(c.sigma / sigma - 1.0);
      worst_ratio = std::max(worst_ratio, e);
      std::printf("%-12s %-8.0f %-13.5e %-13.5e %-11.2e %-11.2e\n",
                  use_cm ? "potential/cm" : "potential", rr, c.dp, c.sigma, e, c.umax);
      std::printf("%-12s          R_eff %-8.3f phi_in %-9.6f phi_out %-9.6f"
                  " p_in %-11.6f p_out %-11.6f\n",
                  "", c.r_eff, c.phi_in, c.phi_out, c.p_in, c.p_out);
    }
    const double form_gap = std::abs(pcs[0].sigma / ref_stress.sigma - 1.0);
    // Normalised by the PRESCRIBED sigma, not by the smallest measured one: a
    // measured value that comes back negative would otherwise flip the sign of
    // the ratio and turn a gross failure into an apparent pass. It did.
    double rmin = pcs[0].sigma, rmax = pcs[0].sigma;
    for (const Case& c : pcs) { rmin = std::min(rmin, c.sigma); rmax = std::max(rmax, c.sigma); }
    const double ratio_spread = std::abs(rmax - rmin) / sigma;
    std::printf("\n  the two forms differ by %.2e at matched density\n", form_gap);
    std::printf("  sigma varies by %.2e over a 100x density ratio\n", ratio_spread);

    // Spread of sigma across the radii. THIS is Laplace's law: dp R constant.
    // It is asserted tightly, where the absolute value is not, because the two
    // fail for different reasons -- a broken stress destroys the 1/R law, while
    // a coarse lattice only shifts the constant. See the header.
    double smin = cs[0].sigma, smax = cs[0].sigma, ssum = 0;
    for (const Case& c : cs) {
      smin = std::min(smin, c.sigma);
      smax = std::max(smax, c.sigma);
      ssum += c.sigma;
    }
    const double spread = (smax - smin) / (ssum / double(cs.size()));
    std::printf("  spread of sigma over the four radii = %.2e\n", spread);

    //--------------------------------------------------------------------------
    const double tol_w = 0.01, tol_sp = 0.02, tol_s = 0.10,
                 tol_uf = 1e-6, tol_u = 1e-3, tol_d = 1e-11,
                 tol_fg = 0.02, tol_rs = 0.08;
    const bool pass_w  = finite && w_err < tol_w;
    const bool pass_uf = finite && flat.umax < tol_uf;
    const bool pass_sp = finite && spread < tol_sp;
    const bool pass_s  = finite && slope_err < tol_s;
    const bool pass_u  = finite && worst_u < tol_u;
    const bool pass_d  = finite && worst_drift < tol_d;
    std::printf("\nacceptance:\n");
    std::printf("  interface width within %.0f%%             rel err %.2e     %s\n",
                100 * tol_w, w_err, pass_w ? "PASS" : "FAIL");
    std::printf("  flat interface drives no flow        max |u| %.2e     %s\n",
                flat.umax, pass_uf ? "PASS" : "FAIL");
    std::printf("  dp R constant across R to %.0f%%          spread  %.2e     %s\n",
                100 * tol_sp, spread, pass_sp ? "PASS" : "FAIL");
    std::printf("  sigma within %.0f%% at this resolution   rel err %.2e     %s\n",
                100 * tol_s, slope_err, pass_s ? "PASS" : "FAIL");
    std::printf("  spurious currents < %.0e            worst %.2e     %s\n",
                tol_u, worst_u, pass_u ? "PASS" : "FAIL");
    std::printf("  phi conserved to round-off           worst %.2e     %s\n",
                worst_drift, pass_d ? "PASS" : "FAIL");
    const bool pass_fg = finite && form_gap < tol_fg;
    const bool pass_rs = finite && ratio_spread < tol_rs;
    std::printf("  stress and potential agree to %.0f%%      gap     %.2e     %s\n",
                100 * tol_fg, form_gap, pass_fg ? "PASS" : "FAIL");
    std::printf("  sigma independent of rho ratio to %.0f%%  spread  %.2e     %s\n",
                100 * tol_rs, ratio_spread, pass_rs ? "PASS" : "FAIL");
    std::printf("\n  -refine asserts the order the absolute value does not.\n");
    if (!(pass_w && pass_uf && pass_sp && pass_s && pass_u && pass_d &&
          pass_fg && pass_rs)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
