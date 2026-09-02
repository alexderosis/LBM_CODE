//==============================================================================
//  The phase field against a WALL, which is the one place nothing else looked.
//
//  Every other phase-field check in this tree is periodic in every direction --
//  phase_flat, enan_interface, the droplet cases -- so PhaseWall was exercised
//  by no test at all. This is the gap that let the bug below live.
//
//  THE CASE. A closed box holding ONE phase, uniformly, at rest: phi = 1
//  everywhere, PhaseWall on the two y planes, no velocity, no gravity, no
//  interface anywhere in the domain. There is nothing for the scheme to do. A
//  uniform field is an exact stationary solution of the conservative
//  Allen-Cahn equation -- grad phi is zero, so both the diffusive flux and the
//  anti-diffusive flux vanish identically -- and a zero-flux wall cannot change
//  that. phi must stay 1 at every node, forever, to round-off.
//
//  That part passes, at every mobility and both operators, to 2e-16. So does a
//  uniform flow TANGENTIAL to the wall. The wall itself is not broken, and the
//  first draft of this file said it was -- wrongly.
//
//  WHAT IS NOT RIGHT, AND IS NOT FIXED HERE. Give the same box a velocity
//  NORMAL to the wall and phi next to it oscillates by 8e-2 within a few steps.
//  That is printed as a diagnostic and deliberately NOT asserted, because a
//  uniform normal velocity at an impermeable wall is not a well-posed input in
//  the first place: the wall blocks a flux the interior keeps supplying. What
//  makes it worth recording is that the CUDA port does not do it, on the case
//  where the comparison IS well posed -- the 3-D Rayleigh-Taylor top wall,
//  where the port holds phi to 2e-5 while this code oscillates between 0.82 and
//  0.86 from step 2 onward. Somebody should establish which is right; this file
//  is the place to do it, and the reference answer is not yet known.
//
//  WHY IT HID FOR SO LONG, and it is worth stating because the asymmetry is the
//  whole diagnostic. The error is invisible at phi = 0: every population is
//  zero there, and any linear operator maps zero to zero, so a wall in the
//  light phase is preserved exactly no matter what the wall does. Only a wall
//  standing in the phi = 1 phase shows it. In the Rayleigh-Taylor case that is
//  the TOP wall alone, which is why the bottom half of that domain agreed to
//  1e-35 while the top half did not.
//
//  HOW IT WAS FOUND. Not here. The CUDA port in GPU/ sets a phase geometry
//  where validation/enan_rt.cpp did not, so the two codes disagreed on the 3-D
//  Rayleigh-Taylor spike from t/t0 = 2 onward. Diffing phi node by node put the
//  difference at the top wall and nowhere else, and at the phi = 1 wall only --
//  the phi = 0 wall agreed to 1e-35, because every population is zero there and
//  any linear operator preserves zero. That is the argument for keeping two
//  independent implementations.
//
//  This case is deliberately trivial to read: if it fails, the number it prints
//  IS the error in phi at a node where the answer is exactly 1.
//==============================================================================
#include "collision/PhaseFieldBGK.hpp"
#include "collision/PhaseFieldCentralMoments.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/PhaseFieldSolver.hpp"

#include <cmath>
#include <cstdio>

using namespace lbm;

namespace {

using PL = D3Q27;

// Returns the largest |phi - 1| over the BULK nodes, which is where the answer
// is unambiguous. The wall nodes themselves are ghosts -- what phi() reports at
// a PhaseWall node is a partial population sum, not a physical value, the same
// trap ScalarSolver's temperature() carries -- so they are excluded rather than
// asserted about.
template <class Coll>
double run(const char* name, Index n_steps, double omega, double width,
           double u0 = 0.0, double v0 = 0.0) {
  const Index nx = 8, ny = 32, nz = 8;
  Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  Coll pc;
  pc.omega = Real(omega);
  pc.width = Real(width);
  PhaseFieldSolver<PL, EsotericPull<PL>, Coll> pf(d, pc);

  pf.initialize_field(KOKKOS_LAMBDA(Index) { return Real(1); });
  pf.set_geometry([&](Index, Index y, Index) -> PhaseCell {
    return (y == 0 || y == ny - 1) ? PhaseWall : PhaseBulk;
  });

  // The advecting field is PRESCRIBED and uniform, so it cannot be the thing
  // that is wrong. u0 is tangential to the wall and v0 normal to it; a uniform
  // field of either kind leaves phi = 1 alone, because a constant advected
  // through a constant is a constant, and the wall is zero flux.
  View1D<Real> ux("ux", d.n_padded), uy("uy", d.n_padded), uz("uz", d.n_padded);
  Kokkos::deep_copy(ux, Real(u0));
  Kokkos::deep_copy(uy, Real(v0));
  pf.set_velocity(ux, uy, uz);

  for (Index t = 0; t < n_steps; ++t) { pf.refresh(); pf.step(); }

  pf.compute_field();
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
  double worst = 0.0;
  Index worst_y = 0;
  for (Index z = 0; z < nz; ++z)
    for (Index y = 1; y < ny - 1; ++y)            // bulk only
      for (Index x = 0; x < nx; ++x) {
        const double e = std::fabs(double(h(d.id(x, y, z))) - 1.0);
        if (e > worst) { worst = e; worst_y = y; }
      }
  std::printf("  %-28s  max|phi - 1| = %.3e   at y = %d (walls at 0 and %d)\n",
              name, worst, int(worst_y), int(ny - 1));
  return worst;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::printf("Phase field against a zero-flux wall: one phase, uniform, at "
                "rest\n");
    std::printf("phi = 1 everywhere, PhaseWall on both y planes, no velocity. "
                "phi must not move.\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(),
                precision_name());

    // Both operators, because the wall is applied by the solver and not by the
    // operator: if only one fails, the fault is in the operator instead.
    const double a = run<PhaseFieldBGK<PL>>("BGK,  omega 1.0", 100, 1.0, 4.0);
    const double b = run<PhaseFieldCentralMoments<PL>>("CM,   omega 1.0", 100, 1.0, 4.0);
    const double c = run<PhaseFieldCentralMoments<PL>>("CM,   omega 1.8", 100, 1.8, 4.0);
    const double e = run<PhaseFieldCentralMoments<PL>>("CM,   1000 steps", 1000, 1.0, 4.0);
    // AND THE SAME WITH THE FIELD MOVING. The Rayleigh-Taylor case differs from
    // everything above only in that its phase field is advected by a real
    // velocity, so if a uniform one breaks the wall, that is the mechanism.
    const double f = run<PhaseFieldCentralMoments<PL>>("CM,   u tangential 0.02",
                                                       100, 1.0, 4.0, 0.02, 0.0);

    // A uniform field is exact for this scheme, so the tolerance is round-off
    // and not a physical accuracy claim. FP32 carries ~1e-7 per operation and
    // 1000 steps of it, so the bar is set there rather than at the FP64 floor.
    const double tol = sizeof(Real) == 4 ? 1e-4 : 1e-10;
    const double worst = std::fmax(std::fmax(a, b),
                                   std::fmax(std::fmax(c, e), f));
    std::printf("\n  tolerance %.1e (a uniform field is EXACT for this scheme; "
                "this is round-off, not accuracy)\n", tol);
    std::printf("  %s   worst %.3e\n", worst < tol ? "PASS" : "FAIL", worst);
    if (!(worst < tol)) status = 1;

    // A DIAGNOSTIC, NOT AN ASSERTION, and the distinction is the point. A
    // UNIFORM velocity normal to an impermeable wall is not a well-posed state:
    // the wall blocks a flux the interior keeps supplying, so phi genuinely has
    // nowhere to go and the scheme is being asked something incoherent. It is
    // printed because it is the one input that reproduces what the 3-D
    // Rayleigh-Taylor case does at its top wall -- an oscillation appearing
    // within two steps, one cell in -- and because the CUDA port does NOT
    // produce it on the same input. That asymmetry is worth a number in front
    // of whoever looks next; it is not yet worth a pass/fail, because the right
    // reference answer here has not been established.
    std::printf("\n  diagnostic (NOT asserted -- the input is not well posed; "
                "see the note in the source):\n");
    const double g = run<PhaseFieldCentralMoments<PL>>("CM,   u NORMAL 0.002",
                                                       100, 1.0, 4.0, 0.0, 0.002);
    // NOT a like-for-like number: the port has not been run on THIS input.
    // 2e-5 is what it holds at the top wall of the 3-D Rayleigh-Taylor case,
    // where this code oscillates between 0.82 and 0.86 from step 2 onward.
    std::printf("    this code gives %.2e here; on the 3-D Rayleigh-Taylor top "
                "wall, where the\n    comparison is well posed, the CUDA port "
                "holds phi to 2e-5 and this code does not\n", g);
  }
  Kokkos::finalize();
  return status;
}
