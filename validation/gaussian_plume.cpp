//==============================================================================
//  Open boundaries for the passive scalar, measured against the Gaussian plume.
//
//  This exists to validate ScalarOutflow -- the only genuinely new piece of
//  physics needed before the scalar can be run on an urban domain, where the
//  plume must LEAVE through the lateral and top faces rather than accumulate.
//  Every other ingredient (D3Q7 transport, adiabatic and Dirichlet walls,
//  advection by an imposed velocity field) is already validated elsewhere.
//
//  THE ANALYTIC SOLUTION. A continuous point source of strength Q at the origin
//  in a uniform wind U along x, with isotropic diffusivity D, neglecting
//  streamwise diffusion, satisfies U dC/dx = D (d2C/dy2 + d2C/dz2). That is the
//  two-dimensional heat equation with x/U in place of time, so
//
//      C(x,y,z) = Q / (4 pi D x) * exp( -U (y^2 + z^2) / (4 D x) )
//
//  giving a plume that spreads as sigma(x) = sqrt(2 D x / U) while its
//  centreline value decays as 1/x. Integrating across the plume returns
//  U * int C dy dz = Q at EVERY station, which is the property the outflow
//  condition has to preserve.
//
//  WHAT EACH CHECK ACTUALLY TESTS, weakest first:
//
//    1. Uniform-state invariance. A constant field under a constant wind is an
//       exact fixed point of equilibrium, collision, streaming AND the outflow
//       condition, so the answer here is round-off or the code is wrong. This
//       isolates the new kernel from all of the physics: it needs no analytic
//       solution, no convergence, and no judgement about what "close" means.
//
//    2. Plume width sigma(x) against sqrt(2 D x / U). This tests the lattice's
//       diffusivity, D = cs^2 (1/omega - 1/2) with cs^2 = 1/4 on D3Q7. A wrong
//       speed of sound shows up immediately as a slope error, and it would be
//       invisible in test 1.
//
//    3. Cross-wind flux balance, U * sum C over each y-z plane, against Q.
//       THIS IS THE ONE THAT TESTS THE BOUNDARY CONDITION. A reflecting exit
//       piles concentration up near the outflow face and the flux climbs; an
//       over-draining one loses it and the flux falls. Reading it station by
//       station up to the last interior cell shows how far any contamination
//       reaches back into the domain, which a single domain-integrated number
//       would hide. It is the same measurement as the aorta's inlet/outlet
//       balance, for the same reason.
//
//  ERRORS TO EXPECT, so that agreement is judged against the right floor:
//
//    * The analytic solution drops d2C/dx2 and the lattice does not, an
//      O(1/Pe) difference -- about 0.2% at the Peclet number used here.
//    * The source is a small box, not a point, so the near field differs; the
//      comparison therefore starts several sigma downstream.
//    * D3Q7's equilibrium is first order in u (see ScalarBGK.hpp), an O(u^2)
//      defect in the advection term.
//    * sigma must span several cells for the lattice to represent a Gaussian at
//      all, which sets how close to the source the comparison can begin.
//==============================================================================
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;

using L      = D3Q7;
using Coll   = ScalarBGK<L>;
using Solver = ScalarSolver<L, EsotericPull<L>, Coll>;

//------------------------------------------------------------------------------
// A uniform wind along +x, as the velocity field the scalar is handed. The
// fluid solver would normally own these; here they are prescribed so that the
// test measures the scalar and its boundaries and nothing else.
//------------------------------------------------------------------------------
struct Wind {
  View1D<Real> ux, uy, uz;
  Wind(const Domain& d, Real U)
      : ux("ux", d.n_padded), uy("uy", d.n_padded), uz("uz", d.n_padded) {
    Kokkos::deep_copy(ux, U);
    Kokkos::deep_copy(uy, Real(0));
    Kokkos::deep_copy(uz, Real(0));
  }
  void attach(Solver& s) { s.set_velocity(ux, uy, uz); }
};

//==============================================================================
// 1. Uniform-state invariance -- an exact test, no analytic solution needed.
//
// THE PROPERTY TESTED IS UNIFORMITY, NOT THE LEVEL, and the distinction is the
// whole point. With every face zero-gradient the problem is pure Neumann:
// nothing anchors the absolute value of C, so ANY constant is a valid steady
// state and asking for 1.0 exactly would be testing the initial condition
// rather than the boundary condition. What the outflow condition must not do is
// introduce spatial structure, or fail to settle.
//
// The level is still reported, because it is informative: at u = 0 it is
// preserved to round-off, while at u != 0 it settles slightly below 1. That
// offset is D3Q7's FIRST-ORDER EQUILIBRIUM, the O(u^2) defect in the advection
// term documented in ScalarBGK.hpp -- measured here at -5.0e-4, -1.9e-3,
// -6.8e-3 and -2.2e-2 for u = 0.0125, 0.025, 0.05 and 0.1, i.e. scaling as
// u^1.9. All-Neumann boundaries pin nothing, so that defect has nowhere to go
// except into the level.
//
// It is NOT the initialiser laying down equilibrium at zero velocity, which was
// the obvious first suspect: attaching the wind only after the field is laid
// down and settled reproduces the offset to six significant figures. Worth
// stating because the fix that suspicion implies would have been wasted work.
//
// The offset is also invisible to any problem with a Dirichlet cell anywhere in
// it -- including the plume below, which is why it does not contaminate the
// measurement that follows.
//==============================================================================
struct Uniform { double structure, level, motion; };

static Uniform uniform_drift(Real U, std::size_t steps) {
  const Index N = 24;
  Domain d(N, N, N, false, false, false);
  Coll coll;
  coll.omega = Coll::omega_from_diffusivity(Real(0.01));

  Solver s(d, coll);
  // EVERY face open, so the uniform state is consistent with the boundaries.
  s.set_geometry([&](Index x, Index y, Index z) -> ScalarCell {
    const bool face = (x == 0 || x == N - 1 || y == 0 || y == N - 1 ||
                       z == 0 || z == N - 1);
    return face ? ScalarOutflow : ScalarBulk;
  });
  s.finalize_geometry();
  s.initialize(Real(1));
  Wind w(d, U); w.attach(s);

  auto snapshot = [&]() {
    s.compute_field();
    auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
    std::vector<double> v;
    v.reserve(std::size_t(N) * N * N);
    for (Index z = 0; z < N; ++z)
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) v.push_back(double(hv(d.id(x, y, z))));
    return v;
  };

  for (std::size_t t = 0; t < steps; ++t) s.step();
  const std::vector<double> a = snapshot();
  for (std::size_t t = 0; t < 2000; ++t) s.step();
  const std::vector<double> b = snapshot();

  double mean = 0.0;
  for (double v : b) mean += v;
  mean /= double(b.size());

  double structure = 0.0, motion = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    structure = std::max(structure, std::abs(b[i] - mean));
    motion    = std::max(motion,    std::abs(b[i] - a[i]));
  }
  return {structure, std::abs(mean - 1.0), motion};
}

//==============================================================================
// 2 and 3. The plume.
//==============================================================================
struct Station { double x, sigma, sigma_exact, cmax, cmax_exact, flux, flux_tot; };

struct Plume {
  std::vector<Station> st;
  double Q, steps_taken;
  bool finite;
};

static Plume run_plume(Index nx, Index ny, Index nz, Real U, Real D,
                       Index xs, Index src_r, Real Q, std::size_t max_steps) {
  Domain d(nx, ny, nz, false, false, false);
  Coll coll;
  coll.omega = Coll::omega_from_diffusivity(D);

  Solver s(d, coll);
  // x = 0 is a clean zero-concentration inlet; every other face is open. The
  // lateral faces are open rather than adiabatic on purpose: it puts the new
  // condition on five faces at once instead of hiding it behind far-field
  // walls the plume never reaches.
  s.set_geometry([&](Index x, Index y, Index z) -> ScalarCell {
    if (x == 0) return ScalarDirichlet;
    const bool face = (x == nx - 1 || y == 0 || y == ny - 1 ||
                       z == 0 || z == nz - 1);
    return face ? ScalarOutflow : ScalarBulk;
  });
  s.set_wall_values([](Index, Index, Index) -> Real { return Real(0); });
  s.finalize_geometry();
  s.initialize(Real(0));
  Wind w(d, U); w.attach(s);

  // Continuous release from a small box, normalised so the total injected per
  // step is exactly Q whatever the box size.
  const Index yc = ny / 2, zc = nz / 2;
  Index nsrc = 0;
  for (Index z = zc - src_r; z <= zc + src_r; ++z)
    for (Index y = yc - src_r; y <= yc + src_r; ++y)
      for (Index x = xs - src_r; x <= xs + src_r; ++x)
        if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1) ++nsrc;
  const Real per_cell = Q / Real(nsrc);
  const Domain dc = d;
  auto src = KOKKOS_LAMBDA(Index n) -> Real {
    Index px, py, pz; dc.coords(n, px, py, pz);
    const Index x = px - dc.hx, y = py - dc.hy, z = pz - dc.hz;
    const bool in = (x >= xs - src_r && x <= xs + src_r &&
                     y >= yc - src_r && y <= yc + src_r &&
                     z >= zc - src_r && z <= zc + src_r);
    return in ? per_cell : Real(0);
  };

  // Steady state is reached when the flux at a downstream station stops moving.
  // Checking the concentration at a point instead would declare convergence
  // while the plume was still filling: the point stops changing long before
  // what leaves the domain equals what enters it.
  // The plume needs nx/U steps merely to CROSS the domain, so a convergence
  // test alone declares success at once: before it arrives the downstream flux
  // is zero, and zero is very stable. Requiring several transit times first is
  // the same trap, and the same fix, as the diffusive warm-up in scalar_walls.
  const Index probe_x = nx - 8;
  const std::size_t warmup = std::size_t(4.0 * double(nx) / double(U));
  double prev = 0.0;
  std::size_t taken = 0;
  const std::size_t chunk = 500;
  bool converged = false;
  for (std::size_t t = 0; t < max_steps && !converged; t += chunk) {
    for (std::size_t k = 0; k < chunk; ++k) { s.add_source(src); s.step(); }
    taken += chunk;
    s.compute_field();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
    double F = 0.0;
    for (Index z = 1; z < nz - 1; ++z)
      for (Index y = 1; y < ny - 1; ++y) F += double(h(d.id(probe_x, y, z)));
    F *= double(U);
    if (!std::isfinite(F)) return {{}, double(Q), double(taken), false};
    if (taken > warmup && std::abs(F - prev) < 1e-7 * std::max(1.0, std::abs(F)))
      converged = true;
    prev = F;
  }

  s.compute_field();
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());

  Plume out; out.Q = double(Q); out.steps_taken = double(taken); out.finite = true;
  for (Index x = 20; x <= nx - 2; x += 6) {
    double m0 = 0.0, m2 = 0.0, cmax = 0.0, fdiff = 0.0;
    for (Index z = 1; z < nz - 1; ++z)
      for (Index y = 1; y < ny - 1; ++y) {
        const double c = double(h(d.id(x, y, z)));
        const double r2 = double(y - yc) * double(y - yc) +
                          double(z - zc) * double(z - zc);
        m0 += c; m2 += c * r2;
        cmax = std::max(cmax, c);
        if (x > 1 && x < nx - 2)
          fdiff += -double(D) * 0.5 * (double(h(d.id(x + 1, y, z))) -
                                       double(h(d.id(x - 1, y, z))));
      }
    // <r^2> = 2 sigma^2 for an isotropic two-dimensional Gaussian.
    const double sigma = (m0 > 0) ? std::sqrt(0.5 * m2 / m0) : 0.0;
    const double xe = double(x - xs);
    const double se = std::sqrt(2.0 * double(D) * xe / double(U));
    const double ce = double(Q) / (4.0 * M_PI * double(D) * xe);
    out.st.push_back({double(x), sigma, se, cmax, ce,
                      double(U) * m0, double(U) * m0 + fdiff});
  }
  return out;
}

//==============================================================================
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("\n3D passive-scalar OPEN BOUNDARY -- Gaussian plume   D3Q7   %s\n",
                precision_name());
    std::printf("%s\n\n", std::string(78, '=').c_str());

    //--------------------------------------------------------------------------
    std::printf("1. UNIFORM-STATE INVARIANCE  (exact: a constant field must stay\n");
    std::printf("   constant and stationary under the outflow condition)\n\n");
    std::printf("   %-22s %15s %14s %14s\n",
                "case", "max|C-mean|", "max|dC/2000|", "|mean-1|");
    std::printf("   %s\n", std::string(68, '-').c_str());
    // 24^2/D ~ 58,000 steps to settle diffusively. Testing after a few thousand
    // measures the tail of the startup transient, not the fixed point.
    const Uniform u0 = uniform_drift(Real(0),    60000);
    const Uniform u1 = uniform_drift(Real(0.05), 60000);
    std::printf("   %-22s %15.3e %14.3e %14.3e\n",
                "u = 0", u0.structure, u0.motion, u0.level);
    std::printf("   %-22s %15.3e %14.3e %14.3e\n",
                "u = 0.05 x", u1.structure, u1.motion, u1.level);
    const bool exact_ok = (u0.structure < 1e-12) && (u1.structure < 1e-12) &&
                          (u0.motion    < 1e-12) && (u1.motion    < 1e-12);
    std::printf("\n   %s\n", exact_ok
        ? "PASS -- no spatial structure and no motion: round-off in both columns."
        : "FAIL -- the outflow condition perturbs or fails to settle a uniform state.");
    std::printf("   The level column is a diagnostic, not a criterion -- see the note\n");
    std::printf("   in the source. All-Neumann boundaries do not determine it.\n\n");

    //--------------------------------------------------------------------------
    const Index nx = 128, ny = 64, nz = 64, xs = 8;
    const Real U = Real(0.05), D = Real(0.01), Q = Real(1);
    const double Pe = double(U) * double(nx) / double(D);
    std::printf("2/3. GAUSSIAN PLUME\n\n");
    std::printf("   grid %dx%dx%d   U = %.3f   D = %.4f   omega = %.4f   Pe = %.0f\n",
                int(nx), int(ny), int(nz), double(U), double(D),
                double(Coll::omega_from_diffusivity(D)), Pe);
    std::printf("   source at x = %d, Q = %.1f per step\n\n", int(xs), double(Q));

    const Plume p = run_plume(nx, ny, nz, U, D, xs, 1, Q, 60000);
    if (!p.finite) {
      std::printf("   DIVERGED after %.0f steps\n", p.steps_taken);
      Kokkos::finalize();
      return 1;
    }
    std::printf("   steady after %.0f steps\n\n", p.steps_taken);

    std::printf("   %5s %9s %9s %8s %11s %11s %8s %9s\n",
                "x", "sigma", "exact", "err%", "C_max", "exact", "err%", "flux/Q");
    std::printf("   %s\n", std::string(78, '-').c_str());
    double worst_flux = 0.0, worst_sigma = 0.0;
    for (const Station& t : p.st) {
      const double es = 100.0 * (t.sigma - t.sigma_exact) / t.sigma_exact;
      const double ec = 100.0 * (t.cmax - t.cmax_exact) / t.cmax_exact;
      const double fr = t.flux / p.Q;
      std::printf("   %5.0f %9.3f %9.3f %+8.2f %11.4e %11.4e %+8.2f %9.5f\n",
                  t.x, t.sigma, t.sigma_exact, es, t.cmax, t.cmax_exact, ec, fr);
      if (t.x >= 30) {
        worst_flux  = std::max(worst_flux,  std::abs(fr - 1.0));
        worst_sigma = std::max(worst_sigma, std::abs(es));
      }
    }
    std::printf("\n   over x >= 30:  worst flux error %.3f%%,  worst sigma error %.2f%%\n",
                100.0 * worst_flux, worst_sigma);
    std::printf("\n   The flux column is the boundary-condition test: it should stay at\n");
    std::printf("   1.0 all the way to the last interior cell. A reflecting exit makes\n");
    std::printf("   it climb near the outflow face; an over-draining one makes it fall.\n");
    std::printf("   The sigma column tests D = cs^2 (1/omega - 1/2) with cs^2 = 1/4,\n");
    std::printf("   which the uniform-state test above cannot see at all.\n\n");
  }
  Kokkos::finalize();
  return 0;
}
