//==============================================================================
//  The flat interface: the smallest thing the conservative Allen-Cahn phase
//  field must get right, and the cheapest place to look when it does not.
//
//  A slab of liquid with two flat interfaces, in a domain periodic in every
//  direction, with the velocity PRESCRIBED and uniform. Two configurations:
//
//    -u0 0      AT REST. The tanh profile of width W is an exact stationary
//               solution of the continuum equation, because the anti-diffusion
//               cancels the diffusive flux identically there:
//
//                   theta = (4/W) phi(1-phi) = |grad phi|   at  phi =
//                       1/2[1 + tanh(2y/W)],
//
//               so grad phi - theta n = 0 and nothing moves. Anything that
//               happens here is discretisation, and it is unambiguous: the
//               width cannot drift, the interface cannot move, and the mass
//               cannot change.
//
//    -u0 > 0    ADVECTED. The same slab carried at constant speed through a
//               periodic box, which is validation/enan_interface's diagonal
//               translation reduced to one dimension and a few thousand cells.
//               Their Table II runs 50 000 steps of that at 100x100 and this
//               runs the same number in a second, so it is where to look first
//               when the translation error is large -- as it currently is.
//
//  WHAT IS MEASURED, and why the width is an integral rather than a fit:
//
//      W_meas = 4 * integral phi(1-phi) dy   per interface,
//
//  because for the tanh profile the integrand is (1/4) sech^2(2y/W) and its
//  integral is exactly W/4. Every node in the profile contributes, so the
//  measure does not degrade as W approaches the grid the way a two-point fit
//  of the steepest slope does.
//
//  WHAT IT FOUND, and it is worth stating because it is the reason this case
//  exists. At rest the scheme is exact at every mobility measured, down to
//  omega = 1.994: the width holds to 1e-4 over 50 000 steps, the interface
//  does not move, and the mass is conserved to 1e-12. Advect the same slab at
//  u0 = 0.02 and it is still exact to omega = 1.94, and then it is not:
//
//      omega     W drift (50 000 steps, W = 3)
//      1.5385    -6.6e-5
//      1.8868    -6.6e-5
//      1.9418    -6.6e-5
//      1.9881    -4.6e-2
//      1.9940    +7.4e-1     <- the interface is 25 % too wide
//
//  So the failure needs BOTH a low mobility and a moving interface, which is
//  what makes it invisible in a static test and what makes their Table II
//  operating point (omega = 1.9881) a demanding one.
//
//  The first hypothesis was the source truncation at u = 0, since the terms it
//  discards are O(|u| A) and so vanish exactly at rest. It is wrong: -fullsrc
//  transforms the source at the actual velocity and moves those numbers by
//  1.3 % and 0.3 %. What is left is the mobility itself -- M multiplies the
//  WHOLE restoring term in div[M(grad phi - theta n)], so as M falls the
//  sharpening can no longer correct the dispersion that advection introduces,
//  and there is no dispersion to correct at rest.
//==============================================================================
#include "collision/PhaseFieldCentralMoments.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/PhaseFieldSolver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;
namespace {

using L     = D3Q27;
using PColl = PhaseFieldCentralMoments<L>;

const char* arg_str(int c, char** v, const char* k, const char* d) {
  for (int i = 1; i + 1 < c; ++i) if (!std::strcmp(v[i], k)) return v[i + 1];
  return d;
}
double arg_num(int c, char** v, const char* k, double d) {
  const char* s = arg_str(c, v, k, nullptr);  return s ? std::atof(s) : d;
}

struct Out {
  double w0 = 0, w1 = 0;        // measured width, start and end
  double pos_drift = 0;         // movement of the phi = 1/2 crossing, cells
  double linf = 0;              // max |phi - phi_exact|, against the SHIFTED exact
  double mass_drift = 0;
  double phi_min = 0, phi_max = 0;
  bool   finite = true;
};

// 4 * integral phi(1-phi) over the column, halved for the two interfaces.
double width_of(const std::vector<double>& p) {
  double s = 0;
  for (double v : p) s += v * (1.0 - v);
  return 4.0 * s / 2.0;
}
// The phi = 1/2 crossing on the rising edge, linearly interpolated.
double edge_of(const std::vector<double>& p) {
  for (std::size_t j = 1; j < p.size(); ++j)
    if (p[j - 1] < 0.5 && p[j] >= 0.5)
      return double(j - 1) + (0.5 - p[j - 1]) / (p[j] - p[j - 1]);
  return -1.0;
}

Out run(Index ny, double R, double W, double M, double u0, std::size_t steps,
        bool full_src) {
  Out o;
  const Index nx = 8, nz = 1;
  Domain d(nx, ny, nz, true, true, true);

  PColl coll;
  coll.omega = PColl::omega_from_mobility(Real(M));
  coll.width = Real(W);
  coll.full_source = full_src;
  PhaseFieldSolver<L, EsotericPull<L>, PColl> pf(d, coll);

  const Real y0 = Real(0.5 * double(ny)), Rr = Real(R), Wr = Real(W);
  const Index hy = d.hy;
  pf.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real y = Real(py - hy);
    const Real dy = (y > y0) ? (y - y0) : (y0 - y);
    return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (Rr - dy) / Wr));
  });
  pf.compute_field();

  auto column = [&]() {
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
    std::vector<double> p(static_cast<std::size_t>(ny), 0.0);
    for (Index j = 0; j < ny; ++j) p[std::size_t(j)] = double(h(d.id(0, j, 0)));
    return p;
  };

  std::vector<double> p0 = column();
  o.w0 = width_of(p0);
  const double e0 = edge_of(p0);
  double m0 = 0; for (double v : p0) m0 += v;

  // The velocity is prescribed and uniform, so the solver needs a field.
  View1D<Real> ux("ux", d.n_padded), uy("uy", d.n_padded), uz("uz", d.n_padded);
  Kokkos::deep_copy(uy, Real(u0));
  pf.set_velocity(ux, uy, uz);

  for (std::size_t t = 0; t < steps; ++t) { pf.refresh(); pf.step(); }
  pf.compute_field();

  std::vector<double> p1 = column();
  o.w1 = width_of(p1);
  double m1 = 0; for (double v : p1) m1 += v;
  o.mass_drift = (m0 != 0) ? (m1 - m0) / m0 : 0;
  o.phi_min = o.phi_max = p1[0];
  for (double v : p1) {
    if (!std::isfinite(v)) o.finite = false;
    o.phi_min = std::min(o.phi_min, v);  o.phi_max = std::max(o.phi_max, v);
  }
  // Where the edge SHOULD be: carried u0 per step, wrapped.
  const double want = std::fmod(e0 + u0 * double(steps), double(ny));
  const double got  = edge_of(p1);
  double dd = got - want;
  while (dd >  0.5 * double(ny)) dd -= double(ny);
  while (dd < -0.5 * double(ny)) dd += double(ny);
  o.pos_drift = dd;
  // L_inf against the exact profile shifted to where it should be.
  for (Index j = 0; j < ny; ++j) {
    double y = double(j) - u0 * double(steps);
    while (y <  0)            y += double(ny);
    while (y >= double(ny))   y -= double(ny);
    const double dy = std::fabs(y - 0.5 * double(ny));
    const double ex = 0.5 * (1.0 + std::tanh(2.0 * (R - dy) / W));
    o.linf = std::max(o.linf, std::fabs(p1[std::size_t(j)] - ex));
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  const Index ny      = Index(arg_num(argc, argv, "-ny", 256));
  const double R      = arg_num(argc, argv, "-r", 64.0);
  const double W      = arg_num(argc, argv, "-w", 3.0);
  const double u0     = arg_num(argc, argv, "-u0", 0.0);
  const std::size_t T = std::size_t(arg_num(argc, argv, "-steps", 50000));
  bool full_src = false;
  for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "-fullsrc")) full_src = true;

  Kokkos::initialize(argc, argv);
  {
    std::printf("Flat interface, D3Q27 phase-field central moments\n");
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("  slab of half-width %g in ny = %d, W = %g, %zu steps, u0 = %g\n",
                R, int(ny), W, T, u0);
    std::printf("  the tanh profile is an EXACT stationary solution at u0 = 0,\n"
                "  so any width drift, motion or overshoot here is the scheme\n");
    std::printf("  source: %s\n\n", full_src ? "TRANSFORMED at the actual u"
                                              : "truncated at u = 0 (as theirs)");
    std::printf("  %10s %9s %10s %10s %10s %11s %9s\n",
                "M", "omega", "W meas", "W drift", "pos drift", "Linf", "mass");
    std::printf("  %s\n", std::string(78, '-').c_str());

    int status = 0;
    for (double M : {0.05, 0.01, 0.005, 0.001, 0.0005}) {
      const Out o = run(ny, R, W, M, u0, T, full_src);
      const double om = 1.0 / (M / (1.0 / 3.0) + 0.5);
      std::printf("  %10.5f %9.5f %10.5f %+10.2e %+10.3f %11.3e %+9.1e%s\n",
                  M, om, o.w1, o.w1 - o.w0, o.pos_drift, o.linf, o.mass_drift,
                  o.finite ? "" : "  DIVERGED");
      std::fflush(stdout);
      if (!o.finite) status = 1;
    }
    std::printf("\n  W drift is against the width measured at t = 0, not against\n"
                "  the prescribed W: the initial profile is already discretised.\n");
    Kokkos::finalize();
    return status;
  }
}
