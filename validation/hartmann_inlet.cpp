//==============================================================================
//  Hartmann flow, INLET DRIVEN.
//
//  The companion to validation/hartmann.cpp, which drives the same flow with a
//  uniform body force through a streamwise-periodic channel. Three things are
//  different here, and each is the point of the test:
//
//    * No applied force. The flow is sustained by the pressure difference the
//      solver establishes between a velocity inlet and a constant back-pressure
//      outlet, exactly as in validation/poiseuille_inlet.cpp.
//    * The central-moments MHD operator. MhdCentralMoments hardcodes NoForcing
//      and so cannot be body-force driven at all -- an inlet is the ONLY way to
//      put it against an analytic profile. The force-driven test covers MhdBGK
//      and could never have covered this.
//    * The new outflow conditions, fluid and magnetic, coupled. Nothing else in
//      the suite exercises both at once.
//
//  GEOMETRY. Dellar's axes are rotated so the streamwise direction is x, which
//  is where the outflow conditions live: walls on the two y-normal planes at
//  y = 0 and y = Ly-1, half-width L = (Ly-1)/2, xi = (y-L)/L in [-1,1]. The
//  applied field B0 is WALL-NORMAL (along y); the flow stretches it into a
//  streamwise induced component b_x.
//
//  EXACT SOLUTION, Dellar Eq. (14) with x and y interchanged, Pr_m = 1 so
//  eta = nu and sqrt(nu eta) = nu, and Ha = B0 L / nu:
//
//     u_x(xi) = A coth(Ha) [ 1 - cosh(Ha xi)/cosh(Ha) ]
//     b_x(xi) = A [ sinh(Ha xi)/sinh(Ha) - xi ]
//
//  where A is fixed by the requested peak speed, A = u_max / (coth(Ha)(1-sech Ha)).
//  Both vanish at xi = +-1. The velocity profile is the flat-cored Hartmann one
//  with boundary layers of thickness ~ L/Ha, so this is a much harder test of
//  the wall region than the parabolic Poiseuille profile: at Ha = 10 the layer
//  is a tenth of the channel.
//
//  BOUNDARY CONDITIONS.
//    u  inlet   x = 0        regularised, analytic Hartmann profile imposed
//    u  outlet  x = Lx-1     constant back-pressure, rho = 1
//    u  walls   y = 0, Ly-1  regularised, u = 0
//    B  inlet   x = 0        moment Dirichlet, (b_x(xi), B0, 0)
//    B  outlet  x = Lx-1     moment zero-gradient
//    B  walls   y = 0, Ly-1  moment Dirichlet, (0, B0, 0)
//
//  b is imposed only at the inlet, not at both ends. Pinning it at the outlet
//  too would make the interior test partly circular -- the induction equation
//  would be told the answer at both boundaries -- so the outlet is left free.
//
//  The error is the relative L2 norm over a y-aligned cross-section at
//  x = Lx/2, reported separately for u and for b.
//
//  RESULT. Second order on both fields, against Dellar's exact solution:
//
//      Ly     relL2(u)   order     relL2(b)   order      (Lx = 241, u_max = 0.005)
//      17    2.281e-02      --    6.499e-02      --
//      33    5.391e-03   2.081    1.590e-02   2.031
//      65    1.286e-03   2.068    3.916e-03   2.022
//     129    3.071e-04   2.066    9.793e-04   2.000
//
//  u and b vanish at the wall node to exactly zero at every resolution, which is
//  what putting both conditions ON the node buys.
//
//  WHY Lx IS SO LARGE, and a defect it exposes. The magnetic zero-gradient
//  outlet (MagOutXp) is NOT clean: it drives b roughly 6% high over the last
//  ~10 nodes, and that overshoot decays upstream only slowly. Measured at
//  Ly = 65 by sampling b along x -- the exact solution is x-independent, so any
//  x-variation is boundary error:
//
//      Lx = 21    b/b_exact runs 1.000 -> 1.054 across the WHOLE domain
//      Lx = 121   flat plateau at 1.005 in the bulk, rising to 1.061 at the end
//      Lx = 241   plateau reached well before the sampling station
//
//  The effect on the score is large: at Ly = 129 the b error falls from
//  3.367e-03 (Lx = 121) to 9.793e-04 (Lx = 241) purely by moving the outlet
//  further away. It is specifically the MAGNETIC outlet, not the fluid one:
//  pinning b analytically at the outlet instead gives errors agreeing to three
//  digits with the zero-gradient case at Ly = 17, 33, 65, so the two only differ
//  through that contamination. Until the condition is fixed, this test needs the
//  outlet far from the sampling station, and MagOutXp should not be treated as
//  validated.
//
//  The Mach floor of validation/poiseuille_inlet.cpp shows up here too, though
//  it is the smaller effect: at u_max = 0.02 (Ma = 0.035) the last u rate falls
//  to 1.461, and lowering to u_max = 0.005 (Ma = 0.0087) recovers it.
//==============================================================================
#include "Campaign.hpp"
#include "FieldDump.hpp"
#include "boundary/MomentDirichlet.hpp"
#include "boundary/Regularized.hpp"
#include "collision/MagneticBGK.hpp"
#include "collision/MhdBGK.hpp"
#include "collision/MhdCentralMoments.hpp"
#include "solver/MagneticSolver.hpp"

#include <chrono>

using namespace lbm;
using namespace campaign;

struct Result {
  double eu = NAN, eb = NAN;      // relative L2 at the sampling station
  double wall_u = NAN, wall_b = NAN;
  double resid = NAN, secs = 0;
  std::size_t steps = 0;
  bool ok = false;
};

// Analytic profile, shared by the inlet condition and the error measure so the
// two can never drift apart.
struct Exact {
  double Ha, A, L;
  double xi(Index y) const { return (double(y) - L) / L; }
  double u(double x) const {
    return A / std::tanh(Ha) * (1.0 - std::cosh(Ha * x) / std::cosh(Ha));
  }
  double b(double x) const {
    return A * (std::sinh(Ha * x) / std::sinh(Ha) - x);
  }
};

template <class FL, class ML, class FluidColl>
static Result run(Index Ly, Index Lx, double Ha, Real nu, Real umax, bool dump,
                  bool bout_dirichlet) {
  using MagColl = MagneticBGK<ML>;
  Result r;

  const Real L   = Real(Ly - 1) / Real(2);
  const Real eta = nu;                       // Pr_m = 1
  const Real B0  = Real(Ha) * nu / L;        // from Ha = B0 L / sqrt(nu eta)
  const double shape = (1.0 / std::tanh(Ha)) * (1.0 - 1.0 / std::cosh(Ha));
  const Exact ex{Ha, double(umax) / shape, double(L)};

  Domain d(Lx, Ly, 1, /*periodic x*/ false, /*y*/ false, /*z*/ true);

  MagColl mc;
  mc.omega = MagColl::omega_from_resistivity(eta);
  MagneticSolver<ML, EsotericPull<ML>, MagColl> mag(d, mc);

  FluidColl fc;
  fc.omega = FluidColl::omega_from_viscosity(nu);
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fc);

  fl.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  using WS = typename decltype(fl)::WallSpec;
  fl.set_regularized_walls([&](Index x, Index y, Index) -> WS {
    if (y == 0)      return WS{NrmYm, Real(0), Real(0), Real(0), Real(1)};
    if (y == Ly - 1) return WS{NrmYp, Real(0), Real(0), Real(0), Real(1)};
    if (x == 0)      return WS{NrmXm, Real(ex.u(ex.xi(y))), Real(0), Real(0), Real(1)};
    if (x == Lx - 1) return WS{NrmOutXp, Real(0), Real(0), Real(0), Real(1)};
    return WS{};
  });

  using WB = typename decltype(mag)::WallB;
  mag.set_moment_walls([&](Index x, Index y, Index) -> WB {
    // Walls first: at the inlet corners the wall value and the profile agree
    // (b and u both vanish at xi = +-1), so the order does not matter there,
    // but it must be consistent with the fluid's choice.
    if (y == 0 || y == Ly - 1) return WB{true, Real(0), B0, Real(0), false};
    if (x == 0)      return WB{true, Real(ex.b(ex.xi(y))), B0, Real(0), false};
    if (x == Lx - 1) {
      // -bout dirichlet pins the analytic profile here instead, which is the
      // discriminator for whether the zero-gradient outlet is at fault.
      if (bout_dirichlet) return WB{true, Real(ex.b(ex.xi(y))), B0, Real(0), false};
      return WB{true, Real(0), Real(0), Real(0), /*outflow*/ true};
    }
    return WB{};
  });

  const Real B0c = B0;
  const Exact exc = ex;
  const Domain dc = d;
  mag.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dc.coords(n, px, py, pz);
    const double x = (double(py - dc.hy) - exc.L) / exc.L;
    Kokkos::Array<Real, 3> b;
    b[0] = Real(exc.b(x)); b[1] = B0c; b[2] = Real(0);
    return b;
  });
  fl.initialize(Real(1));
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // Steady state on a whole-field residual, as in poiseuille_inlet.cpp -- a
  // single-node test saturates at round-off while the field is still moving.
  const double tau_d = double(Ly) * double(Ly) / double(nu);
  const std::size_t probe = std::max<std::size_t>(200, std::size_t(tau_d / 200.0));
  const std::size_t tmin  = std::size_t(0.5  * tau_d);
  const std::size_t cap   = std::size_t(40.0 * tau_d);

  std::vector<double> prev(std::size_t(Lx) * std::size_t(Ly), 0.0);
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t t = 0; t < cap; t += probe) {
    for (std::size_t k = 0; k < probe; ++k) {
      mag.compute_field();      // fluid must collide against B(t), not B(t-1)
      fl.step(true);
      mag.step(true);
    }
    r.steps += probe;
    fl.compute_macroscopic();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    double num = 0, den = 0; bool bad = false;
    for (Index y = 0; y < Ly && !bad; ++y)
      for (Index x = 0; x < Lx; ++x) {
        const std::size_t k = std::size_t(y) * std::size_t(Lx) + std::size_t(x);
        const double v = double(h(d.id(x, y, 0)));
        if (!std::isfinite(v)) { bad = true; break; }
        const double dv = v - prev[k];
        num += dv * dv; den += v * v; prev[k] = v;
      }
    if (bad) return r;
    r.resid = std::sqrt(num / std::max(den, 1e-300));
    if (r.steps >= tmin && r.resid < 1e-10) break;
  }
  r.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  fl.compute_macroscopic(); mag.compute_field();
  auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hbx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());

  double nu_ = 0, du = 0, nb = 0, db = 0;
  for (Index y = 0; y < Ly; ++y) {
    const double xi = ex.xi(y);
    const double au = ex.u(xi), ab = ex.b(xi);
    const double vu = double(hux(d.id(Lx / 2, y, 0)));
    const double vb = double(hbx(d.id(Lx / 2, y, 0)));
    du += (vu - au) * (vu - au); nu_ += au * au;
    db += (vb - ab) * (vb - ab); nb  += ab * ab;
  }
  r.eu = std::sqrt(du / nu_);
  r.eb = std::sqrt(db / nb);
  r.wall_u = std::abs(double(hux(d.id(Lx / 2, 0, 0))));
  r.wall_b = std::abs(double(hbx(d.id(Lx / 2, 0, 0))));
  r.ok = true;

  if (std::getenv("FIGDUMP")) {
    using namespace lbm::figdump;
    const std::string t = "ha" + std::to_string(int(Ha)) + "_ly" + std::to_string(int(Ly));
    scalar_slice("hm_ux_" + t + ".bin", Lx, Ly,
                 [&](Index x, Index y) { return double(hux(d.id(x, y, 0))); });
    scalar_slice("hm_bx_" + t + ".bin", Lx, Ly,
                 [&](Index x, Index y) { return double(hbx(d.id(x, y, 0))); });
  }

  if (dump) {
    std::printf("\n   %8s %13s %13s %13s %13s\n",
                "y/L", "u (LB)", "u (exact)", "b (LB)", "b (exact)");
    for (Index y = 0; y < Ly; y += (Ly > 18 ? Ly / 16 : 1)) {
      const double xi = ex.xi(y);
      std::printf("   %8.4f %13.6e %13.6e %13.6e %13.6e\n", xi,
                  double(hux(d.id(Lx / 2, y, 0))), ex.u(xi),
                  double(hbx(d.id(Lx / 2, y, 0))), ex.b(xi));
    }
    const double xi = ex.xi(Ly - 1);
    std::printf("   %8.4f %13.6e %13.6e %13.6e %13.6e\n\n", xi,
                double(hux(d.id(Lx / 2, Ly - 1, 0))), ex.u(xi),
                double(hbx(d.id(Lx / 2, Ly - 1, 0))), ex.b(xi));
    // The exact solution is x-independent. If it is not here, the streamwise
    // boundaries are driving the error rather than the wall resolution.
    const Index yq = Ly / 4;
    std::printf("   at y/L = %.4f, along x:  b(x)/b_exact =", ex.xi(yq));
    for (Index xv = 0; xv < Lx; xv += 4)
      std::printf(" %.4f", double(hbx(d.id(xv, yq, 0))) / ex.b(ex.xi(yq)));
    std::printf("\n   %36s u(x)/u_exact =", "");
    for (Index xv = 0; xv < Lx; xv += 4)
      std::printf(" %.4f", double(hux(d.id(xv, yq, 0))) / ex.u(ex.xi(yq)));
    std::printf("\n\n");
  }
  return r;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    double Ha = 10.0;
    Real nu = Real(0.1), umax = Real(0.005);
    // Lx = 241, not the 21 of the Poiseuille test: the magnetic outlet
    // contaminates a long stretch upstream. See the header.
    Index Lx = 241;
    std::string op = "cm", lat = "d2q9";
    bool dump = false, boutd = false;
    std::vector<Index> Lys = {17, 33, 65, 129};
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-ha"  && i + 1 < argc) Ha = std::atof(argv[++i]);
      if (a == "-nu"  && i + 1 < argc) nu = Real(std::atof(argv[++i]));
      if (a == "-u0"  && i + 1 < argc) umax = Real(std::atof(argv[++i]));
      if (a == "-lx"  && i + 1 < argc) Lx = std::atoi(argv[++i]);
      if (a == "-op"  && i + 1 < argc) op = argv[++i];
      if (a == "-lat" && i + 1 < argc) lat = argv[++i];
      if (a == "-dump") dump = true;
      if (a == "-bout" && i + 1 < argc) boutd = (std::string(argv[++i]) == "dirichlet");
      if (a == "-ly"  && i + 1 < argc) Lys = {Index(std::atoi(argv[++i]))};
    }

    std::printf("Hartmann flow, inlet driven   lattice %s   operator %s\n", lat.c_str(), op.c_str());
    std::printf("  Ha = %.1f   nu = eta = %.4f   u_max = %.3f   Lx = %d   Lz = 1\n",
                Ha, double(nu), double(umax), int(Lx));
    std::printf("  u: regularised inlet + back-pressure outlet   B: moment walls + zero-gradient outlet\n\n");
    std::printf("  %5s %12s %8s %12s %8s %10s %10s %10s\n",
                "Ly", "relL2(u)", "order", "relL2(b)", "order", "residual", "|u| wall", "|b| wall");
    std::printf("  %s\n", std::string(86, '-').c_str());

    std::FILE* f = open_out("F_hartmann_inlet", "hartmann_ha" + std::to_string(int(Ha)), lat, op);
    if (f) std::fprintf(f, "# Ly relL2u relL2b residual steps   Ha=%.1f nu=%.4f umax=%.3f Lx=%d\n",
                        Ha, double(nu), double(umax), int(Lx));

    double pu = 0, pb = 0; Index pn = 0;
    for (Index Ly : Lys) {
      Result r;
      const bool known =
        (lat == "d2q9" && op == "cm")  ? (r = run<D2Q9, D2Q5, MhdCentralMoments<D2Q9, true>>(Ly, Lx, Ha, nu, umax, dump, boutd), true) :
        (lat == "d2q9" && op == "bgk") ? (r = run<D2Q9, D2Q5, MhdBGK<D2Q9, HighOrderEquilibrium<D2Q9>, ShiftedPopulations, NoForcing>>(Ly, Lx, Ha, nu, umax, dump, boutd), true) :
        (lat == "d3q27" && op == "cm") ? (r = run<D3Q27, D3Q7, MhdCentralMoments<D3Q27, true>>(Ly, Lx, Ha, nu, umax, dump, boutd), true) :
        (lat == "d3q27" && op == "bgk")? (r = run<D3Q27, D3Q7, MhdBGK<D3Q27, HighOrderEquilibrium<D3Q27>, ShiftedPopulations, NoForcing>>(Ly, Lx, Ha, nu, umax, dump, boutd), true) : false;
      if (!known) { std::printf("  unknown lattice/operator combination\n"); break; }
      if (!r.ok)  { std::printf("  %5d   DIVERGED after %zu steps\n", int(Ly), r.steps); break; }

      const double ou = pn ? std::log(pu / r.eu) / std::log(double(Ly - 1) / double(pn - 1)) : NAN;
      const double ob = pn ? std::log(pb / r.eb) / std::log(double(Ly - 1) / double(pn - 1)) : NAN;
      std::printf("  %5d %12.5e", int(Ly), r.eu);
      if (pn) std::printf(" %8.3f", ou); else std::printf(" %8s", "--");
      std::printf(" %12.5e", r.eb);
      if (pn) std::printf(" %8.3f", ob); else std::printf(" %8s", "--");
      std::printf(" %10.2e %10.2e %10.2e   [%.0f s]\n", r.resid, r.wall_u, r.wall_b, r.secs);
      std::fflush(stdout);
      if (f) { std::fprintf(f, "%d %.8e %.8e %.4e %zu\n", int(Ly), r.eu, r.eb, r.resid, r.steps);
               std::fflush(f); }
      pu = r.eu; pb = r.eb; pn = Ly;
    }
    if (f) std::fclose(f);
    std::printf("\n  the wall columns should be at round-off: both conditions put the\n");
    std::printf("  boundary ON the node, so u and b vanish there exactly.\n");
  }
  Kokkos::finalize();
  return 0;
}
