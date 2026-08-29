//==============================================================================
//  A square falling into a free water surface.
//
//  A DEMONSTRATOR, and a more speculative one than the others here: it puts a
//  moving rigid body into a two-phase flow, and there is no exact answer to
//  check it against. Wagner's slamming theory covers a WEDGE, whose wetted
//  length grows smoothly; a flat-bottomed square has a theoretically singular
//  impact pressure at first contact and no closed form at all. So what this can
//  honestly show is the qualitative sequence -- approach, impact, cavity,
//  splash-up jets, closure -- and a force history whose shape is physical, not
//  a number to quote.
//
//  De Rosis & Enan run this problem (Sec. III.G) with a Peskin immersed boundary
//  and a PRESCRIBED constant entry velocity. This uses volume penalisation and
//  lets the square FALL, which is the harder direction: the reaction is fed back
//  into Newton's equation, so the deceleration on impact is a result rather than
//  an input. See PenalisedBody.hpp for why penalisation and not IBM, and for the
//  fictitious-mass correction that makes the free fall well posed.
//
//  THE PHASES. phi = 1 is water, phi = 0 is air, and the free surface starts
//  flat. Both share a kinematic viscosity here rather than the true 15:1 air to
//  water ratio: the air's only job in this problem is to be light and get out of
//  the way, and giving it its physical viscosity buys nothing while costing
//  stability at these relaxation rates.
//
//  READ THE SPLASH, NOT THE MENISCUS. There is no contact-line model -- see the
//  limitation list in PenalisedBody.hpp -- so the angle at which the free
//  surface meets the square's sides is not physics. The jets and the cavity are
//  inertia-dominated and do not depend on it; the meniscus does.
//
//  Output is raw fields for demonstrator/render_rt, plus a body mask so the
//  square can be drawn, and a force history on stdout.
//==============================================================================
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/MultiphasePotentialBGK.hpp"
#include "collision/PhaseFieldBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/PenalisedBody.hpp"
#include "solver/PhaseFieldSolver.hpp"
#include "solver/ViscousInterfaceForce.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

using FL = D2Q9;
using PL = D2Q9;
using BgkColl = MultiphasePotentialBGK<FL, SecondOrderPhi<FL>, RawPopulations>;
using CmColl  = MultiphaseCentralMoments<FL>;
using PColl   = PhaseFieldBGK<PL>;
using PhaseSlv = PhaseFieldSolver<PL, EsotericPull<PL>, PColl>;

template <class Get>
static void dump_field(const std::string& path, Index nx, Index ny, Get get) {
  std::vector<float> v(std::size_t(nx) * std::size_t(ny));
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x)
      v[std::size_t(y) * std::size_t(nx) + std::size_t(x)] = float(get(x, y));
  std::ofstream o(path, std::ios::binary);
  if (!o) { std::printf("  cannot write %s\n", path.c_str()); return; }
  const std::int32_t a = int(nx), b = int(ny);
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}

struct Params {
  Index W;                 // square side, in cells -- everything scales off it
  double ratio, Re, U, iw, M, sigma, body_rho, drop, tmax;
  int nframes; std::string dump; const char* op;
};

template <class FColl>
static void simulate(const Params& P) {
  using FluidSlv = FluidSolver<FL, EsotericPull<FL>, FColl>;

  const Index L  = P.W;
  const Index nx = 6 * L, ny = 7 * L;
  const double y_water = 4.0 * double(L);           // free surface height
  const double rho_l = 1.0, rho_h = P.ratio;
  const double U  = P.U;                            // reference impact speed
  const double g  = U * U / (2.0 * P.drop * double(L));   // so it arrives at U
  const double nu = double(L) * U / P.Re;
  const std::size_t nsteps = std::size_t(P.tmax * double(L) / U);

  std::printf("Water entry of a square   D2Q9 (pressure form, %s) + D2Q9 phase field\n",
              FColl::name);
  std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
  std::printf("%dx%d   L = %d   rho_H/rho_L = %.0f   Re = %.0f   nu = %.3e\n",
              int(nx), int(ny), int(L), rho_h / rho_l, P.Re, nu);
  const double tau = nu / (1.0 / 3.0);
  std::printf("impact U = %.4f   g = %.3e   drop = %.2f L   omega = %.6f\n",
              U, g, P.drop, 1.0 / (tau + 0.5));
  std::printf("body rho = %.1f x water   %zu steps\n\n", P.body_rho, nsteps);

  Domain d(nx, ny, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  PColl pc;
  pc.omega = PColl::omega_from_mobility(Real(P.M));
  pc.width = Real(P.iw);
  PhaseSlv pf(d, pc);

  const Real yw = Real(y_water), iwr = Real(P.iw);
  const Index hx = d.hx, hy = d.hy;
  pf.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real y = Real(py - hy);
    // phi = 1 below the free surface: water underneath, air above.
    return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (yw - y) / iwr));
  });

  ViscousInterfaceForce<FL> vf(d);
  PenalisedBody<FL> body(d);
  body.shape = Rect{Real(0.5 * double(nx)),
                    Real(y_water + P.drop * double(L) + 0.5 * double(L)),
                    Real(0.5 * double(L)), Real(0.5 * double(L)), Real(1.5)};
  body.vx = Real(0);
  body.vy = Real(0);

  FColl fc;
  fc.phi = pf.phi();
  fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
  fc.Lap = pf.laplacian();
  fc.Vx = vf.x();  fc.Vy = vf.y();  fc.Vz = vf.z();
  fc.Ex = body.x(); fc.Ey = body.y(); fc.Ez = body.z();
  fc.rho_L = Real(rho_l);      fc.rho_H = Real(rho_h);
  fc.mu_L  = Real(rho_l * nu); fc.mu_H  = Real(rho_h * nu);
  fc.kappa = FColl::kappa_from_sigma(Real(P.sigma), Real(P.iw));
  fc.beta  = FColl::beta_from_sigma(Real(P.sigma), Real(P.iw));
  fc.by    = Real(-g);
  FluidSlv fl(d, fc);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;      // tank floor and lid
  });

  // Hydrostatic pressure, integrated through the diffuse free surface, with the
  // zero AT the surface -- the gauge argument in MultiphasePotentialBGK.hpp.
  auto phiv = pf.phi();
  const Real rl = Real(rho_l), rh = Real(rho_h), gr = Real(g);
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real y = Real(py - hy);
    const Real dz = y - yw;                            // positive upward, in air
    const Real az = (dz < Real(0) ? -dz : dz) * Real(2) / iwr;
    const Real lnch = az + Kokkos::log(Real(1) + Kokkos::exp(Real(-2) * az))
                    - Real(0.6931471805599453);
    // integral of rho from the surface to y, with rho heavy BELOW
    const Real I = rh * dz - (rh - rl) * Real(0.5) * (dz + Real(0.5) * iwr * lnch);
    const Real p = -gr * I;
    const Real r = rl + phiv(n) * (rh - rl);
    return FlowState{p / (r / Real(3)), Real(0), Real(0), Real(0)};
  });

  pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());
  body.set_velocity(fl.ux(), fl.uy());

  const Real area = body.penalised_area();
  const Real m_body = Real(P.body_rho * rho_h) * area;
  std::printf("penalised area %.1f cells (nominal %d), body mass %.3e\n\n",
              double(area), int(L * L), double(m_body));

  const std::size_t every = nsteps / std::size_t(P.nframes > 0 ? P.nframes : 1);
  int frame = 0;
  std::printf("%-8s %-9s %-10s %-11s %-12s %-11s %6s %6s %10s %10s\n",
              "t U/L", "step", "y_c/L", "V/U", "F_y", "max |u|", "phi<", "phi>",
              "p~ min", "p~ max");
  std::printf("%s\n", std::string(102, '-').c_str());

  auto phi_view = pf.phi();
  // Same clamp as the collision operator's equation of state, and for the same
  // reason: the body's effective mass is m_body - integral(chi rho), and an
  // unclamped rho lets that bracket collapse or change sign, which is a runaway
  // in Newton's equation rather than in the flow. Measured: the fluid stayed
  // finite at max|u| = 2.1e-02 while the body left the domain at 5e18 U.
  auto dens_of = KOKKOS_LAMBDA(Index n) {
    const Real q = phi_view(n);
    const Real c = q < Real(0) ? Real(0) : (q > Real(1) ? Real(1) : q);
    return Real(rho_l) + c * Real(rho_h - rho_l);
  };

  for (std::size_t step = 0; step <= nsteps; ++step) {
    pf.refresh();
    fl.compute_macroscopic();
    vf.refresh(fc);
    const auto R = body.refresh(dens_of);

    // Newton with the fictitious fluid removed; see PenalisedBody.hpp.
    const Real m_eff = m_body - R.fluid_mass;
    const Real acc = (m_eff > Real(1e-9))
                   ? (R.fy + m_eff * Real(-g)) / m_eff : Real(0);

    if (every && step % every == 0) {
      auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
      // p~ is the quantity the populations carry, and the one the pressure form
      // is badly conditioned in at a density ratio -- watching only phi and |u|
      // left the last run's failure with no precursor at all.
      auto hq = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
      auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      double umx = 0, pmin = 1e30, pmax = -1e30, qmin = 1e30, qmax = -1e30;
      bool bad = false;
      for (Index y = 1; y < ny - 1; ++y)
        for (Index x = 0; x < nx; ++x) {
          const double p = double(hp(d.id(x, y)));
          if (!std::isfinite(p)) bad = true;
          pmin = std::min(pmin, p);  pmax = std::max(pmax, p);
          const double q = double(hq(d.id(x, y)));
          qmin = std::min(qmin, q);  qmax = std::max(qmax, q);
          const double a = double(hu(d.id(x, y))), b = double(hv(d.id(x, y)));
          umx = std::max(umx, std::sqrt(a * a + b * b));
        }
      std::printf("%-8.3f %-9zu %-10.4f %-11.4f %-12.4e %-11.3e %6.3f %6.3f %10.2e %10.2e\n",
                  double(step) * U / double(L), step,
                  (double(body.shape.cy) - y_water) / double(L),
                  double(body.vy) / U, double(R.fy), umx, pmin, pmax, qmin, qmax);
      if (!P.dump.empty()) {
        char nm[512];
        auto at = [&](const char* f) {
          std::snprintf(nm, sizeof nm, "%s/rt_%04d_%s.bin", P.dump.c_str(), frame, f);
          return std::string(nm);
        };
        dump_field(at("phi"), nx, ny, [&](Index x, Index y) { return hp(d.id(x, y)); });
        dump_field(at("ux"),  nx, ny, [&](Index x, Index y) { return hu(d.id(x, y)); });
        dump_field(at("uy"),  nx, ny, [&](Index x, Index y) { return hv(d.id(x, y)); });
        const Rect b = body.shape;
        dump_field(at("body"), nx, ny,
                   [&](Index x, Index y) { return b.chi(Real(x), Real(y)); });
      }
      ++frame;
      if (bad) { std::printf("  DIVERGED\n"); break; }
    }
    if (step == nsteps) break;

    fl.step(true);
    pf.step();

    // Body state, explicit Euler at dt = 1 -- the fluid step is the timescale.
    body.vy += acc;
    body.shape.cy += body.vy;
    if (body.shape.cy - body.shape.hy < Real(2)) {   // reached the floor
      body.shape.cy = body.shape.hy + Real(2);
      body.vy = Real(0);
    }
  }
  std::printf("\n%d frame(s)%s\n", frame,
              P.dump.empty() ? " (pass -dump <dir> to write the fields)"
                             : " dumped; render with demonstrator/render_rt -body");
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Params P{48, 50.0, 2000.0, 0.05, 5.0, 0.02, 1e-4, 2.0, 1.0, 6.0, 150, "", "cm"};
    for (int i = 1; i < argc; ++i) {
      auto nx = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-l"))       { if (i+1<argc) P.W = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-ratio"))   nx(P.ratio);
      else if (!std::strcmp(argv[i], "-re"))      nx(P.Re);
      else if (!std::strcmp(argv[i], "-u"))       nx(P.U);
      else if (!std::strcmp(argv[i], "-iw"))      nx(P.iw);
      else if (!std::strcmp(argv[i], "-m"))       nx(P.M);
      else if (!std::strcmp(argv[i], "-sigma"))   nx(P.sigma);
      else if (!std::strcmp(argv[i], "-rhob"))    nx(P.body_rho);
      else if (!std::strcmp(argv[i], "-drop"))    nx(P.drop);
      else if (!std::strcmp(argv[i], "-tmax"))    nx(P.tmax);
      else if (!std::strcmp(argv[i], "-nframes")) { if (i+1<argc) P.nframes = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-dump"))    { if (i+1<argc) P.dump = argv[++i]; }
      else if (!std::strcmp(argv[i], "-op"))      { if (i+1<argc) P.op = argv[++i]; }
    }
    if (!std::strcmp(P.op, "bgk")) simulate<BgkColl>(P);
    else                           simulate<CmColl>(P);
  }
  Kokkos::finalize();
  return 0;
}
