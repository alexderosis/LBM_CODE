//==============================================================================
//  Rayleigh-Taylor instability, by the colour-gradient model.
//
//  THE SAME PROBLEM AS demonstrator/rayleigh_taylor.cpp, ON A DIFFERENT MODEL,
//  which is the only reason it exists. That case runs the conservative
//  Allen-Cahn phase field on D2Q9 with the pressure-based operator; this one
//  runs Saito et al.'s colour gradient on D3Q27 with nonorthogonal central
//  moments. Nothing is shared but the initial condition and the dump format, so
//  putting the two films side by side is a comparison of two interface-capturing
//  schemes on identical physics rather than of two parameter choices.
//
//  A THIN PERIODIC SLAB, and that is a deliberate compromise. The colour-
//  gradient operator is D3Q27 and there is no two-dimensional version of it --
//  phi_i, B_i and sigma = 4 A tau / 9 are all derived for that lattice. Running
//  the full three-dimensional problem would be the honest thing and it is also
//  fifty times the work; a slab four cells deep and periodic in z carries the
//  same physics as long as nothing breaks the symmetry in z, and nothing here
//  does. What it CANNOT show is the three-dimensional secondary instability that
//  a real Rayleigh-Taylor spike develops, and that absence is a property of the
//  slab, not of the model.
//
//  NO-SLIP WALLS TOP AND BOTTOM, periodic in x and z. That is the classic
//  Rayleigh-Taylor benchmark geometry and it is what demonstrator/
//  rayleigh_taylor.cpp uses, so the two films are comparable frame for frame.
//  The walls are halfway bounce-back, which under Esoteric Pull costs nothing at
//  all -- a solid cell is skipped, and the reflection is the identity on the
//  storage. Both colours bounce from the same branch.
//
//  An earlier version of this case was periodic in y instead, because the solver
//  had no wall condition. That is a WORSE problem, not merely a different one:
//  periodicity forces a second interface at the seam, and although that one is
//  stable (light below heavy), the falling spike eventually reaches it and what
//  happens next is the box rather than the instability. With walls the spike
//  simply lands, which is the physical answer.
//
//  THE COLOUR GRADIENT AT THE WALL IS NEUTRAL -- ninety degrees, no wetting
//  preference. See the banner in ColourGradientSolver.hpp: it is the choice that
//  assumes least, not a validated contact angle, and a Rayleigh-Taylor spike is
//  the wrong problem to notice the difference on anyway.
//
//  THE BODY FORCE IS MEASURED AGAINST THE MEAN DENSITY, which the walls no
//  longer force but which is still the right thing. Plain rho g would be
//  physical here -- walls can carry a hydrostatic gradient -- but this equation
//  of state ties pressure to density, so establishing that gradient means
//  compressing the fluid, and at these numbers it is a 6% density change in the
//  heavy phase arriving as an acoustic transient at t = 0. Subtracting the mean
//  leaves exactly the buoyancy difference that drives the instability and starts
//  the run in balance. The full density contrast is retained; only its mean is
//  removed.
//
//  Output is the same raw format demonstrator/render_rt reads, taken on a single
//  z-slice, with phi mapped from its native [-1, 1] to the [0, 1] that renderer
//  expects.
//==============================================================================
#include "collision/ColourGradient.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ColourGradientSolver.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

using L   = D3Q27;
using CG  = ColourGradient<L>;
using Slv = ColourGradientSolver<L, EsotericPull<L>, CG>;

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

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    int W = 64, nz = 4, nframes = 90;
    double At = 0.5, Re = 256, U = 0.04, A = 2e-4, iw = 5.0, tmax = 3.0;
    std::string dump;
    for (int i = 1; i < argc; ++i) {
      auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-w"))       { if (i+1<argc) W = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-nz"))      { if (i+1<argc) nz = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-at"))      num(At);
      else if (!std::strcmp(argv[i], "-re"))      num(Re);
      else if (!std::strcmp(argv[i], "-u"))       num(U);
      else if (!std::strcmp(argv[i], "-a"))       num(A);
      else if (!std::strcmp(argv[i], "-iw"))      num(iw);
      else if (!std::strcmp(argv[i], "-tmax"))    num(tmax);
      else if (!std::strcmp(argv[i], "-nframes")) { if (i+1<argc) nframes = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-dump"))    { if (i+1<argc) dump = argv[++i]; }
    }

    const Index nx = W, ny = 4 * W;
    const double gamma = (1.0 + At) / (1.0 - At);
    const double ab = 8.0 / 27.0;
    const double ar = 1.0 - (1.0 - ab) / gamma;
    const double rho_b0 = 1.0, rho_r0 = gamma;
    const double g  = U * U / double(W);
    const double nu = double(W) * U / Re;
    const double tau = 3.0 * nu + 0.5;
    const double t_ref = std::sqrt(double(W) / (g * At));
    const std::size_t nsteps = std::size_t(tmax * t_ref);

    std::printf("Rayleigh-Taylor   D3Q27 colour gradient, nonorthogonal central moments\n");
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("%dx%dx%d slab, no-slip walls in y   At = %.4f (rho_H/rho_L = %.2f)"
                "   Re = %.0f   nu = %.3e\n",
                int(nx), int(ny), nz, At, gamma, Re, nu);
    std::printf("g = %.3e   U = %.3f   tau = %.4f   A = %.2e   sigma = %.3e\n",
                g, U, tau, A, 4.0 * A * tau / 9.0);
    std::printf("t* = %.1f is %zu steps (t_ref = %.1f)\n\n", tmax, nsteps, t_ref);

    Domain d(nx, ny, Index(nz), /*periodic x*/ true, /*y*/ false, /*z*/ true);
    CG cg;
    cg.alpha_r = Real(ar);      cg.alpha_b = Real(ab);
    cg.rho_r0  = Real(rho_r0);  cg.rho_b0  = Real(rho_b0);
    cg.nu_r    = Real(nu);      cg.nu_b    = Real(nu);
    cg.A       = Real(A);
    cg.beta    = Real(0.7);
    cg.omega_bulk = Real(1);
    cg.by      = Real(-g);
    cg.rho_ref = Real(0.5 * (rho_r0 + rho_b0));   // see the banner
    Slv s(d, cg);
    // Walls before initialize(): the seeding reads the flags, and a wall cell
    // must be seeded like any other -- it holds populations in transit.
    const Index nyi = ny;
    s.set_geometry([&](Index, Index y, Index) -> CellType {
      return (y == 0 || y == nyi - 1) ? Solid : Fluid;
    });

    // Heavy (red) above light (blue), with a single-mode perturbation at the box
    // wavelength -- the same initial condition rayleigh_taylor.cpp uses.
    const Real y0 = Real(0.5 * double(ny)), amp = Real(0.1 * double(W));
    const Real Wr = Real(W), iwr = Real(iw);
    const Real rr0 = Real(rho_r0), rb0 = Real(rho_b0);
    const Index hx = d.hx, hy = d.hy, hz = d.hz;
    const Domain dd = d;
    s.initialize(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Real x = Real(px - hx), y = Real(py - hy);
      const Real yi = y0 + amp * Kokkos::cos(Real(2) * Real(M_PI) * x / Wr);
      const Real t = Kokkos::tanh(Real(2) * (y - yi) / iwr);
      (void)pz; (void)hz;
      return Slv::Colours{Real(0.5) * rr0 * (Real(1) + t),
                          Real(0.5) * rb0 * (Real(1) - t)};
    });

    const std::size_t every = nsteps / std::size_t(nframes > 0 ? nframes : 1);
    int frame = 0;
    std::printf("%-9s %-9s %-11s %-11s %-11s %-11s\n",
                "t*", "step", "max |u|", "min phi", "max phi", "red mass");
    std::printf("%s\n", std::string(70, '-').c_str());
    const double m0 = double(s.total_red());
    const Index zs = Index(nz / 2);           // the slice that gets rendered

    for (std::size_t step = 0; step <= nsteps; ++step) {
      s.refresh();
      if (every && step % every == 0) {
        auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.phi());
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        double umx = 0, pmin = 1e30, pmax = -1e30;
        bool bad = false;
        for (Index y = 0; y < ny; ++y)
          for (Index x = 0; x < nx; ++x) {
            const Index n = d.id(x, y, zs);
            const double p = double(hp(n));
            if (!std::isfinite(p)) bad = true;
            pmin = std::min(pmin, p);  pmax = std::max(pmax, p);
            const double a = double(hu(n)), b = double(hv(n));
            umx = std::max(umx, std::sqrt(a * a + b * b));
          }
        std::printf("%-9.3f %-9zu %-11.3e %-11.4f %-11.4f %-11.6f\n",
                    double(step) / t_ref, step, umx, pmin, pmax,
                    double(s.total_red()) / m0);
        if (!dump.empty()) {
          char nm[512];
          auto at = [&](const char* f) {
            std::snprintf(nm, sizeof nm, "%s/rt_%04d_%s.bin", dump.c_str(), frame, f);
            return std::string(nm);
          };
          // phi is native [-1, 1]; render_rt expects [0, 1].
          dump_field(at("phi"), nx, ny, [&](Index x, Index y) {
            return 0.5 * (1.0 + double(hp(d.id(x, y, zs)))); });
          dump_field(at("ux"), nx, ny,
                     [&](Index x, Index y) { return hu(d.id(x, y, zs)); });
          dump_field(at("uy"), nx, ny,
                     [&](Index x, Index y) { return hv(d.id(x, y, zs)); });
        }
        ++frame;
        if (bad) { std::printf("  DIVERGED\n"); break; }
      }
      if (step == nsteps) break;
      s.step();
    }
    std::printf("\n%d frame(s)%s\n", frame,
                dump.empty() ? " (pass -dump <dir> to write the fields)"
                             : " dumped; render with demonstrator/render_rt");
  }
  Kokkos::finalize();
  return 0;
}
