//==============================================================================
//  A TEMPLATE FOR YOUR OWN PROBLEM: flow past a square cylinder in a channel.
//
//  Copy this file, rename it, change the four marked blocks. It is written to be
//  read top to bottom rather than to be short, and every choice below says why
//  it was made -- because the choices, not the syntax, are where a new case goes
//  wrong.
//
//  The physics: a channel with no-slip walls, periodic along the flow, a square
//  obstacle in the middle, driven by a uniform body force. Above Re ~ 50 the
//  wake becomes unsteady and sheds vortices. It is here because it exercises
//  every piece a real case needs -- geometry, forcing, an initial condition, a
//  diagnostic and an output file -- and because "an obstacle in a flow" is what
//  most people arrive wanting.
//
//  Build and run:
//     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
//           -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_THREADS=ON
//     cmake --build build -j4 --target flow_past_square
//     ./build/examples/flow_past_square --kokkos-num-threads=4
//
//  Flags: -n <ny>  -re <Reynolds>  -steps <n>  -out <file.csv>
//
//  THIS IS NOT A VALIDATION CASE. It has no exact answer to check against, so it
//  lives in examples/ and not in validation/. If you write something with a
//  known answer -- analytic, a published table, a convergence rate -- put it in
//  validation/ instead and add it to the first foreach list there so ctest runs
//  it. That distinction is the repository's, and it is worth keeping.
//==============================================================================

// INCLUDE ORDER MATTERS HERE. FluidSolver.hpp uses Macro, which BGK.hpp defines
// and FluidSolver.hpp does not include, so putting these in alphabetical order
// fails to compile. Validation cases never meet this because Campaign.hpp orders
// the includes for them.
#include "collision/BGK.hpp"
#include "solver/FluidSolver.hpp"
#include "memory/EsotericPull.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace lbm;

//------------------------------------------------------------------------------
//  BLOCK 1 of 4 -- THE DISCRETISATION.
//
//  D2Q9 for a 2D problem; use D3Q27 in 3D. Do not reach for D3Q19 unless you
//  know you want it: it is not a product lattice, so the central-moment and
//  multiphase operators either fall back to a different basis or refuse.
//
//  BGK is the simplest operator and the right default. Swap the alias for a
//  central-moment operator if you need Galilean invariance at higher Mach or
//  better stability at low viscosity; nothing else in this file changes.
//
//  Guo is the forcing policy -- it is a template parameter rather than a runtime
//  flag because a force enters the equilibrium AND the collision, and getting
//  only one of them is a silent second-order error. NoForcing is the default;
//  ask for Guo explicitly, as here, when you drive the flow with a body force.
//------------------------------------------------------------------------------
using L    = D2Q9;
using Coll = BGK<L, SecondOrderEquilibrium<L>, Guo>;
using Sol  = FluidSolver<L, EsotericPull<L>, Coll>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    //--------------------------------------------------------------------------
    //  Parameters, in LATTICE UNITS. Everything in this code is dimensionless:
    //  dx = dt = 1, and you choose the numbers that fix the physics.
    //
    //  The one that bites: u0 is a Mach number in disguise. The lattice sound
    //  speed is 1/sqrt(3), so Ma = u0*sqrt(3), and compressibility error grows
    //  as Ma^2. Keep u0 <= 0.05 unless you have checked what it costs you.
    //--------------------------------------------------------------------------
    Index ny = 96;              // channel height in cells; everything scales off it
    double Re = 40.0;           // Reynolds number based on the obstacle side
    double u0 = 0.04;           // target centreline velocity -> Ma = 0.069
    std::size_t steps = 40000;
    std::string out = "profile.csv";

    for (int i = 1; i < argc; ++i) {
      if      (!std::strcmp(argv[i], "-n")     && i + 1 < argc) ny    = std::atoi(argv[++i]);
      else if (!std::strcmp(argv[i], "-re")    && i + 1 < argc) Re    = std::atof(argv[++i]);
      else if (!std::strcmp(argv[i], "-u0")    && i + 1 < argc) u0    = std::atof(argv[++i]);
      else if (!std::strcmp(argv[i], "-steps") && i + 1 < argc) steps = std::atoll(argv[++i]);
      else if (!std::strcmp(argv[i], "-out")   && i + 1 < argc) out   = argv[++i];
    }

    const Index nx   = 4 * ny;              // long enough for a wake to develop
    const Index side = ny / 6;              // obstacle side, the Reynolds length
    const Index cx   = nx / 4, cy = ny / 2; // obstacle centre, upstream of middle

    // nu follows from Re and the length scale. Deriving it rather than setting
    // it is the habit worth keeping: it makes the Reynolds number the thing you
    // control, and stops tau drifting when you change resolution.
    const double nu  = u0 * double(side) / Re;
    const double tau = 3.0 * nu + 0.5;

    // tau -> 1/2 is the stability floor: omega -> 2 and the scheme rings. This
    // check exists because "it diverged" almost always means tau came out too
    // small, and the number that says so belongs on screen before the run.
    //
    // 0.51 is the threshold below which this warns. The defaults give 0.548,
    // comfortably clear. Raising -re or
    // lowering -n both push tau down, because nu = u0 * side / Re -- so a bigger
    // Reynolds number is bought either with a bigger grid or with a smaller u0,
    // never for free. That trade is the single most useful thing to understand
    // about setting up an LBM case.
    std::printf("Flow past a square cylinder   %s + %s + EsotericPull\n",
                L::name, Coll::name);
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("  %lldx%lld   side = %lld   Re = %.0f   u0 = %.3f (Ma = %.3f)\n",
                (long long)nx, (long long)ny, (long long)side, Re, u0,
                u0 * std::sqrt(3.0));
    std::printf("  nu = %.6f   tau = %.4f%s\n", nu, tau,
                tau < 0.51 ? "   <-- TOO CLOSE TO 1/2, EXPECT INSTABILITY" : "");
    std::printf("  %zu steps\n\n", steps);

    //--------------------------------------------------------------------------
    //  BLOCK 2 of 4 -- THE DOMAIN AND THE GEOMETRY.
    //
    //  Domain(nx, ny, nz, periodic_x, periodic_y, periodic_z). Periodic along
    //  the flow, walled across it.
    //
    //  set_geometry's callback runs ON THE HOST over interior coordinates and
    //  returns a CellType per cell: Fluid, Solid (halfway bounce-back),
    //  RegWall (regularised velocity boundary), Excluded. Being on the host is
    //  deliberate -- it means you can read a mesh, an image or an STL here
    //  without any device-side machinery.
    //
    //  CHANGE THIS BLOCK for your own geometry. A circle is
    //  (x-cx)^2 + (y-cy)^2 < r*r; an aerofoil is a signed distance function; a
    //  city is a height field. Anything you can answer per cell works.
    //--------------------------------------------------------------------------
    Domain d(nx, ny, 1, /*px=*/true, /*py=*/false, /*pz=*/true);

    Coll coll;
    coll.omega = Coll::omega_from_viscosity(Real(nu));
    // The driving force, STARTING value. For a plain channel of height H the
    // balance sustaining a mean velocity U against wall friction is
    // G = 8 nu U / H^2 -- the Poiseuille result. With an obstacle in the way it
    // is only a starting guess, and the loop below corrects it.
    double G = 8.0 * nu * u0 / (double(ny) * double(ny));
    coll.forcing = Guo{Real(G), Real(0), Real(0)};

    Sol s(d, coll);

    const Index hs = side / 2;
    s.set_geometry([&](Index x, Index y, Index) -> CellType {
      if (y == 0 || y == ny - 1) return Solid;          // channel walls
      if (std::abs(long(x) - long(cx)) <= long(hs) &&
          std::abs(long(y) - long(cy)) <= long(hs)) return Solid;   // the obstacle
      return Fluid;
    });

    //--------------------------------------------------------------------------
    //  BLOCK 3 of 4 -- THE INITIAL CONDITION.
    //
    //  The callback runs on the DEVICE (hence KOKKOS_LAMBDA) and returns a
    //  FlowState{rho, ux, uy, uz} per cell. Solid cells are skipped for you.
    //
    //  Starting from a plausible field rather than from rest saves transient
    //  time, and a tiny asymmetry breaks the symmetry that would otherwise keep
    //  the wake artificially steady far past the physical shedding threshold.
    //  Left perfectly symmetric, this case sheds only once round-off has grown
    //  enough to seed it, which can be tens of thousands of steps.
    //--------------------------------------------------------------------------
    const Domain dd = d; const Index hy = d.hy;
    const Real u0r = Real(u0); const Index nyr = ny;
    s.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Real yy = Real(py - hy);
      // Parabolic profile across the channel, plus a 1% transverse kick.
      const Real f = Real(4) * yy * (Real(nyr - 1) - yy)
                   / Real((nyr - 1) * (nyr - 1));
      return FlowState{Real(1), u0r * f, Real(0.01) * u0r * f, Real(0)};
    });

    //--------------------------------------------------------------------------
    //  BLOCK 4 of 4 -- THE TIME LOOP AND WHAT YOU MEASURE.
    //
    //  s.step() advances one timestep. The macroscopic fields are only computed
    //  when you ask -- either step(true) or compute_macroscopic() -- because on
    //  a GPU that is a separate pass over memory and you do not want it every
    //  step just to print occasionally.
    //
    //  CHANGE THIS BLOCK for your own diagnostic. What is here: peak velocity,
    //  the Reynolds number it implies, and total mass. Mass is printed on
    //  purpose -- it is conserved to round-off by construction, so a drifting
    //  value means something is wrong with the geometry or the boundaries, and
    //  it is the cheapest early warning this code has.
    //--------------------------------------------------------------------------
    //  A FIXED BODY FORCE WOULD NOT GIVE YOU THE REYNOLDS NUMBER YOU ASKED FOR,
    //  and this is worth understanding before you copy the file. Held constant,
    //  G drives the flow until it balances the TOTAL drag -- walls plus obstacle
    //  -- and the obstacle's share is not in the Poiseuille estimate above. The
    //  flow therefore drifts to whatever velocity that balance implies, on the
    //  momentum-diffusion timescale H^2/nu, which here is 576,000 steps. Measured
    //  before this controller existed: Re_eff fell from 49 to 26 over 30,000
    //  steps and was still falling.
    //
    //  So G is adjusted instead to hold the mean velocity at u0, which makes the
    //  requested Re the one you actually run at. This is an integral controller
    //  and nothing cleverer: nudge G by the velocity error, slowly enough not to
    //  ring. total_momentum(0)/total_mass() is the mean streamwise velocity over
    //  the fluid, and both are single reductions.
    const std::size_t ctrl_every = 100;
    const double gain = 1.0 / 400.0;

    const double m0 = double(s.total_mass());
    const std::size_t every = steps / 20 ? steps / 20 : 1;

    //  Re_eff is computed from the MEAN velocity, because that is the velocity
    //  nu was derived from -- report it against max|u| instead and a converged
    //  run reads Re = 70 when you asked for 40, purely from peak-over-mean in a
    //  channel. Keep the two consistent, and print max|u| beside it as the
    //  Mach-number and divergence check it actually is.
    std::printf("  %-9s %-12s %-12s %-10s %-12s\n",
                "step", "max|u|", "mean u", "Re_eff", "mass drift");
    std::printf("  %s\n", std::string(62, '-').c_str());

    for (std::size_t t = 0; t <= steps; ++t) {
      if (t % every == 0) {
        s.compute_macroscopic();
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        double um = 0; bool ok = true;
        for (Index y = 1; y < ny - 1; ++y)
          for (Index x = 0; x < nx; ++x) {
            const Index n = d.id(x, y, 0);
            const double a = double(hu(n)), b = double(hv(n));
            if (!std::isfinite(a) || !std::isfinite(b)) { ok = false; continue; }
            um = std::max(um, std::sqrt(a * a + b * b));
          }
        const double ubar = double(s.total_momentum(0)) / double(s.total_mass());
        std::printf("  %-9zu %-12.4e %-12.4e %-10.1f %+.3e\n",
                    t, um, ubar, ubar * double(side) / nu,
                    double(s.total_mass()) / m0 - 1.0);
        if (!ok) { std::printf("  DIVERGED -- lower u0, or raise nu (lower Re)\n"); break; }
      }
      if (t % ctrl_every == 0 && t > 0) {
        const double um = double(s.total_momentum(0)) / double(s.total_mass());
        G += gain * (u0 - um);
        if (G < 0) G = 0;                 // never suck the flow backwards
        // collision() hands back a mutable reference to the operator the solver
        // holds; the next step() picks the new force up.
        s.collision().forcing = Guo{Real(G), Real(0), Real(0)};
      }
      if (t < steps) s.step();
    }

    //--------------------------------------------------------------------------
    //  Output: a CSV of the wake profile one obstacle-width downstream. Plain
    //  text on purpose -- gnuplot, python, a spreadsheet, anything reads it. For
    //  a full field use src/io/VtiWriter.hpp's write_vti(path, solver) and open
    //  the result in ParaView.
    //--------------------------------------------------------------------------
    s.compute_macroscopic();
    auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
    auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
    if (std::FILE* f = std::fopen(out.c_str(), "w")) {
      std::fprintf(f, "# y, ux, uy   at x = %lld (one side downstream of the body)\n",
                   (long long)(cx + side));
      for (Index y = 0; y < ny; ++y) {
        const Index n = d.id(cx + side, y, 0);
        std::fprintf(f, "%lld, %.8e, %.8e\n",
                     (long long)y, double(hu(n)), double(hv(n)));
      }
      std::fclose(f);
      std::printf("\n  wrote %s\n", out.c_str());
    } else {
      std::printf("\n  could not open %s for writing\n", out.c_str());
    }
  }
  Kokkos::finalize();
  return 0;
}
