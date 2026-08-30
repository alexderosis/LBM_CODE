//==============================================================================
//  A square falling into water, on the FREE-SURFACE engine.
//
//  THE SAME PROBLEM AS demonstrator/water_entry.cpp, on a different model, and
//  the pairing is the point. That case runs the conservative Allen-Cahn phase
//  field with the pressure-based operator at a density ratio of 50, and every
//  hard thing in it comes from resolving the air: the pressure-gauge
//  conditioning of MultiphasePotentialBGK.hpp, the clamped equation of state,
//  the viscous interface force. This one does not resolve the air at all.
//
//  WHAT THAT CHANGES FOR THE BODY, and it is not a detail. PenalisedBody's
//  buoyancy is (m_b - m_f) g with m_f = integral of chi rho, the mass of fluid
//  standing in the penalised region. With a phase field that integral includes
//  the AIR the body displaces above the waterline, which is why
//  validation/floating_body.cpp carries Archimedes in the form
//
//      d = H (rho_b - rho_a) / (rho_w - rho_a)
//
//  and why the air is worth two per cent of the answer at a ratio of 50. Here
//  rho_a is not small, it is ZERO -- there are no populations above the surface
//  to integrate -- so the same expression collapses to
//
//      d = H rho_b / rho_w
//
//  exactly, with no correction to keep. An infinite density ratio is what this
//  engine has instead of a large one, and this is where it shows up as an answer
//  rather than as a stability margin.
//
//  THE DENSITY THE BODY SEES is the LIQUID content, epsilon rho, not rho. An
//  interface cell that is a tenth full holds a tenth of the fluid, and the
//  penalisation must scale with what is actually there or the body would feel a
//  full cell's resistance from a nearly empty one. A floor of 1e-3 is kept under
//  it: PenalisedBody divides by the local density to undo its own previous
//  force, and a body flying through gas would otherwise divide by zero. At that
//  floor the fictitious mass the body carries through the air is a thousandth of
//  its own, which is the right size for something the model says does not exist.
//
//==============================================================================
//  WHY OBSTACLE CELLS AND NOT VOLUME PENALISATION, which is what
//  demonstrator/water_entry.cpp uses against the phase field.
//
//  Penalisation was tried here first and it does not work, for a reason that is
//  structural rather than a tuning failure. The body fell correctly -- V/U =
//  -0.5006 against an analytic -0.5 -- and the free surface disintegrated the
//  moment it touched: interface cells from 192 to ten thousand in a hundred
//  steps. Four hypotheses were tested and all four falsified (conversion erasing
//  the force, the rotation solve, the impact speed, forcing the interface
//  cells), and the control settled it: with the body measuring the fluid but its
//  force disconnected, the surface was untouched for ten thousand steps.
//
//  The incompatibility is that direct forcing and explicit mass advection want
//  opposite things from a population. Penalisation moves the velocity a FINITE
//  amount in one step -- that is the factor of two in chi 2 rho (U - u*) -- and
//  at these speeds the Guo source is a fifth of a population. The free surface's
//  mass exchange is a difference of populations across a face, so a fifth of a
//  population is a fifth of the mass flux arriving as a step change.
//
//  So the body is a SOLID here: its cells are marked FsSolid, it bounces the
//  fluid back off a moving wall, and the force is measured by momentum exchange
//  link by link. Three things follow that are worth stating.
//
//   * NO FICTITIOUS MASS. PenalisedBody carries (m_b - m_f) g because its
//     interior is full of fluid it has to discount. A solid body has no
//     interior, so Newton is just m_b dU/dt = F + m_b g, and BUOYANCY IS NOT PUT
//     IN AT ALL -- it falls out of the pressure integral over the wetted
//     surface, which is what the momentum exchange sums.
//   * THE FORCE IS MEASURED, NOT DERIVED. PenalisedBody's reaction is a closed
//     form implied by its own solve, exact against the applied force but not an
//     independent quantity. This one is a sum over the links that actually
//     bounced.
//   * IT COSTS REFILL. A cell the body vacates holds no distribution and has to
//     be given one, and the mass the body displaces at its front has to arrive
//     at its back. Both are in FreeSurfaceSolver; neither has an analogue in the
//     penalised coupling.
//==============================================================================
//
//  WHAT IS NOT MODELLED, beyond FreeSurfaceSolver's own list: no air cushion.
//  A real flat-bottomed body traps a layer of air at impact, and the pressure in
//  that layer is a real part of the slamming load. Here the gas has no dynamics,
//  so the layer cannot form. The phase-field case can express it and this one
//  cannot, which is the trade the engine makes.
//==============================================================================
//  WHERE THIS STANDS, and it is not finished. READ THIS BEFORE TRUSTING A RUN.
//
//  THE DEFAULT CASE ENTERS THE WATER AND IS ARRESTED BY BUOYANCY, which is the
//  thing this demonstrator exists to show, and then it dies at t* ~ 6.2. The
//  fall is free while the body is dry -- the measured force is EXACTLY zero
//  above the waterline, because there are no populations to bounce, which is the
//  right answer rather than a lucky one -- and the deceleration after entry is
//  the pressure integral doing its job:
//
//     t*    y_c/L    V/U      F_y          mass drift   gas/interface/fluid
//     2.0   0.501   -0.921    1.36e-03     -2.9e-05     25272/1767/38121   entry
//     3.0  -0.148   -0.544   -6.92e-03      7.0e-05     25017/1966/38137
//     4.0  -0.675   -0.462    5.35e-02      4.8e-04     26128/ 898/38093
//     5.0  -1.032   -0.239    3.58e-01      1.5e-03     26332/ 779/38008
//     6.0  -1.210   -0.067   -3.72e+00      2.4e-03     26383/ 834/37903
//     6.5      nan
//
//  Four time units of a body being stopped by water, with the fluid count within
//  0.7 per cent of where it started. Then the tilt runs away -- 1.21, -0.95,
//  -4.91 degrees on the three frames before the end -- and the surface is
//  destroyed by the body, not the other way round.
//
//  THE FAILURE IS THE MOVING MASK, AND TWO SEPARATE RUNS SAY SO.
//
//   * ROTATION LOCKED (WE_NOROT=1) IS WORSE, NOT BETTER: it explodes at t* ~ 2.2,
//     at the instant of contact, and in the VERTICAL velocity (V/U = 7487). A
//     perfectly axis-aligned flat bottom meeting a flat surface converts an
//     entire row of cells in a single pass, all correlated.
//   * AN INITIAL TILT IS ALSO WORSE: -theta 5 dies at t* ~ 1.8, on the corner
//     impact, with the interface count going 1210 -> 3372 while the fluid count
//     has not moved at all.
//
//  So it is not the rotation solve and it is not the impact speed. What both
//  failures share with the good run's ending is a mask that changes at many
//  cells in one pass -- a flat row converting at once, a corner converting, a
//  staircase of a rotated rectangle flipping cells as it turns.
//
//  THAT POINTS AT ONE PLACE. mass_exchange()'s antisymmetry is the only thing
//  conserving mass in this method, and it assumes BOTH cells agree on the mask
//  for the whole pass. transfer_covered_mass() conserves the total but does not
//  make the per-face exchange antisymmetric across a boundary that moved, and
//  the fix is to teach the mass exchange about the moving boundary rather than
//  to keep patching the ledger around it. Not attempted here.
//
//  TWO REAL BUGS WERE FOUND AND FIXED GETTING THIS FAR, and both are worth
//  keeping written down because neither can arise in a two-fluid model:
//
//   * TANK WALLS IMPERSONATING THE BODY. Giving every FsSolid cell the body's
//     velocity made the floor and the lid pump the fluid, contribute their links
//     to the body's force, and convert themselves into interface cells on the
//     first call. The body needs its own mask, separate from static geometry.
//   * MOMENTUM EXCHANGE RETURNS ABSOLUTE PRESSURE. Around a fully wetted closed
//     body that cancels, since sum_q w_q c_q = 0, leaving the gradient, which is
//     buoyancy. A PARTLY SUBMERGED body has no such cancellation: its dry side
//     faces gas, and gas here is a void with no populations to push back. The
//     body felt a whole atmosphere on its wetted face and nothing on top --
//     twenty times the physical slamming load, reversing its velocity in forty
//     steps. Subtracting 2 w_q rho_G from every WET link is identical to adding
//     it on every dry one, because the closed surface sums to zero, and it costs
//     no extra pass. What is left is gauge pressure, which is what a surface
//     force is.
//
//  The demonstrator is kept, and kept building, because everything up to t* ~ 6
//  is right and the failure is a specific measured one rather than a mystery.
//==============================================================================
#include "core/Types.hpp"
#include "solver/FreeSurfaceSolver.hpp"


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

using L    = D2Q9;
using Coll = CentralMoments<L, Guo, RawPopulations>;
using FS   = FreeSurfaceSolver<L, Coll>;

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
    Index W = 40;
    double rho_b = 0.6, drop = 1.0, U = 0.04, nu = 5e-3, tmax = 14.0, theta = 0.0;
    int nframes = 180;
    std::string dump;
    for (int i = 1; i < argc; ++i) {
      auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-l"))       { if (i+1<argc) W = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-rhob"))    num(rho_b);
      else if (!std::strcmp(argv[i], "-drop"))    num(drop);
      else if (!std::strcmp(argv[i], "-u"))       num(U);
      else if (!std::strcmp(argv[i], "-nu"))      num(nu);
      else if (!std::strcmp(argv[i], "-tmax"))    num(tmax);
      else if (!std::strcmp(argv[i], "-theta"))   num(theta);
      else if (!std::strcmp(argv[i], "-nframes")) { if (i+1<argc) nframes = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-dump"))    { if (i+1<argc) dump = argv[++i]; }
    }

    const Index nx = 6 * W, ny = 7 * W;
    const double y_water = 4.0 * double(W);
    const double g  = U * U / (2.0 * drop * double(W));
    const std::size_t nsteps = std::size_t(tmax * double(W) / U);
    const double half = 0.5 * double(W);
    const double area = double(W) * double(W);
    const double m_b  = rho_b * area;
    const double I_b  = m_b * (half * half + half * half) / 3.0;
    const double d_exact = double(W) * rho_b;

    std::printf("Water entry, free surface + obstacle cells   %s, central moments\n", L::name);
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("%dx%d   L = %d   body rho = %.2f x water   tilt = %.1f deg\n",
                int(nx), int(ny), int(W), rho_b, theta);
    std::printf("g = %.3e   U = %.3f   nu = %.1e   m_b = %.1f   I_b = %.3e   %zu steps\n",
                g, U, nu, m_b, I_b, nsteps);
    std::printf("Archimedes draft (rho_a = 0): d = H rho_b = %.2f cells\n\n", d_exact);

    Domain d(nx, ny, 1, true, false, true);
    FS s(d);
    s.coll.omega = FS::omega_from_viscosity(Real(nu));
    s.coll.forcing.fy = Real(-g);           // gravity on the liquid

    const Index nyi = ny;
    s.set_geometry([&](Index, Index y, Index) -> FsCell {
      return (y == 0 || y == nyi - 1) ? FsSolid : FsGas;
    });

    const Domain dd = d;
    const Index hx = d.hx, hy = d.hy;
    const Real yw = Real(y_water), gr = Real(g);
    constexpr Real ics = inv_cs2<L, Real>();
    s.initialize(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Real y = Real(py - hy);
      Real e = yw - (y - Real(0.5));
      e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
      const Real dz = yw - y;
      const Real r = Real(1) + (dz > Real(0) ? gr * dz * ics : Real(0));
      return typename FS::Seed{e, r};
    });

    // Body state, integrated by the driver. No fictitious mass: see the banner.
    double bx = 0.5 * double(nx), by = y_water + drop * double(W) + half;
    double vx = 0, vy = 0, om = 0, th = theta * M_PI / 180.0;

    const std::size_t every = nsteps / std::size_t(nframes > 0 ? nframes : 1);
    int frame = 0;
    const double m0 = double(s.total_mass());

    std::printf("%-8s %-9s %-9s %-9s %-8s %-11s %-11s %-9s\n",
                "t U/L", "step", "y_c/L", "V/U", "tilt", "F_y", "massdrift", "g/i/f");
    std::printf("%s\n", std::string(86, '-').c_str());

    for (std::size_t t = 0; t <= nsteps; ++t) {
      const Real cx = Real(bx), cy = Real(by);
      const Real ct = Real(std::cos(th)), st = Real(std::sin(th));
      const Real hh = Real(half);
      const Real bvx = Real(vx), bvy = Real(vy), bom = Real(om);
      // Inside test: an exact rotated square, not a smoothed indicator -- a
      // solid cell either is or is not.
      auto inside = KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; dd.coords(n, px, py, pz);
        const Real ex = Real(px - hx) - cx, ey = Real(py - hy) - cy;
        const Real X =  ct * ex + st * ey;
        const Real Y = -st * ex + ct * ey;
        return (X > -hh && X < hh && Y > -hh && Y < hh);
      };
      auto wall_vel = KOKKOS_LAMBDA(Index n, Real u[3]) {
        Index px, py, pz; dd.coords(n, px, py, pz);
        const Real ex = Real(px - hx) - cx, ey = Real(py - hy) - cy;
        u[0] = bvx - bom * ey;  u[1] = bvy + bom * ex;  u[2] = Real(0);
      };
      s.move_obstacle(inside, wall_vel, cx, cy);

      if (every && t % every == 0) {
        auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
        auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hw = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        bool bad = !std::isfinite(by) || !std::isfinite(vy);
        const auto c = s.census();
        const auto ex = s.exchange();
        std::printf("%-8.3f %-9zu %-9.4f %-9.4f %-8.2f %-11.4e %-11.3e %d/%d/%d\n",
                    double(t) * U / double(W), t, (by - y_water) / double(W),
                    vy / U, th * 180.0 / M_PI, double(ex.fy),
                    double(s.total_mass()) / m0 - 1.0,
                    int(c.gas), int(c.interface_), int(c.fluid));
        if (!dump.empty()) {
          char nm[512];
          auto at = [&](const char* f2) {
            std::snprintf(nm, sizeof nm, "%s/rt_%04d_%s.bin", dump.c_str(), frame, f2);
            return std::string(nm);
          };
          dump_field(at("phi"), nx, ny, [&](Index x, Index y) {
            const Index n = d.id(x, y);
            const std::uint8_t f = hf(n);
            return (f == FsFluid) ? 1.0f : (f == FsInterface ? float(hv(n)) : 0.0f);
          });
          dump_field(at("ux"), nx, ny, [&](Index x, Index y) { return hu(d.id(x, y)); });
          dump_field(at("uy"), nx, ny, [&](Index x, Index y) { return hw(d.id(x, y)); });
          dump_field(at("body"), nx, ny, [&](Index x, Index y) {
            return (hf(d.id(x, y)) == FsSolid && y > 0 && y < ny - 1) ? 1.0f : 0.0f;
          });
        }
        ++frame;
        if (bad) { std::printf("  DIVERGED\n"); break; }
      }
      if (t == nsteps) break;
      s.step();

      // Newton, with no fictitious-mass correction to make.
      const auto ex = s.exchange();
      vx += double(ex.fx) / m_b;
      vy += double(ex.fy) / m_b - g;
      if (!std::getenv("WE_NOROT")) om += double(ex.torque) / I_b;
      bx += vx;  by += vy;  th += om;
      const double diag = std::sqrt(2.0) * half;
      if (by - diag < 2.0) { by = diag + 2.0; if (vy < 0) vy = 0; }
    }
    std::printf("\n%d frame(s)%s\n", frame,
                dump.empty() ? " (pass -dump <dir> to write the fields)"
                             : " dumped; render with demonstrator/render_rt");
  }
  Kokkos::finalize();
  return 0;
}
