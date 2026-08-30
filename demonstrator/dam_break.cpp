//==============================================================================
//  Dam break: a column of liquid released against a floor.
//
//  WHAT THE FREE SURFACE IS FOR, and a demonstrator rather than a validation
//  case for the reason the repository draws that line: the reference for this
//  problem is experimental -- Martin & Moyce's surge-front data -- and comparing
//  against a digitised curve tests the digitisation as much as it tests the
//  solver. The closed-form check lives in validation/gravity_wave.cpp, where the
//  answer is a dispersion relation and not a set of points read off a graph.
//
//  What this shows instead is the regime the gravity wave deliberately avoids.
//  A standing wave of amplitude three cells converts a handful of interface
//  cells a step and never stresses the topology. A collapsing column does
//  nothing else: the surge front runs, the interface stretches, a sheet climbs
//  the far wall, folds back and reconnects, and the cell classification is
//  rebuilt underneath all of it. If the conversion bookkeeping or the excess
//  mass redistribution is wrong this is where it shows, and it shows as mass.
//  So mass is printed every frame, and it is the number to read first.
//
//  ONE THING IS CHECKABLE WITHOUT A REFERENCE. Before the front feels the far
//  wall, shallow-water theory gives Ritter's solution for a dam break on a dry
//  bed: the front advances at
//
//      u_front = 2 sqrt(g h_0),
//
//  independent of the column's width. That is inviscid, it ignores the vertical
//  acceleration of the first instants, and it is wrong early and wrong late --
//  but between them the measured front speed should sit near it, and a solver
//  that had gravity or the free-surface condition wrong by a constant would not
//  land anywhere near. It is printed as a ratio, not asserted.
//
//  NO SURFACE TENSION, which for this problem is the right physics rather than a
//  limitation: the Bond number of a column tens of cells deep is enormous and
//  the flow is inertial throughout. The sheet that climbs the far wall would
//  need it, and the droplets it throws are therefore not to be trusted.
//
//  IT DIVERGES AT t* ~ 6, AND THE DEFAULT STOPS AT 5. This is a limit of the
//  implementation, not of the run length, and it is stated rather than hidden by
//  a shorter default alone. The sequence is reproducible: the surge reaches the
//  far wall at t* = 2.6, a sheet climbs it, and at t* ~ 6 that sheet folds back
//  onto the bulk. Reconnection produces films a cell thick and pockets of gas
//  with liquid on every side, and the cell classification cannot settle: the
//  interface count goes from 1500 to 11600 in one frame -- a conversion storm --
//  and the velocities follow it to NaN.
//
//  What is NOT the cause: the free-fall runaway, which was a separate defect
//  found here and fixed by the detached-cell rule in FreeSurfaceSolver::classify.
//  Before that rule, max|u| grew linearly at exactly g from t* = 2.6 onward and
//  reached Mach 1.2; after it, max|u| peaks at 0.149 and comes back DOWN to
//  0.098, which is a flow decelerating as it should. The divergence that remains
//  is topological and happens later.
//
//  The likely missing piece is the part of the literature this file's solver
//  does not have: a genuine treatment of the thin film, and a bubble model for
//  the enclosed gas that reconnection creates. Both are named in
//  FreeSurfaceSolver.hpp's limitation list. Until one of them exists, read this
//  to t* = 5.
//
//  Output is the same raw format demonstrator/render_rt reads, with the fill
//  level standing in for a phase field.
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

using L  = D2Q9;
using FS = FreeSurfaceSolver<L>;

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
    Index H = 96;                 // column height, in cells; everything scales off it
    double aspect = 4.0;          // domain width in column heights
    double width = 1.0;           // column width in column heights
    // t* = 5 is where this stops being trustworthy, not where it stops running.
    // See the banner.
    double g = 5e-5, nu = 5e-3, tmax = 5.0;
    int nframes = 160;
    std::string dump;
    for (int i = 1; i < argc; ++i) {
      auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-h"))       { if (i+1<argc) H = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-aspect"))  num(aspect);
      else if (!std::strcmp(argv[i], "-width"))   num(width);
      else if (!std::strcmp(argv[i], "-g"))       num(g);
      else if (!std::strcmp(argv[i], "-nu"))      num(nu);
      else if (!std::strcmp(argv[i], "-tmax"))    num(tmax);
      else if (!std::strcmp(argv[i], "-nframes")) { if (i+1<argc) nframes = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-dump"))    { if (i+1<argc) dump = argv[++i]; }
    }

    const Index nx = Index(aspect * double(H)), ny = Index(1.5 * double(H));
    const double h0 = double(H), w0 = width * double(H);
    // The natural timescale of a collapsing column, and what t* counts.
    const double t_ref = std::sqrt(h0 / g);
    const std::size_t nsteps = std::size_t(tmax * t_ref);
    const double u_ritter = 2.0 * std::sqrt(g * h0);

    std::printf("Dam break   %s free surface, central moments\n", L::name);
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("%dx%d   column %.0f wide x %.0f tall   g = %.1e   nu = %.1e\n",
                int(nx), int(ny), w0, h0, g, nu);
    std::printf("t_ref = sqrt(h0/g) = %.1f   t* = %.1f is %zu steps\n",
                t_ref, tmax, nsteps);
    std::printf("Ritter front speed 2 sqrt(g h0) = %.5f\n\n", u_ritter);

    Domain d(nx, ny, 1, /*periodic x*/ false, /*y*/ false, /*z*/ true);
    FS s(d);
    s.coll.omega = FS::omega_from_viscosity(Real(nu));
    s.set_gravity(Real(0), Real(-g));

    const Index nxi = nx, nyi = ny;
    s.set_geometry([&](Index x, Index y, Index) -> FsCell {
      return (y == 0 || y == nyi - 1 || x == 0 || x == nxi - 1) ? FsSolid : FsGas;
    });

    const Domain dd = d;
    const Index hx = d.hx, hy = d.hy;
    const Real w0r = Real(w0), h0r = Real(h0), gr = Real(g);
    constexpr Real ics = inv_cs2<L, Real>();
    s.initialize(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Real x = Real(px - hx), y = Real(py - hy);
      // A rectangular column against the left wall, with the partial cells at
      // its two free faces given the fraction that is actually liquid.
      const Real fx2 = w0r - (x - Real(0.5));
      const Real fy2 = h0r - (y - Real(0.5));
      const Real cx2 = fx2 < Real(0) ? Real(0) : (fx2 > Real(1) ? Real(1) : fx2);
      const Real cy2 = fy2 < Real(0) ? Real(0) : (fy2 > Real(1) ? Real(1) : fy2);
      const Real e = cx2 * cy2;
      const Real dz = h0r - y;
      const Real r = Real(1) + (e > Real(0) && dz > Real(0) ? gr * dz * ics : Real(0));
      return typename FS::Seed{e, r};
    });

    const double m0 = double(s.total_mass());
    const std::size_t every = nsteps / std::size_t(nframes > 0 ? nframes : 1);
    int frame = 0;
    double x_prev = w0; std::size_t t_prev = 0;

    std::printf("%-8s %-9s %-10s %-10s %-11s %-11s %-9s\n",
                "t*", "step", "front/h0", "u_f/Ritter", "max |u|",
                "mass drift", "g/i/f");
    std::printf("%s\n", std::string(78, '-').c_str());

    for (std::size_t t = 0; t <= nsteps; ++t) {
      if (every && t % every == 0) {
        auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
        auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        // The surge front: the farthest column holding any liquid at all.
        double xf = 0, umx = 0;
        bool bad = false;
        for (Index x = 1; x < nx - 1; ++x)
          for (Index y = 1; y < ny - 1; ++y) {
            const Index n = d.id(x, y);
            const std::uint8_t f = hf(n);
            if (f != FsFluid && f != FsInterface) continue;
            if (double(he(n)) > 0.05) xf = std::max(xf, double(x));
            const double a = double(hu(n)), b = double(hv(n));
            if (!std::isfinite(a) || !std::isfinite(b)) bad = true;
            umx = std::max(umx, std::sqrt(a * a + b * b));
          }
        const auto c = s.census();
        const double uf = (t > t_prev) ? (xf - x_prev) / double(t - t_prev) : 0.0;
        std::printf("%-8.3f %-9zu %-10.3f %-10.3f %-11.3e %-11.3e %d/%d/%d\n",
                    double(t) / t_ref, t, xf / h0, uf / u_ritter, umx,
                    double(s.total_mass()) / m0 - 1.0,
                    int(c.gas), int(c.interface_), int(c.fluid));
        x_prev = xf; t_prev = t;

        if (!dump.empty()) {
          char nm[512];
          auto at = [&](const char* f2) {
            std::snprintf(nm, sizeof nm, "%s/rt_%04d_%s.bin", dump.c_str(), frame, f2);
            return std::string(nm);
          };
          dump_field(at("phi"), nx, ny, [&](Index x, Index y) {
            const Index n = d.id(x, y);
            const std::uint8_t f = hf(n);
            return (f == FsFluid) ? 1.0f
                 : (f == FsInterface ? float(he(n)) : 0.0f);
          });
          dump_field(at("ux"), nx, ny,
                     [&](Index x, Index y) { return hu(d.id(x, y)); });
          dump_field(at("uy"), nx, ny,
                     [&](Index x, Index y) { return hv(d.id(x, y)); });
        }
        ++frame;
        if (bad) { std::printf("  DIVERGED\n"); break; }
      }
      if (t == nsteps) break;
      s.step();
    }
    std::printf("\n%d frame(s)%s\n", frame,
                dump.empty() ? " (pass -dump <dir> to write the fields)"
                             : " dumped; render with demonstrator/render_rt");
  }
  Kokkos::finalize();
  return 0;
}
