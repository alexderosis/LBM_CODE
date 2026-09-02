//==============================================================================
//  The free surface, checked WITHOUT a GPU.
//
//  Every pass in freesurface.cuh is a plain LBM_HD gather, so the serial loop
//  here runs the same computation the kernels do -- not an approximation to it.
//  What this cannot check is the launch: grid configuration, register pressure,
//  and whether the kernel boundaries really fence.
//
//  THE ORDER OF THE CHECKS IS THE ORDER OF WHAT CAN GO SILENTLY WRONG.
//
//    1  mass, in a quiescent pool     nothing enforces conservation here but the
//                                     antisymmetry of one expression
//    2  hydrostatic equilibrium       the free-surface condition's two moments
//    3  the interface invariant       a Fluid cell must NEVER touch a Gas cell
//    4  a collapsing column           all of the above, through several hundred
//                                     conversions
//
//  Mass first, and deliberately. Everywhere else in this code density is a
//  moment and conservation is a property of the collision, checkable by algebra.
//  Here mass moves because populations cross faces, and the only thing that
//  conserves it is that what a cell sends equals what its neighbour receives. A
//  scheme that got that wrong would still look like a fluid.
//
//  Build:  c++ -std=c++17 -O2 -Iinclude test/host_freesurface.cpp -o host_freesurface
//==============================================================================
#include "lbm/hostsim.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace lbm;

static int failures = 0;

static void check(bool ok, const char* what, double got, double want) {
  const double rel = (want != 0.0) ? std::fabs(got - want) / std::fabs(want)
                                   : std::fabs(got - want);
  std::printf("  %s  %-54s %13.6g vs %-12.6g (%.2e)\n", ok ? "PASS" : "FAIL",
              what, got, want, rel);
  if (!ok) ++failures;
}
static void note(const char* s) { std::printf("        %s\n", s); }

//------------------------------------------------------------------------------
// Everything one sweep of the field can tell us.
//------------------------------------------------------------------------------
struct Survey {
  double umax = 0;
  long fluid = 0, interf = 0, gas = 0;
  long open = 0;              // Fluid cells touching Gas -- must be zero
  double front = 0;           // furthest liquid in x
};

static Survey survey(host::FreeSurface& fs, int nx, int ny, int nz) {
  std::vector<Real> ux, uy, uz;
  fs.field_to_host(fs.ux_device(), ux);
  fs.field_to_host(fs.uy_device(), uy);
  fs.field_to_host(fs.uz_device(), uz);
  std::vector<std::uint8_t> f;
  fs.flags_to_host(f);

  Survey s;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const long n = node_id(x, y, z, nx, ny);
        const std::uint8_t c = f[std::size_t(n)];
        if (c == FsFluid) ++s.fluid;
        else if (c == FsInterface) ++s.interf;
        else if (c == FsGas) ++s.gas;
        if (c == FsFluid || c == FsInterface) {
          const double a = double(ux[std::size_t(n)]);
          const double b = double(uy[std::size_t(n)]);
          const double d = double(uz[std::size_t(n)]);
          s.umax = std::fmax(s.umax, std::sqrt(a * a + b * b + d * d));
          s.front = std::fmax(s.front, double(x));
        }
        // THE INVARIANT. A full cell beside a void has no boundary condition to
        // apply: the reconstruction is what stands in for the gas, and it only
        // runs on Interface cells. One open face and the pressure at that link
        // is whatever streamed in, which is nothing.
        if (c == FsFluid)
          for (int i = 1; i < 27; ++i)
            if (f[std::size_t(neighbour<D3Q27>(x, y, z, i, nx, ny, nz))] == FsGas) ++s.open;
      }
  return s;
}

static void box_walls(std::vector<std::uint8_t>& fl, int nx, int ny, int nz) {
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        fl[std::size_t(node_id(x, y, z, nx, ny))] =
            (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) ? FsSolid : FsGas;
}

int main() {
  std::printf("Free surface, host build, Real = %s\n",
              sizeof(Real) == 4 ? "float" : "double");
  const bool fp64 = (sizeof(Real) == 8);

  //===========================================================================
  std::printf("\n1-3. A HYDROSTATIC POOL\n\n");
  //===========================================================================
  //
  // Half a closed box filled with liquid, seeded at its own hydrostatic
  // pressure, under gravity. Nothing should happen -- which makes it the case
  // where anything that does happen is the scheme.
  //
  // THE SEED'S DENSITY MATTERS AND IS NOT COSMETIC. Left at 1 the liquid starts
  // with no pressure gradient, so the column has to build its own hydrostatic
  // profile, and it does that by RINGING: an acoustic transient crossing the
  // domain in a few hundred steps and taking far longer to damp. Seeded with
  // rho = rho_G + g (y_surface - y) / cs^2, what is left is a small residual
  // that decays monotonically -- 6.6e-6 to 9.3e-7 over 5000 steps -- and a
  // decaying residual is a very different diagnostic from a standing one.
  {
    const int nx = 48, ny = 48, nz = 4;
    const double nu = 0.01, g = 1e-5, surface = 24.0, cs2 = 1.0 / 3.0;

    host::FreeSurface fs(nx, ny, nz, Real(nu));
    fs.set_gravity(Real(0), Real(-g));
    fs.rho_G = Real(1);

    std::vector<std::uint8_t> fl(std::size_t(nx) * ny * nz, std::uint8_t(FsGas));
    box_walls(fl, nx, ny, nz);
    fs.set_geometry(fl);
    fs.initialise_with([&](int, int y, int) {
      FsSeed s;
      const double f = surface + 0.5 - double(y);
      s.fill = Real(f > 1.0 ? 1.0 : (f < 0.0 ? 0.0 : f));
      const double h = surface - double(y);
      s.rho = Real(1.0 + (h > 0 ? g * h / cs2 : 0.0));
      return s;
    });

    const double m0 = fs.total_mass();
    const Survey s0 = survey(fs, nx, ny, nz);
    for (int t = 0; t < 2000; ++t) fs.step();
    const Survey s1 = survey(fs, nx, ny, nz);
    const double m1 = fs.total_mass();
    for (int t = 0; t < 2000; ++t) fs.step();
    const Survey s2 = survey(fs, nx, ny, nz);
    const double m2 = fs.total_mass();

    // 1. MASS. Two windows rather than one, so a LEAK -- which grows -- is
    // separated from round-off, which need not.
    const double d1 = (m1 - m0) / m0, d2 = (m2 - m1) / m0;
    check(std::fabs(d1) < (fp64 ? 1e-6 : 2e-4), "mass conserved over 2000 steps",
          d1, 0.0);
    check(std::fabs(d2) < (fp64 ? 1e-6 : 2e-4), "  ... and over the next 2000",
          d2, 0.0);
    std::printf("        (total mass %.6f -> %.6f -> %.6f)\n", m0, m1, m2);

    // 2. HYDROSTATIC EQUILIBRIUM. The pool must go quiet, and go quiet
    // MONOTONICALLY: a residual that stops falling is a balance error, not a
    // transient.
    check(s2.umax < s1.umax, "the residual velocity keeps decaying", s2.umax, 0.0);
    check(s2.umax < 1e-5, "and the pool is quiet after 4000 steps", s2.umax, 0.0);
    std::printf("        (max |u|: %.2e -> %.2e -> %.2e)\n", s0.umax, s1.umax, s2.umax);

    // 3. THE INVARIANT.
    check(s0.open == 0 && s1.open == 0 && s2.open == 0,
          "no Fluid cell ever touches a Gas cell", double(s2.open), 0.0);
    check(s2.fluid == s0.fluid && s2.interf == s0.interf,
          "and a quiescent pool converts nothing", double(s2.interf),
          double(s0.interf));
    std::printf("        (%ld fluid, %ld interface, %ld gas, unchanged)\n",
                s2.fluid, s2.interf, s2.gas);
  }

  //===========================================================================
  std::printf("\n4. A COLLAPSING COLUMN\n\n");
  //===========================================================================
  //
  // The same machinery with the topology actually changing: a column of liquid
  // released against a wall, spreading along the floor. Several hundred cells
  // convert in each direction, which is what the promote/settle pair exists for
  // and the only way to exercise it.
  //
  // Measured in FP32 at g = 1e-5, sampled every 250 steps:
  //
  //     step    mass drift    max|u|     fluid  interface   front
  //     0        0            0           1716    204        12
  //     1000    -4.8e-05      1.04e-02    1692    244        17
  //     1500    -1.7e-04      1.44e-02    1680    284        22
  //     2000    -2.1e-04      1.71e-02    1644    332        29
  //     2500    -1.3e-04      1.82e-02    1636    404        36
  //     3000    +1.1e-06      2.19e-02    1604    452        43
  //
  // Two hundred and fifty cells converted, the invariant held at ZERO open
  // faces at every sample, and the ledger stayed inside two parts in ten
  // thousand throughout.
  //
  // NOTE THAT THE DRIFT IS NOT MONOTONIC -- it falls to -2.1e-4 and comes back
  // to +1e-6 by the end. So the endpoint is the WRONG number to assert on: it
  // happens to return near zero, and a test written against it would pass while
  // a genuine leak of 2e-4 was building and draining. The check below tracks the
  // WORST drift over the run instead.
  //
  // THE WINDOW ENDS AT 3000 STEPS BECAUSE THAT IS WHERE THE FRONT REACHES THE
  // FAR WALL, and what happens after is a different problem. Run to 5000 and the
  // column slams into it: the drift jumps to 3.7e-3 and max|u| to 0.24, i.e.
  // Mach 0.41. That is a violent impact against a hard wall with no surface
  // tension and a one-cell interface, and it is not what this case is for --
  // measuring it here would be measuring the wall, and the numbers would move
  // with the domain width rather than with anything about the scheme.
  //
  // The drift that IS here is not round-off. It is the documented
  // no-subgrid-interface limitation arriving in the accounts: a detached cell
  // has, by construction, no interface neighbour to hand its mass to, so
  // deleting it loses that mass. That is the price of not having a speck of
  // liquid free-fall at g for the rest of the run.
  {
    const int nx = 48, ny = 48, nz = 4;
    const double nu = 0.01, g = 1e-5, cs2 = 1.0 / 3.0;
    const double col_w = 12.0, col_h = 40.0;

    host::FreeSurface fs(nx, ny, nz, Real(nu));
    fs.set_gravity(Real(0), Real(-g));

    std::vector<std::uint8_t> fl(std::size_t(nx) * ny * nz, std::uint8_t(FsGas));
    box_walls(fl, nx, ny, nz);
    fs.set_geometry(fl);
    fs.initialise_with([&](int x, int y, int) {
      FsSeed s;
      const double fx = col_w + 0.5 - double(x);
      const double fy = col_h + 0.5 - double(y);
      s.fill = Real((fx > 1.0 ? 1.0 : (fx < 0.0 ? 0.0 : fx)) *
                    (fy > 1.0 ? 1.0 : (fy < 0.0 ? 0.0 : fy)));
      const double h = col_h - double(y);
      s.rho = Real(1.0 + (h > 0 && double(x) < col_w ? g * h / cs2 : 0.0));
      return s;
    });

    const double m0 = fs.total_mass();
    const Survey s0 = survey(fs, nx, ny, nz);
    long worst_open = 0;
    double worst_drift = 0;
    Survey s1;
    for (int k = 0; k < 12; ++k) {           // 3000 steps; see the banner
      for (int t = 0; t < 250; ++t) fs.step();
      s1 = survey(fs, nx, ny, nz);
      worst_open = std::max(worst_open, s1.open);
      const double d = (fs.total_mass() - m0) / m0;
      if (std::fabs(d) > std::fabs(worst_drift)) worst_drift = d;
    }

    check(worst_open == 0, "the invariant holds through every conversion",
          double(worst_open), 0.0);
    check(s1.interf > s0.interf + 100,
          "and the column really did convert (interface cells grew)",
          double(s1.interf), double(s0.interf));
    check(std::fabs(worst_drift) < 1e-3,
          "the WORST mass drift over the run stays under 0.1%", worst_drift, 0.0);
    check(s1.front > s0.front + 20, "the front advanced along the floor",
          s1.front, s0.front);
    check(s1.front < double(nx - 4),
          "  ... and stopped short of the far wall, as the window intends",
          s1.front, double(nx - 4));
    check(s1.umax < 0.05, "and the flow stayed well subsonic", s1.umax, 0.0);
    std::printf("        (fluid %ld -> %ld, interface %ld -> %ld, front %.0f -> %.0f)\n",
                s0.fluid, s1.fluid, s0.interf, s1.interf, s0.front, s1.front);
    note("the drift is detached cells losing their mass, not a leak -- see the banner");
  }

  std::printf("\n[freesurface] %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
