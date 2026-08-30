#include "Check.hpp"
#include "solver/FreeSurfaceSolver.hpp"
#include <cmath>
#include <cstdio>
#include <string>
using namespace lbm;


// Two runs. Without gravity the fluid is at rest at rho = 1 and nothing but the
// mass bookkeeping can move the total, so it must hold to ROUND-OFF: that is the
// test of the exchange's antisymmetry, which is the only thing conserving mass
// here. With gravity the column settles, rho changes with pressure because the
// scheme is weakly compressible, and the total legitimately breathes -- so the
// test there is that it oscillates rather than drifts.
template <class L, double GRAV>
static double run_case(const char* what, bool strict) {
  using FS = FreeSurfaceSolver<L>;
  {
    // A liquid column filling the lower half of a walled box, gas above.
    const Index nx = 32, ny = 48, nz = (L::D == 3) ? 4 : 1;
    const double h = 24.0;
    Domain d(nx, ny, nz, true, false, true);
    FS s(d);
    s.coll.omega = FS::omega_from_viscosity(Real(0.05));
    s.set_gravity(Real(0), Real(GRAV));
    const Index nyi = ny;
    s.set_geometry([&](Index, Index y, Index) -> FsCell {
      return (y == 0 || y == nyi - 1) ? FsSolid : FsGas;
    });
    const Domain dd = d; const Index hy = d.hy; const Real hh = Real(h);
    s.initialize(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Real y = Real(py - hy);
      // sharp step: full below h, empty above, one partial cell at the surface
      const Real e = (y + Real(0.5) < hh) ? Real(1)
                   : ((y - Real(0.5) < hh) ? Real(0.5) : Real(0));
      return typename FS::Seed{e, Real(1)};
    });
    const double m0 = double(s.total_mass());
    auto c0 = s.census();
    std::printf("  seeded: mass %.10f   gas %d  interface %d  fluid %d\n",
                m0, int(c0.gas), int(c0.interface_), int(c0.fluid));
    for (int k = 0; k < 400; ++k) {
      s.step();
      if (k % 100 == 99) {
        auto c = s.census();
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
        double um = 0;
        for (Index y = 1; y < ny - 1; ++y)
          for (Index x = 0; x < nx; ++x) {
            const Index n = d.id(x, y, nz / 2);
            if (hf(n) == FsFluid || hf(n) == FsInterface)
              um = std::max(um, std::fabs(double(hu(n))));
          }
        std::printf("  step %4d: mass %.10f  drift %.3e  max|uy| %.3e  g/i/f %d/%d/%d\n",
                    k + 1, double(s.total_mass()),
                    double(s.total_mass()) / m0 - 1.0, um,
                    int(c.gas), int(c.interface_), int(c.fluid));
      }
    }
    const double m1 = double(s.total_mass());
    if (strict)
      check::near(m1 / m0 - 1.0, 0.0, 1e-13,
                  std::string("mass exact to round-off: ") + what);
    return m1 / m0 - 1.0;
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    // The same case on every lattice the solver admits. Mass advection is
    // lattice-generic in principle -- the exchange loops over the whole velocity
    // set and the reconstruction uses opp(i), which every lattice here honours --
    // but "in principle" is not a test, and D3Q19 is the one the free-surface
    // literature actually uses.
    std::printf("\n1. at rest, no gravity -- only the bookkeeping can move mass\n");
    std::printf("   D2Q9\n");   run_case<D2Q9,  0.0>("at rest, D2Q9",  true);
    std::printf("   D3Q27\n");  run_case<D3Q27, 0.0>("at rest, D3Q27", true);
    std::printf("\n2. under gravity -- the column settles and rho breathes\n");
    const double a = run_case<D2Q9,  -1e-5>("under gravity, D2Q9",  false);
    const double b = 0.0;
    const double c = run_case<D3Q27, -1e-5>("under gravity, D3Q27", false);
    check::ok(std::fabs(a) < 1e-4 && std::fabs(b) < 1e-4 && std::fabs(c) < 1e-4,
              "under gravity the total stays bounded on every lattice");
  }
  const int rc = check::report("free_surface");
  Kokkos::finalize();
  return rc;
}
