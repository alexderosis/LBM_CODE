// Compile check for the skeleton quoted in CLAUDE.md. Not a physics test:
// it exists so that the snippet a newcomer copies is known to build.
#include "collision/BGK.hpp"       // defines Macro, which FluidSolver.hpp uses
#include "solver/FluidSolver.hpp"
#include "memory/EsotericPull.hpp"
#include <cstdio>
using namespace lbm;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Lattice = D2Q9;
    using Coll    = BGK<Lattice>;
    const Real nu = Real(0.05);
    const Index nx = 16, ny = 16, nz = 1;
    const std::size_t T = 10;

    Domain d(nx, ny, nz, /*periodic x,y,z=*/true, true, true);
    Coll coll;  coll.omega = Coll::omega_from_viscosity(nu);
    FluidSolver<Lattice, EsotericPull<Lattice>, Coll> s(d, coll);
    s.initialize_field(KOKKOS_LAMBDA(Index n) {
      (void)n; return FlowState{Real(1), Real(0), Real(0), Real(0)};
    });
    for (std::size_t t = 0; t < T; ++t) s.step();
    std::printf("CLAUDE.md skeleton: built and ran %zu steps\n", T);
  }
  Kokkos::finalize();
  return 0;
}
