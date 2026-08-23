//==============================================================================
//  Benchmark harness.
//
//  Sweeps lattice x streaming scheme x storage on a periodic box and reports
//  throughput. LBM is bandwidth-bound on real hardware, so the figure that
//  actually predicts performance is bytes moved per node per step:
//  two-lattice moves 2*Q*sizeof(Real), the in-place schemes move Q*sizeof(Real).
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/TRT.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "memory/TwoLattice.hpp"
#include "solver/FluidSolver.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>

using namespace lbm;

namespace {
int status = 0;

template <class Coll, class L, template <class> class Streaming, class Store, class Setup>
void bench(Index n, int steps, Setup setup) {
  using S = Streaming<L>;
  Coll coll;
  setup(coll, Coll::omega_from_viscosity(Real(0.01)));

  Domain d(n, n, n, true, true, true);
  FluidSolver<L, S, Coll> s(d, coll);
  s.initialize(Real(1), Real(0.02), Real(0), Real(0));

  const double mass0 = double(s.total_mass());
  s.step();
  Kokkos::fence();                                     // warm-up
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < steps; ++t) s.step();
  Kokkos::fence();
  const auto t1 = std::chrono::steady_clock::now();

  const double sec   = std::chrono::duration<double>(t1 - t0).count();
  const double lups  = double(d.n_interior()) * steps;
  const double bytes = lups * S::bytes_per_node();
  const double drift = std::abs(double(s.total_mass()) - mass0) / mass0;

  std::printf("%-7s %-16s %-14s %-9s %-10.2f %-10.2f %-11.3e\n",
              L::name, Coll::name, S::name, Store::name,
              lups / sec / 1e6, bytes / sec / 1e9, drift);

  const double tol = sizeof(Real) == 4 ? 1e-4 : 1e-11;
  if (!(drift < tol)) { std::printf("  MASS NOT CONSERVED\n"); status = 1; }
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index n = 64;
    int steps = 50;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-n" && i + 1 < argc)     n = std::atoi(argv[++i]);
      if (a == "-steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
    }

    std::printf("backend %s   precision %s   grid %dx%dx%d   steps %d\n\n",
                ExecSpace::name(), precision_name(), int(n), int(n), int(n), steps);
    std::printf("%-7s %-16s %-14s %-9s %-10s %-10s %-11s\n",
                "lattice", "collision", "streaming", "storage", "MLUPS", "GB/s", "mass drift");
    std::printf("%s\n", std::string(84, '-').c_str());

    auto plain = [](auto& c, Real w) { c.omega = w; };
    auto trt   = [](auto& c, Real w) {
      using T = std::decay_t<decltype(c)>;
      c.omega_p = w; c.omega_m = T::omega_minus_for(w, T::magic_3_16);
    };
    using SS = ShiftedPopulations;
    using RR = RawPopulations;

    // streaming / storage comparison, fixed operator
    bench<BGK<D3Q27, SecondOrderEquilibrium<D3Q27>, NoForcing, RR>, D3Q27, TwoLattice,   RR>(n, steps, plain);
    bench<BGK<D3Q27, SecondOrderEquilibrium<D3Q27>, NoForcing, RR>, D3Q27, EsotericPull, RR>(n, steps, plain);
    bench<BGK<D3Q27, SecondOrderEquilibrium<D3Q27>, NoForcing, SS>, D3Q27, EsotericPull, SS>(n, steps, plain);
    std::printf("\n");
    // collision operator comparison, fixed streaming / storage
    bench<BGK<D3Q27, SecondOrderEquilibrium<D3Q27>, NoForcing, SS>, D3Q27, EsotericPull, SS>(n, steps, plain);
    bench<TRT<D3Q27, SecondOrderEquilibrium<D3Q27>, NoForcing, SS>, D3Q27, EsotericPull, SS>(n, steps, trt);
    bench<MomentCollision<D3Q27, NoForcing, SS, false>,             D3Q27, EsotericPull, SS>(n, steps, plain);
    bench<MomentCollision<D3Q27, NoForcing, SS, true>,              D3Q27, EsotericPull, SS>(n, steps, plain);
  }
  Kokkos::finalize();
  return status;
}
