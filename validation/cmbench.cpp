//==============================================================================
//  OPERATOR THROUGHPUT, ON THE HOST. There was no throughput bench for src/ at
//  all -- the MLUPS figures in the notes came from ad-hoc runs -- and this
//  exists because one question needed answering cheaply.
//
//  THE QUESTION. GPU/ runs central moments at 99% of BGK on a T4 (933 against
//  943 MLUPS). src/ on the same card reportedly did not finish 50 steps at 64^3
//  in seventeen minutes, where BGK took 0.03 s. Two possibilities: the
//  arithmetic is expensive, in which case the host sees it too; or something
//  happens only on the device, in which case the host cannot see it at all.
//  That is a fork worth resolving before anyone books GPU time.
//
//  MEASURED, 48^3, FP64, four host threads, MEAN OF FIVE RUNS (spread 3.4 to
//  5.5%, which is why five and not one -- see the caution below):
//
//      BGK                       43.5 MLUPS
//      TRT                       37.4           1.16x BGK
//      MomentCollision central   20.0           2.18x BGK
//      MomentCollision raw MRT   20.3           2.14x BGK
//
//  So the arithmetic is FINE. 2.18x is an unremarkable price for a 27-moment
//  transform, and it is four orders of magnitude away from the device figure.
//  The collapse is device-specific and the host will not reproduce it.
//
//  Note also that central moments and raw MRT cost the SAME here, 2.18 against
//  2.14. They share the transform and differ only in the relaxation schedule,
//  so that is the expected answer and a useful sanity check on the bench.
//
//  A CORRECTION THIS FILE SHOULD CARRY, because it is the mistake this tree
//  keeps making. The first version of these numbers was a SINGLE run each and
//  reported 2.69x and 4.30x. Repeated five times on a quiet machine they are
//  2.18x and 2.14x -- so the MRT figure was wrong by a factor of two, and it was
//  wrong in the direction that would have made raw MRT look like a second
//  problem needing its own explanation. One sample of a wall-clock timing on a
//  four-thread laptop is not a measurement.
//
//  A CAUTION ON THAT DEVICE FIGURE, which this file's existence should not
//  lend credibility to. "Did not finish in seventeen minutes" is a timeout, not
//  a measurement -- this tree's own discipline says so, and it has been bitten
//  before by a metric that moved 2x with the backend alone and by silent
//  throttling on the shared machine those numbers came from. The first thing to
//  do with a GPU is replace it with a NUMBER. If the true ratio is ~50x then the
//  prime suspect is local-memory traffic from the dynamically indexed
//  moment array -- a mechanism this project has already measured at 47x in
//  GPU/'s colour gradient (a 216-byte stack frame, 20.2 against 950 MLUPS,
//  recovered 12.5x by deriving the equilibrium central moments in closed form).
//  If it really is four orders, something pathological is happening instead and
//  the local-memory story is the wrong tree.
//
//  IT IS MEANT TO BE BUILT FOR CUDA TOO, and that is the point of it now: it
//  bounds the experiment. 48^3 x 60 steps is 6.6e6 cell updates, about 1.3 ms at
//  an H200's BGK rate, so even a four-orders-slower central moment finishes
//  inside a minute. A timeout cannot happen, which is exactly what the original
//  seventeen-minute observation could not promise.
//
//      cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
//            -DLBM_GPU_BACKEND=cuda -DLBM_GPU_ARCH=HOPPER90 -DLBM_PRECISION=float
//      cmake --build build-gpu -j8 --target cmbench && ./build-gpu/validation/cmbench
//
//  WHAT THIS DOES NOT MEASURE: accuracy. It is a stopwatch.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/TRT.hpp"
#include "solver/FluidSolver.hpp"
#include "memory/EsotericPull.hpp"

#include <chrono>
#include <cstdio>

using namespace lbm;
using L = D3Q27;

// THE COLLISION ARRIVES CONFIGURED, BY VALUE, and that is an nvcc constraint
// rather than a style choice. This function contains a KOKKOS_LAMBDA, i.e. an
// extended __host__ __device__ lambda, and nvcc forbids a FUNCTION-LOCAL type
// as a template argument of any function that does -- so taking a `Setup`
// callable deduced from a lambda in main() compiles on the host and fails under
// nvcc. That is restriction six in this tree's own list (it cost `galilean` a
// build once, fixed the same way: keep the template arguments at namespace
// scope). Configuring in main and passing the object sidesteps it entirely, and
// this file exists to be built for CUDA.
template <class Coll>
static double run(const char* name, Index N, std::size_t steps, Coll coll) {
  Domain d(N, N, N, true, true, true);
  FluidSolver<L, EsotericPull<L>, Coll> s(d, coll);
  const Real u0 = Real(0.04);
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    const double kx = 2.0 * 3.14159265358979 / double(N);
    return FlowState{Real(1),
                     Real(u0 * Kokkos::sin(kx * double(x)) * Kokkos::cos(kx * double(y))),
                     Real(-u0 * Kokkos::cos(kx * double(x)) * Kokkos::sin(kx * double(y))),
                     Real(0)};
  });
  s.step();                                    // warm up, exclude first-touch
  Kokkos::fence();
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t t = 0; t < steps; ++t) s.step();
  Kokkos::fence();
  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  const double ml = double(N) * N * N * double(steps) / sec / 1e6;
  std::printf("  %-28s %8.1f MLUPS   %8.3f s\n", name, ml, sec);
  return ml;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const Index N = 48;
    const std::size_t steps = 60;
    const Real nu = Real(0.01);
    std::printf("\nsrc/ host throughput, D3Q27 EsotericPull, %d^3, %zu steps, %s\n\n",
                int(N), steps, precision_name());
    using B  = BGK<L, SecondOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;
    using T2 = TRT<L, SecondOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;
    using CM = MomentCollision<L, NoForcing, ShiftedPopulations, true>;
    using MR = MomentCollision<L, NoForcing, ShiftedPopulations, false>;
    B  cb;  cb.omega = B::omega_from_viscosity(nu);
    T2 ct;  ct.omega_p = T2::omega_from_viscosity(nu);
            ct.omega_m = T2::omega_minus_for(ct.omega_p, T2::magic_3_16);
    CM cc;  cc.omega = CM::omega_from_viscosity(nu);
    MR cr;  cr.omega = MR::omega_from_viscosity(nu);

    const double b  = run<B> ("BGK",                     N, steps, cb);
                      run<T2>("TRT",                     N, steps, ct);
    const double cm = run<CM>("MomentCollision central", N, steps, cc);
    const double mr = run<MR>("MomentCollision raw MRT", N, steps, cr);
    std::printf("\n  CM / BGK = %.2fx slower      MRT / BGK = %.2fx slower\n\n",
                b / cm, b / mr);
  }
  Kokkos::finalize();
  return 0;
}
