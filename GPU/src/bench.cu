//==============================================================================
//  Throughput benchmark.
//
//  Reports MLUPS and the implied memory traffic for BGK and central moments at
//  a sweep of problem sizes, with a mass-drift column so a fast wrong answer is
//  visibly a wrong answer.
//
//  THREE THINGS THIS GETS RIGHT ON PURPOSE, because the parent implementation's
//  first GPU benchmark got them wrong or made them hard to read:
//
//  1. cudaDeviceSynchronize() either side of the timed loop. Kernel launches are
//     asynchronous; without the fence you time the launch queue, not the work.
//  2. A warm-up step outside the timing. The first launch pays context setup and
//     JIT, which at small N is a large fraction of the measurement.
//  3. The traffic figure counts what is actually moved: 27 reads + 27 writes per
//     node per step, the same for Esoteric Pull and for a two-lattice scheme.
//     Esoteric Pull's win is FOOTPRINT -- one lattice instead of two -- not
//     traffic, and reporting it as half the bytes flatters it.
//
//    usage: bench [-n N] [-steps S]   (repeats for a small sweep if -n absent)
//==============================================================================
#include "lbm/solver.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

struct UniformInit {
  LBM_HD Macro operator()(int, int, int) const {
    return Macro{Real(1), Real(0.02), Real(0), Real(0)};
  }
};

static double total_mass(Solver& s) {
  std::vector<Real> rho, ux, uy, uz;
  s.macroscopic_to_host(rho, ux, uy, uz);
  double m = 0;
  for (Real v : rho) m += double(v);
  return m;
}

static void run(int n, int steps, Op op, const char* name) {
  Solver s(n, n, n, op, Real(0.01));
  s.initialise_with(UniformInit{});

  const double m0 = total_mass(s);

  s.step();                                   // warm-up, outside the timing
  LBM_CUDA_CHECK(cudaDeviceSynchronize());

  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < steps; ++t) s.step();
  LBM_CUDA_CHECK(cudaDeviceSynchronize());
  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();

  const double lups  = double(s.nodes()) * double(steps);
  const double bytes = lups * 27.0 * 2.0 * double(sizeof(Real));  // read + write
  const double drift = std::abs(total_mass(s) - m0) / m0;

  std::printf("  %-16s %5d^3 %12.2f %12.2f %14.3e\n",
              name, n, lups / sec / 1e6, bytes / sec / 1e9, drift);
  if (!(drift < (sizeof(Real) == 4 ? 1e-4 : 1e-11)))
    std::printf("    ^^ MASS NOT CONSERVED -- treat the rate above as meaningless\n");
}

int main(int argc, char** argv) {
  int n = 0, steps = 100;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-n"     && i + 1 < argc) n     = std::atoi(argv[++i]);
    if (a == "-steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
  }

  cudaDeviceProp p{};
  LBM_CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("device %s   sm_%d%d   %.0f GB/s peak   %s\n\n",
              p.name, p.major, p.minor,
              2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6,
              sizeof(Real) == 4 ? "FP32" : "FP64");

  std::printf("  %-16s %7s %12s %12s %14s\n",
              "operator", "grid", "MLUPS", "GB/s", "mass drift");
  std::printf("  %s\n", std::string(66, '-').c_str());

  const std::vector<int> sizes = n ? std::vector<int>{n}
                                   : std::vector<int>{64, 96, 128, 160};
  for (int s : sizes) run(s, steps, Op::BGK, "BGK");
  std::printf("\n");
  for (int s : sizes) run(s, steps, Op::CentralMoments, "central moments");

  std::printf("\n  Traffic counts 27 reads + 27 writes per node per step. Esoteric\n"
              "  Pull's advantage is footprint (one lattice), not bandwidth.\n");
  return 0;
}
