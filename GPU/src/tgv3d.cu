//==============================================================================
//  3D Taylor-Green vortex, native CUDA.
//
//  Deliberately the same case, the same initial condition and the same output
//  format as validation/tgv3d.cpp in the parent Kokkos implementation, so the
//  two can be diffed directly. The parent's committed reference is
//
//      results/E_tgv3d/tgv3d_re1600_d3q27_cm.dat
//
//  at D = 64, Re = 1600, u0 = 0.02, tau = 0.502400. This code will NOT
//  reproduce it bit for bit -- it is FP32 by default where the parent is FP64,
//  and the reduction order differs -- but the energy and enstrophy curves
//  should agree to a few parts in a thousand. Anything worse is a port bug, and
//  that is the point of matching the format.
//
//    usage: tgv3d [-d N] [-re R] [-tmax T] [-u0 U] [-op bgk|cm]
//==============================================================================
#include "lbm/backend.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
// A STRUCT, not a lambda.
//
// nvcc forbids an extended __host__ __device__ lambda inside a generic lambda,
// and restricts where else one may appear. A plain functor sidesteps the whole
// family of restrictions, which cost the parent implementation six separate
// fixes when it was first taken to a GPU.
//------------------------------------------------------------------------------
struct TaylorGreenInit {
  Real u0, k;
  LBM_HD Macro operator()(int x, int y, int z) const {
    const Real X = k * Real(x), Y = k * Real(y), Z = k * Real(z);
    Macro m;
    m.rho = Real(1);
    m.ux  =  u0 * cosf(X) * sinf(Y) * sinf(Z);
    m.uy  = -Real(0.5) * u0 * sinf(X) * cosf(Y) * sinf(Z);
    m.uz  = -Real(0.5) * u0 * sinf(X) * sinf(Y) * cosf(Z);
    return m;
  }
};

//------------------------------------------------------------------------------
// Volume-integrated kinetic energy and enstrophy, on the host.
//
// Second-order central differences with periodic wrap, matching the parent's
// diagnostics() exactly so the numbers are comparable. This runs on probe steps
// only; doing it on the device would save nothing measurable and would add a
// reduction to maintain.
//------------------------------------------------------------------------------
struct Diag { double energy, enstrophy; bool finite; };

static Diag diagnostics(const std::vector<Real>& ux, const std::vector<Real>& uy,
                        const std::vector<Real>& uz, int n) {
  auto id = [&](int x, int y, int z) {
    return long(((x + n) % n)) + long(n) * (long((y + n) % n) + long(n) * long((z + n) % n));
  };
  double e = 0, w = 0;
  bool ok = true;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const long m = id(x, y, z);
        const double a = ux[m], b = uy[m], c = uz[m];
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) { ok = false; continue; }
        e += 0.5 * (a * a + b * b + c * c);
        const double wx = 0.5 * (double(uz[id(x, y + 1, z)]) - double(uz[id(x, y - 1, z)]))
                        - 0.5 * (double(uy[id(x, y, z + 1)]) - double(uy[id(x, y, z - 1)]));
        const double wy = 0.5 * (double(ux[id(x, y, z + 1)]) - double(ux[id(x, y, z - 1)]))
                        - 0.5 * (double(uz[id(x + 1, y, z)]) - double(uz[id(x - 1, y, z)]));
        const double wz = 0.5 * (double(uy[id(x + 1, y, z)]) - double(uy[id(x - 1, y, z)]))
                        - 0.5 * (double(ux[id(x, y + 1, z)]) - double(ux[id(x, y - 1, z)]));
        w += 0.5 * (wx * wx + wy * wy + wz * wz);
      }
  return {e, w, ok};
}

int main(int argc, char** argv) {
  int D = 64;
  double Re = 1600.0, tmax = 10.0;
  Real u0 = Real(0.02);
  std::string op = "cm";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-d"    && i + 1 < argc) D    = std::atoi(argv[++i]);
    if (a == "-re"   && i + 1 < argc) Re   = std::atof(argv[++i]);
    if (a == "-tmax" && i + 1 < argc) tmax = std::atof(argv[++i]);
    if (a == "-u0"   && i + 1 < argc) u0   = Real(std::atof(argv[++i]));
    if (a == "-op"   && i + 1 < argc) op   = argv[++i];
  }

  const Real nu = Real(double(u0) * double(D) / Re);
  const std::size_t T = std::size_t(tmax * double(D) / double(u0));
  const Op which = (op == "bgk") ? Op::BGK : Op::CentralMoments;

  std::printf("3D Taylor-Green   %s   D3Q27   operator %s   %s\n",
              backend::on_device ? "CUDA native" : "HOST reference",
              which == Op::BGK ? "bgk" : "cm",
              sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  D = %d   Re = %.0f   u0 = %.3f   Ma = %.4f   nu = %.6e   tau = %.6f\n",
              D, Re, double(u0), double(u0) * std::sqrt(3.0),
              double(nu), 3.0 * double(nu) + 0.5);
  std::printf("  t* = t u0 / D up to %.1f   (%zu steps, %d^3 = %ld nodes)\n\n",
              tmax, T, D, long(D) * D * D);

  backend::Fluid s(D, D, D, which, nu);
  s.initialise_with(TaylorGreenInit{u0, Real(2.0 * M_PI) / Real(D)});

  std::vector<Real> rho, ux, uy, uz;
  s.macroscopic_to_host(rho, ux, uy, uz);
  const Diag d0 = diagnostics(ux, uy, uz, D);

  std::printf("  %8s %14s %14s\n", "t*", "E/E0", "Psi/Psi0");
  const std::size_t probe = T / 20 ? T / 20 : 1;
  const auto wall0 = std::chrono::steady_clock::now();

  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      s.macroscopic_to_host(rho, ux, uy, uz);
      const Diag dd = diagnostics(ux, uy, uz, D);
      const double ts = tmax * double(t) / double(T);
      if (!dd.finite) { std::printf("  DIVERGED at t* = %.3f\n", ts); return 1; }
      std::printf("  %8.3f %14.6f %14.6f\n", ts, dd.energy / d0.energy,
                  dd.enstrophy / d0.enstrophy);
      std::fflush(stdout);
    }
    if (t < T) s.step();
  }

  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall0).count();
  std::printf("\n  %zu steps in %.2f s  ->  %.1f MLUPS\n",
              T, sec, double(long(D) * D * D) * double(T) / sec / 1e6);
  return 0;
}
