//==============================================================================
//  Taylor-Green vortex, THE BENCHMARK CONFIGURATION -- CUDA.
//
//  The companion to validation/tgv3d_bench.cpp, and the same distinction from
//  src/tgv3d.cu: that file runs the De Rosis & Coreixas Taylor-Green-TYPE flow,
//  which has no published reference; this one runs the canonical DNS benchmark.
//  Two differences, both easy to miss and fatal to a comparison:
//
//    * THE INITIAL CONDITION HAS w = 0 --
//          u =  V0 sin(x/L) cos(y/L) cos(z/L)
//          v = -V0 cos(x/L) sin(y/L) cos(z/L)
//          w =  0
//      against three non-zero components with factors of 1/2 in the other file.
//
//    * Re IS DEFINED ON L, NOT ON THE BOX. The domain is [0, 2 pi L)^3, so
//      L = D/(2 pi) and Re = V0 L/nu = V0 D/(2 pi nu). The other file uses
//      Re = u0 D/nu, larger by 2 pi: its "Re = 1600" is Re = 255 here.
//
//  Reports the benchmark's own quantities: Ek* = <|u|^2/2>/V0^2, which is
//  exactly 1/8 at t* = 0, and the dissipation rate from BOTH the energy decay
//  and the enstrophy. Their difference is the resolution diagnostic -- the
//  energy-based figure includes numerical dissipation, the enstrophy-based one
//  counts only resolved vorticity. Measured at Re = 100 on 48^3 they agree to
//  0.5%; at Re = 1600 on the same grid the gap exceeds 200%, which is what
//  under-resolution looks like when it is measured rather than assumed.
//
//    usage: tgv3d_bench [-d N] [-re R] [-tmax T] [-u0 U] [-op bgk|cm]
//==============================================================================
#include "lbm/solver.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
// A STRUCT, not a lambda -- see src/tgv3d.cu for why nvcc requires it.
//
// cos/sin rather than cosf/sinf: this runs once per cell at start-up, so the
// double-precision promotion in an FP32 build costs nothing measurable and the
// same source is correct when built with -DLBM_DOUBLE.
//------------------------------------------------------------------------------
struct TaylorGreenBenchInit {
  Real u0, k, dr;
  LBM_HD Macro operator()(int x, int y, int z) const {
    const Real X = k * Real(x), Y = k * Real(y), Z = k * Real(z);
    Macro m;
    // Analytic pressure through p = rho cs^2. A uniform start instead launches
    // an acoustic transient across the first eddy turnover.
    m.rho = Real(1) + dr * (cos(Real(2) * X) + cos(Real(2) * Y))
                         * (cos(Real(2) * Z) + Real(2));
    m.ux  =  u0 * sin(X) * cos(Y) * cos(Z);
    m.uy  = -u0 * cos(X) * sin(Y) * cos(Z);
    m.uz  =  Real(0);
    return m;
  }
};

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
  double Re = 1600.0, tmax = 20.0;
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

  // THE POINT OF THIS FILE: Re is on L = D/(2 pi), not on D.
  const double L  = double(D) / (2.0 * M_PI);
  const Real   nu = Real(double(u0) * L / Re);
  const std::size_t T = std::size_t(tmax * L / double(u0));
  const Op which = (op == "bgk") ? Op::BGK : Op::CentralMoments;

  std::printf("3D Taylor-Green, BENCHMARK   CUDA native   D3Q27   %s   %s\n",
              which == Op::BGK ? "bgk" : "cm", sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  D = %d   L = D/2pi = %.3f   Re = V0 L/nu = %.0f\n", D, L, Re);
  std::printf("  u0 = %.4f   Ma = %.4f   nu = %.6e   tau = %.6f\n",
              double(u0), double(u0) * std::sqrt(3.0), double(nu),
              3.0 * double(nu) + 0.5);
  std::printf("  t* = t V0/L up to %.1f   (%zu steps, %d^3 = %ld nodes)\n",
              tmax, T, D, long(D) * D * D);
  std::printf("  NOTE: src/tgv3d.cu's Re = u0 D/nu would read %.0f for this nu.\n\n",
              double(u0) * double(D) / double(nu));

  Solver s(D, D, D, which, nu);
  s.initialise_with(TaylorGreenBenchInit{
      u0, Real(2.0 * M_PI) / Real(D), Real(3.0) * u0 * u0 / Real(16.0)});

  const double N   = double(D) * double(D) * double(D);
  const double sE  = 1.0 / (double(u0) * double(u0));
  const double sZ  = (L / double(u0)) * (L / double(u0));
  const double nus = 1.0 / Re;

  std::vector<Real> rho, ux, uy, uz;
  std::printf("  %8s %13s %13s %13s %11s\n", "t*", "Ek*", "eps_E", "eps_zeta", "gap %");
  std::printf("  %s\n", std::string(64, '-').c_str());

  const std::size_t probe = T / 40 ? T / 40 : 1;
  const auto wall0 = std::chrono::steady_clock::now();
  double prev_E = 0.0, prev_t = 0.0;
  bool first = true;

  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      s.macroscopic_to_host(rho, ux, uy, uz);
      const Diag dd = diagnostics(ux, uy, uz, D);
      const double ts = tmax * double(t) / double(T);
      if (!dd.finite) { std::printf("  DIVERGED at t* = %.3f\n", ts); return 1; }
      const double Ek = dd.energy / N * sE;
      const double zeta = dd.enstrophy / N * sZ;
      const double eps_z = 2.0 * nus * zeta;
      const double eps_E = first ? 0.0 / 0.0 : -(Ek - prev_E) / (ts - prev_t);
      const double gap = (eps_z > 0 && !first) ? 100.0 * (eps_E - eps_z) / eps_z : 0.0 / 0.0;
      std::printf("  %8.3f %13.6f %13.6f %13.6f %11.1f\n", ts, Ek, eps_E, eps_z, gap);
      std::fflush(stdout);
      prev_E = Ek; prev_t = ts; first = false;
    }
    if (t < T) s.step();
  }

  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall0).count();
  std::printf("\n  %zu steps in %.2f s  ->  %.1f MLUPS\n",
              T, sec, double(long(D) * D * D) * double(T) / sec / 1e6);
  return 0;
}
