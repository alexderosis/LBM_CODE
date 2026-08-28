//==============================================================================
//  Force-driven plane Poiseuille flow -- the geometry and forcing check.
//
//  The same case as validation/poiseuille.cpp in the parent Kokkos code, and
//  deliberately the first thing to run after adding walls to a solver, because
//  it has an exact solution and because the way it fails is informative.
//
//  H fluid nodes sit at y = 1..H, with SOLID layers at y = 0 and y = H+1. Under
//  Esoteric Pull a solid cell is simply not visited -- bounce-back is the
//  identity on the storage -- which puts the no-slip planes half-way between the
//  last fluid node and the first solid node, at y = 0.5 and y = H + 0.5:
//
//      u(y) = (G / 2 rho nu) (y - 0.5)(H + 0.5 - y),   u_max = G H^2 / 8 rho nu.
//
//  HOW TO READ THE OUTPUT. A wall in the wrong place gives an error that falls
//  like 1/H; a correctly placed wall leaves a truncation error that falls like
//  1/H^2. So run two resolutions before believing either. The amplitude is
//  fitted by least squares rather than read off as the largest node value: for
//  even H the parabola peaks at y = (H+1)/2, half-way BETWEEN two nodes, and the
//  largest sample is below the continuous maximum by O(1/H^2) for reasons that
//  have nothing to do with the solver.
//
//    usage: channel [-h H] [-nu NU] [-umax U] [-op bgk|cm] [-steps N]
//==============================================================================
#include "lbm/backend.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

// A STRUCT, not a lambda: nvcc restricts where an extended __host__ __device__
// lambda may appear, and a plain functor sidesteps the whole family of rules.
struct RestInit {
  LBM_HD Macro operator()(int, int, int) const {
    return Macro{Real(1), Real(0), Real(0), Real(0)};
  }
};

int main(int argc, char** argv) {
  int H = 32;
  double nu = 1.0 / 6.0, umax = 0.05;
  std::string op = "cm";
  std::size_t T = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h"     && i + 1 < argc) H    = std::atoi(argv[++i]);
    if (a == "-nu"    && i + 1 < argc) nu   = std::atof(argv[++i]);
    if (a == "-umax"  && i + 1 < argc) umax = std::atof(argv[++i]);
    if (a == "-op"    && i + 1 < argc) op   = argv[++i];
    if (a == "-steps" && i + 1 < argc) T    = std::size_t(std::atol(argv[++i]));
  }
  // Momentum diffuses across the channel in H^2/nu steps; 30 of those is well
  // past steady state and cheap at this size.
  if (T == 0) T = std::size_t(30.0 * double(H) * double(H) / nu);

  const int nx = 4, nz = 4, ny = H + 2;
  const double G = 8.0 * 1.0 * nu * umax / (double(H) * double(H));
  const Op which = (op == "bgk") ? Op::BGK : Op::CentralMoments;

  std::printf("Plane Poiseuille   %s   D3Q27   operator %s   %s\n",
              backend::on_device ? "CUDA native" : "HOST reference",
              which == Op::BGK ? "bgk" : "cm", sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  H = %d fluid nodes   nu = %.6f   tau = %.6f   u_max target %.4f\n",
              H, nu, 3.0 * nu + 0.5, umax);
  std::printf("  G = %.6e per node per step   %zu steps\n\n", G, T);

  backend::Fluid fl(nx, ny, nz, which, Real(nu));

  std::vector<std::uint8_t> flags(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      flags[std::size_t(node_id(x, 0,      z, nx, ny))] = Solid;
      flags[std::size_t(node_id(x, ny - 1, z, nx, ny))] = Solid;
    }
  fl.set_geometry(flags);

  BodyForce b;
  b.fx = Real(G);
  fl.set_force(b, ForceUniform);
  fl.initialise_with(RestInit{});

  const auto wall0 = std::chrono::steady_clock::now();
  for (std::size_t t = 0; t < T; ++t) fl.step();
  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall0).count();

  std::vector<Real> rho, ux, uy, uz;
  fl.macroscopic_to_host(rho, ux, uy, uz);

  std::printf("  %4s %14s %14s %12s\n", "y", "u (lattice)", "exact", "err/u_max");
  double worst = 0, num = 0, den = 0;
  for (int y = 1; y <= H; ++y) {
    double u = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) u += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
    u /= double(nx) * nz;
    const double shape = (y - 0.5) * (H + 0.5 - y);
    const double exact = (G / (2.0 * nu)) * shape;
    const double rel = (u - exact) / umax;
    worst = std::fmax(worst, std::fabs(rel));
    num += u * shape;
    den += shape * shape;
    if (H <= 16 || y % (H / 8) == 0 || y == 1)
      std::printf("  %4d %14.8f %14.8f %12.2e\n", y, u, exact, rel);
  }

  const double amp = num / den, amp_exact = G / (2.0 * nu);
  std::printf("\n  fitted amplitude   %.8e   exact %.8e   rel %.3e\n",
              amp, amp_exact, (amp - amp_exact) / amp_exact);
  std::printf("  worst node error / u_max                              %.3e\n", worst);
  std::printf("  that error times H^2                                  %.3f\n",
              worst * double(H) * double(H));
  std::printf("      A wall in the WRONG place gives an error going like 1/H, so this\n");
  std::printf("      product would grow with H. A correctly placed wall leaves the\n");
  std::printf("      1/H^2 truncation error, and the product stays put.\n");
  std::printf("\n  %zu steps in %.2f s  ->  %.1f MLUPS\n",
              T, sec, double(fl.nodes()) * double(T) / sec / 1e6);
  return 0;
}
