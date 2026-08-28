//==============================================================================
//  Rayleigh-Benard convection -- thermal, geometry and buoyancy, together.
//
//  A layer of depth H heated from below and cooled from above, periodic in the
//  horizontal, no-slip and isothermal on both plates. The controlling groups are
//
//      Ra = g beta dT H^3 / (nu D),      Pr = nu / D.
//
//  WHY THIS CASE AND NOT A CAVITY. It needs no reference table. Linear stability
//  gives the onset of convection between two RIGID plates as
//
//      Ra_c = 1707.762,
//
//  a number that does not depend on Pr, on the working fluid, or on anything
//  else -- so a solver either reproduces it or does not. Below Ra_c a
//  perturbation must decay and Nu return to 1; above it, convection must sustain
//  and Nu exceed 1. That is a two-sided check against a constant of nature, and
//  it exercises every piece added here at once: solid walls for the momentum,
//  Dirichlet walls for the scalar, and the Boussinesq force that couples them.
//
//  THE TWO WALL PLANES COINCIDE, and they must. Halfway bounce-back puts no-slip
//  at y = 0.5 and y = H + 0.5; anti-bounce-back puts the isothermal planes in
//  exactly the same places. If they did not coincide, H would mean two different
//  things in Ra and the onset would come out wrong by O(1/H) -- which looks like
//  ordinary discretisation error and is not.
//
//  Nu = 1 + H <u_y T> / (D dT), averaged over the fluid: total heat flux over
//  the flux conduction alone would carry.
//
//  A NOTE ON RESOLUTION. Ra_c on a lattice of H layers differs from 1707.762 by
//  O(1/H^2), so bracketing it at H = 12 is a demonstration and at H = 48 a
//  measurement. Run two H before quoting a number.
//
//    usage: rayleigh_benard [-h H] [-ra RA] [-pr PR] [-nu NU] [-op bgk|cm]
//                           [-steps N] [-aspect A]
//==============================================================================
#include "lbm/backend.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct RestInit {
  LBM_HD Macro operator()(int, int, int) const {
    return Macro{Real(1), Real(0), Real(0), Real(0)};
  }
};

//------------------------------------------------------------------------------
// The conductive profile plus one roll-shaped perturbation.
//
// Seeded deliberately at the critical wavelength rather than with noise: the
// question is whether THIS mode grows, and a broadband seed would leave the
// answer depending on which mode happened to win.
//------------------------------------------------------------------------------
struct StratifiedInit {
  int H;
  Real amp, kx;
  LBM_HD Real operator()(int x, int y, int) const {
    const double yy = (double(y) - 0.5) / double(H);         // 0 at the hot plane
    const double base = 0.5 - yy;
    const double pert = double(amp) * sin(double(kx) * double(x)) * sin(M_PI * yy);
    return Real(base + pert);
  }
};

int main(int argc, char** argv) {
  int H = 32, aspect = 2;
  double Ra = 5000.0, Pr = 1.0, nu = 0.02, amp = 0.01;
  std::string op = "cm";
  std::size_t T = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h"      && i + 1 < argc) H      = std::atoi(argv[++i]);
    if (a == "-ra"     && i + 1 < argc) Ra     = std::atof(argv[++i]);
    if (a == "-pr"     && i + 1 < argc) Pr     = std::atof(argv[++i]);
    if (a == "-nu"     && i + 1 < argc) nu     = std::atof(argv[++i]);
    if (a == "-op"     && i + 1 < argc) op     = argv[++i];
    if (a == "-aspect" && i + 1 < argc) aspect = std::atoi(argv[++i]);
    if (a == "-amp"    && i + 1 < argc) amp    = std::atof(argv[++i]);
    if (a == "-steps"  && i + 1 < argc) T      = std::size_t(std::atol(argv[++i]));
  }

  const double D  = nu / Pr;
  const double dT = 1.0;                       // plates at +1/2 and -1/2
  const double g  = Ra * nu * D / (dT * double(H) * double(H) * double(H));
  const int nx = aspect * H, nz = 4, ny = H + 2;
  const Op which = (op == "bgk") ? Op::BGK : Op::CentralMoments;

  // The thermal diffusion time across the layer. Growth near onset is slow --
  // the linear rate vanishes AT Ra_c -- so a short run near the threshold shows
  // nothing either way.
  const double t_diff = double(H) * double(H) / D;
  if (T == 0) T = std::size_t(30.0 * t_diff);

  std::printf("Rayleigh-Benard   %s   D3Q27 fluid / D3Q7 scalar   operator %s   %s\n",
              backend::on_device ? "CUDA native" : "HOST reference",
              which == Op::BGK ? "bgk" : "cm", sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  H = %d   %d x %d x %d   Ra = %.1f   Pr = %.2f\n", H, nx, ny, nz, Ra, Pr);
  std::printf("  nu = %.6f (tau %.4f)   D = %.6f (tau %.4f)   g beta dT = %.4e\n",
              nu, 3.0 * nu + 0.5, D, 4.0 * D + 0.5, g);
  std::printf("  Ra_c = 1707.762 between rigid plates;  this run is %s it\n",
              Ra > 1707.762 ? "ABOVE" : "BELOW");
  std::printf("  %zu steps = %.1f thermal diffusion times\n\n", T, double(T) / t_diff);

  backend::Fluid  fl(nx, ny, nz, which, Real(nu));
  backend::Scalar sc(nx, ny, nz, Real(D), Real(0));

  // Geometry. The momentum walls and the thermal walls are the SAME two layers.
  std::vector<std::uint8_t> ff(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  std::vector<std::uint8_t> sf(std::size_t(nx) * ny * nz, std::uint8_t(ScalarBulk));
  std::vector<Real>         sw(std::size_t(nx) * ny * nz, Real(0));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      const std::size_t lo = std::size_t(node_id(x, 0,      z, nx, ny));
      const std::size_t hi = std::size_t(node_id(x, ny - 1, z, nx, ny));
      ff[lo] = Solid;      ff[hi] = Solid;
      sf[lo] = ScalarDirichlet; sf[hi] = ScalarDirichlet;
      sw[lo] = Real( 0.5); sw[hi] = Real(-0.5);          // hot below, cold above
    }
  fl.set_geometry(ff);
  sc.set_geometry(sf, sw);

  fl.enable_velocity_output();
  sc.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  // Buoyancy. gy is POSITIVE here: the force on a parcel hotter than the
  // reference must point away from gravity, so warm fluid rises.
  BodyForce b;
  b.T = sc.field_device();
  b.gx = Real(0); b.gy = Real(g); b.gz = Real(0);
  b.rho0 = Real(1); b.beta = Real(1); b.T0 = Real(0);
  fl.set_force(b, ForceBoussinesq);

  fl.initialise_with(RestInit{});
  sc.initialise_with(StratifiedInit{H, Real(amp), Real(2.0 * M_PI / nx)});

  std::vector<Real> rho, ux, uy, uz, Tf;
  auto diagnose = [&](double& nu_number, double& umax) {
    fl.macroscopic_to_host(rho, ux, uy, uz);
    sc.field_to_host(Tf);
    double flux = 0, peak = 0;
    long cells = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = 1; y <= H; ++y)
        for (int x = 0; x < nx; ++x) {
          const std::size_t n = std::size_t(node_id(x, y, z, nx, ny));
          flux += double(uy[n]) * double(Tf[n]);
          const double s = std::sqrt(double(ux[n]) * double(ux[n]) +
                                     double(uy[n]) * double(uy[n]) +
                                     double(uz[n]) * double(uz[n]));
          peak = std::fmax(peak, s);
          ++cells;
        }
    flux /= double(cells);
    nu_number = 1.0 + double(H) * flux / (D * dT);
    umax = peak;
  };

  std::printf("  %10s %12s %14s %14s\n", "t/t_diff", "Nu", "Nu - 1", "max |u|");
  const std::size_t probe = T / 20 ? T / 20 : 1;
  const auto wall0 = std::chrono::steady_clock::now();

  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      double nun, umax;
      diagnose(nun, umax);
      if (!std::isfinite(nun)) { std::printf("  DIVERGED at t = %zu\n", t); return 1; }
      std::printf("  %10.2f %12.6f %14.3e %14.3e\n",
                  double(t) / t_diff, nun, nun - 1.0, umax);
      std::fflush(stdout);
    }
    if (t < T) {
      // THE COUPLING ORDER. Refresh T first, so the fluid collides against the
      // temperature at its OWN time level. Stepping the fluid first and letting
      // the scalar catch up is a first-order splitting error that does not
      // vanish under refinement. On the device these are three launches on the
      // default stream, so they are already ordered; no fence is needed.
      sc.compute_field();
      fl.step();
      sc.step();
    }
  }

  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall0).count();
  double nun, umax;
  diagnose(nun, umax);
  std::printf("\n  final Nu = %.6f   max |u| = %.4e\n", nun, umax);
  std::printf("  %s: at Ra = %.1f %s Ra_c = 1707.762, convection %s\n",
              (nun - 1.0 > 1e-3) == (Ra > 1707.762) ? "AS EXPECTED" : "UNEXPECTED",
              Ra, Ra > 1707.762 ? ">" : "<",
              nun - 1.0 > 1e-3 ? "is sustained" : "died out");
  std::printf("  %zu steps in %.2f s  ->  %.1f MLUPS (fluid nodes only)\n",
              T, sec, double(fl.nodes()) * double(T) / sec / 1e6);
  return 0;
}
