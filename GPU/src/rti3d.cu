//==============================================================================
//  Rayleigh-Taylor in three dimensions, phase field + central moments.
//
//  De Rosis & Enan, Phys. Fluids 33, 043315 (2021), Sec. III I, Table IX. This
//  is a deliberate re-run of ../../validation/enan_rt.cpp on the independent
//  CUDA implementation: same case, same parameters, same reported quantity, no
//  shared headers. The point is that two implementations written months apart
//  from the same paper should land in the same place, and any gap between them
//  is a defect in one of them rather than a property of the model.
//
//  SETUP, theirs exactly. A box W wide, 4W tall and W deep; heavy fluid ABOVE
//  light; periodic across, no-slip top and bottom. The interface carries a
//  single-mode perturbation, their Eq. (83):
//
//      y_i = 2W + 0.05 W [ cos(2 pi x / W) + cos(2 pi z / W) ]
//
//  so the spike starts at y = 1.9 W. Gravity is set by sqrt(gW) = U = 0.04,
//  Re = W sqrt(gW) / nu, At = 0.5 with rho_L = 1.
//
//  THREE PARAMETER READINGS COME FROM THEIR DRIVERS RATHER THAN THEIR PROSE,
//  and the parent found all three the hard way. They are reproduced here
//  because the comparison is only meaningful if both codes are solving the same
//  problem:
//
//   1. sigma = nu U / Ca, with NO rho_H, although Ca = mu_H U / sigma would put
//      one there. At At = 0.5 that is a factor of three.
//   2. M = U W / Pe with W the DOMAIN width, not U xi / Pe. A factor of W/xi.
//   3. THE REFERENCE TIME HAS NO ATWOOD FACTOR IN 3-D: t0 = sqrt(W/g), where
//      their 2-D drivers use sqrt(W/(g At)). Using the 2-D form here stretches
//      the clock by 1/sqrt(At) = 1.41, so a spike reported at t/t0 = 3 has
//      actually been run 41% further than theirs.
//
//  THE INTERFACE WIDTH IS NOT GIVEN BY THE PAPER for these cases. xi = 5 is
//  used, matching the parent's committed run; -iw sweeps it.
//
//  THE REPORTED QUANTITY is their y-dagger: the vertical position of the SPIKE,
//  the lowest point the heavy fluid has reached, over W. It runs down from 1.9.
//  Three measures are computed, for the reason the parent's banner gives at
//  length: the global lowest node with phi > 0.5 is a different quantity from
//  the tip of a coherent finger the moment anything detaches, and it can only
//  ever read LOWER. Measuring all three separates a physics difference from a
//  measurement artefact.
//
//  Run:  ./rti3d [-w 64] [-re 256] [-ca 960] [-pe 1024] [-iw 5] [-tmax 3]
//                [-frames 60] [-dump PREFIX] [-vol] [-bgk]
//==============================================================================
#include "lbm/backend.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The paper's reporting instants.
static const double T_STAR[7] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

//------------------------------------------------------------------------------
// phi and the hydrostatic pressure, from one analytic profile.
//
// THE PRESSURE IS SEEDED THROUGH THE DIFFUSE INTERFACE, by integrating the
// ACTUAL density profile away from it rather than assuming a sharp step. At
// At = 0.5 the imbalance either way is small, but the closed form is free and
// the same expression is correct at any density ratio -- and the alternative is
// an acoustic transient that has to damp before anything can be measured.
//
// The zeroth moment here is p~ = p / (rho cs^2), which is why the integral is
// divided by rho/3 and not by rho.
//------------------------------------------------------------------------------
struct RtInit {
  Real half, amp, k, iw, rl, rh, g;
  int nx, nz;

  LBM_HD void operator()(int x, int y, int z, Real& ph, Real& pt) const {
    const float fx = float(x), fy = float(y), fz = float(z);
    const float yi = float(half) + float(amp) * (cosf(float(k) * fx) + cosf(float(k) * fz));
    const float dz = fy - yi;
    ph = Real(0.5f * (1.0f + tanhf(2.0f * dz / float(iw))));

    // I = integral of rho dy from the interface, through the tanh profile.
    // log(1 + e^-2a) is written as a shifted softplus so it stays accurate for
    // large |a| instead of underflowing to log(1) = 0.
    const float az = fabsf(dz) * 2.0f / float(iw);
    const float lnch = az + logf(1.0f + expf(-2.0f * az)) - 0.6931471805599453f;
    const float I = float(rl) * dz
                  + (float(rh) - float(rl)) * 0.5f * (dz + 0.5f * float(iw) * lnch);
    const float r = float(rl) + float(ph) * (float(rh) - float(rl));
    pt = Real((-float(g) * I) / (r / 3.0f));
  }
};

int main(int argc, char** argv) {
  int W = 64, frames = 60;
  double Re = 256.0, At = 0.5, Ca = 960.0, Pe = 1024.0, U = 0.04, iw = 5.0, tmax = 3.0;
  const char* dump = "";
  bool cm = true, vol = false;

  for (int i = 1; i < argc; ++i) {
    auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-w"))      { if (i+1<argc) W = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-frames")) { if (i+1<argc) frames = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-re"))     num(Re);
    else if (!std::strcmp(argv[i], "-at"))     num(At);
    else if (!std::strcmp(argv[i], "-ca"))     num(Ca);
    else if (!std::strcmp(argv[i], "-pe"))     num(Pe);
    else if (!std::strcmp(argv[i], "-u"))      num(U);
    else if (!std::strcmp(argv[i], "-iw"))     num(iw);
    else if (!std::strcmp(argv[i], "-tmax"))   num(tmax);
    else if (!std::strcmp(argv[i], "-bgk"))    cm = false;
    else if (!std::strcmp(argv[i], "-vol"))    vol = true;
    else if (!std::strcmp(argv[i], "-dump") && i + 1 < argc) dump = argv[++i];
  }

  const int nx = W, ny = 4 * W, nz = W;
  const double g     = U * U / double(W);            // so sqrt(gW) = U
  const double nu    = double(W) * U / Re;
  const double rho_l = 1.0, rho_h = (1.0 + At) / (1.0 - At);
  const double sigma = nu * U / Ca;                  // NO rho_H -- see the banner
  const double M     = U * double(W) / Pe;           // domain-based
  const double t_ref = std::sqrt(double(W) / g);     // NO Atwood factor in 3-D
  const std::size_t nsteps = std::size_t(tmax * t_ref);

  const backend::DeviceInfo dev = backend::device_info();
  std::printf("Rayleigh-Taylor 3-D   phase field %s + fluid %s   %s, %s\n",
              cm ? "CM" : "BGK", cm ? "CM" : "BGK", dev.name.c_str(),
              sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  %d x %d x %d   Re %g   At %g   Ca %g   Pe %g   xi %g\n",
              nx, ny, nz, Re, At, Ca, Pe, iw);
  std::printf("  nu %.6e   sigma %.6e   M %.6e   rho_H %.2f   g %.6e\n",
              nu, sigma, M, rho_h, g);
  std::printf("  t_ref %.1f   steps %zu\n\n", t_ref, nsteps);

  //---- solver ----------------------------------------------------------------
  backend::PhaseFieldOn<D3Q27> pf(nx, ny, nz);
  pf.phase.width = Real(iw);
  pf.set_mobility(Real(M));                 // cs^2 = 1/3 on D3Q27, not D3Q7's 1/4
  pf.set_phase_op(cm ? PhaseOp::CentralMoments : PhaseOp::BGK);
  pf.set_fluid_op(cm ? MultiOp::CentralMoments : MultiOp::BGK);
  pf.fluid.rho_L = Real(rho_l);          pf.fluid.rho_H = Real(rho_h);
  pf.fluid.mu_L  = Real(rho_l * nu);     pf.fluid.mu_H  = Real(rho_h * nu);
  pf.fluid.kappa = Real(MultiphaseModel::kappa_from_sigma(Real(sigma), Real(iw)));
  pf.fluid.beta  = Real(MultiphaseModel::beta_from_sigma (Real(sigma), Real(iw)));
  pf.fluid.by    = Real(-g);
  pf.enable_viscous_force(true);

  // No-slip top and bottom; periodic elsewhere, which this code's indexing gives
  // for free. The phase field sees the same two planes as zero-flux walls.
  const std::size_t NN = static_cast<std::size_t>(long(nx) * ny * nz);
  const std::uint8_t kFluid = Fluid, kSolid = Solid;
  const std::uint8_t kBulk = PhaseBulk, kWall = PhaseWall;
  std::vector<std::uint8_t> ff(NN, kFluid), pfl(NN, kBulk);
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x)
      for (int y : {0, ny - 1}) {
        const std::size_t n = std::size_t(node_id(x, y, z, nx, ny));
        ff[n] = kSolid;  pfl[n] = kWall;
      }
  pf.set_geometry(pfl, ff);

  RtInit init;
  init.half = Real(0.5 * double(ny));
  init.amp  = Real(0.05 * double(W));
  init.k    = Real(2.0 * M_PI / double(W));
  init.iw   = Real(iw);
  init.rl   = Real(rho_l);
  init.rh   = Real(rho_h);
  init.g    = Real(g);
  init.nx   = nx;
  init.nz   = nz;
  pf.initialise_with(init);

  //---- march, measuring at their instants and dumping frames -----------------
  const std::size_t frame_every =
      (frames > 1) ? std::max<std::size_t>(1, nsteps / std::size_t(frames - 1)) : nsteps + 1;
  int next = 0, nframe = 0;
  std::vector<Real> phi;
  bool finite = true;
  std::printf("  t/t0     y+ global   y+ on-axis  y+ pop 0.1%%   max|u|\n");

  for (std::size_t step = 0; step <= nsteps; ++step) {
    const bool want_report = (next < 7 && step >= std::size_t(T_STAR[next] * t_ref));
    const bool want_frame  = (*dump && step % frame_every == 0);
    if (want_report || want_frame) {
      pf.field_to_host(pf.phi_device(), phi);

      if (want_report) {
        std::vector<Real> ux, uy;
        pf.field_to_host(pf.ux_device(), ux);
        pf.field_to_host(pf.uy_device(), uy);
        int spike = ny - 1, spike_ax = ny - 1, spike_pop = ny - 1;
        double um = 0;
        const int xc = nx / 2, zc = nz / 2;
        const long plane = long(nx) * nz;
        const long need = std::max(1L, plane / 1000);
        for (int y = 0; y < ny; ++y) {
          long cnt = 0;
          for (int z = 0; z < nz; ++z)
            for (int x = 0; x < nx; ++x) {
              const std::size_t n = std::size_t(node_id(x, y, z, nx, ny));
              const double p = double(phi[n]);
              if (!(p == p)) finite = false;
              if (p > 0.5) { ++cnt; if (y < spike) spike = y; }
              const double a = double(ux[n]), b = double(uy[n]);
              um = std::fmax(um, std::sqrt(a * a + b * b));
            }
          if (cnt >= need && y < spike_pop) spike_pop = y;
          if (double(phi[std::size_t(node_id(xc, y, zc, nx, ny))]) > 0.5 && y < spike_ax)
            spike_ax = y;
        }
        std::printf("  %4.1f     %8.4f    %8.4f    %8.4f     %.3e\n",
                    T_STAR[next], double(spike) / W, double(spike_ax) / W,
                    double(spike_pop) / W, um);
        std::fflush(stdout);
        ++next;
      }

      // One frame of the order parameter.
      //
      // A PLANE IS NOT THE SAME PICTURE AS THE SURFACE once the spike rolls up.
      // The mid-z plane cuts through the finger and is what the spike position
      // is read from, but the mushroom cap and the saddles between the four
      // sides are exactly what a single cut misses -- and they are what the
      // paper's Fig. 15 shows. -vol writes the whole volume so the phi = 1/2
      // isosurface can be extracted; it is 4 MB a frame against 64 kB.
      //
      // The volume layout matches doc/fig/enan_rt3d_render.py: three int32
      // dimensions, then nx*ny*nz float32 with x fastest.
      if (want_frame) {
        char fp[256];
        std::snprintf(fp, sizeof fp, "%s_%04d.bin", dump, nframe);
        std::FILE* f = std::fopen(fp, "wb");
        if (f) {
          if (vol) {
            const int hdr[3] = {nx, ny, nz};
            std::fwrite(hdr, sizeof(int), 3, f);
            std::vector<float> v(std::size_t(nx) * ny * nz);
            for (int z = 0; z < nz; ++z)
              for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                  v[(std::size_t(z) * ny + y) * nx + x] =
                      float(phi[std::size_t(node_id(x, y, z, nx, ny))]);
            std::fwrite(v.data(), sizeof(float), v.size(), f);
          } else {
            const int hdr[2] = {nx, ny};
            std::fwrite(hdr, sizeof(int), 2, f);
            std::vector<float> plane_data(std::size_t(nx) * ny);
            for (int y = 0; y < ny; ++y)
              for (int x = 0; x < nx; ++x)
                plane_data[std::size_t(y) * nx + x] =
                    float(phi[std::size_t(node_id(x, y, nz / 2, nx, ny))]);
            std::fwrite(plane_data.data(), sizeof(float), plane_data.size(), f);
          }
          std::fclose(f);
          ++nframe;
        }
      }
    }
    if (step < nsteps) pf.step();
  }

  std::printf("\n  %s   %d frame(s) written%s%s\n",
              finite ? "finite throughout" : "NON-FINITE VALUES APPEARED",
              nframe, *dump ? " to " : "", dump);
  std::printf("  Their Table IX (3-D, Re=256): "
              "1.898 1.858 1.741 1.553 1.304 1.001 0.648\n");
  std::printf("  Kokkos M3LB, W=64 xi=5:       "
              "1.9062 1.8750 1.7969 1.6406 1.4375 1.2031 0.9219\n");
  return finite ? 0 : 1;
}
