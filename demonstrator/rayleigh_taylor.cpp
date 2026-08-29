//==============================================================================
//  Rayleigh-Taylor instability -- heavy fluid over light, at a density ratio.
//
//  A DEMONSTRATOR, not a validation case: there is no closed-form answer to a
//  nonlinear instability, and this is registered with no test. What it shows is
//  the multiphase module doing the thing it was built for -- an interface that
//  rolls up, reconnects and keeps going -- on the potential-form operator, which
//  is the only one here that reaches a density ratio at all.
//
//  SETUP, the standard one. A box W wide and 4W tall, periodic in x, no-slip on
//  the top and bottom. Heavy fluid above, light below, and the interface given a
//  single-mode perturbation of amplitude 0.1 W at the box wavelength:
//
//      y_i(x) = 2W + 0.1 W cos(2 pi x / W).
//
//  The controlling groups are the Atwood number and a Reynolds number built on
//  the free-fall velocity,
//
//      At = (rho_H - rho_L) / (rho_H + rho_L),    Re = W sqrt(g W) / nu,
//
//  and time is reported in units of sqrt(W / (g At)), so t* is comparable with
//  any other run at the same At regardless of the lattice numbers underneath.
//
//  HYDROSTATIC INITIALISATION, THROUGH THE DIFFUSE INTERFACE. The pressure is
//  seeded by integrating the ACTUAL density profile away from the interface,
//
//      p(y) = -g I(y),   I(y) = integral from y_i to y of rho(phi(s)) ds,
//
//  which the tanh profile makes closed-form: with phi = 1/2 (1 + tanh(2 dz/W)),
//
//      I = rho_L dz + (rho_H - rho_L) [ dz + (W/2) ln cosh(2 dz / W) ] / 2 .
//
//  Assuming a SHARP interface instead leaves an imbalance of order
//  (rho_H - rho_L) g W_int spread over the interface. At At = 0.5 that is 1e-4
//  and invisible. At At = 0.998 it is 0.25, p~ on the light side starts at 0.75,
//  and |u| reaches the free-fall velocity within THIRTY steps -- before the
//  instability could have grown at all -- and the run is gone by step 120.
//  A diffuse-interface model deserves a diffuse-interface initial condition.
//
//  ln cosh is evaluated as |z| + log1p(exp(-2|z|)) - ln 2 rather than through
//  cosh, which overflows in the far field at any useful box height.
//
//  It is then divided by rho(phi) cs2 to get p~, which is what the populations
//  carry. Seeding a uniform p~ instead would start the box with the whole
//  hydrostatic imbalance as an acoustic transient, larger than the instability.
//
//  WHY THE ZERO GOES AT THE INTERFACE. Only grad p is physical, so the absolute
//  level is free -- but it is NOT free numerically. p~ = p/(rho cs2), so a
//  constant added to p is not a constant added to p~ when rho varies, and both
//  of the large terms that cancel in the interface, rho cs2 grad p~ and
//  F_p = -p~ cs2 grad rho, scale with the level. Since p is continuous across
//  the interface while rho is not, p~ jumps by the density ratio wherever p is
//  not near zero -- so the gauge that keeps BOTH sides small is the one that
//  puts p = 0 where the two fluids meet.
//
//  Two measurements of how much this matters. Referencing p to rho_L cs2 (a
//  uniform physical pressure) made this case diverge inside 424 steps at
//  At = 0.5, with |u| at twice the free-fall velocity at step zero before
//  anything had moved. Referencing to the top of the box instead of the
//  interface is harmless at At = 0.5, where g rho_H 2W is 1e-2, and is not at
//  At = 0.998, where the same quantity is 3.2 and p~ in the light fluid starts
//  at 9.6.
//
//  COLLISION OPERATOR. -op cm (the default) is the central-moment scheme of the
//  reference's Sec. II.C; -op bgk is single-relaxation-time. The difference is
//  not cosmetic at high Reynolds number: BGK relaxes every mode at omega, so as
//  omega approaches 2 the non-hydrodynamic modes ring instead of decaying. At
//  Re = 30000 on a 128-wide box omega is 1.99795, and only the moment operator
//  runs there at all.
//
//  WHAT TO DISTRUST HERE. Two things, both stated in the module headers and both
//  live in this case:
//
//   * The interface never touches a wall in a run to t* = 3, which is what makes
//     it legitimate to run with no wetting condition. Take this further and the
//     spike reaches the floor, and what happens there is not modelled.
//   * The pressure form's F_p and the LBE's own pressure gradient are different
//     discrete operators, and their mismatch scales with the density jump and
//     with 1/W_int^2 (measured in validation/laplace). A well resolved interface
//     matters more here than the phase-field literature's usual W_int = 4
//     suggests; the default is 5.
//==============================================================================
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/MultiphasePotentialBGK.hpp"
#include "collision/PhaseFieldBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/PhaseFieldSolver.hpp"
#include "solver/ViscousInterfaceForce.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

// The lattices are template parameters rather than typedefs so that this case
// can run on D3Q27 as well as D2Q9. That is not a generality for its own sake:
// comparing this model against the colour gradient of demonstrator/rt_colour.cpp
// is only a comparison of the two INTERFACE-CAPTURE SCHEMES if the lattice is
// held fixed, and the colour-gradient operator exists only on D3Q27. Run on
// different lattices, the difference between the two films confounds the model
// with the velocity set, and there is no way afterwards to say which produced
// what. -3d puts this case on D3Q27 in the same four-cell slab.
template <class FLat> using BgkOf = MultiphasePotentialBGK<FLat, SecondOrderPhi<FLat>, RawPopulations>;
template <class FLat> using CmOf  = MultiphaseCentralMoments<FLat>;

//------------------------------------------------------------------------------
// Raw field dumps, NOT pictures.
//
// This case used to rasterise its own frames. That was a mistake for the reason
// vol_aorta.cpp and vol_urban.cpp already encode: a renderer welded to the
// simulation means every change of colour map costs a full re-run, and at this
// resolution that is eight and a half minutes to alter a hue. Fields go out raw
// and demonstrator/render_rt.cpp turns them into frames in seconds.
//
// The format is FieldDump.hpp's: int32 nx, int32 ny, then nx*ny float32 in row
// major order. It is written out here rather than by including that header
// because figdump::write_raw logs every file it writes, and three files a frame
// over a hundred and eighty frames would bury the run table this case exists to
// print.
//------------------------------------------------------------------------------
template <class Get>
static void dump_field(const std::string& path, Index nx, Index ny, Get get) {
  std::vector<float> v(std::size_t(nx) * std::size_t(ny));
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x)
      v[std::size_t(y) * std::size_t(nx) + std::size_t(x)] = float(get(x, y));
  std::ofstream o(path, std::ios::binary);
  // Say so rather than reporting a frame count for files that were never
  // written -- a missing output directory is otherwise completely silent, and
  // the run only fails at render time, minutes later.
  if (!o) { std::printf("  cannot write %s\n", path.c_str()); return; }
  const std::int32_t a = int(nx), b = int(ny);
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}

//------------------------------------------------------------------------------
// Everything the run needs, so the simulation body can be one template over the
// collision operator rather than two copies of it.
//------------------------------------------------------------------------------
struct Params {
  Index W; double At, Re, U, iw, M, sigma, tmax;
  int nframes; std::string dump; const char* op;
  bool three_d; int nz;
};

template <class FLat, class PLat, class FColl>
static void simulate(const Params& P) {
  using FluidSlv = FluidSolver<FLat, EsotericPull<FLat>, FColl>;
  using PColl    = PhaseFieldBGK<PLat>;
  using PhaseSlv = PhaseFieldSolver<PLat, EsotericPull<PLat>, PColl>;

  const Index W = P.W, nx = P.W, ny = 4 * P.W;
  const Index nz = Index(FLat::D == 3 ? P.nz : 1);
  const double g   = P.U * P.U / double(W);
  const double nu  = double(W) * P.U / P.Re;
  const double rho_l = 1.0, rho_h = (1.0 + P.At) / (1.0 - P.At);
  const double t_ref = std::sqrt(double(W) / (g * P.At));
  const std::size_t nsteps = std::size_t(P.tmax * t_ref);

  std::printf("Rayleigh-Taylor   %s fluid (pressure form, %s) + %s phase field\n",
              FLat::name, FColl::name, PLat::name);
  std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
  if (nz > 1)
    std::printf("%dx%dx%d slab, no-slip walls in y   At = %.4f (rho_H/rho_L = %.2f)"
                "   Re = %.0f   nu = %.3e\n",
                int(nx), int(ny), int(nz), P.At, rho_h / rho_l, P.Re, nu);
  else
    std::printf("%dx%d   At = %.4f (rho_H/rho_L = %.2f)   Re = %.0f   nu = %.3e\n",
                int(nx), int(ny), P.At, rho_h / rho_l, P.Re, nu);
  std::printf("g = %.3e   U = %.3f   W_int = %.1f   M = %.3f   sigma = %.1e\n",
              g, P.U, P.iw, P.M, P.sigma);
  const double tau = nu / (1.0 / 3.0);
  std::printf("tau = %.3e   omega = %.6f\n", tau, 1.0 / (tau + 0.5));
  std::printf("t* = %.1f is %zu steps (t_ref = %.1f)\n\n", P.tmax, nsteps, t_ref);

  Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  PColl pc;
  pc.omega = PColl::omega_from_mobility(Real(P.M));
  pc.width = Real(P.iw);
  PhaseSlv pf(d, pc);

  const Real half = Real(0.5) * Real(ny), amp = Real(0.1) * Real(W);
  const Real k = Real(2.0 * M_PI) / Real(W), iwr = Real(P.iw);
  const Index hx = d.hx, hy = d.hy;
  pf.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real x = Real(px - hx), y = Real(py - hy);
    const Real yi = half + amp * Kokkos::cos(k * x);
    return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (y - yi) / iwr));
  });

  ViscousInterfaceForce<FLat> vf(d);

  FColl fc;
  fc.phi = pf.phi();
  fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
  fc.Lap = pf.laplacian();
  fc.Vx = vf.x();  fc.Vy = vf.y();  fc.Vz = vf.z();
  fc.rho_L = Real(rho_l);      fc.rho_H = Real(rho_h);
  fc.mu_L  = Real(rho_l * nu); fc.mu_H  = Real(rho_h * nu);
  fc.kappa = FColl::kappa_from_sigma(Real(P.sigma), Real(P.iw));
  fc.beta  = FColl::beta_from_sigma(Real(P.sigma), Real(P.iw));
  fc.by    = Real(-g);
  FluidSlv fl(d, fc);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });

  auto phiv = pf.phi();
  const Real rl = Real(rho_l), rh = Real(rho_h), gr = Real(g);
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real x = Real(px - hx), y = Real(py - hy);
    const Real yi = half + amp * Kokkos::cos(k * x);
    const Real dz = y - yi;
    const Real az = (dz < Real(0) ? -dz : dz) * Real(2) / iwr;
    const Real lnch = az + Kokkos::log(Real(1) + Kokkos::exp(Real(-2) * az))
                    - Real(0.6931471805599453);
    const Real I = rl * dz + (rh - rl) * Real(0.5) * (dz + Real(0.5) * iwr * lnch);
    const Real p = -gr * I;
    const Real r = rl + phiv(n) * (rh - rl);
    return FlowState{p / (r / Real(3)), Real(0), Real(0), Real(0)};
  });

  pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());

  const Index zs = nz / 2;      // the slice that is measured and rendered
  const std::size_t every = nsteps / std::size_t(P.nframes > 0 ? P.nframes : 1);
  int frame = 0;
  std::printf("%-8s %-9s %-11s %-11s %-11s %-11s\n",
              "t*", "step", "spike/W", "bubble/W", "max |u|", "max |wz|");
  std::printf("%s\n", std::string(66, '-').c_str());

  for (std::size_t step = 0; step <= nsteps; ++step) {
    if (every && step % every == 0) {
      pf.compute_field();
      fl.compute_macroscopic();
      auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
      auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());

      Index spike = ny - 1, bubble = 0;
      double umx = 0, wmx = 0;
      bool bad = false;
      for (Index y = 1; y < ny - 1; ++y)
        for (Index x = 0; x < nx; ++x) {
          const double p = double(hp(d.id(x, y, zs)));
          if (!std::isfinite(p)) bad = true;
          if (p > 0.5 && y < spike)  spike = y;
          if (p < 0.5 && y > bubble) bubble = y;
          const double a = double(hu(d.id(x, y, zs))), b = double(hv(d.id(x, y, zs)));
          umx = std::max(umx, std::sqrt(a * a + b * b));
          // Peak vorticity, reported so the colour scale of the render can be
          // set from the flow instead of guessed at.
          const Index xp = (x + 1) % nx, xm = (x + nx - 1) % nx;
          const double wz = 0.5 * (double(hv(d.id(xp, y, zs))) - double(hv(d.id(xm, y, zs))))
                          - 0.5 * (double(hu(d.id(x, y + 1, zs))) - double(hu(d.id(x, y - 1, zs))));
          wmx = std::max(wmx, std::abs(wz));
        }
      const double ts = double(step) / t_ref;
      std::printf("%-8.3f %-9zu %-11.4f %-11.4f %-11.3e %-11.3e\n", ts, step,
                  (double(spike) - 0.5 * double(ny)) / double(W),
                  (double(bubble) - 0.5 * double(ny)) / double(W), umx, wmx);
      if (!P.dump.empty()) {
        char nm[512];
        auto at = [&](const char* f) {
          std::snprintf(nm, sizeof nm, "%s/rt_%04d_%s.bin", P.dump.c_str(), frame, f);
          return std::string(nm);
        };
        dump_field(at("phi"), nx, ny, [&](Index x, Index y) { return hp(d.id(x, y, zs)); });
        dump_field(at("ux"),  nx, ny, [&](Index x, Index y) { return hu(d.id(x, y, zs)); });
        dump_field(at("uy"),  nx, ny, [&](Index x, Index y) { return hv(d.id(x, y, zs)); });
      }
      ++frame;
      if (bad) { std::printf("  DIVERGED\n"); break; }
    }
    if (step == nsteps) break;
    pf.refresh();
    fl.compute_macroscopic();
    vf.refresh(fc);
    fl.step(true);
    pf.step();
  }
  std::printf("\n%d frame(s)%s\n", frame,
              P.dump.empty() ? " (pass -dump <dir> to write the fields)"
                             : " dumped; render with demonstrator/render_rt");
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Params P{128, 0.5, 256.0, 0.04, 5.0, 0.02, 1e-4, 3.0, 120, "", "cm", false, 4};
    for (int i = 1; i < argc; ++i) {
      auto nx = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-w"))       { if (i+1<argc) P.W = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-at"))      nx(P.At);
      else if (!std::strcmp(argv[i], "-re"))      nx(P.Re);
      else if (!std::strcmp(argv[i], "-u"))       nx(P.U);
      else if (!std::strcmp(argv[i], "-iw"))      nx(P.iw);
      else if (!std::strcmp(argv[i], "-m"))       nx(P.M);
      else if (!std::strcmp(argv[i], "-sigma"))   nx(P.sigma);
      else if (!std::strcmp(argv[i], "-tmax"))    nx(P.tmax);
      else if (!std::strcmp(argv[i], "-nframes")) { if (i+1<argc) P.nframes = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-dump"))    { if (i+1<argc) P.dump = argv[++i]; }
      else if (!std::strcmp(argv[i], "-op"))      { if (i+1<argc) P.op = argv[++i]; }
      else if (!std::strcmp(argv[i], "-3d"))      P.three_d = true;
      else if (!std::strcmp(argv[i], "-nz"))      { if (i+1<argc) P.nz = std::atoi(argv[++i]); }
    }
    const bool bgk = !std::strcmp(P.op, "bgk");
    if (P.three_d) {
      if (bgk) simulate<D3Q27, D3Q27, BgkOf<D3Q27>>(P);
      else     simulate<D3Q27, D3Q27, CmOf<D3Q27>>(P);
    } else {
      if (bgk) simulate<D2Q9, D2Q9, BgkOf<D2Q9>>(P);
      else     simulate<D2Q9, D2Q9, CmOf<D2Q9>>(P);
    }
  }
  Kokkos::finalize();
  return 0;
}
