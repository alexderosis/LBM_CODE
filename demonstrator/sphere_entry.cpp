//==============================================================================
//  A SPHERE falling into a free water surface -- the 3-D case, on Kokkos.
//
//  WHY THIS EXISTS WHEN GPU/src/impact.cu ALREADY DOES IT. Because that was the
//  only one. This tree keeps two independent implementations of the physics so
//  that agreement between them is evidence and disagreement is a bug in one of
//  them, and until now the 3-D body lived on the CUDA side alone, with nothing
//  to check it against. That is not a hypothetical worry: diffing the two codes
//  on the 3-D Rayleigh-Taylor case found three real defects in one morning -- a
//  missing phase-field geometry, an inverted coupling order, and a solid node
//  seeded at p~ = 0 against a hydrostatic neighbour. A 3-D body with no second
//  opinion is exactly the situation that produced them.
//
//  It is also the only way to run a 3-D body on THIS machine at a useful size.
//  GPU/ compiles as plain C++, but that host path is a correctness harness with
//  no threading -- one core, ~1.2 MLUPS -- so a converged sphere there is hours.
//  Kokkos threads it.
//
//  WHAT IS AND IS NOT 3-D. The sphere is a real sphere: chi depends on all three
//  coordinates and its volume integrates to 4 pi R^3 / 3, which the driver
//  checks at startup rather than assuming. The DYNAMICS are three translations
//  with rotation switched off. For a sphere entering on its axis that is exact
//  and not an approximation -- there is no torque about any axis, and chi is
//  invariant under rotation anyway, so the roll equation would measure and
//  change nothing. It is NOT a general 3-D rigid body: no quaternion, no
//  rotating inertia tensor. A tilted cube is not available here.
//
//  THE PHASES. phi = 1 is water, phi = 0 is air, the free surface starts flat,
//  and both share a KINEMATIC viscosity: matching the dynamic one across a
//  ratio of 100 leaves the heavy phase at nu/100 and drives omega to 1.994
//  against a limit of 2, which reads as the model failing when it is the setup.
//
//  THE DENSITY RATIO IS 100, NOT 1000. Water against air is about 830 by mass.
//  The conservative Allen-Cahn phase field here is documented to ~100 and that
//  is what is used, so this is a sphere entering a liquid 100x its surroundings
//  -- right for the cavity and the splash, wrong for the air.
//
//  READ THE CAVITY, NOT THE MENISCUS: there is no contact-line model, so the
//  angle at which the surface meets the sphere is not physics.
//
//  Output matches GPU/src/impact.cu byte for byte in layout, so
//  doc/fig/rt2d_anim.py and doc/fig/impact3d_anim.py read either code's dumps
//  with no argument changes -- which is the point of having two.
//==============================================================================
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/PhaseFieldCentralMoments.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/PenalisedBody.hpp"
#include "solver/PhaseFieldSolver.hpp"
#include "solver/ViscousInterfaceForce.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

using FL = D3Q27;
using PL = D3Q27;
using FColl = MultiphaseCentralMoments<FL>;
using PColl = PhaseFieldCentralMoments<PL>;

constexpr double PI = 3.14159265358979323846;

const char* arg_str(int argc, char** argv, const char* k, const char* d) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], k)) return argv[i + 1];
  return d;
}
double arg_num(int argc, char** argv, const char* k, double d) {
  const char* s = arg_str(argc, argv, k, nullptr);
  return s ? std::atof(s) : d;
}

// One writer for both products -- the mid-z plane and the whole volume -- so a
// reader that works on one works on the other. Layout is GPU/src/impact.cu's:
// two int32 then nx*ny float32 for a plane, three int32 then nx*ny*nz for a
// volume, x fastest in both.
template <class HostF>
void write_field(const char* path, const HostF& h, const Domain& d,
                 Index nx, Index ny, Index nz, Index hx, Index hy, Index hz,
                 bool vol) {
  std::FILE* f = std::fopen(path, "wb");
  if (!f) return;
  if (vol) {
    const std::int32_t hdr[3] = {std::int32_t(nx), std::int32_t(ny),
                                 std::int32_t(nz)};
    std::fwrite(hdr, sizeof(std::int32_t), 3, f);
  } else {
    const std::int32_t hdr[2] = {std::int32_t(nx), std::int32_t(ny)};
    std::fwrite(hdr, sizeof(std::int32_t), 2, f);
  }
  const Index z0 = vol ? Index(0) : nz / 2;
  const Index z1 = vol ? nz : (z0 + 1);
  std::vector<float> buf(std::size_t(nx) * std::size_t(ny));
  for (Index z = z0; z < z1; ++z) {
    for (Index y = 0; y < ny; ++y)
      for (Index x = 0; x < nx; ++x)
        buf[std::size_t(y) * std::size_t(nx) + std::size_t(x)] =
            float(h(d.id(x + hx, y + hy, z + hz)));
    std::fwrite(buf.data(), sizeof(float), buf.size(), f);
  }
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  const Index D      = Index(arg_num(argc, argv, "-d", 24));
  const double span  = arg_num(argc, argv, "-span", 5.0);
  const double aspect = arg_num(argc, argv, "-aspect", 2.0);
  const double Fr    = arg_num(argc, argv, "-fr", 2.0);
  const double Re    = arg_num(argc, argv, "-re", 256.0);
  const double We    = arg_num(argc, argv, "-we", 200.0);
  const double Pe    = arg_num(argc, argv, "-pe", 128.0);
  const double ratio = arg_num(argc, argv, "-ratio", 100.0);
  const double chib  = arg_num(argc, argv, "-chib", 2.0);
  const double U     = arg_num(argc, argv, "-u", 0.04);
  const double iw    = arg_num(argc, argv, "-iw", 4.0);
  const double tmax  = arg_num(argc, argv, "-tmax", 6.0);
  const int frames   = int(arg_num(argc, argv, "-frames", 48));
  const char* dump   = arg_str(argc, argv, "-dump", "");
  const bool volume  = [&] {
    for (int i = 1; i < argc; ++i)
      if (!std::strcmp(argv[i], "-vol")) return true;
    return false;
  }();

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const Index nx = Index(span * double(D) + 0.5);
    const Index nz = nx;
    const Index ny = Index(aspect * span * double(D) + 0.5) / 2 * 2;
    const double R     = 0.5 * double(D);
    const double g     = U * U / (Fr * Fr * double(D));
    const double nu    = U * double(D) / Re;
    const double rho_l = 1.0, rho_h = ratio;
    const double sigma = rho_h * U * U * double(D) / We;
    const double M     = U * double(D) / Pe;
    const double t_ref = double(D) / U;              // one diameter travelled
    const std::size_t nsteps = std::size_t(tmax * t_ref);
    const double ysurf = 0.55 * double(ny);

    std::printf("Sphere entering water   3-D, Kokkos   %s   precision %s\n",
                ExecSpace::name(), precision_name());
    std::printf("  %d x %d x %d   D %d   Fr %g   Re %g   We %g   "
                "rho_H/rho_L %g   rho_b/rho_H %g\n",
                int(nx), int(ny), int(nz), int(D), Fr, Re, We, ratio, chib);
    std::printf("  nu %.6e   sigma %.6e   g %.6e   U %.4f   xi %g\n",
                nu, sigma, g, U, iw);
    // BOTH FLOORS, PRINTED, because Re is bought with viscosity and the
    // mobility is bought at the other end. tau -> 1/2 is where the fluid stops;
    // a SMALL mobility drives the phase relaxation toward 2, which is where
    // that one stops, and M = U D / Pe means a large Pe on a small D walks into
    // it. Reading both back before running is this tree's standing rule.
    std::printf("  tau %.6f (floor 0.5; margin %.2e)   phase omega %.6f "
                "(ceiling 2; margin %.2e)\n",
                3.0 * nu + 0.5, 3.0 * nu,
                1.0 / (M / (1.0 / 3.0) + 0.5),
                2.0 - 1.0 / (M / (1.0 / 3.0) + 0.5));
    std::printf("  t_ref %.1f steps   %zu steps to t/t0 = %g\n\n",
                t_ref, nsteps, tmax);
    std::fflush(stdout);
    Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);
    const Index hx = d.hx, hy = d.hy, hz = d.hz;
    PColl pc;
    pc.omega = PColl::omega_from_mobility(Real(M));
    pc.width = Real(iw);
    PhaseFieldSolver<PL, EsotericPull<PL>, PColl> pf(d, pc);

    const Real ysr = Real(ysurf), iwr = Real(iw);
    pf.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real y = Real(py - hy);
      // phi = 1 below the surface, 0 above, through the tanh profile.
      return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (ysr - y) / iwr));
    });
    ViscousInterfaceForce<FL> vf(d);
    // No single omega on this operator: it takes the two DYNAMIC viscosities
    // and derives a per-node rate from phi, which is what carries the viscosity
    // contrast across the interface.
    FColl fc;
    PenalisedBody<FL, Sphere> body(d);
    body.shape.cx = Real(0.5 * double(nx));
    body.shape.cz = Real(0.5 * double(nz));
    body.shape.cy = Real(ysurf + R);        // tangent to the undisturbed surface
    body.shape.R  = Real(R);
    body.shape.smooth = Real(1.5);
    body.vx = Real(0);  body.vz = Real(0);
    body.vy = Real(-U);                     // the impact speed, imposed
    body.omega = Real(0);
    body.by = Real(-g);                     // the SAME vector the collision gets
    body.free_translation = true;
    // OFF, AND NOT AS A SIMPLIFICATION: a sphere's chi is invariant under
    // rotation, so the roll equation measures nothing and would change nothing.
    body.free_rotation = false;
    // FILL fc BEFORE CONSTRUCTING THE SOLVER. FluidSolver COPIES the collision
    // operator, so a view assigned to fc afterwards never reaches the copy the
    // kernels use -- they read a default-constructed View and the first
    // macroscopic pass segfaults. enan_rt.cpp and water_entry.cpp both fill it
    // first; getting this backwards is how this driver crashed on its first
    // run, and the failure is a hard one rather than a wrong answer only
    // because an unassigned View has no data pointer.
    fc.phi = pf.phi();
    fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
    fc.Lap = pf.laplacian();
    fc.Vx = vf.x();  fc.Vy = vf.y();  fc.Vz = vf.z();
    fc.Ex = body.x(); fc.Ey = body.y(); fc.Ez = body.z();
    fc.rho_L = Real(rho_l);      fc.rho_H = Real(rho_h);
    fc.mu_L  = Real(rho_l * nu); fc.mu_H  = Real(rho_h * nu);
    fc.kappa = FColl::kappa_from_sigma(Real(sigma), Real(iw));
    fc.beta  = FColl::beta_from_sigma(Real(sigma), Real(iw));
    fc.by    = Real(-g);

    FluidSolver<FL, EsotericPull<FL>, FColl> fl(d, fc);

    // THE HYDROSTATIC SEED, integrating the ACTUAL tanh density profile rather
    // than a sharp step: the closed form is free and the alternative is an
    // acoustic transient that has to damp before the impact means anything. The
    // gauge is p = 0 AT THE SURFACE -- on this path the populations carry
    // p~ = p/(rho cs^2), so seeding a pressure rather than a density is not a
    // stylistic choice, and a wrong gauge is the difference between a scheme
    // that works and one that does not.
    auto phiv = pf.phi();
    const Real rl = Real(rho_l), rh = Real(rho_h), gr = Real(g);
    fl.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real dep = ysr - Real(py - hy);            // depth, + below surface
      const Real a = Real(2) * dep / iwr;
      const Real aa = (a < Real(0) ? -a : a);
      const Real lnch = aa + Kokkos::log(Real(1) + Kokkos::exp(Real(-2) * aa))
                      - Real(0.6931471805599453);
      const Real W = rl * dep
                   + (rh - rl) * Real(0.5) * (dep + Real(0.5) * iwr * lnch);
      const Real r = rl + phiv(n) * (rh - rl);
      return FlowState{gr * W / (r / Real(3)), Real(0), Real(0), Real(0)};
    });

    // GEOMETRY AFTER THE SEED, and both solvers get it. initialize_field seeds
    // FlowState{} -- rho = 0, hence p~ = 0 -- at any node not flagged Fluid, so
    // flagging first writes a zero pressure into the tank floor against a
    // hydrostatic neighbour. Measured in validation/enan_rt.cpp: a spurious
    // 1.7e-1 velocity, Ma 0.29, at the adjacent node in quiescent fluid. The
    // phase field needs the same planes as zero-flux walls.
    fl.set_geometry([&](Index, Index y, Index) -> CellType {
      return (y == 0 || y == ny - 1) ? Solid : Fluid;
    });
    pf.set_geometry([&](Index, Index y, Index) -> PhaseCell {
      return (y == 0 || y == ny - 1) ? PhaseWall : PhaseBulk;
    });
    pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
    vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
    vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());
    // THREE components, which is what switches the z limb on. The two-argument
    // form leaves it off and every prism case keeps its old arithmetic.
    body.set_velocity(fl.ux(), fl.uy(), fl.uz());
    body.set_uniform_density(Real(chib * rho_h));

    // THE VOLUME IS CHECKED, NOT ASSUMED. chi is smoothed over 1.5 cells, so
    // the penalised sphere is slightly larger than the nominal one and the
    // integral is the honest measure of the body the force acts on. A gross
    // disagreement means the 3-D indicator or the prism/sphere culling is
    // wrong, and every number downstream would be wrong with it.
    {
      const double vchi = double(body.penalised_area());
      const double exact = 4.0 / 3.0 * PI * R * R * R;
      std::printf("  sphere volume from chi %.1f   4 pi R^3/3 = %.1f  (%+.2f %%)"
                  "\n", vchi, exact, 100.0 * (vchi - exact) / exact);
      std::printf("  body mass %.4e   displaced water %.4e   net weight %+.3e\n",
                  double(body.mass), rho_h * exact,
                  (double(body.mass) - rho_h * exact) * g);
    }

    auto dens_of = KOKKOS_LAMBDA(Index n) {
      return rl + phiv(n) * (rh - rl);
    };

    const std::size_t every =
        (frames > 1) ? std::max<std::size_t>(1, nsteps / std::size_t(frames - 1))
                     : nsteps + 1;
    int nframe = 0;
    bool finite = true;

    // The last column is the AXISYMMETRY CHECK: an on-axis entry has no lateral
    // force, so vx and vz must stay at zero. Anything growing there is either a
    // real symmetry break or a bug in the z limb, and it costs nothing to watch.
    std::printf("\n  t/t0    depth/R     v/U      Fy/(rho U^2 D^2)   m_fluid"
                "     |v_lat|/U\n");
    const double fnorm = rho_h * U * U * double(D) * double(D);

    for (std::size_t step = 0; step <= nsteps; ++step) {
      pf.refresh();                     // phi(t) and grad phi(t)
      fl.compute_macroscopic();         // u(t), so the body sees it fresh
      vf.refresh(fc);
      const auto Rx = body.refresh(dens_of);   // after macroscopic, before step

      if (step % (nsteps / 24 ? nsteps / 24 : 1) == 0) {
        const double depth = (ysurf - double(body.shape.cy)) / R;
        const double vlat = std::sqrt(double(body.vx) * double(body.vx)
                                    + double(body.vz) * double(body.vz)) / U;
        std::printf("  %5.2f   %8.3f   %+7.4f   %+.4e      %9.1f   %.2e\n",
                    double(step) / t_ref, depth, double(body.vy) / U,
                    double(Rx.fy) / fnorm, double(Rx.fluid_mass), vlat);
        std::fflush(stdout);
      }

      if (*dump && step % every == 0) {
        pf.compute_field();
        auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
        char p[256];
        std::snprintf(p, sizeof p, "%s_%04d.bin", dump, nframe);
        write_field(p, hp, d, nx, ny, nz, hx, hy, hz, false);
        // The SOLID INDICATOR beside it: phi alone cannot tell a sphere from
        // the air around it, so a submerged body and its cavity plot as one
        // blob and a buoyant one is invisible in air that is already pale.
        const Sphere b = body.shape;
        auto chi_of = [&](Index x, Index y, Index z) {
          return b.chi(Real(x), Real(y), Real(z));
        };
        {
          std::vector<float> buf(std::size_t(nx) * std::size_t(ny));
          std::snprintf(p, sizeof p, "%s_chi_%04d.bin", dump, nframe);
          if (std::FILE* f = std::fopen(p, "wb")) {
            const std::int32_t hdr[2] = {std::int32_t(nx), std::int32_t(ny)};
            std::fwrite(hdr, sizeof(std::int32_t), 2, f);
            for (Index y = 0; y < ny; ++y)
              for (Index x = 0; x < nx; ++x)
                buf[std::size_t(y) * std::size_t(nx) + std::size_t(x)] =
                    float(chi_of(x, y, nz / 2));
            std::fwrite(buf.data(), sizeof(float), buf.size(), f);
            std::fclose(f);
          }
        }
        // The body's pose as four numbers, not a chi volume: chi is analytic,
        // so a renderer rebuilds the sphere exactly from these.
        std::snprintf(p, sizeof p, "%s_body.dat", dump);
        if (std::FILE* f = std::fopen(p, nframe == 0 ? "wb" : "ab")) {
          std::fprintf(f, "%d %.4f %.4f %.4f %.4f\n", nframe,
                       double(body.shape.cx), double(body.shape.cy),
                       double(body.shape.cz), double(body.shape.R));
          std::fclose(f);
        }
        if (volume) {
          std::snprintf(p, sizeof p, "%s_vol_%04d.bin", dump, nframe);
          write_field(p, hp, d, nx, ny, nz, hx, hy, hz, true);
        }
        ++nframe;
      }

      fl.step(true);
      pf.step();
      body.advance();

      const double cy = double(body.shape.cy);
      if (!(cy == cy) || !(double(body.vy) == double(body.vy))) {
        std::printf("  NON-FINITE body state at step %zu -- the run diverged, "
                    "it did not travel\n", step);
        finite = false;  break;
      }
      if (!(cy > 1.0 && cy < double(ny - 1))) {
        std::printf("  the sphere reached the domain edge at step %zu "
                    "(cy %.2f)\n", step, cy);
        finite = false;  break;
      }
    }

    std::printf("\n  %s   %d frame(s) written%s%s\n",
                finite ? "finite throughout" : "STOPPED EARLY",
                nframe, *dump ? " to " : "", dump);
    if (!finite) status = 1;
  }
  Kokkos::finalize();
  return status;
}
