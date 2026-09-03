//==============================================================================
//  STONE SKIPPING -- a spinning disc striking water at a shallow attack angle.
//
//  THE KOKKOS TWIN of GPU/src/skip.cu, and the reason it exists is this tree's
//  standing one: two implementations sharing no headers, so that agreement is
//  evidence and disagreement is a bug in one of them. The CUDA side ran the
//  first skip; until this file existed that result had no second opinion, which
//  is exactly the situation that produced three real defects when the 3-D
//  Rayleigh-Taylor case was finally diffed against its twin.
//
//  It is also the only way to run this case at all without a GPU. GPU/ compiles
//  as plain C++, but that host path is a correctness harness with no threading
//  -- one core, ~1.2 MLUPS -- so a converged skip there is hours. Kokkos threads
//  it.
//
//  THE CASE THE 6-DOF SOLVE EXISTS FOR. Every body in this tree before it was
//  either planar (a rotation about z and nothing else) or a sphere (whose chi is
//  rotation invariant, so the angular equation measures nothing). A skipping
//  stone is neither: what holds its angle of attack through the impact is
//  gyroscopic stiffness from spin about its own axis, which needs the full 6x6,
//  the rotating inertia tensor and the quaternion. It is also the case that
//  found the gyroscopic term missing from that solve -- see RigidBody3D.hpp,
//  and note that a cube could not have, since an isotropic inertia makes
//  omega x (I omega) vanish identically.
//
//  WHAT A SKIP IS, in the terms this driver prints. Bocquet and Clanet's
//  account is inviscid: the disc planes on the water it wets, the pressure on
//  that wetted area has a component along -g, and the stone rebounds if the
//  impulse of that force reverses its vertical momentum before it has sunk.
//  Three numbers decide it -- the attack angle (they report ~20 degrees as
//  optimal), the entry Froude number, and the spin, which provides no lift at
//  all and only keeps the attack angle from collapsing.
//
//  THE MEASURED SKIP on the CUDA side at D = 48, for this one to be checked
//  against (results/skip/skip_d48_attack20.dat): contact at t/t0 = 0.45,
//  deepest immersion 0.064 D, vy reverses at 1.49 while submerged, exit at 2.9,
//  27 % of the horizontal speed lost against ~20 % per skip for real stones.
//  And the attack angle NUTATES rather than collapsing -- 20 deg down to 11.5,
//  back up to 17.1, then oscillating about 13 -- which is the gyroscopic term
//  doing its job and the sharpest thing to compare between the two codes.
//
//  WHAT IS NOT MODELLED, in the order it bites.
//   * THICKNESS. A real stone is about 1:10 diameter to thickness, which at
//     D = 48 is 4.8 cells -- thinner than the interface it must deflect
//     (xi = 4) plus the chi smoothing. 1:5 is the honest floor, so this is a
//     flat plate with a rounded rim and not a lenticular stone. Below D = 40
//     even 1:5 puts the thickness under two interface widths; the CUDA banner
//     records D = 48 as the size the result was measured at.
//   * NO CONTACT LINE OR WETTING MODEL, and a skip ENDS in a trailing-edge
//     separation. The exit is not physics.
//   * Re ~ 1e3 against a real 1.75e5. Defensible rather than fatal, since the
//     lift is a planing pressure on the wetted area and not a boundary-layer
//     effect -- Bocquet and Clanet's model has no viscosity in it at all. tau
//     is the binding constraint, not Re, and the driver prints its margin.
//   * Density ratio 100, not water-against-air's 830.
//   * x IS PERIODIC, so the stone eventually wraps into its own wake. The run
//     stops before it can, rather than producing a second bounce off water it
//     has already disturbed.
//
//  Output layout is GPU/src/skip.cu's byte for byte, including the ten-field
//  _body.dat, so doc/fig/impact3d_anim.py reads either code's dumps unchanged.
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
bool arg_flag(int argc, char** argv, const char* k) {
  for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], k)) return true;
  return false;
}

// GPU/src/skip.cu's writer, so the same readers work: two int32 then nx*ny
// float32 for a plane, three int32 then nx*ny*nz for a volume, x fastest.
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
  const Index D       = Index(arg_num(argc, argv, "-d", 48));
  const double span   = arg_num(argc, argv, "-span", 8.0);
  const double aspect = arg_num(argc, argv, "-aspect", 4.0);
  const double Fr     = arg_num(argc, argv, "-fr", 5.0);
  const double Re     = arg_num(argc, argv, "-re", 500.0);
  const double We     = arg_num(argc, argv, "-we", 1000.0);
  const double Pe     = arg_num(argc, argv, "-pe", 128.0);
  const double ratio  = arg_num(argc, argv, "-ratio", 100.0);
  const double chib   = arg_num(argc, argv, "-chib", 2.6);
  const double U      = arg_num(argc, argv, "-u", 0.05);
  const double iw     = arg_num(argc, argv, "-iw", 4.0);
  const double tmax   = arg_num(argc, argv, "-tmax", 6.0);
  const double thick  = arg_num(argc, argv, "-thick", 0.2);
  const double attack = arg_num(argc, argv, "-attack", 20.0);
  const double descent = arg_num(argc, argv, "-descent", 15.0);
  const double spin_i = arg_num(argc, argv, "-spin", 0.64);
  const double h0     = arg_num(argc, argv, "-h0", 1.0);
  const double smooth = arg_num(argc, argv, "-smooth", 1.0);
  const int frames    = int(arg_num(argc, argv, "-frames", 60));
  const char* dump    = arg_str(argc, argv, "-dump", "");
  const bool volume   = arg_flag(argc, argv, "-vol");
  const bool norot    = arg_flag(argc, argv, "-norot");
  const bool nospin   = arg_flag(argc, argv, "-nospin");
  const double spin   = nospin ? 0.0 : spin_i;

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const Index nx = Index(span * double(D) + 0.5);
    const Index ny = Index(aspect * double(D) + 0.5) / 2 * 2;
    const Index nz = 3 * D;
    const double R  = 0.5 * double(D);
    const double hy = 0.5 * thick * double(D);      // HALF thickness
    const double g  = U * U / (Fr * Fr * double(D));
    const double nu = U * double(D) / Re;
    const double rho_l = 1.0, rho_h = ratio;
    const double sigma = rho_h * U * U * double(D) / We;
    const double M     = U * double(D) / Pe;
    const double t_ref = double(D) / U;             // one diameter travelled
    const std::size_t nsteps = std::size_t(tmax * t_ref);
    const double ysurf = 0.5 * double(ny);
    const double att = attack * PI / 180.0;
    const double des = descent * PI / 180.0;
    const double Omega = spin * U / R;

    std::printf("Stone skipping   6-DOF disc, Kokkos   %s   precision %s\n",
                ExecSpace::name(), precision_name());
    std::printf("  %d x %d x %d   D %d   thickness %.1f cells (1:%.0f)   "
                "rho_b/rho_H %g\n", int(nx), int(ny), int(nz), int(D),
                2.0 * hy, 1.0 / thick, chib);
    std::printf("  Fr %g   Re %g   We %g   rho_H/rho_L %g   nu %.6e   "
                "sigma %.6e   g %.6e   U %.4f   xi %g\n",
                Fr, Re, We, ratio, nu, sigma, g, U, iw);
    std::printf("  tau %.6f   (stability floor 0.5; margin %.2e)\n",
                3.0 * nu + 0.5, 3.0 * nu);
    std::printf("  Pe %g   M %.6e   phase omega %.6f   (ceiling 2; "
                "margin %.2e)\n", Pe, M, 1.0 / (M / (1.0 / 3.0) + 0.5),
                2.0 - 1.0 / (M / (1.0 / 3.0) + 0.5));
    // The spin as a RIM SPEED, because that is the number with a
    // compressibility limit on it. A real stone at 90 rad/s, D = 5 cm and
    // U = 3.5 m/s has a rim speed of 0.64 U -- which is the default -- and in
    // lattice units that is 0.032, well under the 0.05 the case already runs
    // at. Matching the RATIO rather than the rate is what makes it possible.
    std::printf("  attack %.1f deg   descent %.1f deg   spin %.3f U at the rim "
                "-> Omega %.4e rad/step (%.2f rev over the run)\n",
                attack, descent, spin, Omega,
                Omega * double(nsteps) / (2.0 * PI));
    std::printf("  rotation %s   spin %s\n", norot ? "HELD" : "free",
                nospin ? "ZERO (the control)" : "on");
    std::printf("  t_ref %.1f steps   %zu steps to t/t0 = %g\n",
                t_ref, nsteps, tmax);
    std::fflush(stdout);
    if (3.0 * nu <= 0.0) {
      std::printf("  tau AT the floor -- refusing\n");
      Kokkos::finalize();  return 1;
    }

    Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);
    const Index hxp = d.hx, hyp = d.hy, hzp = d.hz;

    PColl pc;
    pc.omega = PColl::omega_from_mobility(Real(M));
    pc.width = Real(iw);
    PhaseFieldSolver<PL, EsotericPull<PL>, PColl> pf(d, pc);

    const Real ysr = Real(ysurf), iwr = Real(iw);
    pf.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real y = Real(py - hyp);
      return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (ysr - y) / iwr));
    });
    ViscousInterfaceForce<FL> vf(d);
    FColl fc;

    PenalisedBody<FL, Disc> body(d);
    body.shape.R = Real(R);
    body.shape.hy = Real(hy);
    body.shape.smooth = Real(smooth);
    // A rotation about +z by the attack angle tips the symmetry axis from +y
    // toward -x, which RAISES the leading edge at +x. That sign is pinned in
    // tests/test_rigid3d block 8 rather than reasoned about here: backwards, it
    // buries the leading edge and the stone dives -- a plausible result that is
    // the wrong experiment.
    body.shape.set_orientation(
        Quat::from_axis_angle(Real(0), Real(0), Real(1), Real(att)));
    body.shape.cx = Real(1.5 * double(D));
    body.shape.cz = Real(0.5 * double(nz));
    // Clear of the surface by h0 half-thicknesses PLUS the vertical reach of
    // the TILTED disc, R sin(attack) + hy cos(attack) -- not hy. A 20-degree
    // disc of radius R hangs a long way below its own centre, and using hy
    // alone starts it already cut by the water.
    const double reach_y = R * std::sin(att) + hy * std::cos(att);
    body.shape.cy = Real(ysurf + reach_y + h0 * hy);
    body.vx = Real(U * std::cos(des));
    body.vy = Real(-U * std::sin(des));
    body.vz = Real(0);
    // Spin about the body's OWN axis, read from the pose rather than written
    // out as (-sin a, cos a, 0), so it stays right if the pose changes.
    {
      Real ax, ay, az;
      body.shape.axis(ax, ay, az);
      body.wx = Real(Omega * double(ax));
      body.wy = Real(Omega * double(ay));
      body.wz = Real(Omega * double(az));
    }
    body.by = Real(-g);
    body.free_translation = true;
    body.free_rotation = !norot;

    // FILL fc BEFORE CONSTRUCTING THE SOLVER: FluidSolver COPIES the collision
    // operator, so a view assigned afterwards never reaches the copy the
    // kernels use, and the first macroscopic pass segfaults.
    fc.phi = pf.phi();
    fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
    fc.Lap = pf.laplacian();
    fc.Vx = vf.x();  fc.Vy = vf.y();  fc.Vz = vf.z();
    fc.Ex = body.x(); fc.Ey = body.y(); fc.Ez = body.z();
    fc.rho_L = Real(rho_l);      fc.rho_H = Real(rho_h);
    // KINEMATIC viscosity matched across the ratio, so mu scales with rho.
    fc.mu_L  = Real(rho_l * nu); fc.mu_H  = Real(rho_h * nu);
    fc.kappa = FColl::kappa_from_sigma(Real(sigma), Real(iw));
    fc.beta  = FColl::beta_from_sigma(Real(sigma), Real(iw));
    fc.by    = Real(-g);

    FluidSolver<FL, EsotericPull<FL>, FColl> fl(d, fc);

    // The hydrostatic seed, integrating the ACTUAL tanh density profile rather
    // than a sharp step: the closed form is free and the alternative is an
    // acoustic transient that has to damp before the impact means anything.
    // The gauge is p = 0 AT THE SURFACE -- the populations carry p~, so seeding
    // a pressure rather than a density is not a stylistic choice.
    auto phiv = pf.phi();
    const Real rl = Real(rho_l), rh = Real(rho_h), gr = Real(g);
    fl.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real dep = ysr - Real(py - hyp);
      const Real a = Real(2) * dep / iwr;
      const Real aa = (a < Real(0) ? -a : a);
      const Real lnch = aa + Kokkos::log(Real(1) + Kokkos::exp(Real(-2) * aa))
                      - Real(0.6931471805599453);
      const Real W = rl * dep
                   + (rh - rl) * Real(0.5) * (dep + Real(0.5) * iwr * lnch);
      const Real r = rl + phiv(n) * (rh - rl);
      return FlowState{gr * W / (r / Real(3)), Real(0), Real(0), Real(0)};
    });

    // GEOMETRY AFTER THE SEED, and both solvers get it: initialize_field seeds
    // FlowState{} -- hence p~ = 0 -- at any node not flagged Fluid, so flagging
    // first writes a zero pressure into the tank floor against a hydrostatic
    // neighbour. Measured elsewhere in this tree as a spurious 1.7e-1 velocity.
    fl.set_geometry([&](Index, Index y, Index) -> CellType {
      return (y == 0 || y == ny - 1) ? Solid : Fluid;
    });
    pf.set_geometry([&](Index, Index y, Index) -> PhaseCell {
      return (y == 0 || y == ny - 1) ? PhaseWall : PhaseBulk;
    });
    pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
    vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
    vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());
    body.set_velocity(fl.ux(), fl.uy(), fl.uz());
    // set_uniform_density6, NOT set_uniform_density: the 2-D call fills the
    // mass and leaves the tensor at ZERO, which makes the angular half of the
    // 6x6 singular and produces a plausible tumble rather than a failure.
    body.set_uniform_density6(Real(chib * rho_h));

    {
      const double v = double(body.penalised_volume());
      const double exact = PI * R * R * 2.0 * hy;
      std::printf("  disc volume from chi %.1f   pi R^2 (2h) = %.1f   "
                  "(%+.2f %%)\n", v, exact, 100.0 * (v - exact) / exact);
      const double m = double(body.mass);
      std::printf("  body mass %.1f   displaced water %.1f   "
                  "net weight %+.3e\n", m, rho_h * exact,
                  (m - rho_h * exact) * g);
      // The closed-form cylinder, as a check on the whole measurement path.
      // The smoothing raises both by a computed amount -- +0.71 % axial and
      // +1.22 % diametral at R = 24, hy = 4.8, smooth = 1, from quadrature --
      // so a per cent or two high is correct and a factor is not.
      std::printf("  I_body axial %.4e (m R^2/2 = %.4e)   diametral %.4e "
                  "(m(R^2/4+h^2/3) = %.4e)\n",
                  double(body.inertia_body(1, 1)), m * R * R / 2.0,
                  double(body.inertia_body(0, 0)),
                  m * (R * R / 4.0 + hy * hy / 3.0));
    }

    auto dens_of = KOKKOS_LAMBDA(Index n) {
      return rl + phiv(n) * (rh - rl);
    };

    const std::size_t every =
        (frames > 1) ? std::max<std::size_t>(1, nsteps / std::size_t(frames - 1))
                     : nsteps + 1;
    int nframe = 0;
    bool finite = true;
    const double fnorm = rho_h * U * U * double(D) * double(D);
    const double tnorm = fnorm * double(D);
    double vy_min = 0, vy_max = -1e30, y_low = 1e30;
    bool rebounded = false;
    double t_rebound = -1;

    // clear/D is the LOWEST POINT of the tilted disc above the undisturbed
    // surface, which is what decides whether it is in the water -- the centre
    // height is misleading for a plate at 20 degrees. attack is read back from
    // the pose every print, because whether it HOLDS is the question, and Tz
    // rather than Ty because the attack angle is a rotation about z.
    std::printf("\n  t/t0     x/D    clear/D    vx/U     vy/U    attack   "
                "|w|D/U   Fy/(rU^2D^2)  Tz/(rU^2D^3)   m_fluid\n");

    for (std::size_t step = 0; step <= nsteps; ++step) {
      pf.refresh();
      fl.compute_macroscopic();
      vf.refresh(fc);
      const auto Rx = body.refresh6(dens_of);

      Real axr, ayr, azr;
      body.shape.axis(axr, ayr, azr);
      const double a_now = std::acos(std::fmax(-1.0, std::fmin(1.0,
                              double(ayr)))) * 180.0 / PI;
      const double reach_now = R * std::sin(a_now * PI / 180.0)
                             + hy * std::cos(a_now * PI / 180.0);
      const double clear = (double(body.shape.cy) - reach_now - ysurf)
                         / double(D);
      if (clear < y_low) y_low = clear;
      const double vyU = double(body.vy) / U;
      if (vyU < vy_min) vy_min = vyU;
      if (vyU > vy_max) vy_max = vyU;
      // A REBOUND IS A SIGN CHANGE IN vy WHILE IN CONTACT, not merely vy > 0:
      // a stone still on its way down has vy < 0 throughout, and one lifted by
      // the initial transient alone would show vy > 0 before it ever reached
      // the surface.
      if (!rebounded && vyU > 0.0 && clear < 0.5 && step > 10) {
        rebounded = true;  t_rebound = double(step) / t_ref;
      }

      if (step % (nsteps / 24 ? nsteps / 24 : 1) == 0) {
        const double wmag = std::sqrt(double(body.wx) * double(body.wx)
                                    + double(body.wy) * double(body.wy)
                                    + double(body.wz) * double(body.wz))
                          * double(D) / U;
        std::printf("  %5.2f  %6.2f  %+8.3f  %+7.4f  %+7.4f  %7.2f  %.3e  "
                    "%+.4e   %+.4e   %.4e\n",
                    double(step) / t_ref, double(body.shape.cx) / double(D),
                    clear, double(body.vx) / U, vyU, a_now, wmag,
                    double(Rx.fy) / fnorm, double(Rx.tz) / tnorm,
                    double(Rx.fluid_mass));
        std::fflush(stdout);
      }

      if (*dump && step % every == 0) {
        pf.compute_field();
        auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
        char p[256];
        std::snprintf(p, sizeof p, "%s_%04d.bin", dump, nframe);
        write_field(p, hp, d, nx, ny, nz, hxp, hyp, hzp, false);
        const Disc b = body.shape;
        {
          std::vector<float> buf(std::size_t(nx) * std::size_t(ny));
          std::snprintf(p, sizeof p, "%s_chi_%04d.bin", dump, nframe);
          if (std::FILE* f = std::fopen(p, "wb")) {
            const std::int32_t hdr[2] = {std::int32_t(nx), std::int32_t(ny)};
            std::fwrite(hdr, sizeof(std::int32_t), 2, f);
            for (Index y = 0; y < ny; ++y)
              for (Index x = 0; x < nx; ++x)
                buf[std::size_t(y) * std::size_t(nx) + std::size_t(x)] =
                    float(b.chi(Real(x), Real(y), Real(nz / 2)));
            std::fwrite(buf.data(), sizeof(float), buf.size(), f);
            std::fclose(f);
          }
        }
        // TEN numbers, matching GPU/src/skip.cu exactly: centre, radius, half
        // thickness, quaternion. A reader tells a disc from a sphere's five and
        // a box's eleven by the field count, so a renderer written for one
        // skips the others rather than drawing a wrong body.
        std::snprintf(p, sizeof p, "%s_body.dat", dump);
        if (std::FILE* f = std::fopen(p, nframe == 0 ? "wb" : "ab")) {
          std::fprintf(f, "%d %.4f %.4f %.4f %.4f %.4f %.6f %.6f %.6f %.6f\n",
                       nframe, double(b.cx), double(b.cy), double(b.cz),
                       double(b.R), double(b.hy), double(b.q.w), double(b.q.x),
                       double(b.q.y), double(b.q.z));
          std::fclose(f);
        }
        if (volume) {
          std::snprintf(p, sizeof p, "%s_vol_%04d.bin", dump, nframe);
          write_field(p, hp, d, nx, ny, nz, hxp, hyp, hzp, true);
        }
        ++nframe;
      }

      fl.step(true);
      pf.step();
      body.advance6();

      const double cxv = double(body.shape.cx), cyv = double(body.shape.cy);
      if (!(cxv == cxv) || !(cyv == cyv)
          || !(double(body.vy) == double(body.vy))) {
        std::printf("  NON-FINITE body state at step %zu -- the run diverged, "
                    "it did not travel\n", step);
        finite = false;  break;
      }
      if (!(cyv > 1.0 && cyv < double(ny - 1))) {
        std::printf("  the stone left the tank in y at step %zu (cy %.2f)\n",
                    step, cyv);
        finite = false;  break;
      }
      // x is PERIODIC, so the stone does not leave -- it wraps into its own
      // wake, which is not a skip off undisturbed water.
      if (cxv > double(nx) - 1.5 * double(D)) {
        std::printf("  the stone reached the end of the tank at step %zu "
                    "(x/D %.2f of %.2f) -- x is periodic, so it would wrap "
                    "into its own wake\n", step, cxv / double(D),
                    double(nx) / double(D));
        finite = false;  break;
      }
    }

    std::printf("\n  %s\n", finite ? "finite throughout" : "STOPPED EARLY");
    std::printf("  lowest clearance %+.3f D   vy from %+.4f U to %+.4f U\n",
                y_low, vy_min, vy_max);
    if (rebounded)
      std::printf("  SKIPPED: vy changed sign at t/t0 = %.2f while in "
                  "contact\n", t_rebound);
    else
      std::printf("  NO SKIP: vy never became positive while in contact\n");
    if (*dump) std::printf("  %d frame(s) written to %s\n", nframe, dump);
    if (!finite) status = 1;
  }
  Kokkos::finalize();
  return status;
}
