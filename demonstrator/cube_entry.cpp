//==============================================================================
//  A CUBE, RELEASED CORNER-DOWN, falling into a free water surface.
//
//  WHY A CUBE AND NOT ANOTHER SPHERE. demonstrator/sphere_entry.cpp runs three
//  translations with rotation switched OFF, and for a sphere on its axis that is
//  exact rather than a simplification: chi is invariant under rotation, so the
//  angular equation measures nothing and would change nothing. Nothing in this
//  tree exercised the other three degrees of freedom. A cube held corner-down
//  does: its indicator depends on the orientation, its inertia tensor is carried
//  in the body frame and rotated out each step, and the pose is a quaternion.
//  Every line of the 6x6 solve is on the critical path here and on none of it in
//  the sphere case.
//
//  THE POSE IS AN EQUILIBRIUM IN THE CONTINUUM, AND NOT QUITE ONE ON A LATTICE.
//  A cube whose body diagonal is exactly along gravity has three-fold symmetry
//  about that diagonal, and both the hydrodynamic torque and the hydrostatic
//  couple -(S x g) vanish identically under it. So the continuum problem has no
//  reason to turn.
//
//  THE DISCRETE PROBLEM DOES, and the reason is worth stating rather than
//  discovering: a three-fold axis cannot be aligned with D3Q27, whose symmetry
//  about y is four-fold. The three faces meeting at the lower corner therefore
//  sit at azimuths the grid does not treat alike, and the measured torque at
//  -tilt 0 is not zero -- it is a discretisation residual. At D = 12 it turns
//  the diagonal by 0.016 deg over t/t0 = 1.8, which is the number to beat: it
//  should shrink with D, and anything of order the tilt itself would mean the
//  6-DOF path is manufacturing torque rather than measuring it. That is what
//  the -tilt 0 run is for, and it is a bound rather than an exact zero.
//
//  The interesting run is -tilt a few degrees. Then the question is whether the
//  equilibrium restores or runs away, which is a genuinely 3-D question with no
//  answer available from a 2-D section: the restoring couple, if there is one,
//  acts about a horizontal axis that is not a symmetry axis of the tilted body.
//  Both runs are printed with the same diagnostics so the pair is a comparison
//  rather than two separate stories.
//
//  THE MOBILITY IS THE BINDING CONSTRAINT, NOT THE BODY. sphere_entry.cpp's
//  Pe = 128 diverges here, and it does so through the PHASE FIELD rather than
//  through the solve: at D = 12 it puts the phase relaxation rate at 1.956
//  against a ceiling of 2, the phi undershoot then grows monotonically
//  (-4e-3 at t/t0 = 0.4, -4.3e-2 at 0.8) and the field explodes at 0.84 while
//  the body's own numbers are still smooth. Measured with rotation both free
//  and held, failing at the identical step, so it is not the angular limb. The
//  default here is Pe = 32, rate 1.835, margin 0.165 -- a cube's corner has
//  steeper gradients than a sphere of the same size and needs the room. The
//  whole-field scan in the step table is what makes this visible, and it is
//  printed for the same reason it was added: a body diagnostic cannot tell a
//  diverging fluid from a diverging solve.
//
//  WHAT IS NOT MODELLED, and it matters for reading the result. There is no
//  contact-line model, so the angle at which the surface meets a FACE of the
//  cube is not physics -- and a cube has faces where a sphere has none, so this
//  omission is more visible here. There is no collision model either; the cube
//  is stopped at the tank floor by the diagnostic, not by contact.
//
//  THE SHARP CORNER IS SMOOTHED over `smooth` cells in each of the three body
//  directions, so the penalised body is a cube with rounded edges and a rounded
//  corner. At D = 24 and smooth = 1.5 the rounding radius is about 6 % of the
//  edge. The volume is measured from chi rather than taken as D^3 for exactly
//  this reason, and the driver prints both.
//
//  THE PHASES, THE RATIO, AND THE VISCOSITY MATCH are sphere_entry.cpp's and
//  argued there: phi = 1 is water, ratio 100 (not water-against-air's 830,
//  because the conservative Allen-Cahn here is documented to ~100), and a
//  matched KINEMATIC viscosity, since matching the dynamic one across a ratio
//  of 100 drives omega to 1.994 against a limit of 2.
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

// sphere_entry.cpp's writer, unchanged, so the same readers work.
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

// The rotation that takes the cube's body diagonal (1,1,1)/sqrt(3) to world
// -y, i.e. puts one corner straight down. Composed from the axis-angle form
// rather than written out, because the axis is the cross product and the angle
// the dot product and those are the two things worth reading in the source.
Quat corner_down() {
  // d.w = -1/sqrt(3), so the angle is acos(-1/sqrt(3)) = 125.264 deg, about
  // the axis d x w = (1,0,-1)/sqrt(2).
  return Quat::from_axis_angle(Real(1), Real(0), Real(-1),
                               Real(2.1862760354652844));
}

}  // namespace

int main(int argc, char** argv) {
  const Index D       = Index(arg_num(argc, argv, "-d", 24));
  const double span   = arg_num(argc, argv, "-span", 5.0);
  const double aspect = arg_num(argc, argv, "-aspect", 2.0);
  const double Fr     = arg_num(argc, argv, "-fr", 2.0);
  const double Re     = arg_num(argc, argv, "-re", 256.0);
  const double We     = arg_num(argc, argv, "-we", 200.0);
  // 32, NOT sphere_entry.cpp's 128; see the banner. A larger Pe is a smaller
  // mobility and a phase relaxation rate nearer 2, and at D = 12 the sphere's
  // value diverges here.
  const double Pe     = arg_num(argc, argv, "-pe", 32.0);
  const double ratio  = arg_num(argc, argv, "-ratio", 100.0);
  const double chib   = arg_num(argc, argv, "-chib", 2.0);
  const double U      = arg_num(argc, argv, "-u", 0.04);
  const double iw     = arg_num(argc, argv, "-iw", 4.0);
  const double tmax   = arg_num(argc, argv, "-tmax", 6.0);
  // The deliberate perturbation off the symmetric pose, in DEGREES, about x.
  // Zero is the symmetry test; a few degrees is the experiment.
  const double tilt   = arg_num(argc, argv, "-tilt", 5.0);
  const double smooth = arg_num(argc, argv, "-smooth", 1.5);
  const int frames    = int(arg_num(argc, argv, "-frames", 48));
  const char* dump    = arg_str(argc, argv, "-dump", "");
  const bool volume   = arg_flag(argc, argv, "-vol");
  // Rotation held, to isolate the translation against sphere_entry.cpp's path.
  const bool norot    = arg_flag(argc, argv, "-norot");

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const Index nx = Index(span * double(D) + 0.5);
    const Index nz = nx;
    const Index ny = Index(aspect * span * double(D) + 0.5) / 2 * 2;
    const double h     = 0.5 * double(D);           // half edge
    const double diag  = std::sqrt(3.0) * double(D);
    const double g     = U * U / (Fr * Fr * double(D));
    const double nu    = U * double(D) / Re;
    const double rho_l = 1.0, rho_h = ratio;
    const double sigma = rho_h * U * U * double(D) / We;
    const double M     = U * double(D) / Pe;
    const double t_ref = double(D) / U;             // one edge travelled
    const std::size_t nsteps = std::size_t(tmax * t_ref);
    const double ysurf = 0.55 * double(ny);

    std::printf("Cube entering water, corner-down   3-D 6-DOF, Kokkos   %s   "
                "precision %s\n", ExecSpace::name(), precision_name());
    std::printf("  %d x %d x %d   D %d   diagonal %.1f   Fr %g   Re %g   We %g"
                "   rho_H/rho_L %g   rho_b/rho_H %g\n",
                int(nx), int(ny), int(nz), int(D), diag, Fr, Re, We, ratio, chib);
    std::printf("  nu %.6e   sigma %.6e   g %.6e   U %.4f   xi %g   "
                "smooth %g\n", nu, sigma, g, U, iw, smooth);
    std::printf("  tau %.6f (floor 0.5; margin %.2e)   phase omega %.6f "
                "(ceiling 2; margin %.2e)\n",
                3.0 * nu + 0.5, 3.0 * nu,
                1.0 / (M / (1.0 / 3.0) + 0.5),
                2.0 - 1.0 / (M / (1.0 / 3.0) + 0.5));
    std::printf("  tilt off the symmetric pose %.2f deg%s   rotation %s\n",
                tilt, tilt == 0.0 ? "  (THE SYMMETRY TEST)" : "",
                norot ? "HELD" : "free");
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
      return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (ysr - y) / iwr));
    });
    ViscousInterfaceForce<FL> vf(d);
    FColl fc;

    PenalisedBody<FL, Box> body(d);
    body.shape.hx = Real(h);  body.shape.hy = Real(h);  body.shape.hz = Real(h);
    body.shape.smooth = Real(smooth);
    // THE POSE. Corner-down first, then the perturbation ON TOP of it, applied
    // about x in the WORLD frame -- so the tilt is a tilt of the whole body and
    // not a rotation of the cube about its own diagonal, which by the symmetry
    // above would do nothing at all.
    {
      const Quat cd = corner_down();
      const Real a = Real(tilt * 3.14159265358979323846 / 180.0);
      const Quat pt = Quat::from_axis_angle(Real(1), Real(0), Real(0), a);
      // pt * cd: corner-down applied first, then the tilt.
      const Quat q{pt.w * cd.w - pt.x * cd.x - pt.y * cd.y - pt.z * cd.z,
                   pt.w * cd.x + pt.x * cd.w + pt.y * cd.z - pt.z * cd.y,
                   pt.w * cd.y - pt.x * cd.z + pt.y * cd.w + pt.z * cd.x,
                   pt.w * cd.z + pt.x * cd.y - pt.y * cd.x + pt.z * cd.w};
      body.shape.set_orientation(q);
    }
    body.shape.cx = Real(0.5 * double(nx));
    body.shape.cz = Real(0.5 * double(nz));
    // The lowest point of a corner-down cube is half the diagonal below the
    // centre, so THAT is what sits tangent to the undisturbed surface -- not
    // half an edge. Releasing it at cy = ysurf + h would start with the corner
    // already 0.37 D under water.
    body.shape.cy = Real(ysurf + 0.5 * diag);
    body.vx = Real(0);  body.vy = Real(-U);  body.vz = Real(0);
    body.wx = Real(0);  body.wy = Real(0);   body.wz = Real(0);
    body.by = Real(-g);
    body.free_translation = true;
    body.free_rotation = !norot;

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

    auto phiv = pf.phi();
    const Real rl = Real(rho_l), rh = Real(rho_h), gr = Real(g);
    fl.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real dep = ysr - Real(py - hy);
      const Real a = Real(2) * dep / iwr;
      const Real aa = (a < Real(0) ? -a : a);
      const Real lnch = aa + Kokkos::log(Real(1) + Kokkos::exp(Real(-2) * aa))
                      - Real(0.6931471805599453);
      const Real W = rl * dep
                   + (rh - rl) * Real(0.5) * (dep + Real(0.5) * iwr * lnch);
      const Real r = rl + phiv(n) * (rh - rl);
      return FlowState{gr * W / (r / Real(3)), Real(0), Real(0), Real(0)};
    });

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
    // The 6-DOF measurement: mass AND the body-frame inertia tensor, both from
    // chi at the release pose. set_uniform_density() -- the 2-D one -- would
    // fill the mass and leave the tensor at zero, and a zero inertia makes the
    // angular half of the 6x6 singular in exactly the way that produces a
    // plausible tumble rather than a failure.
    body.set_uniform_density6(Real(chib * rho_h));

    {
      const double vchi = double(body.penalised_volume());
      const double exact = double(D) * double(D) * double(D);
      std::printf("  cube volume from chi %.1f   D^3 = %.1f  (%+.2f %%, the "
                  "rounded edges)\n", vchi, exact,
                  100.0 * (vchi - exact) / exact);
      const Mat3 I = body.inertia_body;
      std::printf("  body mass %.4e   displaced water %.4e   net weight %+.3e\n",
                  double(body.mass), rho_h * exact,
                  (double(body.mass) - rho_h * exact) * g);
      // A uniform cube's inertia is ISOTROPIC -- m D^2 / 6 on every axis, for
      // every orientation -- so the measured tensor has a closed form to be
      // checked against, and the smoothing's effect on it has one too.
      //
      // THE SMOOTHING RAISES I WITHOUT CHANGING THE VOLUME, and by a computable
      // amount. Replacing a face by 0.5(1 + tanh((h - |X|)/s)) convolves the
      // sharp indicator with the kernel sech^2(u/s)/(2s), whose variance is
      // pi^2 s^2 / 12. A convolution preserves the mass and adds that variance
      // to the second moment of each axis, so
      //     I / (m D^2 / 6) = 1 + 12 sigma^2 / D^2,   sigma^2 = pi^2 s^2 / 12.
      // At s = 1.5 and D = 12 that predicts +15.422 %, and the measurement came
      // out at +15.42 %. Volume matching D^3 to 0.01 % while I sits 15 % high
      // is therefore the CORRECT behaviour of a rounded cube, not a defect in
      // the measurement -- which is worth printing, because a 15 % discrepancy
      // in an inertia tensor otherwise looks exactly like a bug.
      const double iso = double(body.mass) * double(D) * double(D) / 6.0;
      const double PI2 = 9.869604401089358;
      const double sig2 = PI2 * smooth * smooth / 12.0;
      const double pred = 1.0 + 12.0 * sig2 / (double(D) * double(D));
      std::printf("  I_body diag %.4e %.4e %.4e   m D^2/6 = %.4e\n",
                  double(I(0, 0)), double(I(1, 1)), double(I(2, 2)), iso);
      std::printf("  I/(m D^2/6) measured %.5f   1 + 12 sigma^2/D^2 = %.5f "
                  " (%+.3f %%, the rounded edges)\n",
                  double(I(0, 0)) / iso, pred,
                  100.0 * (double(I(0, 0)) / iso - pred) / pred);
      std::printf("  I_body off-diag %.2e %.2e %.2e  (isotropic, so these are "
                  "zero up to discretisation)\n",
                  double(I(0, 1)), double(I(0, 2)), double(I(1, 2)));
    }

    auto dens_of = KOKKOS_LAMBDA(Index n) {
      return rl + phiv(n) * (rh - rl);
    };

    const std::size_t every =
        (frames > 1) ? std::max<std::size_t>(1, nsteps / std::size_t(frames - 1))
                     : nsteps + 1;
    int nframe = 0;
    bool finite = true;

    // THE COLUMN THAT MATTERS IS THE LAST BUT ONE: the angle between the cube's
    // body diagonal and gravity. It starts at `tilt`, and whether it grows or
    // decays is the whole question. |w| beside it says whether the body is
    // actually turning or merely leaning.
    // A WHOLE-FIELD SCAN beside the body's own numbers. A body diagnostic
    // cannot tell a diverging fluid from a diverging solve -- both arrive as a
    // NaN in the same reduction -- and this run needed exactly that
    // distinction: the force went non-finite while the pose was still moving
    // smoothly, which is only possible if one of them was already gone.
    auto uxv = fl.ux(), uyv = fl.uy(), uzv = fl.uz();
    auto scan = [&](double& umax, double& phlo, double& phhi) {
      const Domain dd = d;
      Real umx = Real(0);
      Kokkos::parallel_reduce("cube_scan_u", Range(0, d.n_padded),
        KOKKOS_LAMBDA(Index n, Real& acc) {
          Index a, b, c; dd.coords(n, a, b, c);
          if (!dd.is_interior(a, b, c)) return;
          const Real m = Kokkos::sqrt(uxv(n) * uxv(n) + uyv(n) * uyv(n)
                                    + uzv(n) * uzv(n));
          if (!(m == m)) { acc = Real(1e30); return; }
          if (m > acc) acc = m;
        }, Kokkos::Max<Real>(umx));
      Real lo = Real(1e30), hi = Real(-1e30);
      Kokkos::parallel_reduce("cube_scan_lo", Range(0, d.n_padded),
        KOKKOS_LAMBDA(Index n, Real& acc) {
          Index a, b, c; dd.coords(n, a, b, c);
          if (!dd.is_interior(a, b, c)) return;
          const Real v = phiv(n);
          if (!(v == v)) { acc = Real(-1e30); return; }
          if (v < acc) acc = v;
        }, Kokkos::Min<Real>(lo));
      Kokkos::parallel_reduce("cube_scan_hi", Range(0, d.n_padded),
        KOKKOS_LAMBDA(Index n, Real& acc) {
          Index a, b, c; dd.coords(n, a, b, c);
          if (!dd.is_interior(a, b, c)) return;
          const Real v = phiv(n);
          if (!(v == v)) { acc = Real(1e30); return; }
          if (v > acc) acc = v;
        }, Kokkos::Max<Real>(hi));
      Kokkos::fence();
      umax = double(umx);  phlo = double(lo);  phhi = double(hi);
    };

    std::printf("\n  t/t0    depth/D     v/U      Fy/(rho U^2 D^2)   "
                "Ty/(rho U^2 D^3)   diag-tilt   |w| D/U    max|u|/cs   "
                "phi range        m_fluid\n");
    const double fnorm = rho_h * U * U * double(D) * double(D);
    const double tnorm = fnorm * double(D);
    const double cy0 = double(body.shape.cy);

    for (std::size_t step = 0; step <= nsteps; ++step) {
      pf.refresh();
      fl.compute_macroscopic();
      vf.refresh(fc);
      const auto Rx = body.refresh6(dens_of);

      if (step % (nsteps / 24 ? nsteps / 24 : 1) == 0) {
        // The body diagonal, rotated into the world, against -y.
        const Real s3 = Real(0.5773502691896258);
        Real dx, dy, dz;
        body.shape.Rm.mul(s3, s3, s3, dx, dy, dz);
        const double ang = std::acos(std::fmax(-1.0, std::fmin(1.0,
                              -double(dy)))) * 57.29577951308232;
        const double wmag = std::sqrt(double(body.wx) * double(body.wx)
                                    + double(body.wy) * double(body.wy)
                                    + double(body.wz) * double(body.wz))
                          * double(D) / U;
        const double tmag = std::sqrt(double(Rx.tx) * double(Rx.tx)
                                    + double(Rx.ty) * double(Rx.ty)
                                    + double(Rx.tz) * double(Rx.tz));
        double umax = 0, phlo = 0, phhi = 0;
        scan(umax, phlo, phhi);
        std::printf("  %5.2f   %8.3f   %+7.4f   %+.4e        %.4e      "
                    "%7.3f    %.3e   %.4f   %+.3f..%+.3f   %.4e\n",
                    double(step) / t_ref,
                    (cy0 - double(body.shape.cy)) / double(D),
                    double(body.vy) / U, double(Rx.fy) / fnorm,
                    tmag / tnorm, ang, wmag,
                    umax / 0.5773502691896258, phlo, phhi,
                    double(Rx.fluid_mass));
        std::fflush(stdout);
      }

      if (*dump && step % every == 0) {
        pf.compute_field();
        auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
        char p[256];
        std::snprintf(p, sizeof p, "%s_%04d.bin", dump, nframe);
        write_field(p, hp, d, nx, ny, nz, hx, hy, hz, false);
        const Box b = body.shape;
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
        // ELEVEN numbers, not the sphere's five: centre, half extents, and the
        // quaternion. A reader tells the two apart by the field count, and a
        // renderer written for the sphere skips these lines rather than drawing
        // a wrong body -- which is why the arity differs rather than being
        // padded to match.
        std::snprintf(p, sizeof p, "%s_body.dat", dump);
        if (std::FILE* f = std::fopen(p, nframe == 0 ? "wb" : "ab")) {
          std::fprintf(f, "%d %.4f %.4f %.4f %.4f %.4f %.4f "
                          "%.6f %.6f %.6f %.6f\n", nframe,
                       double(b.cx), double(b.cy), double(b.cz),
                       double(b.hx), double(b.hy), double(b.hz),
                       double(b.q.w), double(b.q.x), double(b.q.y),
                       double(b.q.z));
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
      body.advance6();

      const double cy = double(body.shape.cy);
      if (!(cy == cy) || !(double(body.vy) == double(body.vy))) {
        std::printf("  NON-FINITE body state at step %zu -- the run diverged, "
                    "it did not travel\n", step);
        finite = false;  break;
      }
      if (!(cy > 0.5 * diag + 1.0 && cy < double(ny - 1))) {
        std::printf("  the cube reached the tank floor at step %zu (cy %.2f); "
                    "there is no contact model, so it is stopped here\n",
                    step, cy);
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
