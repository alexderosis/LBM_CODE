//==============================================================================
//  STONE SKIPPING -- a spinning disc striking water at a shallow attack angle.
//
//  THE CASE THE 6-DOF SOLVE EXISTS FOR. Every body in this tree before it was
//  either planar (a rotation about z and nothing else) or a sphere (whose chi is
//  rotation invariant, so the angular equation measures nothing). A skipping
//  stone is neither: what holds its angle of attack through the impact is
//  gyroscopic stiffness from spin about its own axis, and that cannot be
//  expressed with one angular coordinate. Every entry of the 6x6, the rotating
//  inertia tensor and the quaternion is on the critical path here.
//
//  WHAT MAKES A STONE SKIP, in the terms this driver prints. Bocquet and
//  Clanet's account is inviscid: the disc planes on the water it wets, the
//  pressure on that wetted area has a component along -g, and the stone
//  rebounds if the impulse of that force reverses its vertical momentum before
//  the stone has sunk. Three numbers therefore decide it -- the attack angle
//  (they report ~20 degrees as optimal), the entry Froude number, and the spin,
//  which does not lift at all but keeps the attack angle from collapsing.
//
//  ONE IMPACT AT THE DEFAULTS, AND A SEQUENCE IS CHEAPER THAN I FIRST THOUGHT.
//  I sized a multi-skip run before measuring one, guessing the rebound would
//  leave at about half the translational speed. It does not: the measured exit
//  is vy = +0.070 U, seven times smaller, so the ballistic flight is 3370 steps
//  and 2.5 diameters of travel rather than the ~25 D I had assumed. One whole
//  skip cycle is about 5 D of x.
//
//  So three skips need nx ~ 18 D = 862 cells: 23.8 M cells, 5.1 GB in FP32, and
//  ~14400 steps -- about 25 minutes on a T4, not the ten hours the first
//  estimate claimed. That is well inside one device. Run it with -span 18
//  -tmax 15 and raise the x cutoff; the default span of 8 stops at the first
//  rebound only because the domain runs out at x/D = 6.5.
//
//  What DOES stay out of reach is a translating frame, which would need only
//  ~10 D of x for any number of skips: the phase field has no open boundary
//  (PhaseCell is Bulk, Wall or Excluded), so the free surface cannot be
//  advected out of the domain. Periodic x is the constraint, not the cell count.
//
//  THE MEASURED SKIP, at the defaults (results/skip/skip_d48_attack20.dat):
//  water contact at t/t0 = 0.45, deepest immersion 0.064 D, vy reverses at
//  1.49 while submerged, exit at 2.9, apex at 5.4, and it is descending again
//  by 6.0 -- set up for a second bounce the domain has no room for. It loses
//  27 % of its horizontal speed and 25 % of its spin, against ~20 % per skip
//  reported for real stones.
//
//  AND THE ATTACK ANGLE NUTATES RATHER THAN COLLAPSING: 20 deg at release, down
//  to 11.5 at the deepest point, back up to 17.1, then oscillating about 13.
//  That oscillation IS the gyroscopic term -- before it was added the same case
//  lost the angle monotonically and dug in.
//
//  WHAT IS NOT MODELLED, in the order it bites.
//   * THICKNESS. A real stone is about 1:10 diameter to thickness, which at
//     D = 48 is 4.8 cells -- thinner than the interface it is meant to deflect
//     (xi = 4) plus the 1 cell of chi smoothing. 1:5 is the honest floor here,
//     so this is a flat plate with a rounded rim, not a lenticular stone.
//   * NO CONTACT LINE OR WETTING MODEL, and a skip ENDS in a trailing-edge
//     separation. The exit is not physics.
//   * Re ~ 1e3 against a real 1.75e5. Defensible rather than fatal, because the
//     lift is a planing pressure on the wetted area and not a boundary-layer
//     effect -- Bocquet and Clanet's model has no viscosity in it at all. But
//     tau is the binding constraint, not Re: nu = U D / Re, so Re is bought
//     against the tau -> 1/2 floor and the driver prints the margin.
//   * Density ratio 100, not water-against-air's 830. Negligible during the
//     contact; it would accumulate over any flight.
//
//  Output layout is impact.cu's, except that _body.dat carries ELEVEN numbers
//  rather than five -- centre, radius, half thickness and the quaternion -- so
//  a reader tells a disc from a sphere by the field count rather than a flag.
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/body.cuh"
#include "lbm/phasefield.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// A flat free surface with the hydrostatic pressure that belongs to it,
// integrating the ACTUAL tanh density profile rather than a sharp step: the
// closed form is free and the alternative is an acoustic transient that has to
// damp before the impact means anything. The gauge is p = 0 AT THE SURFACE --
// on this path the populations carry p~ = p/(rho cs^2), so seeding a pressure
// rather than a density is not a stylistic choice, and a wrong gauge is the
// difference between a scheme that works and one that does not.
//------------------------------------------------------------------------------
struct SkipInit {
  Real ysurf, iw, rl, rh, g;

  LBM_HD void operator()(int, int y, int, Real& ph, Real& pt) const {
    const float d = float(ysurf) - float(y);          // depth, positive in water
    ph = Real(0.5f * (1.0f + tanhf(2.0f * d / float(iw))));
    const float a  = 2.0f * d / float(iw);
    const float aa = fabsf(a);
    const float lnch = aa + logf(1.0f + expf(-2.0f * aa)) - 0.6931471805599453f;
    const float W = float(rl) * d
                  + (float(rh) - float(rl)) * 0.5f * (d + 0.5f * float(iw) * lnch);
    const float r = float(rl) + float(ph) * (float(rh) - float(rl));
    pt = Real(float(g) * W / (r / 3.0f));
  }
};

struct PhiDensity {
  const Real* phi = nullptr;
  Real rl = Real(1), dr = Real(0);
  LBM_HD LBM_INLINE Real operator()(long n) const { return rl + dr * phi[n]; }
};

int main(int argc, char** argv) {
  int D = 48, frames = 60;
  double Fr = 5.0, Re = 500.0, We = 1000.0, ratio = 100.0, chib = 2.6;
  double U = 0.05, iw = 4.0, tmax = 6.0, Pe = 128.0;
  double aspect = 4.0, span = 8.0;     // ny = aspect*D, nx = span*D, nz = 3*D
  double thick = 0.2;                  // thickness / D  -- 1:5, see the banner
  double attack = 20.0;                // degrees, leading edge UP
  double descent = 15.0;               // degrees below horizontal
  double spin = 0.64;                  // rim speed / U -- a real stone's value
  double h0 = 1.0;                     // release height, in HALF-THICKNESSES
  double smooth = 1.0;
  const char* dump = "";
  bool cm = true, vol = false, norot = false, nospin = false;

  for (int i = 1; i < argc; ++i) {
    auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-d"))       { if (i+1<argc) D = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-frames"))  { if (i+1<argc) frames = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-fr"))      num(Fr);
    else if (!std::strcmp(argv[i], "-re"))      num(Re);
    else if (!std::strcmp(argv[i], "-we"))      num(We);
    else if (!std::strcmp(argv[i], "-pe"))      num(Pe);
    else if (!std::strcmp(argv[i], "-ratio"))   num(ratio);
    else if (!std::strcmp(argv[i], "-chib"))    num(chib);
    else if (!std::strcmp(argv[i], "-u"))       num(U);
    else if (!std::strcmp(argv[i], "-iw"))      num(iw);
    else if (!std::strcmp(argv[i], "-tmax"))    num(tmax);
    else if (!std::strcmp(argv[i], "-span"))    num(span);
    else if (!std::strcmp(argv[i], "-aspect"))  num(aspect);
    else if (!std::strcmp(argv[i], "-thick"))   num(thick);
    else if (!std::strcmp(argv[i], "-attack"))  num(attack);
    else if (!std::strcmp(argv[i], "-descent")) num(descent);
    else if (!std::strcmp(argv[i], "-spin"))    num(spin);
    else if (!std::strcmp(argv[i], "-h0"))      num(h0);
    else if (!std::strcmp(argv[i], "-smooth"))  num(smooth);
    else if (!std::strcmp(argv[i], "-bgk"))     cm = false;
    else if (!std::strcmp(argv[i], "-vol"))     vol = true;
    // Two bisection switches, not physical options. -norot holds the
    // orientation, which turns this into a plate on rails and isolates the
    // translation; -nospin keeps the rotation free but releases with zero spin,
    // which is the run that should NOT skip if gyroscopic stiffness is doing
    // what the literature says.
    else if (!std::strcmp(argv[i], "-norot"))   norot = true;
    else if (!std::strcmp(argv[i], "-nospin"))  nospin = true;
    else if (!std::strcmp(argv[i], "-dump"))    { if (i+1<argc) dump = argv[++i]; }
  }
  if (nospin) spin = 0.0;

  const int nx = int(span * D + 0.5);
  const int ny = int(aspect * D + 0.5) / 2 * 2;
  const int nz = 3 * D;
  const double R = 0.5 * double(D);
  const double hy = 0.5 * thick * double(D);        // HALF thickness
  const double g = U * U / (Fr * Fr * double(D));
  const double nu = U * double(D) / Re;
  const double rho_l = 1.0, rho_h = ratio;
  const double sigma = rho_h * U * U * double(D) / We;
  const double t_ref = double(D) / U;               // one diameter travelled
  const std::size_t nsteps = std::size_t(tmax * t_ref);
  const double ysurf = 0.5 * double(ny);
  const double att = attack * M_PI / 180.0;
  const double des = descent * M_PI / 180.0;

  const backend::DeviceInfo dev = backend::device_info();
  std::printf("Stone skipping   6-DOF disc   phase field %s + fluid %s   %s, %s\n",
              cm ? "CM" : "BGK", cm ? "CM" : "BGK", dev.name.c_str(),
              sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  %d x %d x %d   D %d   thickness %.1f cells (1:%.0f)   "
              "rho_b/rho_H %g\n", nx, ny, nz, D, 2.0 * hy, 1.0 / thick, chib);
  std::printf("  Fr %g   Re %g   We %g   rho_H/rho_L %g   nu %.6e   "
              "sigma %.6e   g %.6e   U %.4f   xi %g\n",
              Fr, Re, We, ratio, nu, sigma, g, U, iw);
  std::printf("  tau %.6f   (stability floor 0.5; margin %.2e)\n",
              3.0 * nu + 0.5, 3.0 * nu);
  {
    const double Mm = U * double(D) / Pe;
    const double om = 1.0 / (Mm / (1.0 / 3.0) + 0.5);
    std::printf("  Pe %g   M %.6e   phase omega %.6f   (ceiling 2; margin %.2e)\n",
                Pe, Mm, om, 2.0 - om);
    if (om > 1.99) std::printf("  WARNING phase omega within 1e-2 of its "
                               "ceiling -- lower -pe or raise -d\n");
  }
  if (3.0 * nu <= 0.0) { std::printf("  tau AT the floor -- refusing\n"); return 1; }

  // THE SPIN, PRINTED AS A RIM SPEED, because that is the number with a
  // compressibility limit on it. A real stone at 90 rad/s, D = 5 cm, U = 3.5
  // m/s has a rim speed of 0.64 U -- which is why that is the default -- and in
  // lattice units 0.64 U is 0.032, comfortably under the ~0.05 the rest of the
  // case already runs at. Matching the RATIO rather than the rate is what makes
  // that possible.
  const double Omega = spin * U / R;
  std::printf("  attack %.1f deg   descent %.1f deg   spin %.3f U at the rim "
              "-> Omega %.4e rad/step (%.2f rev over the run)\n",
              attack, descent, spin, Omega, Omega * double(nsteps) / (2.0 * M_PI));
  std::printf("  rotation %s   spin %s\n", norot ? "HELD" : "free",
              nospin ? "ZERO (the control)" : "on");
  std::printf("  t_ref %.1f steps   %zu steps to t/t0 = %g\n",
              t_ref, nsteps, tmax);
  std::fflush(stdout);

  //---- solver ----------------------------------------------------------------
  backend::PhaseFieldOn<D3Q27> pf(nx, ny, nz);
  pf.phase.width = Real(iw);
  const double M = U * double(D) / Pe;
  pf.set_mobility(Real(M));
  pf.set_phase_op(cm ? PhaseOp::CentralMoments : PhaseOp::BGK);
  pf.set_fluid_op(cm ? MultiOp::CentralMoments : MultiOp::BGK);
  pf.fluid.rho_L = Real(rho_l);          pf.fluid.rho_H = Real(rho_h);
  pf.fluid.mu_L  = Real(rho_l * nu);     pf.fluid.mu_H  = Real(rho_h * nu);
  pf.fluid.kappa = Real(MultiphaseModel::kappa_from_sigma(Real(sigma), Real(iw)));
  pf.fluid.beta  = Real(MultiphaseModel::beta_from_sigma (Real(sigma), Real(iw)));
  pf.fluid.by    = Real(-g);
  pf.enable_viscous_force(true);

  // A tank needs a floor and a lid; see impact.cu's note on why leaving them
  // out puts a zero-width second interface at the periodic seam in y.
  {
    const std::size_t NN = std::size_t(long(nx) * ny * nz);
    const std::uint8_t kF = Fluid, kS = Solid, kB = PhaseBulk, kW = PhaseWall;
    std::vector<std::uint8_t> ff(NN, kF), pfl(NN, kB);
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x)
        for (int y : {0, ny - 1}) {
          const std::size_t n = std::size_t(node_id(x, y, z, nx, ny));
          ff[n] = kS;  pfl[n] = kW;
        }
    pf.set_geometry(pfl, ff);
  }

  SkipInit init;
  init.ysurf = Real(ysurf);  init.iw = Real(iw);
  init.rl = Real(rho_l);     init.rh = Real(rho_h);   init.g = Real(g);
  pf.initialise_with(init);

  PhiDensity dens;
  dens.phi = pf.phi_device();
  dens.rl  = Real(rho_l);
  dens.dr  = Real(rho_h - rho_l);

  //---- the stone -------------------------------------------------------------
  backend::Body<Disc> body(nx, ny, nz);
  body.shape.R = Real(R);
  body.shape.hy = Real(hy);
  body.shape.smooth = Real(smooth);
  // THE POSE. A rotation about +z by the attack angle tips the symmetry axis
  // from +y toward -x, which RAISES the leading edge at +x. That sign is pinned
  // in test/host_body.cpp block 6 rather than left to be reasoned about here,
  // because getting it backwards buries the leading edge and the stone dives --
  // a plausible result that is the wrong experiment.
  body.shape.set_orientation(
      Quat::from_axis_angle(Real(0), Real(0), Real(1), Real(att)));
  body.shape.cx = Real(1.5 * double(D));
  body.shape.cz = Real(0.5 * double(nz));
  // Released clear of the surface by h0 half-thicknesses PLUS the vertical
  // reach of the tilted disc, which for a tilted plate is R sin(attack) + hy
  // cos(attack) rather than hy: a 20-degree disc of radius R hangs a long way
  // below its own centre, and using hy alone starts it already cut by the water.
  const double reach_y = R * std::sin(att) + hy * std::cos(att);
  body.shape.cy = Real(ysurf + reach_y + h0 * hy);
  body.vx = Real(U * std::cos(des));
  body.vy = Real(-U * std::sin(des));
  body.vz = Real(0);
  // Spin about the body's OWN axis, which is where a stone's angular momentum
  // is and the only place it does no lifting. Read from the pose rather than
  // written out as (-sin a, cos a, 0), so it stays right if the pose changes.
  {
    Real ax, ay, az;
    body.shape.axis(ax, ay, az);
    body.wx = Real(Omega * double(ax));
    body.wy = Real(Omega * double(ay));
    body.wz = Real(Omega * double(az));
  }
  body.props.by = Real(-g);
  body.props.free_translation = true;
  body.props.free_rotation = !norot;
  body.couple_velocity(pf.ux_device(), pf.uy_device(), pf.uz_device());
  // set_uniform_density6, NOT set_uniform_density: the 2-D call fills the mass
  // and leaves the tensor at ZERO, which makes the angular half of the 6x6
  // singular and produces a plausible tumble rather than a failure.
  body.set_uniform_density6(Real(chib * rho_h));

  {
    const double v = body.penalised_volume();
    const double exact = M_PI * R * R * 2.0 * hy;
    std::printf("  disc volume from chi %.1f   pi R^2 (2h) = %.1f   (%+.2f %%)\n",
                v, exact, 100.0 * (v - exact) / exact);
    const double m = double(body.props.mass);
    std::printf("  body mass %.1f   displaced water %.1f   net weight %+.3e\n",
                m, rho_h * exact, (m - rho_h * exact) * g);
    // The closed-form cylinder, as a check on the whole measurement path. The
    // smoothing raises both by a computed amount -- see test/host_body.cpp
    // block 6 -- so a few per cent high is correct and a factor is not.
    std::printf("  I_body axial %.4e (m R^2/2 = %.4e)   diametral %.4e "
                "(m(R^2/4+h^2/3) = %.4e)\n",
                double(body.inertia_body(1, 1)), m * R * R / 2.0,
                double(body.inertia_body(0, 0)),
                m * (R * R / 4.0 + hy * hy / 3.0));
  }

  typename backend::Body<Disc>::Reaction6 last_rx;
  pf.couple_external_force(body.fx(), body.fy(), body.fz());
  // THE BODY RUNS INSIDE pf.step(). Volume penalisation needs the window
  // between the macroscopic field and the collision -- PASS 4b in
  // phasefield.cuh. Calling it around pf.step() reads a u one step stale, which
  // inverts the sign of the measured momentum deficit.
  pf.set_pre_fluid([&]{ last_rx = body.refresh6(dens); });

  //---- march -----------------------------------------------------------------
  const std::size_t frame_every =
      (frames > 1) ? std::max<std::size_t>(1, nsteps / std::size_t(frames - 1))
                   : nsteps + 1;
  int nframe = 0;
  std::vector<Real> phi;
  bool finite = true;
  const double fnorm = rho_h * U * U * double(D) * double(D);
  const double tnorm = fnorm * double(D);
  double vy_min = 0, vy_max = -1e30;
  double y_low = 1e30;
  bool rebounded = false;
  double t_rebound = -1;

  // clear/D is the LOWEST POINT of the tilted disc above the undisturbed
  // surface, which is what decides whether it is in the water -- the centre
  // height is misleading for a plate at 20 degrees. attack is read back from
  // the pose every print, because whether it holds is the question.
  // Tz, NOT Ty: the attack angle is a rotation about z, so the torque that
  // changes it is the z one. Ty is the torque about the vertical, which for a
  // spinning disc is mostly the gyroscopic reaction and says nothing about
  // whether the stone is pitching into the water.
  std::printf("\n  t/t0     x/D    clear/D    vx/U     vy/U    attack   "
              "|w|D/U   Fy/(rU^2D^2)  Tz/(rU^2D^3)   m_fluid\n");

  for (std::size_t step = 0; step <= nsteps; ++step) {
    auto rx = last_rx;
    if (!(rx.fluid_mass == rx.fluid_mass) || !(rx.fy == rx.fy)) {
      std::printf("  NON-FINITE FLUID at step %zu -- the flow diverged "
                  "(m_fluid %g)\n", step, rx.fluid_mass);
      finite = false;  break;
    }

    Real axr, ayr, azr;
    body.shape.axis(axr, ayr, azr);
    const double a_now = std::acos(std::fmax(-1.0, std::fmin(1.0, double(ayr))))
                       * 180.0 / M_PI;
    const double reach_now = R * std::sin(a_now * M_PI / 180.0)
                           + hy * std::cos(a_now * M_PI / 180.0);
    const double clear = (double(body.shape.cy) - reach_now - ysurf) / double(D);
    if (clear < y_low) y_low = clear;
    const double vyU = double(body.vy) / U;
    if (vyU < vy_min) vy_min = vyU;
    if (vyU > vy_max) vy_max = vyU;
    // A REBOUND IS A SIGN CHANGE IN vy WHILE IN CONTACT, not merely vy > 0:
    // a stone that never touched the water and is still on its way down has
    // vy < 0 throughout, and one lifted by nothing but the initial transient
    // would show vy > 0 before it ever reached the surface.
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
                  double(rx.fy) / fnorm, double(rx.tz) / tnorm, rx.fluid_mass);
      std::fflush(stdout);
    }

    if (*dump && step % frame_every == 0) {
      pf.field_to_host(pf.phi_device(), phi);
      char fp[256];
      const int hdr[2] = {nx, ny};
      std::snprintf(fp, sizeof fp, "%s_%04d.bin", dump, nframe);
      if (std::FILE* f = std::fopen(fp, "wb")) {
        std::fwrite(hdr, sizeof(int), 2, f);
        std::vector<float> pl(std::size_t(nx) * ny);
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            pl[std::size_t(y) * nx + x] =
                float(phi[std::size_t(node_id(x, y, nz / 2, nx, ny))]);
        std::fwrite(pl.data(), sizeof(float), pl.size(), f);
        std::fclose(f);
      }
      std::snprintf(fp, sizeof fp, "%s_chi_%04d.bin", dump, nframe);
      if (std::FILE* c = std::fopen(fp, "wb")) {
        std::fwrite(hdr, sizeof(int), 2, c);
        std::vector<float> cl(std::size_t(nx) * ny);
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            cl[std::size_t(y) * nx + x] =
                float(body.shape.chi(Real(x), Real(y), Real(nz / 2)));
        std::fwrite(cl.data(), sizeof(float), cl.size(), c);
        std::fclose(c);
      }
      // ELEVEN numbers: centre, radius, half thickness, quaternion. A reader
      // tells a disc from a sphere's five and a box's eleven-with-three-extents
      // by the field count, so a renderer written for one skips the other
      // rather than drawing a wrong body from the first four numbers.
      std::snprintf(fp, sizeof fp, "%s_body.dat", dump);
      if (std::FILE* b = std::fopen(fp, nframe == 0 ? "wb" : "ab")) {
        std::fprintf(b, "%d %.4f %.4f %.4f %.4f %.4f %.6f %.6f %.6f %.6f\n",
                     nframe, double(body.shape.cx), double(body.shape.cy),
                     double(body.shape.cz), double(body.shape.R),
                     double(body.shape.hy), double(body.shape.q.w),
                     double(body.shape.q.x), double(body.shape.q.y),
                     double(body.shape.q.z));
        std::fclose(b);
      }
      if (vol) {
        std::snprintf(fp, sizeof fp, "%s_vol_%04d.bin", dump, nframe);
        if (std::FILE* v = std::fopen(fp, "wb")) {
          const int vh[3] = {nx, ny, nz};
          std::fwrite(vh, sizeof(int), 3, v);
          std::vector<float> vv(std::size_t(nx) * ny * nz);
          for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
              for (int x = 0; x < nx; ++x)
                vv[(std::size_t(z) * ny + y) * nx + x] =
                    float(phi[std::size_t(node_id(x, y, z, nx, ny))]);
          std::fwrite(vv.data(), sizeof(float), vv.size(), v);
          std::fclose(v);
        }
      }
      ++nframe;
    }

    pf.step();
    body.advance6();

    const double cxv = double(body.shape.cx), cyv = double(body.shape.cy);
    if (!(cxv == cxv) || !(cyv == cyv) || !(double(body.vy) == double(body.vy))) {
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
    // wake, which is not a skip off undisturbed water. Stopped rather than
    // allowed to produce a second bounce that means nothing.
    if (cxv > double(nx) - 1.5 * double(D)) {
      std::printf("  the stone reached the end of the tank at step %zu "
                  "(x/D %.2f of %.2f) -- x is periodic, so it would wrap into "
                  "its own wake\n", step, cxv / double(D), double(nx) / double(D));
      finite = false;  break;
    }
  }

  std::printf("\n  %s\n", finite ? "finite throughout" : "STOPPED EARLY");
  std::printf("  lowest clearance %+.3f D   vy from %+.4f U to %+.4f U\n",
              y_low, vy_min, vy_max);
  if (rebounded)
    std::printf("  SKIPPED: vy changed sign at t/t0 = %.2f while in contact\n",
                t_rebound);
  else
    std::printf("  NO SKIP: vy never became positive while in contact\n");
  if (*dump) std::printf("  %d frame(s) written to %s\n", nframe, dump);
  return finite ? 0 : 1;
}
