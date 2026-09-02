//==============================================================================
//  A SPHERE ENTERING WATER: gravity, a deformable free surface, and the cavity.
//
//  WHY THE PHASE FIELD AND NOT THE FREE SURFACE. A "free surface" in this tree
//  means the Korner sharp-interface method, and freesurface.cuh does not have a
//  moving obstacle -- deliberately. The parent has one and its own banner says
//  it is not reliable: two defects are identified, measured, and LEFT IN,
//  because fixing either makes the demonstrator fail sooner. Building a water
//  entry on that would inherit a known-bad ledger. The diffuse interface here is
//  a genuine deformable surface with surface tension and needs no such
//  compromise; what it costs is that the air is RESOLVED rather than ignored,
//  so the density ratio is a parameter and not infinity.
//
//  WHAT IS AND IS NOT 3-D HERE. The sphere is a real sphere: chi depends on all
//  three coordinates and its volume integrates to 4 pi R^3 / 3, which the driver
//  checks at startup rather than assuming. The DYNAMICS are three translations
//  with rotation switched off. For a sphere entering on its axis that is exact
//  and not an approximation -- there is no torque about any axis, and a sphere's
//  chi is invariant under rotation anyway, so the roll equation would measure
//  and change nothing. It is NOT a general 3-D rigid body: no quaternion, no
//  rotating inertia tensor, and nothing here would carry a tumbling body.
//
//  THE DENSITY RATIO IS 100, NOT 1000. Water against air is about 830 by mass.
//  The conservative Allen-Cahn phase field in this tree is documented to ~100
//  and that is what is used; the entry is therefore quantitatively a sphere
//  entering a liquid 100x its surroundings, which is the right physics for the
//  cavity and the splash and the wrong number for the air. Raising it is a
//  colour-gradient question, not a parameter change.
//
//  THE SPHERE IS RELEASED AT THE SURFACE, ALREADY MOVING. Dropping it through
//  resolved air would spend most of the run integrating an air column that
//  contributes 1/100 of the force, so the impact speed U is imposed at
//  touchdown, which is what a water-entry experiment reports anyway. Gravity
//  acts throughout on both fluid and body, so the sphere keeps accelerating
//  under its own excess weight after entry.
//
//  THE DIMENSIONLESS GROUPS, and the resolution is not one of them:
//
//      Fr = U / sqrt(g D)      how ballistic the entry is
//      Re = U D / nu           set by the viscosity, and capped by tau -> 1/2
//      We = rho_H U^2 D / sigma    inertia against surface tension
//      chi_rho = rho_H / rho_L, chi_b = rho_body / rho_H
//
//  tau is printed against its floor because Re is bought with viscosity and
//  this is where a case stops being a simulation.
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
// phi and the hydrostatic pressure, from one analytic profile -- the same
// argument enan_rt/rti3d make: integrating the ACTUAL tanh density profile
// costs nothing in closed form and saves an acoustic transient that would
// otherwise have to damp before the impact means anything.
//
// The gauge is p = 0 AT THE FREE SURFACE. On this path the populations carry
// p~ = p/(rho cs^2), so seeding a pressure rather than a density is not a
// stylistic choice; and a wrong gauge here is the difference between a scheme
// that works and one that does not.
//------------------------------------------------------------------------------
struct ImpactInit {
  Real ysurf, iw, rl, rh, g;

  LBM_HD void operator()(int, int y, int, Real& ph, Real& pt) const {
    const float d = float(ysurf) - float(y);          // depth, positive in water
    ph = Real(0.5f * (1.0f + tanhf(2.0f * d / float(iw))));

    // W(d) = integral of rho from the surface down to d, through the profile:
    //   integral 1/2 (1 + tanh(2s/w)) ds = 1/2 [ s + (w/2) ln cosh(2s/w) ]
    // ln cosh is written as a shifted softplus so it stays accurate for large
    // |a| instead of overflowing in the exponential.
    const float a  = 2.0f * d / float(iw);
    const float aa = fabsf(a);
    const float lnch = aa + logf(1.0f + expf(-2.0f * aa)) - 0.6931471805599453f;
    const float W = float(rl) * d
                  + (float(rh) - float(rl)) * 0.5f * (d + 0.5f * float(iw) * lnch);
    const float r = float(rl) + float(ph) * (float(rh) - float(rl));
    pt = Real(float(g) * W / (r / 3.0f));
  }
};

//------------------------------------------------------------------------------
// The local fluid density from the order parameter. A plain POD so it captures
// into a kernel by value, exactly like body.cuh's own UniformDensity: the body
// must feel the thousandfold-heavier medium when it reaches the water, and a
// penalisation given a constant density would decelerate it as though it were
// still in air.
//------------------------------------------------------------------------------
struct PhiDensity {
  const Real* phi = nullptr;
  Real rl = Real(1), dr = Real(0);
  LBM_HD LBM_INLINE Real operator()(long n) const { return rl + dr * phi[n]; }
};

int main(int argc, char** argv) {
  int D = 32, frames = 120;
  double Fr = 2.0, Re = 256.0, We = 200.0, ratio = 100.0, chib = 2.0;
  double U = 0.04, iw = 4.0, tmax = 12.0, Pe = 128.0;
  double aspect = 2.0, span = 4.0;     // ny = aspect*D*2, nx = nz = span*D
  const char* dump = "";
  bool cm = true, driven = false, nobody = false;

  for (int i = 1; i < argc; ++i) {
    auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-d"))      { if (i+1<argc) D = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-frames")) { if (i+1<argc) frames = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-fr"))     num(Fr);
    else if (!std::strcmp(argv[i], "-re"))     num(Re);
    else if (!std::strcmp(argv[i], "-we"))     num(We);
    else if (!std::strcmp(argv[i], "-pe"))     num(Pe);
    else if (!std::strcmp(argv[i], "-ratio"))  num(ratio);
    else if (!std::strcmp(argv[i], "-chib"))   num(chib);
    else if (!std::strcmp(argv[i], "-u"))      num(U);
    else if (!std::strcmp(argv[i], "-iw"))     num(iw);
    else if (!std::strcmp(argv[i], "-tmax"))   num(tmax);
    else if (!std::strcmp(argv[i], "-span"))   num(span);
    else if (!std::strcmp(argv[i], "-aspect")) num(aspect);
    else if (!std::strcmp(argv[i], "-bgk"))    cm = false;
    // A BISECTION SWITCH, not a physical option. Driving the sphere at constant
    // speed removes the rigid-body solve from the loop while leaving the fluid,
    // the penalisation and the interface exactly as they are -- so if -driven is
    // stable and the free body is not, the fault is in the coupling and not in
    // the flow. It is also the classic constant-velocity Wagner entry.
    else if (!std::strcmp(argv[i], "-driven")) driven = true;
    // The other half of the bisection: the same flow with no solid in it at
    // all. A hydrostatic column with a flat interface should sit still forever,
    // so if THIS diverges the fault is in the flow setup and the body is
    // innocent.
    else if (!std::strcmp(argv[i], "-nobody")) nobody = true;
    else if (!std::strcmp(argv[i], "-dump") && i + 1 < argc) dump = argv[++i];
  }

  const int nx = int(span * D + 0.5), nz = nx;
  const int ny = int(aspect * span * D + 0.5) / 2 * 2;
  const double R = 0.5 * double(D);
  const double g     = U * U / (Fr * Fr * double(D));
  const double nu    = U * double(D) / Re;
  const double rho_l = 1.0, rho_h = ratio;
  const double sigma = rho_h * U * U * double(D) / We;
  const double t_ref = double(D) / U;                 // one diameter travelled
  const std::size_t nsteps = std::size_t(tmax * t_ref);
  const double ysurf = 0.55 * double(ny);             // deep water below it

  const backend::DeviceInfo dev = backend::device_info();
  std::printf("Sphere entering water   phase field %s + fluid %s   %s, %s\n",
              cm ? "CM" : "BGK", cm ? "CM" : "BGK", dev.name.c_str(),
              sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  %d x %d x %d   D %d   Fr %g   Re %g   We %g   rho_H/rho_L %g"
              "   rho_b/rho_H %g\n", nx, ny, nz, D, Fr, Re, We, ratio, chib);
  std::printf("  nu %.6e   sigma %.6e   g %.6e   U %.4f   xi %g\n",
              nu, sigma, g, U, iw);
  std::printf("  tau %.6f   (stability floor 0.5; margin %.2e)\n",
              3.0 * nu + 0.5, 3.0 * nu);
  // THE PHASE FIELD HAS ITS OWN FLOOR AND IT IS AT THE OTHER END. The mobility
  // sets omega through M = (1/omega - 1/2) cs^2, so a SMALL mobility pushes
  // omega toward 2, which is where that relaxation stops -- the mirror image of
  // tau -> 1/2 for the fluid. On the diameter M = U D / Pe, so a large Pe on a
  // small D is exactly the combination that walks into it, and the first
  // version of this driver did: Pe = 1024 at D = 12 gives omega = 1.9944 and
  // the run produced non-finite values within 13 steps. Printed for the same
  // reason tau is.
  {
    const double Mm = U * double(D) / Pe;
    const double om = 1.0 / (Mm / (1.0 / 3.0) + 0.5);
    std::printf("  Pe %g   M %.6e   phase omega %.6f   (ceiling 2; margin %.2e)\n",
                Pe, Mm, om, 2.0 - om);
    if (om > 1.99) std::printf("  WARNING phase omega within 1e-2 of its "
                               "ceiling -- lower -pe or raise -d\n");
  }
  std::printf("  t_ref %.1f steps (one diameter at U)   %zu steps to t/t0 = %g%s\n",
              t_ref, nsteps, tmax, driven ? "   [DRIVEN at U]" : "   [free]");
  if (3.0 * nu <= 0.0) { std::printf("  tau AT the floor -- refusing\n"); return 1; }

  //---- solver ----------------------------------------------------------------
  backend::PhaseFieldOn<D3Q27> pf(nx, ny, nz);
  pf.phase.width = Real(iw);
  const double M = U * double(D) / Pe;                 // Peclet on the DIAMETER
  pf.set_mobility(Real(M));
  pf.set_phase_op(cm ? PhaseOp::CentralMoments : PhaseOp::BGK);
  pf.set_fluid_op(cm ? MultiOp::CentralMoments : MultiOp::BGK);
  pf.fluid.rho_L = Real(rho_l);          pf.fluid.rho_H = Real(rho_h);
  // KINEMATIC viscosity matched across the ratio, so mu scales with rho. The
  // parent's CLAUDE.md records the alternative as a units error wearing another
  // hat: matching mu across a ratio of 100 leaves the heavy phase at nu/100 and
  // drives omega to 1.994 against a limit of 2.
  pf.fluid.mu_L  = Real(rho_l * nu);     pf.fluid.mu_H  = Real(rho_h * nu);
  pf.fluid.kappa = Real(MultiphaseModel::kappa_from_sigma(Real(sigma), Real(iw)));
  pf.fluid.beta  = Real(MultiphaseModel::beta_from_sigma (Real(sigma), Real(iw)));
  pf.fluid.by    = Real(-g);
  pf.enable_viscous_force(true);

  // A TANK NEEDS A FLOOR AND A LID, and leaving them out is not a small
  // omission here. With no geometry this solver is periodic in every direction,
  // so the air above the surface wraps directly into the water below it: a
  // SECOND interface at the seam, of zero width, across which grad phi is a
  // one-cell jump. The capillary force reads that as an enormous curvature. The
  // flow alone survived it for a couple of hundred steps, which is what made it
  // hard to see -- add the body's perturbation and the run went non-finite in
  // under twenty. x and z stay periodic, which is what this code's indexing
  // gives for free.
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

  ImpactInit init;
  init.ysurf = Real(ysurf);  init.iw = Real(iw);
  init.rl = Real(rho_l);     init.rh = Real(rho_h);   init.g = Real(g);
  pf.initialise_with(init);

  BodyReaction last_rx;
  PhiDensity dens;
  dens.phi = pf.phi_device();
  dens.rl  = Real(rho_l);
  dens.dr  = Real(rho_h - rho_l);

  //---- the sphere ------------------------------------------------------------
  backend::Body<Sphere> body(nx, ny, nz);
  body.shape.cx = Real(0.5 * double(nx));
  body.shape.cz = Real(0.5 * double(nz));
  body.shape.cy = Real(ysurf + R);        // tangent to the undisturbed surface
  body.shape.R  = Real(R);
  body.shape.smooth = Real(1.5);
  body.vx = Real(0);  body.vz = Real(0);
  body.vy = Real(-U);                     // the impact speed, imposed
  body.omega = Real(0);
  body.props.by = Real(-g);               // the SAME vector the collision gets
  body.props.free_translation = !driven;
  // OFF, AND NOT AS A SIMPLIFICATION: a sphere's chi is invariant under
  // rotation, so the roll equation measures nothing and would change nothing.
  body.props.free_rotation = false;
  body.set_uniform_density(Real(chib * rho_h));
  body.couple_velocity(pf.ux_device(), pf.uy_device(), pf.uz_device());
  if (!nobody) {
    pf.couple_external_force(body.fx(), body.fy(), body.fz());
    // THE BODY RUNS INSIDE pf.step(), not around it. Volume penalisation needs
    // the window between the macroscopic field and the collision -- see PASS 4b
    // in phasefield.cuh. Calling it before pf.step() instead reads a u that is
    // one step stale, which inverts the sign of the measured momentum deficit
    // and diverges; that is how this hook came to exist.
    pf.set_pre_fluid([&]{ last_rx = body.refresh(dens); });
  }

  // THE VOLUME IS CHECKED, NOT ASSUMED. chi is smoothed over 1.5 cells, so the
  // penalised sphere is slightly larger than the nominal one and the integral
  // is the honest measure of the body the force actually acts on. A gross
  // disagreement here means the 3-D indicator or the prism/sphere culling is
  // wrong, and every number downstream would be wrong with it.
  {
    const auto m = body.indicator_moments();
    const double exact = 4.0 / 3.0 * M_PI * R * R * R;
    std::printf("  sphere volume from chi %.1f   4 pi R^3/3 = %.1f   (%+.2f %%)\n",
                m.area, exact, 100.0 * (m.area - exact) / exact);
    std::printf("  body mass %.1f   displaced water %.1f   net weight %+.3e\n",
                double(body.props.mass), rho_h * exact,
                (double(body.props.mass) - rho_h * exact) * g);
  }


  //---- march -----------------------------------------------------------------
  const std::size_t frame_every =
      (frames > 1) ? std::max<std::size_t>(1, nsteps / std::size_t(frames - 1)) : nsteps + 1;
  int nframe = 0;
  std::vector<Real> phi;
  bool finite = true;

  // The last column is the AXISYMMETRY CHECK. An on-axis entry has no lateral
  // force, so vx and vz must stay at zero; anything growing there is either a
  // real symmetry break or a bug in the new z limb, and it costs nothing to
  // watch. It is the cheapest test that the three-translation path is right.
  std::printf("\n  t/t0    depth/R     v/U      Fy/(rho U^2 D^2)   m_fluid"
              "     |v_lat|/U\n");
  const double fnorm = rho_h * U * U * double(D) * double(D);

  for (std::size_t step = 0; step <= nsteps; ++step) {
    // AFTER the macroscopic pass and BEFORE the fluid steps, which on this
    // solver means before pf.step(): the fluid kernel fuses the macroscopic
    // pass with the collision, so the velocity the body reads is the one the
    // previous step wrote -- which is exactly the field body_probe_node
    // subtracts its own previous force back out of.
    // The reaction from the PREVIOUS step's refresh, which ran inside pf.step().
    // Reported one step late rather than measured at the wrong time.
    BodyReaction rx = last_rx;
    // THE FLUID IS CHECKED, NOT THE BODY. A DRIVEN body advances mechanically
    // whatever the flow does, so watching only cy reports "finite throughout"
    // over a domain full of NaN -- which the first version of this driver did.
    // The reaction integrates chi rho over the flow, so it is NaN as soon as
    // the field is, and it costs nothing: it is already computed.
    if (!(rx.fluid_mass == rx.fluid_mass) || !(rx.fy == rx.fy)) {
      std::printf("  NON-FINITE FLUID at step %zu -- the flow diverged "
                  "(m_fluid %g)\n", step, rx.fluid_mass);
      finite = false;
      break;
    }

    if (step % (nsteps / 24 ? nsteps / 24 : 1) == 0) {
      const double depth = (ysurf - double(body.shape.cy)) / R;
      const double vlat = std::sqrt(double(body.vx) * double(body.vx)
                                  + double(body.vz) * double(body.vz)) / U;
      std::printf("  %5.2f   %8.3f   %+7.4f   %+.4e      %9.1f   %.2e\n",
                  double(step) / t_ref, depth, double(body.vy) / U,
                  rx.fy / fnorm, rx.fluid_mass, vlat);
      std::fflush(stdout);
    }

    if (*dump && step % frame_every == 0) {
      pf.field_to_host(pf.phi_device(), phi);
      char fp[256];
      std::snprintf(fp, sizeof fp, "%s_%04d.bin", dump, nframe);
      if (std::FILE* f = std::fopen(fp, "wb")) {
        // The mid-z plane through the sphere's axis: two int32 then nx*ny
        // float32, the layout doc/fig/rt2d_anim.py already reads.
        const int hdr[2] = {nx, ny};
        std::fwrite(hdr, sizeof(int), 2, f);
        std::vector<float> pl(std::size_t(nx) * ny);
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            pl[std::size_t(y) * nx + x] =
                float(phi[std::size_t(node_id(x, y, nz / 2, nx, ny))]);
        std::fwrite(pl.data(), sizeof(float), pl.size(), f);
        std::fclose(f);
        ++nframe;
      }
    }

    pf.step();
    body.advance();

    // TWO DIFFERENT FAILURES, REPORTED AS TWO. The first version printed "left
    // the domain" for both, because !(cy > 1 && cy < ny-1) is also true of NaN
    // -- so a blown-up run was reported as a geometric one, which sent the
    // diagnosis in the wrong direction for a while.
    const double cyv = double(body.shape.cy), vyv = double(body.vy);
    if (!(cyv == cyv) || !(vyv == vyv)) {
      std::printf("  NON-FINITE body state at step %zu (cy %g, vy %g) -- the "
                  "run diverged, it did not travel\n", step, cyv, vyv);
      finite = false;
      break;
    }
    if (!(cyv > 1.0 && cyv < double(ny - 1))) {
      std::printf("  the sphere reached the domain edge at step %zu (cy %.2f)\n",
                  step, cyv);
      finite = false;
      break;
    }
  }

  std::printf("\n  %s   %d frame(s) written%s%s\n",
              finite ? "finite throughout" : "STOPPED EARLY",
              nframe, *dump ? " to " : "", dump);
  return finite ? 0 : 1;
}
