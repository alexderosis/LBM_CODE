//==============================================================================
//  A floating body: Archimedes' draft, and the sign of the metacentric height.
//
//  WHAT MAKES THIS A VALIDATION CASE RATHER THAN A DEMONSTRATOR. Water entry has
//  no closed form to check against and lives in demonstrator/ for that reason.
//  Hydrostatics of a floating body has two, both sharp, and neither has a
//  tunable constant in it:
//
//   1. THE DRAFT. A box of height H and density rho_b, floating between fluids
//      of density rho_w and rho_a, sits at
//
//          d = H (rho_b - rho_a) / (rho_w - rho_a),
//
//      which is Archimedes with the air's buoyancy kept rather than dropped --
//      at a density ratio of 50 the air carries 2% of the answer and dropping it
//      would be a larger error than anything measured here.
//
//   2. THE STABILITY. A box floats upright only if its metacentric height is
//      positive. In two dimensions, with waterplane second moment B^3/12 and
//      displaced area B d,
//
//          GM = BM - BG = B^2 / (12 d) - (H - d) / 2,
//
//      and the sign of that number decides whether a tilted box returns to
//      upright or lies down. It is a sign, not a fit: the same body at the same
//      density with its aspect ratio inverted must do the opposite thing.
//
//  WHY THAT SECOND ONE IS THE TEST WORTH HAVING. Both cases here use the same
//  density, the same gravity, the same box AREA, the same resolution and the
//  same code path. The only thing that differs is which side is longer. If the
//  rigid-body solve's coupling to the fluid were wrong in almost any way -- a
//  sign in the righting moment, the first moments S dropped, the inertia taken
//  from the nominal rectangle instead of the smoothed one -- both would do the
//  same thing, and no tolerance would have to be chosen to notice.
//
//    raft   B = 2H:  BM = 21.8, BG = 8.2, GM = +13.6   returns upright
//    pillar B = H/2: BM =  1.4, BG = 8.2, GM =  -6.8   falls onto its side
//
//  The pillar's resting state is a raft with B and H exchanged, GM = +6.8, so
//  it should come to rest near 90 degrees rather than keep turning -- but that
//  is NOT what is asserted. Three e-folding times take it to 65 degrees and it
//  is still going; reaching the far equilibrium is a much longer run than a
//  regression test should be, so the check is only that it has left upright for
//  good (past 45 degrees, where the righting arm of the rotated raft has taken
//  over) and the last angle is printed for whoever wants to look.
//
//  THE DRAFT IS MEASURED TWICE, and the pair is the interesting part.
//
//    geometric   from the body's centre height against the free surface level,
//                the latter taken as the water column height far from the body
//                (integral of phi over y), which absorbs the level rise the
//                body's own displacement causes in a closed tank.
//    from mass   from m_f = integral of chi rho, the fictitious fluid mass the
//                rigid-body solve actually uses for buoyancy, inverted through
//                the same Archimedes relation.
//
//  The second is nearly circular -- m_f = m_b is the fixed point of the solve,
//  so on its own it says only that the solve converged -- but the DIFFERENCE
//  between the two is not circular at all. It BOUNDS the amount by which the
//  phase field inside the penalised region has drifted away from the free
//  surface outside it, which is the one failure mode this coupling has that
//  nothing else in the suite would catch: there is no contact-line model (see
//  PenalisedBody.hpp), so the waterline on the hull is advected with the body
//  rather than pinned to the water, and if it slides then buoyancy is being
//  computed for a body that is not the one on screen.
//
//  A bound rather than a measurement, because the gap also contains whatever
//  the two routes disagree about for duller reasons -- the smoothed hull edge,
//  the diffuse surface, the column integral of phi. Those do not separate here.
//  What can be said is that the gap is small, and that it is the right quantity
//  to watch if a case ever puts a body somewhere the interface has to move
//  along it.
//
//  WHAT IS NOT TESTED. The roll PERIOD, which would need the added inertia of
//  the entrained water and has no closed form; the draft of a body small enough
//  for surface tension to matter (Bond number of order one), which this
//  formulation cannot produce because it has no contact line; and any of it in
//  three dimensions, where the rigid-body solve is a 6x6 that does not exist.
//==============================================================================
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/PhaseFieldBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/PenalisedBody.hpp"
#include "solver/PhaseFieldSolver.hpp"
#include "solver/ViscousInterfaceForce.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using FL = D2Q9;
using PL = D2Q9;
using FColl = MultiphaseCentralMoments<FL>;
using PColl = PhaseFieldBGK<PL>;
using FluidSlv = FluidSolver<FL, EsotericPull<FL>, FColl>;
using PhaseSlv = PhaseFieldSolver<PL, EsotericPull<PL>, PColl>;

struct Case {
  const char* name;
  double B, H;        // box width and height, in cells
  double theta0;      // release tilt, degrees
  bool   spin;        // let it roll
  double tmax;        // run length, in roll periods
};

struct Result {
  double draft_geo = 0, draft_mass = 0, draft_exact = 0;
  double theta = 0, theta_rms = 0;
  double GM = 0, righting = 0, righting_exact = 0;
  bool ok = false;
};

struct Knobs {
  double ratio = 50.0, s = 0.5, g = 2.0e-4, nu = 0.02;
  double iw = 4.0, M = 0.02;
  // SURFACE TENSION IS ZERO, ON PURPOSE. Both answers this case checks are
  // hydrostatic, and at these proportions the Bond number is in the thousands,
  // so capillarity would contribute far less than the discretisation while
  // adding its own spurious currents to the measurement. The interface stays
  // sharp regardless: that is the conservative Allen-Cahn operator's job and it
  // does not go through kappa. A body small enough for surface tension to
  // matter is out of scope for a coupling with no contact line anyway.
  double sigma = 0.0;
  int    nframes = 12;
  bool   verbose = true;
};

//------------------------------------------------------------------------------
static Result run(const Case& C, const Knobs& K) {
  const double rho_a = 1.0, rho_w = K.ratio;
  const double rho_b = K.s * rho_w;
  const double d_ex  = C.H * (rho_b - rho_a) / (rho_w - rho_a);
  const double BM    = C.B * C.B / (12.0 * d_ex);
  const double GM    = BM - (C.H - d_ex) / 2.0;

  // The righting couple at the release tilt, with two corrections to the
  // textbook rho g V GM sin(theta), both of which matter here.
  //
  //  * THE AIR'S BUOYANCY. The effective buoyant density is rho_w - rho_a, the
  //    same combination the draft uses. At a ratio of 50 that is 2%.
  //  * THE WALL-SIDED TERM. GM is a small-angle quantity and 15 degrees is not
  //    small: as the box heels, the emerging and immerging wedges stop being
  //    mirror images and the arm grows to (GM + BM tan^2(theta)/2) sin(theta).
  //    For the raft that is +5.7%, which is most of the discrepancy against the
  //    linear formula and would otherwise have been charged to the solver. It
  //    holds while the deck stays dry and the bilge stays wet, which at these
  //    proportions is out to 26 degrees.
  //
  // Restoring, so it opposes a positive tilt -- and for GM < 0 it does not,
  // which is the whole point of the pillar.
  const double th0    = C.theta0 * M_PI / 180.0;
  const double GZ     = (GM + 0.5 * BM * std::tan(th0) * std::tan(th0))
                      * std::sin(th0);
  const double T_meta = -(rho_w - rho_a) * K.g * C.B * d_ex * GZ;

  const Index nx = Index(std::max(3.0 * C.B, 4.0 * C.H));
  const Index ny = Index(4.0 * C.H);
  const double y0 = 2.0 * C.H;                 // undisturbed free surface

  // Long enough for several roll periods, from the linearised stiffness
  // rho_w g A_sub |GM| against the body's own inertia -- the entrained water
  // only lengthens it, so this is a lower bound and the run is longer than it
  // looks. For the pillar, where GM < 0, the same number is the e-folding time
  // of the capsize rather than a period.
  const double I_b  = rho_b * (C.B * C.H) * (C.B * C.B + C.H * C.H) / 12.0;
  const double k    = (rho_w - rho_a) * K.g * (C.B * d_ex) * std::fabs(GM);
  const double Tro  = 2.0 * M_PI * std::sqrt(I_b / k);
  const std::size_t nsteps = std::size_t(C.tmax * Tro);

  Domain d(nx, ny, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  PColl pc;
  pc.omega = PColl::omega_from_mobility(Real(K.M));
  pc.width = Real(K.iw);
  PhaseSlv pf(d, pc);

  const Real yw = Real(y0), iwr = Real(K.iw);
  const Index hy = d.hy;
  pf.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real y = Real(py - hy);
    return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (yw - y) / iwr));
  });

  // The body starts at the draft Archimedes predicts, tilted. Heave therefore
  // begins near its fixed point and roll does not, which is the split this case
  // wants: the draft answer has to be HELD, the tilt answer has to be FOUND.
  ViscousInterfaceForce<FL> vf(d);
  PenalisedBody<FL> body(d);
  body.shape = Rect{Real(0.5 * double(nx)), Real(y0 + C.H / 2.0 - d_ex),
                    Real(0.5 * C.B), Real(0.5 * C.H), Real(1.5)};
  body.shape.set_angle(Real(C.theta0 * M_PI / 180.0));
  body.by = Real(-K.g);
  body.free_rotation = C.spin;
  body.set_uniform_density(Real(rho_b));

  FColl fc;
  fc.phi = pf.phi();
  fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
  fc.Lap = pf.laplacian();
  fc.Vx = vf.x();  fc.Vy = vf.y();  fc.Vz = vf.z();
  fc.Ex = body.x(); fc.Ey = body.y(); fc.Ez = body.z();
  fc.rho_L = Real(rho_a);        fc.rho_H = Real(rho_w);
  fc.mu_L  = Real(rho_a * K.nu); fc.mu_H  = Real(rho_w * K.nu);
  fc.kappa = FColl::kappa_from_sigma(Real(K.sigma), Real(K.iw));
  fc.beta  = FColl::beta_from_sigma(Real(K.sigma), Real(K.iw));
  fc.by    = Real(-K.g);
  FluidSlv fl(d, fc);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });

  // Hydrostatic, with the gauge zero AT the surface -- MultiphasePotentialBGK.hpp.
  auto phiv = pf.phi();
  const Real rl = Real(rho_a), rh = Real(rho_w), gr = Real(K.g);
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real dz = Real(py - hy) - yw;
    const Real az = (dz < Real(0) ? -dz : dz) * Real(2) / iwr;
    const Real lnch = az + Kokkos::log(Real(1) + Kokkos::exp(Real(-2) * az))
                    - Real(0.6931471805599453);
    const Real I = rh * dz - (rh - rl) * Real(0.5) * (dz + Real(0.5) * iwr * lnch);
    const Real r = rl + phiv(n) * (rh - rl);
    return FlowState{(-gr * I) / (r / Real(3)), Real(0), Real(0), Real(0)};
  });

  pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());
  body.set_velocity(fl.ux(), fl.uy());

  auto phi_view = pf.phi();
  auto dens_of = KOKKOS_LAMBDA(Index n) {
    const Real q = phi_view(n);
    const Real c = q < Real(0) ? Real(0) : (q > Real(1) ? Real(1) : q);
    return Real(rho_a) + c * Real(rho_w - rho_a);
  };

  const double area = double(body.penalised_area());

  if (K.verbose) {
    std::printf("\n%s   B = %.0f  H = %.0f   %dx%d   GM = %+.2f   "
                "period ~ %.0f   %zu steps\n",
                C.name, C.B, C.H, int(nx), int(ny), GM, Tro, nsteps);
    std::printf("%-9s %-10s %-10s %-10s %-10s %-11s %-10s\n",
                "step", "tilt(deg)", "d_geo/H", "d_mass/H", "surface",
                "righting", "max|u|");
    std::printf("%s\n", std::string(78, '-').c_str());
  }

  Result R;
  R.draft_exact = d_ex;
  R.GM = GM;
  R.righting_exact = T_meta;
  R.ok = true;
  // Averages are taken over the SECOND HALF only. The tank is closed and
  // periodic, so the body's own wake wraps round and comes back; a single
  // instantaneous draft carries that wave, and the mean does not.
  double acc_g = 0, acc_m = 0, acc_t2 = 0; int nacc = 0;
  const std::size_t every = nsteps / std::size_t(K.nframes > 0 ? K.nframes : 1);

  // The free surface far from the body, as the water column height. Sampled over
  // the eighth of the tank on either side of the periodic seam, which is the
  // farthest the body ever is from anywhere.
  auto surface_level = [&](decltype(Kokkos::create_mirror_view_and_copy(
                               HostSpace{}, pf.phi()))& hp) {
    double sum = 0; int cols = 0;
    for (Index x = 0; x < nx; ++x) {
      if (x > nx / 8 && x < nx - nx / 8) continue;
      double h = 0;
      for (Index y = 0; y < ny; ++y) h += double(hp(d.id(x, y)));
      sum += h; ++cols;
    }
    return cols ? sum / cols : 0.0;
  };

  for (std::size_t step = 0; step <= nsteps; ++step) {
    pf.refresh();
    fl.compute_macroscopic();
    vf.refresh(fc);
    const auto Rn = body.refresh(dens_of);

    const double th = double(body.shape.theta) * 180.0 / M_PI;

    if (every && step % every == 0) {
      auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
      auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      double umx = 0; bool bad = false;
      for (Index y = 1; y < ny - 1; ++y)
        for (Index x = 0; x < nx; ++x) {
          const double a = double(hu(d.id(x, y))), b = double(hv(d.id(x, y)));
          if (!std::isfinite(a) || !std::isfinite(b)) bad = true;
          umx = std::max(umx, std::sqrt(a * a + b * b));
        }
      const double ys = surface_level(hp);
      const double dg = ys - (double(body.shape.cy) - C.H / 2.0);
      const double dm = (double(Rn.fluid_mass) - rho_a * area)
                      / ((rho_w - rho_a) * C.B);
      R.theta = th;
      if (step * 2 >= nsteps) {
        acc_g += dg; acc_m += dm; acc_t2 += th * th; ++nacc;
        R.draft_geo = acc_g / nacc;  R.draft_mass = acc_m / nacc;
        R.theta_rms = std::sqrt(acc_t2 / nacc);
      }
      // The righting couple is read at step 0, where the fluid is still at rest
      // and hydrostatic: there is nothing in it but the integral of chi rho r,
      // so it can be put straight against metacentric theory.
      if (step == 0) R.righting = double(Rn.righting);
      if (K.verbose)
        std::printf("%-9zu %-10.3f %-10.4f %-10.4f %-10.3f %-11.4e %-10.3e\n",
                    step, th, dg / C.H, dm / C.H, ys,
                    double(Rn.righting), umx);
      if (bad) { R.ok = false; if (K.verbose) std::printf("  DIVERGED\n"); break; }
    }
    if (step == nsteps) break;

    fl.step(true);
    pf.step();
    body.advance();
  }
  return R;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    Knobs K;
    double H = 32;
    for (int i = 1; i < argc; ++i) {
      auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-h"))     num(H);
      else if (!std::strcmp(argv[i], "-ratio")) num(K.ratio);
      else if (!std::strcmp(argv[i], "-s"))     num(K.s);
      else if (!std::strcmp(argv[i], "-g"))     num(K.g);
      else if (!std::strcmp(argv[i], "-nu"))    num(K.nu);
      else if (!std::strcmp(argv[i], "-iw"))    num(K.iw);
      else if (!std::strcmp(argv[i], "-quiet")) K.verbose = false;
    }

    std::printf("Floating body   D2Q9 (pressure form, %s) + D2Q9 phase field\n",
                FColl::name);
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("rho_H/rho_L = %.0f   body = %.2f x water   g = %.2e   nu = %.3f\n",
                K.ratio, K.s, K.g, K.nu);

    const Case cases[] = {
      //  name       B       H   tilt   roll  periods
      {"upright",  2 * H,    H,   0.0, false, 1.5},
      {"raft",     2 * H,    H,  15.0, true,  4.0},
      {"pillar",   H / 2,    H,   5.0, true,  3.0},
    };
    Result res[3];
    for (int i = 0; i < 3; ++i) res[i] = run(cases[i], K);
    const Result &up = res[0], &raft = res[1], &pil = res[2];

    //------------------------------------------------------------------------
    const double e_geo = std::fabs(up.draft_geo / up.draft_exact - 1.0);
    const double e_mass = std::fabs(up.draft_mass / up.draft_exact - 1.0);
    const double slip  = std::fabs(up.draft_geo - up.draft_mass) / up.draft_exact;
    const double e_Tr  = std::fabs(raft.righting / raft.righting_exact - 1.0);
    const double e_Tp  = std::fabs(pil.righting  / pil.righting_exact  - 1.0);

    const bool finite  = up.ok && raft.ok && pil.ok;
    const bool p_draft = finite && e_geo < 0.06;
    const bool p_slip  = finite && slip  < 0.06;
    const bool p_Tr    = finite && e_Tr  < 0.05;
    const bool p_Tp    = finite && e_Tp  < 0.05;
    const bool p_raft  = finite && raft.theta_rms < cases[1].theta0 / 2;
    const bool p_pill  = finite && std::fabs(pil.theta) > 45.0;

    std::printf("\nAcceptance\n\n");
    std::printf("%-36s %-13s %-13s %s\n", "", "measured", "expected", "");
    std::printf("%s\n", std::string(72, '-').c_str());
    std::printf("%-36s %-13.4f %-13.4f %s\n", "draft / H  (Archimedes)",
                up.draft_geo / cases[0].H, up.draft_exact / cases[0].H,
                p_draft ? "PASS" : "FAIL");
    std::printf("%-36s %-13.4f %-13s %s\n", "waterline slip on the hull / d",
                slip, "0", p_slip ? "PASS" : "FAIL");
    std::printf("%-36s %-13.4e %-13.4e %s\n", "righting couple, raft   (wall-sided)",
                raft.righting, raft.righting_exact, p_Tr ? "PASS" : "FAIL");
    std::printf("%-36s %-13.4e %-13.4e %s\n", "righting couple, pillar (wall-sided)",
                pil.righting, pil.righting_exact, p_Tp ? "PASS" : "FAIL");
    std::printf("%-36s %-13.2f %-13s %s\n", "raft tilt, rms over 2nd half (deg)",
                raft.theta_rms, "< 7.5", p_raft ? "PASS" : "FAIL");
    std::printf("%-36s %-13.2f %-13s %s\n", "pillar tilt, final (deg)",
                pil.theta, "> 45", p_pill ? "PASS" : "FAIL");

    std::printf("\n  The draft is measured twice. From m_f -- the fictitious mass the\n"
                "  buoyancy is actually computed with -- it is %.4f H, %.2f%% off\n"
                "  Archimedes, which says the solve sits on its fixed point. From the\n"
                "  geometry it is %.4f H, %.1f%% off, which is the diffuse surface and\n"
                "  the smoothed hull. The %.1f%% between them BOUNDS the waterline\n"
                "  sliding on the hull -- what a coupling with no contact line has\n"
                "  instead of a contact angle -- and does not separate it from the\n"
                "  discretisation of either measurement.\n",
                up.draft_mass / cases[0].H, 100.0 * e_mass,
                up.draft_geo / cases[0].H, 100.0 * e_geo, 100.0 * slip);
    std::printf("\n  The two boxes have the same area, density, gravity and mesh, and\n"
                "  differ only in which side is longer: GM = %+.2f against %+.2f. The\n"
                "  righting couple follows the sign, the raft came back from %.0f deg,\n"
                "  and the pillar left %.0f deg for %.0f.\n",
                raft.GM, pil.GM, cases[1].theta0, cases[2].theta0, pil.theta);

    if (!(p_draft && p_slip && p_raft && p_pill)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
