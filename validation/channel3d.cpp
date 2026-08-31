//==============================================================================
//  Channel flow in PHYSICAL units -- 5 m x 1 m x 1 m, water, parabolic inlet.
//
//  A rectangular duct, length 5 m along x, height 1 m along y, breadth 1 m
//  along z. No-slip on the two y-normal planes, a fixed parabolic profile on
//  the west face, a constant back-pressure outflow on the east face, periodic
//  in z. The fluid is water (nu = 1.0e-6 m^2/s, rho = 1000 kg/m^3).
//
//  This is the one case in the tree that starts from METRES AND SECONDS and
//  carries the conversion out in the open, so it doubles as a worked example of
//  the "From a physical problem to lattice units" section of CLAUDE.md. The
//  conversion is the velocity-capping strategy: u_lat is chosen, which fixes
//  dt = u_lat * dx / u_phys, and omega follows. nu_lat is then computed twice --
//  once as nu_phys * dt / dx^2 and once as u_lat * N / Re -- and the two are
//  printed side by side. They are algebraically identical, so a mismatch means
//  the conversion has been edited wrongly, not that the physics is interesting.
//
//  WHAT THE FLOW IS. The inlet profile is the fully developed solution, so the
//  exact answer is that same parabola at every station:
//
//      ux(y) = 4 U (y/H) (1 - y/H),     dp/dx = -8 mu U / H^2.
//
//  Both are checked: the profile at the mid-section, and the streamwise density
//  gradient, which in lattice units is drho/dx = (dp/dx)_lat / cs2.
//
//  THE REYNOLDS NUMBER IS THE WHOLE DIFFICULTY. u_phys = 1 m/s in a 1 m channel
//  of water is Re = u H / nu = 1.0e6. Plane Poiseuille is linearly unstable
//  from Re ~ 5772 and transitions in practice near Re ~ 2000, and there is no
//  turbulence model in this tree, so the laminar parabola is NOT the physical
//  answer there -- it is the answer to the equations being solved. A run at
//  Re = 1.0e6 is therefore an under-resolved laminar computation, and it is
//  reported as such. Re is a command-line parameter precisely so the same case
//  can be run at a Re where the parabola IS the physics.
//
//  CONVERGENCE: THE SCALING DECIDES WHAT IS BEING MEASURED. This channel has a
//  fixed PHYSICAL aspect ratio, 5:1, so Nx grows with N and the density variation
//  along it does NOT shrink under refinement. Refining at fixed u0 therefore
//  measures the compressibility error, not the scheme:
//
//    u0      fitted rate   A (2nd-order coeff)   floor B     B / finest err
//    0.02      0.865         0.521               9.48e-4        0.78
//    0.005     1.888         0.589               2.49e-5        0.17
//
//  A is the same in both -- it belongs to the scheme -- while the non-refining
//  floor B falls ~50x for a 4x smaller u0, which is steeper than u0^2. The
//  exponent is NOT claimed: poiseuille_inlet.cpp declines to pin it from a
//  two-point extraction and there is no more information here.
//
//  Under DIFFUSIVE scaling -- tau held fixed, so u0 = (tau-1/2) cs2 Re / N falls
//  as 1/N and the Mach error falls as 1/N^2, the same rate as discretisation --
//  the ladder measures the order of the scheme and no floor appears at all
//  (fitted B < 0). That is the scaling to use for a convergence claim here.
//
//  Note that poiseuille_inlet.cpp holds Lx = 21 FIXED while Ly grows, so its
//  aspect ratio shrinks and its compressibility error falls as 1/Ly^2. Its
//  fitted floor at the same U0 = 0.02 is 2.1e-5, 50x below the one above. Both
//  numbers are right; the geometries are not comparable.
//
//  INITIALISATION IS NOT A DETAIL. From rest, the transient is momentum
//  diffusion across the channel, tau_d = N^2 / nu_lat = N Re / u_lat steps --
//  4e8 steps at N = 9, Re = 1e6, u_lat = 0.02. That is not affordable and it is
//  not what a high-Re run can test anyway. So -init parabola seeds the exact
//  solution and asks whether the scheme MAINTAINS it, which probes the wall,
//  inlet and outlet closures but not the evolution. -init rest is the stronger
//  test and is the one to use at moderate Re; the clock then printed as
//  t/tau_d is the one that matters, and a run that stops below 2 is not
//  converged whatever the residual says (see validation/poiseuille_inlet.cpp,
//  which measured a run quitting at 0.03 tau_d with a 400x error).
//==============================================================================
#include "Campaign.hpp"
#include "boundary/Regularized.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;
using namespace campaign;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    //-------------------------------------------------------------------------
    // The physical problem. Metres, seconds, kilograms.
    //-------------------------------------------------------------------------
    double Lx_phys = 5.0;        // m, streamwise
    double H_phys  = 1.0;        // m, wall to wall
    double W_phys  = 1.0;        // m, periodic breadth
    double u_phys  = 1.0;        // m/s, peak of the inlet parabola
    double nu_phys = 1.0e-6;     // m^2/s, water at 20 C
    double rho_phys = 1000.0;    // kg/m^3

    //-------------------------------------------------------------------------
    // The numerical choices.
    //-------------------------------------------------------------------------
    Index Ny   = 10;             // grid POINTS across the height, walls included
    Real  u0   = Real(0.02);     // peak inlet velocity in lattice units
    std::string lat = "d3q27", op = "cm", init = "parabola";
    std::string walls = "bb", outlet = "pressure";
    double t_cap = 400.0;        // stop after this many advective times
    double t_min = 20.0;         // never stop before this many
    int    oo    = 1;            // outflow tangential extrapolation order
    bool   dump  = false;
    Index  nz_override = 0;      // 0 = from the geometry; 1 = exploit z-invariance
    double res_tol = 1e-11;      // whole-field residual that counts as steady
    double err_tol = 1e-5;       // relative change in the REPORTED error that counts
    int    n_avg = 240;          // time-average samples taken AFTER the transient
    int    avg_stride = 37;      // steps between samples (see the banner note)
    int    servo_every = 200;    // steps between mass-servo updates
    std::size_t fevery = 0;      // >0: dump a mid z-plane frame every this many steps
    std::size_t probe_override = 0;  // >0: fix the probe interval in steps
    double servo_kp = 1.0, servo_ki = 0.05;   // PI gains, -outlet servo
    bool   verbose = false;      // per-probe progress, flushed
    double re_override = 0.0;    // if set, overrides nu_phys via Re

    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-ny"   && i + 1 < argc) Ny = std::atoi(argv[++i]);
      if (a == "-u0"   && i + 1 < argc) u0 = Real(std::atof(argv[++i]));
      if (a == "-re"   && i + 1 < argc) re_override = std::atof(argv[++i]);
      if (a == "-uphys"&& i + 1 < argc) u_phys = std::atof(argv[++i]);
      if (a == "-nuphys" && i + 1 < argc) nu_phys = std::atof(argv[++i]);
      if (a == "-rhophys" && i + 1 < argc) rho_phys = std::atof(argv[++i]);
      if (a == "-lat"  && i + 1 < argc) lat = argv[++i];
      if (a == "-op"   && i + 1 < argc) op  = argv[++i];
      if (a == "-init" && i + 1 < argc) init = argv[++i];
      if (a == "-walls"  && i + 1 < argc) walls = argv[++i];   // bb | reg
      if (a == "-outlet" && i + 1 < argc) outlet = argv[++i];  // pressure | copy
      if (a == "-tcap" && i + 1 < argc) t_cap = std::atof(argv[++i]);
      if (a == "-tmin" && i + 1 < argc) t_min = std::atof(argv[++i]);
      if (a == "-oo"   && i + 1 < argc) oo = std::atoi(argv[++i]);
      if (a == "-nz"   && i + 1 < argc) nz_override = std::atoi(argv[++i]);
      if (a == "-res"  && i + 1 < argc) res_tol = std::atof(argv[++i]);
      if (a == "-etol" && i + 1 < argc) err_tol = std::atof(argv[++i]);
      if (a == "-navg" && i + 1 < argc) n_avg = std::atoi(argv[++i]);
      if (a == "-astride" && i + 1 < argc) avg_stride = std::atoi(argv[++i]);
      if (a == "-servo" && i + 1 < argc) servo_every = std::atoi(argv[++i]);
      if (a == "-fevery" && i + 1 < argc) fevery = std::strtoull(argv[++i], nullptr, 10);
      if (a == "-probe"  && i + 1 < argc) probe_override = std::strtoull(argv[++i], nullptr, 10);
      if (a == "-kp" && i + 1 < argc) servo_kp = std::atof(argv[++i]);
      if (a == "-ki" && i + 1 < argc) servo_ki = std::atof(argv[++i]);
      if (a == "-v")    verbose = true;
      if (a == "-dump") dump = true;
    }

    // Re is a property of the physical problem. Allowing -re to set it means
    // solving for the viscosity that would produce it, so that the printed
    // fluid is always consistent with the printed Reynolds number.
    // -re solves for the viscosity that would give that Reynolds number, so the
    // printed fluid always matches the printed Re. Given -nuphys, the fluid is
    // the fixed thing and Re is whatever it is -- do not overwrite it.
    bool nu_given = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "-nuphys") == 0) nu_given = true;
    if (re_override > 0.0 && !nu_given) nu_phys = u_phys * H_phys / re_override;
    else if (re_override > 0.0 && nu_given)
      std::printf("  NOTE: -nuphys given, so -re is ignored; Re follows from u and nu.\n");
    const double Re = u_phys * H_phys / nu_phys;

    //-------------------------------------------------------------------------
    // Physical -> lattice. dx is fixed by the resolution; dt by capping u_lat.
    //-------------------------------------------------------------------------
    // WALL CONVENTION. The two treatments do not place the wall in the same
    // place, and the channel height in cells differs by one:
    //
    //   reg  regularised velocity wall. The no-slip plane sits ON the boundary
    //        node, so the walls are y = 0 and y = Ny-1 and H = (Ny-1) dx.
    //   bb   halfway bounce-back. y = 0 and y = Ny-1 are SOLID, the fluid is
    //        y = 1..Ny-2, and the no-slip planes fall halfway into the solid at
    //        y = 0.5 and y = Ny-1.5, so H = (Ny-2) dx -- the number of fluid
    //        nodes, which is the "ny = H + 2" convention validation/poiseuille.cpp
    //        and CLAUDE.md's Wall-conventions invariant both refer to.
    //
    // Getting this wrong rescales Re by (1 +- 1/N) -- 11% at Ny = 10 -- and
    // nothing fails.
    const bool bb   = (walls == "bb");
    const Index N   = bb ? Ny - 2 : Ny - 1;        // cell spacings across H
    const Index yf0 = bb ? 1 : 0;                  // first fluid row
    const Index yf1 = bb ? Ny - 2 : Ny - 1;        // last fluid row
    const double y_off = bb ? 0.5 : 0.0;           // wall plane, in cells
    const double dx = H_phys / double(N);          // m per cell
    const double dt = double(u0) * dx / u_phys;    // s per step
    const Index Nx = Index(std::llround(Lx_phys / dx)) + 1;   // walls are nodes
    // The exact solution has no z dependence and z is periodic, so Nz changes
    // the cost and nothing else. -nz 1 makes a convergence sweep affordable;
    // running the same point at both Nz is then a real check that the 3D
    // machinery is not adding anything. (poiseuille_inlet.cpp makes the same
    // argument for its Lz = 1 deviation from the paper's 21.)
    const Index Nz = nz_override > 0 ? nz_override
                                     : Index(std::llround(W_phys / dx));  // periodic: no +1

    const double nu_lat_from_dt = nu_phys * dt / (dx * dx);
    const double nu_lat_from_Re = double(u0) * double(N) / Re;
    const double mu_phys = rho_phys * nu_phys;
    const double dpdx_phys = -8.0 * mu_phys * u_phys / (H_phys * H_phys);  // Pa/m

    std::printf("Channel, inlet driven, physical units   lattice %s   operator %s\n", lat.c_str(), op.c_str());
    std::printf("  walls %s   outlet %s\n",
                bb ? "halfway bounce-back (solid rows, H = Ny-2)"
                   : "regularised velocity (H = Ny-1)",
                outlet == "copy" ? "zero-gradient populations, f(Nx-1) = f(Nx-2)"
                                 : "imposed back-pressure rho = 1");
    std::printf("  sizeof(Real) = %zu bytes%s\n", sizeof(Real),
                sizeof(Real) == 4 ? "   <-- FP32: tau-1/2 below ~1e-6 is unresolvable" : "");
    std::printf("\n  PHYSICAL\n");
    std::printf("    box              %.3f x %.3f x %.3f m  (x streamwise, y walls, z periodic)\n",
                Lx_phys, H_phys, W_phys);
    std::printf("    fluid            nu = %.4g m^2/s   rho = %.1f kg/m^3   mu = %.4g Pa s\n",
                nu_phys, rho_phys, mu_phys);
    std::printf("    inlet peak       %.4g m/s\n", u_phys);
    std::printf("    Re = u H / nu    %.4g%s\n", Re,
                Re > 2000.0 ? "   <-- ABOVE TRANSITION (~2000); no turbulence model here" : "");
    std::printf("    dp/dx (exact)    %.6g Pa/m\n", dpdx_phys);
    std::printf("\n  CONVERSION  (cap u_lat, let dt follow)\n");
    std::printf("    dx               %.6g m/cell        grid %d x %d x %d = %lld cells%s\n",
                dx, int(Nx), int(Ny), int(Nz),
                (long long)Nx * (long long)Ny * (long long)Nz,
                nz_override > 0 ? "   [nz overridden]" : "");
    std::printf("    dt               %.6g s/step\n", dt);
    std::printf("    u_lat            %.6g            Ma = u/cs = %.4f\n",
                double(u0), double(u0) * std::sqrt(3.0));
    std::printf("    nu_lat           %.6e  (from dt/dx)\n", nu_lat_from_dt);
    std::printf("                     %.6e  (from u0 N / Re)   rel diff %.2e\n",
                nu_lat_from_Re,
                std::abs(nu_lat_from_dt - nu_lat_from_Re) / std::max(nu_lat_from_Re, 1e-300));

    if (Nx < 4 || Ny < 4 || Nz < 1) {
      std::printf("\n  grid too small\n"); Kokkos::finalize(); return 1;
    }

    //-------------------------------------------------------------------------
    const double T_adv = double(Nx) / double(u0);          // steps per pass
    const double tau_d = double(N) * Re / double(u0);      // diffusive steps
    double err = NAN, resid = NAN, secs = 0, tau_print = NAN;
    double u_peak_num = NAN, dpdx_num = NAN, flux_spread = NAN, flux_interior = NAN;
    double u_peak_exact_node = NAN; bool peak_is_node = false;
    std::size_t taken = 0;
    bool blew = false;
    Index blow_x = -1, blow_y = -1; const char* blow_where = "?";
    double drift_out = NAN, err_mean = NAN, err_lo = NAN, err_hi = NAN;
    double rho_mean = NAN, rho_lo = NAN, rho_hi = NAN;
    double servo_rho_out = NAN, servo_err_out = NAN, mass_rel_err = NAN;
    const bool is_servo = (outlet == "servo" || outlet == "freeservo");
    int frames_written = 0;
    std::size_t n_samples = 0;

    const bool ok = dispatch(lat, op, [&](auto coll) {
      using Coll = decltype(coll);
      using LL   = typename Coll::Lattice;
      Domain d(Nx, Ny, Nz, /*periodic x*/ false, /*y*/ false, /*z*/ true);
      coll.omega = Coll::omega_from_viscosity(Real(nu_lat_from_dt));
      tau_print = 1.0 / double(coll.omega);
      FluidSolver<LL, EsotericPull<LL>, Coll> s(d, coll);

      const bool bbc = bb;
      const Index Nyc = Ny, Nxc = Nx;
      s.set_geometry([&](Index, Index y, Index) -> CellType {
        if (bbc && (y == 0 || y == Nyc - 1)) return Solid;   // bounce-back walls
        return Fluid;
      });
      using WS = typename decltype(s)::WallSpec;
      const Real U0c = u0;
      const Index Nl = N; const double yoff = y_off;
      // Two readings of "the populations at Nx-1 equal those at Nx-2":
      //
      //   copy     ALL Q directions are replaced. The outlet node becomes a
      //            clone of its neighbour. This also discards the populations
      //            that legitimately streamed in FROM the interior, so the node
      //            is over-determined -- interior information that has arrived
      //            is thrown away and replaced by a copy.
      //   copyunk  only the UNKNOWN directions are replaced, i.e. those with
      //            c_x < 0, which are the ones that would have streamed in from
      //            x = Nx (outside the domain) and are the only ones actually
      //            missing. Everything the interior sent is kept. This is the
      //            textbook zero-gradient outflow.
      const bool copy_out = (outlet == "copy" || outlet == "copyunk");
      const bool copy_all = (outlet == "copy");
      // -outlet servo: NrmOutFree, whose imposed density is the scalar
      // outflow_rho_ rather than a per-node table value, so the driver can move
      // it every step without rebuilding anything. A PI loop then drives it so
      // that TOTAL BULK MASS stays at its initial value. That is what
      // set_outflow_density's banner is for: pinning the outflow density to a
      // constant makes the boundary a mass source and sink, because the nodes
      // are overwritten every step and whatever rho is imposed is injected
      // regardless of what arrived. Letting total mass choose the level instead
      // makes the boundary conserving, and removes the arbitrariness of rho = 1.
      // Two servo targets, because the knob and the closure are separable:
      //
      //   freeservo  NrmOutFree + set_outflow_density(). The knob is a scalar
      //              captured fresh each step, so this is cheap -- but the
      //              closure is wrong for a flat face. Measured here: the outlet
      //              column runs 110% high and uniform in y, all-station flux
      //              spread 0.66 against 7e-5 in the interior. NrmOutFree is
      //              built for a voxelised oblique cap, where NrmOutXp's
      //              one-open-axis partition is what fails instead.
      //   servo      NrmOutXp, the right closure for a flat face (outlet column
      //              ~3% high, mass-neutral to 1e-6 at rho = 1). Its density
      //              lives in the deduplicated wall TABLE, not in a scalar, so
      //              moving it means re-issuing set_regularized_walls. That
      //              rebuilds the node lists, so the loop runs every few hundred
      //              steps rather than every step -- which is ample, because
      //              what is being corrected is a slow gauge drift.
      const bool servo_xp   = (outlet == "servo");
      const bool servo_free = (outlet == "freeservo");
      const bool servo = servo_xp || servo_free;
      double servo_rho = 1.0;
      // -outlet identity keeps the back-pressure wall but still runs the
      // gather/seed round trip. It exists to separate a bug in that plumbing
      // from a defect in the boundary condition: the round trip is an exact
      // inverse pair, so an identity run MUST reproduce -outlet pressure. If it
      // does not, the fault is here and not in the physics.
      const bool round_trip = copy_out || (outlet == "identity");
      // eta in [0,1] across the channel, per the wall convention above.
      auto eta_of = [=](Index y) { return (double(y) - yoff) / double(Nl); };
      auto wallfn = [&](Index x, Index y, Index) -> WS {
        // A solid node must be left alone: returning any normal would promote it
        // to RegWall and delete the bounce-back wall.
        if (bbc && (y == 0 || y == Nyc - 1)) return WS{};
        if (!bbc) {
          // Walls are tested BEFORE the inlet, so an inlet/wall edge node is a
          // wall. The parabola vanishes there anyway; what this decides is
          // which closure runs on the edge.
          if (y == 0)       return WS{NrmYm, Real(0), Real(0), Real(0), Real(1)};
          if (y == Nyc - 1) return WS{NrmYp, Real(0), Real(0), Real(0), Real(1)};
        }
        if (x == 0) {
          const double e = eta_of(y);
          return WS{NrmXm, Real(4.0 * double(U0c) * e * (1.0 - e)), Real(0), Real(0), Real(1)};
        }
        // The copy outlet is applied AFTER each step, not through this table:
        // it replaces the node's populations wholesale rather than closing for
        // a macroscopic unknown, so it is not expressible as a WallSpec.
        if (x == Nxc - 1 && servo_free)
          return WS{NrmOutFree, Real(0), Real(0), Real(0), Real(1)};
        if (x == Nxc - 1 && outlet == "free")
          return WS{NrmOutFree, Real(0), Real(0), Real(0), Real(1)};
        if (x == Nxc - 1 && !copy_out)
          return WS{NrmOutXp, Real(0), Real(0), Real(0), Real(servo_rho)};
        return WS{};
      };
      s.set_regularized_walls(wallfn);
      s.set_outflow_order(oo);

      if (init == "rest") {
        s.initialize(Real(1));
      } else {
        // The fully developed state is the parabola AND a linear pressure drop.
        // Seeding rho = 1 uniformly instead is not a small error: it launches an
        // acoustic transient whose slowest mode decays as 1/(nu k^2) with
        // k = pi/Nx, i.e. ~5e5 steps at nu_lat = 8e-3, Nx = 201. Measured: the
        // whole-field residual then stalls, oscillating around 1e-2 for
        // thousands of steps instead of falling, and the run hits its cap.
        //
        //   u_max = (-dp/dx) H^2 / (8 mu)   =>   -dp/dx = 8 nu u0 / N^2   (rho0=1)
        //   drho/dx = (dp/dx) / cs2
        //
        // rho = 1 is imposed at the outlet, so the ramp is measured back from it.
        const Index Nxi = Nx;
        const Real drdx = Real(8.0 * nu_lat_from_dt * double(u0)
                               / (double(N) * double(N) * double(cs2<LL, Real>())));
        const Real Nlr = Real(Nl), yoffr = Real(yoff);
        s.initialize_field(KOKKOS_LAMBDA(Index n) {
          Index px, py, pz; d.coords(n, px, py, pz);
          const Real e  = (Real(py - d.hy) - yoffr) / Nlr;
          const Real xr = Real(Nxi - 1 - (px - d.hx));   // cells upstream of outlet
          return FlowState{Real(1) + drdx * xr,
                           Real(4) * U0c * e * (Real(1) - e), Real(0), Real(0)};
        });
      }

      // Probe often enough that a run seeded with the exact solution can stop
      // early. With -init parabola the transient is ACOUSTIC (Nx/cs steps), not
      // advective (Nx/u_lat), and the two differ by 1/Ma ~ 30, so a probe tied
      // to T_adv/4 would overshoot convergence by more than an order of
      // magnitude on the fine grids and dominate the cost of a sweep.
      //-----------------------------------------------------------------------
      // ZERO-GRADIENT POPULATION OUTFLOW, -outlet copy.
      //
      // After each step the populations arriving at x = Nx-1 are replaced by
      // those arriving at x = Nx-2, for every direction. This is applied
      // OUTSIDE the WallSpec table because it replaces the node's state
      // wholesale instead of closing for a macroscopic unknown, so there is no
      // normal code that can express it.
      //
      // It goes through gather_populations()/seed_populations() rather than
      // touching the arrays directly. Those two are exact inverses and they
      // resolve the streaming parity themselves, so the same code is correct for
      // Esoteric Pull and TwoLattice -- which a raw array copy would not be,
      // because Esoteric Pull's stored value depends on the direction and the
      // step parity. The cost is two extra full passes and one allocation per
      // step; their own banner says they are "not on any hot path", and that is
      // why the copy runs are kept to nz = 1.
      //
      // NOTE ON WELL-POSEDNESS: this outlet imposes nothing on rho, so nothing
      // pins the pressure level. Regularized.hpp records rho -> 181 for a
      // zero-gradient outlet, but that was with REGULARISED walls, which
      // overwrite populations and so are a mass source. With bounce-back walls
      // the only non-conserving node left is the inlet. Whether that is enough
      // is measured below, not assumed.
      //-----------------------------------------------------------------------
      const Index Nxa = Nx;

      // Bulk mass over the interior columns only: x = 0 and x = Nx-1 are
      // boundary nodes whose rho is imposed or closed for, so including them
      // would feed the controller its own output.
      auto bulk_mass = [&]() {
        s.compute_macroscopic();
        auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
        double m = 0;
        for (Index z = 0; z < Nz; ++z)
          for (Index y = yf0; y <= yf1; ++y)
            for (Index x = 1; x < Nx - 1; ++x) m += double(hr(d.id(x, y, z)));
        return m;
      };
      const double n_bulk = double(Nz) * double(yf1 - yf0 + 1) * double(Nx - 2);
      const double M0 = bulk_mass();
      double integ = 0.0, last_e = NAN;
      std::size_t since_servo = 0;

      auto advance = [&]() {
        s.step();
        if (servo && ++since_servo >= std::size_t(servo_every)) {
          since_servo = 0;
          // e is the MEAN density excess. Raising the outlet density raises the
          // back pressure, cuts the pressure drop and so cuts the outflow, which
          // makes mass accumulate -- hence the correction is negative in e.
          //
          // The integral term is normalised by the advective time so that the
          // gain means the same thing on every grid. Accumulating raw e*dt
          // instead wound up: at ki = 0.05 the loop drove the near-wall row
          // unstable in 6526 steps.
          const double e = (bulk_mass() - M0) / n_bulk;
          integ += e * double(servo_every) / T_adv;
          servo_rho = 1.0 - servo_kp * e - servo_ki * integ;
          last_e = e;
          if (servo_free) s.set_outflow_density(Real(servo_rho));
          else            s.set_regularized_walls(wallfn);   // rebuilds the table
        }
        if (!round_trip) return;
        const bool modify = copy_out; const bool mall = copy_all;
        auto g = s.gather_populations();
        s.seed_populations(KOKKOS_LAMBDA(Index n, int i) {
          Index px, py, pz; d.coords(n, px, py, pz);
          if (modify && d.is_interior(px, py, pz) && (px - d.hx) == Nxa - 1 &&
              (mall || cvel<LL>(i, 0) < 0))
            return g(d.id(Nxa - 2, py - d.hy, pz - d.hz), i);
          return g(n, i);
        });
      };

      // Frames are written at probe boundaries, so the probe also sets the
      // animation's frame rate; -probe overrides it for that purpose.
      const std::size_t probe = probe_override > 0 ? probe_override :
          std::min<std::size_t>(5000, std::max<std::size_t>(200, std::size_t(T_adv / 20.0)));
      const std::size_t tmin  = std::size_t(t_min * T_adv);
      const std::size_t cap   = std::size_t(t_cap * T_adv);
      std::vector<double> prev(std::size_t(Nx) * std::size_t(Ny), 0.0);
      double res = NAN;
      Index bad_x = -1, bad_y = -1;
      const Index xmid = Nx / 2;
      std::size_t next_frame = 0; int frame_no = 0;
      char frame_tag[512];
      std::snprintf(frame_tag, sizeof frame_tag,
                    "results/I_channel/anim_ny%d_nz%d_re%.0e_u%.0e_%s",
                    int(Ny), int(Nz), Re, u_phys, walls.c_str());
      double err_now = NAN, err_prev = 1e300, err_drift = NAN;
      std::vector<double> err_hist;

      const auto t0 = std::chrono::steady_clock::now();
      for (std::size_t t = 0; t < cap; t += probe) {
        for (std::size_t k = 0; k < probe; ++k) advance();
        taken += probe;
        s.compute_macroscopic();
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        double num = 0, den = 0; bool bad = false;
        for (Index y = 0; y < Ny && !bad; ++y)
          for (Index x = 0; x < Nx; ++x) {
            const std::size_t k = std::size_t(y) * std::size_t(Nx) + std::size_t(x);
            const double v = double(h(d.id(x, y, Nz / 2)));
            // A blow-up here is a real outcome, not an error: report it.
            if (!std::isfinite(v) || std::abs(v) > 1.0) {
              bad = true; bad_x = x; bad_y = y; break;
            }
            const double dv = v - prev[k];
            num += dv * dv; den += v * v;
            prev[k] = v;
          }
        if (bad) { blew = true; break; }
        res = std::sqrt(num / std::max(den, 1e-300));
        // Converge on the QUANTITY BEING REPORTED, not on a proxy. The
        // whole-field residual plateaus near 5e-4 here: the constant-pressure
        // outlet reflects weakly and the standing wave it leaves never damps on
        // an affordable timescale. But that wave is ~1e-4 of the velocity and
        // does not move the mid-section error at all, so demanding res < 1e-11
        // just burns 40 minutes to change no reported digit. The mid-section L2
        // error going stationary is the criterion that means what it says.
        {
          double a2 = 0, b2 = 0;
          for (Index y = yf0; y <= yf1; ++y) {
            const double e = (double(y) - y_off) / double(N);
            const double an = 4.0 * double(u0) * e * (1.0 - e);
            const double dv = double(h(d.id(xmid, y, Nz / 2))) - an;
            a2 += dv * dv; b2 += an * an;
          }
          err_now = std::sqrt(a2 / b2);
          err_drift = std::abs(err_now - err_prev) / std::max(err_now, 1e-300);
          err_prev = err_now;
        }
        // Progress must be FLUSHED. stdout is block-buffered when redirected, so
        // a long run writes nothing at all until it exits and there is no way to
        // tell a converging run from a stuck one.
        if (verbose) {
          std::printf("    [%8zu steps  %.2f passes  residual %.3e  relL2 %.6e  drift %.2e]\n",
                      taken, double(taken) / T_adv, res, err_now, err_drift);
          std::fflush(stdout);
        }
        // The reported error does not settle to a point, it OSCILLATES: the
        // constant-pressure outlet reflects weakly, and the standing wave that
        // leaves modulates the mid-section L2 by ~8% about a stable mean while
        // the mean itself stops moving after ~1 pass. So no drift threshold can
        // fire, and chasing one runs to the cap. Collect the samples instead and
        // report mean and spread over the second half of the run -- which is
        // what "converged" honestly means for this quantity.
        err_hist.push_back(err_now);

        // TRANSIENT FRAMES. Written from the same host mirror the residual
        // already needed, so a frame is an fprintf and nothing more. These are
        // instantaneous snapshots on purpose -- the whole point is the time
        // evolution, so they must NOT be time-averaged the way the final
        // diagnostics are.
        if (fevery > 0 && taken >= next_frame) {
          next_frame = taken + fevery;
          auto hrf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
          char fp[600];
          std::snprintf(fp, sizeof fp, "%s_frame%05d.dat", frame_tag, frame_no);
          if (std::FILE* ff = std::fopen(fp, "w")) {
            std::fprintf(ff, "# step=%zu t_s=%.8e nx=%d ny=%d dx=%.8e uscale=%.8e\n",
                         taken, double(taken) * dt, int(Nx), int(yf1 - yf0 + 1), dx,
                         dx / dt);
            for (Index y = yf0; y <= yf1; ++y)
              for (Index x = 0; x < Nx; ++x) {
                const Index nn = d.id(x, y, Nz / 2);
                std::fprintf(ff, "%.7e %.7e\n", double(h(nn)), double(hrf(nn)));
              }
            std::fclose(ff);
          }
          ++frame_no;
        }
        if (taken >= tmin && res < res_tol) break;
      }
      secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      resid = res; drift_out = err_drift; frames_written = frame_no;
      servo_rho_out = servo_rho; servo_err_out = last_e;
      mass_rel_err = (bulk_mass() - M0) / M0;
      if (!err_hist.empty()) {
        const std::size_t h0 = err_hist.size() / 2;         // second half only
        double sum = 0, lo = 1e300, hi = -1e300;
        for (std::size_t k = h0; k < err_hist.size(); ++k) {
          sum += err_hist[k];
          lo = std::min(lo, err_hist[k]); hi = std::max(hi, err_hist[k]);
        }
        err_mean = sum / double(err_hist.size() - h0);
        err_lo = lo; err_hi = hi; n_samples = err_hist.size() - h0;
      }
      if (blew) {
        blow_x = bad_x; blow_y = bad_y;
        const bool at_in = bad_x <= 1, at_out = bad_x >= Nx - 2;
        const bool at_wall = bad_y <= 1 || bad_y >= Ny - 2;
        blow_where = at_in ? "inlet column" : at_out ? "outlet column"
                   : at_wall ? "wall row" : "BULK";
        return;
      }

      //-----------------------------------------------------------------------
      // TIME AVERAGE, then diagnostics and dumps.
      //
      // A snapshot does not measure the scheme here, it measures a wave. The
      // constant-pressure outlet reflects weakly, and the standing acoustic
      // mode it leaves damps as 1/(nu k^2) with k = pi/Nx -- about 5e5 steps at
      // nu_lat = 8e-3, Nx = 201, which no affordable run reaches. Its amplitude
      // is only ~1e-4 in lattice velocity, but the peak velocity is u0 = 0.02,
      // so it is ~5e-3 RELATIVE and it swamps the discretisation error on any
      // grid finer than about N = 30. Measured, with snapshots: the
      // mid-section L2 stopped falling and then rose, 1.82e-3 (N=18) ->
      // 8.32e-4 (N=36) -> 1.08e-3 (N=72), an apparent error floor 50x above the
      // one poiseuille_inlet.cpp fits for the same U0. Averaging over ~10
      // acoustic periods removes it and restores the expected rate.
      //
      // avg_stride is 37: the acoustic period is ~2 Nx / cs, and a stride that
      // divides it would sample the same phase every time and average to the
      // snapshot it is meant to replace.
      //-----------------------------------------------------------------------
      const Index xm = Nx / 2, zm = Nz / 2;
      const std::size_t nplane = std::size_t(Nx) * std::size_t(Ny);
      std::vector<double> AX(nplane, 0.0), AY(nplane, 0.0), AR(nplane, 0.0);
      for (int k = 0; k < n_avg; ++k) {
        for (int j = 0; j < avg_stride; ++j) advance();
        taken += std::size_t(avg_stride);
        s.compute_macroscopic();
        auto ax = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto ay = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        auto ar = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
        for (Index y = 0; y < Ny; ++y)
          for (Index x = 0; x < Nx; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(Nx) + std::size_t(x);
            const Index n = d.id(x, y, zm);
            AX[i] += double(ax(n)); AY[i] += double(ay(n)); AR[i] += double(ar(n));
          }
      }
      for (std::size_t i = 0; i < nplane; ++i) {
        AX[i] /= double(n_avg); AY[i] /= double(n_avg); AR[i] /= double(n_avg);
      }
      auto AXf = [&](Index x, Index y) { return AX[std::size_t(y)*std::size_t(Nx)+std::size_t(x)]; };
      auto AYf = [&](Index x, Index y) { return AY[std::size_t(y)*std::size_t(Nx)+std::size_t(x)]; };
      auto ARf = [&](Index x, Index y) { return AR[std::size_t(y)*std::size_t(Nx)+std::size_t(x)]; };

      double num = 0, den = 0, pk = -1e300; Index pky = 0;
      for (Index y = yf0; y <= yf1; ++y) {         // fluid rows only
        const double e = eta_of(y);
        const double ana = 4.0 * double(u0) * e * (1.0 - e);
        const double v = AXf(xm, y);
        num += (v - ana) * (v - ana); den += ana * ana;
        if (v > pk) { pk = v; pky = y; }
      }
      err = std::sqrt(num / den);
      u_peak_num = pk;
      // The analytic value AT THE SAMPLED NODE. Comparing the discrete maximum
      // against the true peak conflates the scheme's error with the fact that
      // the centreline need not be a grid node: at Ny = 10 the centre is
      // y = 4.5 and no node can do better than -1.23%.
      {
        const double e = eta_of(pky);
        u_peak_exact_node = 4.0 * double(u0) * e * (1.0 - e);
        // The centreline is at eta = 1/2, i.e. y = y_off + N/2. That is a grid
        // node when N is even for the regularised convention (y_off = 0) and
        // when N is odd for bounce-back (y_off = 1/2).
        const double yc = y_off + 0.5 * double(N);
        peak_is_node = std::abs(yc - std::round(yc)) < 1e-12;
      }

      // Streamwise density gradient, fitted by least squares over the interior
      // stations only: the two boundary columns run a different closure.
      {
        double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
        for (Index x = 2; x < Nx - 2; ++x) {
          const double r = ARf(x, Ny / 2);
          sx += double(x); sy += r; sxx += double(x) * double(x); sxy += double(x) * r; ++n;
        }
        const double slope = (double(n) * sxy - sx * sy) / (double(n) * sxx - sx * sx);
        // p = rho cs2, so dp/dx = cs2 drho/dx. cs2 is read off the LATTICE and
        // not written as 1/3: it is 1/4 on D3Q7, and a copied literal would be
        // wrong by 4/3 without failing anything.
        dpdx_num = slope * double(cs2<LL, Real>());
      }

      // MEAN DENSITY. With -outlet copy nothing imposes rho, so the pressure
      // level is free to drift; this is the number that says whether it did.
      {
        double sum = 0, lo = 1e300, hi = -1e300; std::size_t cnt = 0;
        for (Index y = yf0; y <= yf1; ++y)
          for (Index x = 0; x < Nx; ++x) {
            const double r = ARf(x, y);
            sum += r; ++cnt; lo = std::min(lo, r); hi = std::max(hi, r);
          }
        rho_mean = sum / double(cnt); rho_lo = lo; rho_hi = hi;
      }

      // Mass conservation: the flux must be identical at every station. Reported
      // twice, because the two boundary columns run a different closure and a
      // single number over all stations just reports the outlet.
      {
        // The conserved quantity is the MASS flux, integral of rho*u, not the
        // integral of u. At Ma = 0.035 rho varies by ~5e-4 along the channel, so
        // a spread measured on u alone reports weak compressibility as if it
        // were an error in the scheme.
        double lo = 1e300, hi = -1e300, ilo = 1e300, ihi = -1e300;
        for (Index x = 0; x < Nx; ++x) {
          double q = 0;
          for (Index y = yf0; y <= yf1; ++y) q += ARf(x, y) * AXf(x, y);
          q /= double(yf1 - yf0 + 1);
          lo = std::min(lo, q); hi = std::max(hi, q);
          if (x >= 2 && x < Nx - 2) { ilo = std::min(ilo, q); ihi = std::max(ihi, q); }
        }
        flux_spread   = (hi - lo) / std::max(std::abs(hi), 1e-300);
        flux_interior = (ihi - ilo) / std::max(std::abs(ihi), 1e-300);
      }

      //-----------------------------------------------------------------------
      // ASCII dumps. Plain text on purpose: this tree has five binary output
      // formats and no reader, and 5 kB of numbers needs neither.
      //-----------------------------------------------------------------------
      const double u_scale = dx / dt;                     // lattice -> m/s
      // The tag must name every input that changes the dumped fields, or runs
      // collide silently. It did: a sweep at the same Ny and Re but nz = 1 and a
      // different outlet overwrote a 25-minute 3D run's dumps, and the only
      // symptom was a velocity in the figure that was 1e4 times too large.
      // u_phys is in the name because it sets dx/dt and so the physical units of
      // every dumped column, while leaving the lattice solution untouched.
      char tag[512];
      std::snprintf(tag, sizeof tag,
                    "results/I_channel/ny%d_nz%d_re%.0e_u%.0e_%s_%s_%s_%s",
                    int(Ny), int(Nz), Re, u_phys, walls.c_str(), outlet.c_str(),
                    lat.c_str(), op.c_str());

      if (std::FILE* f = std::fopen((std::string(tag) + "_profile.dat").c_str(), "w")) {
        std::fprintf(f, "# mid-section x=%d z=%d   y_m  ux_lat  ux_mps  ux_exact_mps  uy_lat\n",
                     int(xm), int(zm));
        for (Index y = yf0; y <= yf1; ++y) {
          const double e = eta_of(y);
          const double v = AXf(xm, y);
          std::fprintf(f, "%.8e %.8e %.8e %.8e %.8e\n", e * H_phys, v, v * u_scale,
                       4.0 * u_phys * e * (1.0 - e), AYf(xm, y));
        }
        std::fclose(f);
      }
      if (std::FILE* f = std::fopen((std::string(tag) + "_contour.dat").c_str(), "w")) {
        std::fprintf(f, "# mid z-plane z=%d   nx=%d ny=%d dx=%.8e   x_m y_m ux_mps uy_mps rho\n",
                     int(zm), int(Nx), int(yf1 - yf0 + 1), dx);
        for (Index y = yf0; y <= yf1; ++y)
          for (Index x = 0; x < Nx; ++x) {
            std::fprintf(f, "%.8e %.8e %.8e %.8e %.8e\n", double(x) * dx,
                         (double(y) - y_off) * dx,
                         AXf(x, y) * u_scale, AYf(x, y) * u_scale, ARf(x, y));
          }
        std::fclose(f);
      }
      if (std::FILE* f = std::fopen((std::string(tag) + "_centreline.dat").c_str(), "w")) {
        std::fprintf(f, "# x_m  ux_mps  rho_lat  p_pa_rel\n");
        for (Index x = 0; x < Nx; ++x) {
          const double p = (ARf(x, Ny / 2) - 1.0) * double(cs2<LL, Real>())
                         * rho_phys * (dx / dt) * (dx / dt);
          std::fprintf(f, "%.8e %.8e %.8e %.8e\n", double(x) * dx,
                       AXf(x, Ny / 2) * u_scale, ARf(x, Ny / 2), p);
        }
        std::fclose(f);
      }

      if (dump) {
        std::printf("\n   y/H      ux(inlet)    ux(mid)      ux(outlet)   exact        rho(mid)\n");
        for (Index y = yf0; y <= yf1; ++y) {
          const double e = eta_of(y);
          std::printf("  %5.3f %12.5e %12.5e %12.5e %12.5e %12.8f\n", e,
                      AXf(1, y), AXf(xm, y), AXf(Nx - 1, y),
                      4.0 * double(u0) * e * (1.0 - e), ARf(xm, y));
        }
      }
    });

    if (!ok) { std::printf("\n  unknown lattice/operator\n"); Kokkos::finalize(); return 1; }

    const double u_scale = dx / dt;
    std::printf("\n  RUN\n");
    std::printf("    tau              %.9f   (tau - 1/2 = %.3e)\n", tau_print, tau_print - 0.5);
    std::printf("    init             %s\n", init.c_str());
    std::printf("    advective time   %.0f steps/pass      diffusive time %.4g steps\n", T_adv, tau_d);
    std::printf("    steps taken      %zu   = %.1f passes = %.3g diffusive times   [%.1f s]\n",
                taken, double(taken) / T_adv, double(taken) / tau_d, secs);
    if (fevery > 0)
      std::printf("    frames           %d written every %zu steps (%.4g s of physical time each)\n",
                  frames_written, fevery, double(fevery) * dt);
    if (blew) {
      std::printf("    residual         --\n");
      std::printf("\n  DIVERGED after %zu steps (non-finite or |u| > 1)\n", taken);
      std::printf("    first bad node   x=%d y=%d  of %dx%d  ->  %s\n",
                  int(blow_x), int(blow_y), int(Nx), int(Ny), blow_where);
      std::printf("\n  FAIL\n");
      Kokkos::finalize();
      return 1;
    }
    std::printf("    residual         %.3e   (whole field, over one probe)\n", resid);
    std::printf("    (snapshots)      mean %.5e  range [%.5e, %.5e] over %zu\n",
                err_mean, err_lo, err_hi, n_samples);
    std::printf("                     +-%.1f%% -- the acoustic wave the average removes\n",
                100.0 * (err_hi - err_lo) / (2.0 * std::max(err_mean, 1e-300)));
    std::printf("\n  RESULT\n");
    std::printf("    peak ux          %.6e lat = %.6g m/s\n", u_peak_num, u_peak_num * u_scale);
    std::printf("                     vs exact AT THAT NODE  %+.4f%%%s\n",
                100.0 * (u_peak_num - u_peak_exact_node) / u_peak_exact_node,
                peak_is_node ? "   (centreline is a node)"
                             : "   (centreline is BETWEEN nodes)");
    std::printf("                     vs true peak %.6g m/s  %+.4f%%   <- includes the sampling gap\n",
                u_phys, 100.0 * (u_peak_num * u_scale - u_phys) / u_phys);
    std::printf("    rel L2 profile   %.5e   mid-section, TIME AVERAGED over %d samples\n",
                err, n_avg);
    // (dp/dx)_phys = rho_phys (dx/dt)^2 (dp/dx)_lat / dx = rho_phys dx / dt^2 * ...
    const double dpdx_conv = dpdx_num * rho_phys * dx / (dt * dt);
    std::printf("    dp/dx            %.6g Pa/m   (exact %.6g, %+.2f%%)\n",
                dpdx_conv, dpdx_phys,
                100.0 * (dpdx_conv - dpdx_phys) / dpdx_phys);
    std::printf("    bulk mass error  %+.3e  (relative to the initial value)\n", mass_rel_err);
    if (is_servo)
      std::printf("    servo            outflow rho = %.8f   last mean-density error %+.2e\n",
                  servo_rho_out, servo_err_out);
    std::printf("    rho              mean %.8f  range [%.8f, %.8f]\n",
                rho_mean, rho_lo, rho_hi);
    std::printf("                     mean-1 = %+.3e   (drifts freely with -outlet copy)\n",
                rho_mean - 1.0);
    std::printf("    mass flux spread %.3e all stations   %.3e interior only\n",
                flux_spread, flux_interior);
    // The wall closure is second order, so the profile error at N cells across
    // scales as 1/N^2 and a fixed tolerance is meaningless: at N = 9 it is
    // ~1e-2 by construction. Judge against the expected discretisation error
    // with a factor of 3 of headroom, not against an absolute number.
    const double tol = 3.0 * 1.0 / (double(N) * double(N));
    std::printf("    tolerance        %.3e   = 3/N^2 with N = %d\n", tol, int(N));
    const bool pass = std::isfinite(err) && err < tol && flux_interior < 5e-3;
    std::printf("\n  %s\n", pass ? "PASS" : "FAIL");
    Kokkos::finalize();
    return pass ? 0 : 1;
  }
}
