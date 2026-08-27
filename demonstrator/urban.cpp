//==============================================================================
//  Urban pollutant dispersion.
//
//  A passive scalar released into a city, transported by an atmospheric wind
//  over building geometry read from a height field. This is the Pollutant
//  project's D3Q7 stage, rebuilt on the validated solver: the same physics, the
//  same output format, but the transport, the walls and -- the part that did not
//  previously exist -- the OPEN BOUNDARIES are the ones validated in
//  validation/gaussian_plume.cpp and validation/scalar_walls.cpp.
//
//  A DEMONSTRATOR, NOT A VALIDATION CASE, and the distinction matters here more
//  than usual. There is no analytic solution for a plume in a real city and no
//  measurement to compare against, so nothing in this file can fail except by
//  crashing. What can be checked is checked, and reported as a number:
//    * the wind's divergence, which bounds the error it injects (see below);
//    * whether the scalar has reached steady state, as dM/dt;
//    * the fraction of the release still inside the domain.
//
//  THE WIND IS PRESCRIBED, AND THAT IS THE MODEL'S MAIN LIMITATION.
//  u(z) = (u*/kappa) ln((z+z0)/z0) is a neutral-stability surface layer, simply
//  zeroed inside buildings. It has no street-canyon recirculation, no building
//  wake and no channelling along avenues -- which are exactly the phenomena an
//  urban dispersion study is about.
//
//  HOW NOT TO MEASURE THAT. A log profile zeroed inside buildings is not
//  divergence-free, and it is tempting to look for the damage in a mass budget.
//  That budget cannot see it. Bounce-back is conservative and sum_i w_i c_i = 0,
//  so the advective term contributes nothing to sum_i h_i: the budget closes to
//  round-off however wrong the wind is. Here it is worse still, because the
//  outflow condition of ScalarSolver PRESCRIBES its cells rather than conserving
//  them, so "what left the domain" is not an independent quantity at all -- it
//  is injected minus stored, by construction.
//
//  The error appears in the CONCENTRATION instead. Expanding the transport term,
//  div(uC) = u.grad(C) + C div(u), and the second piece is spurious: it
//  concentrates the scalar where div(u) < 0 and dilutes it where div(u) > 0. So
//  div(u) is the diagnostic, reported below over the open cells and separately
//  over the cells next to a wall, where the zeroing bites hardest. Those numbers
//  are the quantitative case for replacing this wind with a solved one.
//
//  CHECKED AGAINST THE ORIGINAL, AND WHERE THE TWO DIVERGE. On the synthetic
//  60x30x20 case both codes agree exactly on dt (0.0129 s), tau (0.5646) and
//  every div(u) figure -- RMS 1.063e-3 over open cells, 9.213e-3 over
//  wall-adjacent cells, max 1.867e-2 -- so the wind, the scaling and the
//  geometry are the same problem. Along the plume centreline the concentrations
//  agree to within 0.8% through the interior.
//
//  They part company at the outflow, and it is worth recording which one is
//  wrong. Steady-state mass here is 11.4% higher than the original's, entirely
//  in the cells near an open face: 12.5% of the field lies within three cells of
//  one, against 5.9% there. Measuring the advective flux through successive
//  planes settles it. In the original that flux falls 52% between x = 30 and the
//  last cell, dropping from 0.82 to 0.48 of its reference value across a SINGLE
//  cell; here it rises 7.6% over the same span. A steady state cannot destroy
//  half its flux at a boundary, so the original's exit is an artificial sink: it
//  discards outgoing populations and supplies no incoming ones, starving the
//  last cell and pulling material out diffusively for some distance upstream.
//
//  The 7.6% rise is not error either, or not all of it. The wind is sheared, so
//  scalar mixing upward into faster-moving air genuinely increases u.C through
//  successive planes while the TOTAL flux, advective plus diffusive, is what is
//  conserved. The condition used here was measured against a case with no shear
//  and an analytic answer -- validation/gaussian_plume.cpp -- where it holds the
//  flux to 0.038% all the way to the last interior cell.
//==============================================================================
#include "collision/ScalarBGK.hpp"
#include "collision/TRT.hpp"
#include "core/Types.hpp"
#include "io/HeightField.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"

#include <chrono>
#include <memory>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using L      = D3Q7;
using Coll   = ScalarBGK<L>;
using Solver = ScalarSolver<L, EsotericPull<L>, Coll>;

// The wind, when it is solved rather than prescribed. D3Q27 because D3Q19 is
// out of scope for validation in this code, and TRT rather than BGK because the
// effective viscosity here puts tau within a few thousandths of 1/2, where BGK
// is not usable and TRT's free antisymmetric rate is exactly the lever needed.
using FL   = D3Q27;
using FOp  = TRT<FL>;
using Flow = FluidSolver<FL, EsotericPull<FL>, FOp>;

//------------------------------------------------------------------------------
// Neutral-stability logarithmic surface layer.
//
// Bearing is meteorological -- the direction the wind comes FROM -- so 270 is a
// westerly, blowing toward +x. Getting that convention backwards is silent: the
// plume simply goes the wrong way, and a city is not symmetric enough for that
// to be obvious in a picture.
//------------------------------------------------------------------------------
struct LogLaw {
  double u_ref = 5.0, z_ref = 10.0, z0 = 1.0, kappa = 0.41, bearing_deg = 270.0;

  double ustar() const { return kappa * u_ref / std::log((z_ref + z0) / z0); }
  double speed(double z_m) const { return ustar() / kappa * std::log((z_m + z0) / z0); }
  void direction(double& ex, double& ey) const {
    const double r = bearing_deg * M_PI / 180.0;
    ex = -std::sin(r); ey = -std::cos(r);
  }
};

//------------------------------------------------------------------------------
// Physical <-> lattice scaling. dt is fixed by capping the fastest cell at
// u_lat_max, then the relaxation rate follows from the physical diffusivity.
// Deriving dt from the wind and omega from dt (rather than choosing tau and
// letting dt fall out) is what keeps the advective Courant number bounded, which
// is the constraint that actually bites in an advection-dominated problem.
//------------------------------------------------------------------------------
struct Scaling {
  double dx, dt, u_lat_max, D_phys;
  double to_u_lat(double u_phys) const { return u_phys * dt / dx; }
  double D_lat() const { return D_phys * dt / (dx * dx); }
  double seconds(std::size_t steps) const { return double(steps) * dt; }
};

//------------------------------------------------------------------------------
// Legacy VTK, STRUCTURED_POINTS, binary big-endian -- deliberately byte-for-byte
// the format the Pollutant project's renderers already read, so the existing
// visualisation pipeline transfers with the physics. Solid cells are written as
// -1 so they can be masked rather than appearing as zero concentration.
//
// Binary, not the ASCII of src/io/VtiWriter.hpp: at 400x400x60 an ASCII dump is
// several hundred megabytes per frame, which makes a time series unusable.
//------------------------------------------------------------------------------
static void write_vtk(const std::string& path, const Domain& d, const HeightField& g,
                      const std::vector<float>& C, int step,
                      const char* field = "concentration") {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  const std::size_t n = g.cells();
  std::fprintf(f, "# vtk DataFile Version 3.0\nurban plume step %d\nBINARY\n"
                  "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\n"
                  "ORIGIN 0 0 0\nSPACING %g %g %g\nPOINT_DATA %zu\n"
                  "SCALARS %s float 1\nLOOKUP_TABLE default\n",
               step, int(g.nx), int(g.ny), int(g.nz), g.dx, g.dx, g.dx, n, field);
  std::vector<unsigned char> buf(n * 4);
  std::size_t p = 0;
  for (Index z = 0; z < g.nz; ++z)
    for (Index y = 0; y < g.ny; ++y)
      for (Index x = 0; x < g.nx; ++x) {
        const float v = C[std::size_t(d.id(x, y, z))];
        std::uint32_t w; std::memcpy(&w, &v, 4);
        w = __builtin_bswap32(w);                    // legacy VTK is big-endian
        std::memcpy(&buf[p], &w, 4); p += 4;
      }
  std::fwrite(buf.data(), 1, buf.size(), f);
  std::fclose(f);
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string prefix = "city", outdir = "vtk";
  LogLaw wind;
  double D_phys = 5.0, u_lat_max = 0.05, minutes = 10.0, out_every_s = 10.0;
  double rate = 1.0;
  int src_i = -1, src_j = -1, src_k = -1, src_r = 2;
  bool diffusion_only = false;
  std::string top = "lid", mode = "prescribed", lateral = "prescribed", wind_out;
  double nu_phys = 2.0;              // effective (turbulent) viscosity, m2/s
  std::size_t flow_max = 40000;
  double flow_tol = 1e-6;

  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    auto num = [&](double d) { return (a + 1 < argc) ? std::atof(argv[++a]) : d; };
    if      (s == "-geom")      prefix = argv[++a];
    else if (s == "-out")       outdir = argv[++a];
    else if (s == "-u-ref")     wind.u_ref = num(wind.u_ref);
    else if (s == "-z-ref")     wind.z_ref = num(wind.z_ref);
    else if (s == "-z0")        wind.z0 = num(wind.z0);
    else if (s == "-bearing")   wind.bearing_deg = num(wind.bearing_deg);
    else if (s == "-diff")      D_phys = num(D_phys);
    else if (s == "-u-lat")     u_lat_max = num(u_lat_max);
    else if (s == "-minutes")   minutes = num(minutes);
    else if (s == "-out-every") out_every_s = num(out_every_s);
    else if (s == "-rate")      rate = num(rate);
    else if (s == "-src")     { src_i = int(num(0)); src_j = int(num(0)); src_k = int(num(0)); }
    else if (s == "-src-r")     src_r = int(num(src_r));
    else if (s == "-diffusion-only") diffusion_only = true;
    else if (s == "-top")       top = argv[++a];
    else if (s == "-mode")      mode = argv[++a];
    else if (s == "-lateral")   lateral = argv[++a];
    else if (s == "-wind-out")  wind_out = argv[++a];
    else if (s == "-visc")      nu_phys = num(nu_phys);
    else if (s == "-flow-steps") flow_max = std::size_t(num(double(flow_max)));
    else if (s == "-flow-tol")  flow_tol = num(flow_tol);
    // Kokkos parses its own --kokkos-* flags out of the argc/argv handed to
    // initialize() below, so they must reach it rather than being rejected
    // here. That matters more than it sounds: the Threads backend defaults to
    // ONE thread, and --kokkos-num-threads=4 is 2.1x on this grid.
    else if (s.rfind("--kokkos-", 0) == 0) continue;
    else { std::printf("unknown argument %s\n", s.c_str()); return 2; }
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::printf("\nUrban pollutant dispersion   D3Q7   %s\n", precision_name());
    std::printf("%s\n\n", std::string(70, '=').c_str());

    const HeightField g = load_height_field(prefix + "_heights.npy", prefix + "_meta.json");
    report(g, "geometry");

    //--------------------------------------------------------------------------
    // Scaling. With no wind there is no speed to size dt from, so it comes from
    // the relaxation rate instead -- which also lifts the Courant limit and
    // makes the pure-diffusion case far cheaper than the advective one.
    Scaling sc{g.dx, 0.0, u_lat_max, D_phys};
    double u_peak = 0.0;
    if (diffusion_only) {
      const double tau = 0.8;
      sc.dt = 0.25 * (tau - 0.5) * g.dx * g.dx / D_phys;
    } else {
      u_peak = wind.speed(g.nz * g.dx);
      sc.dt = u_lat_max * g.dx / u_peak;
    }
    const Real omega = Coll::omega_from_diffusivity(Real(sc.D_lat()));
    const double tau = 1.0 / double(omega);

    std::printf("\n  wind:  ");
    if (diffusion_only) std::printf("DIFFUSION ONLY, u = 0 everywhere\n");
    else std::printf("log law, u_ref %.1f m/s at %.0f m, z0 %.2f m, bearing %.0f deg\n"
                     "         u* = %.3f m/s, peak %.2f m/s at %.0f m\n",
                     wind.u_ref, wind.z_ref, wind.z0, wind.bearing_deg,
                     wind.ustar(), u_peak, g.nz * g.dx);
    std::printf("  top:   %s\n", top == "open"
                ? "OPEN -- held at C = 0, the scalar diffuses out into clean air"
                : "LID -- zero gradient, so nothing escapes upward (-top open to change)");
    std::printf("  scale: dt %.4f s, tau %.4f, D %.2f m2/s, max |u| %.3f lattice\n",
                sc.dt, tau, D_phys, diffusion_only ? 0.0 : u_lat_max);
    if (tau < 0.51)
      std::printf("  WARNING tau is close to 1/2; raise the diffusivity or dt\n");

    //--------------------------------------------------------------------------
    // PERIODIC LATERALS: the standard atmospheric arrangement, and the fix the
    // -mode solved header has been pointing at since it was written.
    //
    // The wind runs along x, so the y faces are the lateral pair. Making them
    // periodic deletes two of the four faces where a velocity-prescribed wall
    // meets a pressure outlet, and it is at those junctions that the corner jet
    // forms. It is not a cosmetic improvement: on a real city the prescribed
    // arrangement does not merely jet, it fails to converge at all.
    //
    // What it costs is a modelling choice rather than accuracy. The domain
    // becomes an infinite strip: the city repeats every ny cells across the
    // wind, and a plume that drifts sideways out of one edge re-enters at the
    // other. That is the usual idealisation of an extended urban canopy, and it
    // is exactly right for a cross-wind-homogeneous problem and wrong for one
    // where the city has an edge that matters. The seam is also not smooth --
    // real building heights at y = 0 and y = ny-1 were never chosen to match --
    // so the mismatch is counted and reported below rather than left implicit.
    //
    // Periodicity is handled inside Domain::fill_neighbours, which wraps within
    // the interior range and allocates no halo in a periodic direction, so
    // neither solver needs to know.
    const bool per_y = (lateral == "periodic");
    if (lateral != "periodic" && lateral != "prescribed")
      throw std::runtime_error("-lateral takes 'prescribed' or 'periodic'");
    Domain d(g.nx, g.ny, g.nz, false, per_y, false);
    Coll coll; coll.omega = omega;
    Solver s(d, coll);

    // Which faces the wind enters through, from the bearing. Inflow faces are
    // held at C = 0 -- clean air arriving -- and everything else is open. Making
    // every lateral face open instead would leave the upwind boundary free to
    // drift, and a zero-gradient inflow is not a boundary condition on an
    // incoming characteristic.
    double ex = 0, ey = 0;
    if (!diffusion_only) wind.direction(ex, ey);
    const bool in_xm = ex > 1e-9, in_xp = ex < -1e-9;
    if (per_y) {
      // The city wraps across the y seam, and real building heights at y = 0
      // and y = ny-1 were never chosen to match. Counting the mismatch is the
      // difference between a stated idealisation and a hidden one.
      Index seam = 0; double worst = 0;
      for (Index x = 0; x < g.nx; ++x) {
        const double a = g.at(x, 0), b = g.at(x, g.ny - 1);
        if (std::abs(a - b) > 1e-6) { ++seam; worst = std::max(worst, std::abs(a - b)); }
      }
      std::printf("\n  lateral: PERIODIC in y. The domain is an infinite strip across the\n"
                  "           wind: the city repeats every %.0f m and a plume drifting\n"
                  "           sideways out of one edge re-enters at the other.\n"
                  "           Seam: %d of %d columns disagree in height, worst %.1f m.\n",
                  g.ny * g.dx, int(seam), int(g.nx), worst);
      // Periodicity in y is the right idealisation for a wind ALONG x. With a
      // cross-wind component the strip is still well posed, but the plume is
      // then advected into the seam rather than merely diffusing across it, and
      // the wrap stops being a detail.
      if (!diffusion_only && std::abs(ey) > 0.05)
        std::printf("           WARNING the wind has a %.0f%% cross-wind component, so the\n"
                    "           plume is ADVECTED through the seam, not just diffused.\n",
                    100.0 * std::abs(ey));
    }
    const bool in_ym = ey > 1e-9, in_yp = ey < -1e-9;

    // ONE classification, named, used by set_geometry AND by the source-cell
    // count below. They were two separate hand-written predicates and they
    // drifted: the count required z > 0 while the injection ran over every
    // ScalarBulk cell, which after the ground fix includes z = 0. A source
    // sitting on the ground therefore had its per-cell rate divided by three
    // layers and applied to four -- 33% over-injection, visible as "retained
    // 127.9%" and invisible for any source high enough to clear z = 2.
    auto classify = [&](Index x, Index y, Index z) -> ScalarCell {
      if (g.solid(x, y, z)) return ScalarAdiabatic;      // buildings: zero flux
      // THE GROUND IS NOT MARKED. Under Esoteric Pull the z = 0 cells already
      // bounce back off the layer beneath them -- that layer is never processed,
      // so the update there is the identity and a population emitted downward is
      // read back as the opposite direction, with the wall exactly half a cell
      // below the first cell centre, i.e. at z = 0. Marking z = 0 adiabatic
      // instead makes those cells WALLS rather than fluid, which deletes the
      // lowest layer of air and lifts the ground to z = dx. That is a 5 m error
      // on this grid, in the layer where the source sits.
      const bool inflow = (x == 0 && in_xm) || (x == g.nx - 1 && in_xp) ||
                          (!per_y && ((y == 0 && in_ym) || (y == g.ny - 1 && in_yp)));
      if (inflow) return ScalarDirichlet;
      // THE TOP IS A MODELLING CHOICE, not a technicality, and the two options
      // differ by more than the boundary layer.
      //
      // A horizontal wind has u_z = 0 at the top face, so a zero-gradient
      // condition there passes no advective flux (there is none) AND no
      // diffusive flux (that is what zero-gradient means). The top is therefore
      // a LID: nothing escapes upward, and material that mixes to the top stays
      // in the domain. That is right when the top is far above the plume and
      // wrong when it is not.
      //
      // Holding the top at C = 0 instead says the atmosphere above is clean and
      // lets the scalar diffuse away into it, which is the more physical
      // statement for a domain whose top is an arbitrary cut through the
      // boundary layer -- at the cost of forcing C = 0 there, so it is wrong in
      // the other direction if the plume genuinely reaches the top.
      if (z == g.nz - 1) return (top == "open") ? ScalarDirichlet : ScalarOutflow;
      const bool face = (x == 0 || x == g.nx - 1 ||
                         (!per_y && (y == 0 || y == g.ny - 1)));
      return face ? ScalarOutflow : ScalarBulk;
    };
    s.set_geometry(classify);
    s.set_wall_values([](Index, Index, Index) -> Real { return Real(0); });
    s.finalize_geometry();
    s.initialize(Real(0));

    //--------------------------------------------------------------------------
    // The wind field, in lattice units, zero inside solids.
    View1D<Real> ux("ux", d.n_padded), uy("uy", d.n_padded), uz("uz", d.n_padded);
    auto h_ux = Kokkos::create_mirror_view(ux);
    auto h_uy = Kokkos::create_mirror_view(uy);
    auto h_uz = Kokkos::create_mirror_view(uz);
    for (Index n = 0; n < d.n_padded; ++n) { h_ux(n) = 0; h_uy(n) = 0; h_uz(n) = 0; }
    if (!diffusion_only)
      for (Index z = 0; z < g.nz; ++z) {
        const double ul = sc.to_u_lat(wind.speed((double(z) + 0.5) * g.dx));
        for (Index y = 0; y < g.ny; ++y)
          for (Index x = 0; x < g.nx; ++x) {
            if (g.solid(x, y, z)) continue;
            const Index n = d.id(x, y, z);
            h_ux(n) = Real(ul * ex); h_uy(n) = Real(ul * ey);
          }
      }
    Kokkos::deep_copy(ux, h_ux); Kokkos::deep_copy(uy, h_uy); Kokkos::deep_copy(uz, h_uz);
    s.set_velocity(ux, uy, uz);

    //--------------------------------------------------------------------------
    // SOLVED WIND. The prescribed profile above becomes the far-field boundary
    // condition and the initial state; the interior is solved, so the flow
    // separates behind buildings and channels along streets instead of passing
    // through them as if they were porous.
    //
    // The payoff is not realism for its own sake -- it is that a solved field is
    // divergence-free to round-off, which removes the spurious C div(u) source
    // entirely, and with it both the concentration error reported above and the
    // stability floor on the diffusivity.
    //
    // Same Domain, same dx, same dt, so the fluid's lattice velocity IS the
    // scalar's lattice velocity and the two couple with no conversion. That is
    // the module composition the whole design is for.
    //
    // *** INCOMPLETE. READ THIS BEFORE USING -mode solved. ***
    //
    // The interior is right and is the point: three cells clear of every face,
    // div u falls from RMS 1.27e-3 (prescribed) to 2.12e-4, and the peak speed
    // is 8.8 m/s against a 7.7 m/s free stream -- a 14% speed-up over the
    // building, which is the physical result the prescribed profile cannot
    // produce at all.
    //
    // The domain EDGES are not right. Where the pressure outlet meets the
    // velocity-prescribed lateral and top faces, the corner is over-determined,
    // and a jet forms along it: 31.6 m/s and div u of 9.5e-2, against 8.8 m/s
    // and 7.0e-3 in the interior. Giving those edges the free stream (below)
    // removed the inert-outflow-node warning but did NOT remove the jet, so the
    // cause is the mixed condition itself and not the donor lookup.
    //
    // The identified fix is periodic lateral boundaries -- the standard
    // atmospheric setup, which deletes two of the four offending faces outright
    // and leaves only the outlet/top edge. It is not done here because the
    // Domain is shared with the scalar, for which periodic laterals mean a
    // plume leaving one side re-enters the other, and that is a modelling
    // decision rather than a bug fix.
    //
    // Until then: the field is usable away from the boundary, the plume must
    // not be allowed to reach the outlet edges, and -mode prescribed remains
    // the default.
    std::unique_ptr<Flow> flow;
    if (mode == "solved") {
      if (diffusion_only) throw std::runtime_error("-mode solved and -diffusion-only conflict");
      const double nu_lat = nu_phys * sc.dt / (g.dx * g.dx);
      FOp fcoll;
      fcoll.omega_p = FOp::omega_from_viscosity(Real(nu_lat));
      // The magic parameter is the reason this is TRT. At nu_lat this small,
      // tau_plus sits a few thousandths above 1/2, where BGK is unusable;
      // Lambda = 3/16 additionally puts the bounce-back wall exactly halfway,
      // which is what makes a voxelised building the size it looks.
      fcoll.omega_m = FOp::omega_minus_for(fcoll.omega_p, FOp::magic_3_16);
      std::printf("\n  SOLVING the wind: D3Q27 TRT, nu %.2f m2/s -> nu_lat %.3e,"
                  " tau+ %.6f, Lambda 3/16\n",
                  nu_phys, nu_lat, 1.0 / double(fcoll.omega_p));

      flow = std::make_unique<Flow>(d, fcoll);
      // A FACE CELL SHADOWED BY A BUILDING IS ILL-POSED, and on a real city
      // there are thousands of them.
      //
      // The regularised condition reconstructs density at a wall node from the
      // populations that streamed into it, splitting them by the wall normal
      // into known and unknown. That split assumes the known ones arrived from
      // the interior. When the cell one step inward is a building, what
      // actually arrives is bounce-back off that building, the reconstructed
      // density is not a density, and the solve diverges: Manchester has 3975
      // built columns touching the domain edge and NaNs within 500 steps.
      // Clearing an 8-cell band around the edge removes the NaN outright, which
      // is what identified the cause; periodic laterals alone do NOT, because
      // they leave the inlet and outlet faces where the same thing happens.
      //
      // Making such a cell Solid extends the building by the one cell it was
      // already effectively occupying, and turns an ill-posed node into a
      // well-posed wall. It also subsumes the "outflow node has no fluid
      // neighbour and is inert" case, which was the same geometry seen from the
      // outlet side and left frozen rather than closed.
      Index shadowed = 0;
      for (Index z = 0; z < g.nz; ++z)
        for (Index y = 0; y < g.ny; ++y)
          for (Index x = 0; x < g.nx; ++x)
            if (!g.solid(x, y, z) &&
                ((x == 0 && g.solid(x + 1, y, z)) ||
                 (x == g.nx - 1 && g.solid(x - 1, y, z)) ||
                 (z == g.nz - 1 && g.solid(x, y, z - 1)) ||
                 (!per_y && y == 0 && g.solid(x, y + 1, z)) ||
                 (!per_y && y == g.ny - 1 && g.solid(x, y - 1, z)))) ++shadowed;
      if (shadowed)
        std::printf("  [walls] %d face cell(s) shadowed by a building, closed as solid\n",
                    int(shadowed));

      flow->set_geometry([&](Index x, Index y, Index z) -> CellType {
        if (g.solid(x, y, z)) return Solid;
        if ((x == 0 && g.solid(x + 1, y, z)) ||
            (x == g.nx - 1 && g.solid(x - 1, y, z)) ||
            (z == g.nz - 1 && g.solid(x, y, z - 1)) ||
            (!per_y && y == 0 && g.solid(x, y + 1, z)) ||
            (!per_y && y == g.ny - 1 && g.solid(x, y - 1, z))) return Solid;
        // z = 0 is deliberately NOT a wall here, for the same reason as in the
        // scalar: Esoteric Pull already bounces it off the layer below, putting
        // the no-slip ground at z = 0 exactly.
        const bool face = (x == 0 || x == g.nx - 1 || z == g.nz - 1 ||
                           (!per_y && (y == 0 || y == g.ny - 1)));
        return face ? RegWall : Fluid;
      });

      using WS = Flow::WallSpec;
      flow->set_regularized_walls([&](Index x, Index y, Index z) -> WS {
        if (g.solid(x, y, z)) return WS{};
        // Same predicate as the geometry above: a closed cell is a wall, not a
        // boundary node, and must not be handed a velocity to impose.
        if ((x == 0 && g.solid(x + 1, y, z)) ||
            (x == g.nx - 1 && g.solid(x - 1, y, z)) ||
            (z == g.nz - 1 && g.solid(x, y, z - 1)) ||
            (!per_y && y == 0 && g.solid(x, y + 1, z)) ||
            (!per_y && y == g.ny - 1 && g.solid(x, y - 1, z))) return WS{};
        const double ul = sc.to_u_lat(wind.speed((double(z) + 0.5) * g.dx));
        const Real vx = Real(ul * ex), vy = Real(ul * ey), vz = Real(0);
        // A face is an inlet where the wind enters it, an outlet where it
        // leaves, and free-stream where the flow runs parallel -- prescribing
        // the undisturbed profile there rather than letting the boundary drift.
        auto face = [&](double inward, std::uint8_t code) -> WS {
          if (inward < -1e-9) return WS{NrmOutFree, Real(0), Real(0), Real(0), Real(1)};
          return WS{code, vx, vy, vz, Real(1)};
        };
        // EDGES AND CORNERS GET THE FREE STREAM, NOT OUTFLOW. An outflow node
        // takes its distribution from an interior donor, and FluidSolver
        // requires that donor to be a Fluid cell -- but the cell inward of an
        // outlet EDGE lies on the adjoining lateral face, which is a wall. Such
        // nodes are reported inert and behave as walls, and the resulting
        // partial blockage of the outlet plane drove a corner jet: 28.7 m/s
        // against a 7.7 m/s free stream, with div u of 8.9e-2 concentrated
        // there, while the interior was meanwhile perfectly well behaved at
        // 9.1 m/s and 7.3e-3. Prescribing the undisturbed profile on the edges
        // is both defined and defensible -- they are far field by construction.
        const int faces = int(x == 0) + int(x == g.nx - 1) + int(z == g.nz - 1) +
                          (per_y ? 0 : int(y == 0) + int(y == g.ny - 1));
        if (faces > 1) return WS{NrmCorner, vx, vy, vz, Real(1)};
        if (x == 0)         return face( ex, NrmXm);
        if (x == g.nx - 1)  return face(-ex, NrmXp);
        if (!per_y && y == 0)        return face( ey, NrmYm);
        if (!per_y && y == g.ny - 1) return face(-ey, NrmYp);
        if (z == g.nz - 1)  return face(Real(0), NrmZp);
        return WS{};
      });
      // The finite-difference corner stress walks two nodes off each wall node.
      // In a city those neighbours are frequently inside a building, whose
      // populations Esoteric Pull never updates, so the stencil would read
      // stale memory -- the same defect the aorta hit.
      flow->set_fd_corners(false);

      // Start from the undisturbed profile rather than from rest: only the
      // wakes then have to develop, instead of the whole boundary layer being
      // advected in from the inlet over several flow-through times.
      View1D<Real> prof("prof", g.nz);
      auto h_prof = Kokkos::create_mirror_view(prof);
      for (Index z = 0; z < g.nz; ++z)
        h_prof(z) = Real(sc.to_u_lat(wind.speed((double(z) + 0.5) * g.dx)));
      Kokkos::deep_copy(prof, h_prof);
      const Domain fd = d;
      const Real fex = Real(ex), fey = Real(ey);
      flow->initialize_field(KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; fd.coords(n, px, py, pz);
        const Index z = pz - fd.hz;
        const Real u = (z >= 0 && z < fd.nz) ? prof(z) : Real(0);
        return FlowState{Real(1), u * fex, u * fey, Real(0)};
      });

      // Convergence on the mean speed. A residual on a single probe point would
      // settle long before the wakes do.
      const std::size_t probe = 500;
      double prev = 0.0;
      std::size_t used = 0;
      std::printf("  %10s %16s %14s\n", "steps", "mean |u|", "rel change");
      std::printf("  %s\n", std::string(44, '-').c_str());
      for (std::size_t t = 0; t < flow_max; t += probe) {
        for (std::size_t k = 0; k < probe; ++k) flow->step();
        used += probe;
        flow->compute_macroscopic();
        auto a = Kokkos::create_mirror_view_and_copy(HostSpace{}, flow->ux());
        auto b = Kokkos::create_mirror_view_and_copy(HostSpace{}, flow->uy());
        auto c = Kokkos::create_mirror_view_and_copy(HostSpace{}, flow->uz());
        double sum = 0.0; long long cnt = 0;
        for (Index z = 0; z < g.nz; ++z)
          for (Index y = 0; y < g.ny; ++y)
            for (Index x = 0; x < g.nx; ++x) {
              if (g.solid(x, y, z)) continue;
              const Index n = d.id(x, y, z);
              sum += std::sqrt(double(a(n))*double(a(n)) + double(b(n))*double(b(n)) +
                               double(c(n))*double(c(n)));
              ++cnt;
            }
        const double cur = cnt ? sum / double(cnt) : 0.0;
        const double rel = (prev > 0) ? std::abs(cur - prev) / cur : 1.0;
        std::printf("  %10zu %16.8e %14.3e\n", used, cur, rel);
        std::fflush(stdout);
        if (!std::isfinite(cur)) throw std::runtime_error("the wind solve diverged");
        if (prev > 0 && rel < flow_tol) break;
        prev = cur;
      }

      // Hand the solved field to the scalar, and re-take the host mirrors so the
      // divergence diagnostic below measures the field actually being used.
      s.set_velocity(flow->ux(), flow->uy(), flow->uz());
      Kokkos::deep_copy(h_ux, flow->ux());
      Kokkos::deep_copy(h_uy, flow->uy());
      Kokkos::deep_copy(h_uz, flow->uz());
      for (Index z = 0; z < g.nz; ++z)
        for (Index y = 0; y < g.ny; ++y)
          for (Index x = 0; x < g.nx; ++x)
            if (g.solid(x, y, z)) {
              const Index n = d.id(x, y, z);
              h_ux(n) = 0; h_uy(n) = 0; h_uz(n) = 0;
            }
      double umax = 0;
      for (Index n = 0; n < d.n_padded; ++n)
        umax = std::max(umax, std::sqrt(double(h_ux(n))*double(h_ux(n)) +
                                        double(h_uy(n))*double(h_uy(n)) +
                                        double(h_uz(n))*double(h_uz(n))));
      std::printf("  solved: peak |u| %.4f lattice = %.2f m/s\n",
                  umax, umax * g.dx / sc.dt);
    }

    //--------------------------------------------------------------------------
    // div(u): the diagnostic that CAN see a bad wind, unlike a mass budget.
    double max_div = 0.0, rms_div = 0.0;
    {
      double s2 = 0, s2w = 0, mx = 0, mxw = 0;
      long long nn = 0, nw = 0;
      Index mxi = 0, mxj = 0, mxk = 0;
      // With periodic laterals the y = 0 and y = ny-1 planes are ordinary
      // interior cells and belong in the statistics; their neighbours wrap.
      // Leaving them out would drop 0.5% of the domain from an rms the
      // stability threshold is now built on.
      const Index y0 = per_y ? 0 : 1, y1 = per_y ? g.ny : g.ny - 1;
      auto wy = [&](Index y) { return per_y ? (y + g.ny) % g.ny : y; };
      for (Index z = 1; z < g.nz - 1; ++z)
        for (Index y = y0; y < y1; ++y)
          for (Index x = 1; x < g.nx - 1; ++x) {
            if (g.solid(x, y, z)) continue;
            const Index yp = wy(y + 1), ym = wy(y - 1);
            const double dv =
                0.5 * (double(h_ux(d.id(x+1,y,z))) - double(h_ux(d.id(x-1,y,z)))) +
                0.5 * (double(h_uy(d.id(x,yp,z)))  - double(h_uy(d.id(x,ym,z)))) +
                0.5 * (double(h_uz(d.id(x,y,z+1))) - double(h_uz(d.id(x,y,z-1))));
            s2 += dv * dv; ++nn;
            if (std::abs(dv) > mx) { mx = std::abs(dv); mxi = x; mxj = y; mxk = z; }
            const bool wall = g.solid(x+1,y,z) || g.solid(x-1,y,z) ||
                              g.solid(x,yp,z)  || g.solid(x,ym,z) ||
                              g.solid(x,y,z+1) || g.solid(x,y,z-1);
            if (wall) { s2w += dv * dv; ++nw; mxw = std::max(mxw, std::abs(dv)); }
          }
      std::printf("\n  div u:  RMS %.3e (max %.3e) over %lld open cells\n",
                  nn ? std::sqrt(s2 / double(nn)) : 0.0, mx, nn);
      std::printf("  div u:  RMS %.3e (max %.3e) over %lld wall-adjacent cells\n",
                  nw ? std::sqrt(s2w / double(nw)) : 0.0, mxw, nw);
      std::printf("          a divergence-free field gives 0. This is the error the\n");
      std::printf("          wind injects into CONCENTRATION, and no mass budget sees it.\n");
      max_div = mx;
      rms_div = nn ? std::sqrt(s2 / double(nn)) : 0.0;
      // Where the worst divergence sits matters more than its value: at a
      // domain face it is a boundary-condition artefact, in the interior it is
      // the wind model.
      const bool on_face = mxi <= 1 || mxi >= g.nx - 2 || mxk >= g.nz - 2 ||
                           (!per_y && (mxj <= 1 || mxj >= g.ny - 2));
      std::printf("          worst at (%d,%d,%d)%s\n", int(mxi), int(mxj), int(mxk),
                  on_face ? "  <- ON A DOMAIN FACE" : "  (interior)");
      // Interior-only statistics, three cells clear of every face, so a
      // boundary artefact cannot masquerade as a bulk error.
      double is2 = 0, imx = 0, iu = 0; long long inn = 0;
      // "Clear of every face" means the faces that exist: a periodic direction
      // has none, so the whole span is interior there.
      const Index iy0 = per_y ? 0 : 3, iy1 = per_y ? g.ny : g.ny - 3;
      for (Index z = 1; z < g.nz - 3; ++z)
        for (Index y = iy0; y < iy1; ++y)
          for (Index x = 3; x < g.nx - 3; ++x) {
            if (g.solid(x, y, z)) continue;
            const double dv =
                0.5 * (double(h_ux(d.id(x+1,y,z))) - double(h_ux(d.id(x-1,y,z)))) +
                0.5 * (double(h_uy(d.id(x,wy(y+1),z))) - double(h_uy(d.id(x,wy(y-1),z)))) +
                0.5 * (double(h_uz(d.id(x,y,z+1))) - double(h_uz(d.id(x,y,z-1))));
            is2 += dv * dv; ++inn; imx = std::max(imx, std::abs(dv));
            const Index nn2 = d.id(x, y, z);
            iu = std::max(iu, std::sqrt(double(h_ux(nn2))*double(h_ux(nn2)) +
                                        double(h_uy(nn2))*double(h_uy(nn2)) +
                                        double(h_uz(nn2))*double(h_uz(nn2))));
          }
      std::printf("  div u:  RMS %.3e (max %.3e) over %lld cells 3 clear of every face\n",
                  inn ? std::sqrt(is2 / double(inn)) : 0.0, imx, inn);
      std::printf("          peak |u| there %.4f lattice = %.2f m/s\n",
                  iu, iu * g.dx / sc.dt);
    }

    //--------------------------------------------------------------------------
    // STABILITY MARGIN, and why it is worth a line of output.
    //
    // The spurious term is dC/dt = -C div(u), so wherever the divergence is
    // negative the scalar GROWS at rate |div u|. What opposes that is diffusion,
    // which damps a cell-scale blob at roughly D_lat pi^2. When the two are
    // comparable the run does not fail, it EXPONENTIATES -- slowly enough to
    // look plausible for a few thousand steps and then by nine orders of
    // magnitude per frame.
    //
    // THE MAX IS THE WRONG STATISTIC, and it took two cities to see it.
    //
    //   case                 max|div u|  margin   RMS |div u|  margin   outcome
    //   Manchester diagonal   3.278e-2    5.1x     9.208e-4     180x    stable
    //   Manhattan, D = 20     3.345e-2    4.6x     2.432e-3      63x    EXPONENTIATED
    //   Manhattan, D = 35     3.345e-2    8.0x     2.432e-3     110x    stable
    //
    // Those two cities have the SAME peak divergence to two per cent, and the
    // max-based margin puts a stable run and a divergent one eleven per cent
    // apart -- it cannot separate them, and using it to set a threshold means
    // drawing a line through the middle of a coin flip. The RMS separates the
    // same pair by a factor of 2.9.
    //
    // Which makes sense. The spurious source is damped at the cell scale by
    // D_lat pi^2, so a single bad cell is held down by diffusion from its
    // neighbours; what outruns diffusion is an EXTENDED region of divergence,
    // and how much of the domain is in that state is what the RMS measures and
    // the max does not. The decomposition is exact and worth knowing:
    //
    //     RMS over open cells = sqrt(wall-adjacent fraction) x RMS at walls
    //
    // (0.112 x 8.216e-3 for Manchester, 0.183 x 1.329e-2 for Manhattan, both to
    // three figures), so this really is "how much city", not "how bad is the
    // worst cell". Manhattan has 4.3 times as many wall-adjacent cells.
    //
    // The threshold below is 100x, which sits just under the lowest verified
    // stable point and well above the one failure. It is calibrated on ONE
    // failure and should be treated as such: it is a warning, not a proof. Both
    // margins are printed, because the max is still worth seeing.
    //
    // Note that D_lat = D_phys u_lat_max / (u_peak dx) has no dt in it, so
    // shortening the time step does NOT buy margin -- only a larger
    // diffusivity, a slower wind or a finer grid does. u_lat_max does not buy
    // it either: it scales D_lat and div u by the same factor and cancels.
    if (!diffusion_only && max_div > 0 && rms_div > 0) {
      const double damping = sc.D_lat() * M_PI * M_PI;
      const double margin_max = damping / max_div;
      const double margin = damping / rms_div;
      std::printf("\n  stability: D_lat %.5f, damping %.4f\n"
                  "             vs RMS |div u| %.4e  ->  margin %.0fx   <- the test\n"
                  "             vs max |div u| %.4e  ->  margin %.1fx\n",
                  sc.D_lat(), damping, rms_div, margin, max_div, margin_max);
      if (margin < 100.0) {
        std::printf("  *** MARGIN IS LOW. The spurious C div(u) source is close to\n"
                    "  *** outrunning diffusion, and this run may exponentiate rather\n"
                    "  *** than diverge visibly. Raise -diff (the turbulent diffusivity\n"
                    "  *** is a calibrated stand-in, not a measured constant) or use a\n"
                    "  *** solved, divergence-free wind. A shorter time step will NOT\n"
                    "  *** help: D_lat does not depend on dt.\n");
        std::printf("  *** For reference, -diff %.0f would give a margin of about 100x.\n",
                    std::ceil(D_phys * 100.0 / margin));
      }
    }

    //--------------------------------------------------------------------------
    // The wind itself, in m/s, in the same format as the concentration frames.
    // Solid is written as -1 so a renderer masks it the same way.
    if (!wind_out.empty()) {
      std::vector<float> W(std::size_t(d.n_padded), 0.f);
      for (Index z = 0; z < g.nz; ++z)
        for (Index y = 0; y < g.ny; ++y)
          for (Index x = 0; x < g.nx; ++x) {
            const Index n = d.id(x, y, z);
            W[std::size_t(n)] = g.solid(x, y, z) ? -1.f : float(
                std::sqrt(double(h_ux(n))*double(h_ux(n)) +
                          double(h_uy(n))*double(h_uy(n)) +
                          double(h_uz(n))*double(h_uz(n))) * g.dx / sc.dt);
          }
      write_vtk(wind_out, d, g, W, 0, "speed");
      std::printf("  wind written to %s (m/s)\n", wind_out.c_str());
    }

    if (src_i < 0) { src_i = g.nx / 2; src_j = g.ny / 2; src_k = 1; }
    while (src_k < g.nz - 1 && g.solid(src_i, src_j, src_k)) ++src_k;   // out of any building
    // add_source injects into every ScalarBulk cell of the box, so that is
    // exactly the set to divide by. Asking classify() rather than re-deriving
    // the condition is what keeps the two from drifting apart again.
    Index nsrc = 0;
    for (Index z = src_k - src_r; z <= src_k + src_r; ++z)
      for (Index y = src_j - src_r; y <= src_j + src_r; ++y)
        for (Index x = src_i - src_r; x <= src_i + src_r; ++x)
          if (x >= 0 && x < g.nx && y >= 0 && y < g.ny && z >= 0 && z < g.nz &&
              classify(x, y, z) == ScalarBulk) ++nsrc;
    if (nsrc == 0)
      throw std::runtime_error("no source cell is a bulk cell -- move the source");

    const Real per_cell = Real(rate * sc.dt / double(nsrc));
    const Domain dc = d;
    const Index si = src_i, sj = src_j, sk = src_k, sr = src_r;
    auto source = KOKKOS_LAMBDA(Index n) -> Real {
      Index px, py, pz; dc.coords(n, px, py, pz);
      const Index x = px - dc.hx, y = py - dc.hy, z = pz - dc.hz;
      return (x >= si - sr && x <= si + sr && y >= sj - sr && y <= sj + sr &&
              z >= sk - sr && z <= sk + sr) ? per_cell : Real(0);
    };
    std::printf("\n  source: cell (%d,%d,%d) r=%d over %d open cells, %.3g units/s at z = %.1f m\n",
                int(src_i), int(src_j), int(src_k), int(src_r), int(nsrc),
                rate, (double(src_k) + 0.5) * g.dx);

    //--------------------------------------------------------------------------
    const std::size_t nsteps = std::size_t(minutes * 60.0 / sc.dt);
    const std::size_t every = std::max<std::size_t>(1, std::size_t(out_every_s / sc.dt));
    std::printf("  run:    %.1f min = %zu steps, frame every %zu steps\n\n",
                minutes, nsteps, every);
    std::string mk = "mkdir -p " + outdir;
    if (std::system(mk.c_str()) != 0) throw std::runtime_error("cannot create " + outdir);

    // min C is reported because it is not zero and should not be hidden. A
    // first-order equilibrium with no limiter undershoots at sharp gradients,
    // so a few parts in ten thousand of the peak appear as small NEGATIVE
    // concentrations. They are bounded and they do not grow, but anything
    // downstream that takes a logarithm or treats "negative" as "solid" has to
    // know: the plan-view plot in doc/fig/plot_urban.py originally painted three
    // million undershooting cells as buildings for exactly that reason.
    std::printf("  %9s %12s %12s %10s %9s %10s %11s\n",
                "t (s)", "in domain", "injected", "retained", "in fluid",
                "min C", "dM/dt");
    std::printf("  %s\n", std::string(80, '-').c_str());

    std::vector<float> Cout(std::size_t(d.n_padded));
    const auto t0 = std::chrono::steady_clock::now();
    double injected = 0.0, prev_mass = 0.0, prev_t = 0.0;
    int frame = 0;
    for (std::size_t t = 0; t <= nsteps; ++t) {
      if (t % every == 0) {
        s.compute_field();
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
        double fluid = 0.0, cmin = 0.0, cmax = 0.0;
        for (Index z = 0; z < g.nz; ++z)
          for (Index y = 0; y < g.ny; ++y)
            for (Index x = 0; x < g.nx; ++x) {
              const Index n = d.id(x, y, z);
              const double c = g.solid(x, y, z) ? -1.0 : double(h(n));
              fluid += g.solid(x, y, z) ? 0.0 : c;
              if (!g.solid(x, y, z)) { cmin = std::min(cmin, c); cmax = std::max(cmax, c); }
              Cout[std::size_t(n)] = float(c);
            }
        // THE BUDGET IS THE WHOLE LATTICE, NOT THE FLUID.
        //
        // Summing the macroscopic field over fluid cells is the obvious thing
        // to do and it is systematically short. Under Esoteric Pull a
        // population emitted toward a wall spends a step in a slot the WALL
        // owns, and no node's field includes it while it is there. The
        // shortfall is a standing boundary-layer quantity -- it grows with wall
        // area and with the advective flux into walls, and it does not
        // oscillate with the parity -- so on a city it is large: 0.7% for a
        // bare box, 2.4% once there is a block in it, 6.9% with advection on,
        // and 13% over Manhattan's 6.87% solid fraction. Thirteen per cent
        // reads exactly like a leak, which is how it was first read here, and
        // it took a closed box with an exact answer to show it was not one.
        // validation/scalar_mass.cpp is that box: the total over the whole
        // lattice comes back as the injection to round-off in every case, with
        // and without buildings, with and without a divergent wind.
        //
        // So the budget uses the conserved total and the fluid sum is reported
        // beside it, as a fraction, rather than in place of it.
        const double mass = double(s.total_population());
        const double now = sc.seconds(t);
        const double dM = (now > prev_t) ? (mass - prev_mass) / (now - prev_t) : 0.0;
        // Finiteness is NOT a sufficient check. An exponentiating run stays
        // perfectly finite for a long time -- the Manchester case that motivated
        // this reached 1e70 without ever producing a NaN, and ran to completion
        // writing 400 MB of garbage. Mass in the domain cannot exceed what was
        // injected by any meaningful factor, so that is the real test.
        // A conservative bound would be mass <= injected. It is not quite that:
        // the outflow condition prescribes its cells and can add a little at a
        // boundary the plume has reached. 1.05 catches a normalisation error --
        // the 1.28 above -- while tolerating that.
        if (injected > 0 && mass > 1.05 * injected && now < 60.0)
          std::printf("  *** mass exceeds injection by %.1f%% before the plume can have\n"
                      "  *** reached a boundary -- the source normalisation is wrong\n",
                      100.0 * (mass / injected - 1.0));
        if (!std::isfinite(mass) || (injected > 0 && mass > 2.0 * injected)) {
          std::printf("  DIVERGED at t = %.1f s: %.4e in the domain against %.4e injected\n",
                      now, mass, injected);
          std::printf("  Check the stability margin reported above.\n");
          rc = 1; break;
        }
        // THE UNDERSHOOT IS THE EARLY WARNING, and the mass test is not.
        //
        // A first-order equilibrium undershoots at sharp gradients, so a few
        // parts in ten thousand of the peak are expected and are not a fault.
        // An exponentiating C div(u) is different: it shows up here first, and
        // by a long way. The Manhattan run at margin 4.6x went 0.03% -> 0.4% ->
        // 3% -> 10% of the peak over nine frames, doubling every eighteen
        // seconds, while mass tracked injection to within a per cent the whole
        // time and never came close to the factor-of-two test above. Whatever
        // is growing geometrically will eventually trip that test, but only
        // after the field has been garbage for several minutes of simulated
        // time -- and on a run that writes a frame a minute, that is the whole
        // animation.
        if (cmax > 0 && -cmin > 0.5 * cmax) {
          std::printf("  DIVERGED at t = %.1f s: undershoot %.3e is %.0f%% of the "
                      "peak %.3e\n", now, cmin, -100.0 * cmin / cmax, cmax);
          std::printf("  Check the stability margin reported above.\n");
          rc = 1; break;
        }
        if (cmax > 0 && -cmin > 0.05 * cmax)
          std::printf("  *** undershoot is %.0f%% of the peak and a first-order\n"
                      "  *** equilibrium does not do that. If it is still growing at\n"
                      "  *** the next frame, the run is exponentiating -- stop it.\n",
                      -100.0 * cmin / cmax);
        std::printf("  %9.1f %12.4e %12.4e %9.1f%% %8.1f%% %10.2e %11.3e\n",
                    now, mass, injected, injected > 0 ? 100.0 * mass / injected : 0.0,
                    mass > 0 ? 100.0 * fluid / mass : 0.0, cmin, dM);
        std::fflush(stdout);
        char p[512];
        std::snprintf(p, sizeof p, "%s/conc_%04d.vtk", outdir.c_str(), frame++);
        write_vtk(p, d, g, Cout, int(t));
        prev_mass = mass; prev_t = now;
      }
      if (t < nsteps) { s.add_source(source); s.step(); injected += rate * sc.dt; }
    }
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("\n  %zu steps in %.1f s wall  (%.1f MLUPS), %d frames in %s/\n\n",
                nsteps, wall,
                double(g.cells()) * double(nsteps) / wall / 1e6, frame, outdir.c_str());
    std::printf("  \"retained\" is the fraction of the release still inside the domain,\n");
    std::printf("  summed over the WHOLE lattice. It is NOT a conservation check: the\n");
    std::printf("  outflow condition prescribes its cells, so what left is injected minus\n");
    std::printf("  retained by construction. What it does show is when the plume reaches\n");
    std::printf("  steady state -- dM/dt -> 0.\n\n");
    std::printf("  \"in fluid\" is how much of that total the macroscopic field sees. The\n");
    std::printf("  rest is sitting in slots owned by walls, in flight for one step, and it\n");
    std::printf("  is the plotted field that is missing it -- not the solver. The fraction\n");
    std::printf("  is set by wall area and by the advective flux into walls; see\n");
    std::printf("  validation/scalar_mass.cpp, which measures it against an exact answer.\n\n");
  } catch (const std::exception& e) {
    std::printf("\nerror: %s\n\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
