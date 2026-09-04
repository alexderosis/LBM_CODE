//==============================================================================
//  Rayleigh-Benard at high Rayleigh number -- the free-fall parameterisation.
//
//  A port of a D3Q19 reference driver onto this code's D3Q27 fluid and D3Q7
//  scalar. Layer of depth H, heated from below, periodic in x, isothermal
//  no-slip plates, Boussinesq buoyancy, central moments on both distributions
//  in the reference and on the fluid here.
//
//  ================= WHY THIS IS NOT rayleigh_benard.cu ======================
//  That driver fixes nu and DERIVES g from Ra. It is the right way round for
//  the onset problem it solves -- bracketing Ra_c = 1707.762 -- and it is the
//  wrong way round above Ra ~ 1e6, because the free-fall velocity it implies,
//
//      U_f = sqrt(g beta dT H) = (nu / H) sqrt(Ra / Pr),
//
//  then grows without bound. At Ra = 1e14 with the default nu = 0.02 it is 1.5
//  at H = 4337 and 24.7 at H = 256: supersonic, and the run is nonsense while
//  still printing numbers. Keeping U_f <= 0.05 that way would need H >= 126491.
//
//  So this driver inverts it, exactly as the reference does. U_f is the INPUT
//  and the transport coefficients follow:
//
//      g beta = U_f^2 / (dT H),     nu = U_f H sqrt(Pr / Ra),     alpha = nu/Pr.
//
//  Three choices -- H, Ra, U_f -- and tau is not a fourth. Read it back before
//  running: at the reference point it is 0.5000001264.
//  ===========================================================================
//
//  cs2 IS 1/4 ON D3Q7, AND THAT IS THE ONE LINE THAT DOES NOT TRANSCRIBE.
//  The reference carries its temperature on D3Q19 and so writes
//  tauT = 3 alpha + 1/2. Here the scalar is D3Q7. Copying that line would give
//  alpha' = 3 alpha / 4 -- a Prandtl number 4/3 too large, converged, stable,
//  and wrong. `Scalar` takes the diffusivity and reads its own lattice's cs2,
//  which is why the constructor is handed alpha and not a relaxation rate.
//
//  BUILD THIS IN FP64 -- but the reason is H-dependent, so it is printed rather
//  than asserted. nu = U H sqrt(Pr/Ra), so tau - 1/2 grows LINEARLY with H while
//  the FP32 spacing at 0.5 stays at 2^-24 = 5.96e-08. At the reference point
//  (H = 50) the excess viscosity is 1.264e-07, TWO ulp: nu is quantised by ~24%
//  and Ra with it, and FP32 is simply solving a different problem. By H = 1000
//  it is 2.53e-06, forty-two ulp, and the quantisation is ~1.2% -- immaterial
//  next to a run this under-resolved. The startup line prints the ulp count and
//  says which case it is; do not carry the H = 50 answer to another H.
//
//  MEASURED COST, T4, FP64, central moments (2026-09-04). H = 50 is 5200 cells
//  and launch-latency-bound: 134 MLUPS, 38.7 us/step. H = 1000 is 2.0M cells and
//  runs at ~17 ms/step, i.e. ~120 MLUPS -- NOT bandwidth-bound (that would be
//  ~300), because a consumer card's FP64 ALU rate is 1/32 of its FP32 one and
//  the central-moment collision is arithmetic-heavy. One free-fall time at
//  H = 1000 is 100000 steps, so about 28 minutes.
//  This is not a stability question, it is a question of which Ra is being
//  solved. Configure with -DLBM_DOUBLE=ON. The cost is ~2x here (the kernels
//  are bandwidth-bound, not FP64-throughput-bound) and at these grid sizes the
//  run is launch-latency-bound anyway.
//
//  nz = 1, AND ESOTERIC PULL SURVIVES IT. In a periodic direction one cell
//  deep, wrap(z +/- 1, 1) = z, so a +/-z pair's neighbour IS the node. The
//  in-place scheme still writes slots i and i+1 -- two distinct slots at one
//  node -- so there is no collision and the population simply stays put, which
//  is what streaming into yourself means. That makes the run genuinely 2-D
//  rather than quasi-2-D, and 4x cheaper than the nz = 4 slab in the sibling
//  driver. `-nz 4` is kept so the two can be compared; they agree.
//
//  WALL FAMILY, AND THE HALF CELL. The reference puts BOTH walls ON the node
//  (regularised velocity, regularised temperature) so its layer is ny - 1 deep.
//  This driver uses the halfway pair instead -- bounce-back for the momentum,
//  anti-bounce-back for the scalar -- so both planes sit at y = 0.5 and
//  y = H + 0.5 and the layer is exactly H deep with ny = H + 2. That pairing is
//  the one whose Ra_c this code reproduces, which is the evidence that H means
//  what Ra says it means. The cost of the difference is half a cell in H: 1% in
//  H, 3% in Ra, 0.9% in Nu -- far below this run's discretisation error, see
//  below. An on-node scalar Dirichlet does not exist here yet; adding one is
//  the way to match the reference's family exactly, not a flag on this file.
//
//  ========================= WHAT Nu MEANS HERE ==============================
//  READ THIS BEFORE QUOTING A NUMBER. The 2-D correlation Nu ~ 0.14 Ra^0.29
//  gives Nu ~ 1600 at Ra = 1e14, hence a thermal boundary layer
//  delta = H / 2Nu of 0.016 CELLS at H = 50. Nothing about the boundary layer
//  is resolved; a resolved 2-D DNS at this Ra wants H ~ 25000. Whatever Nu this
//  prints is a property of the discretisation -- an implicit LES, with the
//  central-moment operator at omega -> 2 supplying the dissipation -- and not a
//  measurement of Ra = 1e14.
//
//  The driver prints the evidence rather than the claim. Three estimators:
//
//    Nu_vol  = 1 + H <v T> / (alpha dT),   the volume average;
//    Nu_bot  = H (T_hot - <T>_{y=1}) / (0.5 dT),   the plate gradient below;
//    Nu_top  = H (<T>_{y=H} - T_cold) / (0.5 dT),  and above.
//
//  At a resolved steady state all three agree. The size of their disagreement
//  is the honest error bar, and it is why `-h` is the first flag to sweep.
//
//  ============ AND AT THE REFERENCE POINT IT DOES NOT SURVIVE =============
//  H = 50, Ra = 1e14, FP64, central moments, T4: conduction runs cleanly for
//  ~500 free-fall times, then the run DIVERGES the moment buoyancy wins ---
//  Nu_vol 1.0383 -> 6.50 -> 504.5 -> nan over 100 free-fall times, with max|u|
//  climbing 2.8e-06 -> 5.8e-05 -> 1.0e-03. tau_f - 1/2 = 1.26e-07 leaves nothing
//  to damp the flow once it starts, and a Nu_vol of 504 is impossible anyway
//  against the H/2 = 25 ceiling above.
//
//  The STEP at which it blows up is deliberately not quoted as a measurement:
//  this tree has been burned by that before (the blow-up step moves by 2x with
//  the Kokkos backend alone). What is reproducible is the ORDER of events ---
//  clean conduction, then divergence at onset --- and that raising H is the
//  lever, because nu = U H sqrt(Pr/Ra) grows with H: H = 1000 gives
//  tau_f - 1/2 = 2.53e-06, twenty times further off the floor, and lifts the Nu
//  ceiling from 25 to 500 at the same time. Lowering Ra at fixed H does the same
//  thing to tau and does NOT help the resolution, which is why it is the worse
//  of the two knobs.
//  ===========================================================================
//
//  ONE MORE ARITHMETIC NOTE ON Nu. The reference sums v T over ALL nx*ny nodes
//  and divides by (nx - 1). The exact volume average times H/alpha divides by
//  nx*ny*nz/H, which is 103.02 for its 101 x 51 grid against the 100 it uses:
//  its Nu - 1 runs 3.0% high. `Nu_ref` below reproduces that normalisation
//  verbatim (with nz folded in, which its nz = 1 grid leaves implicit) so the
//  two codes can be compared directly; `Nu_vol` is the exact
//  one. Do not mix them in a table.
//  ===========================================================================
//
//  THE SCALAR OPERATOR IS NOT BGK, AND IT CANNOT BE. The reference relaxes only
//  the first-order thermal moments and puts every higher one at equilibrium.
//  BGK on D3Q7 instead relaxes the ghost moments at the same omega, and at
//  omega = 1.99999905 that is a reflection: with `-sop bgk` the near-wall
//  temperature RINGS rather than relaxing -- Nu_bot ran 100 -> 39.4 -> 78.7 over
//  ten free-fall times, bounded, so easy to average over and quote. `-sop reg`
//  is the default and is the D3Q7 form of the reference's operator; the flag is
//  kept because reproducing the ringing on demand is how it stays documented.
//
//  TWO SMALLER DEVIATIONS FROM THE REFERENCE, both deliberate.
//  1. Buoyancy uses rho0 = 1 rather than the local rho (core.cuh's
//     ForceBoussinesq). Under Boussinesq those differ by the density
//     perturbation, 1% at t = 0 and decaying -- and holding rho0 constant in
//     the buoyancy term is what Boussinesq means.
//  2. The seeded density mode is cos(2 pi x / nx), not the reference's
//     cos(2 pi x / (nx-1)), which has period nx-1 on an nx-periodic domain and
//     so does not close. The seed only has to pick a mode.
//
//  THE RESIDUAL IS MEASURED OVER THE OUTPUT INTERVAL, NOT PER STEP. A per-step
//  change is bounded by the timestep and shrinks as the grid refines whether or
//  not the flow has settled; the reference's 1e-12 per-step threshold cannot
//  fire on a turbulent field and would fire on a slow one. Over an interval it
//  is a statement about the field.
//
//  WHAT THIS DOES NOT DO: no MPI, no grid stretching (so the boundary layer
//  costs the same as the bulk), no restart -- a run that outlives its session
//  is lost, which at these step counts is the binding constraint, not memory.
//
//    usage: rb_high_ra [-h H] [-aspect A] [-ra RA] [-pr PR] [-u U_REF]
//                      [-amp A] [-tf N] [-out N] [-op bgk|cm] [-sop reg|bgk]
//                      [-nz NZ] [-vtk]
//==============================================================================
#include "lbm/backend.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// Fluid at rest. One row near mid-depth carries a single-mode density
// perturbation -- the reference's seed, and deliberately not noise: the
// question it asks is whether THIS mode grows.
//------------------------------------------------------------------------------
struct RbInit {
  int nx, seed_row;
  Real amp;
  LBM_HD Macro operator()(int x, int y, int) const {
    Real r = Real(1);
    if (y == seed_row)
      r += amp * Real(cos(2.0 * M_PI * double(x) / double(nx)));
    return Macro{r, Real(0), Real(0), Real(0)};
  }
};

// The whole layer starts cold. The hot plate is a wall value, not an initial
// condition, so the entire dT is dropped across the bottom half cell at t = 0.
struct ColdInit {
  Real Tc;
  LBM_HD Real operator()(int, int, int) const { return Tc; }
};

//------------------------------------------------------------------------------
// The reference's VTK, same fields and same ordering, so the two can be opened
// side by side. Off by default: one file per free-fall time for 10000 of them
// is not an output, it is a disk.
//------------------------------------------------------------------------------
static void write_vtk(int step, int nx, int ny, int nz,
                      const std::vector<Real>& T, const std::vector<Real>& ux,
                      const std::vector<Real>& uy, const std::vector<Real>& uz) {
  std::ostringstream name;
  name << "vtk_fluid/fluid_t" << step << ".vtk";
  std::ofstream o(name.str().c_str());
  o << "# vtk DataFile Version 3.0\nfluid_state\nASCII\nDATASET RECTILINEAR_GRID\n";
  o << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
  o << "X_COORDINATES " << nx << " float\n";
  for (int i = 0; i < nx; ++i) o << i << " ";
  o << "\nY_COORDINATES " << ny << " float\n";
  for (int j = 0; j < ny; ++j) o << j << " ";
  o << "\nZ_COORDINATES " << nz << " float\n";
  for (int k = 0; k < nz; ++k) o << k << " ";
  o << "\nPOINT_DATA " << nx * ny * nz << "\n";
  o << "SCALARS Temperature float 1\nLOOKUP_TABLE default\n";
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        o << double(T[std::size_t(node_id(x, y, z, nx, ny))]) << "\n";
  o << "VECTORS velocity_vector float\n";
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const std::size_t n = std::size_t(node_id(x, y, z, nx, ny));
        o << double(ux[n]) << " " << double(uy[n]) << " " << double(uz[n]) << "\n";
      }
}

int main(int argc, char** argv) {
  int H = 50, aspect = 2, nz = 1;
  double Ra = 1e14, Pr = 0.71, U = 0.01, amp = 0.01;
  double tf = 10000.0, out_every = 1.0;
  std::string op = "cm", sop = "reg";
  bool vtk = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h"      && i + 1 < argc) H         = std::atoi(argv[++i]);
    if (a == "-aspect" && i + 1 < argc) aspect    = std::atoi(argv[++i]);
    if (a == "-nz"     && i + 1 < argc) nz        = std::atoi(argv[++i]);
    if (a == "-ra"     && i + 1 < argc) Ra        = std::atof(argv[++i]);
    if (a == "-pr"     && i + 1 < argc) Pr        = std::atof(argv[++i]);
    if (a == "-u"      && i + 1 < argc) U         = std::atof(argv[++i]);
    if (a == "-amp"    && i + 1 < argc) amp       = std::atof(argv[++i]);
    if (a == "-tf"     && i + 1 < argc) tf        = std::atof(argv[++i]);
    if (a == "-out"    && i + 1 < argc) out_every = std::atof(argv[++i]);
    if (a == "-op"     && i + 1 < argc) op        = argv[++i];
    if (a == "-sop"    && i + 1 < argc) sop       = argv[++i];
    if (a == "-vtk")                    vtk       = true;
  }

  // The plates and the gauge. T_ref = T0 is the shifted-storage reference: the
  // populations then carry T - T0, symmetric about zero, which is the whole
  // point of the shift (core.cuh: "set it to the mean temperature").
  const double T_hot = 1.0, T_cold = 0.0, T0 = 0.5 * (T_hot + T_cold);
  const double dT = T_hot - T_cold;

  const double gbeta = U * U / (dT * double(H));
  const double nu    = U * double(H) * std::sqrt(Pr / Ra);
  const double alpha = nu / Pr;

  const int nx = aspect * H, ny = H + 2;
  const Op which = (op == "bgk") ? Op::BGK : Op::CentralMoments;
  const ScalarOp swhich = (sop == "bgk") ? ScalarOp::BGK : ScalarOp::Regularised;

  const double t_ff = double(H) / U;                       // one free-fall time
  const std::size_t T_end = std::size_t(tf * t_ff);
  const std::size_t probe = std::size_t(out_every * t_ff) ? std::size_t(out_every * t_ff) : 1;

  // The resolution statement, printed rather than assumed. See the banner.
  const double Nu_est = 0.14 * std::pow(Ra, 0.29);
  const double cells_in_bl = double(H) / (2.0 * Nu_est);

  std::printf("Rayleigh-Benard, free-fall scaling   %s   D3Q27 fluid / D3Q7 scalar"
              "   operator %s   %s\n",
              backend::on_device ? "CUDA native" : "HOST reference",
              which == Op::BGK ? "bgk" : "cm", sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  scalar operator: %s\n",
              swhich == ScalarOp::BGK ? "BGK  (rings at omega -> 2; see the banner)"
                                      : "regularised (ghost moments annihilated)");
  std::printf("  %d x %d x %d   H = %d   aspect = %d   %.3e cells\n",
              nx, ny, nz, H, aspect, double(nx) * ny * nz);
  std::printf("  Ra = %.3e   Pr = %.4f   U_f = %.4g   Ma = %.4f\n",
              Ra, Pr, U, U * std::sqrt(3.0));
  std::printf("  g beta = %.6e   nu = %.6e   alpha = %.6e\n", gbeta, nu, alpha);
  std::printf("  tau_f  = %.10f (omega %.8f)   [D3Q27, cs2 = 1/3]\n",
              nu / (1.0 / 3.0) + 0.5, 1.0 / (nu / (1.0 / 3.0) + 0.5));
  std::printf("  tau_g  = %.10f (omega %.8f)   [D3Q7,  cs2 = 1/4 -- NOT 3a+1/2]\n",
              alpha / 0.25 + 0.5, 1.0 / (alpha / 0.25 + 0.5));
  if (sizeof(Real) == 4) {
    const double ulps = nu * 3.0 / 5.96e-8;        // FP32 spacing at 0.5 is 2^-24
    std::printf("  ** FP32: tau - 1/2 = %.3e is %.1f ulp at 0.5, so nu and Ra are\n"
                "     quantised by about %.2f%%.%s **\n",
                nu * 3.0, ulps, 50.0 / ulps,
                ulps < 10.0 ? "  REBUILD WITH -DLBM_DOUBLE=ON."
                            : "  Tolerable, but FP64 is the reference.");
  }
  std::printf("  one free-fall time = %.0f steps;  %zu steps = %.0f of them\n",
              t_ff, T_end, tf);
  std::printf("  RESOLUTION: Nu ~ %.0f (2-D, 0.14 Ra^0.29) -> thermal BL = %.4f cells."
              "  %s\n\n", Nu_est, cells_in_bl,
              cells_in_bl >= 10.0 ? "Resolved." : "UNDER-RESOLVED: Nu below is the scheme, not Ra.");

  backend::Fluid  fl(nx, ny, nz, which, Real(nu));
  backend::Scalar sc(nx, ny, nz, Real(alpha), Real(T0), swhich);

  // Geometry. Momentum and thermal walls are the SAME two layers, so the two
  // halfway planes coincide at y = 0.5 and y = H + 0.5.
  std::vector<std::uint8_t> ff(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  std::vector<std::uint8_t> sf(std::size_t(nx) * ny * nz, std::uint8_t(ScalarBulk));
  std::vector<Real>         sw(std::size_t(nx) * ny * nz, Real(0));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      const std::size_t lo = std::size_t(node_id(x, 0,      z, nx, ny));
      const std::size_t hi = std::size_t(node_id(x, ny - 1, z, nx, ny));
      ff[lo] = Solid;            ff[hi] = Solid;
      sf[lo] = ScalarDirichlet;  sf[hi] = ScalarDirichlet;
      sw[lo] = Real(T_hot);      sw[hi] = Real(T_cold);      // hot below
    }
  fl.set_geometry(ff);
  sc.set_geometry(sf, sw);

  fl.enable_velocity_output();
  sc.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  // gy is POSITIVE: a parcel hotter than T0 must be pushed away from gravity.
  BodyForce b;
  b.T = sc.field_device();
  b.gx = Real(0); b.gy = Real(gbeta); b.gz = Real(0);
  b.rho0 = Real(1); b.beta = Real(1); b.T0 = Real(T0);
  fl.set_force(b, ForceBoussinesq);

  fl.initialise_with(RbInit{nx, ny / 2, Real(amp)});
  sc.initialise_with(ColdInit{Real(T_cold)});

  if (vtk) { if (std::system("mkdir -p vtk_fluid")) {} }
  std::FILE* series = std::fopen("rb_high_ra.dat", "wt");
  std::fprintf(series, "# t/t_ff  Nu_vol  Nu_bot  Nu_top  Nu_ref  max|u|  Ma  residual\n");

  std::vector<Real> rho, ux, uy, Tf, uz, Tprev;

  std::printf("  %10s %11s %11s %11s %11s %11s %11s\n",
              "t/t_ff", "Nu_vol", "Nu_bot", "Nu_top", "Nu_ref", "max|u|", "residual");

  const auto wall0 = std::chrono::steady_clock::now();
  bool diverged = false;
  // The summary must count the steps actually TAKEN, not the steps asked for.
  // A run that diverges at 3.25e6 of 5e7 otherwise reports the full 5e7 and an
  // MLUPS fifteen times too high -- a throughput number nothing measured.
  std::size_t steps_run = T_end;

  for (std::size_t t = 0; t <= T_end; ++t) {
    if (t % probe == 0) {
      fl.macroscopic_to_host(rho, ux, uy, uz);
      sc.field_to_host(Tf);

      // Volume average over the H FLUID rows only. The exact normalisation:
      // <vT> = sum / (nx H nz), and Nu = 1 + H <vT> / (alpha dT), so the
      // divisor collapses to nx nz alpha dT.
      double flux = 0.0, peak = 0.0, bot = 0.0, top = 0.0, all = 0.0;
      double num = 0.0, den = 0.0;
      for (int z = 0; z < nz; ++z)
        for (int y = 1; y <= H; ++y)
          for (int x = 0; x < nx; ++x) {
            const std::size_t n = std::size_t(node_id(x, y, z, nx, ny));
            flux += double(uy[n]) * double(Tf[n]);
            const double s = std::sqrt(double(ux[n]) * double(ux[n]) +
                                       double(uy[n]) * double(uy[n]) +
                                       double(uz[n]) * double(uz[n]));
            if (s > peak) peak = s;
            if (y == 1) bot += double(Tf[n]);
            if (y == H) top += double(Tf[n]);
          }
      // The reference's normalisation, verbatim: every node, divided by nx-1.
      for (std::size_t n = 0; n < std::size_t(nx) * ny * nz; ++n)
        all += double(uy[n]) * double(Tf[n]);

      const double plate = double(nx) * nz;
      const double Nu_vol = 1.0 + flux / (double(nx) * nz * alpha * dT);
      const double Nu_bot = double(H) * (T_hot - bot / plate) / (0.5 * dT);
      const double Nu_top = double(H) * (top / plate - T_cold) / (0.5 * dT);
      const double Nu_ref = 1.0 + all / (alpha * dT * double(nx - 1) * nz);

      // Whole-field residual over the interval, not per step. See the banner.
      double resid = 0.0;
      if (!Tprev.empty()) {
        for (std::size_t n = 0; n < Tf.size(); ++n) {
          const double d = double(Tf[n]) - double(Tprev[n]);
          num += d * d;  den += double(Tf[n]) * double(Tf[n]);
        }
        resid = (den > 0.0) ? std::sqrt(num / den) : 0.0;
      }
      Tprev = Tf;

      std::printf("  %10.2f %11.4f %11.4f %11.4f %11.4f %11.3e %11.3e\n",
                  double(t) / t_ff, Nu_vol, Nu_bot, Nu_top, Nu_ref, peak, resid);
      std::fflush(stdout);
      std::fprintf(series, "%.6f %.8f %.8f %.8f %.8f %.6e %.6e %.6e\n",
                   double(t) / t_ff, Nu_vol, Nu_bot, Nu_top, Nu_ref,
                   peak, peak * std::sqrt(3.0), resid);
      std::fflush(series);

      if (vtk) write_vtk(int(t), nx, ny, nz, Tf, ux, uy, uz);

      if (!std::isfinite(Nu_vol) || peak > 1.0) {
        std::printf("  DIVERGED at t = %zu  (max|u| = %.3e)\n", t, peak);
        diverged = true;
        steps_run = t;
        break;
      }
    }
    if (t < T_end) {
      // Refresh T first, so the fluid collides against the temperature at its
      // OWN time level -- a first-order splitting error otherwise, and one that
      // does not vanish under refinement.
      sc.compute_field();
      fl.step();
      sc.step();
    }
  }

  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall0).count();
  std::fclose(series);
  std::printf("\n  %zu steps in %.2f s  ->  %.1f MLUPS\n",
              steps_run, sec, double(fl.nodes()) * double(steps_run) / sec / 1e6);
  std::printf("  series in rb_high_ra.dat%s\n", vtk ? ", fields in vtk_fluid/" : "");
  return diverged ? 1 : 0;
}
