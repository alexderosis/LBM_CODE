//==============================================================================
//  Taylor-Green vortex, THE BENCHMARK CONFIGURATION.
//
//  validation/tgv3d.cpp runs the Taylor-Green case of De Rosis & Coreixas
//  (2020), which is a Taylor-Green-TYPE flow and explicitly not a benchmark:
//  its own header says there is no exact solution and the comparison is between
//  lattices. This file runs the other one -- the canonical DNS benchmark, the
//  case with published reference data -- and it differs in two ways that are
//  easy to miss and fatal to a comparison.
//
//  1. THE INITIAL CONDITION HAS w = 0.
//
//         u =  V0 sin(x/L) cos(y/L) cos(z/L)
//         v = -V0 cos(x/L) sin(y/L) cos(z/L)
//         w =  0
//
//     against tgv3d.cpp's three non-zero components with factors of 1/2. Both
//     are solenoidal; they are different flows. Here
//         du/dx + dv/dy = (V0/L)cos cos cos - (V0/L)cos cos cos = 0.
//
//  2. THE REYNOLDS NUMBER IS DEFINED ON L, NOT ON THE BOX.
//     The domain is [0, 2 pi L)^3, so a grid of D cells has L = D/(2 pi), and
//         Re = V0 L / nu = V0 D / (2 pi nu).
//     tgv3d.cpp uses Re = u0 D / nu. The two differ by 2 pi: what that file
//     calls Re = 1600 is Re = 255 here. Running the benchmark at its stated
//     Reynolds number therefore needs nu smaller by 2 pi -- at D = 512,
//     u0 = 0.02 that is tau = 0.503056 rather than 0.519200.
//
//  The initial density carries the analytic pressure,
//      p = p0 + (rho0 V0^2/16)(cos(2x/L) + cos(2y/L))(cos(2z/L) + 2),
//  mapped through p = rho cs^2. Starting at uniform density instead launches an
//  acoustic transient that contaminates the first eddy turnover.
//
//  WHAT IS REPORTED, in the benchmark's own non-dimensionalisation
//  (t* = t V0/L, lengths in L, velocities in V0):
//
//    Ek*        = <|u|^2/2> / V0^2, which is exactly 1/8 at t* = 0
//    eps_E      = -d(Ek*)/d(t*), the dissipation rate from the energy decay
//    eps_zeta   = 2 nu* <zeta*>, the dissipation rate from the enstrophy
//
//  The two dissipation estimates agree only while the flow is resolved, so
//  their DIFFERENCE is the resolution diagnostic: eps_E includes numerical
//  dissipation, eps_zeta counts only what the resolved vorticity accounts for.
//  A gap between them is under-resolution measured rather than assumed.
//
//  No reference values are hard-coded here. The published DNS peak dissipation
//  for Re = 1600 falls near t* = 9; compare against your own reference data
//  rather than against a number transcribed into a comment.
//
//    usage: tgv3d_bench [-d N] [-re R] [-tmax T] [-u0 U] [-lat d3q27] [-op cm]
//==============================================================================
#include "Campaign.hpp"

using namespace lbm;
using namespace campaign;

//------------------------------------------------------------------------------
// A struct with a template operator(), not a generic lambda -- nvcc forbids an
// extended __host__ __device__ lambda inside a generic lambda, and the
// KOKKOS_LAMBDA laying down the initial condition sits directly in this body.
//------------------------------------------------------------------------------
struct TaylorGreenBench {
  Index D;
  double Re, tmax;
  Real u0, nu;
  std::size_t T;
  std::string lat, op;

  template <class Coll>
  void operator()(Coll coll) const {
    using LL = typename Coll::Lattice;
    Domain d(D, D, D, true, true, true);
    coll.omega = Coll::omega_from_viscosity(nu);
    FluidSolver<LL, EsotericPull<LL>, Coll> s(d, coll);

    const Real u0c = u0;
    // k x = x/L, since L = D/(2 pi).
    const Real k = Real(2.0 * M_PI) / Real(D);
    const Real dr = Real(3.0) * u0 * u0 / Real(16.0);   // (V0^2/16)/cs^2, cs^2 = 1/3
    s.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real x = k * Real(px - d.hx), y = k * Real(py - d.hy), z = k * Real(pz - d.hz);
      FlowState st;
      st.rho = Real(1) + dr * (Kokkos::cos(Real(2) * x) + Kokkos::cos(Real(2) * y))
                            * (Kokkos::cos(Real(2) * z) + Real(2));
      st.ux =  u0c * Kokkos::sin(x) * Kokkos::cos(y) * Kokkos::cos(z);
      st.uy = -u0c * Kokkos::cos(x) * Kokkos::sin(y) * Kokkos::cos(z);
      st.uz =  Real(0);
      return st;
    });

    // Non-dimensionalisation. diagnostics() returns SUMS in lattice units, so
    // divide by the cell count for means, then scale: velocities by V0 and
    // lengths by L. Vorticity carries L/V0 because it is a velocity gradient.
    const double N   = double(D) * double(D) * double(D);
    const double L   = double(D) / (2.0 * M_PI);
    const double sE  = 1.0 / (double(u0) * double(u0));
    const double sZ  = (L / double(u0)) * (L / double(u0));
    const double nus = 1.0 / Re;                       // nu* = nu/(V0 L) = 1/Re

    std::FILE* f = open_out("E_tgv3d_bench",
                            "tgvb_re" + std::to_string(int(Re)), lat, op);
    if (f) std::fprintf(f, "# t*  Ek*  eps_E  eps_zeta   D=%d Re=%.0f u0=%.4f tau=%.6f\n",
                        int(D), Re, double(u0), 3.0 * double(nu) + 0.5);
    std::printf("  %8s %13s %13s %13s %11s\n",
                "t*", "Ek*", "eps_E", "eps_zeta", "gap %");
    std::printf("  %s\n", std::string(64, '-').c_str());

    const std::size_t probe = T / 40 ? T / 40 : 1;
    double prev_E = 0.0, prev_t = 0.0;
    bool first = true;
    for (std::size_t t = 0; t <= T; ++t) {
      if (t % probe == 0) {
        const Diag dd = diagnostics(s, d, D, D, D);
        const double ts = tmax * double(t) / double(T);
        if (!dd.finite) { std::printf("  DIVERGED at t* = %.3f\n", ts);
                          if (f) std::fprintf(f, "# DIVERGED\n"); break; }
        const double Ek = dd.energy / N * sE;
        const double zeta = dd.enstrophy / N * sZ;
        const double eps_z = 2.0 * nus * zeta;
        // Backward difference: the first probe has no predecessor, so its
        // energy-based dissipation is undefined rather than zero.
        const double eps_E = first ? 0.0 / 0.0 : -(Ek - prev_E) / (ts - prev_t);
        const double gap = (eps_z > 0 && !first) ? 100.0 * (eps_E - eps_z) / eps_z : 0.0 / 0.0;
        std::printf("  %8.3f %13.6f %13.6f %13.6f %11.1f\n", ts, Ek, eps_E, eps_z, gap);
        if (f) { std::fprintf(f, "%.6f %.8e %.8e %.8e\n", ts, Ek, eps_E, eps_z);
                 std::fflush(f); }
        std::fflush(stdout);
        prev_E = Ek; prev_t = ts; first = false;
      }
      if (t < T) s.step();
    }
    if (f) std::fclose(f);
  }
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index D = 64;
    double Re = 1600.0, tmax = 20.0;    // t* = t V0/L, benchmark convention
    Real u0 = Real(0.02);
    std::string lat = "d3q27", op = "cm";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-d"    && i + 1 < argc) D    = Index(std::atoi(argv[++i]));
      if (a == "-re"   && i + 1 < argc) Re   = std::atof(argv[++i]);
      if (a == "-tmax" && i + 1 < argc) tmax = std::atof(argv[++i]);
      if (a == "-u0"   && i + 1 < argc) u0   = Real(std::atof(argv[++i]));
      if (a == "-lat"  && i + 1 < argc) lat  = argv[++i];
      if (a == "-op"   && i + 1 < argc) op   = argv[++i];
    }

    // THE WHOLE POINT OF THIS FILE. Re is defined on L = D/(2 pi), not on D.
    const double L  = double(D) / (2.0 * M_PI);
    const Real   nu = Real(double(u0) * L / Re);
    const std::size_t T = std::size_t(tmax * L / double(u0));

    std::printf("\n3D Taylor-Green, BENCHMARK configuration   %s   %s   %s\n",
                lat.c_str(), op.c_str(), precision_name());
    std::printf("  D = %d   L = D/2pi = %.3f   Re = V0 L/nu = %.0f\n", int(D), L, Re);
    std::printf("  u0 = %.4f   Ma = %.4f   nu = %.6e   tau = %.6f\n",
                double(u0), double(u0) * std::sqrt(3.0), double(nu),
                3.0 * double(nu) + 0.5);
    std::printf("  t* = t V0/L up to %.1f   (%zu steps, %d^3 = %.1f M nodes)\n",
                tmax, T, int(D), double(D) * D * D / 1e6);
    std::printf("  NOTE: tgv3d.cpp's Re = u0 D/nu would be %.0f for this nu.\n\n",
                double(u0) * double(D) / double(nu));

    if (!dispatch(lat, op, TaylorGreenBench{D, Re, tmax, u0, nu, T, lat, op}))
      std::printf("  unknown lattice/operator combination\n");
  }
  Kokkos::finalize();
  return 0;
}
