//==============================================================================
//  III.B -- Double shear layer.
//
//  De Rosis & Coreixas (2020), Phys. Fluids 32, 117101, Sec. III B, Eqs. (29)-(30).
//
//  Two longitudinal shear layers with a transverse perturbation, in a periodic
//  square of side L. The layers roll up into a Kelvin-Helmholtz pair; the test
//  is a stability probe, and the reported quantities are the kinetic energy and
//  enstrophy normalised by their initial values.
//
//      ux(y) = u0 tanh[k(y/L - 1/4)]   for y/L <= 1/2
//              u0 tanh[k(3/4 - y/L)]   otherwise
//      uy(x) = u0 d sin[2 pi (x/L + 1/4)]
//
//  with k = 80 and d = 0.05.
//
//  DEVIATION FROM THE PAPER. The paper uses Ma = 0.57, i.e. u0 = 0.329. This
//  campaign fixes u0 = 0.02 throughout, and holding Re = 3e4 then forces
//  tau = 0.50051 -- essentially the stability limit. This makes the test far
//  more severe than the published one rather than less, and BGK is expected to
//  fail. That failure is a property of tau -> 1/2, not of the physics the paper
//  illustrates, and is reported as such.
//==============================================================================
#include "Campaign.hpp"
#include "FieldDump.hpp"

using namespace lbm;
using namespace campaign;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index L = 256;
    double Re = 3e4;
    Real u0 = Real(0.02);
    std::string lat = "d2q9", op = "bgk";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-l"   && i + 1 < argc) L  = std::atoi(argv[++i]);
      if (a == "-re"  && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-u0"  && i + 1 < argc) u0 = Real(std::atof(argv[++i]));
      if (a == "-lat" && i + 1 < argc) lat = argv[++i];
      if (a == "-op"  && i + 1 < argc) op  = argv[++i];
    }

    const Real nu = Real(double(u0) * double(L) / Re);
    const double kap = 80.0, del = 0.05;
    const std::size_t t0 = std::size_t(double(L) / double(u0));   // one eddy turnover

    std::printf("III.B double shear layer   lattice %s   operator %s\n", lat.c_str(), op.c_str());
    std::printf("  L = %d   Re = %.0f   u0 = %.3f   nu = %.6e   tau = %.6f\n",
                int(L), Re, double(u0), double(nu), 3.0 * double(nu) + 0.5);
    std::printf("  t0 = L/u0 = %zu steps   (paper: Ma = 0.57; here Ma = %.4f)\n\n",
                t0, double(u0) * std::sqrt(3.0));

    const bool ok = dispatch(lat, op, [&](auto coll) {
      using Coll = decltype(coll);
      using LL   = typename Coll::Lattice;
      const Index nz = 1;
      Domain d(L, L, nz, true, true, true);
      coll.omega = Coll::omega_from_viscosity(nu);
      FluidSolver<LL, EsotericPull<LL>, Coll> s(d, coll);

      const Real u0c = u0; const Index Lc = L;
      const Real kc = Real(kap), dc = Real(del);
      s.initialize_field(KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real x = Real(px - d.hx), y = Real(py - d.hy);
        const Real yy = y / Real(Lc), xx = x / Real(Lc);
        FlowState st;
        st.rho = Real(1);
        st.ux = (yy <= Real(0.5)) ? u0c * Kokkos::tanh(kc * (yy - Real(0.25)))
                                  : u0c * Kokkos::tanh(kc * (Real(0.75) - yy));
        st.uy = u0c * dc * Kokkos::sin(Real(2) * Real(M_PI) * (xx + Real(0.25)));
        st.uz = Real(0);
        return st;
      });

      const Diag d0 = diagnostics(s, d, L, L, nz);
      std::FILE* f = open_out("B_shear_layer", "shear", lat, op);
      if (f) std::fprintf(f, "# t/t0  E/E0  Psi/Psi0   L=%d Re=%.0f u0=%.3f tau=%.6f\n",
                          int(L), Re, double(u0), 3.0 * double(nu) + 0.5);
      std::printf("  %8s %14s %14s\n", "t/t0", "E/E0", "Psi/Psi0");
      const std::size_t probe = t0 / 40;
      double blow = -1;
      for (std::size_t t = 0; t <= t0; ++t) {
        if (t % probe == 0) {
          const Diag dd = diagnostics(s, d, L, L, nz);
          const double tt = double(t) / double(t0);
          if (!dd.finite) { blow = tt; std::printf("  DIVERGED at t/t0 = %.3f\n", tt); break; }
          std::printf("  %8.3f %14.6f %14.6f\n", tt, dd.energy / d0.energy,
                      dd.enstrophy / d0.enstrophy);
          if (f) std::fprintf(f, "%.6f %.8e %.8e\n", tt, dd.energy / d0.energy,
                              dd.enstrophy / d0.enstrophy);
          if (std::getenv("FIGDUMP")) {
            // snapshot at the two instants the figure shows: the roll-up and
            // the fully formed pair of counter-rotating vortices
            const bool want = (std::abs(tt - 0.5) < 1e-9) || (std::abs(tt - 1.0) < 1e-9);
            if (want) {
              using namespace lbm::figdump;
              s.compute_macroscopic();
              auto gx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
              auto gy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
              char nmb[64];
              std::snprintf(nmb, sizeof nmb, "sl_zeta_L%d_t%02d.bin", int(L), int(tt * 10 + 0.5));
              curl_z_slice(nmb, L, L,
                           [&](Index x, Index y) { return double(gx(d.id(x, y))); },
                           [&](Index x, Index y) { return double(gy(d.id(x, y))); });
            }
          }
          std::fflush(stdout);
        }
        if (t < t0) s.step();
      }
      if (f) { std::fprintf(f, "# %s\n", blow < 0 ? "COMPLETED" : "DIVERGED"); std::fclose(f); }
      std::printf("\n  %s\n", blow < 0 ? "completed t/t0 = 1" : "DIVERGED");
    });
    if (!ok) { std::printf("unknown lattice/operator\n"); }
  }
  Kokkos::finalize();
  return 0;
}
