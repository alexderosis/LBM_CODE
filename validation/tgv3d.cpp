//==============================================================================
//  III.E -- Three-dimensional Taylor-Green vortex.
//
//  De Rosis & Coreixas (2020), Phys. Fluids 32, 117101, Sec. III E, Eq. (33).
//
//  Cubic periodic domain of edge D, initialised with
//
//      ux =  u0 cos x sin y sin z
//      uy = -u0/2 sin x cos y sin z
//      uz = -u0/2 sin x sin y cos z
//
//  on x, y, z in [0, 2 pi). Unlike the classical Taylor-Green vortex there is no
//  exact solution: the reported quantity is the kinetic energy normalised by its
//  initial value, and the comparison is between lattices.
//
//  This field IS solenoidal:
//      d(ux)/dx + d(uy)/dy + d(uz)/dz
//        = -u0 sin x sin y sin z + u0/2 sin x sin y sin z + u0/2 sin x sin y sin z = 0,
//  which is why the two transverse components carry the factor 1/2.
//
//  DEVIATIONS FROM THE PAPER, both forced by the campaign settings:
//    * D = 64 rather than 128, for cost;
//    * u0 = 0.02 fixed, so Ma = 0.035 rather than the paper's 0.2/0.4/0.6 sweep.
//      Run length scales as 1/u0, so this is 17x longer per unit physical time
//      than their Ma = 0.6 case. At Re = 30000 it also puts tau at 0.50026.
//  Enstrophy is reported alongside energy since it is the more sensitive of the
//  two and is what resolves differences between lattices.
//==============================================================================
#include "Campaign.hpp"

using namespace lbm;
using namespace campaign;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index D = 64;
    double Re = 1600.0, tmax = 10.0;      // t* = t u0 / D
    Real u0 = Real(0.02);
    std::string lat = "d3q19", op = "cm";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-d"    && i + 1 < argc) D  = std::atoi(argv[++i]);
      if (a == "-re"   && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-tmax" && i + 1 < argc) tmax = std::atof(argv[++i]);
      if (a == "-u0"   && i + 1 < argc) u0 = Real(std::atof(argv[++i]));
      if (a == "-lat"  && i + 1 < argc) lat = argv[++i];
      if (a == "-op"   && i + 1 < argc) op  = argv[++i];
    }

    const Real nu = Real(double(u0) * double(D) / Re);
    const std::size_t T = std::size_t(tmax * double(D) / double(u0));

    std::printf("III.E 3D Taylor-Green   lattice %s   operator %s\n", lat.c_str(), op.c_str());
    std::printf("  D = %d   Re = %.0f   u0 = %.3f   Ma = %.4f   nu = %.6e   tau = %.6f\n",
                int(D), Re, double(u0), double(u0) * std::sqrt(3.0),
                double(nu), 3.0 * double(nu) + 0.5);
    std::printf("  t* = t u0 / D up to %.1f   (%zu steps, %d^3 = %d nodes)\n\n",
                tmax, T, int(D), int(D * D * D));

    const bool ok = dispatch(lat, op, [&](auto coll) {
      using Coll = decltype(coll);
      using LL   = typename Coll::Lattice;
      Domain d(D, D, D, true, true, true);
      coll.omega = Coll::omega_from_viscosity(nu);
      FluidSolver<LL, EsotericPull<LL>, Coll> s(d, coll);

      const Real u0c = u0;
      const Real k = Real(2.0 * M_PI) / Real(D);     // one period across the box
      s.initialize_field(KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real x = k * Real(px - d.hx), y = k * Real(py - d.hy), z = k * Real(pz - d.hz);
        FlowState st;
        st.rho = Real(1);
        st.ux =  u0c * Kokkos::cos(x) * Kokkos::sin(y) * Kokkos::sin(z);
        st.uy = -Real(0.5) * u0c * Kokkos::sin(x) * Kokkos::cos(y) * Kokkos::sin(z);
        st.uz = -Real(0.5) * u0c * Kokkos::sin(x) * Kokkos::sin(y) * Kokkos::cos(z);
        return st;
      });

      const Diag d0 = diagnostics(s, d, D, D, D);
      std::FILE* f = open_out("E_tgv3d", "tgv3d_re" + std::to_string(int(Re)), lat, op);
      if (f) std::fprintf(f, "# t*  E/E0  Psi/Psi0   D=%d Re=%.0f u0=%.3f tau=%.6f\n",
                          int(D), Re, double(u0), 3.0 * double(nu) + 0.5);
      std::printf("  %8s %14s %14s\n", "t*", "E/E0", "Psi/Psi0");
      const std::size_t probe = T / 20;
      for (std::size_t t = 0; t <= T; ++t) {
        if (t % probe == 0) {
          const Diag dd = diagnostics(s, d, D, D, D);
          const double ts = tmax * double(t) / double(T);
          if (!dd.finite) { std::printf("  DIVERGED at t* = %.3f\n", ts);
                            if (f) std::fprintf(f, "# DIVERGED\n"); break; }
          std::printf("  %8.3f %14.6f %14.6f\n", ts, dd.energy / d0.energy,
                      dd.enstrophy / d0.enstrophy);
          if (f) { std::fprintf(f, "%.6f %.8e %.8e\n", ts, dd.energy / d0.energy,
                                dd.enstrophy / d0.enstrophy); std::fflush(f); }
          std::fflush(stdout);
        }
        if (t < T) s.step();
      }
      if (f) std::fclose(f);
    });
    if (!ok) std::printf("unknown lattice/operator\n");
  }
  Kokkos::finalize();
  return 0;
}
