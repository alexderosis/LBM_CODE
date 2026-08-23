//==============================================================================
//  III.D -- Dipole-wall collision.
//
//  De Rosis & Coreixas (2020), Phys. Fluids 32, 117101, Sec. III D, Eqs. (31)-(32),
//  after Clercx & Bruneau, Comput. Fluids 35, 245 (2006) and Mohammed, Graham &
//  Reis, Comput. Fluids 176, 79 (2018).
//
//  Square domain (x, y) in [-1, 1]^2 enclosed by no-slip walls. Two counter-
//  rotating monopoles at (0, +-0.1) of radius r0 = 0.1 and strength |we| = 299.56
//  translate toward the right wall and collide with it. The reference quantities
//  are the kinetic energy and enstrophy at t = 0.25, 0.5, 0.75, normalised so
//  that E(0) = 2 and Psi(0) = 800 exactly:
//
//      E   = 1/2 int |u|^2 dx dy
//      Psi = 1/2 int |omega_z|^2 dx dy
//
//  with the integrals over [-1,1]^2. The characteristic Reynolds number is
//  Re = U D / nu with U = 1 the rms velocity and D = 1 the half width, so in
//  lattice units nu = u0 D_lb / Re.
//
//  DEVIATIONS FROM THE PAPER, both from the campaign settings:
//    * D_lb = 512 only (the paper uses 512, 768, 1024 for Re = 625, 1250, 2500);
//    * u0 = 0.02 fixed rather than chosen per case.
//  Walls are the regularised condition, as the paper states it uses throughout.
//  Corners take rho by extrapolation with Pi^(1) from finite-difference
//  gradients -- the only place in this campaign where the corner treatment
//  matters, since A, B and E are periodic.
//==============================================================================
#include "Campaign.hpp"
#include "boundary/Regularized.hpp"

using namespace lbm;
using namespace campaign;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index Dlb = 512;
    double Re = 625.0, tmax = 0.75;
    Real u0 = Real(0.02);
    std::string lat = "d2q9", op = "cm";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-d"    && i + 1 < argc) Dlb = std::atoi(argv[++i]);
      if (a == "-re"   && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-tmax" && i + 1 < argc) tmax = std::atof(argv[++i]);
      if (a == "-u0"   && i + 1 < argc) u0 = Real(std::atof(argv[++i]));
      if (a == "-lat"  && i + 1 < argc) lat = argv[++i];
      if (a == "-op"   && i + 1 < argc) op  = argv[++i];
    }

    const Index N = 2 * Dlb + 1;               // nodes across [-1, 1], walls included
    const Real nu = Real(double(u0) * double(Dlb) / Re);
    const std::size_t T = std::size_t(tmax * double(Dlb) / double(u0));
    const double dx = 1.0 / double(Dlb);       // lattice spacing in physical units

    std::printf("III.D dipole-wall collision   lattice %s   operator %s\n", lat.c_str(), op.c_str());
    std::printf("  D_lb = %d  (%d x %d nodes)   Re = %.0f   u0 = %.3f   nu = %.6e   tau = %.6f\n",
                int(Dlb), int(N), int(N), Re, double(u0), double(nu), 3.0 * double(nu) + 0.5);
    std::printf("  t up to %.2f   (%zu steps)\n\n", tmax, T);

    const bool ok = dispatch(lat, op, [&](auto coll) {
      using Coll = decltype(coll);
      using LL   = typename Coll::Lattice;
      Domain d(N, N, 1, false, false, true);
      coll.omega = Coll::omega_from_viscosity(nu);
      FluidSolver<LL, EsotericPull<LL>, Coll> s(d, coll);

      s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
      using WS = typename decltype(s)::WallSpec;
      s.set_regularized_walls([&](Index x, Index y, Index) -> WS {
        const bool xl = (x == 0), xr = (x == N - 1), yb = (y == 0), yt = (y == N - 1);
        if ((xl || xr) && (yb || yt)) return WS{NrmCorner, Real(0), Real(0), Real(0)};
        if (xl) return WS{NrmXm, Real(0), Real(0), Real(0)};
        if (xr) return WS{NrmXp, Real(0), Real(0), Real(0)};
        if (yb) return WS{NrmYm, Real(0), Real(0), Real(0)};
        if (yt) return WS{NrmYp, Real(0), Real(0), Real(0)};
        return WS{};
      });

      // The published initial condition is written for |we| = 299.56 with the
      // velocity in physical units; here it is scaled so that the peak speed is
      // u0 in lattice units, which is what fixes the Mach number.
      const Real we = Real(299.56), r0 = Real(0.1);
      const Real x1 = Real(0), y1 = Real(0.1), x2 = Real(0), y2 = Real(-0.1);
      const Real dxc = Real(dx), u0c = u0;
      const Index Dc = Dlb;
      s.initialize_field(KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real X = (Real(px - d.hx) - Real(Dc)) * dxc;   // physical, in [-1,1]
        const Real Y = (Real(py - d.hy) - Real(Dc)) * dxc;
        const Real ra = (X - x1) * (X - x1) + (Y - y1) * (Y - y1);
        const Real rb = (X - x2) * (X - x2) + (Y - y2) * (Y - y2);
        const Real ea = Kokkos::exp(-ra / (r0 * r0));
        const Real eb = Kokkos::exp(-rb / (r0 * r0));
        FlowState st;
        st.rho = Real(1);
        st.ux = -Real(0.5) * we * (Y - y1) * ea + Real(0.5) * we * (Y - y2) * eb;
        st.uy =  Real(0.5) * we * (X - x1) * ea - Real(0.5) * we * (X - x2) * eb;
        st.uz = Real(0);
        // The Reynolds number is defined on U_rms, not on the peak speed, and
        // the analytic field with |we| = 299.56 has U_rms = 1 exactly -- that is
        // what makes E(0) = 2 and Psi(0) = 800 in Eq. (32). So scaling by u0
        // alone puts the lattice rms at u0, which is the velocity that nu and
        // the step count are both built on.
        //
        // Scaling by u0/peak instead (peak/rms = 11.019 for this field) leaves
        // the true Re a factor 11 from its nominal value and the elapsed
        // physical time likewise -- the flow then dissipates far too slowly and
        // the error grows with t, which is exactly what was seen.
        st.ux *= u0c;  st.uy *= u0c;
        return st;
      });

      const Diag d0 = diagnostics(s, d, N, N, 1);
      std::FILE* f = open_out("D_dipole", "dipole_re" + std::to_string(int(Re)), lat, op);
      if (f) std::fprintf(f, "# t  E/E0  Psi/Psi0  E(scaled to 2)  Psi(scaled to 800)"
                             "   D_lb=%d Re=%.0f u0=%.3f tau=%.6f\n",
                          int(Dlb), Re, double(u0), 3.0 * double(nu) + 0.5);
      std::printf("  %8s %12s %12s %12s %12s\n", "t", "E/E0", "Psi/Psi0", "E", "Psi");
      const std::size_t probe = T / 30;
      for (std::size_t t = 0; t <= T; ++t) {
        if (t % probe == 0) {
          const Diag dd = diagnostics(s, d, N, N, 1);
          const double ts = tmax * double(t) / double(T);
          if (!dd.finite) { std::printf("  DIVERGED at t = %.3f\n", ts);
                            if (f) std::fprintf(f, "# DIVERGED\n"); break; }
          const double eR = dd.energy / d0.energy, pR = dd.enstrophy / d0.enstrophy;
          std::printf("  %8.3f %12.6f %12.6f %12.4f %12.2f\n", ts, eR, pR, 2.0 * eR, 800.0 * pR);
          if (f) { std::fprintf(f, "%.6f %.8e %.8e %.6f %.4f\n", ts, eR, pR, 2.0*eR, 800.0*pR);
                   std::fflush(f); }
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
