//==============================================================================
//  III.A -- Taylor-Green vortex (two-dimensional).
//
//  De Rosis & Coreixas (2020), Phys. Fluids 32, 117101, Sec. III A, Eqs. (26)-(28).
//
//  Periodic square of side 2 pi discretised by N x N points, one point in z.
//  The flow decays exactly, so this is a pure convergence test:
//
//      u(x, t) = u(x, 0) exp(-t / T),      T = 1 / (2 xi^2 nu),   xi = 2 pi / N,
//
//  and the error is the relative L2 norm of (analytic - numerical) at t = T.
//
//  SIGN CORRECTION. The paper prints the initial velocity as
//  u0 [cos(xi x) sin(xi y), sin(xi x) cos(xi y), 0], which has divergence
//  -2 u0 xi sin(xi x) sin(xi y) and is therefore NOT solenoidal -- almost
//  certainly a lost minus sign in typesetting. The standard, divergence-free
//  Taylor-Green field is used here:
//
//      u(x, 0) = u0 [-cos(xi x) sin(xi y), sin(xi x) cos(xi y), 0].
//
//  Likewise the density is initialised from the exact pressure field,
//  rho = rho0 [1 - (3 u0^2 / 4)(cos 2 xi x + cos 2 xi y)], which is the
//  standard form; the leading factor 3 in the printed equation would give
//  rho ~ 3 and cannot be right.
//==============================================================================
#include "Campaign.hpp"

using namespace lbm;
using namespace campaign;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    double Re = 1000.0;
    Real u0 = Real(0.02);
    std::string lat = "d2q9", op = "cm";
    std::vector<Index> Ns = {8, 16, 32, 64, 128, 256};
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-re"  && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-u0"  && i + 1 < argc) u0 = Real(std::atof(argv[++i]));
      if (a == "-lat" && i + 1 < argc) lat = argv[++i];
      if (a == "-op"  && i + 1 < argc) op  = argv[++i];
      if (a == "-nmax" && i + 1 < argc) {
        const Index m = std::atoi(argv[++i]);
        std::vector<Index> k; for (Index n : Ns) if (n <= m) k.push_back(n); Ns = k;
      }
    }

    std::printf("III.A Taylor-Green vortex   lattice %s   operator %s\n", lat.c_str(), op.c_str());
    std::printf("  Re = %.0f   u0 = %.3f   error measured at t = T\n\n", Re, double(u0));
    std::printf("  %5s %11s %11s %13s %9s\n", "N", "tau", "steps (T)", "L2 error", "order");
    std::printf("  %s\n", std::string(56, '-').c_str());

    std::FILE* f = open_out("A_taylor_green", "tgv2d", lat, op);
    if (f) std::fprintf(f, "# N  tau  steps  L2error   Re=%.0f u0=%.3f\n", Re, double(u0));

    double prev = 0; Index prevN = 0;
    for (Index N : Ns) {
      const double xi = 2.0 * M_PI / double(N);
      const Real nu = Real(double(u0) * double(N) / Re);
      const std::size_t T = std::size_t(1.0 / (2.0 * xi * xi * double(nu)));
      double err = NAN;

      dispatch(lat, op, [&](auto coll) {
        using Coll = decltype(coll);
        using LL   = typename Coll::Lattice;
        Domain d(N, N, 1, true, true, true);
        coll.omega = Coll::omega_from_viscosity(nu);
        FluidSolver<LL, EsotericPull<LL>, Coll> s(d, coll);

        const Real xic = Real(xi), u0c = u0;
        s.initialize_field(KOKKOS_LAMBDA(Index n) {
          Index px, py, pz; d.coords(n, px, py, pz);
          const Real x = Real(px - d.hx), y = Real(py - d.hy);
          FlowState st;
          st.rho = Real(1) - Real(0.75) * u0c * u0c *
                   (Kokkos::cos(Real(2) * xic * x) + Kokkos::cos(Real(2) * xic * y));
          st.ux = -u0c * Kokkos::cos(xic * x) * Kokkos::sin(xic * y);
          st.uy =  u0c * Kokkos::sin(xic * x) * Kokkos::cos(xic * y);
          st.uz = Real(0);
          return st;
        });

        for (std::size_t t = 0; t < T; ++t) s.step();
        s.compute_macroscopic();
        auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        const double decay = std::exp(-1.0);          // t = T
        double num = 0, den = 0;
        for (Index y = 0; y < N; ++y)
          for (Index x = 0; x < N; ++x) {
            const Index n = d.id(x, y);
            const double ax = -double(u0) * std::cos(xi * x) * std::sin(xi * y) * decay;
            const double ay =  double(u0) * std::sin(xi * x) * std::cos(xi * y) * decay;
            const double dx = double(hx(n)) - ax, dy = double(hy(n)) - ay;
            num += dx * dx + dy * dy;
            den += ax * ax + ay * ay;
          }
        err = std::sqrt(num / den);
      });

      const double ord = prevN ? std::log(prev / err) / std::log(double(N) / double(prevN)) : NAN;
      if (prevN) std::printf("  %5d %11.6f %11zu %13.5e %9.3f\n",
                             int(N), 3.0 * double(nu) + 0.5, T, err, ord);
      else       std::printf("  %5d %11.6f %11zu %13.5e %9s\n",
                             int(N), 3.0 * double(nu) + 0.5, T, err, "--");
      std::fflush(stdout);
      if (f) { std::fprintf(f, "%d %.6f %zu %.8e\n", int(N), 3.0*double(nu)+0.5, T, err); std::fflush(f); }
      prev = err; prevN = N;
    }
    if (f) std::fclose(f);
  }
  Kokkos::finalize();
  return 0;
}
