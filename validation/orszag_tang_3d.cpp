//==============================================================================
//  Orszag-Tang vortex in THREE dimensions.
//
//  De Rosis, Phys. Rev. E 95, 013310 (2017), Sec. III D, Eqs. (28)-(30) and
//  Fig. 6, with the initial condition taken from Mininni, Pouquet and
//  Montgomery (the paper's Ref. [38]).
//
//  Cubic periodic box of side 2 pi discretised by M^3 points, delta = 2 pi / M:
//
//     u(x,0) = v0 [ -2 sin(delta y),  2 sin(delta x),  0 ]
//     B(x,0) = 0.8 b0 [ -2 sin(2 delta y) + sin(delta z),
//                        2 sin(delta x)   + sin(delta z),
//                        sin(delta x)     + sin(delta y) ]
//
//  Both are solenoidal: every component depends only on the coordinates it is
//  not differentiated by, so each term of the divergence vanishes identically.
//
//  SIGN. Equation (28) as printed reads u = v0[2 sin(delta y), 2 sin(delta x), 0],
//  without the leading minus. Ref. [38] has -2 sin y in the first component, and
//  that is what is used here. Unlike the two-dimensional Taylor-Green case in
//  the same paper, the divergence does NOT discriminate -- both signs are
//  solenoidal -- so this rests on the cited source rather than on a consistency
//  check, and it is a genuine ambiguity rather than a proven typo.
//
//  TWO PARAMETERS THE PAPER DOES NOT PIN DOWN. It says only that "the values of
//  v0 and b0 lead to a Mach number, Ma ~ 0.034", and Ma depends on which speed
//  it is built on. Taken here on the PEAK initial speed, which for this field is
//  2 sqrt(2) v0, so v0 = Ma / (2 sqrt(2) sqrt(3)) = 6.94e-3. And b0 = v0, the
//  usual equipartition choice. Both are stated as assumptions, not readings.
//
//  Re = 100, on the lattice definition Re = v0 M / nu used elsewhere in this
//  suite, so nu = v0 M / Re. Pr_m = 1, so eta = nu.
//
//  WHAT IS AND IS NOT SCORED. Figure 6 is a plot against a high-resolution
//  pseudospectral run, and those reference values are not available numerically,
//  so no per-point comparison is possible here. Three things are checked instead:
//
//    * that J_max peaks at t ~ 1, which the paper states in words -- "the
//      current shows a maximum at about t = 1, which rapidly reduces as the
//      time increases";
//    * that refining M moves the curves toward a limit, which is the trend
//      Fig. 6 reports;
//    * the M-to-M discrepancy on the same L-infinity measure the paper uses in
//      Eq. (30), which it quotes as of order 1e-2 for the finest grid against
//      the spectral reference.
//==============================================================================
#include "Campaign.hpp"
#include "FieldDump.hpp"
#include "collision/MagneticBGK.hpp"
#include "collision/MhdCentralMoments.hpp"
#include "solver/MagneticSolver.hpp"

#include <cmath>

using namespace lbm;
using namespace campaign;

namespace {
constexpr double PI3 = 3.14159265358979323846;

struct Sample { double t, energy, jmax; bool finite; };

// |curl B|_inf and the volume-averaged kinetic energy, both in LATTICE units.
template <class FS, class MS>
Sample measure(FS& fl, MS& mag, Index M, double t) {
  const Domain& d = fl.domain();
  fl.compute_macroscopic();
  mag.compute_field();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto uz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uz());
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  auto bz = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bz());
  auto w = [&](Index v) { return ((v % M) + M) % M; };

  double e = 0, jm = 0; bool finite = true;
  for (Index z = 0; z < M; ++z)
    for (Index y = 0; y < M; ++y)
      for (Index x = 0; x < M; ++x) {
        const Index n = d.id(x, y, z);
        const double a = double(ux(n)), b = double(uy(n)), c = double(uz(n));
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) finite = false;
        e += a * a + b * b + c * c;
        const Index xp = d.id(w(x+1), y, z), xm = d.id(w(x-1), y, z);
        const Index yp = d.id(x, w(y+1), z), ym = d.id(x, w(y-1), z);
        const Index zp = d.id(x, y, w(z+1)), zm = d.id(x, y, w(z-1));
        const double jx = 0.5 * (double(bz(yp)) - double(bz(ym)))
                        - 0.5 * (double(by(zp)) - double(by(zm)));
        const double jy = 0.5 * (double(bx(zp)) - double(bx(zm)))
                        - 0.5 * (double(bz(xp)) - double(bz(xm)));
        const double jz = 0.5 * (double(by(xp)) - double(by(xm)))
                        - 0.5 * (double(bx(yp)) - double(bx(ym)));
        const double j = std::sqrt(jx*jx + jy*jy + jz*jz);
        if (!std::isfinite(j)) finite = false;
        jm = std::max(jm, j);
      }
  if (std::getenv("FIGDUMP") && t > 0) {
    // |curl B| on the mid-z plane, at the instants the figure shows
    const double want[2] = {1.0, 2.0};
    for (double tw : want)
      if (std::abs(t - tw) < 0.06) {
        using namespace lbm::figdump;
        const Index zc = M / 2;
        auto ww = [&](Index v) { return ((v % M) + M) % M; };
        char nmv[64];
        std::snprintf(nmv, sizeof nmv, "ot3d_vol_m%d_t%d.bin", int(M), int(tw + 0.5));
        scalar_volume(nmv, M, M, M, [&](Index x, Index y, Index z) {
          const Index xp = d.id(ww(x+1), y, z), xm = d.id(ww(x-1), y, z);
          const Index yp = d.id(x, ww(y+1), z), ym = d.id(x, ww(y-1), z);
          const Index zp = d.id(x, y, ww(z+1)), zm2 = d.id(x, y, ww(z-1));
          const double jx = 0.5*(double(bz(yp))-double(bz(ym))) - 0.5*(double(by(zp))-double(by(zm2)));
          const double jy = 0.5*(double(bx(zp))-double(bx(zm2))) - 0.5*(double(bz(xp))-double(bz(xm)));
          const double jz = 0.5*(double(by(xp))-double(by(xm))) - 0.5*(double(bx(yp))-double(bx(ym)));
          return std::sqrt(jx*jx + jy*jy + jz*jz);
        });
        char nmb[64];
        std::snprintf(nmb, sizeof nmb, "ot3d_j_m%d_t%d.bin", int(M), int(tw + 0.5));
        scalar_slice(nmb, M, M, [&](Index x, Index y) {
          const Index xp = d.id(ww(x+1), y, zc), xm = d.id(ww(x-1), y, zc);
          const Index yp = d.id(x, ww(y+1), zc), ym = d.id(x, ww(y-1), zc);
          const Index zp = d.id(x, y, ww(zc+1)), zm2 = d.id(x, y, ww(zc-1));
          const double jx = 0.5*(double(bz(yp))-double(bz(ym))) - 0.5*(double(by(zp))-double(by(zm2)));
          const double jy = 0.5*(double(bx(zp))-double(bx(zm2))) - 0.5*(double(bz(xp))-double(bz(xm)));
          const double jz = 0.5*(double(by(xp))-double(by(xm))) - 0.5*(double(bx(yp))-double(bx(ym)));
          return std::sqrt(jx*jx + jy*jy + jz*jz);
        });
      }
  }
  return {t, e / double(M) / double(M) / double(M), jm, finite};
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    double Re = 100.0, Ma = 0.034, tmax = 4.0;
    Index M = 64;
    int nsample = 60;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-m"    && i + 1 < argc) M = std::atoi(argv[++i]);
      if (a == "-re"   && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-ma"   && i + 1 < argc) Ma = std::atof(argv[++i]);
      if (a == "-tmax" && i + 1 < argc) tmax = std::atof(argv[++i]);
      if (a == "-ns"   && i + 1 < argc) nsample = std::atoi(argv[++i]);
    }

    // Ma is taken on the PEAK initial speed 2 sqrt(2) v0. See the header.
    const double v0 = Ma / (2.0 * std::sqrt(2.0) * std::sqrt(3.0));
    const double b0 = v0;                       // equipartition, assumed
    const Real   nu = Real(v0 * double(M) / Re);
    const double dx = 2.0 * PI3 / double(M);
    const double dt = v0 * dx;                  // u_phys amplitude = 1
    const std::size_t T = std::size_t(tmax / dt);

    std::printf("3D Orszag-Tang (PRE 2017 Fig. 6)   D3Q27 fluid + D3Q7 magnetic, central moments\n");
    std::printf("  M = %d   Re = %.0f   Ma = %.3f   v0 = b0 = %.5e\n", int(M), Re, Ma, v0);
    std::printf("  nu = eta = %.6e   tau = %.6f   t up to %.1f  (%zu steps, %d^3 nodes)\n\n",
                double(nu), 3.0 * double(nu) + 0.5, tmax, T, int(M));

    Domain d(M, M, M, true, true, true);

    MagneticBGK<D3Q7> mc;
    mc.omega = MagneticBGK<D3Q7>::omega_from_resistivity(nu);   // Pr_m = 1
    MagneticSolver<D3Q7, EsotericPull<D3Q7>, MagneticBGK<D3Q7>> mag(d, mc);

    MhdCentralMoments<D3Q27, true> fc;
    fc.omega = MhdCentralMoments<D3Q27, true>::omega_from_viscosity(nu);
    fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
    FluidSolver<D3Q27, EsotericPull<D3Q27>, MhdCentralMoments<D3Q27, true>> fl(d, fc);

    const Real dl = Real(2.0 * PI3 / double(M));
    const Real v0c = Real(v0), b0c = Real(0.8 * b0);
    fl.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real X = dl * Real(px - d.hx), Y = dl * Real(py - d.hy);
      FlowState st;
      st.rho = Real(1);
      st.ux = -Real(2) * v0c * Kokkos::sin(Y);
      st.uy =  Real(2) * v0c * Kokkos::sin(X);
      st.uz =  Real(0);
      return st;
    });
    mag.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real X = dl * Real(px - d.hx), Y = dl * Real(py - d.hy),
                 Z = dl * Real(pz - d.hz);
      Kokkos::Array<Real, 3> b;
      b[0] = b0c * (-Real(2) * Kokkos::sin(Real(2) * Y) + Kokkos::sin(Z));
      b[1] = b0c * ( Real(2) * Kokkos::sin(X)           + Kokkos::sin(Z));
      b[2] = b0c * ( Kokkos::sin(X) + Kokkos::sin(Y));
      return b;
    });
    mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

    std::FILE* f = open_out("G_orszag_tang_3d", "ot3d_m" + std::to_string(int(M)), "d3q27", "cm");
    if (f) std::fprintf(f, "# t  E(t)  E(t)/E(0)  Jmax(lattice)  Jmax(physical)"
                           "   M=%d Re=%.0f Ma=%.3f tau=%.6f\n",
                        int(M), Re, Ma, 3.0 * double(nu) + 0.5);

    const Sample s0 = measure(fl, mag, M, 0.0);
    std::printf("  %8s %14s %12s %14s\n", "t", "E(t)", "E/E(0)", "Jmax (phys)");

    // Sample times LOGARITHMICALLY, because Fig. 6 plots against a log time
    // axis: uniform sampling puts most points in the last decade, where the
    // curve is flat, and leaves the rise and the peak under-resolved. t = 0 is
    // kept for the normalisation and then dropped from the log plot.
    std::vector<std::size_t> when;
    when.push_back(0);
    {
      const double t0log = 0.05;
      for (int i = 0; i < nsample; ++i) {
        const double t = t0log * std::pow(tmax / t0log, double(i) / double(nsample - 1));
        const std::size_t k = std::size_t(t / dt);
        if (k > 0 && k <= T && k != when.back()) when.push_back(k);
      }
      if (when.back() != T) when.push_back(T);
    }
    std::size_t wi = 0;

    double jpeak = 0, tpeak = 0;
    for (std::size_t k = 0; k <= T; ++k) {
      if (wi < when.size() && k == when[wi]) {
        ++wi;
        const double t = double(k) * dt;
        const Sample s = measure(fl, mag, M, t);
        if (!s.finite) { std::printf("  DIVERGED at t = %.3f\n", t);
                         if (f) std::fprintf(f, "# DIVERGED\n"); break; }
        const double jp = s.jmax / dt;
        if (jp > jpeak) { jpeak = jp; tpeak = t; }
        std::printf("  %8.3f %14.6e %12.6f %14.4f\n", t, s.energy, s.energy / s0.energy, jp);
        if (f) { std::fprintf(f, "%.6f %.8e %.8e %.8e %.8e\n",
                              t, s.energy, s.energy / s0.energy, s.jmax, jp);
                 std::fflush(f); }
        std::fflush(stdout);
      }
      if (k < T) { mag.compute_field(); fl.step(true); mag.step(true); }
    }
    if (f) std::fclose(f);
    std::printf("\n  J_max peaks at t = %.3f  (paper: \"a maximum at about t = 1\")\n", tpeak);
  }
  Kokkos::finalize();
  return 0;
}
