//==============================================================================
//  Orszag-Tang vortex at Re ~ 628, Pr_m = 1, against Table 1 of
//
//    A. De Rosis, E. Leveque, R. Chahine, "Advanced lattice Boltzmann scheme for
//    high-Reynolds-number magneto-hydrodynamic flows", J. Turbulence 19(6), 2018.
//
//  Reference values there are the L-infinity norms of the electric current
//  j = curl b and the vorticity zeta = curl u, at t = 0.5 s and t = 1 s, taken
//  from a high-resolution pseudo-spectral simulation.
//
//  Setup exactly as in the paper: L = 2 pi m, u0 = b0 = 2 (physical), rho = 1,
//  N = 1024, dt = 5e-5 s. That gives u0_lat = 2 dt/dx = 1.6297e-2 and
//  Ma = 2.82e-2, matching the paper's quoted ~1.6e-2 and ~3e-2. The Reynolds
//  number is the paper's lattice definition Re = u0 N / nu.
//
//  Coarser grids use ACOUSTIC scaling (dt ~ dx) so u0_lat and the Mach number
//  stay fixed, which is what the paper does when it refines.
//
//  CAVEATS on the comparison, stated up front:
//    * -op bgk  runs the Dellar baseline (the paper's ref [13]);
//      -op cm   runs the paper's own hybrid central-moment fluid operator,
//               Equations (7)-(13), with BGK for the magnetic field.
//    * The paper computes j locally from the distributions. Here j and zeta come
//      from second-order central differences, so a small difference in the peak
//      of an L-infinity norm is expected from the operator alone.
//==============================================================================
#include "Mhd.hpp"
#include "collision/MhdCentralMoments.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <type_traits>

using namespace lbm;
using namespace lbm::mhd;

namespace {
constexpr double L_PHYS = 2.0 * PI;
constexpr double U_PHYS = 2.0;
constexpr double DT_REF = 5.0e-5;      // paper's time step at N = 1024
constexpr int    N_REF  = 1024;

// L-infinity norms of curl u and curl b, in LATTICE units, plus the peak Mach
// number and a NaN flag -- the two things a blow-up shows up in first.
struct Peaks { double j_max, z_max, ma_max; bool finite; };

//------------------------------------------------------------------------------
// What Orszag-Tang actually tests, which the Table 1 peaks do not.
//
// There is no closed-form solution here, so the peaks are a comparison against
// somebody else's numbers and nothing more. The properties this case is the only
// one in the suite able to test are divergence preservation and the energy
// budget -- and divergence specifically, because in the resistive-decay and
// Alfven-wave cases B_x depends only on y and B_y only on x, so div B is
// STRUCTURALLY zero there and those cases report round-off whatever the scheme
// does. Here the nonlinear dynamics makes every component depend on every
// coordinate, so the number means something.
//
// max |div B| is normalised by k|B|, the field's own gradient scale, which is
// the only normalisation that stays meaningful as the grid refines. It is not
// preserved to machine precision in this scheme and must not be reported as if
// it were: it sits at truncation level and converges at about order 1.65.
//------------------------------------------------------------------------------
struct Budget { double div_b, e_kin, e_mag; };

template <class FS, class MS>
Budget budget(FS& fl, MS& mag, double k) {
  const Domain& d = fl.domain();
  fl.compute_macroscopic();
  mag.compute_field();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  auto w = [&](Index v, Index n) { return ((v % n) + n) % n; };
  double worst = 0, scale = 0, ek = 0, em = 0;
  for (Index y = 0; y < d.ny; ++y)
    for (Index x = 0; x < d.nx; ++x) {
      const Index n = d.id(x, y);
      const Index xp = w(x + 1, d.nx), xm = w(x - 1, d.nx);
      const Index yp = w(y + 1, d.ny), ym = w(y - 1, d.ny);
      const double dv =
          0.5 * (double(bx(d.id(xp, y))) - double(bx(d.id(xm, y)))) +
          0.5 * (double(by(d.id(x, yp))) - double(by(d.id(x, ym))));
      worst = std::max(worst, std::abs(dv));
      scale = std::max(scale, std::hypot(double(bx(n)), double(by(n))));
      ek += 0.5 * double(hr(n)) * (double(ux(n)) * double(ux(n)) +
                                   double(uy(n)) * double(uy(n)));
      em += 0.5 * (double(bx(n)) * double(bx(n)) + double(by(n)) * double(by(n)));
    }
  const double cells = double(d.nx) * double(d.ny);
  return {(scale > 0) ? worst / (scale * k) : worst, ek / cells, em / cells};
}

template <class FS, class MS>
Peaks peaks(FS& fl, MS& mag) {
  const Domain& d = fl.domain();
  fl.compute_macroscopic();
  mag.compute_field();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  auto w = [&](Index v, Index n) { return ((v % n) + n) % n; };
  double jm = 0, zm = 0, sm = 0;
  bool finite = true;
  for (Index y = 0; y < d.ny; ++y)
    for (Index x = 0; x < d.nx; ++x) {
      const double sx = double(ux(d.id(x, y))), sy = double(uy(d.id(x, y)));
      if (!std::isfinite(sx) || !std::isfinite(sy)) finite = false;
      sm = std::max(sm, std::hypot(sx, sy));
      const Index xp = w(x + 1, d.nx), xm = w(x - 1, d.nx);
      const Index yp = w(y + 1, d.ny), ym = w(y - 1, d.ny);
      const double j = 0.5 * (double(by(d.id(xp, y))) - double(by(d.id(xm, y))))
                     - 0.5 * (double(bx(d.id(x, yp))) - double(bx(d.id(x, ym))));
      const double z = 0.5 * (double(uy(d.id(xp, y))) - double(uy(d.id(xm, y))))
                     - 0.5 * (double(ux(d.id(x, yp))) - double(ux(d.id(x, ym))));
      if (!std::isfinite(j) || !std::isfinite(z)) finite = false;
      jm = std::max(jm, std::abs(j));
      zm = std::max(zm, std::abs(z));
    }
  return {jm, zm, sm * std::sqrt(3.0), finite};
}

// Dump the vorticity and current fields as raw float32, row-major, for the
// figure generator. Values are converted to physical units (divided by dt).
template <class FS, class MS>
void dump_fields(FS& fl, MS& mag, double dt, const std::string& tag) {
  const Domain& d = fl.domain();
  fl.compute_macroscopic();
  mag.compute_field();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  auto w = [&](Index v, Index n) { return ((v % n) + n) % n; };
  std::vector<float> jf(std::size_t(d.nx * d.ny)), zf(std::size_t(d.nx * d.ny));
  for (Index y = 0; y < d.ny; ++y)
    for (Index x = 0; x < d.nx; ++x) {
      const Index xp = w(x + 1, d.nx), xm = w(x - 1, d.nx);
      const Index yp = w(y + 1, d.ny), ym = w(y - 1, d.ny);
      const double j = 0.5 * (double(by(d.id(xp, y))) - double(by(d.id(xm, y))))
                     - 0.5 * (double(bx(d.id(x, yp))) - double(bx(d.id(x, ym))));
      const double z = 0.5 * (double(uy(d.id(xp, y))) - double(uy(d.id(xm, y))))
                     - 0.5 * (double(ux(d.id(x, yp))) - double(ux(d.id(x, ym))));
      jf[std::size_t(y * d.nx + x)] = float(j / dt);
      zf[std::size_t(y * d.nx + x)] = float(z / dt);
    }
  auto put = [&](const char* what, const std::vector<float>& v) {
    const std::string fn = "ot_" + std::string(what) + "_" + tag + ".bin";
    std::ofstream o(fn, std::ios::binary);
    const int nx = int(d.nx), ny = int(d.ny);
    o.write(reinterpret_cast<const char*>(&nx), sizeof nx);
    o.write(reinterpret_cast<const char*>(&ny), sizeof ny);
    o.write(reinterpret_cast<const char*>(v.data()),
            std::streamsize(v.size() * sizeof(float)));
    std::printf("  wrote %s (%dx%d)\n", fn.c_str(), nx, ny);
  };
  put("j", jf);
  put("zeta", zf);
}

}  // namespace

//------------------------------------------------------------------------------
// Stability run. Marches to t_max, recording the current maximum as the paper's
// Figures 1 and 3 do, and stops at the first sign of blow-up: a non-finite field
// or a Mach number past 0.5, by which point the low-Mach regime the scheme is
// built on has been abandoned regardless of what happens next.
//------------------------------------------------------------------------------
template <class FluidColl, class ML, class Setup>
void stability(Index N, double Re, double t_max, const char* opname, Setup setup) {
  const double dx = L_PHYS / double(N);
  const double dt = DT_REF * double(N_REF) / double(N);
  const double u0 = U_PHYS * dt / dx;
  const double nu = u0 * double(N) / Re;
  const std::size_t n_max = std::size_t(t_max / dt);
  const std::size_t probe = std::size_t(0.02 / dt);

  std::printf("  operator = %-38s N = %d   Re = %.0f\n", opname, int(N), Re);
  std::printf("  nu = eta = %.6e   tau = %.6f   t_max = %.2f s (%zu steps)\n\n",
              nu, 3.0 * nu + 0.5, t_max, n_max);

  using FL = typename FluidColl::Lattice;
  Domain d(N, N, 1, true, true, true);
  MagneticBGK<ML> mc;
  mc.omega = MagneticBGK<ML>::omega_from_resistivity(Real(nu));
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);
  FluidColl fc;
  setup(fc, Real(nu));
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fc);

  const double k = 2.0 * PI / double(N);
  const Real kk = Real(k), U = Real(u0), B = Real(u0);
  mag.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    Kokkos::Array<Real, 3> b;
    b[0] = -B * Kokkos::sin(kk * Real(y));
    b[1] =  B * Kokkos::sin(Real(2) * kk * Real(x));
    b[2] = Real(0);
    return b;
  });
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    FlowState s;
    s.rho = Real(1);
    s.ux = -U * Kokkos::sin(kk * Real(y));
    s.uy =  U * Kokkos::sin(kk * Real(x));
    s.uz = Real(0);
    return s;
  });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  std::printf("%-10s %-14s %-14s %-10s\n", "t (s)", "j_max", "zeta_max", "Ma_max");
  std::printf("%s\n", std::string(52, '-').c_str());
  double blow_t = -1.0;
  for (std::size_t t = 0; t < n_max; ++t) {
    mag.compute_field(); fl.step(true); mag.step(true);
    if ((t + 1) % probe == 0) {
      const Peaks p = peaks(fl, mag);
      const double tt = double(t + 1) * dt;
      std::printf("%-10.3f %-14.4f %-14.4f %-10.4f\n",
                  tt, p.j_max / dt, p.z_max / dt, p.ma_max);
      std::fflush(stdout);
      if (!p.finite || p.ma_max > 0.5) { blow_t = tt; break; }
    }
  }
  if (blow_t > 0)
    std::printf("\n  BLOW-UP at t = %.3f s\n\n", blow_t);
  else
    std::printf("\n  STABLE through t = %.2f s\n\n", t_max);
}

template <class FluidColl, class ML, class Setup>
void run(Index N, double Re, const char* opname, Setup setup, bool dump = false) {
  const double dx   = L_PHYS / double(N);
  const double dt   = DT_REF * double(N_REF) / double(N);   // acoustic scaling
  const double u0   = U_PHYS * dt / dx;                     // = 1.6297e-2
  const double nu   = u0 * double(N) / Re;                  // paper's Re definition
  const std::size_t n_half = std::size_t(0.5 / dt);
  const std::size_t n_full = std::size_t(1.0 / dt);

  std::printf("  operator = %s   N = %d   Re = %.0f   Pr_m = 1\n", opname, int(N), Re);
  std::printf("  dx = %.6e m   dt = %.3e s\n", dx, dt);
  std::printf("  u0_lat = %.6e   Ma = %.4f\n", u0, u0 * std::sqrt(3.0));
  std::printf("  nu = eta = %.6e   tau = %.5f\n", nu, 3.0 * nu + 0.5);
  std::printf("  steps: t=0.5s -> %zu, t=1s -> %zu\n\n", n_half, n_full);

  using FL = typename FluidColl::Lattice;
  Domain d(N, N, 1, true, true, true);
  MagneticBGK<ML> mc;
  mc.omega = MagneticBGK<ML>::omega_from_resistivity(Real(nu));
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);
  FluidColl fc;
  setup(fc, Real(nu));
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fc);

  const double k = 2.0 * PI / double(N);
  const Real kk = Real(k), U = Real(u0), B = Real(u0);   // b0 = u0 in Alfven units
  mag.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    Kokkos::Array<Real, 3> b;
    b[0] = -B * Kokkos::sin(kk * Real(y));
    b[1] =  B * Kokkos::sin(Real(2) * kk * Real(x));
    b[2] = Real(0);
    return b;
  });
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    FlowState s;
    s.rho = Real(1);
    s.ux = -U * Kokkos::sin(kk * Real(y));
    s.uy =  U * Kokkos::sin(kk * Real(x));
    s.uz = Real(0);
    return s;
  });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  const double ref_j[2] = {18.24, 46.59}, ref_z[2] = {6.758, 14.20};
  const double pap_j[2] = {18.24, 46.65}, pap_z[2] = {6.756, 14.18};
  double got_j[2] = {0, 0}, got_z[2] = {0, 0};
  Budget bud[3];
  bud[0] = budget(fl, mag, k);                 // t = 0

  for (std::size_t t = 0; t < n_full; ++t) {
    mag.compute_field(); fl.step(true); mag.step(true);
    if (t + 1 == n_half || t + 1 == n_full) {
      const Peaks p = peaks(fl, mag);
      const int s = (t + 1 == n_half) ? 0 : 1;
      got_j[s] = p.j_max / dt;                 // lattice -> physical (1/s)
      got_z[s] = p.z_max / dt;
      bud[s + 1] = budget(fl, mag, k);
      if (dump) dump_fields(fl, mag, dt, s ? "t1" : "t05");
    }
  }

  std::printf("%-10s %-6s %-12s %-12s %-12s %-10s %-10s\n",
              "quantity", "t(s)", "spectral", "paper LB", "this run",
              "err vs sp", "err vs LB");
  std::printf("%s\n", std::string(78, '-').c_str());
  for (int s = 0; s < 2; ++s)
    std::printf("%-10s %-6.1f %-12.2f %-12.2f %-12.2f %-+10.2f %-+10.2f\n",
                "j_max", s ? 1.0 : 0.5, ref_j[s], pap_j[s], got_j[s],
                100.0 * (got_j[s] / ref_j[s] - 1.0),
                100.0 * (got_j[s] / pap_j[s] - 1.0));
  for (int s = 0; s < 2; ++s)
    std::printf("%-10s %-6.1f %-12.3f %-12.3f %-12.3f %-+10.2f %-+10.2f\n",
                "zeta_max", s ? 1.0 : 0.5, ref_z[s], pap_z[s], got_z[s],
                100.0 * (got_z[s] / ref_z[s] - 1.0),
                100.0 * (got_z[s] / pap_z[s] - 1.0));
  std::printf("\nerrors are percent. spectral = ref [13] in the paper's Table 1.\n");

  // The properties this case is actually the only one able to test.
  const double e0 = bud[0].e_kin + bud[0].e_mag;
  std::printf("\n%-10s %-13s %-13s %-13s %-13s\n",
              "t (s)", "max|div B|", "E_kin", "E_mag", "E_tot/E_tot(0)");
  std::printf("%s\n", std::string(66, '-').c_str());
  for (int s = 0; s < 3; ++s)
    std::printf("%-10.1f %-13.3e %-13.6e %-13.6e %-13.6f\n",
                s == 0 ? 0.0 : (s == 1 ? 0.5 : 1.0), bud[s].div_b,
                bud[s].e_kin, bud[s].e_mag, (bud[s].e_kin + bud[s].e_mag) / e0);
  std::printf("\nmax|div B| is normalised by k|B| and is NOT preserved to machine\n"
              "precision -- it sits at truncation level, order about 1.65. The wave\n"
              "cases in this suite report round-off for it and that is meaningless:\n"
              "there div B is structurally zero whatever the scheme does.\n");
}

//------------------------------------------------------------------------------
// The whole lattice/operator dispatch, parameterised on the equilibrium order so
// both can be built: HO = true uses the highest order each lattice admits (D3Q27
// product form, sixth order; D3Q19 fourth order; D2Q9 product form, fourth),
// HO = false the second-order truncation the published results were produced
// with. Selected at run time by -eq2.
//------------------------------------------------------------------------------
template <class L, bool HO>
using EqOf = std::conditional_t<HO, HighOrderEquilibrium<L>, SecondOrderEquilibrium<L>>;

template <bool HO, class BgkSetup, class CmSetup>
void dispatch(Index N, double Re, double tmax, const std::string& op,
              const std::string& lat, const std::string& maglat, double wbulk,
              bool dump, BgkSetup bgk_setup, CmSetup cm_setup) {
    // The Orszag-Tang vortex is a two-dimensional problem. Running it on a 3D
  // lattice with a single cell in z is a genuine reduction test: with nz = 1
  // and periodic z the wrap sends the z-neighbour back to the node itself, so
  // the out-of-plane populations stream in place and the answer must match the
  // native 2D lattice. The magnetic field follows the fluid's dimensionality
  // (D2Q5 beside D2Q9, D3Q7 beside D3Q19/D3Q27).
  using F9  = MhdBGK<D2Q9,  EqOf<D2Q9, HO>,  ShiftedPopulations>;
  using F19 = MhdBGK<D3Q19, EqOf<D3Q19, HO>, ShiftedPopulations>;
  using F27 = MhdBGK<D3Q27, EqOf<D3Q27, HO>, ShiftedPopulations>;

  using CM9  = MhdCentralMoments<D2Q9, HO>;
  using CM27 = MhdCentralMoments<D3Q27, HO>;

  if (tmax > 0) {
    if (op == "cm" && lat == "d3q27")
                             stability<CM27, D3Q7>(N, Re, tmax,
                               "hybrid central moments, D3Q27 + D3Q7 (nz = 1)", cm_setup);
    else if (op == "cm")     stability<CM9, D2Q5>(N, Re, tmax,
                               "hybrid central moments (Eqs. 7-13)", cm_setup);
    else if (lat == "d3q19") {
        // -maglat lets the magnetic lattice be held fixed across fluid lattices,
        // which is the only way to separate the fluid lattice's effect from the
        // magnetic one (D3Q7 has cs2 = 1/4, D2Q5 has 1/3).
        if (maglat == "d2q5") stability<F19, D2Q5>(N, Re, tmax, "BGK, D3Q19 + D2Q5", bgk_setup);
        else                  stability<F19, D3Q7>(N, Re, tmax, "BGK, D3Q19 + D3Q7", bgk_setup);
      } else if (lat == "d3q27") {
        if (maglat == "d2q5") stability<F27, D2Q5>(N, Re, tmax, "BGK, D3Q27 + D2Q5", bgk_setup);
        else                  stability<F27, D3Q7>(N, Re, tmax, "BGK, D3Q27 + D3Q7", bgk_setup);
      }
    else                     stability<F9,  D2Q5>(N, Re, tmax, "BGK, D2Q9 + D2Q5", bgk_setup);
  } else if (op == "cm" && lat == "d3q27") {
    // The 3D extension the paper prescribes for the velocity field: D3Q27 for
    // the fluid, D3Q7 for the magnetic field. On the Orszag-Tang vortex, with
    // nz = 1, it must land on the 2D scheme's answer.
    std::printf("  omega_3 (bulk) = %.3f\n", wbulk);
    if (maglat == "d2q5")
      run<CM27, D2Q5>(N, Re, "hybrid CM, D3Q27 fluid + D2Q5 magnetic (nz = 1)", cm_setup, dump);
    else
      run<CM27, D3Q7>(N, Re, "hybrid CM, D3Q27 fluid + D3Q7 magnetic (nz = 1)", cm_setup, dump);
  } else if (op == "cm") {
    std::printf("  omega_3 (bulk) = %.3f\n", wbulk);
    run<CM9, D2Q5>(N, Re, "hybrid central moments (Eqs. 7-13)", cm_setup, dump);
  } else if (lat == "d3q19") {
    // b_z is identically zero for this flow, so a 2D magnetic lattice is a
    // legitimate pairing and isolates the magnetic lattice's contribution.
    if (maglat == "d2q5")
      run<F19, D2Q5>(N, Re, "BGK, D3Q19 fluid + D2Q5 magnetic (nz = 1)", bgk_setup, dump);
    else
      run<F19, D3Q7>(N, Re, "BGK, D3Q19 fluid + D3Q7 magnetic (nz = 1)", bgk_setup, dump);
  } else if (lat == "d3q27") {
    if (maglat == "d2q5")
      run<F27, D2Q5>(N, Re, "BGK, D3Q27 fluid + D2Q5 magnetic (nz = 1)", bgk_setup, dump);
    else
      run<F27, D3Q7>(N, Re, "BGK, D3Q27 fluid + D3Q7 magnetic (nz = 1)", bgk_setup, dump);
  } else {
    run<F9, D2Q5>(N, Re, "BGK, D2Q9 fluid + D2Q5 magnetic", bgk_setup, dump);
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index N = 512;
    double Re = 628.0;
    std::string op = "cm";
    double wbulk = 1.0;
    double tmax = -1.0;          // > 0 selects the stability run
    bool dump = false;           // write field snapshots for the figures
    std::string lat = "d2q9";    // fluid lattice; the 3D ones run with nz = 1
    std::string maglat = "";     // magnetic lattice override, to isolate its effect
    bool eq2 = false;            // -eq2: second-order equilibrium (published form)
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-n" && i + 1 < argc)  N = std::atoi(argv[++i]);
      if (a == "-re" && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-op" && i + 1 < argc) op = argv[++i];
      if (a == "-wb" && i + 1 < argc) wbulk = std::atof(argv[++i]);
      if (a == "-tmax" && i + 1 < argc) tmax = std::atof(argv[++i]);
      if (a == "-dump") dump = true;
      if (a == "-lat" && i + 1 < argc) lat = argv[++i];
      if (a == "-maglat" && i + 1 < argc) maglat = argv[++i];
      if (a == "-eq2") eq2 = true;
    }
    std::printf("Orszag-Tang vortex vs De Rosis, Leveque & Chahine (2018), Table 1\n");
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("equilibrium: %s\n\n",
                eq2 ? "second order (as published)"
                    : "highest order the lattice admits");

    auto bgk_setup = [](auto& c, Real nu) {
      c.omega = std::decay_t<decltype(c)>::omega_from_viscosity(nu);
    };
    auto cm_setup = [wbulk](auto& c, Real nu) {
      c.omega = std::decay_t<decltype(c)>::omega_from_viscosity(nu);
      c.omega_bulk = Real(wbulk);
    };
    if (eq2) dispatch<false>(N, Re, tmax, op, lat, maglat, wbulk, dump,
                             bgk_setup, cm_setup);
    else     dispatch<true> (N, Re, tmax, op, lat, maglat, wbulk, dump,
                             bgk_setup, cm_setup);
  }
  Kokkos::finalize();
  return 0;
}
