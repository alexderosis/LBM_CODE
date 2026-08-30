//==============================================================================
//  A standing gravity wave: omega^2 = g k tanh(k h).
//
//  THE FREE SURFACE'S OWN VALIDATION CASE, and the reason it is this one rather
//  than a dam break. Dam break is what the method is FOR, but its reference is
//  experimental (Martin & Moyce) and comparing against a digitised curve tests
//  the digitisation as much as the solver. The linear dispersion relation for
//  surface gravity waves is exact, closed form, has no fitted constant, and --
//  the part that matters -- it is a statement about precisely the two conditions
//  a free-surface scheme exists to impose:
//
//    the KINEMATIC condition, that the surface moves with the fluid, which here
//      is the mass exchange and nothing else;
//    the DYNAMIC condition, that the normal stress at the surface equals the gas
//      pressure, which here is the population reconstruction and nothing else.
//
//  Get either wrong and the period moves. Nothing else in this file can hide it:
//  there is no viscosity in the dispersion relation, no surface tension, and the
//  amplitude is small enough that the nonlinear correction is below the
//  measurement.
//
//  TWO DEPTHS, ON PURPOSE. At k h = 3.1 the tanh is 0.9963 and the relation is
//  indistinguishable from the deep-water limit omega^2 = g k; at k h = 0.79 it is
//  0.657, and a scheme that had the surface condition right but the depth
//  coupling wrong would pass the first and fail the second. Testing one depth
//  tests a number; testing two tests the relation.
//
//  HOW THE PERIOD IS MEASURED. The surface height of column x is the column's
//  total fill, sum_y epsilon(x, y) -- exact in this representation, since that
//  IS how much liquid the column holds. Its first Fourier mode
//
//      a(t) = (2/Lx) sum_x h(x, t) cos(k x)
//
//  is the standing wave's amplitude, and the period comes from its zero
//  crossings by linear interpolation. Zero crossings rather than a peak fit
//  because the wave DAMPS -- viscously, at 2 nu k^2 -- and a damped sinusoid's
//  peaks drift while its zeros do not.
//
//  THE TWO ERRORS, AND WHICH KNOB MOVES EACH. Neither is a defect in the scheme;
//  both are the distance between a finite mesh at finite viscosity and an
//  inviscid continuum relation, and each was identified by making it move.
//
//    DEEP is limited by the free surface's own discretisation. The surface is
//    resolved to about a cell, so a wave of amplitude A carries a relative
//    position error of order 1/A. Refining at FIXED k h -- the depths below are
//    fractions of the wavelength for exactly this reason -- took it from 6.15%
//    at Lx = 64, A = 1.5 to 1.77% at Lx = 128, A = 3.
//
//    SHALLOW is limited by the viscous boundary layer on the floor. At h = 16
//    cells the layer is sqrt(nu T) ~ 4 cells, a quarter of the depth, and the
//    relation being compared against has no viscosity in it at all. Dropping nu
//    fivefold took it from 8.44% to 4.23%, monotonically.
//
//  Neither knob touches the other's number, which is what says these are two
//  separate effects rather than one tolerance being chased.
//
//  WHAT THIS DOES NOT TEST. Surface tension, which the solver does not model.
//  Large amplitude: at A/lambda = 0.023 here the Stokes correction to the period
//  is order (kA)^2 = 2e-2 of a per cent, deliberately below what is measured.
//  And conversion: a wave this small moves the surface across about three cells,
//  so the interface converts constantly but never violently. A dam break is the
//  case that stresses conversion, and it is a demonstrator because it has no
//  closed form.
//==============================================================================
#include "core/Types.hpp"
#include "solver/FreeSurfaceSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using L  = D2Q9;
using FS = FreeSurfaceSolver<L>;

struct Result {
  double T_meas = 0, T_exact = 0, err = 0;
  double kh = 0, decay = 0, amp0 = 0;
  int crossings = 0;
  bool ok = false;
};

static Result run(Index Lx, Index Ly, double h, double A, double g, double nu,
                  std::size_t nsteps, bool verbose) {
  const double k = 2.0 * M_PI / double(Lx);
  Result R;
  R.kh = k * h;
  R.T_exact = 2.0 * M_PI / std::sqrt(g * k * std::tanh(k * h));

  Domain d(Lx, Ly, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);
  FS s(d);
  s.coll.omega = FS::omega_from_viscosity(Real(nu));
  s.set_gravity(Real(0), Real(-g));

  const Index Lyi = Ly;
  s.set_geometry([&](Index, Index y, Index) -> FsCell {
    return (y == 0 || y == Lyi - 1) ? FsSolid : FsGas;
  });

  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const Real hr = Real(h), Ar = Real(A), kr = Real(k);
  const Real gr = Real(g);
  constexpr Real ics = inv_cs2<L, Real>();
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real x = Real(px - hx), y = Real(py - hy);
    const Real ys = hr + Ar * Kokkos::cos(kr * x);       // the free surface
    // Fraction of this cell that is below the surface.
    Real e = ys - (y - Real(0.5));
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    // Hydrostatic, so the run does not start by ringing.
    const Real dz = ys - y;
    const Real r = Real(1) + (dz > Real(0) ? gr * dz * ics : Real(0));
    return typename FS::Seed{e, r};
  });

  const double m0 = double(s.total_mass());

  // Amplitude history of the k = 2 pi / Lx mode.
  std::vector<double> amp;
  amp.reserve(nsteps + 1);
  auto sample = [&]() {
    auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
    double a = 0;
    for (Index x = 0; x < Lx; ++x) {
      double col = 0;
      for (Index y = 1; y < Ly - 1; ++y) {
        const Index n = d.id(x, y);
        const std::uint8_t f = hf(n);
        if (f == FsFluid) col += 1.0;
        else if (f == FsInterface) col += double(he(n));
      }
      a += col * std::cos(k * double(x));
    }
    return 2.0 * a / double(Lx);
  };

  amp.push_back(sample());
  R.amp0 = amp[0];
  const bool trace = std::getenv("GW_TRACE") != nullptr;
  bool bad = false;
  if (trace) std::printf("    a(0) = %+.5e\n", R.amp0);
  for (std::size_t t = 1; t <= nsteps; ++t) {
    s.step();
    const double a = sample();
    if (trace && (t % (nsteps / 20 ? nsteps / 20 : 1) == 0)) {
      const auto c = s.census();
      std::printf("    t %6zu  a %+12.5e   massdrift %+.3e   g/i/f %d/%d/%d\n",
                  t, a, double(s.total_mass()) / m0 - 1.0,
                  int(c.gas), int(c.interface_), int(c.fluid));
    }
    if (!std::isfinite(a)) { bad = true; break; }
    amp.push_back(a);
  }

  // Zero crossings, linearly interpolated. Half a period apart.
  std::vector<double> zc;
  for (std::size_t t = 1; t < amp.size(); ++t)
    if ((amp[t - 1] > 0) != (amp[t] > 0)) {
      const double f = amp[t - 1] / (amp[t - 1] - amp[t]);
      zc.push_back(double(t - 1) + f);
    }
  R.crossings = int(zc.size());
  if (zc.size() >= 3) {
    // Mean half-period over the whole record, from first to last crossing --
    // more robust than averaging the individual gaps.
    R.T_meas = 2.0 * (zc.back() - zc.front()) / double(zc.size() - 1);
    R.err = std::fabs(R.T_meas / R.T_exact - 1.0);
  }
  // Viscous decay, for information: the exact rate is 2 nu k^2.
  if (amp.size() > 1 && std::fabs(amp[0]) > 0)
    R.decay = -std::log(std::fabs(amp.back() / amp[0])) / double(amp.size() - 1);
  R.ok = !bad && zc.size() >= 3
       && std::fabs(double(s.total_mass()) / m0 - 1.0) < 1e-3;

  if (verbose)
    std::printf("%-7.2f %-9.1f %-11.1f %-11.1f %-9.2f %-7d %-11.3e %-11.3e\n",
                R.kh, h, R.T_exact, R.T_meas, 100.0 * R.err, R.crossings,
                R.decay, 2.0 * nu * k * k);
  return R;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    Index Lx = 128, Ly = 128;
    double A = 3.0, g = 1e-4, nu = 1e-3;
    double periods = 4.0;
    for (int i = 1; i < argc; ++i) {
      auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-lx")) { if (i+1<argc) Lx = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-ly")) { if (i+1<argc) Ly = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-a"))  num(A);
      else if (!std::strcmp(argv[i], "-g"))  num(g);
      else if (!std::strcmp(argv[i], "-nu")) num(nu);
      else if (!std::strcmp(argv[i], "-periods")) num(periods);
    }

    std::printf("Standing gravity wave   %s free surface, central moments\n", L::name);
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("%dx%d   A = %.2f   g = %.1e   nu = %.1e   omega = %.4f\n",
                int(Lx), int(Ly), A, g, nu,
                double(FS::omega_from_viscosity(Real(nu))));
    std::printf("exact:  omega^2 = g k tanh(k h),  k = 2 pi / Lx = %.5f\n\n",
                2.0 * M_PI / double(Lx));

    std::printf("%-7s %-9s %-11s %-11s %-9s %-7s %-11s %-11s\n",
                "k h", "depth", "T exact", "T measured", "err (%)", "zeros",
                "decay/step", "2 nu k^2");
    std::printf("%s\n", std::string(84, '-').c_str());

    const double k = 2.0 * M_PI / double(Lx);
    // Depths as FRACTIONS of the wavelength, so that k h -- and therefore the
    // whole dispersion relation -- is fixed while the mesh is refined. Depths in
    // absolute cells would change the physics and the resolution together and
    // could not separate them.
    const double depths[2] = {0.5 * double(Lx), 0.125 * double(Lx)};
    Result res[2];
    for (int j = 0; j < 2; ++j) {
      const double h = depths[j];
      const double T = 2.0 * M_PI / std::sqrt(g * k * std::tanh(k * h));
      res[j] = run(Lx, Ly, h, A, g, nu, std::size_t(periods * T), true);
    }

    const bool finite = res[0].ok && res[1].ok;
    const double worst = std::max(res[0].err, res[1].err);
    // The two depths differ by a factor 1.23 in period through tanh(k h) alone.
    // Requiring both within 5% is therefore a test of the tanh, not just of a
    // constant: a scheme that got the depth coupling wrong could not pass both.
    const bool p_deep    = finite && res[0].err < 0.05;
    const bool p_shallow = finite && res[1].err < 0.05;
    const double ratio_meas  = res[1].T_meas  / res[0].T_meas;
    const double ratio_exact = res[1].T_exact / res[0].T_exact;
    const bool p_ratio = finite
                       && std::fabs(ratio_meas / ratio_exact - 1.0) < 0.05;

    std::printf("\nacceptance:\n");
    std::printf("  deep    (k h = %.2f) period within 5%%    %.2f%%      %s\n",
                res[0].kh, 100.0 * res[0].err, p_deep ? "PASS" : "FAIL");
    std::printf("  shallow (k h = %.2f) period within 5%%    %.2f%%      %s\n",
                res[1].kh, 100.0 * res[1].err, p_shallow ? "PASS" : "FAIL");
    std::printf("  the tanh itself: T_shallow / T_deep      %.4f vs %.4f   %s\n",
                ratio_meas, ratio_exact, p_ratio ? "PASS" : "FAIL");
    std::printf("\n  The two depths are the same wave on the same mesh with the same\n"
                "  gravity; only tanh(k h) separates them, by a factor %.3f. Passing\n"
                "  both is a test of the dispersion RELATION rather than of one\n"
                "  period, and the third row states that directly.\n", ratio_exact);
    std::printf("  Worst period error %.2f%%.\n", 100.0 * worst);
    if (!(p_deep && p_shallow && p_ratio)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
