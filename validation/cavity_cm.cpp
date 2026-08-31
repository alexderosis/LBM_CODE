//==============================================================================
//  III.C -- Lid-driven cavity.
//
//  De Rosis & Coreixas (2020), Phys. Fluids 32, 117101, Sec. III C, against the
//  reference solution of Ghia, Ghia & Shin, J. Comput. Phys. 48, 387 (1982).
//
//  Square cavity, uniform rightward lid velocity, no-slip on the other three
//  sides, started from rest at rho = 1. Reported quantities are the horizontal
//  velocity along the vertical mid-section and the vertical velocity along the
//  horizontal mid-section, both normalised by the lid speed.
//
//  DEVIATIONS FROM THE PAPER, from the campaign settings:
//    * L = 129 rather than 201, for cost;
//    * u_lid = 0.02 rather than 0.01.
//  Walls use the regularised condition, as the paper states. The two upper
//  corners sit on a genuine velocity discontinuity and are given u = 0, the
//  side-wall value, so no spurious momentum enters at the singular point.
//
//  REFERENCE DATA PROVENANCE. All three tables have now been checked against an
//  independent digitisation of Ghia, Ghia & Shin (1982) and they match, entry
//  for entry, including the Re = 400 rows. The earlier note in this file saying
//  the Re = 400 table was unverified recall and that one entry was mis-copied
//  was WRONG on the second count: the transcription is right.
//
//  What is true is that the published table itself has an anomalous entry, and
//  the anomaly is real rather than ours:
//
//      v at Re = 400, x = 0.9063  reads  -0.23827.
//
//  Three independent reasons to distrust it:
//    * Ghia's own extremum table gives min v = -0.44993 at x = 0.8594 for
//      Re = 400. The tabulated neighbours are then -0.44993 (0.8594),
//      -0.23827 (0.9063), -0.22847 (0.9453): almost the entire recovery from
//      the minimum happens in one interval and then the profile goes flat.
//    * The same three stations at Re = 100 (-0.22445, -0.16914, -0.10313) and
//      Re = 1000 (-0.42665, -0.51550, -0.39188) are smooth. Only Re = 400 kinks,
//      and it kinks at exactly one point.
//    * Every other station here agrees with Ghia to 0.0071 or better at
//      Re = 400; this one disagrees by 0.1415, a factor of twenty.
//
//  The value is therefore left in the table -- it is what the paper prints, and
//  silently substituting a number nobody published would be worse -- but it is
//  scored separately. The run reports max |diff| both including and excluding it.
//
//  UPDATE 2026-08-31: the paper's own Table I was supplied directly, and all
//  three u-columns here (Re = 100, 400, 1000) plus the 17 y-stations match it
//  ENTRY FOR ENTRY. The transcription is confirmed against the primary source,
//  not merely against a second digitisation.
//
//  That table also carries a SECOND published anomaly, in a column this file
//  does not yet use:
//
//      u at Re = 3200, y = 0.4531  reads  -0.86636.
//
//  It cannot be right. The same station reads -0.10648 at Re = 1000 and
//  -0.07404 at Re = 5000, and every u entry in the whole table lies in
//  [-0.44, 1.0]. It is almost certainly -0.08636 with a shifted digit. If a
//  Re = 3200 column is ever added here, that is the entry to handle.
//
//  CONVERGENCE -- MEASURED, three criteria at Re = 100, N = 129, D3Q27/CM:
//
//    criterion                              steps    t/tau_d   max|du|  max|dv|
//    single node (centre), 1e-11           262,000    0.410     0.0049   0.0067
//    whole field, per STEP, 1e-5            20,391    0.032     0.0485   0.0414
//    whole field, per 500 steps, 1e-5       93,500    0.146     0.0048   0.0066
//
//  Two lessons, and they point in opposite directions.
//
//  A PER-STEP tolerance is not a convergence test. For a field relaxing as
//  u_inf + A exp(-t/T) the change per step is (remaining error)/T, so bounding
//  the per-step change by eps bounds the remaining error only by eps*T. Here
//  T = L^2/nu = 6.4e5 steps, so eps = 1e-5 permits a remaining error of 633% of
//  the velocity scale -- and the run duly stops at 3% of a diffusive time with
//  an error against Ghia ten times the converged one. It gets LOOSER as Re
//  rises, which is backwards. The run therefore reports the residual and the
//  implied remaining error e_hat = (r/probe)*T side by side; e_hat is the number
//  that means something.
//
//  But the WHOLE-FIELD measure is strictly better than the single node, and with
//  the same 1e-5 threshold taken over a 500-step interval it reproduces the
//  converged answer to the last digit in 2.8x FEWER steps than the single-node
//  test needed. A single node can sit still while boundary layers are still
//  growing (validation/poiseuille_inlet.cpp records that failure costing a
//  factor of 400 in error); the field cannot.
//
//  Recommended: -restol 1e-5 -probe 500. -minsteps forces a longer run.
//
//  IS 1e-5 TIGHT ENOUGH AT HIGHER Re? Checked, at Re = 1000, N = 129:
//
//    restol   steps     t/tau_d   e_hat     max|du|   max|dv|
//    1e-5     321,000   0.050     1.28e-1   0.0097    0.0104
//    1e-6     528,500   0.083     1.28e-2   0.0094    0.0100
//
//  A 10x tighter tolerance and a 1.65x longer run move the comparison by 0.0003
//  and 0.0004 -- about 4% of the discrepancy itself, and 0.04% of the lid speed.
//  So the remaining transient is NOT what separates this from Ghia at 129^2; the
//  grid is.
//
//  Note also that e_hat is a CONSERVATIVE bound, not an estimate. It uses the
//  diffusive time L^2/nu as the relaxation scale, and the cavity equilibrates
//  convectively rather than diffusively, so e_hat = 0.13 corresponded to an
//  actual movement of 0.0003. Read it as "no worse than", not "is".
//==============================================================================
#include "Campaign.hpp"
#include "FieldDump.hpp"
#include "boundary/Regularized.hpp"

using namespace lbm;
using namespace campaign;

static const double GY[17] = {0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                              0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                              0.9531, 0.9609, 0.9688, 0.9766, 1.0000};
static const double GX[17] = {0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                              0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                              0.9453, 0.9531, 0.9609, 0.9688, 1.0000};
// Re = 100 (verified)
static const double U100[17] = {0.00000, -0.03717, -0.04192, -0.04775, -0.06434,
                                -0.10150, -0.15662, -0.21090, -0.20581, -0.13641,
                                0.00332, 0.23151, 0.68717, 0.73722, 0.78871,
                                0.84123, 1.00000};
static const double V100[17] = {0.00000, 0.09233, 0.10091, 0.10890, 0.12317,
                                0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
                                -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
                                -0.05906, 0.00000};
// Re = 400. Verified against an independent digitisation; matches entry for
// entry. The v entry at x = 0.9063 (index 11 below) is anomalous in the
// PUBLISHED data -- see the provenance note in the header -- and is excluded
// from the headline score.
static const double U400[17] = {0.00000, -0.08186, -0.09266, -0.10338, -0.14612,
                                -0.24299, -0.32726, -0.17119, -0.11477, 0.02135,
                                0.16256, 0.29093, 0.55892, 0.61756, 0.68439,
                                0.75837, 1.00000};
static const double V400[17] = {0.00000, 0.18360, 0.19713, 0.20920, 0.22965,
                                0.28124, 0.30203, 0.30174, 0.05186, -0.38598,
                                -0.44993, -0.23827, -0.22847, -0.19254, -0.15663,
                                -0.12146, 0.00000};
// Index into GX/V400 of the anomalous published station, x = 0.9063.
static const int V400_ANOMALY = 11;
// Re = 1000 (verified)
static const double U1K[17] = {0.00000, -0.18109, -0.20196, -0.22220, -0.29730,
                               -0.38289, -0.27805, -0.10648, -0.06080, 0.05702,
                               0.18719, 0.33304, 0.46604, 0.51117, 0.57492,
                               0.65928, 1.00000};
static const double V1K[17] = {0.00000, 0.27485, 0.29012, 0.30353, 0.32627,
                               0.37095, 0.33075, 0.32235, 0.02526, -0.31966,
                               -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
                               -0.21388, 0.00000};

static double lerp_at(const std::vector<double>& v, double s) {
  const double p = s * double(v.size() - 1);
  const int i = int(p);
  if (i >= int(v.size()) - 1) return v.back();
  return v[i] + (p - double(i)) * (v[i + 1] - v[i]);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Index N = 129;
    double Re = 100.0;
    Real U = Real(0.02);
    std::string lat = "d2q9", op = "cm";
    std::size_t minsteps = 0;   // force at least this many steps before stopping
    std::size_t probe_n = 500;  // steps between residual probes
    double restol = 0.0;        // >0: use the WHOLE-FIELD residual instead of
                                // the single-node test
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-n"   && i + 1 < argc) N  = std::atoi(argv[++i]);
      if (a == "-re"  && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-u0"  && i + 1 < argc) U  = Real(std::atof(argv[++i]));
      if (a == "-lat" && i + 1 < argc) lat = argv[++i];
      if (a == "-op"  && i + 1 < argc) op  = argv[++i];
      if (a == "-minsteps" && i + 1 < argc) minsteps = std::strtoull(argv[++i], nullptr, 10);
      if (a == "-probe"  && i + 1 < argc) probe_n = std::strtoull(argv[++i], nullptr, 10);
      if (a == "-restol" && i + 1 < argc) restol = std::atof(argv[++i]);
    }
    const Real Lc = Real(N - 1);                 // walls sit ON the boundary nodes
    const Real nu = Real(double(U) * double(Lc) / Re);

    std::printf("III.C lid-driven cavity   lattice %s   operator %s\n", lat.c_str(), op.c_str());
    std::printf("  N = %d   L = %.0f   u_lid = %.3f   nu = %.6e   tau = %.6f\n\n",
                int(N), double(Lc), double(U), double(nu), 3.0 * double(nu) + 0.5);

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
        if (yt) return WS{NrmYp, U, Real(0), Real(0)};
        if (yb) return WS{NrmYm, Real(0), Real(0), Real(0)};
        if (xl) return WS{NrmXm, Real(0), Real(0), Real(0)};
        if (xr) return WS{NrmXp, Real(0), Real(0), Real(0)};
        return WS{};
      });
      s.initialize(Real(1));

      // March to steady state on the centreline velocity.
      // STOPPING. Two criteria, because they measure different things.
      //
      // The default watches ONE node, the cavity centre. -restol switches to the
      // relative L2 change of the WHOLE velocity field over `probe` steps, which
      // is strictly better: a single node can sit still while boundary layers
      // are still growing, and poiseuille_inlet.cpp records that exact failure.
      //
      // The tolerance must NOT be read as a per-step change. For a field
      // relaxing as u_inf + A exp(-t/T), the change per step is (remaining
      // error)/T, so a per-step bound eps only bounds the remaining error by
      // eps*T -- and T = L^2/nu is 6.4e5 steps at Re = 100 here, so eps = 1e-5
      // permits a remaining error of 640% of the velocity scale and fires
      // immediately. Worse, T grows with Re, so the bound loosens exactly where
      // the run needs to be longer. The residual is therefore reported over the
      // probe interval AND converted to an estimated remaining error,
      //     e_hat = (r / probe) * T,
      // which is the number that means something.
      const std::size_t probe = probe_n, cap = 40000000;
      const double tau_d = double(Lc) * double(Lc) / double(nu);
      Real prev = 0; std::size_t taken = 0;
      std::vector<double> pu, pv;
      double res = NAN, ehat = NAN;
      for (std::size_t t = 0; t < cap; t += probe) {
        for (std::size_t k = 0; k < probe; ++k) s.step();
        taken += probe;
        s.compute_macroscopic();
        auto h  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto h2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        const Real cur = h(d.id(N / 2, N / 2));
        if (!std::isfinite(double(cur))) { std::printf("  DIVERGED at step %zu\n", taken); return; }
        if (restol > 0.0) {
          const std::size_t np = std::size_t(N) * std::size_t(N);
          if (pu.empty()) { pu.assign(np, 0.0); pv.assign(np, 0.0); }
          double num = 0, den = 0;
          for (Index y = 0; y < N; ++y)
            for (Index x = 0; x < N; ++x) {
              const std::size_t k = std::size_t(y) * std::size_t(N) + std::size_t(x);
              const double a = double(h(d.id(x, y))), b = double(h2(d.id(x, y)));
              num += (a - pu[k]) * (a - pu[k]) + (b - pv[k]) * (b - pv[k]);
              den += a * a + b * b;
              pu[k] = a; pv[k] = b;
            }
          res  = std::sqrt(num / std::max(den, 1e-300));
          ehat = res / double(probe) * tau_d;
          if (t > 0 && taken >= minsteps && res < restol) break;
        } else {
          if (t > 0 && taken >= minsteps &&
              std::abs(double(cur - prev)) < 1e-11 * (std::abs(double(cur)) + 1e-12)) break;
        }
        prev = cur;
      }
      std::printf("  steady after %zu steps   (t/tau_d = %.3f)\n",
                  taken, double(taken) / tau_d);
      if (restol > 0.0)
        std::printf("  whole-field residual %.3e over %zu steps"
                    "   -> estimated remaining error %.2e\n", res, probe, ehat);

      s.compute_macroscopic();
      auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
      auto huy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
      if (std::getenv("FIGDUMP")) {
        using namespace lbm::figdump;
        const std::string t = "re" + std::to_string(int(Re));
        scalar_slice("cav_speed_" + t + ".bin", N, N, [&](Index x, Index y) {
          const double a = double(hux(d.id(x, y))), b = double(huy(d.id(x, y)));
          return std::sqrt(a * a + b * b) / double(U);
        });
        curl_z_slice("cav_zeta_" + t + ".bin", N, N,
                     [&](Index x, Index y) { return double(hux(d.id(x, y))); },
                     [&](Index x, Index y) { return double(huy(d.id(x, y))); });
      }
      std::vector<double> uc(N), vc(N);
      for (Index j = 0; j < N; ++j) uc[j] = double(hux(d.id(N / 2, j))) / double(U);
      for (Index i = 0; i < N; ++i) vc[i] = double(huy(d.id(i, N / 2))) / double(U);

      // The comparison table below holds only Ghia's 17 stations, which is the
      // right thing to SCORE against but makes a plotted profile look
      // piecewise-linear. Dump the full centrelines as well, so a figure can
      // show a smooth curve with Ghia's points on top of it.
      {
        const std::string fp = "results/C_cavity/cavity_re" + std::to_string(int(Re))
                             + "_" + lat + "_" + op + "_full.dat";
        if (std::FILE* g = std::fopen(fp.c_str(), "w")) {
          std::fprintf(g, "# s  u/U(x=0.5, y=s)  v/U(x=s, y=0.5)   N=%d Re=%.0f\n",
                       int(N), Re);
          for (Index k = 0; k < N; ++k)
            std::fprintf(g, "%.6f %.8f %.8f\n",
                         double(k) / double(N - 1), uc[k], vc[k]);
          std::fclose(g);
          std::printf("  -> %s\n", fp.c_str());
        }
      }

      const bool is400 = std::abs(Re - 400.0) < 1e-9;
      const bool have = (std::abs(Re - 100.0) < 1e-9) || is400 ||
                        (std::abs(Re - 1000.0) < 1e-9);
      const double* ru = (std::abs(Re - 1000.0) < 1e-9) ? U1K : (is400 ? U400 : U100);
      const double* rv = (std::abs(Re - 1000.0) < 1e-9) ? V1K : (is400 ? V400 : V100);

      std::FILE* f = open_out("C_cavity", "cavity_re" + std::to_string(int(Re)), lat, op);
      if (f) std::fprintf(f, "# y u/U  x v/U   (Ghia reference where available)"
                             "  N=%d Re=%.0f u_lid=%.3f\n", int(N), Re, double(U));
      double eu = 0, ev = 0, ev_all = 0; int worst = -1;
      for (int k = 0; k < 17; ++k) {
        const double su = lerp_at(uc, GY[k]), sv = lerp_at(vc, GX[k]);
        if (have) {
          eu = std::max(eu, std::abs(su - ru[k]));
          const double dv = std::abs(sv - rv[k]);
          if (dv > ev_all) { ev_all = dv; worst = k; }
          // The one station known to be bad in the published Re = 400 table is
          // kept in the file and reported, but not allowed to set the score.
          if (!(is400 && k == V400_ANOMALY)) ev = std::max(ev, dv);
        }
        if (f) std::fprintf(f, "%.4f %.6f %.4f %.6f %.5f %.5f\n",
                            GY[k], su, GX[k], sv, have ? ru[k] : NAN, have ? rv[k] : NAN);
      }
      if (f) std::fclose(f);
      if (have) {
        std::printf("  max |diff| vs Ghia:  u %.4f   v %.4f\n", eu, ev);
        if (is400)
          std::printf("  excluded x=%.4f (v=%.5f), the anomalous published entry:"
                      " |diff| there %.4f\n", GX[V400_ANOMALY], V400[V400_ANOMALY],
                      std::abs(lerp_at(vc, GX[V400_ANOMALY]) - V400[V400_ANOMALY]));
        else if (worst >= 0 && ev_all > 0)
          std::printf("  worst v station x = %.4f\n", GX[worst]);
      }
      else      std::printf("  no verified reference table for Re = %.0f; profiles written only\n", Re);
    });
    if (!ok) std::printf("unknown lattice/operator\n");
  }
  Kokkos::finalize();
  return 0;
}
