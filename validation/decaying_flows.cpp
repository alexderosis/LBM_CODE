//==============================================================================
//  Convergence and consistency of the decaying flows (see DecayingFlows.hpp).
//==============================================================================
#include "DecayingFlows.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
static double order_of(double e_coarse, int n_coarse, double e_fine, int n_fine) {
  return std::log(e_coarse / e_fine) / std::log(double(n_fine) / double(n_coarse));
}

namespace {
double g_worst_mass = 0;

struct Study {
  double nu_ord[2] = {0, 0};   // orders across the successive resolution pairs
  double l2_ord[2] = {0, 0};
  double nu_err_fine = 0, l2_fine = 0;
  int    pairs = 0;
};

// Run one case over a resolution ladder and fit the order across each pair.
template <class Fn>
Study convergence(const char* lattice, Fn run, const std::vector<int>& Ns) {
  Study st;
  std::vector<double> nu, l2;
  for (int N : Ns) {
    const Result r = run(N);
    g_worst_mass = std::max(g_worst_mass, r.mass_drift);
    std::printf("%-8s %-5d %-14.9f %-11.2e %-11.3e %-8zu\n",
                lattice, N, r.nu_eff, r.nu_err, r.l2_u, r.steps);
    nu.push_back(r.nu_err); l2.push_back(r.l2_u);
  }
  st.pairs = int(Ns.size()) - 1;
  for (int i = 0; i < st.pairs && i < 2; ++i) {
    st.nu_ord[i] = order_of(nu[i], Ns[i], nu[i + 1], Ns[i + 1]);
    st.l2_ord[i] = order_of(l2[i], Ns[i], l2[i + 1], Ns[i + 1]);
  }
  st.nu_err_fine = nu.back(); st.l2_fine = l2.back();
  std::printf("         order:");
  for (int i = 0; i < st.pairs && i < 2; ++i)
    std::printf("  %d->%d  nu %.2f  L2 %.2f  ", Ns[i], Ns[i + 1], st.nu_ord[i], st.l2_ord[i]);
  std::printf("\n");
  return st;
}

bool orders_ok(const Study& s, double tol) {
  for (int i = 0; i < s.pairs && i < 2; ++i)
    if (!(s.nu_ord[i] > tol && s.l2_ord[i] > tol)) return false;
  return true;
}

void header() {
  std::printf("%-8s %-5s %-14s %-11s %-11s %-8s\n",
              "lattice", "N", "nu_eff", "nu err", "L2(u)", "steps");
  std::printf("%s\n", std::string(62, '-').c_str());
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const Real tau = Real(0.8);
    const double nu_input = double((tau - Real(0.5)) * cs2<D2Q9, Real>());
    const double frac = 0.5;

    std::printf("Decaying flows with exact NS solutions   (nu = %.9f, tau = %.2f)\n",
                nu_input, double(tau));
    std::printf("backend %s   precision %s   streaming EsotericPull   storage shifted\n",
                ExecSpace::name(), precision_name());
    std::printf("\nDiffusive scaling U ~ 1/N, so both the Mach-number and spatial errors are\n"
                "second order. The effective viscosity does NOT equal nu at finite resolution --\n"
                "a sinusoid decays at the rate set by the discrete Laplacian, which differs at\n"
                "O(k^2) -- so what is asserted is that the error converges away at order 2.\n\n");

    bool all_orders_ok = true;
    const double ord_tol = 1.8;

    //--------------------------------------------------------------------------
    std::printf("TAYLOR-GREEN, D2Q9  (nonlinear; z-independent, so 2D)\n");
    header();
    const std::vector<int> lad2 = {16, 32, 64};
    const Study tg_bgk = convergence("BGK", [&](int N) {
      return taylor_green<ops::Bgk<D2Q9>, D2Q9, EsotericPull>(N, 1, tau, Real(0.64 / N), frac, ops::plain); }, lad2);
    const Study tg_trt = convergence("TRT", [&](int N) {
      return taylor_green<ops::Trt<D2Q9>, D2Q9, EsotericPull>(N, 1, tau, Real(0.64 / N), frac, ops::trt); }, lad2);
    const Study tg_mrt = convergence("MRT", [&](int N) {
      return taylor_green<ops::Mrt<D2Q9>, D2Q9, EsotericPull>(N, 1, tau, Real(0.64 / N), frac, ops::plain); }, lad2);
    const Study tg_cm  = convergence("CM", [&](int N) {
      return taylor_green<ops::Cm<D2Q9>,  D2Q9, EsotericPull>(N, 1, tau, Real(0.64 / N), frac, ops::plain); }, lad2);

    //--------------------------------------------------------------------------
    std::printf("\nABC / BELTRAMI, D3Q27  (nonlinear and genuinely 3D)\n");
    header();
    const std::vector<int> lad3 = {16, 32, 48};   // 48 not 64: the ladder is the cost
    const Study abc_bgk = convergence("BGK", [&](int N) {
      return abc_flow<ops::Bgk<D3Q27>, D3Q27, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad3);
    const Study abc_trt = convergence("TRT", [&](int N) {
      return abc_flow<ops::Trt<D3Q27>, D3Q27, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::trt); }, lad3);
    const Study abc_mrt = convergence("MRT", [&](int N) {
      return abc_flow<ops::Mrt<D3Q27>, D3Q27, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad3);
    const Study abc_cm  = convergence("CM", [&](int N) {
      return abc_flow<ops::Cm<D3Q27>,  D3Q27, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad3);

    std::printf("\nABC / BELTRAMI, D3Q19  (moment operators via the 19-monomial basis)\n");
    header();
    const Study a19_bgk = convergence("BGK", [&](int N) {
      return abc_flow<ops::Bgk<D3Q19>, D3Q19, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad3);
    const Study a19_mrt = convergence("MRT", [&](int N) {
      return abc_flow<ops::Mrt<D3Q19>, D3Q19, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad3);
    const Study a19_cm  = convergence("CM", [&](int N) {
      return abc_flow<ops::Cm<D3Q19>,  D3Q19, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad3);

    //--------------------------------------------------------------------------
    std::printf("\nDIAGONAL SHEAR WAVE  (linear, off-axis -- probes lattice isotropy)\n");
    header();
    const Study sw19 = convergence("D3Q19 BGK", [&](int N) {
      return shear_wave<ops::Bgk<D3Q19>, D3Q19, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad2);
    const Study sw27 = convergence("D3Q27 BGK", [&](int N) {
      return shear_wave<ops::Bgk<D3Q27>, D3Q27, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad2);
    const Study sw19cm = convergence("D3Q19 CM", [&](int N) {
      return shear_wave<ops::Cm<D3Q19>,  D3Q19, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad2);
    const Study sw27cm = convergence("D3Q27 CM", [&](int N) {
      return shear_wave<ops::Cm<D3Q27>,  D3Q27, EsotericPull>(N, tau, Real(0.32 / N), frac, ops::plain); }, lad2);
    std::printf("\n  off-diagonal viscosity error at the finest N, relative to D3Q19+BGK:\n");
    std::printf("    D3Q19 BGK 1.00x   D3Q19 CM %.2fx   D3Q27 BGK %.2fx   D3Q27 CM %.2fx\n",
                sw19.nu_err_fine / sw19cm.nu_err_fine,
                sw19.nu_err_fine / sw27.nu_err_fine,
                sw19.nu_err_fine / sw27cm.nu_err_fine);

    for (const Study* st : {&tg_bgk, &tg_trt, &tg_mrt, &tg_cm,
                            &abc_bgk, &abc_trt, &abc_mrt, &abc_cm,
                            &a19_bgk, &a19_mrt, &a19_cm,
                            &sw19, &sw27, &sw19cm, &sw27cm})
      all_orders_ok = all_orders_ok && orders_ok(*st, ord_tol);

    //--------------------------------------------------------------------------
    std::printf("\nCROSS-LATTICE CONSISTENCY  (Taylor-Green, N=32, BGK -- same 2D field)\n");
    const Real U32 = Real(0.64 / 32.0);
    const Result c9  = taylor_green<ops::Bgk<D2Q9>,  D2Q9,  EsotericPull>(32, 1, tau, U32, frac, ops::plain);
    const Result c19 = taylor_green<ops::Bgk<D3Q19>, D3Q19, EsotericPull>(32, 8, tau, U32, frac, ops::plain);
    const Result c27 = taylor_green<ops::Bgk<D3Q27>, D3Q27, EsotericPull>(32, 8, tau, U32, frac, ops::plain);
    std::printf("  D2Q9 %.12f   D3Q19 %.12f   D3Q27 %.12f\n", c9.nu_eff, c19.nu_eff, c27.nu_eff);
    const double spread = std::max(std::abs(c19.nu_eff - c9.nu_eff),
                                   std::abs(c27.nu_eff - c9.nu_eff)) / c9.nu_eff;
    std::printf("  spread %.2e\n", spread);

    //--------------------------------------------------------------------------
    std::printf("\nSTREAMING / STORAGE EQUIVALENCE  (ABC, D3Q27, N=32, CentralMoments)\n");
    const Real U3 = Real(0.32 / 32.0);
    const Result q[4] = {
      abc_flow<MomentCollision<D3Q27, NoForcing, RawPopulations,     true>, D3Q27, TwoLattice>  (32, tau, U3, frac, ops::plain),
      abc_flow<MomentCollision<D3Q27, NoForcing, ShiftedPopulations, true>, D3Q27, TwoLattice>  (32, tau, U3, frac, ops::plain),
      abc_flow<MomentCollision<D3Q27, NoForcing, RawPopulations,     true>, D3Q27, EsotericPull>(32, tau, U3, frac, ops::plain),
      abc_flow<MomentCollision<D3Q27, NoForcing, ShiftedPopulations, true>, D3Q27, EsotericPull>(32, tau, U3, frac, ops::plain),
    };
    const char* qn[4] = {"TwoLattice/raw", "TwoLattice/shifted",
                         "EsotericPull/raw", "EsotericPull/shifted"};
    for (int i = 0; i < 4; ++i)
      std::printf("  %-22s nu_eff %.12f   L2(u) %.4e\n", qn[i], q[i].nu_eff, q[i].l2_u);
    const bool same_scheme = (q[0].nu_eff == q[2].nu_eff) && (q[1].nu_eff == q[3].nu_eff);
    double store_spread = 0;
    for (int i = 0; i < 4; ++i)
      store_spread = std::max(store_spread, std::abs(q[i].nu_eff - q[0].nu_eff) / q[0].nu_eff);

    //--------------------------------------------------------------------------
    const double mass_tol  = sizeof(Real) == 4 ? 1e-4 : 1e-12;
    const double cross_tol = sizeof(Real) == 4 ? 1e-4 : 1e-10;
    const bool pass_cross = spread < cross_tol;
    const bool pass_iso   = sw27.nu_err_fine < sw19.nu_err_fine;
    const bool pass_mass  = g_worst_mass < mass_tol;
    const bool pass_equiv = same_scheme && store_spread < cross_tol;

    std::printf("\nacceptance:\n");
    std::printf("  every fitted order > %.1f  (15 studies x 2 quantities x 2 pairs)  %s\n",
                ord_tol, all_orders_ok ? "PASS" : "FAIL");
    std::printf("  D2Q9/D3Q19/D3Q27 agree on a 2D field   spread %.1e        %s\n",
                spread, pass_cross ? "PASS" : "FAIL");
    std::printf("  D3Q27 beats D3Q19 off-axis (isotropy)                     %s\n",
                pass_iso ? "PASS" : "FAIL");
    std::printf("  streaming/storage equivalence          spread %.1e        %s\n",
                store_spread, pass_equiv ? "PASS" : "FAIL");
    std::printf("  mass drift  %.2e  < %.1e                            %s\n",
                g_worst_mass, mass_tol, pass_mass ? "PASS" : "FAIL");
    if (!(all_orders_ok && pass_cross && pass_iso && pass_mass && pass_equiv)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
