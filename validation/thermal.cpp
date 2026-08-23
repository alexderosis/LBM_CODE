//==============================================================================
//  Scalar transport convergence, plus one coupled natural-convection case.
//  See Thermal.hpp for the cases and natural_convection.cpp for the full sweep.
//==============================================================================
#include "Thermal.hpp"

#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

using namespace lbm;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::printf("Scalar transport   (D = cs^2 (1/omega - 1/2); cs2 = 1/3 on D2Q5, 1/4 on D3Q7)\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    const Real tau = Real(0.8);
    const std::vector<int> Ns = {16, 32, 64};

    // Diffusive scaling for the advected cases too: the run length grows as N^2,
    // so holding U fixed would grow the advected DISTANCE with N and with it the
    // accumulated phase error. Keeping the Peclet number U N / D fixed -- i.e.
    // U ~ 1/N -- is what makes the comparison a convergence study rather than a
    // measurement of how far the wave travelled.
    auto ladder = [&](const char* nm, auto run) {
      double e[3];
      std::printf("%-22s", nm);
      for (int i = 0; i < 3; ++i) {
        const Decay r = run(Ns[i]);
        e[i] = r.l2;
        std::printf(" %-11.3e", r.l2);
      }
      const double ord = std::log(e[1] / e[2]) / std::log(2.0);
      std::printf("  %6.2f\n", ord);
      return ord;
    };

    std::printf("%-22s %-11s %-11s %-11s  %6s\n", "case", "N=16", "N=32", "N=64", "order");
    std::printf("%s\n", std::string(66, '-').c_str());
    const double o1 = ladder("D2Q5 diffusion", [&](int N) {
      return sinusoid<D2Q5>(N, tau, Real(0.01), Real(0)); });
    const double o2 = ladder("D3Q7 diffusion", [&](int N) {
      return sinusoid<D3Q7>(N, tau, Real(0.01), Real(0)); });
    const double o3 = ladder("D2Q5 advect-diffuse", [&](int N) {
      return sinusoid<D2Q5>(N, tau, Real(0.01), Real(0.8 / N)); });
    const double o4 = ladder("D3Q7 advect-diffuse", [&](int N) {
      return sinusoid<D3Q7>(N, tau, Real(0.01), Real(0.8 / N)); });

    std::printf("\n%-22s %-14s %-13s %-13s\n", "case (N=64)", "D_eff", "rel err", "phase err");
    std::printf("%s\n", std::string(66, '-').c_str());
    const Decay a = sinusoid<D2Q5>(64, tau, Real(0.01), Real(0));
    const Decay b = sinusoid<D3Q7>(64, tau, Real(0.01), Real(0));
    const Decay c = sinusoid<D2Q5>(64, tau, Real(0.01), Real(0.8 / 64));
    const Decay e = sinusoid<D3Q7>(64, tau, Real(0.01), Real(0.8 / 64));
    const char* nm[4] = {"D2Q5 diffusion", "D3Q7 diffusion",
                         "D2Q5 advect-diffuse", "D3Q7 advect-diffuse"};
    const Decay* rs[4] = {&a, &b, &c, &e};
    double worst_d = 0, worst_ph = 0;
    for (int i = 0; i < 4; ++i) {
      std::printf("%-22s %-14.9f %-13.2e %-13.2e\n", nm[i], rs[i]->d_eff,
                  rs[i]->d_err, rs[i]->phase_err);
      worst_d = std::max(worst_d, rs[i]->d_err);
      worst_ph = std::max(worst_ph, rs[i]->phase_err);
    }

    const double ord_tol = 1.8;
    const bool pass_ord = o1 > ord_tol && o2 > ord_tol && o3 > ord_tol && o4 > ord_tol;
    const bool pass_ph  = worst_ph < 2e-3;
    std::printf("\nacceptance:\n");
    std::printf("  all four orders > %.1f  (%.2f %.2f %.2f %.2f)          %s\n",
                ord_tol, o1, o2, o3, o4, pass_ord ? "PASS" : "FAIL");
    std::printf("  advected phase error < 2e-3 rad   worst %.2e     %s\n",
                worst_ph, pass_ph ? "PASS" : "FAIL");
    if (!(pass_ord && pass_ph)) status = 1;

    //--------------------------------------------------------------------------
    // One coupled case, small enough to belong in the regression suite. The full
    // Rayleigh sweep against the benchmark lives in `natural_convection`, which
    // takes minutes and is an analysis run, not a test.
    std::printf("\n\nCoupled check: natural convection, Ra = 1e4, N = 48\n");
    auto trt = [](auto& c, Real w) {
      using T = std::decay_t<decltype(c)>;
      c.omega_p = w; c.omega_m = T::omega_minus_for(w, T::magic_3_16);
    };
    using FT = TRT<D2Q9, SecondOrderEquilibrium<D2Q9>, BoussinesqGuo, ShiftedPopulations>;
    const Cavity cv = cavity<FT>(48, 1e4, 0.71, Real(0.05), trt);
    const double dev = std::abs(cv.nu_avg / 2.243 - 1.0);
    std::printf("  Nu = %.4f   benchmark 2.243   deviation %.2f%%   (%zu steps)\n",
                cv.nu_avg, 100.0 * dev, cv.steps);
    const bool pass_nu = dev < 0.02;
    std::printf("\n  coupled Nusselt within 2%% of de Vahl Davis (1983)        %s\n",
                pass_nu ? "PASS" : "FAIL");
    if (!pass_nu) status = 1;
  }
  Kokkos::finalize();
  return status;
}
