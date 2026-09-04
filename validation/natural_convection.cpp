//==============================================================================
//  Natural convection in a differentially heated square cavity, swept over
//  Rayleigh number and compared against de Vahl Davis (1983).
//
//  Analysis run, not a regression test: it takes minutes. `thermal` carries one
//  small coupled case for the suite.
//==============================================================================
#include "Thermal.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

using namespace lbm;

int main(int argc, char** argv) {
  bool conv = false;
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "-conv") conv = true;
  Kokkos::initialize(argc, argv);
  {
    std::printf("Natural convection, differentially heated cavity (Pr = 0.71)\n");
    std::printf("D2Q9 fluid + D2Q5 scalar, coupled both ways, EsotericPull on both\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());
    std::printf("%-8s %-5s %-6s %-11s %-12s %-9s %-9s %-10s %-9s\n",
                "Ra", "N", "op", "Nu (this)", "Nu (dVD83)", "dev", "u_max",
                "u_max ref", "steps");
    std::printf("%s\n", std::string(88, '-').c_str());

    auto plain0 = [](auto& c, Real w) { c.omega = w; };
    auto plain = plain0;
    auto trt   = [](auto& c, Real w) {
      using T = std::decay_t<decltype(c)>;
      c.omega_p = w; c.omega_m = T::omega_minus_for(w, T::magic_3_16);
    };
    using FB = BGK<D2Q9, SecondOrderEquilibrium<D2Q9>, BoussinesqGuo, ShiftedPopulations>;
    using FT = TRT<D2Q9, SecondOrderEquilibrium<D2Q9>, BoussinesqGuo, ShiftedPopulations>;

    //=======================================================================
    // GRID CONVERGENCE TOWARD THE TABLE, which the sweep below does not show.
    //
    // The sweep runs ONE resolution per Rayleigh number, so it reports a
    // deviation from de Vahl Davis without saying whether that deviation is
    // the scheme's error or the grid's. Those are different claims, and only
    // the second one shrinks. This mode fixes Ra and refines N instead, so the
    // deviation can be watched falling -- which is the difference between
    // "agrees with the reference" and "converges to the reference".
    //
    // BGK only: the operator comparison is the sweep's job, and doubling the
    // runs here to repeat it would cost more than it tells.
    //=======================================================================
    if (conv) {
      std::printf("  --- Nu vs resolution at fixed Ra, against de Vahl Davis ---\n");
      std::printf("  %-8s %-5s %-11s %-12s %-9s %-10s %-9s\n",
                  "Ra", "N", "Nu (this)", "Nu (dVD83)", "dev %", "order", "steps");
      std::printf("  %s\n", std::string(72, '-').c_str());
      const double refnu[3] = {1.118, 2.243, 4.519};
      const double rav[3]   = {1e3, 1e4, 1e5};
      for (int j = 0; j < 3; ++j) {
        double prev_dev = 0; Index prev_N = 0;
        for (Index N : {Index(32), Index(48), Index(64), Index(96)}) {
          const Cavity c = cavity<FB>(N, rav[j], 0.71, Real(0.05), plain0);
          const double dev = 100.0 * (c.nu_avg / refnu[j] - 1.0);
          char ord[16] = "   -";
          if (prev_N && std::abs(dev) > 0 && std::abs(prev_dev) > 0)
            std::snprintf(ord, sizeof ord, "%6.2f",
                          std::log(std::abs(prev_dev / dev))
                          / std::log(double(N) / double(prev_N)));
          std::printf("  %-8.0e %-5d %-11.4f %-12.3f %-+9.2f %-10s %-9zu\n",
                      rav[j], int(N), c.nu_avg, refnu[j], dev, ord, c.steps);
          std::fflush(stdout);
          prev_dev = dev; prev_N = N;
        }
      }
      std::printf("\n  `order` is the local slope of |dev| against N between"
                  " consecutive rows.\n"
                  "  reference values are de Vahl Davis (1983), Pr = 0.71.\n");
      // AN ORDER IS ONLY MEASURABLE WHERE THE ERROR EXCEEDS THE REFERENCE'S OWN
      // PRECISION, and here it often does not. de Vahl Davis prints Nu to four
      // figures -- 1.118, 2.243, 4.519 -- so the table itself is quantised at
      // half a last digit: +/-0.045%, +/-0.022% and +/-0.011% of its own value.
      //
      // Measured against that: at Ra = 1e3 the deviation is already -0.12% on
      // the COARSEST grid and reaches -0.02%, inside the table's precision, so
      // the falling `order` there (1.96 -> 1.16) is the table's rounding and not
      // the scheme's convergence. At Ra = 1e4 the deviation CROSSES ZERO between
      // N = 64 and 96 (-0.06 -> +0.00 -> +0.05), and no order can be fitted to a
      // quantity that changes sign -- the 9.34 and -5.83 in that column are
      // arithmetic on noise and are printed only because suppressing them would
      // hide the crossing. Ra = 1e5 is the one row that measures anything: the
      // error starts at -1.00%, an order of magnitude above the reference's
      // precision, and falls monotonically at order 2.4 to 3.4.
      //
      // So the honest reading is two different claims. At Ra = 1e3 and 1e4 the
      // code agrees with de Vahl Davis to within the table's printed precision.
      // At Ra = 1e5 it CONVERGES to it, at roughly second order or better. Only
      // the second is a convergence result.
      std::printf("  NOTE the table is quantised at +/-0.045, 0.022, 0.011%% of its\n"
                  "  own value, so only the Ra = 1e5 row has room to show an order;\n"
                  "  the others are already inside the reference's precision.\n\n");
      Kokkos::finalize();
      return 0;
    }

    struct Ref { double Ra, nu, umax; Index N; };
    const std::vector<Ref> refs = {
      {1e3, 1.118,  3.649, 64},
      {1e4, 2.243, 16.178, 64},
      {1e5, 4.519, 34.73,  96},
    };
    for (const auto& r : refs) {
      const Cavity cb = cavity<FB>(r.N, r.Ra, 0.71, Real(0.05), plain);
      const Cavity ct = cavity<FT>(r.N, r.Ra, 0.71, Real(0.05), trt);
      std::printf("%-8.0e %-5d %-6s %-11.4f %-12.3f %-+9.2f %-9.2f %-10.3f %-9zu\n",
                  r.Ra, int(r.N), "BGK", cb.nu_avg, r.nu,
                  100.0 * (cb.nu_avg / r.nu - 1.0), cb.umax, r.umax, cb.steps);
      std::printf("%-8s %-5s %-6s %-11.4f %-12.3f %-+9.2f %-9.2f %-10s %-9zu\n",
                  "", "", "TRT", ct.nu_avg, r.nu,
                  100.0 * (ct.nu_avg / r.nu - 1.0), ct.umax, "", ct.steps);
    }
    std::printf("\ndeviations are percent; reference values are de Vahl Davis (1983).\n");
  }
  Kokkos::finalize();
  return 0;
}
