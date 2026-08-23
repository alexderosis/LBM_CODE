//==============================================================================
//  Natural convection in a differentially heated square cavity, swept over
//  Rayleigh number and compared against de Vahl Davis (1983).
//
//  Analysis run, not a regression test: it takes minutes. `thermal` carries one
//  small coupled case for the suite.
//==============================================================================
#include "Thermal.hpp"

#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

using namespace lbm;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Natural convection, differentially heated cavity (Pr = 0.71)\n");
    std::printf("D2Q9 fluid + D2Q5 scalar, coupled both ways, EsotericPull on both\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());
    std::printf("%-8s %-5s %-6s %-11s %-12s %-9s %-9s %-10s %-9s\n",
                "Ra", "N", "op", "Nu (this)", "Nu (dVD83)", "dev", "u_max",
                "u_max ref", "steps");
    std::printf("%s\n", std::string(88, '-').c_str());

    auto plain = [](auto& c, Real w) { c.omega = w; };
    auto trt   = [](auto& c, Real w) {
      using T = std::decay_t<decltype(c)>;
      c.omega_p = w; c.omega_m = T::omega_minus_for(w, T::magic_3_16);
    };
    using FB = BGK<D2Q9, SecondOrderEquilibrium<D2Q9>, BoussinesqGuo, ShiftedPopulations>;
    using FT = TRT<D2Q9, SecondOrderEquilibrium<D2Q9>, BoussinesqGuo, ShiftedPopulations>;

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
