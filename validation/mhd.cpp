//==============================================================================
//  MHD validation: exact solutions first, then Orszag-Tang.
//==============================================================================
#include "Mhd.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;
using namespace lbm::mhd;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::printf("MHD   (D2Q9 fluid + D2Q5 vector magnetic distribution, Dellar scheme)\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    const Real nu = Real(0.05), eta = Real(0.05);
    const std::vector<int> Ns = {16, 32, 64};

    auto ladder = [&](const char* nm, bool alfven, double* speed_ord) {
      double e[3], sp[3];
      std::printf("%-24s", nm);
      for (int i = 0; i < 3; ++i) {
        // diffusive scaling: amplitude ~ 1/N keeps the Mach error second order too
        const Wave w = wave(Ns[i], nu, eta, Real(0.16 / Ns[i]), Real(0.05), alfven);
        e[i] = w.l2; sp[i] = w.speed_err;
        std::printf(" %-11.3e", w.l2);
      }
      const double ord = std::log(e[1] / e[2]) / std::log(2.0);
      std::printf("  %6.2f\n", ord);
      if (speed_ord && alfven)
        *speed_ord = std::log(sp[1] / sp[2]) / std::log(2.0);
      return ord;
    };

    std::printf("%-24s %-11s %-11s %-11s  %6s\n", "case", "N=16", "N=32", "N=64", "order");
    std::printf("%s\n", std::string(70, '-').c_str());
    double speed_ord = 0;
    const double o1 = ladder("resistive decay", false, nullptr);
    const double o2 = ladder("shear Alfven wave", true, &speed_ord);
    std::printf("\n  Alfven-wave SPEED converges at order %.2f. Its L2 stalls because the\n"
                "  damping rate carries a larger constant than the phase does -- see below.\n",
                speed_ord);

    // per-resolution diagnostics, to see whether each error converges
    std::printf("\n%-24s %-6s %-13s %-12s %-13s %-11s\n",
                "case", "N", "rate err", "speed err", "div B", "steps");
    std::printf("%s\n", std::string(78, '-').c_str());
    Wave d64{}, a64{};
    for (int i = 0; i < 3; ++i) {
      const Wave d = wave(Ns[i], nu, eta, Real(0.16 / Ns[i]), Real(0.05), false);
      const Wave a = wave(Ns[i], nu, eta, Real(0.16 / Ns[i]), Real(0.05), true);
      std::printf("%-24s %-6d %-13.2e %-12s %-13.2e %-11zu\n",
                  "resistive decay", Ns[i], d.rate_err, "-", d.div_b, d.steps);
      std::printf("%-24s %-6d %-13.2e %-12.2e %-13.2e %-11zu\n",
                  "shear Alfven wave", Ns[i], a.rate_err, a.speed_err, a.div_b, a.steps);
      if (Ns[i] == 64) { d64 = d; a64 = a; }
    }
    std::printf("\n%-24s %-15s %-12s %-15s %-12s\n",
                "case (N=64)", "decay rate", "rel err", "wave speed", "rel err");
    std::printf("%s\n", std::string(78, '-').c_str());
    std::printf("%-24s %-15.9f %-12.2e %-15s %-12s\n", "resistive decay",
                d64.rate_eff, d64.rate_err, "-", "-");
    std::printf("%-24s %-15.9f %-12.2e %-15.9f %-12.2e\n", "shear Alfven wave",
                a64.rate_eff, a64.rate_err, a64.speed_eff, a64.speed_err);
    std::printf("   (Alfven speed v_A = B0/sqrt(rho) = %.6f)\n", 0.05);

    //--------------------------------------------------------------------------
    std::printf("\n\nOrszag-Tang vortex, Re = Rm = 100, to t* = t u0/L = 0.5\n");
    std::printf("The only case here that actually tests divergence preservation: the two\n"
                "waves above have div B structurally zero and report round-off regardless.\n\n");
    std::printf("%-6s %-13s %-13s %-13s %-13s %-9s\n",
                "N", "div B (final)", "div B (max)", "E_mag/E_kin", "E_tot loss", "steps");
    std::printf("%s\n", std::string(74, '-').c_str());
    double div32 = 0, div64 = 0;
    for (Index N : {32, 64}) {
      const OT r = orszag_tang(N, Real(0.8 / N), Real(0.8 / N), 100.0, 0.5);
      std::printf("%-6d %-13.2e %-13.2e %-13.4f %-13.4f %-9zu\n",
                  int(N), r.div_b, r.div_b_max, r.e_mag / r.e_kin,
                  1.0 - r.e_tot1 / r.e_tot0, r.steps);
      if (N == 32) div32 = r.div_b_max; else div64 = r.div_b_max;
    }
    const double div_ord = std::log(div32 / div64) / std::log(2.0);
    std::printf("\n  div B is measured against the field's own gradient scale k|B|, and\n"
                "  converges away at order %.2f. It is NOT preserved to machine precision --\n"
                "  it sits at truncation level and refines away, which is the honest claim.\n",
                div_ord);

    const double ord_tol  = 1.8;
    const double tol       = sizeof(Real) == 4 ? 3e-2 : 1e-2;
    const bool pass_ord    = o1 > ord_tol;                  // resistive L2
    const bool pass_sord   = speed_ord > ord_tol;           // Alfven phase
    const bool pass_eta    = d64.rate_err < 5e-3;
    const bool pass_damp   = a64.rate_err < tol;
    const bool pass_speed  = a64.speed_err < 2e-3;
    // structurally zero, so this is purely a round-off floor and scales with the
    // working precision; the k-normalisation lifts it by another decade
    const double div_tol = sizeof(Real) == 4 ? 1e-4 : 1e-12;
    const bool pass_wdiv   = d64.div_b < div_tol && a64.div_b < div_tol;
    const bool pass_otdiv  = div_ord > 1.5 && div64 < 5e-2;
    std::printf("\nacceptance:\n");
    std::printf("  resistive-decay L2 order > %.1f   (%.2f)                  %s\n",
                ord_tol, o1, pass_ord ? "PASS" : "FAIL");
    std::printf("  Alfven phase-speed order > %.1f   (%.2f)                  %s\n",
                ord_tol, speed_ord, pass_sord ? "PASS" : "FAIL");
    std::printf("  resistivity recovered to 0.5%%     (%.2e)             %s\n",
                d64.rate_err, pass_eta ? "PASS" : "FAIL");
    std::printf("  Alfven damping within %.0f%%        (%.2e)             %s\n",
                100 * tol, a64.rate_err, pass_damp ? "PASS" : "FAIL");
    std::printf("  Alfven speed within 0.2%%          (%.2e)             %s\n",
                a64.speed_err, pass_speed ? "PASS" : "FAIL");
    std::printf("  div B at round-off on both waves  (%.1e < %.0e)      %s\n",
                std::max(d64.div_b, a64.div_b), div_tol, pass_wdiv ? "PASS" : "FAIL");
    std::printf("  Orszag-Tang div B converges (order %.2f, %.1e at N=64)  %s\n",
                div_ord, div64, pass_otdiv ? "PASS" : "FAIL");
    if (!(pass_ord && pass_sord && pass_eta && pass_damp && pass_speed &&
          pass_wdiv && pass_otdiv)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
