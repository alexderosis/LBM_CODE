//==============================================================================
//  GALILEAN INVARIANCE -- the test that justifies central moments.
//
//  The Navier-Stokes equations do not care what uniform velocity the observer
//  moves at: superimpose a mean flow U0 on any solution and the physics is
//  unchanged. A lattice Boltzmann scheme is not automatically Galilean
//  invariant, because the discrete velocity set picks out a preferred frame; the
//  standard BGK operator relaxes RAW moments, whose equilibria carry O(u^3)
//  errors, so its effective viscosity drifts as the mean flow grows.
//
//  Relaxing CENTRAL moments -- moments taken about the local fluid velocity --
//  is precisely the fix: the collision no longer knows which frame it is in.
//
//  Test: run the ABC/Beltrami flow (nonlinear, 3D, exact) with a superimposed
//  uniform velocity U0, and measure the effective viscosity from the decay of
//  ||u - U0||. Adding U0 only translates the solution, so nu_eff MUST be
//  independent of U0. What each operator actually does about that is the result.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/TRT.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "solver/FluidSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;
constexpr double PI = 3.14159265358979323846;

namespace {

// ||u - U0|| over the fluid, plus the drift of the mean velocity.
template <class Solver>
void norms(Solver& s, const double U0[3], double& perturbation) {
  const Domain& d = s.domain();
  s.compute_macroscopic();
  auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hz = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
  double sum = 0, mx = 0;
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) {
        const Index n = d.id(x, y, z);
        const double a = double(hx(n)) - U0[0];
        const double b = double(hy(n)) - U0[1];
        const double c = double(hz(n)) - U0[2];
        sum += a * a + b * b + c * c;
        mx  += double(hx(n));
      }
  perturbation = std::sqrt(sum);
  (void)mx;
}

// Effective viscosity of the ABC flow with a superimposed mean velocity U0.
template <class Coll, class L, class Setup>
double nu_effective(Index N, Real tau, Real U, Real U0, Setup setup) {
  const Real nu = (tau - Real(0.5)) * cs2<L, Real>();
  const double k = 2.0 * PI / double(N);
  const double coef = k * k;
  const auto steps = std::size_t(0.5 / (double(nu) * coef));

  Domain d(N, N, N, true, true, true);
  Coll coll;
  setup(coll, Real(1) / tau);
  FluidSolver<L, EsotericPull<L>, Coll> s(d, coll);

  const Real Uk = U, kk = Real(k), ics2 = inv_cs2<L, Real>(), U0k = U0;
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    const Real X = kk * Real(x), Y = kk * Real(y), Z = kk * Real(z);
    const Real ax = Uk * (Kokkos::sin(Z) + Kokkos::cos(Y));
    const Real ay = Uk * (Kokkos::sin(X) + Kokkos::cos(Z));
    const Real az = Uk * (Kokkos::sin(Y) + Kokkos::cos(X));
    FlowState st;
    // only the ABC part needs a balancing pressure; a uniform translation does not
    st.rho = Real(1) - Real(0.5) * (ax * ax + ay * ay + az * az) * ics2;
    st.ux = ax + U0k;  st.uy = ay + U0k;  st.uz = az + U0k;
    return st;
  });

  const double bg[3] = {double(U0), double(U0), double(U0)};
  double n0 = 0, n1 = 0;
  norms(s, bg, n0);
  for (std::size_t t = 0; t < steps; ++t) s.step();
  norms(s, bg, n1);
  return -std::log(n1 / n0) / (coef * double(steps));
}

struct Op { const char* name; std::vector<double> nu; double drift; };

}  // namespace

//------------------------------------------------------------------------------
// Operator tags at NAMESPACE scope, deliberately not inside main().
//
// These are template arguments for sweep(), whose body contains a
// KOKKOS_LAMBDA, and nvcc rejects "a type local to a function" in the template
// arguments of an extended lambda. Declaring them here costs nothing and is the
// whole fix; the Threads backend accepts either form, which is why they were
// local to begin with.
//------------------------------------------------------------------------------
struct TB { using type = BGK<D3Q27, SecondOrderEquilibrium<D3Q27>, NoForcing, ShiftedPopulations>; };
struct TT { using type = TRT<D3Q27, SecondOrderEquilibrium<D3Q27>, NoForcing, ShiftedPopulations>; };
struct TM { using type = MomentCollision<D3Q27, NoForcing, ShiftedPopulations, false>; };
struct TC { using type = MomentCollision<D3Q27, NoForcing, ShiftedPopulations, true>; };
struct QB { using type = BGK<D3Q19, SecondOrderEquilibrium<D3Q19>, NoForcing, ShiftedPopulations>; };
struct QM { using type = MomentCollision<D3Q19, NoForcing, ShiftedPopulations, false>; };
struct QC { using type = MomentCollision<D3Q19, NoForcing, ShiftedPopulations, true>; };

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const Index N = 32;
    const Real tau = Real(0.8);
    const Real U = Real(0.01);
    const double nu_in = double((tau - Real(0.5)) * cs2<D3Q27, Real>());
    const std::vector<Real> U0s = {Real(0.0), Real(0.05), Real(0.10), Real(0.15)};

    std::printf("Galilean invariance   (ABC/Beltrami flow, N=%d, tau=%.2f, nu=%.6f)\n",
                int(N), double(tau), nu_in);
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());
    std::printf("Superimposing a uniform velocity U0 only translates the solution, so the\n"
                "recovered viscosity must not depend on it. Each column is nu_eff at that U0.\n\n");

    std::printf("%-16s", "operator");
    for (Real u0 : U0s) std::printf(" U0=%-11.2f", double(u0));
    std::printf("  %-11s\n", "drift");
    std::printf("%s\n", std::string(80, '-').c_str());

    auto sweep = [&](const char* nm, auto tag, auto setup) -> Op {
      using C = typename decltype(tag)::type;
      using L = typename C::Lattice;
      Op o{nm, {}, 0};
      for (Real u0 : U0s) o.nu.push_back(nu_effective<C, L>(N, tau, U, u0, setup));
      for (double v : o.nu) o.drift = std::max(o.drift, std::abs(v / o.nu[0] - 1.0));
      std::printf("%-16s", nm);
      for (double v : o.nu) std::printf(" %-14.9f", v);
      std::printf("  %-11.2e\n", o.drift);
      return o;
    };

    auto plain = [](auto& c, Real w) { c.omega = w; };
    auto trt   = [](auto& c, Real w) {
      using T = std::decay_t<decltype(c)>;
      c.omega_p = w; c.omega_m = T::omega_minus_for(w, T::magic_3_16);
    };

    std::printf("D3Q27:\n");
    const Op b = sweep("  BGK",            TB{}, plain);
    const Op t = sweep("  TRT(3/16)",      TT{}, trt);
    const Op m = sweep("  MRT (raw)",      TM{}, plain);
    const Op c = sweep("  CentralMoments", TC{}, plain);
    std::printf("D3Q19  (moment operators via the 19-monomial basis):\n");
    const Op qb = sweep("  BGK",            QB{}, plain);
    const Op qm = sweep("  MRT (raw)",      QM{}, plain);
    const Op qc = sweep("  CentralMoments", QC{}, plain);

    // How the drift scales with U0 tells us what KIND of error is left.
    auto power = [&](const Op& o) {
      const double d1 = std::abs(o.nu[1] - o.nu[0]);
      const double d3 = std::abs(o.nu[3] - o.nu[0]);
      return std::log(d3 / d1) / std::log(double(U0s[3]) / double(U0s[1]));
    };
    std::printf("\n%-16s %-12s %-12s %-10s\n", "operator", "drift", "vs BGK", "power in U0");
    std::printf("%s\n", std::string(56, '-').c_str());
    for (const Op* o : {&b, &t, &m, &c, &qb, &qm, &qc})
      std::printf("%-16s %-12.2e %-12s %-10.2f\n", o->name, o->drift,
                  (o == &b ? "-" : (std::to_string(int(b.drift / o->drift)) + "x").c_str()),
                  power(*o));

    std::printf("\nEvery operator's residual scales as U0^2, i.e. it is the weakly-compressible\n"
                "O(Ma^2) error of the method itself, not a frame dependence of the collision.\n"
                "What the moment operators change is the COEFFICIENT, by three orders of\n"
                "magnitude. Note also that TRT is no better than BGK here: the magic parameter\n"
                "fixes where the WALL sits, not which frame the collision prefers -- these are\n"
                "independent defects and it takes both operators to address both.\n");

    const double tol_cm = sizeof(Real) == 4 ? 5e-4 : 2e-4;
    const bool pass_cm     = c.drift < tol_cm;
    const bool pass_better = c.drift < b.drift / 100.0 && m.drift < b.drift / 100.0;
    const bool pass_raw    = c.drift <= m.drift;     // central at least as good as raw
    const bool pass_q19    = qc.drift < qb.drift / 100.0;
    const bool pass_power  = std::abs(power(c) - 2.0) < 0.25;
    std::printf("\nacceptance:\n");
    std::printf("  CentralMoments viscosity drift %.2e < %.1e            %s\n",
                c.drift, tol_cm, pass_cm ? "PASS" : "FAIL");
    std::printf("  moment operators at least 100x better than BGK          %s\n",
                pass_better ? "PASS" : "FAIL");
    std::printf("  central moments no worse than raw moments               %s\n",
                pass_raw ? "PASS" : "FAIL");
    std::printf("  D3Q19 CM also at least 100x better than D3Q19 BGK       %s\n",
                pass_q19 ? "PASS" : "FAIL");
    std::printf("  residual scales as U0^2 (compressibility, power %.2f)    %s\n",
                power(c), pass_power ? "PASS" : "FAIL");
    if (!(pass_cm && pass_better && pass_raw && pass_power && pass_q19)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
