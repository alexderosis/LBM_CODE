//==============================================================================
//  Grid convergence study: every lattice x every collision operator.
//
//  Emits CSV on stdout. Not part of ctest -- this is an analysis run, not a
//  regression check; `decaying_flows` is the one that asserts.
//
//  Two cases, both with exact Navier-Stokes solutions:
//    taylor_green  a z-independent field, so it runs on D2Q9, D3Q19 and D3Q27
//                  alike and is the only case that lets all three be compared
//                  on the same problem;
//    abc           genuinely 3D, so D3Q19 and D3Q27 only.
//
//  Diffusive scaling U ~ 1/N keeps the Mach-number error second order alongside
//  the spatial error, so a clean slope of -2 on a log-log plot of error against
//  N is what a correct implementation should give.
//
//  D2Q5 and D3Q7 are absent on purpose: they are advection-diffusion lattices
//  with no isotropic 4th-order moments, cannot support Navier-Stokes, and have
//  no fluid solver here to converge. They arrive with the thermal module.
//==============================================================================
#include "DecayingFlows.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;

namespace {

const Real TAU = Real(0.8);
const double FRAC = 0.5;

void row(const char* cs, const char* lat, const char* op, int N,
         const Result& r, int nodes) {
  std::printf("%s,%s,%s,%d,%d,%.10e,%.10e,%.10e,%zu\n",
              cs, lat, op, N, nodes, r.nu_eff, r.nu_err, r.l2_u, r.steps);
}

template <class L, template <class> class Op, class Setup>
void tg_ladder(const char* op, Setup setup, const std::vector<int>& Ns) {
  const Index Nz = (L::D == 3) ? 8 : 1;
  for (int N : Ns) {
    const Result r = taylor_green<Op<L>, L, EsotericPull>(
        N, Nz, TAU, Real(0.64 / N), FRAC, setup);
    row("taylor_green", L::name, op, N, r, N * N * int(Nz));
  }
}

template <class L, template <class> class Op, class Setup>
void abc_ladder(const char* op, Setup setup, const std::vector<int>& Ns) {
  for (int N : Ns) {
    const Result r = abc_flow<Op<L>, L, EsotericPull>(
        N, TAU, Real(0.32 / N), FRAC, setup);
    row("abc", L::name, op, N, r, N * N * N);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("case,lattice,operator,N,nodes,nu_eff,nu_err,l2_u,steps\n");
    const std::vector<int> tg = {16, 24, 32, 48, 64};
    const std::vector<int> ab = {16, 24, 32, 48};

    // ---- Taylor-Green: all three Navier-Stokes lattices ----
    tg_ladder<D2Q9,  ops::Bgk>("BGK", ops::plain, tg);
    tg_ladder<D2Q9,  ops::Trt>("TRT", ops::trt,   tg);
    tg_ladder<D2Q9,  ops::Mrt>("MRT", ops::plain, tg);
    tg_ladder<D2Q9,  ops::Cm >("CM",  ops::plain, tg);

    tg_ladder<D3Q19, ops::Bgk>("BGK", ops::plain, tg);
    tg_ladder<D3Q19, ops::Trt>("TRT", ops::trt,   tg);
    tg_ladder<D3Q19, ops::Mrt>("MRT", ops::plain, tg);
    tg_ladder<D3Q19, ops::Cm >("CM",  ops::plain, tg);

    tg_ladder<D3Q27, ops::Bgk>("BGK", ops::plain, tg);
    tg_ladder<D3Q27, ops::Trt>("TRT", ops::trt,   tg);
    tg_ladder<D3Q27, ops::Mrt>("MRT", ops::plain, tg);
    tg_ladder<D3Q27, ops::Cm >("CM",  ops::plain, tg);

    // ---- ABC/Beltrami: genuinely 3D, so the 3D lattices only ----
    abc_ladder<D3Q19, ops::Bgk>("BGK", ops::plain, ab);
    abc_ladder<D3Q19, ops::Trt>("TRT", ops::trt,   ab);
    abc_ladder<D3Q19, ops::Mrt>("MRT", ops::plain, ab);
    abc_ladder<D3Q19, ops::Cm >("CM",  ops::plain, ab);

    abc_ladder<D3Q27, ops::Bgk>("BGK", ops::plain, ab);
    abc_ladder<D3Q27, ops::Trt>("TRT", ops::trt,   ab);
    abc_ladder<D3Q27, ops::Mrt>("MRT", ops::plain, ab);
    abc_ladder<D3Q27, ops::Cm >("CM",  ops::plain, ab);
  }
  Kokkos::finalize();
  return 0;
}
