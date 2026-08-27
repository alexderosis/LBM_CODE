//==============================================================================
//  Does an INSULATING wall conserve the scalar?
//
//  scalar_walls.cpp covers Dirichlet walls, where mass is deliberately not
//  conserved -- that is what a fixed-value boundary is for. Nothing covered the
//  insulating one, and under Esoteric Pull it is the boundary that does nothing
//  at all: an adiabatic node is skipped, and the population its fluid
//  neighbour emitted into the shared slot is read back reversed. "Costs
//  nothing" is a strong claim to leave untested, because a closed box with a
//  source in it has an exact answer -- what went in is what is there -- and any
//  scheme that does not reproduce it is losing material somewhere.
//
//  The four cases below add one thing at a time, so a failure names its cause:
//
//    box       closed box, no flow. Only the six outer walls.
//    block     the same box with a solid slab in the middle, source beside it.
//              Adds interior wall faces, edges and corners.
//    uniform   the block case with a constant wind. Advection on, div u = 0.
//    sheared   the same wind zeroed inside the block, which is what
//              demonstrator/urban does to a log profile. div u is now nonzero,
//              and the question is whether the mass budget can see it.
//
//  WHAT IS SUMMED MATTERS. The macroscopic field is zero by definition at an
//  adiabatic node, so summing it answers "how much is in the fluid", not "how
//  much exists". Both are reported: if they disagree, material is parked in
//  wall slots rather than destroyed, and those are different bugs.
//==============================================================================
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace lbm;

using L      = D3Q7;
using Coll   = ScalarBGK<L>;
using Solver = ScalarSolver<L, EsotericPull<L>, Coll>;

namespace {

struct Case {
  const char* name;
  bool block;        // solid slab in the middle
  bool wind;         // prescribed advection
  bool zero_in_solid;// zero the wind inside the slab, as urban.cpp does
};

struct Result { double injected, in_fluid, in_fluid_next, everywhere; };

Result run(const Case& c, Index N = 32, std::size_t steps = 400) {
  Domain d(N, N, N, false, false, false);
  Coll coll;
  coll.omega = Coll::omega_from_diffusivity(Real(0.05));
  coll.T_ref = Real(0);

  // A slab spanning the middle third in x and y, the full height in z minus a
  // gap top and bottom, so the source sees faces, edges and corners of it.
  const Index lo = N / 3, hi = 2 * N / 3, zlo = 4, zhi = N - 5;
  auto solid = [&](Index x, Index y, Index z) {
    return c.block && x >= lo && x <= hi && y >= lo && y <= hi && z >= zlo && z <= zhi;
  };

  Solver s(d, coll);
  s.set_geometry([&](Index x, Index y, Index z) -> ScalarCell {
    if (solid(x, y, z)) return ScalarAdiabatic;
    // Every outer face insulating: a CLOSED box, so the only way for the total
    // to change is the source.
    if (x == 0 || x == N - 1 || y == 0 || y == N - 1 || z == 0 || z == N - 1)
      return ScalarAdiabatic;
    return ScalarBulk;
  });
  s.set_wall_values([](Index, Index, Index) -> Real { return Real(0); });
  s.finalize_geometry();
  s.initialize(Real(0));

  View1D<Real> ux, uy, uz;
  if (c.wind) {
    ux = View1D<Real>("ux", d.n_padded);
    uy = View1D<Real>("uy", d.n_padded);
    uz = View1D<Real>("uz", d.n_padded);
    auto hx = Kokkos::create_mirror_view(ux);
    auto hy = Kokkos::create_mirror_view(uy);
    auto hz = Kokkos::create_mirror_view(uz);
    for (Index n = 0; n < d.n_padded; ++n) { hx(n) = hy(n) = hz(n) = Real(0); }
    for (Index z = 0; z < N; ++z)
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) {
          const Index n = d.id(x, y, z);
          const bool in = solid(x, y, z);
          hx(n) = (c.zero_in_solid && in) ? Real(0) : Real(0.04);
        }
    Kokkos::deep_copy(ux, hx); Kokkos::deep_copy(uy, hy); Kokkos::deep_copy(uz, hz);
    s.set_velocity(ux, uy, uz);
  }

  // Source beside the slab on the upwind side, well clear of the outer walls.
  const Index si = lo - 3, sj = N / 2, sk = N / 2;
  const Real per_cell = Real(1) / Real(27);
  const Domain dc = d;
  auto source = KOKKOS_LAMBDA(Index n) -> Real {
    Index px, py, pz; dc.coords(n, px, py, pz);
    const Index x = px - dc.hx, y = py - dc.hy, z = pz - dc.hz;
    return (x >= si - 1 && x <= si + 1 && y >= sj - 1 && y <= sj + 1 &&
            z >= sk - 1 && z <= sk + 1) ? per_cell : Real(0);
  };

  double injected = 0.0;
  for (std::size_t t = 0; t < steps; ++t) {
    s.add_source(source);
    s.step();
    injected += 1.0;                       // 27 cells x 1/27 per step
  }

  auto fluid_sum = [&]() {
    s.compute_field();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
    auto f = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
    double m = 0.0;
    for (Index z = 0; z < N; ++z)
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) {
          const Index n = d.id(x, y, z);
          if (f(n) == ScalarBulk) m += double(h(n));
        }
    return m;
  };
  const double fluid = fluid_sum();
  // Everything that exists, wall slots included. Esoteric Pull keeps in-flight
  // populations in the slots of whichever node owns them, so this is the only
  // sum that answers "was it destroyed" rather than "is it in the fluid".
  const double all = double(s.total_population());
  // The same fluid sum one step later. Esoteric Pull alternates which half of a
  // wall node's slots its fluid neighbours read, so if the deficit were purely
  // a parity artefact this would swing by roughly the deficit. It does not --
  // the shortfall is steady, which says the material sitting in wall slots is a
  // standing boundary-layer quantity, not something that appears and vanishes.
  s.step();
  const double fluid_next = fluid_sum();
  return {injected, fluid, fluid_next, all};
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("\nScalar mass conservation   D3Q7   %s\n", precision_name());
    std::printf("%s\n\n", std::string(86, '=').c_str());
    std::printf("  A closed box with a source in it. What went in is what must be there.\n\n");
    std::printf("  %-10s %13s %13s %9s %13s %9s %11s\n",
                "case", "injected", "in fluid", "error", "everywhere", "error",
                "next step");
    std::printf("  %s\n", std::string(86, '-').c_str());

    const Case cases[] = {
      {"box",     false, false, false},
      {"block",   true,  false, false},
      {"uniform", true,  true,  false},
      {"sheared", true,  true,  true },
    };
    for (const Case& c : cases) {
      const Result r = run(c);
      const double ef = (r.in_fluid - r.injected) / r.injected;
      const double ea = (r.everywhere - r.injected) / r.injected;
      std::printf("  %-10s %13.6f %13.6f %8.4f%% %13.6f %8.4f%% %11.6f\n",
                  c.name, r.injected, r.in_fluid, 100 * ef, r.everywhere, 100 * ea,
                  r.in_fluid_next);
      // The closed box has no sink at all, so the total must be the injection
      // to round-off. A tolerance of 1e-9 is far above double round-off over
      // 400 steps and far below anything a real leak would produce.
      if (std::abs(ea) > 1e-9) rc = 1;
    }
    std::printf("\n  \"in fluid\" sums the macroscopic field over bulk nodes;\n"
                "  \"everywhere\" sums every population in the lattice, wall slots\n"
                "  included. A gap between them is material parked in a wall rather\n"
                "  than destroyed; a gap in the second column is destroyed material.\n\n");
    std::printf("  %s\n\n", rc ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return rc;
}
