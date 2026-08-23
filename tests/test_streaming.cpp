//==============================================================================
//  STREAMING TESTS
//
//  1. PERMUTATION TEST. Tag every population with a unique value, disable
//     collision, run T steps on a fully periodic domain, then assert that every
//     value has moved exactly T * c_i lattice sites. Run against BOTH streaming
//     schemes through the same logical interface, so Esoteric Pull's parity
//     bookkeeping has nowhere to hide.
//
//  2. CROSS-CHECK. Run TwoLattice and EsotericPull side by side on a real flow
//     with walls, and compare the full population field after every step. The
//     two schemes share no storage code, so agreement to round-off is strong
//     evidence that the in-place scheme is right.
//==============================================================================
#include "Check.hpp"
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/TwoLattice.hpp"
#include "solver/FluidSolver.hpp"

#include <cmath>

using namespace lbm;

// Collision that does nothing: isolates streaming completely.
template <class L>
struct NullCollision {
  using Equilibrium = SecondOrderEquilibrium<L>;
  using Storage     = RawPopulations;
  KOKKOS_INLINE_FUNCTION Macro macroscopic(const Real[L::Q], Index = 0) const {
    return Macro{Real(1), Real(0), Real(0), Real(0)};
  }
  KOKKOS_INLINE_FUNCTION void collide(Real[L::Q], const Macro&, Index = 0) const {}
  KOKKOS_INLINE_FUNCTION static Real density(const Macro& m) { return m.dens; }
  KOKKOS_INLINE_FUNCTION static Real seed_value(int, Real r, Real, Real, Real) { return r; }
};

//------------------------------------------------------------------------------
template <class L, class Streaming>
void permutation(Index nx, Index ny, Index nz, int steps) {
  const std::string nm = std::string(Streaming::name) + " " + L::name + " " +
                         std::to_string(nx) + "x" + std::to_string(ny) + "x" +
                         std::to_string(nz) + " T=" + std::to_string(steps);
  Domain d(nx, ny, nz, true, true, true);
  FluidSolver<L, Streaming, NullCollision<L>> s(d, NullCollision<L>{});

  const int Q = L::Q;
  s.seed_populations(KOKKOS_LAMBDA(Index n, int i) {
    Index x, y, z; d.coords(n, x, y, z);
    return Real(((z * ny + y) * nx + x) * Q + i + 1);
  });

  for (int t = 0; t < steps; ++t) s.step();

  auto g  = s.gather_populations();
  auto hg = Kokkos::create_mirror_view_and_copy(HostSpace{}, g);

  auto wrap = [](Index v, Index n) { return ((v % n) + n) % n; };
  auto tag  = [&](Index x, Index y, Index z, int i) {
    return Real(((z * ny + y) * nx + x) * Q + i + 1);
  };

  int bad = 0;
  for (Index z = 0; z < nz; ++z)
    for (Index y = 0; y < ny; ++y)
      for (Index x = 0; x < nx; ++x)
        for (int i = 0; i < Q; ++i) {
          const Index sx = wrap(x - steps * cvel<L>(i, 0), nx);
          const Index sy = wrap(y - steps * cvel<L>(i, 1), ny);
          const Index sz = wrap(z - steps * cvel<L>(i, 2), nz);
          if (hg(d.id(x, y, z), i) != tag(sx, sy, sz, i)) ++bad;
        }
  check::ok(bad == 0, nm + ": every population landed exactly T*c_i away (" +
                          std::to_string(bad) + " misplaced)");
}

//------------------------------------------------------------------------------
template <class L, class Store>
void cross_check(int steps) {
  const std::string nm = std::string(L::name) + "/" + Store::name;
  const Index nx = 12, ny = 10, nz = (L::D == 3) ? 8 : 1;

  using Coll = BGK<L, SecondOrderEquilibrium<L>, Guo, Store>;
  Coll coll;
  coll.omega   = Real(1.4);
  coll.forcing = Guo{Real(2e-5), Real(-1e-5), Real(0)};

  // Periodic in x, walls in y: exercises bounce-back on both schemes.
  Domain d(nx, ny, nz, true, false, true);
  FluidSolver<L, TwoLattice<L>,   Coll> a(d, coll);
  FluidSolver<L, EsotericPull<L>, Coll> b(d, coll);
  auto geom = [&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  };
  a.set_geometry(geom);  b.set_geometry(geom);
  a.initialize(Real(1), Real(0.03), Real(0.01), Real(0));
  b.initialize(Real(1), Real(0.03), Real(0.01), Real(0));

  const Real tol = sizeof(Real) == 4 ? Real(1e-6) : Real(1e-13);
  double worst = 0;
  int first_bad_step = -1;

  for (int t = 0; t < steps; ++t) {
    a.step(); b.step();
    auto ga = Kokkos::create_mirror_view_and_copy(HostSpace{}, a.gather_populations());
    auto gb = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.gather_populations());
    auto fa = Kokkos::create_mirror_view_and_copy(HostSpace{}, a.flags());
    for (Index n = 0; n < d.n_padded; ++n) {
      if (fa(n) != Fluid) continue;      // only fluid cells have a shared meaning
      for (int i = 0; i < L::Q; ++i) {
        const double e = std::abs(double(ga(n, i)) - double(gb(n, i)));
        if (e > worst) worst = e;
        if (e > double(tol) && first_bad_step < 0) first_bad_step = t;
      }
    }
  }
  char buf[96];
  std::snprintf(buf, sizeof buf, " after %d steps (worst |diff| = %.3e)", steps, worst);
  check::ok(first_bad_step < 0,
            nm + ": EsotericPull matches TwoLattice" + buf +
            (first_bad_step >= 0 ? ", diverged at step " + std::to_string(first_bad_step) : ""));
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    permutation<D2Q9,  TwoLattice<D2Q9>>(8, 6, 1, 1);
    permutation<D2Q9,  TwoLattice<D2Q9>>(8, 6, 1, 24);
    permutation<D2Q9,  EsotericPull<D2Q9>>(8, 6, 1, 1);
    permutation<D2Q9,  EsotericPull<D2Q9>>(8, 6, 1, 2);
    permutation<D2Q9,  EsotericPull<D2Q9>>(8, 6, 1, 7);
    permutation<D2Q9,  EsotericPull<D2Q9>>(8, 6, 1, 24);
    permutation<D3Q19, TwoLattice<D3Q19>>(5, 4, 3, 11);
    permutation<D3Q19, EsotericPull<D3Q19>>(5, 4, 3, 1);
    permutation<D3Q19, EsotericPull<D3Q19>>(5, 4, 3, 11);
    permutation<D3Q27, EsotericPull<D3Q27>>(5, 4, 3, 13);

    cross_check<D2Q9,  RawPopulations>(40);
    cross_check<D2Q9,  ShiftedPopulations>(40);
    cross_check<D3Q19, RawPopulations>(20);
    cross_check<D3Q27, ShiftedPopulations>(20);
  }
  const int r = check::report("streaming");
  Kokkos::finalize();
  return r;
}
