#pragma once
//==============================================================================
//  Shared harness for the De Rosis & Coreixas (2020) validation campaign,
//  Phys. Fluids 32, 117101, tests III.A - III.E.
//
//  One dispatch and one set of diagnostics for every test, so that a difference
//  between two runs is a difference in the physics rather than in the
//  measurement. Every configuration uses:
//
//    * the highest-order equilibrium the lattice admits (HighOrderEquilibrium),
//      as the paper does -- product form on D2Q9 and D3Q27, fourth order on
//      D3Q19;
//    * shifted population storage;
//    * no external force -- the campaign is deliberately force-free, so the
//      wall-force interaction of the regularised condition never arises.
//
//  The three operators are BGK, raw MRT and central moments. MRT and CM share
//  MomentCollision and differ only in the basis velocity, so any discrepancy
//  between them is that choice and nothing else.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace campaign {

using namespace lbm;

template <class L> using OpBGK = BGK<L, HighOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;
template <class L> using OpMRT = MRT<L, NoForcing, ShiftedPopulations>;
template <class L> using OpCM  = CentralMoments<L, NoForcing, ShiftedPopulations>;

// fn is a generic lambda: fn(Collision{}) with Collision deduced.
template <class Fn>
bool dispatch(const std::string& lat, const std::string& op, Fn&& fn) {
  auto pick = [&](auto lattice) {
    using L = decltype(lattice);
    if (op == "bgk") { fn(OpBGK<L>{}); return true; }
    if (op == "mrt") { fn(OpMRT<L>{}); return true; }
    if (op == "cm")  { fn(OpCM<L>{});  return true; }
    return false;
  };
  if (lat == "d2q9")  return pick(D2Q9{});
  if (lat == "d3q19") return pick(D3Q19{});
  if (lat == "d3q27") return pick(D3Q27{});
  return false;
}

//------------------------------------------------------------------------------
// Volume-integrated kinetic energy and enstrophy over the interior, with
// central differences for the vorticity. Both are returned unnormalised; the
// caller divides by the value at t = 0.
//------------------------------------------------------------------------------
struct Diag { double energy, enstrophy; bool finite; };

template <class Solver>
Diag diagnostics(Solver& s, const Domain& d, Index nx, Index ny, Index nz = 1) {
  s.compute_macroscopic();
  auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hz = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
  double e = 0, w = 0;
  bool ok = true;
  auto U = [&](Index x, Index y, Index z, int c) {
    const Index n = d.id((x + nx) % nx, (y + ny) % ny, (z + nz) % nz);
    return c == 0 ? double(hx(n)) : (c == 1 ? double(hy(n)) : double(hz(n)));
  };
  for (Index z = 0; z < nz; ++z)
    for (Index y = 0; y < ny; ++y)
      for (Index x = 0; x < nx; ++x) {
        const Index n = d.id(x, y, z);
        const double a = double(hx(n)), b = double(hy(n)), c = double(hz(n));
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) { ok = false; continue; }
        e += 0.5 * (a * a + b * b + c * c);
        const double wz = 0.5 * (U(x + 1, y, z, 1) - U(x - 1, y, z, 1))
                        - 0.5 * (U(x, y + 1, z, 0) - U(x, y - 1, z, 0));
        double wx = 0, wy = 0;
        if (nz > 1) {
          wx = 0.5 * (U(x, y + 1, z, 2) - U(x, y - 1, z, 2))
             - 0.5 * (U(x, y, z + 1, 1) - U(x, y, z - 1, 1));
          wy = 0.5 * (U(x, y, z + 1, 0) - U(x, y, z - 1, 0))
             - 0.5 * (U(x + 1, y, z, 2) - U(x - 1, y, z, 2));
        }
        w += 0.5 * (wx * wx + wy * wy + wz * wz);
      }
  return {e, w, ok};
}

// A results file per (test, lattice, operator), written into its own folder.
inline std::FILE* open_out(const std::string& dir, const std::string& test,
                           const std::string& lat, const std::string& op) {
  const std::string path = "results/" + dir + "/" + test + "_" + lat + "_" + op + ".dat";
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (f) std::printf("  -> %s\n", path.c_str());
  return f;
}

}  // namespace campaign
