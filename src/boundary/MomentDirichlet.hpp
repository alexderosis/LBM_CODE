#pragma once
//==============================================================================
//  Moment-based Dirichlet boundary conditions for the advection-diffusion
//  lattices (D2Q5, D3Q7).
//
//  P. J. Dellar, "Moment-Based Boundary Conditions for Lattice Boltzmann
//  Magnetohydrodynamics", Eqs. (13a)-(13b).
//
//  The idea is that the field carried by these lattices is a ZEROTH moment,
//
//      B_a = sum_i g_{a,i},        T = sum_i h_i,
//
//  so imposing its boundary value is one linear equation in the unknown
//  populations. On a cross lattice a straight wall leaves EXACTLY ONE unknown
//  direction -- the one pointing into the domain along the wall normal -- so
//  that single equation determines it uniquely and exactly:
//
//      g_1 = B_0 - (g_0 + g_2 + g_3 + g_4).            Dellar, Eq. (13a)
//
//  That is the whole method. No closure assumption, no free parameter, and the
//  boundary value is attained AT the node rather than half-way to the next one,
//  which is what bounce-back and anti-bounce-back give.
//
//  Corners and edges leave more than one unknown and the single moment no
//  longer closes the system. There the deficit is shared out in proportion to
//  the lattice weights: that is the choice which introduces no directional
//  preference, and it still reproduces the imposed moment exactly. It is a
//  fallback, not part of the published method, and Hartmann flow -- periodic
//  along the channel -- never reaches it.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Directions whose source node lies outside the fluid, as a bitmask. Host-side:
// geometry does not change during a run, so this is built once at setup.
//
// `is_fluid(x, y, z)` is called with INTERIOR coordinates and must report
// whether that node participates in the transport.
//------------------------------------------------------------------------------
template <class L, class IsFluid>
std::uint32_t unknown_mask(const Domain& d, Index x, Index y, Index z, IsFluid is_fluid) {
  std::uint32_t m = 0;
  for (int i = 0; i < L::Q; ++i) {
    Index sx = x - cvel<L>(i, 0), sy = y - cvel<L>(i, 1), sz = z - cvel<L>(i, 2);
    bool outside = false;
    if (d.periodic[0]) sx = (sx + d.nx) % d.nx; else if (sx < 0 || sx >= d.nx) outside = true;
    if (d.periodic[1]) sy = (sy + d.ny) % d.ny; else if (sy < 0 || sy >= d.ny) outside = true;
    if (d.periodic[2]) sz = (sz + d.nz) % d.nz; else if (sz < 0 || sz >= d.nz) outside = true;
    if (!outside && !is_fluid(sx, sy, sz)) outside = true;
    if (outside) m |= (1u << i);
  }
  return m;
}

//------------------------------------------------------------------------------
// Set the unknown populations so that sum_i g_i equals `target`.
//
// One unknown  -> exact and unique, Dellar Eq. (13).
// Several      -> weight-proportional share of the deficit; the moment is still
//                 reproduced exactly, only its distribution is a choice.
//------------------------------------------------------------------------------
template <class L>
KOKKOS_INLINE_FUNCTION
void impose_moment(Real g[L::Q], Real target, std::uint32_t unknown) {
  Real known = Real(0), wsum = Real(0);
  for (int i = 0; i < L::Q; ++i) {
    if (unknown & (1u << i)) wsum += weight<L, Real>(i);
    else                     known += g[i];
  }
  if (!(wsum > Real(0))) return;          // nothing streamed in: leave it alone
  const Real deficit = target - known;
  for (int i = 0; i < L::Q; ++i)
    if (unknown & (1u << i)) g[i] = deficit * weight<L, Real>(i) / wsum;
}

}  // namespace lbm
