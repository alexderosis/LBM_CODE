#pragma once
//==============================================================================
//  Cartesian block, periodicity, and neighbour lookup.
//
//  Periodicity is resolved by WRAPPING the neighbour index, not by copying halo
//  layers. That matters for the in-place streaming schemes: Esoteric Pull writes
//  into its neighbours' storage, so a halo-copy scheme would need a two-way
//  exchange whose slot set depends on parity. Wrapping sidesteps that entirely
//  and is what the reference implementations do.
//
//  A halo layer is therefore only allocated in NON-periodic directions, where
//  edge cells genuinely need somewhere out-of-domain to read from. MPI will
//  reintroduce a halo on partitioned directions; `mpi_halo` is the hook.
//
//  Padded storage order is x-fastest:  idp = (pz*sy + py)*sx + px
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Neighbour indices for one node, built once per node and reused by the
// streaming policy: nb[i] is the node at n + c_i.
//
// Sized Q even for schemes that only use half the directions: compacting it to
// the odd half was tried and measured, and made no difference on D3Q27 while
// costing ~4% on D3Q19 for the extra index shift. The kernel is not limited by
// this array.
//------------------------------------------------------------------------------
template <class L>
struct Neighbours {
  Index n = 0;
  Index j[L::Q] = {};
};

class Domain {
 public:
  Index nx = 1, ny = 1, nz = 1;   // interior extents
  Index hx = 0, hy = 0, hz = 0;   // halo widths (non-periodic directions only)
  Index sx = 1, sy = 1, sz = 1;   // padded extents
  Index n_padded = 1;
  bool  periodic[3] = {true, true, true};

  Domain() = default;

  Domain(Index nx_, Index ny_, Index nz_ = 1,
         bool px = true, bool py = true, bool pz = true, Index mpi_halo = 0) {
    nx = nx_; ny = ny_; nz = nz_;
    periodic[0] = px; periodic[1] = py; periodic[2] = pz;
    hx = halo_for(nx, px, mpi_halo);
    hy = halo_for(ny, py, mpi_halo);
    hz = halo_for(nz, pz, mpi_halo);
    sx = nx + 2 * hx; sy = ny + 2 * hy; sz = nz + 2 * hz;
    n_padded = sx * sy * sz;
  }

  KOKKOS_INLINE_FUNCTION Index n_interior() const { return nx * ny * nz; }

  KOKKOS_INLINE_FUNCTION Index idp(Index px, Index py, Index pz) const {
    return (pz * sy + py) * sx + px;
  }
  KOKKOS_INLINE_FUNCTION Index id(Index x, Index y, Index z = 0) const {
    return idp(x + hx, y + hy, z + hz);
  }
  KOKKOS_INLINE_FUNCTION void coords(Index n, Index& px, Index& py, Index& pz) const {
    px = n % sx;
    py = (n / sx) % sy;
    pz = n / (sx * sy);
  }
  KOKKOS_INLINE_FUNCTION bool is_interior(Index px, Index py, Index pz) const {
    return px >= hx && px < hx + nx &&
           py >= hy && py < hy + ny &&
           pz >= hz && pz < hz + nz;
  }

  //----------------------------------------------------------------------------
  // Fill neighbour indices for the directions i = first, first+stride, ...
  // Esoteric Pull only ever needs the odd-indexed directions, which halves the
  // register footprint of the list, so the streaming policy chooses the stride.
  //----------------------------------------------------------------------------
  template <class L, int First, int Stride>
  KOKKOS_INLINE_FUNCTION void fill_neighbours(Index n, Neighbours<L>& nb) const {
    Index x, y, z;
    coords(n, x, y, z);
    nb.n = n;

    // Fast path: away from every face no wrapping or clamping can apply.
    bool core = (x > 0 && x < sx - 1) && (y > 0 && y < sy - 1);
    if constexpr (L::D == 3) core = core && (z > 0 && z < sz - 1);

    if (core) {
      for (int i = First; i < L::Q; i += Stride)
        nb.j[i] = n + cvel<L>(i, 0) + cvel<L>(i, 1) * sx + cvel<L>(i, 2) * sx * sy;
    } else {
      for (int i = First; i < L::Q; i += Stride)
        nb.j[i] = idp(shift(x + cvel<L>(i, 0), hx, nx, sx, periodic[0]),
                      shift(y + cvel<L>(i, 1), hy, ny, sy, periodic[1]),
                      shift(z + cvel<L>(i, 2), hz, nz, sz, periodic[2]));
    }
  }

 private:
  static Index halo_for(Index n, bool per, Index mpi_halo) {
    if (n <= 1) return 0;            // flat dimension: nothing to wrap into
    if (per) return mpi_halo;        // wrapping handles it on one rank
    return mpi_halo > 0 ? mpi_halo : 1;
  }

  // Periodic directions wrap inside the interior range [h, h+n).
  // Non-periodic directions clamp into the padded array: interior cells always
  // land on a real halo cell, and halo cells (never processed) stay in bounds.
  KOKKOS_INLINE_FUNCTION
  static Index shift(Index v, Index h, Index n, Index s, bool per) {
    if (per) {
      if (v < h)          return v + n;
      if (v >= h + n)     return v - n;
      return v;
    }
    if (v < 0)      return 0;
    if (v >= s)     return s - 1;
    return v;
  }
};

}  // namespace lbm
