#pragma once
//==============================================================================
//  Esoteric Pull streaming, and the cell flags that geometry rides on.
//
//  ESOTERIC PULL, briefly. One lattice, updated in place. The pair of opposite
//  directions (i, i+1) shares a slot, and which of the two a node owns flips
//  with the parity of the timestep. Every slot has exactly one reader and one
//  writer, and they are the same node, so the update is race-free with no
//  temporary buffer and no second lattice.
//
//  Three consequences worth stating because they are the reasons to use it:
//    * half the memory footprint of a two-lattice scheme;
//    * bounce-back becomes the IDENTITY on the storage, so a solid cell needs no
//      work at all -- it is simply not visited. This is what makes arbitrary
//      geometry nearly free here, and it is why `Solid` costs a skipped thread
//      rather than a reflected write;
//    * because each slot has one writer, a whole step may be applied to nodes in
//      any order. The host reference driver in hostsim.hpp relies on exactly
//      that, and so does host_check.
//
//  EVERYTHING IS TEMPLATED ON THE LATTICE, with D3Q27 as the default so that
//  existing call sites read unchanged. The scalar and the magnetic field run on
//  D3Q7 and use the same functions -- which is legitimate only because both
//  lattices obey the same ordering contract (opp(i) == i + 1 for odd i, and
//  direction 0 at rest). If that is ever broken on either lattice, populations
//  still move, just in the wrong directions, and the flow looks plausible.
//  host_check tests the round trip on both for that reason.
//
//  LAYOUT. f[i * n_nodes + node]. Consecutive threads take consecutive nodes,
//  so a warp's 32 accesses to a given direction are contiguous.
//==============================================================================
#include "core.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace lbm {

#if defined(__CUDACC__)
#define LBM_CUDA_CHECK(call)                                                   \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                   cudaGetErrorString(_e), __FILE__, __LINE__);                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
#endif

//==============================================================================
//  Cell flags.
//
//  A boundary condition is an ALTERNATIVE COLLISION applied to marked cells, so
//  there is one branch per NODE rather than one per direction and the population
//  loads stay unconditional and coalesced.
//
//  Fluid    collide normally
//  Solid    halfway bounce-back -- the wall plane sits midway between the last
//           fluid node and the first solid node. Under Esoteric Pull this is the
//           identity on the storage, so the node is skipped outright.
//  Excluded not part of the simulation at all; never visited, never seeded with
//           anything meaningful.
//
//  The flags array is always allocated, but the kernels are TEMPLATED on whether
//  any geometry exists, so a periodic run never issues the load at all.
//
//  THAT TEMPLATE PARAMETER LOOKS LIKE OVER-ENGINEERING AND IS NOT. The first
//  version of this code loaded the flags unconditionally, on the reasoning that
//  one byte per node against 216 of population traffic is half a per cent -- "a
//  rounding error in bandwidth", as the comment then said. Measured on a T4 at
//  128^3 against the commit immediately before geometry existed:
//
//      BGK              978.67 -> 924.14 MLUPS   (-5.6%)
//      central moments  977.99 -> 921.85 MLUPS   (-5.7%)
//
//  Ten times the predicted cost. What a bandwidth-bound kernel pays for is not
//  the byte but the extra memory STREAM: more transactions, more cache pressure,
//  another address to keep in flight. Predicting this kind of cost from a byte
//  count is how it gets got wrong, and the fix is a compile-time bool.
//==============================================================================
enum CellType : std::uint8_t {
  Fluid    = 0,
  Solid    = 1,
  Excluded = 2,
};

//------------------------------------------------------------------------------
// Cell flags for a scalar field. The GEOMETRY is shared with the fluid, but the
// boundary CONDITIONS are not: a wall that is no-slip for momentum may be either
// insulating or held at a fixed temperature, and which one it is has nothing to
// do with the flow.
//
//  ScalarBulk       transport
//  ScalarAdiabatic  zero flux    -- bounce-back, hence skipped (see above)
//  ScalarDirichlet  fixed value  -- anti-bounce-back toward T_wall
//  ScalarExcluded   not part of the simulation
//
// Note the asymmetry with the fluid. Bounce-back is the identity on Esoteric
// Pull's storage so an insulating cell can be skipped, but anti-bounce-back
// flips a sign and adds a source, so a Dirichlet cell must be processed every
// step. Both still write back into the same two slots they read, so neither
// disturbs the in-place scheme.
//------------------------------------------------------------------------------
enum ScalarCell : std::uint8_t {
  ScalarBulk      = 0,
  ScalarAdiabatic = 1,
  ScalarDirichlet = 2,
  ScalarExcluded  = 3,
};

//------------------------------------------------------------------------------
// Periodic node index. Kept as free functions so host_check.cpp and hostsim.hpp
// exercise the same arithmetic the kernels use.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE int wrap(int v, int n) { return (v + n) % n; }

LBM_HD LBM_INLINE long node_id(int x, int y, int z, int nx, int ny) {
  return long(x) + long(nx) * (long(y) + long(ny) * long(z));
}

//------------------------------------------------------------------------------
// Neighbour in direction i, with periodic wrap.
//------------------------------------------------------------------------------
template <class L = D3Q27>
LBM_HD LBM_INLINE long neighbour(int x, int y, int z, int i, int nx, int ny, int nz) {
  return node_id(wrap(x + L::cx(i), nx),
                 wrap(y + L::cy(i), ny),
                 wrap(z + L::cz(i), nz), nx, ny);
}

//------------------------------------------------------------------------------
// Esoteric Pull load / store, templated on parity so there is no runtime test
// inside the hot loop.
//------------------------------------------------------------------------------
template <int Parity>
LBM_HD LBM_INLINE void load_pair(const Real* f, long N, long self, long nb,
                                 int i, Real& a, Real& b) {
  if (Parity == 0) { a = f[long(i) * N + self];       b = f[long(i + 1) * N + nb]; }
  else             { a = f[long(i + 1) * N + self];   b = f[long(i) * N + nb];     }
}

template <int Parity>
LBM_HD LBM_INLINE void store_pair(Real* f, long N, long self, long nb,
                                  int i, Real a, Real b) {
  if (Parity == 0) { f[long(i + 1) * N + nb] = a;  f[long(i) * N + self]     = b; }
  else             { f[long(i) * N + nb]     = a;  f[long(i + 1) * N + self] = b; }
}

//------------------------------------------------------------------------------
// Gather the populations a node should collide with, in direction order.
//------------------------------------------------------------------------------
template <int Parity, class L = D3Q27>
LBM_HD LBM_INLINE void gather(const Real* f, long N, int x, int y, int z,
                              int nx, int ny, int nz, Real* out) {
  const long self = node_id(x, y, z, nx, ny);
  out[0] = f[self];
  for (int i = 1; i < L::Q; i += 2) {
    const long nb = neighbour<L>(x, y, z, i, nx, ny, nz);
    load_pair<Parity>(f, N, self, nb, i, out[i], out[i + 1]);
  }
}

//------------------------------------------------------------------------------
// INVERSE OF gather -- for initialisation only, and it is NOT the same function
// as `scatter` below.
//
// `scatter` writes POST-COLLISION populations into the slots the NEXT parity
// will read, i.e. it streams. Applying it to lay down an initial condition
// therefore shifts every population by one cell before the first step, and the
// paired slots end up describing a state no node ever held. The flow still
// evolves and still looks like a flow, which is what makes this worth a named
// function and a test rather than a comment.
//
// This writes each value into the very slot gather would read it back from, so
// gather(init_scatter(x)) == x exactly. host_check.cpp asserts that.
//------------------------------------------------------------------------------
template <int Parity, class L = D3Q27>
LBM_HD LBM_INLINE void init_scatter(Real* f, long N, int x, int y, int z,
                                    int nx, int ny, int nz, const Real* in) {
  const long self = node_id(x, y, z, nx, ny);
  f[self] = in[0];
  for (int i = 1; i < L::Q; i += 2) {
    const long nb = neighbour<L>(x, y, z, i, nx, ny, nz);
    if (Parity == 0) { f[long(i) * N + self]     = in[i]; f[long(i + 1) * N + nb] = in[i + 1]; }
    else             { f[long(i + 1) * N + self] = in[i]; f[long(i) * N + nb]     = in[i + 1]; }
  }
}

template <int Parity, class L = D3Q27>
LBM_HD LBM_INLINE void scatter(Real* f, long N, int x, int y, int z,
                               int nx, int ny, int nz, const Real* in) {
  const long self = node_id(x, y, z, nx, ny);
  f[self] = in[0];
  for (int i = 1; i < L::Q; i += 2) {
    const long nb = neighbour<L>(x, y, z, i, nx, ny, nz);
    store_pair<Parity>(f, N, self, nb, i, in[i], in[i + 1]);
  }
}

//------------------------------------------------------------------------------
// Decompose a linear node index. Used identically by the kernels and by the
// host reference driver.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void coords(long n, int nx, int ny, int& x, int& y, int& z) {
  x = int(n % nx);
  y = int((n / nx) % ny);
  z = int(n / (long(nx) * ny));
}

}  // namespace lbm
