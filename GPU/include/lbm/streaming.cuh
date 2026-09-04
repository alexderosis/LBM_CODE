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
//  rounding error in bandwidth", as the comment then said. Three points on one
//  T4 at 128^3, FP32:
//
//      before geometry existed   978.15 BGK   977.35 CM
//      flags always loaded       923.99       919.81      (-5.5%, -5.9%)
//      geometry templated        950.16       950.37      (-2.9%, -2.8%)
//
//  So the byte-count estimate was out by a factor of ten, and the template
//  recovers ABOUT HALF of what it cost. What a bandwidth-bound kernel pays for
//  is not the byte but the extra memory STREAM: more transactions, more cache
//  pressure, another address to keep in flight.
//
//  THE REMAINING 2.9% IS NOT EXPLAINED, and it is not the obvious candidates.
//  -Xptxas -v on the same two builds: the old stream_collide used 96 registers
//  across its 4 instantiations, the new fluid_kernel uses 63-80 across its 40,
//  and BOTH spill zero bytes. So it is neither register pressure nor spilling.
//  What else differs is that the kernel now takes a much larger parameter struct
//  and carries a runtime `if (p.ux_out)` for the coupled-velocity write. `ncu
//  --set full` on one kernel would settle it; measure before theorising.
//==============================================================================
//  RegWall  a REGULARISED velocity wall: the node is a FLUID node whose
//           populations are all replaced before it collides, so the imposed
//           velocity holds exactly AT the node rather than half a cell away.
//           See regularized.cuh.
//
//  THE ASYMMETRY WITH Solid IS THE POINT, and it is a live trap. A Solid cell
//  does not collide and is not forced; a RegWall cell does both. The parent's
//  CLAUDE.md records what mixing them cost there: a Boussinesq force reads
//  T = 0 at a RegWall node against the intended T0 and drives a body force
//  along the whole wall. Whatever reads a field at a wall has to know which of
//  the two families it is looking at.
enum CellType : std::uint8_t {
  Fluid    = 0,
  Solid    = 1,
  Excluded = 2,
  RegWall  = 3,
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
//  ScalarOutflow    open         -- zero gradient, by equilibrium extrapolation
//                                  from an interior donor. SECOND PASS; see below
//  ScalarExcluded   not part of the simulation
//
// Note the asymmetry with the fluid. Bounce-back is the identity on Esoteric
// Pull's storage so an insulating cell can be skipped, but anti-bounce-back
// flips a sign and adds a source, so a Dirichlet cell must be processed every
// step. Both still write back into the same two slots they read, so neither
// disturbs the in-place scheme.
//
// OUTFLOW IS THE ONE THAT DOES NOT FIT THAT PATTERN, and the reason is worth
// stating because the obvious implementation is a race.
//
// An open boundary must be zero-gradient, so the node has to learn its value
// from an interior DONOR. Reading the donor's POPULATIONS to get it is a
// genuine read/write conflict under Esoteric Pull: a scatter writes into the
// node's own slots AND its neighbours', so a donor being processed
// concurrently is rewriting slots the outflow node is reading, and which state
// you get depends on thread scheduling. Nothing crashes; the boundary is
// simply somewhere between pre- and post-collision, differently each run.
//
// So outflow nodes are SKIPPED by the main kernel and handled by a second one
// after the kernel boundary, and that pass reads the CONCENTRATION FIELD, not
// populations. Donors are guaranteed to be ScalarBulk, so nothing the second
// pass reads is written by the second pass, and every storage slot still has
// exactly one writer per step. Race-free by construction rather than by
// argument -- and the whole thing is skipped when no cell is marked outflow,
// so a closed problem pays nothing.
//------------------------------------------------------------------------------
enum ScalarCell : std::uint8_t {
  ScalarBulk      = 0,
  ScalarAdiabatic = 1,
  ScalarDirichlet = 2,
  ScalarExcluded  = 3,
  ScalarOutflow   = 4,
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
// Directions whose SOURCE node lies outside the field, as a bitmask. Host-side
// and built once: geometry does not change during a run. D3Q7 needs 7 bits, so
// a std::uint8_t per node is enough and that is what the kernels carry.
//
// `outside(x, y, z)` is called with wrapped coordinates and reports whether that
// node is absent from the transport.
//------------------------------------------------------------------------------
template <class L, class Outside>
inline std::uint8_t unknown_mask(int x, int y, int z, int nx, int ny, int nz,
                                 Outside outside) {
  std::uint8_t m = 0;
  for (int i = 0; i < L::Q; ++i) {
    // The SOURCE of direction i is at x - c_i: that is where the population
    // arriving along i came from. Using x + c_i instead builds the mirror image
    // of the right mask, which imposes the moment on the outgoing directions --
    // it still reproduces the moment and is still wrong.
    const int sx = wrap(x - L::cx(i), nx);
    const int sy = wrap(y - L::cy(i), ny);
    const int sz = wrap(z - L::cz(i), nz);
    if (outside(sx, sy, sz)) m = std::uint8_t(m | (1u << i));
  }
  return m;
}

//==============================================================================
//  FP16 STORAGE, EMULATED -- the accuracy half of the mixed-precision question,
//  answered before any memory layout changes and without FP16 hardware.
//
//  A real implementation would store the populations as 16-bit and compute in
//  32-bit, roughly halving the traffic of a kernel this tree has measured at
//  64% of a T4's peak bandwidth: 328 -> 192 B per cell per step for the coupled
//  fluid and D3Q7 scalar, so about 1.7x. The question is what it costs in
//  accuracy, and that does NOT need the layout change: quantising the value on
//  its way INTO the existing 32-bit array reproduces the arithmetic exactly,
//  because memory then holds only FP16-representable numbers. Zero speed-up,
//  full accuracy effect, and it runs on a laptop.
//
//  ON STORE ONLY, deliberately. Loads need no quantisation because what is in
//  memory is already quantised; wrapping them too would round twice and
//  overstate the damage.
//
//  WHY IT MIGHT NOT WORK, so the measurement has something to refute. FP16
//  carries 10 explicit mantissa bits, a relative precision of 4.88e-04. Shifted
//  storage (g_i = f_i - w_i, see core.cuh) is what makes this plausible at all:
//  it spends those bits on the DEVIATION rather than on the w_i ~ 0.03 pedestal.
//  But the physically meaningful part of g_i is the non-equilibrium part that
//  carries the viscous stress, and to order of magnitude
//
//      f_neq / g  ~  cs^2 (tau - 1/2) / L,
//
//  which equals 4.88e-04 at Re ~ 100 independently of L. A naive per-cell
//  reading of that says FP16 cannot see the stress above Re ~ 100 at all.
//  FluidX3D runs far higher Re in FP16, so the naive reading must be too strict
//  -- the storage error is zero-mean and the hydrodynamic limit averages it over
//  cells and steps. Which of those wins is an empirical question and this switch
//  exists to settle it.
//
//  THE INSTRUMENT IS ALREADY CALIBRATED. Raw (unshifted) FP32 storage broke the
//  Poiseuille error x H^2 constant -- 0.336 at H = 16 drifting to 0.498 at
//  H = 32 -- and shifted storage was added to fix it. So that constant is known
//  to be sensitive to storage precision at exactly this level, and it is the
//  first place damage will show.
//
//  This quantises EVERY population array that goes through these functions --
//  fluid, scalar, magnetic, phase field -- which is the honest "what if all
//  storage were 16-bit" question rather than a fluid-only one.
//
//  ======================= THE ANSWER, MEASURED ==============================
//  NO-GO for plain IEEE binary16, and the reason is arithmetic rather than
//  circumstantial. Host suites, FP32 against FP16-emulated storage:
//
//      host_check     0 failures  ->   2
//      host_physics   0 failures  ->  30
//
//  The sharp one is newpaths' shear-decay error, because it is the test the
//  tree already used to justify shifted storage. With the shift ON:
//
//      A = 1e-2    FP32  0.0052153      FP16  0.0071934   (38% off)
//      A = 1e-5    FP32  0.0052145      FP16 -0.965987    (signal gone)
//      A = 1e-5    raw FP32            -0.967929
//
//  FP16-with-shift at A = 1e-5 lands on -0.966 where raw FP32 lands on -0.968.
//  That is the whole finding: FP16 STORAGE COSTS EXACTLY WHAT REMOVING THE
//  SHIFT COSTS. It hands back the precision shifted storage was added to
//  recover.
//
//  The arithmetic, which predicts it: FP16 loses 2^13 = 8192 against FP32, i.e.
//  3.9 decimal digits. The shift buys 1/(3u), because the stored deviation is
//  g/f ~ 3u -- 0.8 digits at u = 0.05, 1.5 at u = 0.01. So at any useful
//  velocity the shift returns about a fifth of what FP16 takes, and the two
//  cancel exactly at 1/(3u) = 8192, u = 4.1e-05. The measurement crosses over
//  at A = 1e-5. Same mechanism, no fitting.
//
//  WHY FLUIDX3D IS NOT REFUTED BY THIS, and it is worth being clear. It does
//  not store plain binary16: it uses a custom 16-bit layout that spends fewer
//  bits on the exponent, since populations have a narrow dynamic range -- worth
//  perhaps 0.9 digits, which still leaves 2.2 short at u = 0.05. The larger
//  difference is the accuracy STANDARD. This tree asserts a Poiseuille
//  error x H^2 constant to three digits, cross-code energy agreement to 5.3e-04
//  and mass drift of exactly 0.000e+00. A code targeting percent-level
//  engineering accuracy passes its own tests in FP16 and would fail every one
//  of those. So the real question is not technical: it is whether a 1.7x is
//  worth lowering the bar, and at the moment the bar is the asset.
//
//  KEPT rather than deleted, because it is a cheap standing instrument: if the
//  shift baseline ever changes, or a custom 16-bit format is tried, this is how
//  to find out whether it helped without touching a memory layout or a GPU.
//  ===========================================================================
#if defined(LBM_STORE16_EMULATE)
LBM_HD LBM_INLINE Real q16(Real v) {
#if defined(__CUDA_ARCH__)
  return Real(__half2float(__float2half(float(v))));
#else
  return Real(float(_Float16(float(v))));
#endif
}
#else
LBM_HD LBM_INLINE Real q16(Real v) { return v; }
#endif

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
  if (Parity == 0) { f[long(i + 1) * N + nb] = q16(a);  f[long(i) * N + self]     = q16(b); }
  else             { f[long(i) * N + nb]     = q16(a);  f[long(i + 1) * N + self] = q16(b); }
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
  f[self] = q16(in[0]);
  for (int i = 1; i < L::Q; i += 2) {
    const long nb = neighbour<L>(x, y, z, i, nx, ny, nz);
    if (Parity == 0) { f[long(i) * N + self]     = q16(in[i]); f[long(i + 1) * N + nb] = q16(in[i + 1]); }
    else             { f[long(i + 1) * N + self] = q16(in[i]); f[long(i) * N + nb]     = q16(in[i + 1]); }
  }
}

template <int Parity, class L = D3Q27>
LBM_HD LBM_INLINE void scatter(Real* f, long N, int x, int y, int z,
                               int nx, int ny, int nz, const Real* in) {
  const long self = node_id(x, y, z, nx, ny);
  f[self] = q16(in[0]);
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
