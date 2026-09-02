#pragma once
//==============================================================================
//  The fluid: per-node update, kernels, and the host-side driver.
//
//  The per-node update is a plain LBM_HD function, not kernel code. The CUDA
//  kernel is three lines of index arithmetic around it, and the host reference
//  driver in hostsim.hpp calls the very same function in a serial loop. So a
//  physics check run on a laptop with no GPU exercises the code the device will
//  execute, rather than a second implementation of it that can drift.
//
//  COUPLING ORDER, AND WHY IT IS NOT AN IMPLEMENTATION DETAIL.
//
//  Two-way coupling must be evaluated SIMULTANEOUSLY. If the fluid is stepped
//  against a magnetic field or a temperature from the previous step, that is a
//  first-order splitting error -- and it does not vanish under refinement. Under
//  diffusive scaling the ratio omega^2 dt / (nu k^2) is independent of N, so it
//  appears as a damping offset that survives every grid refinement. The parent
//  implementation found this the hard way: its shear-Alfven damping error GREW
//  with resolution, 1.55e-2 -> 2.79e-2 -> 3.16e-2, while the phase speed
//  converged cleanly at second order. A non-converging error sitting beside a
//  converging one is the signature.
//
//  So the driver order is: refresh T and B, step the fluid against them, then
//  step T and B against the velocity the fluid just wrote. That costs one extra
//  light pass over the coupled populations per step.
//
//  It is the cheaper of the two mirror images. The alternative -- run the fluid
//  first and give the coupled fields its velocity -- would need a separate pass
//  over 27 populations to recover u, where refreshing T costs 7 and B costs 21.
//  The fluid writes u as a by-product of a gather it was doing anyway.
//==============================================================================
#include "regularized.cuh"
#include "streaming.cuh"

namespace lbm {

//------------------------------------------------------------------------------
// Everything a fluid step needs. Passed to the kernel by value; the pointers it
// does not use are null and are never dereferenced, because the template flags
// that would read them are false.
//------------------------------------------------------------------------------
struct FluidParams {
  Real* f = nullptr;
  const std::uint8_t* flags = nullptr;
  const Real* Bx = nullptr;                  // magnetic field, owned elsewhere
  const Real* By = nullptr;
  const Real* Bz = nullptr;
  Real* ux_out = nullptr;                    // velocity handed to coupled fields
  Real* uy_out = nullptr;
  Real* uz_out = nullptr;
  BodyForce force{};
  int nx = 0, ny = 0, nz = 0;
  Real omega = Real(1), omega_bulk = Real(1);
  Real omega_minus = Real(1);               // TRT only
  bool shifted = false;                     // storage: f_i, or f_i - w_i

  //---- regularised walls; all null unless set_regularized_walls was called ----
  const std::uint8_t*  bc_nrm = nullptr;    // outward normal code, per node
  const std::uint16_t* bc_tag = nullptr;    // index into wall_u
  const std::uint32_t* bc_unk = nullptr;    // which directions streamed from outside
  const std::uint8_t*  bc_ext = nullptr;    // corner rho stencil direction
  const Real*          wall_u = nullptr;    // 3 per state
  Real* bc_rho = nullptr;                   // written by the wall passes
  Real* bc_pi  = nullptr;                   // 6 per node, corners only
  Real  bc_shear_omega = Real(1);           // for the FD corner route
  bool  fd_corners = true;
};

//------------------------------------------------------------------------------
// One node, one step: gather, collide, scatter.
//
// A Solid cell returns immediately. That is not an optimisation -- under
// Esoteric Pull it IS halfway bounce-back. The slot a fluid node emits into
// toward a wall is a slot the wall owns; the wall never touches it, so the
// fluid node reads its own emission back one step later, reversed. Skipping the
// node is the boundary condition, and it is why arbitrary geometry costs
// nothing here beyond one byte per node.
//
// The consequence worth knowing: a population in flight toward a wall spends a
// step in a slot the WALL owns, so a sum over fluid cells alone does not see it.
// Nothing is lost; the macroscopic field is what is missing it.
//------------------------------------------------------------------------------
template <int Parity, int OpKind, int FKind, bool Mhd, bool HasGeometry, bool HasWalls = false>
LBM_HD LBM_INLINE void fluid_node_update(const FluidParams& p, long N, long n) {
  // Short-circuit on a compile-time constant: with HasGeometry false the load
  // is not merely predicted away, it is never emitted. Measured worth 5.6% on a
  // T4 -- see the note in streaming.cuh on why a one-byte-per-node array costs
  // ten times what its byte count suggests.
  const std::uint8_t cell = HasGeometry ? p.flags[n] : std::uint8_t(Fluid);
  if (HasGeometry && cell != Fluid && cell != RegWall) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  Real f[27];
  gather<Parity>(p.f, N, x, y, z, p.nx, p.ny, p.nz, f);

  // A REGULARISED WALL REPLACES EVERY POPULATION AND THEN COLLIDES NORMALLY.
  // It is a fluid node, not a skipped one -- see the note on CellType in
  // streaming.cuh for why that asymmetry with Solid is a live trap.
  //
  // The reconstruction works on RAW populations, so shifted storage is undone
  // around it -- one add and one subtract, on the wall nodes alone.
  if (HasWalls && cell == RegWall) {
    const std::uint8_t code = p.bc_nrm[n];
    if (code != NrmNone) {
      if (p.shifted) for (int i = 0; i < 27; ++i) f[i] += D3Q27::w(i);

      const int tg = int(p.bc_tag[n]);
      const Real ur[3] = {p.wall_u[3 * tg], p.wall_u[3 * tg + 1], p.wall_u[3 * tg + 2]};

      Real F[3] = {Real(0), Real(0), Real(0)};
      if (FKind != ForceNone) force_at<FKind>(p.force, n, F);
      const Real* Fv = (FKind != ForceNone) ? F : nullptr;
      constexpr bool product = (OpKind == 1);   // the CM operator's own equilibrium

      if (code == NrmCorner) {
        // rho was extrapolated along a wall in the corner pass: Eq. (27) needs a
        // single normal and a corner has none.
        const Real rw = p.bc_rho[n];
        if (p.fd_corners && p.bc_pi) reg_apply_with_stress(f, rw, ur, &p.bc_pi[6 * n], product);
        else                         reg_apply(f, rw, ur, p.bc_unk[n], Fv, product);
      } else {
        int nrm[3];  normal_of(code, nrm);
        const Real rw = reg_density(f, nrm, ur);
        reg_apply(f, rw, ur, p.bc_unk[n], Fv, product);
      }

      if (p.shifted) for (int i = 0; i < 27; ++i) f[i] -= D3Q27::w(i);
    }
  }

  Macro m = macroscopic(f, p.shifted);

  Coupling cp;
  force_at<FKind>(p.force, n, cp.F);
  if (FKind != ForceNone) shift_velocity(m, cp.F);
  if (Mhd) { cp.B[0] = p.Bx[n]; cp.B[1] = p.By[n]; cp.B[2] = p.Bz[n]; }

  // The velocity the coupled fields advect with is the one Guo's half-shift has
  // already been applied to, i.e. the physical one.
  if (p.ux_out) { p.ux_out[n] = m.ux; p.uy_out[n] = m.uy; p.uz_out[n] = m.uz; }

  if      (OpKind == 0) collide_bgk_gen<FKind != ForceNone, Mhd>(f, m, p.omega, cp, p.shifted);
  else if (OpKind == 1) collide_cm_gen <FKind != ForceNone, Mhd>(f, m, p.omega, p.omega_bulk, cp, p.shifted);
  else                  collide_trt_gen<FKind != ForceNone, Mhd>(f, m, p.omega, p.omega_minus, cp, p.shifted);

  scatter<Parity>(p.f, N, x, y, z, p.nx, p.ny, p.nz, f);
}

//------------------------------------------------------------------------------
// Macroscopic readout at one node, for diagnostics. Non-fluid cells report rest
// at rho0 rather than whatever their storage happens to hold, so a plot of the
// field does not show wall slots as flow.
//
// GUO'S HALF-FORCE SHIFT BELONGS HERE TOO, and its absence was a real bug.
// The physical velocity of a forced scheme is u = sum_i c_i f_i / rho + F/(2 rho)
// -- that is what the step kernel collides with and what it writes into ux_out --
// so a diagnostic pass that omits it reports a DIFFERENT velocity from the one
// the solver is using, uniformly low by F/(2 rho).
//
// Measured on the Poiseuille channel of host_physics case 1, G = 2.604e-4: the
// two paths differed by exactly 1.302083e-04 = G/2 at every node, to eight
// digits. It is a constant offset, so it is invisible in a profile SHAPE and
// shows up only in an amplitude -- which is why it survived: the fitted
// parabola projects a constant onto the shape function with a coefficient of
// sum(shape)/sum(shape^2) = 0.0196, turning a 0.26% velocity offset into a
// 0.33% amplitude error that sat inside a 0.3%-ish tolerance alongside the
// wall-position error, partly cancelling it.
//
// It also mattered more than a diagnostic normally would, because
// refresh_velocity() runs this pass to seed the velocity a COUPLED field
// advects with, and a penalised body reads that array to recover u* -- which
// subtracts F/(2 rho) from a velocity that never contained it. That is the
// double-counting body.cuh's banner records as diverging in a few hundred
// steps, arriving through the diagnostic rather than through the body.
//------------------------------------------------------------------------------
template <int Parity, int FKind>
LBM_HD LBM_INLINE void macro_node(const FluidParams& p, long N, long n,
                                  Real* rho, Real* ux, Real* uy, Real* uz) {
  const std::uint8_t cell = p.flags[n];
  if (cell != Fluid && cell != RegWall) {
    rho[n] = Real(1); ux[n] = uy[n] = uz[n] = Real(0);
    return;
  }
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real fl[27];
  gather<Parity>(p.f, N, x, y, z, p.nx, p.ny, p.nz, fl);

  // AT A REGULARISED WALL THE STREAMED POPULATIONS ARE NOT A STATE. This pass
  // runs between steps, so the unknown directions hold whatever was last left
  // in those slots -- the step kernel rebuilds them at the START of a step and
  // they are stale by the time anyone reads the field. Their raw moments are
  // therefore meaningless, and reporting them is not a small error: measured on
  // a forced channel it put a uniform 15%-of-u_max slip into the profile, which
  // looks exactly like a boundary condition that does not hold.
  //
  // The velocity there is the IMPOSED one, by definition. The density comes
  // from the same closure Eq. (27) the wall itself uses, so the reported state
  // is the one the wall is enforcing rather than a snapshot of the plumbing.
  if (cell == RegWall && p.bc_nrm) {
    const std::uint8_t code = p.bc_nrm[n];
    const int tg = int(p.bc_tag[n]);
    const Real uw[3] = {p.wall_u[3 * tg], p.wall_u[3 * tg + 1], p.wall_u[3 * tg + 2]};
    if (p.shifted) for (int i = 0; i < 27; ++i) fl[i] += D3Q27::w(i);
    int nrm[3];  normal_of(code, nrm);
    rho[n] = (code == NrmCorner) ? (p.bc_rho ? p.bc_rho[n] : Real(1))
                                 : reg_density(fl, nrm, uw);
    ux[n] = uw[0];  uy[n] = uw[1];  uz[n] = uw[2];
    return;
  }

  Macro m = macroscopic(fl, p.shifted);
  if (FKind != ForceNone) {
    Real F[3];
    force_at<FKind>(p.force, n, F);
    shift_velocity(m, F);
  }
  rho[n] = m.rho; ux[n] = m.ux; uy[n] = m.uy; uz[n] = m.uz;
}

//------------------------------------------------------------------------------
// THE TWO WALL PRE-PASSES, and why they are separate kernels.
//
// A corner cannot evaluate Eq. (27) -- it has no single normal -- so its rho is
// extrapolated from two STRAIGHT-wall nodes along a wall. Those nodes' rho has
// to exist first, which is one fence. And the finite-difference corner route
// reads its NEIGHBOURS' populations to build a velocity gradient, which is a
// race against the step kernel writing them -- so it has to happen while
// nothing is writing populations at all.
//
// Both passes are therefore read-only on the populations and run before the
// step. BOTH ARE SKIPPED ENTIRELY WHEN THERE ARE NO CORNERS: a channel with two
// straight walls closes for rho inside the step kernel and pays nothing here,
// which is the common case and the one Hartmann is.
//
// This is a deliberate departure from the parent, which does the FD gradient
// inside its step kernel. That is a genuine read/write race on a neighbour's
// populations; it survives there because corners are few and the damage is
// small and local, but it is not something to reproduce on a device with
// thousands of concurrent threads.
//------------------------------------------------------------------------------
template <int Parity>
LBM_HD LBM_INLINE void wall_rho_node(const FluidParams& p, long N, long n) {
  if (p.flags[n] != RegWall) return;
  const std::uint8_t code = p.bc_nrm[n];
  if (code == NrmNone || code == NrmCorner) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real f[27];
  gather<Parity>(p.f, N, x, y, z, p.nx, p.ny, p.nz, f);
  if (p.shifted) for (int i = 0; i < 27; ++i) f[i] += D3Q27::w(i);

  int nrm[3];  normal_of(code, nrm);
  const int tg = int(p.bc_tag[n]);
  const Real uw[3] = {p.wall_u[3 * tg], p.wall_u[3 * tg + 1], p.wall_u[3 * tg + 2]};
  p.bc_rho[n] = reg_density(f, nrm, uw);
}

// Velocity at a node, for the corner gradient. AT A WALL NODE IT IS THE IMPOSED
// ONE, not the population sum: a wall's streamed populations still hold their
// unfixed unknown directions at this point in the step, so their moments are
// not a velocity. Reading them anyway is a wrong gradient at exactly the place
// the gradient is the whole boundary condition.
template <int Parity>
LBM_HD LBM_INLINE void wall_vel_at(const FluidParams& p, long N, long m, Real out[3]) {
  if (p.flags[m] == RegWall) {
    const int t = int(p.bc_tag[m]);
    out[0] = p.wall_u[3 * t];  out[1] = p.wall_u[3 * t + 1];  out[2] = p.wall_u[3 * t + 2];
    return;
  }
  int x, y, z;
  coords(m, p.nx, p.ny, x, y, z);
  Real g[27];
  gather<Parity>(p.f, N, x, y, z, p.nx, p.ny, p.nz, g);
  const Macro mm = macroscopic(g, p.shifted);
  out[0] = mm.ux;  out[1] = mm.uy;  out[2] = mm.uz;
}

template <int Parity>
LBM_HD LBM_INLINE void wall_corner_node(const FluidParams& p, long N, long n) {
  if (p.flags[n] != RegWall || p.bc_nrm[n] != NrmCorner) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  const int dirs[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

  // ---- rho, extrapolated along a wall ----
  const std::uint8_t k = p.bc_ext[n];
  if (k == NrmNone) {
    p.bc_rho[n] = Real(1);           // no valid stencil; reported at setup
  } else {
    const int* e = dirs[k - 1];
    const long n1 = node_id(wrap(x + e[0], p.nx), wrap(y + e[1], p.ny),
                            wrap(z + e[2], p.nz), p.nx, p.ny);
    const long n2 = node_id(wrap(x + 2 * e[0], p.nx), wrap(y + 2 * e[1], p.ny),
                            wrap(z + 2 * e[2], p.nz), p.nx, p.ny);
    p.bc_rho[n] = Real(2) * p.bc_rho[n1] - p.bc_rho[n2];
  }

  if (!p.fd_corners || !p.bc_pi) return;

  // ---- Pi from a finite-difference velocity gradient, Eq. (21) ----
  const int tg = int(p.bc_tag[n]);
  const Real uw[3] = {p.wall_u[3 * tg], p.wall_u[3 * tg + 1], p.wall_u[3 * tg + 2]};
  const int ic[3] = {x, y, z};
  const int nn[3] = {p.nx, p.ny, p.nz};
  Real grad[3][3] = {{Real(0),Real(0),Real(0)},
                     {Real(0),Real(0),Real(0)},
                     {Real(0),Real(0),Real(0)}};
  for (int a = 0; a < 3; ++a) {
    if (nn[a] < 3) continue;
    // One-sided INTO the fluid at a domain face; centred otherwise.
    int dir = 0;
    if (ic[a] == 0)             dir =  1;
    else if (ic[a] == nn[a] - 1) dir = -1;
    int c1[3] = {ic[0], ic[1], ic[2]}, c2[3] = {ic[0], ic[1], ic[2]};
    Real u1[3], u2[3];
    if (dir != 0) {
      c1[a] += dir;  c2[a] += 2 * dir;
      wall_vel_at<Parity>(p, N, node_id(c1[0], c1[1], c1[2], p.nx, p.ny), u1);
      wall_vel_at<Parity>(p, N, node_id(c2[0], c2[1], c2[2], p.nx, p.ny), u2);
      for (int b = 0; b < 3; ++b)            // (-3 f0 + 4 f1 - f2) / 2
        grad[a][b] = Real(dir) * (Real(-1.5) * uw[b] + Real(2) * u1[b] - Real(0.5) * u2[b]);
    } else {
      c1[a] = wrap(ic[a] + 1, nn[a]);
      c2[a] = wrap(ic[a] - 1, nn[a]);
      wall_vel_at<Parity>(p, N, node_id(c1[0], c1[1], c1[2], p.nx, p.ny), u1);
      wall_vel_at<Parity>(p, N, node_id(c2[0], c2[1], c2[2], p.nx, p.ny), u2);
      for (int b = 0; b < 3; ++b) grad[a][b] = Real(0.5) * (u1[b] - u2[b]);
    }
  }
  reg_stress_from_gradient(p.bc_rho[n], p.bc_shear_omega, grad, &p.bc_pi[6 * n]);
}

//==============================================================================
//  CUDA kernels -- thin wrappers around the functions above.
//==============================================================================
#if defined(__CUDACC__)

template <int Parity, int OpKind, int FKind, bool Mhd, bool HasGeometry, bool HasWalls>
__global__ void fluid_kernel(FluidParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  fluid_node_update<Parity, OpKind, FKind, Mhd, HasGeometry, HasWalls>(p, N, n);
}

template <int Parity>
__global__ void wall_rho_kernel(FluidParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  wall_rho_node<Parity>(p, N, n);
}

template <int Parity>
__global__ void wall_corner_kernel(FluidParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  wall_corner_node<Parity>(p, N, n);
}

template <int Parity, int FKind>
__global__ void compute_macro(FluidParams p, Real* __restrict__ rho,
                              Real* __restrict__ ux, Real* __restrict__ uy,
                              Real* __restrict__ uz) {
  const long N = long(p.nx) * p.ny * p.nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  macro_node<Parity, FKind>(p, N, n, rho, ux, uy, uz);
}

//------------------------------------------------------------------------------
// Initialise every slot exactly once.
//
// `init_scatter` writes the value that `gather` would read back, so applying it
// over all nodes at parity 0 leaves a consistent lattice. Doing this any other
// way (writing f[i*N+n] directly) desynchronises the paired slots and the first
// step transports garbage.
//
// SOLID AND EXCLUDED CELLS ARE SEEDED AT REST, NOT LEFT EMPTY. A bounce-back
// cell is a real storage cell holding in-transit populations; starting it at
// zero makes the wall layer soak mass out of the fluid and leaves the bulk
// density low by O(area/volume) = O(1/H) -- a first-order error in a quantity
// the Poiseuille amplitude depends on, since mu = rho nu.
//------------------------------------------------------------------------------
template <class Init>
__global__ void initialise(Real* __restrict__ f, const std::uint8_t* __restrict__ flags,
                           int nx, int ny, int nz, bool shifted, Init init) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int x, y, z;
  coords(n, nx, ny, x, y, z);

  Macro m = (flags[n] == Fluid) ? init(x, y, z)
                                : Macro{Real(1), Real(0), Real(0), Real(0)};
  // A seed comes from a caller as a DENSITY, whichever storage is in use, so
  // `dens` is derived here rather than demanded. A shifted run that was handed
  // a raw seed would start with every population off by w_i, which is a
  // pressure of 1 in lattice units -- not subtle, but silent.
  m.dens = shifted ? (m.rho - Real(1)) : m.rho;
  Real fl[27];
  for (int i = 0; i < 27; ++i) fl[i] = feq_of(i, m, shifted);
  init_scatter<0>(f, N, x, y, z, nx, ny, nz, fl);   // NOT scatter -- see above
}

//------------------------------------------------------------------------------
// Total population, as a small array of partial sums.
//
// The sum runs over EVERY slot, wall slots included, because that -- not a sum
// over fluid cells -- is what Esoteric Pull conserves exactly: collision keeps
// the local sum, and streaming only permutes values between slots each of which
// has exactly one writer per step. A geometry bug that visited a node twice, or
// skipped one it should not have, shows up here and nowhere else.
//
// Two details that matter at 10^9 nodes:
//
//   * the accumulator is double even in an FP32 build. Summing 2.9e10 FP32
//     values pairwise-naively loses the low half of the mantissa, and the drift
//     this check exists to detect is smaller than that.
//   * the result comes back as G partial sums summed on the host, not through
//     atomicAdd on a double. Same answer, no compute-capability floor, and the
//     order of accumulation is deterministic run to run.
//------------------------------------------------------------------------------
__global__ void reduce_population(const Real* __restrict__ f, long M,
                                  double* __restrict__ partial) {
  extern __shared__ double sm[];
  const long stride = long(blockDim.x) * gridDim.x;
  double acc = 0;
  for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < M; i += stride)
    acc += double(f[i]);
  sm[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = sm[0];
}

//==============================================================================
//  Host-side driver.
//==============================================================================
class Solver {
 public:
  Solver(int nx, int ny, int nz, Op op, Real nu, Real omega_bulk = Real(1))
      : nx_(nx), ny_(ny), nz_(nz), op_(op),
        omega_(omega_from_viscosity(nu)), omega_bulk_(omega_bulk),
        omega_minus_(omega_minus_for(omega_from_viscosity(nu), magic_3_16)) {
    N_ = long(nx) * ny * nz;
    LBM_CUDA_CHECK(cudaMalloc(&f_, sizeof(Real) * 27 * N_));
    LBM_CUDA_CHECK(cudaMalloc(&flags_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMemset(flags_, Fluid, sizeof(std::uint8_t) * N_));
  }
  ~Solver() {
    cudaFree(f_); cudaFree(flags_);
    cudaFree(ux_); cudaFree(uy_); cudaFree(uz_);
    cudaFree(bc_nrm_); cudaFree(bc_ext_); cudaFree(bc_tag_); cudaFree(bc_unk_);
    cudaFree(bc_rho_); cudaFree(bc_pi_); cudaFree(wall_u_);
  }

  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  //--------------------------------------------------------------------------
  // Geometry. One byte per node, in the same linear order as everything else:
  // flags[node_id(x,y,z,nx,ny)]. Call before initialise_with, so solid cells
  // are seeded at rest rather than with the initial condition.
  //--------------------------------------------------------------------------
  void set_geometry(const std::vector<std::uint8_t>& flags) {
    if (long(flags.size()) != N_) {
      std::fprintf(stderr, "set_geometry: %zu flags for %ld nodes\n", flags.size(), N_);
      std::exit(1);
    }
    LBM_CUDA_CHECK(cudaMemcpy(flags_, flags.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;
  }

  void set_force(const BodyForce& b, int kind) { force_ = b; fkind_ = kind; }

  //--------------------------------------------------------------------------
  // Regularised velocity walls. `spec` is one RegWallSpec per node; a node with
  // normal == NrmNone is left alone, any other value marks it RegWall and
  // imposes that velocity AT the node. See regularized.cuh.
  //
  // CALL set_geometry FIRST if the run has any solid cells: the unknown-direction
  // mask is built from the geometry, and a wall whose neighbours are not yet
  // marked gets the wrong set of unknowns -- which does not fail, it just
  // reconstructs against directions that did stream.
  //
  // THE CHANNEL IS H-1 WIDE, NOT H. That is the whole difference from
  // bounce-back and the commonest way to get a Reynolds number wrong here.
  //--------------------------------------------------------------------------
  void set_regularized_walls(const std::vector<RegWallSpec>& spec) {
    if (long(spec.size()) != N_) {
      std::fprintf(stderr, "set_regularized_walls: %zu specs for %ld nodes\n",
                   spec.size(), N_);
      std::exit(1);
    }
    std::vector<std::uint8_t> geo(std::size_t(N_), std::uint8_t(Fluid));
    if (has_geometry_)
      LBM_CUDA_CHECK(cudaMemcpy(geo.data(), flags_, sizeof(std::uint8_t) * N_,
                                cudaMemcpyDeviceToHost));

    std::vector<std::uint8_t>  nrm, ext;
    std::vector<std::uint16_t> tag;
    std::vector<std::uint32_t> unk;
    std::vector<Real> table;
    n_walls_ = build_reg_walls(spec, geo, nx_, ny_, nz_, nrm, tag, unk, ext,
                               table, has_corners_);
    if (n_walls_ == 0) return;

    for (long n = 0; n < N_; ++n)
      if (nrm[std::size_t(n)] != NrmNone) geo[std::size_t(n)] = RegWall;
    LBM_CUDA_CHECK(cudaMemcpy(flags_, geo.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;
    has_walls_ = true;

    LBM_CUDA_CHECK(cudaMalloc(&bc_nrm_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&bc_ext_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&bc_tag_, sizeof(std::uint16_t) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&bc_unk_, sizeof(std::uint32_t) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&bc_rho_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&wall_u_, sizeof(Real) * table.size()));
    LBM_CUDA_CHECK(cudaMemcpy(bc_nrm_, nrm.data(), sizeof(std::uint8_t) * N_, cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(bc_ext_, ext.data(), sizeof(std::uint8_t) * N_, cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(bc_tag_, tag.data(), sizeof(std::uint16_t) * N_, cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(bc_unk_, unk.data(), sizeof(std::uint32_t) * N_, cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wall_u_, table.data(), sizeof(Real) * table.size(),
                              cudaMemcpyHostToDevice));
    // Six Reals per node, and ONLY when there are corners to need them: on a
    // channel that is 24 bytes/node of nothing.
    if (has_corners_ && fd_corners_)
      LBM_CUDA_CHECK(cudaMalloc(&bc_pi_, sizeof(Real) * 6 * N_));
  }

  // Corners: how Pi^(1) is obtained there. The finite-difference route is what
  // Latt et al. use (Sec. V) and the parent measures ~35% lower error with it on
  // the Re = 1000 cavity; the local closure is cheaper and operator-agnostic but
  // a corner offers few streamed directions to build a stress from.
  void set_fd_corners(bool on) { fd_corners_ = on; }
  long wall_count() const { return n_walls_; }

  // TRT's free rate, set through the magic parameter rather than directly --
  // Lambda is the quantity with a meaning (3/16 puts the bounce-back wall
  // halfway) and omega_minus is the quantity the kernel wants. Ignored by
  // every other operator. Defaults to 3/16, so a TRT run that says nothing
  // gets the wall-consistent value rather than BGK by accident.
  void set_magic(Real lambda) { omega_minus_ = omega_minus_for(omega_, lambda); }

  // Store f_i - w_i instead of f_i. Call BEFORE initialise_with -- it changes
  // what the arrays MEAN, so switching it on a seeded lattice reinterprets
  // every value. See the banner in core.cuh for what it buys in FP32.
  void set_shifted(bool on) { shifted_ = on; }
  bool shifted() const { return shifted_; }
  Real omega_minus() const { return omega_minus_; }
  Real magic() const { return magic_parameter(omega_, omega_minus_); }

  // Device pointers owned by a MagneticSolver. Switches the fluid to the MHD
  // equilibrium; the Maxwell stress then enters through the second moment
  // rather than as a body force, which is what keeps it second-order accurate.
  void couple_magnetic(const Real* Bx, const Real* By, const Real* Bz) {
    Bx_ = Bx; By_ = By; Bz_ = Bz; mhd_ = true;
    enable_velocity_output();
  }

  // Allocate the velocity field the coupled solvers advect with. Not allocated
  // by default: 12 bytes per node against 108 for the populations is 11% more
  // traffic, which an uncoupled run should not pay.
  void enable_velocity_output() {
    if (ux_) return;
    LBM_CUDA_CHECK(cudaMalloc(&ux_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&uy_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&uz_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(ux_, 0, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(uy_, 0, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(uz_, 0, sizeof(Real) * N_));
  }
  const Real* ux_device() const { return ux_; }
  const Real* uy_device() const { return uy_; }
  const Real* uz_device() const { return uz_; }

  template <class Init>
  void initialise_with(Init init) {
    const int B = 128;
    initialise<<<int((N_ + B - 1) / B), B>>>(f_, flags_, nx_, ny_, nz_, shifted_, init);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    t_ = 0;
    if (ux_) refresh_velocity();
  }

  void step() {
    if (t_ % 2 == 0) launch_op<0>();
    else             launch_op<1>();
    ++t_;
  }

  // Write ux/uy/uz without advancing. Only needed before the first step, so a
  // coupled field advects with the initial velocity rather than with zero.
  void refresh_velocity() {
    if (!ux_) return;
    Real* dr = nullptr;
    LBM_CUDA_CHECK(cudaMalloc(&dr, sizeof(Real) * N_));
    macro_launch(dr, ux_, uy_, uz_);
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dr);
  }

  // rho/u on the host. Allocates its own device scratch: this is a diagnostic
  // path, called every few thousand steps, so the allocation is not worth
  // keeping resident.
  void macroscopic_to_host(std::vector<Real>& rho, std::vector<Real>& ux,
                           std::vector<Real>& uy, std::vector<Real>& uz) {
    Real *dr, *dx, *dy, *dz;
    LBM_CUDA_CHECK(cudaMalloc(&dr, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&dx, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&dy, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&dz, sizeof(Real) * N_));
    macro_launch(dr, dx, dy, dz);
    rho.resize(N_); ux.resize(N_); uy.resize(N_); uz.resize(N_);
    LBM_CUDA_CHECK(cudaMemcpy(rho.data(), dr, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(ux.data(),  dx, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(uy.data(),  dy, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(uz.data(),  dz, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    cudaFree(dr); cudaFree(dx); cudaFree(dy); cudaFree(dz);
  }

  // Summed over every slot, including those owned by solid cells -- see the
  // kernel above. Costs one full pass over the lattice, so it belongs either
  // side of a timed loop, never inside one.
  // NOTE WHAT THIS MEANS UNDER SHIFTED STORAGE: the sum is of g_i = f_i - w_i,
  // so it is (total mass) - (number of slots), and it is near zero rather than
  // near N. Still exactly conserved, which is what the check is for, but a
  // caller comparing it against a node count will be confused.
  double total_mass() {
    const int B = 256, G = 1024;
    double* d = nullptr;
    LBM_CUDA_CHECK(cudaMalloc(&d, sizeof(double) * G));
    reduce_population<<<G, B, sizeof(double) * B>>>(f_, 27 * N_, d);
    LBM_CUDA_CHECK(cudaGetLastError());
    std::vector<double> h(G);
    LBM_CUDA_CHECK(cudaMemcpy(h.data(), d, sizeof(double) * G, cudaMemcpyDeviceToHost));
    cudaFree(d);
    double s = 0;
    for (double v : h) s += v;
    return s;
  }

  long nodes() const { return N_; }
  std::size_t timestep() const { return t_; }
  Real omega() const { return omega_; }

 private:
  // The macro pass has to know the force, and which KIND it is, for the same
  // reason the step does -- see macro_node. Dispatched so an unforced run emits
  // no force code at all.
  void macro_launch(Real* dr, Real* dx, Real* dy, Real* dz) {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) macro_force<0>(G, B, dr, dx, dy, dz);
    else             macro_force<1>(G, B, dr, dx, dy, dz);
    LBM_CUDA_CHECK(cudaGetLastError());
  }
  template <int P>
  void macro_force(int G, int B, Real* dr, Real* dx, Real* dy, Real* dz) {
    if (fkind_ == ForceUniform)
      compute_macro<P, ForceUniform><<<G, B>>>(params(), dr, dx, dy, dz);
    else if (fkind_ == ForceBoussinesq)
      compute_macro<P, ForceBoussinesq><<<G, B>>>(params(), dr, dx, dy, dz);
    else if (fkind_ == ForceField)
      compute_macro<P, ForceField><<<G, B>>>(params(), dr, dx, dy, dz);
    else
      compute_macro<P, ForceNone><<<G, B>>>(params(), dr, dx, dy, dz);
  }

  FluidParams params() {
    FluidParams p;
    p.f = f_; p.flags = flags_;
    p.Bx = Bx_; p.By = By_; p.Bz = Bz_;
    p.ux_out = ux_; p.uy_out = uy_; p.uz_out = uz_;
    p.force = force_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_; p.omega_bulk = omega_bulk_;
    p.omega_minus = omega_minus_;
    p.shifted = shifted_;
    p.bc_nrm = bc_nrm_;  p.bc_tag = bc_tag_;  p.bc_unk = bc_unk_;
    p.bc_ext = bc_ext_;  p.wall_u = wall_u_;
    p.bc_rho = bc_rho_;  p.bc_pi = bc_pi_;
    // THE SHEAR RATE, not just any rate: TRT's is omega_plus. See
    // reg_stress_from_gradient.
    p.bc_shear_omega = omega_;
    p.fd_corners = fd_corners_;
    return p;
  }

  //--------------------------------------------------------------------------
  // Dispatch. Nested so that only the combinations actually used are ever
  // instantiated -- MHD with buoyancy, for instance, is never launched and
  // therefore never compiled.
  //--------------------------------------------------------------------------
  template <int P> void launch_op() {
    if      (op_ == Op::BGK) launch_force<P, 0>();
    else if (op_ == Op::TRT) launch_force<P, 2>();
    else                     launch_force<P, 1>();
  }
  template <int P, int O> void launch_force() {
    if (mhd_) {
      if (fkind_ == ForceUniform) run<P, O, ForceUniform, true>();
      else                        run<P, O, ForceNone,    true>();
    } else if (fkind_ == ForceUniform) {
      run<P, O, ForceUniform, false>();
    } else if (fkind_ == ForceBoussinesq) {
      run<P, O, ForceBoussinesq, false>();
    } else if (fkind_ == ForceField) {
      run<P, O, ForceField, false>();
    } else {
      run<P, O, ForceNone, false>();
    }
  }
  template <int P, int O, int F, bool M> void run() {
    if (has_walls_)         launch<P, O, F, M, true, true>();
    else if (has_geometry_) launch<P, O, F, M, true, false>();
    else                    launch<P, O, F, M, false, false>();
  }
  template <int P, int O, int F, bool M, bool G, bool W> void launch() {
    const int B = 128, Gr = int((N_ + B - 1) / B);
    // The wall pre-passes exist only for corners -- a straight wall closes for
    // rho inside the step kernel. See wall_rho_kernel.
    if (W && has_corners_) {
      wall_rho_kernel<P><<<Gr, B>>>(params(), N_);
      wall_corner_kernel<P><<<Gr, B>>>(params(), N_);
    }
    fluid_kernel<P, O, F, M, G, W><<<Gr, B>>>(params(), N_);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  int nx_, ny_, nz_;
  long N_;
  Op op_;
  Real omega_, omega_bulk_, omega_minus_;
  Real* f_ = nullptr;
  std::uint8_t* flags_ = nullptr;
  Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  const Real *Bx_ = nullptr, *By_ = nullptr, *Bz_ = nullptr;
  BodyForce force_{};
  int fkind_ = ForceNone;
  bool mhd_ = false;
  bool has_geometry_ = false;
  bool shifted_ = false;
  std::uint8_t*  bc_nrm_ = nullptr;
  std::uint8_t*  bc_ext_ = nullptr;
  std::uint16_t* bc_tag_ = nullptr;
  std::uint32_t* bc_unk_ = nullptr;
  Real* bc_rho_ = nullptr;
  Real* bc_pi_  = nullptr;
  Real* wall_u_ = nullptr;
  long n_walls_ = 0;
  bool has_walls_ = false;
  bool has_corners_ = false;
  bool fd_corners_ = true;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
