#pragma once
//==============================================================================
//  Passive scalar on D3Q7 -- temperature, or any advected-diffused quantity.
//
//  Structurally different from the fluid in one way: the velocity is an INPUT.
//  The scalar carries only its zeroth moment and is advected by whatever field
//  the fluid hands it, which is why it has its own solver rather than another
//  collision operator plugged into the fluid one.
//
//  BOUNDARY CONDITIONS, as alternative collisions on marked cells:
//
//    adiabatic  h_i^out = h_opp(i)^in                              (zero flux)
//    Dirichlet  h_i^out = -h_opp(i)^in + 2 w_i (T_wall - T_ref)    (anti-bounce-back)
//
//  The Dirichlet form carries T_ref because the arrays hold h = g - w_i T_ref;
//  since w_i == w_opp(i), the reference simply shifts the target value.
//
//  WHERE THE TWO WALLS PUT THEIR PLANE, which is the thing most easily got
//  wrong: anti-bounce-back places T_wall half-way between the Dirichlet node and
//  its fluid neighbour, exactly as halfway bounce-back places the no-slip plane.
//  So a conduction problem between two Dirichlet layers has its exact linear
//  solution measured between the PLANES, not between the nodes, and a slab of H
//  fluid nodes spans a gap of H + 1 lattice units. Getting this wrong shows up
//  as an O(1/H) error that looks like a convergence problem and is not one.
//
//  NOT INCLUDED, and absent rather than untested: the open (outflow) boundary
//  and Dellar's moment condition. Outflow needs a donor map and a second kernel
//  after a fence -- reading a donor inside the main kernel is a genuine race
//  under Esoteric Pull, because the two slots a node reads are the two it
//  writes. That is a self-contained piece of work and it is not here.
//==============================================================================
#include "streaming.cuh"

namespace lbm {

using ScalarLattice = D3Q7;

struct ScalarParams {
  Real* h = nullptr;
  const std::uint8_t* flags = nullptr;
  const Real* wall = nullptr;                // Dirichlet values, per node
  const Real* ux = nullptr;                  // advecting velocity, owned by the fluid
  const Real* uy = nullptr;
  const Real* uz = nullptr;
  Real* T_out = nullptr;                     // the concentration field
  // Donor node for every ScalarOutflow cell: the interior cell whose value it
  // copies. Built once on the host at set_geometry; see build_donors.
  const long* donor = nullptr;
  int nx = 0, ny = 0, nz = 0;
  Real omega = Real(1), T_ref = Real(0);
  // Runtime rather than a template parameter, unlike HasGeometry/HasOutflow.
  // Those gate a memory STREAM, which is what a bandwidth-bound kernel pays
  // for; this gates a branch that every thread in the grid takes the same way,
  // which costs nothing. Another template axis would double eight kernels.
  bool regularised = false;
};

//------------------------------------------------------------------------------
// One node, one step.
//
// An adiabatic cell returns immediately, for the same reason a Solid fluid cell
// does: bounce-back is the identity on Esoteric Pull's storage. A Dirichlet cell
// cannot be skipped -- anti-bounce-back flips a sign and adds a source -- but it
// still writes back into the same two slots it read, so the in-place scheme is
// undisturbed either way.
//------------------------------------------------------------------------------
template <int Parity, bool Advected, bool HasGeometry, bool HasOutflow = false>
LBM_HD LBM_INLINE void scalar_node_update(const ScalarParams& p, long N, long n) {
  // With HasGeometry false every cell is bulk and the flags load is never
  // emitted -- see solver.cuh. `fl` is then a compile-time constant.
  const std::uint8_t fl = HasGeometry ? p.flags[n] : std::uint8_t(ScalarBulk);
  if (fl == ScalarExcluded || fl == ScalarAdiabatic) return;
  // Handled by the second pass, after the kernel boundary. See streaming.cuh.
  if (HasOutflow && fl == ScalarOutflow) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  constexpr int Q = ScalarLattice::Q;
  Real h[Q];
  gather<Parity, ScalarLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);

  if (fl == ScalarDirichlet) {
    const Real dTw = p.wall[n] - p.T_ref;
    Real out[Q];
    for (int i = 0; i < Q; ++i)
      out[i] = -h[opp(i)] + Real(2) * ScalarLattice::w(i) * dTw;
    scatter<Parity, ScalarLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, out);
    // The donor of an outflow node may be a Dirichlet cell's neighbour but is
    // never the Dirichlet cell itself, so this write is for completeness of the
    // field rather than for the second pass.
    if (HasOutflow) p.T_out[n] = p.wall[n];
    return;
  }

  const Real dT = scalar_deviation<ScalarLattice>(h);
  Real vx = Real(0), vy = Real(0), vz = Real(0);
  if (Advected) { vx = p.ux[n]; vy = p.uy[n]; vz = p.uz[n]; }
  if (p.regularised)
    collide_scalar_regularised(h, dT, p.T_ref, vx, vy, vz, p.omega);
  else
    collide_scalar<ScalarLattice>(h, dT, p.T_ref, vx, vy, vz, p.omega);
  scatter<Parity, ScalarLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);

  // ONE EXTRA STORE, AND ONLY WHERE IT IS NEEDED. The second pass reads the
  // field at its donor, so the main pass has to publish it -- but a closed
  // problem should not pay 4 bytes per node for a boundary it does not have,
  // so the write is behind a template flag and is not emitted at all when no
  // cell is marked outflow. This is the same argument the flags load makes in
  // streaming.cuh, and the same reason it is a template parameter.
  if (HasOutflow) p.T_out[n] = p.T_ref + dT;
}

//------------------------------------------------------------------------------
// PASS 2. Outflow, after the kernel boundary.
//
// Equilibrium extrapolation: the DONOR's concentration, the node's OWN
// velocity, and no non-equilibrium part. Every population is overwritten, so
// whatever streamed in is irrelevant and the inward-pointing directions carry
// the interior's value back into the domain.
//
// Discarding the non-equilibrium part slightly damps the diffusive flux at the
// exit. That is negligible when the exit is advection-dominated, which is the
// only situation an open boundary belongs in, and it is a known approximation
// rather than an accident.
//
// A node whose donor is itself is INERT: it was found at setup to have no bulk
// neighbour, it was counted and reported there, and it keeps whatever streamed
// into it -- i.e. it falls back to bounce-back. Silently reading a neighbour's
// garbage would be the alternative.
//------------------------------------------------------------------------------
template <int Parity, bool Advected>
LBM_HD LBM_INLINE void scalar_outflow_node(const ScalarParams& p, long N, long n) {
  if (p.flags[n] != ScalarOutflow) return;
  const long src = p.donor[n];
  if (src == n) return;                       // degenerate, reported at setup

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  const Real dT = p.T_out[src] - p.T_ref;
  Real vx = Real(0), vy = Real(0), vz = Real(0);
  if (Advected) { vx = p.ux[n]; vy = p.uy[n]; vz = p.uz[n]; }

  constexpr int Q = ScalarLattice::Q;
  Real g[Q];
  for (int i = 0; i < Q; ++i)
    g[i] = scalar_eq<ScalarLattice>(i, dT, p.T_ref, vx, vy, vz);
  scatter<Parity, ScalarLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, g);
  p.T_out[n] = p.T_ref + dT;
}

//------------------------------------------------------------------------------
// The field at one node, without advancing anything.
//
// This is the pass that makes the coupling simultaneous rather than lagged: the
// fluid must collide against T at the SAME time level, and the step kernel above
// cannot supply it, because the value it computes is consumed and overwritten in
// the same launch. Seven reads per node.
//------------------------------------------------------------------------------
template <int Parity, bool HasGeometry>
LBM_HD LBM_INLINE void scalar_field_node(const ScalarParams& p, long N, long n) {
  const std::uint8_t fl = HasGeometry ? p.flags[n] : std::uint8_t(ScalarBulk);
  if (fl == ScalarExcluded) return;
  if (fl == ScalarDirichlet) { p.T_out[n] = p.wall[n]; return; }
  // An outflow node IS an on-node fluid cell holding a real concentration --
  // unlike an adiabatic one, which is a ghost -- so it reports its populations
  // like any other. That is the whole reason to prefer it where an on-node
  // zero-gradient condition is what is actually wanted.

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real h[ScalarLattice::Q];
  gather<Parity, ScalarLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);
  p.T_out[n] = p.T_ref + scalar_deviation<ScalarLattice>(h);
}

//------------------------------------------------------------------------------
// Donor for every ScalarOutflow node: the interior cell whose value it copies.
//
// The OUTWARD directions are the axes whose neighbour is outside the field, and
// the donor is the neighbour one step INWARD along all of them at once -- so a
// face node takes its axis neighbour, an edge node the diagonal and a corner
// node the body diagonal. Without that, a box edge has no purely axial interior
// neighbour at all and every edge and corner of an open box would be inert.
//
// A donor must be ScalarBulk, NEVER another outflow node: the second pass reads
// the field at the donor, and reading it at a node the same pass writes would
// put the race straight back. A node with no bulk neighbour is left pointing at
// itself -- the condition is then a no-op there and it falls back to
// bounce-back -- and such nodes are counted and reported rather than silently
// reading whatever is next door.
//
// Plain host code, run once at set_geometry, shared by the CUDA and host
// drivers. Returns the number of outflow nodes found; `degenerate` comes back
// with how many of them are inert.
//------------------------------------------------------------------------------
inline long build_scalar_donors(const std::vector<std::uint8_t>& flags,
                                int nx, int ny, int nz,
                                std::vector<long>& donor, long& degenerate) {
  const long N = long(nx) * ny * nz;
  donor.resize(std::size_t(N));
  for (long n = 0; n < N; ++n) donor[std::size_t(n)] = n;

  auto flag_at = [&](int x, int y, int z) -> std::uint8_t {
    // Periodic indexing, matching neighbour() in streaming.cuh: "outside the
    // field" is a flag, not an index, so a periodic axis simply wraps.
    return flags[std::size_t(node_id(wrap(x, nx), wrap(y, ny), wrap(z, nz), nx, ny))];
  };

  const int dirs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  long nout = 0;
  degenerate = 0;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const long n = node_id(x, y, z, nx, ny);
        if (flags[std::size_t(n)] != ScalarOutflow) continue;
        ++nout;
        int ix = 0, iy = 0, iz = 0;
        for (int k = 0; k < 6; ++k)
          if (flag_at(x + dirs[k][0], y + dirs[k][1], z + dirs[k][2]) == ScalarExcluded) {
            ix -= dirs[k][0];  iy -= dirs[k][1];  iz -= dirs[k][2];
          }
        long best = n;
        if ((ix || iy || iz) && flag_at(x + ix, y + iy, z + iz) == ScalarBulk)
          best = node_id(wrap(x + ix, nx), wrap(y + iy, ny), wrap(z + iz, nz), nx, ny);
        if (best == n)                        // fall back to any bulk neighbour
          for (int k = 0; k < 6; ++k)
            if (flag_at(x + dirs[k][0], y + dirs[k][1], z + dirs[k][2]) == ScalarBulk) {
              best = node_id(wrap(x + dirs[k][0], nx), wrap(y + dirs[k][1], ny),
                             wrap(z + dirs[k][2], nz), nx, ny);
              break;
            }
        donor[std::size_t(n)] = best;
        if (best == n) ++degenerate;
      }
  if (degenerate)
    std::fprintf(stderr,
                 "  [scalar] %ld of %ld outflow node(s) have no bulk neighbour "
                 "and are inert\n", degenerate, nout);
  return nout;
}

#if defined(__CUDACC__)

template <int Parity, bool Advected, bool HasGeometry, bool HasOutflow>
__global__ void scalar_kernel(ScalarParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_node_update<Parity, Advected, HasGeometry, HasOutflow>(p, N, n);
}

template <int Parity, bool Advected>
__global__ void scalar_outflow_kernel(ScalarParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_outflow_node<Parity, Advected>(p, N, n);
}

template <int Parity, bool HasGeometry>
__global__ void scalar_field_kernel(ScalarParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_field_node<Parity, HasGeometry>(p, N, n);
}

//------------------------------------------------------------------------------
// Seed at equilibrium with zero velocity. A Dirichlet node is seeded at its own
// wall value, so the boundary is consistent from step zero rather than relaxing
// into place over the first few steps.
//------------------------------------------------------------------------------
template <class Init>
__global__ void scalar_initialise(Real* __restrict__ h,
                                  const std::uint8_t* __restrict__ flags,
                                  const Real* __restrict__ wall,
                                  int nx, int ny, int nz, Real T_ref, Init init) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int x, y, z;
  coords(n, nx, ny, x, y, z);
  const Real T = (flags[n] == ScalarDirichlet) ? wall[n] : init(x, y, z);
  Real g[ScalarLattice::Q];
  for (int i = 0; i < ScalarLattice::Q; ++i)
    g[i] = scalar_eq<ScalarLattice>(i, T - T_ref, T_ref, Real(0), Real(0), Real(0));
  init_scatter<0, ScalarLattice>(h, N, x, y, z, nx, ny, nz, g);
}

//==============================================================================
//  Host-side driver.
//==============================================================================
class ScalarSolver {
 public:
  ScalarSolver(int nx, int ny, int nz, Real diffusivity, Real T_ref = Real(0),
               ScalarOp op = ScalarOp::BGK)
      : nx_(nx), ny_(ny), nz_(nz), T_ref_(T_ref), op_(op) {
    omega_ = omega_from_diffusivity<ScalarLattice>(diffusivity);
    N_ = long(nx) * ny * nz;
    LBM_CUDA_CHECK(cudaMalloc(&h_, sizeof(Real) * ScalarLattice::Q * N_));
    LBM_CUDA_CHECK(cudaMalloc(&flags_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMemset(flags_, ScalarBulk, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&wall_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(wall_, 0, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&T_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(T_, 0, sizeof(Real) * N_));
  }
  ~ScalarSolver() {
    cudaFree(h_); cudaFree(flags_); cudaFree(wall_); cudaFree(T_);
    cudaFree(donor_);
  }

  ScalarSolver(const ScalarSolver&) = delete;
  ScalarSolver& operator=(const ScalarSolver&) = delete;

  // Cell roles and, for Dirichlet cells, the value each one holds.
  void set_geometry(const std::vector<std::uint8_t>& flags,
                    const std::vector<Real>& wall) {
    LBM_CUDA_CHECK(cudaMemcpy(flags_, flags.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wall_, wall.data(), sizeof(Real) * N_,
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;

    std::vector<long> donor;
    long degenerate = 0;
    const long nout = build_scalar_donors(flags, nx_, ny_, nz_, donor, degenerate);
    has_outflow_ = nout > 0;
    if (has_outflow_) {
      if (!donor_) LBM_CUDA_CHECK(cudaMalloc(&donor_, sizeof(long) * N_));
      LBM_CUDA_CHECK(cudaMemcpy(donor_, donor.data(), sizeof(long) * N_,
                                cudaMemcpyHostToDevice));
    }
  }

  // Device pointers owned by the fluid solver. Without this the scalar diffuses
  // but does not advect, which is a legitimate mode (pure conduction).
  void advect_with(const Real* ux, const Real* uy, const Real* uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  template <class Init>
  void initialise_with(Init init) {
    const int B = 128;
    scalar_initialise<<<int((N_ + B - 1) / B), B>>>(h_, flags_, wall_,
                                                    nx_, ny_, nz_, T_ref_, init);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    t_ = 0;
    compute_field();
  }

  void step() {
    const int B = 128, G = int((N_ + B - 1) / B);
    // The outflow pass must see the main pass's field writes. A kernel boundary
    // on the same stream IS that fence, so no explicit synchronise is needed --
    // and the pass is not launched at all without an open boundary.
    if (t_ % 2 == 0) {
      launch_advect<0>(G, B);
      if (has_outflow_) launch_outflow<0>(G, B);
    } else {
      launch_advect<1>(G, B);
      if (has_outflow_) launch_outflow<1>(G, B);
    }
    LBM_CUDA_CHECK(cudaGetLastError());
    ++t_;
  }

  // Refresh T on the device. Call before the fluid step in a coupled run --
  // see the coupling-order note in solver.cuh.
  void compute_field() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) {
      if (has_geometry_) scalar_field_kernel<0, true><<<G, B>>>(params(), N_);
      else               scalar_field_kernel<0, false><<<G, B>>>(params(), N_);
    } else {
      if (has_geometry_) scalar_field_kernel<1, true><<<G, B>>>(params(), N_);
      else               scalar_field_kernel<1, false><<<G, B>>>(params(), N_);
    }
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  void field_to_host(std::vector<Real>& T) {
    compute_field();
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    T.resize(N_);
    LBM_CUDA_CHECK(cudaMemcpy(T.data(), T_, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
  }

  const Real* field_device() const { return T_; }
  Real omega() const { return omega_; }
  Real diffusivity() const { return diffusivity_from_omega<ScalarLattice>(omega_); }
  std::size_t timestep() const { return t_; }

 private:
  template <int P> void launch_advect(int G, int B) {
    if (ux_) launch_geom<P, true>(G, B);
    else     launch_geom<P, false>(G, B);
  }
  template <int P, bool A> void launch_geom(int G, int B) {
    if (has_outflow_) {
      // has_outflow_ implies has_geometry_: outflow is a flag.
      scalar_kernel<P, A, true, true><<<G, B>>>(params(), N_);
    } else if (has_geometry_) {
      scalar_kernel<P, A, true, false><<<G, B>>>(params(), N_);
    } else {
      scalar_kernel<P, A, false, false><<<G, B>>>(params(), N_);
    }
  }
  template <int P> void launch_outflow(int G, int B) {
    if (ux_) scalar_outflow_kernel<P, true><<<G, B>>>(params(), N_);
    else     scalar_outflow_kernel<P, false><<<G, B>>>(params(), N_);
  }

  ScalarParams params() const {
    ScalarParams p;
    p.h = h_; p.flags = flags_; p.wall = wall_;
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.T_out = T_;
    p.donor = donor_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_; p.T_ref = T_ref_;
    p.regularised = (op_ == ScalarOp::Regularised);
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real T_ref_, omega_;
  ScalarOp op_ = ScalarOp::BGK;
  Real* h_ = nullptr;
  std::uint8_t* flags_ = nullptr;
  Real* wall_ = nullptr;
  Real* T_ = nullptr;
  long* donor_ = nullptr;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  bool has_geometry_ = false;
  bool has_outflow_ = false;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
