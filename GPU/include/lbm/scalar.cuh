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
  Real* T_out = nullptr;                     // written by compute_field only
  int nx = 0, ny = 0, nz = 0;
  Real omega = Real(1), T_ref = Real(0);
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
template <int Parity, bool Advected, bool HasGeometry>
LBM_HD LBM_INLINE void scalar_node_update(const ScalarParams& p, long N, long n) {
  // With HasGeometry false every cell is bulk and the flags load is never
  // emitted -- see solver.cuh. `fl` is then a compile-time constant.
  const std::uint8_t fl = HasGeometry ? p.flags[n] : std::uint8_t(ScalarBulk);
  if (fl == ScalarExcluded || fl == ScalarAdiabatic) return;

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
    return;
  }

  const Real dT = scalar_deviation<ScalarLattice>(h);
  Real vx = Real(0), vy = Real(0), vz = Real(0);
  if (Advected) { vx = p.ux[n]; vy = p.uy[n]; vz = p.uz[n]; }
  collide_scalar<ScalarLattice>(h, dT, p.T_ref, vx, vy, vz, p.omega);
  scatter<Parity, ScalarLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);
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

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real h[ScalarLattice::Q];
  gather<Parity, ScalarLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);
  p.T_out[n] = p.T_ref + scalar_deviation<ScalarLattice>(h);
}

#if defined(__CUDACC__)

template <int Parity, bool Advected, bool HasGeometry>
__global__ void scalar_kernel(ScalarParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_node_update<Parity, Advected, HasGeometry>(p, N, n);
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
  ScalarSolver(int nx, int ny, int nz, Real diffusivity, Real T_ref = Real(0))
      : nx_(nx), ny_(ny), nz_(nz), T_ref_(T_ref) {
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
  ~ScalarSolver() { cudaFree(h_); cudaFree(flags_); cudaFree(wall_); cudaFree(T_); }

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
    if (t_ % 2 == 0) launch_advect<0>(G, B);
    else             launch_advect<1>(G, B);
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
    if (has_geometry_) scalar_kernel<P, A, true><<<G, B>>>(params(), N_);
    else               scalar_kernel<P, A, false><<<G, B>>>(params(), N_);
  }

  ScalarParams params() const {
    ScalarParams p;
    p.h = h_; p.flags = flags_; p.wall = wall_;
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.T_out = T_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_; p.T_ref = T_ref_;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real T_ref_, omega_;
  Real* h_ = nullptr;
  std::uint8_t* flags_ = nullptr;
  Real* wall_ = nullptr;
  Real* T_ = nullptr;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  bool has_geometry_ = false;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
