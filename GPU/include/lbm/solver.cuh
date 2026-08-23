#pragma once
//==============================================================================
//  Esoteric Pull streaming and the fused stream-collide kernel.
//
//  ESOTERIC PULL, briefly. One lattice, updated in place. The pair of opposite
//  directions (i, i+1) shares a slot, and which of the two a node owns flips
//  with the parity of the timestep. Every slot has exactly one reader and one
//  writer, and they are the same node, so the update is race-free with no
//  temporary buffer and no second lattice.
//
//  Two consequences worth stating because they are the reason to use it:
//    * half the memory footprint of a two-lattice scheme;
//    * bounce-back becomes the identity on the storage, so a solid cell needs
//      no work at all -- it is simply not visited.
//
//  The parity tables below are transcribed from the same contract the parent
//  implementation uses (opp(i) == i + 1 for odd i). If the direction ordering in
//  core.cuh is ever changed, this breaks silently: populations still move, just
//  in the wrong directions, and the flow looks plausible. host_check.cpp tests
//  the round trip for exactly that reason.
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

//------------------------------------------------------------------------------
// Periodic node index. Kept as a free function so host_check.cpp can exercise
// the same arithmetic the kernel uses.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE int wrap(int v, int n) { return (v + n) % n; }

LBM_HD LBM_INLINE long node_id(int x, int y, int z, int nx, int ny) {
  return long(x) + long(nx) * (long(y) + long(ny) * long(z));
}

//------------------------------------------------------------------------------
// Neighbour in direction i, with periodic wrap.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE long neighbour(int x, int y, int z, int i, int nx, int ny, int nz) {
  return node_id(wrap(x + D3Q27::cx(i), nx),
                 wrap(y + D3Q27::cy(i), ny),
                 wrap(z + D3Q27::cz(i), nz), nx, ny);
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
// Gather the 27 populations a node should collide with, in direction order.
//------------------------------------------------------------------------------
template <int Parity>
LBM_HD LBM_INLINE void gather(const Real* f, long N, int x, int y, int z,
                              int nx, int ny, int nz, Real out[27]) {
  const long self = node_id(x, y, z, nx, ny);
  out[0] = f[self];
  for (int i = 1; i < 27; i += 2) {
    const long nb = neighbour(x, y, z, i, nx, ny, nz);
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
template <int Parity>
LBM_HD LBM_INLINE void init_scatter(Real* f, long N, int x, int y, int z,
                                    int nx, int ny, int nz, const Real in[27]) {
  const long self = node_id(x, y, z, nx, ny);
  f[self] = in[0];
  for (int i = 1; i < 27; i += 2) {
    const long nb = neighbour(x, y, z, i, nx, ny, nz);
    if (Parity == 0) { f[long(i) * N + self]     = in[i]; f[long(i + 1) * N + nb] = in[i + 1]; }
    else             { f[long(i + 1) * N + self] = in[i]; f[long(i) * N + nb]     = in[i + 1]; }
  }
}

template <int Parity>
LBM_HD LBM_INLINE void scatter(Real* f, long N, int x, int y, int z,
                               int nx, int ny, int nz, const Real in[27]) {
  const long self = node_id(x, y, z, nx, ny);
  f[self] = in[0];
  for (int i = 1; i < 27; i += 2) {
    const long nb = neighbour(x, y, z, i, nx, ny, nz);
    store_pair<Parity>(f, N, self, nb, i, in[i], in[i + 1]);
  }
}

//==============================================================================
//  The fused stream-collide kernel.
//
//  Templated on parity and on the operator so that both fold at compile time --
//  a runtime `if (op == ...)` inside the kernel would serialise divergent warps
//  and defeat the point.
//==============================================================================
#if defined(__CUDACC__)

template <int Parity, int OpKind>
__global__ void stream_collide(Real* __restrict__ f, int nx, int ny, int nz,
                               Real omega, Real omega_bulk) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;

  const int x = int(n % nx);
  const int y = int((n / nx) % ny);
  const int z = int(n / (long(nx) * ny));

  Real fl[27];
  gather<Parity>(f, N, x, y, z, nx, ny, nz, fl);

  const Macro m = macroscopic(fl);
  if (OpKind == 0) collide_bgk(fl, m, omega);
  else             collide_cm(fl, m, omega, omega_bulk);

  scatter<Parity>(f, N, x, y, z, nx, ny, nz, fl);
}

//------------------------------------------------------------------------------
// Macroscopic readout, for diagnostics. Separate kernel: it runs on probe steps
// only, so folding it into the hot kernel would cost registers every step to
// save a launch every few thousand.
//------------------------------------------------------------------------------
template <int Parity>
__global__ void compute_macro(const Real* __restrict__ f, int nx, int ny, int nz,
                              Real* __restrict__ rho, Real* __restrict__ ux,
                              Real* __restrict__ uy, Real* __restrict__ uz) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const int x = int(n % nx);
  const int y = int((n / nx) % ny);
  const int z = int(n / (long(nx) * ny));

  Real fl[27];
  gather<Parity>(f, N, x, y, z, nx, ny, nz, fl);
  const Macro m = macroscopic(fl);
  rho[n] = m.rho; ux[n] = m.ux; uy[n] = m.uy; uz[n] = m.uz;
}

//------------------------------------------------------------------------------
// Initialise every slot exactly once.
//
// `scatter` writes the value that `gather` would read back, so applying it over
// all nodes at parity 0 leaves a consistent lattice. Doing this any other way
// (writing f[i*N+n] directly) desynchronises the paired slots and the first
// step transports garbage.
//------------------------------------------------------------------------------
template <class Init>
__global__ void initialise(Real* __restrict__ f, int nx, int ny, int nz, Init init) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const int x = int(n % nx);
  const int y = int((n / nx) % ny);
  const int z = int(n / (long(nx) * ny));

  Macro m = init(x, y, z);
  Real fl[27];
  for (int i = 0; i < 27; ++i) fl[i] = feq(i, m.rho, m.ux, m.uy, m.uz);
  init_scatter<0>(f, N, x, y, z, nx, ny, nz, fl);   // NOT scatter -- see above
}

//==============================================================================
//  Host-side driver.
//==============================================================================
class Solver {
 public:
  Solver(int nx, int ny, int nz, Op op, Real nu, Real omega_bulk = Real(1))
      : nx_(nx), ny_(ny), nz_(nz), op_(op),
        omega_(omega_from_viscosity(nu)), omega_bulk_(omega_bulk) {
    N_ = long(nx) * ny * nz;
    LBM_CUDA_CHECK(cudaMalloc(&f_, sizeof(Real) * 27 * N_));
  }
  ~Solver() { cudaFree(f_); }

  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  template <class Init>
  void initialise_with(Init init) {
    const int B = 128;
    initialise<<<int((N_ + B - 1) / B), B>>>(f_, nx_, ny_, nz_, init);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    t_ = 0;
  }

  void step() {
    const int B = 128;
    const int G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) {
      if (op_ == Op::BGK) stream_collide<0, 0><<<G, B>>>(f_, nx_, ny_, nz_, omega_, omega_bulk_);
      else                stream_collide<0, 1><<<G, B>>>(f_, nx_, ny_, nz_, omega_, omega_bulk_);
    } else {
      if (op_ == Op::BGK) stream_collide<1, 0><<<G, B>>>(f_, nx_, ny_, nz_, omega_, omega_bulk_);
      else                stream_collide<1, 1><<<G, B>>>(f_, nx_, ny_, nz_, omega_, omega_bulk_);
    }
    ++t_;
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
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) compute_macro<0><<<G, B>>>(f_, nx_, ny_, nz_, dr, dx, dy, dz);
    else             compute_macro<1><<<G, B>>>(f_, nx_, ny_, nz_, dr, dx, dy, dz);
    LBM_CUDA_CHECK(cudaGetLastError());
    rho.resize(N_); ux.resize(N_); uy.resize(N_); uz.resize(N_);
    LBM_CUDA_CHECK(cudaMemcpy(rho.data(), dr, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(ux.data(),  dx, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(uy.data(),  dy, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(uz.data(),  dz, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    cudaFree(dr); cudaFree(dx); cudaFree(dy); cudaFree(dz);
  }

  long nodes() const { return N_; }
  std::size_t timestep() const { return t_; }
  Real omega() const { return omega_; }

 private:
  int nx_, ny_, nz_;
  long N_;
  Op op_;
  Real omega_, omega_bulk_;
  Real* f_ = nullptr;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
