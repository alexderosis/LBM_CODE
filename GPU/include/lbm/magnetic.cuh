#pragma once
//==============================================================================
//  Magnetohydrodynamics -- the induction half.
//
//  The magnetic field is carried by Dellar's vector-valued distribution: a
//  separate D3Q7 population set for each Cartesian component of B, so 3 x 7 = 21
//  populations per node against the fluid's 27. The equilibrium is in core.cuh;
//  what lives here is the streaming, the coupling to u, and the driver.
//
//  Storage is g[(a * Q + i) * N + n] -- component-major, so each component's
//  block is contiguous and the SAME streaming functions the fluid uses operate
//  on it unchanged, one component at a time, by offsetting the base pointer.
//
//  THE LORENTZ COUPLING DOES NOT LIVE HERE. It is applied by giving the FLUID
//  equilibrium the Maxwell stress in its second moment (see `maxwell` in
//  core.cuh and `couple_magnetic` in solver.cuh), not by computing div(BB) and
//  applying it as a body force. The force route needs derivatives of B and loses
//  an order; this route is exact at the level of the equilibrium.
//
//  ALL THREE COMPONENTS ARE GATHERED BEFORE ANY IS COLLIDED. The equilibrium of
//  component a depends on the whole vector B, so collide-as-you-go would feed
//  post-collision values of B_x into the equilibrium of B_y. That is a silent
//  error: the field still evolves, still looks like a field, and damps at the
//  wrong rate.
//
//  BOUNDARIES. Periodic only, in practice. A non-fluid cell is skipped, which on
//  Esoteric Pull's storage means bounce-back on g -- and bounce-back on the
//  induction distribution is NOT a physical magnetic wall. It is neither the
//  perfectly conducting nor the insulating condition; Dellar's moment-based
//  wall is what those need, and it is not implemented here. The skip exists so a
//  domain with geometry runs rather than reads uninitialised storage. Do not
//  read a wall-bounded MHD result off this code.
//==============================================================================
#include "streaming.cuh"

namespace lbm {

using MagneticLattice = D3Q7;

struct MagneticParams {
  Real* g = nullptr;                         // 3 * Q * N
  const std::uint8_t* flags = nullptr;       // the fluid's geometry
  const Real* ux = nullptr;
  const Real* uy = nullptr;
  const Real* uz = nullptr;
  Real* Bx = nullptr;                        // written by compute_field only
  Real* By = nullptr;
  Real* Bz = nullptr;
  int nx = 0, ny = 0, nz = 0;
  Real omega = Real(1);
};

LBM_HD LBM_INLINE long magnetic_offset(int a, long N) {
  return long(a) * long(MagneticLattice::Q) * N;
}

// `Advected` is false for a motionless conductor -- resistive decay isolates the
// induction equation and its resistivity with no coupling to a flow at all, and
// it is the first thing to check when a magnetic result looks wrong.
template <int Parity, bool Advected, bool HasGeometry>
LBM_HD LBM_INLINE void magnetic_node_update(const MagneticParams& p, long N, long n) {
  if (HasGeometry && p.flags[n] != Fluid) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  constexpr int Q = MagneticLattice::Q;
  Real g[3][Q], B[3];
  for (int a = 0; a < 3; ++a) {
    gather<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                    x, y, z, p.nx, p.ny, p.nz, g[a]);
    B[a] = Real(0);
    for (int i = 0; i < Q; ++i) B[a] += g[a][i];
  }

  Real u[3] = {Real(0), Real(0), Real(0)};
  if (Advected) { u[0] = p.ux[n]; u[1] = p.uy[n]; u[2] = p.uz[n]; }
  for (int a = 0; a < 3; ++a) {
    collide_magnetic<MagneticLattice>(g[a], a, B, u, p.omega);
    scatter<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                     x, y, z, p.nx, p.ny, p.nz, g[a]);
  }
}

template <int Parity, bool HasGeometry>
LBM_HD LBM_INLINE void magnetic_field_node(const MagneticParams& p, long N, long n) {
  if (HasGeometry && p.flags[n] != Fluid) { p.Bx[n] = p.By[n] = p.Bz[n] = Real(0); return; }
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real g[MagneticLattice::Q], B[3];
  for (int a = 0; a < 3; ++a) {
    gather<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                    x, y, z, p.nx, p.ny, p.nz, g);
    B[a] = Real(0);
    for (int i = 0; i < MagneticLattice::Q; ++i) B[a] += g[i];
  }
  p.Bx[n] = B[0]; p.By[n] = B[1]; p.Bz[n] = B[2];
}

#if defined(__CUDACC__)

template <int Parity, bool Advected, bool HasGeometry>
__global__ void magnetic_kernel(MagneticParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  magnetic_node_update<Parity, Advected, HasGeometry>(p, N, n);
}

template <int Parity, bool HasGeometry>
__global__ void magnetic_field_kernel(MagneticParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  magnetic_field_node<Parity, HasGeometry>(p, N, n);
}

// Seeded at equilibrium with the initial velocity, not at rest: the equilibrium
// is u-dependent, and seeding it with u = 0 puts a transient into the induction
// equation that a decay-rate measurement then has to outlive.
template <class InitB, class InitU>
__global__ void magnetic_initialise(Real* __restrict__ g, int nx, int ny, int nz,
                                    InitB initB, InitU initU) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int x, y, z;
  coords(n, nx, ny, x, y, z);

  Real B[3], u[3];
  initB(x, y, z, B);
  initU(x, y, z, u);
  Real gl[MagneticLattice::Q];
  for (int a = 0; a < 3; ++a) {
    for (int i = 0; i < MagneticLattice::Q; ++i)
      gl[i] = magnetic_eq<MagneticLattice>(i, a, B, u);
    init_scatter<0, MagneticLattice>(g + magnetic_offset(a, N), N,
                                     x, y, z, nx, ny, nz, gl);
  }
}

//==============================================================================
//  Host-side driver.
//==============================================================================
class MagneticSolver {
 public:
  MagneticSolver(int nx, int ny, int nz, Real resistivity)
      : nx_(nx), ny_(ny), nz_(nz) {
    omega_ = omega_from_resistivity<MagneticLattice>(resistivity);
    N_ = long(nx) * ny * nz;
    LBM_CUDA_CHECK(cudaMalloc(&g_, sizeof(Real) * 3 * MagneticLattice::Q * N_));
    LBM_CUDA_CHECK(cudaMalloc(&flags_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMemset(flags_, Fluid, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&Bx_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&By_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&Bz_, sizeof(Real) * N_));
  }
  ~MagneticSolver() {
    cudaFree(g_); cudaFree(flags_); cudaFree(Bx_); cudaFree(By_); cudaFree(Bz_);
  }

  MagneticSolver(const MagneticSolver&) = delete;
  MagneticSolver& operator=(const MagneticSolver&) = delete;

  void set_geometry(const std::vector<std::uint8_t>& flags) {
    LBM_CUDA_CHECK(cudaMemcpy(flags_, flags.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;
  }
  void advect_with(const Real* ux, const Real* uy, const Real* uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  template <class InitB, class InitU>
  void initialise_with(InitB initB, InitU initU) {
    const int B = 128;
    magnetic_initialise<<<int((N_ + B - 1) / B), B>>>(g_, nx_, ny_, nz_, initB, initU);
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

  void compute_field() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) {
      if (has_geometry_) magnetic_field_kernel<0, true><<<G, B>>>(params(), N_);
      else               magnetic_field_kernel<0, false><<<G, B>>>(params(), N_);
    } else {
      if (has_geometry_) magnetic_field_kernel<1, true><<<G, B>>>(params(), N_);
      else               magnetic_field_kernel<1, false><<<G, B>>>(params(), N_);
    }
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  void field_to_host(std::vector<Real>& bx, std::vector<Real>& by, std::vector<Real>& bz) {
    compute_field();
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    bx.resize(N_); by.resize(N_); bz.resize(N_);
    LBM_CUDA_CHECK(cudaMemcpy(bx.data(), Bx_, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(by.data(), By_, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(bz.data(), Bz_, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
  }

  const Real* Bx_device() const { return Bx_; }
  const Real* By_device() const { return By_; }
  const Real* Bz_device() const { return Bz_; }
  Real omega() const { return omega_; }
  Real resistivity() const { return diffusivity_from_omega<MagneticLattice>(omega_); }
  std::size_t timestep() const { return t_; }

 private:
  template <int P> void launch_advect(int G, int B) {
    if (ux_) launch_geom<P, true>(G, B);
    else     launch_geom<P, false>(G, B);
  }
  template <int P, bool A> void launch_geom(int G, int B) {
    if (has_geometry_) magnetic_kernel<P, A, true><<<G, B>>>(params(), N_);
    else               magnetic_kernel<P, A, false><<<G, B>>>(params(), N_);
  }

  MagneticParams params() const {
    MagneticParams p;
    p.g = g_; p.flags = flags_;
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.Bx = Bx_; p.By = By_; p.Bz = Bz_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real omega_;
  Real* g_ = nullptr;
  std::uint8_t* flags_ = nullptr;
  Real *Bx_ = nullptr, *By_ = nullptr, *Bz_ = nullptr;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  bool has_geometry_ = false;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
