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
template <int Parity, int OpKind, int FKind, bool Mhd, bool HasGeometry>
LBM_HD LBM_INLINE void fluid_node_update(const FluidParams& p, long N, long n) {
  // Short-circuit on a compile-time constant: with HasGeometry false the load
  // is not merely predicted away, it is never emitted. Measured worth 5.6% on a
  // T4 -- see the note in streaming.cuh on why a one-byte-per-node array costs
  // ten times what its byte count suggests.
  if (HasGeometry && p.flags[n] != Fluid) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  Real f[27];
  gather<Parity>(p.f, N, x, y, z, p.nx, p.ny, p.nz, f);

  Macro m = macroscopic(f);

  Coupling cp;
  force_at<FKind>(p.force, n, cp.F);
  if (FKind != ForceNone) shift_velocity(m, cp.F);
  if (Mhd) { cp.B[0] = p.Bx[n]; cp.B[1] = p.By[n]; cp.B[2] = p.Bz[n]; }

  // The velocity the coupled fields advect with is the one Guo's half-shift has
  // already been applied to, i.e. the physical one.
  if (p.ux_out) { p.ux_out[n] = m.ux; p.uy_out[n] = m.uy; p.uz_out[n] = m.uz; }

  if (OpKind == 0) collide_bgk_gen<FKind != ForceNone, Mhd>(f, m, p.omega, cp);
  else             collide_cm_gen <FKind != ForceNone, Mhd>(f, m, p.omega, p.omega_bulk, cp);

  scatter<Parity>(p.f, N, x, y, z, p.nx, p.ny, p.nz, f);
}

//------------------------------------------------------------------------------
// Macroscopic readout at one node, for diagnostics. Non-fluid cells report rest
// at rho0 rather than whatever their storage happens to hold, so a plot of the
// field does not show wall slots as flow.
//------------------------------------------------------------------------------
template <int Parity>
LBM_HD LBM_INLINE void macro_node(const Real* f, const std::uint8_t* flags, long N, long n,
                                  int nx, int ny, int nz,
                                  Real* rho, Real* ux, Real* uy, Real* uz) {
  if (flags[n] != Fluid) {
    rho[n] = Real(1); ux[n] = uy[n] = uz[n] = Real(0);
    return;
  }
  int x, y, z;
  coords(n, nx, ny, x, y, z);
  Real fl[27];
  gather<Parity>(f, N, x, y, z, nx, ny, nz, fl);
  const Macro m = macroscopic(fl);
  rho[n] = m.rho; ux[n] = m.ux; uy[n] = m.uy; uz[n] = m.uz;
}

//==============================================================================
//  CUDA kernels -- thin wrappers around the functions above.
//==============================================================================
#if defined(__CUDACC__)

template <int Parity, int OpKind, int FKind, bool Mhd, bool HasGeometry>
__global__ void fluid_kernel(FluidParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  fluid_node_update<Parity, OpKind, FKind, Mhd, HasGeometry>(p, N, n);
}

template <int Parity>
__global__ void compute_macro(const Real* __restrict__ f,
                              const std::uint8_t* __restrict__ flags,
                              int nx, int ny, int nz,
                              Real* __restrict__ rho, Real* __restrict__ ux,
                              Real* __restrict__ uy, Real* __restrict__ uz) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  macro_node<Parity>(f, flags, N, n, nx, ny, nz, rho, ux, uy, uz);
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
                           int nx, int ny, int nz, Init init) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int x, y, z;
  coords(n, nx, ny, x, y, z);

  Macro m = (flags[n] == Fluid) ? init(x, y, z)
                                : Macro{Real(1), Real(0), Real(0), Real(0)};
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
    LBM_CUDA_CHECK(cudaMalloc(&flags_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMemset(flags_, Fluid, sizeof(std::uint8_t) * N_));
  }
  ~Solver() {
    cudaFree(f_); cudaFree(flags_);
    cudaFree(ux_); cudaFree(uy_); cudaFree(uz_);
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
    initialise<<<int((N_ + B - 1) / B), B>>>(f_, flags_, nx_, ny_, nz_, init);
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
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) compute_macro<0><<<G, B>>>(f_, flags_, nx_, ny_, nz_, dr, ux_, uy_, uz_);
    else             compute_macro<1><<<G, B>>>(f_, flags_, nx_, ny_, nz_, dr, ux_, uy_, uz_);
    LBM_CUDA_CHECK(cudaGetLastError());
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
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) compute_macro<0><<<G, B>>>(f_, flags_, nx_, ny_, nz_, dr, dx, dy, dz);
    else             compute_macro<1><<<G, B>>>(f_, flags_, nx_, ny_, nz_, dr, dx, dy, dz);
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
  FluidParams params() {
    FluidParams p;
    p.f = f_; p.flags = flags_;
    p.Bx = Bx_; p.By = By_; p.Bz = Bz_;
    p.ux_out = ux_; p.uy_out = uy_; p.uz_out = uz_;
    p.force = force_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_; p.omega_bulk = omega_bulk_;
    return p;
  }

  //--------------------------------------------------------------------------
  // Dispatch. Nested so that only the combinations actually used are ever
  // instantiated -- MHD with buoyancy, for instance, is never launched and
  // therefore never compiled.
  //--------------------------------------------------------------------------
  template <int P> void launch_op() {
    if (op_ == Op::BGK) launch_force<P, 0>();
    else                launch_force<P, 1>();
  }
  template <int P, int O> void launch_force() {
    if (mhd_) {
      if (fkind_ == ForceUniform) run<P, O, ForceUniform, true>();
      else                        run<P, O, ForceNone,    true>();
    } else if (fkind_ == ForceUniform) {
      run<P, O, ForceUniform, false>();
    } else if (fkind_ == ForceBoussinesq) {
      run<P, O, ForceBoussinesq, false>();
    } else {
      run<P, O, ForceNone, false>();
    }
  }
  template <int P, int O, int F, bool M> void run() {
    if (has_geometry_) launch<P, O, F, M, true>();
    else               launch<P, O, F, M, false>();
  }
  template <int P, int O, int F, bool M, bool G> void launch() {
    const int B = 128;
    fluid_kernel<P, O, F, M, G><<<int((N_ + B - 1) / B), B>>>(params(), N_);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  int nx_, ny_, nz_;
  long N_;
  Op op_;
  Real omega_, omega_bulk_;
  Real* f_ = nullptr;
  std::uint8_t* flags_ = nullptr;
  Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  const Real *Bx_ = nullptr, *By_ = nullptr, *Bz_ = nullptr;
  BodyForce force_{};
  int fkind_ = ForceNone;
  bool mhd_ = false;
  bool has_geometry_ = false;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
