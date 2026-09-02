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
  Real omega_minus = Real(1);               // TRT only
  bool shifted = false;                     // storage: f_i, or f_i - w_i
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
LBM_HD LBM_INLINE void macro_node(const Real* f, const std::uint8_t* flags, long N, long n,
                                  int nx, int ny, int nz, const BodyForce& force,
                                  bool shifted,
                                  Real* rho, Real* ux, Real* uy, Real* uz) {
  if (flags[n] != Fluid) {
    rho[n] = Real(1); ux[n] = uy[n] = uz[n] = Real(0);
    return;
  }
  int x, y, z;
  coords(n, nx, ny, x, y, z);
  Real fl[27];
  gather<Parity>(f, N, x, y, z, nx, ny, nz, fl);
  Macro m = macroscopic(fl, shifted);
  if (FKind != ForceNone) {
    Real F[3];
    force_at<FKind>(force, n, F);
    shift_velocity(m, F);
  }
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

template <int Parity, int FKind>
__global__ void compute_macro(const Real* __restrict__ f,
                              const std::uint8_t* __restrict__ flags,
                              int nx, int ny, int nz, BodyForce force,
                              bool shifted,
                              Real* __restrict__ rho, Real* __restrict__ ux,
                              Real* __restrict__ uy, Real* __restrict__ uz) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  macro_node<Parity, FKind>(f, flags, N, n, nx, ny, nz, force, shifted, rho, ux, uy, uz);
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
      compute_macro<P, ForceUniform><<<G, B>>>(f_, flags_, nx_, ny_, nz_, force_, shifted_, dr, dx, dy, dz);
    else if (fkind_ == ForceBoussinesq)
      compute_macro<P, ForceBoussinesq><<<G, B>>>(f_, flags_, nx_, ny_, nz_, force_, shifted_, dr, dx, dy, dz);
    else if (fkind_ == ForceField)
      compute_macro<P, ForceField><<<G, B>>>(f_, flags_, nx_, ny_, nz_, force_, shifted_, dr, dx, dy, dz);
    else
      compute_macro<P, ForceNone><<<G, B>>>(f_, flags_, nx_, ny_, nz_, force_, shifted_, dr, dx, dy, dz);
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
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
