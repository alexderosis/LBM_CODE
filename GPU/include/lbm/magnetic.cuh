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

//------------------------------------------------------------------------------
// Per-node magnetic boundary kind. A port of MagWallCode in
// ../src/solver/MagneticSolver.hpp.
//
//   MagBulk       collide normally
//   MagDirichlet  the node takes an imposed external field, by Dellar's
//                 moment condition (core.cuh) -- attained AT the node
//   MagOutflow    zero gradient: the node takes B from its upstream neighbour
//
// TWO KINDS, AND NEITHER IS A CONDUCTING OR INSULATING WALL. A real conducting
// wall couples the interior field to a wall current and an insulating one
// matches onto an exterior vacuum field; both are a separate piece of work and
// are NOT faked here. What these give is the wall-bounded MHD that a Hartmann
// channel actually needs, where the external field is prescribed.
//
// MagOutflow IS NOT VALIDATED AND HAS A MEASURED DEFECT, inherited from the
// parent along with the condition: on an inlet-driven Hartmann flow whose exact
// solution is independent of x -- so any streamwise variation IS boundary error
// -- it drives B about 6% high over the last ten nodes, and moving the outlet
// from Lx = 121 to 241 cut the sampled error 3.4x. Keep the outlet far from
// anything being measured. The cause is not identified; the natural suspects
// are the interaction with div B = 0 and the fact that a pure Neumann condition
// does not constrain the field LEVEL at all.
//------------------------------------------------------------------------------
enum MagCell : std::uint8_t {
  MagBulk      = 0,
  MagDirichlet = 1,
  MagOutflow   = 2,
};

struct MagneticParams {
  Real* g = nullptr;                         // 3 * Q * N
  const std::uint8_t* flags = nullptr;       // the fluid's geometry
  const Real* ux = nullptr;
  const Real* uy = nullptr;
  const Real* uz = nullptr;
  Real* Bx = nullptr;                        // written by compute_field only
  Real* By = nullptr;
  Real* Bz = nullptr;
  // Magnetic boundary kinds, and the imposed field at a Dirichlet node.
  //
  // THREE FULL ARRAYS RATHER THAN THE PARENT'S TAG TABLE. The parent stores one
  // byte per node indexing a table of at most 256 distinct wall fields, which
  // is thriftier and imposes a limit; here it is 12 bytes per node with no
  // limit, allocated only when walls are set at all. On a device where the
  // magnetic populations already cost 84 bytes per node that is a 14% surcharge
  // on runs that have walls and nothing on runs that do not.
  const std::uint8_t* mwall = nullptr;
  const std::uint8_t* unknown = nullptr;     // directions that streamed in from outside
  const Real* wBx = nullptr;
  const Real* wBy = nullptr;
  const Real* wBz = nullptr;
  int nx = 0, ny = 0, nz = 0;
  Real omega = Real(1);
};

LBM_HD LBM_INLINE long magnetic_offset(int a, long N) {
  return long(a) * long(MagneticLattice::Q) * N;
}

// `Advected` is false for a motionless conductor -- resistive decay isolates the
// induction equation and its resistivity with no coupling to a flow at all, and
// it is the first thing to check when a magnetic result looks wrong.
template <int Parity, bool Advected, bool HasGeometry, bool HasWalls = false>
LBM_HD LBM_INLINE void magnetic_node_update(const MagneticParams& p, long N, long n) {
  if (HasGeometry && p.flags[n] != Fluid) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  constexpr int Q = MagneticLattice::Q;
  Real g[3][Q];
  for (int a = 0; a < 3; ++a)
    gather<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                    x, y, z, p.nx, p.ny, p.nz, g[a]);

  // THE FIELD COMES FROM THE FIELD ARRAY, NOT FROM THE RAW SUM, whenever walls
  // exist. At a moment wall the streamed populations are still missing their
  // inward direction, so their sum is NOT B -- feeding that into the induction
  // equilibrium is how a wall-bounded run diverges with nothing to point at.
  // The field pass has already put the right number there for every node,
  // imposed at a Dirichlet node and taken from upstream at an outflow one.
  Real B[3];
  if (HasWalls) {
    B[0] = p.Bx[n];  B[1] = p.By[n];  B[2] = p.Bz[n];
  } else {
    for (int a = 0; a < 3; ++a) {
      B[a] = Real(0);
      for (int i = 0; i < Q; ++i) B[a] += g[a][i];
    }
  }

  Real u[3] = {Real(0), Real(0), Real(0)};
  if (Advected) { u[0] = p.ux[n]; u[1] = p.uy[n]; u[2] = p.uz[n]; }

  const std::uint8_t code = HasWalls ? p.mwall[n] : std::uint8_t(MagBulk);
  for (int a = 0; a < 3; ++a) {
    // Eqs. (13a)-(13b): choose the inward-pointing populations so the zeroth
    // moment is the target, THEN collide as usual. Dirichlet and outflow differ
    // only in where the target number comes from.
    if (HasWalls && code != MagBulk)
      impose_moment<MagneticLattice>(g[a], B[a], p.unknown[n]);
    collide_magnetic<MagneticLattice>(g[a], a, B, u, p.omega);
    scatter<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                     x, y, z, p.nx, p.ny, p.nz, g[a]);
  }
}

template <int Parity, bool HasGeometry, bool HasWalls = false>
LBM_HD LBM_INLINE void magnetic_field_node(const MagneticParams& p, long N, long n) {
  if (HasGeometry && p.flags[n] != Fluid) { p.Bx[n] = p.By[n] = p.Bz[n] = Real(0); return; }
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  if (HasWalls) {
    const std::uint8_t code = p.mwall[n];
    // A Dirichlet node's field is the imposed one BY CONSTRUCTION. Reading its
    // raw population sum instead gives whatever streamed in minus the direction
    // that has not been chosen yet, which is not a field.
    if (code == MagDirichlet) {
      p.Bx[n] = p.wBx[n];  p.By[n] = p.wBy[n];  p.Bz[n] = p.wBz[n];
      return;
    }
    // Zero-gradient outflow: same problem, different answer. Take the UPSTREAM
    // neighbour's populations. Reading a neighbour is safe in this kernel and
    // only in this kernel -- it reads populations and writes one B per thread,
    // so nothing it reads is written here. Reading B(upstream) instead would
    // race against whoever writes it.
    if (code == MagOutflow) x = wrap(x - 1, p.nx);
  }

  Real g[MagneticLattice::Q], B[3];
  for (int a = 0; a < 3; ++a) {
    gather<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                    x, y, z, p.nx, p.ny, p.nz, g);
    B[a] = Real(0);
    for (int i = 0; i < MagneticLattice::Q; ++i) B[a] += g[i];
  }
  p.Bx[n] = B[0]; p.By[n] = B[1]; p.Bz[n] = B[2];
}

//------------------------------------------------------------------------------
// Build the unknown-direction mask for every magnetic wall node.
//
// A direction is unknown when the node it would have streamed FROM is not part
// of the transport. `geom` is the fluid's geometry; empty means every node is
// fluid, so only nodes on a solid face are walls and a fully periodic array has
// none. Returns the number of wall nodes; `blind` comes back with how many of
// them have no unknown direction at all.
//
// A BLIND WALL NODE IS ALMOST ALWAYS A MIS-SPECIFIED GEOMETRY -- typically the
// wall marked one cell INSIDE the solid rather than on the fluid face -- and
// impose_moment then leaves it alone, so the condition silently does nothing.
// Hence the count, and the report.
//
// Plain host code, run once at setup, shared by the CUDA and host drivers.
//------------------------------------------------------------------------------
inline long build_magnetic_walls(const std::vector<std::uint8_t>& kind,
                                 const std::vector<std::uint8_t>& geom,
                                 int nx, int ny, int nz,
                                 std::vector<std::uint8_t>& unk, long& blind) {
  const long N = long(nx) * ny * nz;
  unk.assign(std::size_t(N), 0);
  auto outside = [&](int x, int y, int z) {
    if (geom.empty()) return false;
    return geom[std::size_t(node_id(x, y, z, nx, ny))] != Fluid;
  };
  long nwall = 0;
  blind = 0;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const long n = node_id(x, y, z, nx, ny);
        if (kind[std::size_t(n)] == MagBulk) continue;
        ++nwall;
        unk[std::size_t(n)] = unknown_mask<MagneticLattice>(x, y, z, nx, ny, nz, outside);
        if (unk[std::size_t(n)] == 0) ++blind;
      }
  if (blind)
    std::fprintf(stderr,
                 "  [magnetic] %ld of %ld wall node(s) have no inward unknown "
                 "direction and are inert -- check the wall is on the fluid face\n",
                 blind, nwall);
  return nwall;
}

#if defined(__CUDACC__)

template <int Parity, bool Advected, bool HasGeometry, bool HasWalls>
__global__ void magnetic_kernel(MagneticParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  magnetic_node_update<Parity, Advected, HasGeometry, HasWalls>(p, N, n);
}

template <int Parity, bool HasGeometry, bool HasWalls>
__global__ void magnetic_field_kernel(MagneticParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  magnetic_field_node<Parity, HasGeometry, HasWalls>(p, N, n);
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
    cudaFree(mwall_); cudaFree(unk_); cudaFree(wBx_); cudaFree(wBy_); cudaFree(wBz_);
  }

  MagneticSolver(const MagneticSolver&) = delete;
  MagneticSolver& operator=(const MagneticSolver&) = delete;

  void set_geometry(const std::vector<std::uint8_t>& flags) {
    LBM_CUDA_CHECK(cudaMemcpy(flags_, flags.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;
    geom_ = flags;
  }

  //--------------------------------------------------------------------------
  // Magnetic walls. `kind` is one MagCell per node and `wall` the imposed field
  // at each Dirichlet node (three components, ignored elsewhere).
  //
  // CALL set_geometry FIRST if the run has any. The unknown-direction mask is
  // built from BOTH: a direction is unknown when its source is outside the
  // domain OR is a non-fluid cell, and this needs the fluid's geometry to know
  // the second. With no geometry set, every node is fluid and only the domain
  // edge counts -- which is right for a channel bounded by the array itself.
  //--------------------------------------------------------------------------
  void set_walls(const std::vector<std::uint8_t>& kind,
                 const std::vector<Real>& wall_bx,
                 const std::vector<Real>& wall_by,
                 const std::vector<Real>& wall_bz) {
    if (long(kind.size()) != N_) {
      std::fprintf(stderr, "set_walls: %zu flags for %ld nodes\n", kind.size(), N_);
      std::exit(1);
    }
    std::vector<std::uint8_t> unk;
    long blind = 0;
    const long nwall = build_magnetic_walls(kind, geom_, nx_, ny_, nz_, unk, blind);

    if (!mwall_) {
      LBM_CUDA_CHECK(cudaMalloc(&mwall_, sizeof(std::uint8_t) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&unk_,   sizeof(std::uint8_t) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&wBx_,   sizeof(Real) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&wBy_,   sizeof(Real) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&wBz_,   sizeof(Real) * N_));
    }
    LBM_CUDA_CHECK(cudaMemcpy(mwall_, kind.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(unk_, unk.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wBx_, wall_bx.data(), sizeof(Real) * N_, cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wBy_, wall_by.data(), sizeof(Real) * N_, cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wBz_, wall_bz.data(), sizeof(Real) * N_, cudaMemcpyHostToDevice));
    has_walls_ = nwall > 0;
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
    // WITH WALLS THE COLLISION READS B FROM THE FIELD ARRAY, not from the raw
    // population sum -- at a moment wall the sum is not the field. So the field
    // pass must have run since the last step. A coupled driver calls
    // compute_field() itself before stepping the fluid, and the flag stops that
    // from being paid for twice; an uncoupled one gets it here and does not have
    // to know. A periodic run takes neither branch.
    if (has_walls_ && !field_current_) compute_field();
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) launch_advect<0>(G, B);
    else             launch_advect<1>(G, B);
    LBM_CUDA_CHECK(cudaGetLastError());
    field_current_ = false;
    ++t_;
  }

  void compute_field() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) launch_field<0>(G, B);
    else             launch_field<1>(G, B);
    LBM_CUDA_CHECK(cudaGetLastError());
    field_current_ = true;
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
    if (has_walls_)         magnetic_kernel<P, A, true,  true ><<<G, B>>>(params(), N_);
    else if (has_geometry_) magnetic_kernel<P, A, true,  false><<<G, B>>>(params(), N_);
    else                    magnetic_kernel<P, A, false, false><<<G, B>>>(params(), N_);
  }
  template <int P> void launch_field(int G, int B) {
    if (has_walls_)         magnetic_field_kernel<P, true,  true ><<<G, B>>>(params(), N_);
    else if (has_geometry_) magnetic_field_kernel<P, true,  false><<<G, B>>>(params(), N_);
    else                    magnetic_field_kernel<P, false, false><<<G, B>>>(params(), N_);
  }

  MagneticParams params() const {
    MagneticParams p;
    p.g = g_; p.flags = flags_;
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.Bx = Bx_; p.By = By_; p.Bz = Bz_;
    p.mwall = mwall_; p.unknown = unk_;
    p.wBx = wBx_; p.wBy = wBy_; p.wBz = wBz_;
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
  std::uint8_t *mwall_ = nullptr, *unk_ = nullptr;
  Real *wBx_ = nullptr, *wBy_ = nullptr, *wBz_ = nullptr;
  std::vector<std::uint8_t> geom_;            // host copy, for the unknown mask
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  bool has_geometry_ = false;
  bool has_walls_ = false;
  bool field_current_ = false;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
