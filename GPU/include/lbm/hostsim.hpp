#pragma once
//==============================================================================
//  The host reference driver.
//
//  Runs complete simulations -- fluid, scalar, magnetic, coupled -- on a machine
//  with no GPU, by calling THE SAME per-node update functions the CUDA kernels
//  call. Not a second implementation: `fluid_node_update`, `scalar_node_update`
//  and `magnetic_node_update` are LBM_HD, and the kernels are three lines of
//  index arithmetic around them. So a physics failure found here is a physics
//  failure on the device, and a physics check that passes here has exercised the
//  arithmetic the device will run.
//
//  WHY A SERIAL LOOP IS A VALID SUBSTITUTE FOR A GRID OF THREADS. Under Esoteric
//  Pull each storage slot has exactly one writer per step, and that writer is
//  also its only reader. So the nodes of one step commute: any order gives the
//  same answer, and a for-loop is not an approximation to the kernel launch, it
//  is the same computation. (This is also why the scheme needs no temporary
//  buffer on the device.)
//
//  What this CANNOT catch: launch configuration, register pressure, an actual
//  race, a host-device transfer bug. It catches wrong physics, which is the
//  failure mode that is expensive to diagnose remotely and cheap to find here.
//
//  Slow, and meant to be. Grids of 32^3 in seconds, not 512^3.
//==============================================================================
#include "magnetic.cuh"
#include "scalar.cuh"
#include "solver.cuh"

#include <vector>

namespace lbm {
namespace host {

//==============================================================================
//  Fluid.
//==============================================================================
class Fluid {
 public:
  Fluid(int nx, int ny, int nz, Op op, Real nu, Real omega_bulk = Real(1))
      : nx_(nx), ny_(ny), nz_(nz), op_(op), omega_bulk_(omega_bulk) {
    omega_ = omega_from_viscosity(nu);
    N_ = long(nx) * ny * nz;
    f_.assign(std::size_t(27 * N_), Real(0));
    flags_.assign(std::size_t(N_), Fluid_);
  }

  void set_geometry(const std::vector<std::uint8_t>& fl) { flags_ = fl; }
  void set_force(const BodyForce& b, int kind) { force_ = b; fkind_ = kind; }
  void couple_magnetic(const Real* bx, const Real* by, const Real* bz) {
    Bx_ = bx; By_ = by; Bz_ = bz; mhd_ = true;
    enable_velocity_output();
  }
  void enable_velocity_output() {
    if (!ux_.empty()) return;
    ux_.assign(std::size_t(N_), Real(0));
    uy_.assign(std::size_t(N_), Real(0));
    uz_.assign(std::size_t(N_), Real(0));
  }

  // Solid and excluded cells are seeded at REST, not left empty -- an empty wall
  // layer soaks mass out of the fluid and leaves the bulk density low by
  // O(1/H), which is a first-order error in mu = rho nu.
  template <class Init>
  void initialise_with(Init init) {
    for (long n = 0; n < N_; ++n) {
      int x, y, z;
      coords(n, nx_, ny_, x, y, z);
      const Macro m = (flags_[std::size_t(n)] == Fluid_)
                          ? init(x, y, z)
                          : Macro{Real(1), Real(0), Real(0), Real(0)};
      Real fl[27];
      for (int i = 0; i < 27; ++i) fl[i] = feq(i, m.rho, m.ux, m.uy, m.uz);
      init_scatter<0>(f_.data(), N_, x, y, z, nx_, ny_, nz_, fl);
    }
    t_ = 0;
    if (!ux_.empty()) refresh_velocity();
  }

  void step() {
    if (t_ % 2 == 0) dispatch_op<0>();
    else             dispatch_op<1>();
    ++t_;
  }

  void refresh_velocity() {
    std::vector<Real> rho;
    macroscopic_to_host(rho, ux_, uy_, uz_);
  }

  void macroscopic_to_host(std::vector<Real>& rho, std::vector<Real>& ux,
                           std::vector<Real>& uy, std::vector<Real>& uz) const {
    rho.resize(std::size_t(N_)); ux.resize(std::size_t(N_));
    uy.resize(std::size_t(N_));  uz.resize(std::size_t(N_));
    for (long n = 0; n < N_; ++n) {
      if (t_ % 2 == 0)
        macro_node<0>(f_.data(), flags_.data(), N_, n, nx_, ny_, nz_,
                      rho.data(), ux.data(), uy.data(), uz.data());
      else
        macro_node<1>(f_.data(), flags_.data(), N_, n, nx_, ny_, nz_,
                      rho.data(), ux.data(), uy.data(), uz.data());
    }
  }

  const Real* ux_device() const { return ux_.data(); }
  const Real* uy_device() const { return uy_.data(); }
  const Real* uz_device() const { return uz_.data(); }

  // Summed over EVERY slot, including those owned by solid cells. Exactly
  // conserved: collision preserves the local sum and streaming only moves
  // values between slots, each of which has one writer per step. A geometry bug
  // that visited a cell twice, or skipped one it should not have, breaks this.
  double total_mass() const {
    double s = 0;
    for (Real v : f_) s += double(v);
    return s;
  }

  long nodes() const { return N_; }
  Real omega() const { return omega_; }
  std::size_t timestep() const { return t_; }

 private:
  static constexpr std::uint8_t Fluid_ = lbm::Fluid;

  FluidParams params() {
    FluidParams p;
    p.f = f_.data(); p.flags = flags_.data();
    p.Bx = Bx_; p.By = By_; p.Bz = Bz_;
    if (!ux_.empty()) { p.ux_out = ux_.data(); p.uy_out = uy_.data(); p.uz_out = uz_.data(); }
    p.force = force_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_; p.omega_bulk = omega_bulk_;
    return p;
  }

  template <int P> void dispatch_op() {
    if (op_ == Op::BGK) dispatch_force<P, 0>();
    else                dispatch_force<P, 1>();
  }
  template <int P, int O> void dispatch_force() {
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
    const FluidParams p = params();
    for (long n = 0; n < N_; ++n) fluid_node_update<P, O, F, M>(p, N_, n);
  }

  int nx_, ny_, nz_;
  long N_;
  Op op_;
  Real omega_, omega_bulk_;
  std::vector<Real> f_;
  std::vector<std::uint8_t> flags_;
  std::vector<Real> ux_, uy_, uz_;
  const Real *Bx_ = nullptr, *By_ = nullptr, *Bz_ = nullptr;
  BodyForce force_{};
  int fkind_ = ForceNone;
  bool mhd_ = false;
  std::size_t t_ = 0;
};

//==============================================================================
//  Passive scalar.
//==============================================================================
class Scalar {
 public:
  Scalar(int nx, int ny, int nz, Real diffusivity, Real T_ref = Real(0))
      : nx_(nx), ny_(ny), nz_(nz), T_ref_(T_ref) {
    omega_ = omega_from_diffusivity<ScalarLattice>(diffusivity);
    N_ = long(nx) * ny * nz;
    h_.assign(std::size_t(ScalarLattice::Q * N_), Real(0));
    flags_.assign(std::size_t(N_), std::uint8_t(ScalarBulk));
    wall_.assign(std::size_t(N_), Real(0));
    T_.assign(std::size_t(N_), Real(0));
  }

  void set_geometry(const std::vector<std::uint8_t>& fl, const std::vector<Real>& wall) {
    flags_ = fl; wall_ = wall;
  }
  void advect_with(const Real* ux, const Real* uy, const Real* uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  template <class Init>
  void initialise_with(Init init) {
    for (long n = 0; n < N_; ++n) {
      int x, y, z;
      coords(n, nx_, ny_, x, y, z);
      const Real T = (flags_[std::size_t(n)] == ScalarDirichlet) ? wall_[std::size_t(n)]
                                                                 : init(x, y, z);
      Real g[ScalarLattice::Q];
      for (int i = 0; i < ScalarLattice::Q; ++i)
        g[i] = scalar_eq<ScalarLattice>(i, T - T_ref_, T_ref_, Real(0), Real(0), Real(0));
      init_scatter<0, ScalarLattice>(h_.data(), N_, x, y, z, nx_, ny_, nz_, g);
    }
    t_ = 0;
    compute_field();
  }

  void step() {
    const ScalarParams p = params();
    if (t_ % 2 == 0) {
      if (ux_) for (long n = 0; n < N_; ++n) scalar_node_update<0, true>(p, N_, n);
      else     for (long n = 0; n < N_; ++n) scalar_node_update<0, false>(p, N_, n);
    } else {
      if (ux_) for (long n = 0; n < N_; ++n) scalar_node_update<1, true>(p, N_, n);
      else     for (long n = 0; n < N_; ++n) scalar_node_update<1, false>(p, N_, n);
    }
    ++t_;
  }

  void compute_field() {
    const ScalarParams p = params();
    if (t_ % 2 == 0) for (long n = 0; n < N_; ++n) scalar_field_node<0>(p, N_, n);
    else             for (long n = 0; n < N_; ++n) scalar_field_node<1>(p, N_, n);
  }

  const std::vector<Real>& field() { compute_field(); return T_; }
  void field_to_host(std::vector<Real>& T) { compute_field(); T = T_; }
  const Real* field_device() const { return T_.data(); }

  // The conserved quantity when every wall is adiabatic. Summed over the WHOLE
  // lattice, not over fluid cells: a population in flight toward a wall spends a
  // step in a slot the wall owns, and a fluid-only sum does not see it.
  double total_population() const {
    double s = 0;
    for (Real v : h_) s += double(v);
    return s;
  }

  Real omega() const { return omega_; }
  Real diffusivity() const { return diffusivity_from_omega<ScalarLattice>(omega_); }
  std::size_t timestep() const { return t_; }

 private:
  ScalarParams params() {
    ScalarParams p;
    p.h = h_.data(); p.flags = flags_.data(); p.wall = wall_.data();
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.T_out = T_.data();
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_; p.T_ref = T_ref_;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real T_ref_, omega_;
  std::vector<Real> h_, wall_, T_;
  std::vector<std::uint8_t> flags_;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  std::size_t t_ = 0;
};

//==============================================================================
//  Magnetic induction.
//==============================================================================
class Magnetic {
 public:
  Magnetic(int nx, int ny, int nz, Real resistivity) : nx_(nx), ny_(ny), nz_(nz) {
    omega_ = omega_from_resistivity<MagneticLattice>(resistivity);
    N_ = long(nx) * ny * nz;
    g_.assign(std::size_t(3 * MagneticLattice::Q * N_), Real(0));
    flags_.assign(std::size_t(N_), std::uint8_t(lbm::Fluid));
    Bx_.assign(std::size_t(N_), Real(0));
    By_.assign(std::size_t(N_), Real(0));
    Bz_.assign(std::size_t(N_), Real(0));
  }

  void set_geometry(const std::vector<std::uint8_t>& fl) { flags_ = fl; }
  void advect_with(const Real* ux, const Real* uy, const Real* uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  template <class InitB, class InitU>
  void initialise_with(InitB initB, InitU initU) {
    for (long n = 0; n < N_; ++n) {
      int x, y, z;
      coords(n, nx_, ny_, x, y, z);
      Real B[3], u[3];
      initB(x, y, z, B);
      initU(x, y, z, u);
      Real gl[MagneticLattice::Q];
      for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < MagneticLattice::Q; ++i)
          gl[i] = magnetic_eq<MagneticLattice>(i, a, B, u);
        init_scatter<0, MagneticLattice>(g_.data() + magnetic_offset(a, N_), N_,
                                         x, y, z, nx_, ny_, nz_, gl);
      }
    }
    t_ = 0;
    compute_field();
  }

  void step() {
    const MagneticParams p = params();
    if (t_ % 2 == 0) {
      if (ux_) for (long n = 0; n < N_; ++n) magnetic_node_update<0, true>(p, N_, n);
      else     for (long n = 0; n < N_; ++n) magnetic_node_update<0, false>(p, N_, n);
    } else {
      if (ux_) for (long n = 0; n < N_; ++n) magnetic_node_update<1, true>(p, N_, n);
      else     for (long n = 0; n < N_; ++n) magnetic_node_update<1, false>(p, N_, n);
    }
    ++t_;
  }

  void compute_field() {
    const MagneticParams p = params();
    if (t_ % 2 == 0) for (long n = 0; n < N_; ++n) magnetic_field_node<0>(p, N_, n);
    else             for (long n = 0; n < N_; ++n) magnetic_field_node<1>(p, N_, n);
  }

  void field_to_host(std::vector<Real>& bx, std::vector<Real>& by, std::vector<Real>& bz) {
    compute_field(); bx = Bx_; by = By_; bz = Bz_;
  }
  const std::vector<Real>& bx() { compute_field(); return Bx_; }
  const std::vector<Real>& by() { compute_field(); return By_; }
  const std::vector<Real>& bz() { compute_field(); return Bz_; }
  const Real* Bx_device() const { return Bx_.data(); }
  const Real* By_device() const { return By_.data(); }
  const Real* Bz_device() const { return Bz_.data(); }

  Real omega() const { return omega_; }
  Real resistivity() const { return diffusivity_from_omega<MagneticLattice>(omega_); }
  std::size_t timestep() const { return t_; }

 private:
  MagneticParams params() {
    MagneticParams p;
    p.g = g_.data(); p.flags = flags_.data();
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.Bx = Bx_.data(); p.By = By_.data(); p.Bz = Bz_.data();
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real omega_;
  std::vector<Real> g_, Bx_, By_, Bz_;
  std::vector<std::uint8_t> flags_;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  std::size_t t_ = 0;
};

//------------------------------------------------------------------------------
// One step of a fully coupled system, in the order the coupling requires.
//
// Refresh the coupled fields FIRST, so the fluid collides against T and B at its
// own time level. Stepping the fluid first and letting the scalar and the field
// catch up is a first-order splitting error that does NOT vanish under
// refinement -- see the note at the top of solver.cuh.
//------------------------------------------------------------------------------
inline void coupled_step(Fluid& fluid, Scalar* scalar, Magnetic* magnetic) {
  if (scalar)   scalar->compute_field();
  if (magnetic) magnetic->compute_field();
  fluid.step();
  if (scalar)   scalar->step();
  if (magnetic) magnetic->step();
}

}  // namespace host
}  // namespace lbm
