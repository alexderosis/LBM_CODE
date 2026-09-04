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
#include <functional>
#include "body.cuh"
#include "colour.cuh"
#include "freesurface.cuh"
#include "magnetic.cuh"
#include "phasefield.cuh"
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
    omega_minus_ = omega_minus_for(omega_, magic_3_16);
    N_ = long(nx) * ny * nz;
    f_.assign(std::size_t(27 * N_), Real(0));
    flags_.assign(std::size_t(N_), Fluid_);
  }

  void set_geometry(const std::vector<std::uint8_t>& fl) { flags_ = fl; has_geometry_ = true; }
  void set_force(const BodyForce& b, int kind) { force_ = b; fkind_ = kind; }

  // Regularised velocity walls. Same contract as the device class: call
  // set_geometry FIRST if there are solid cells, and remember the channel is
  // H-1 wide, not H. See regularized.cuh.
  void set_regularized_walls(const std::vector<RegWallSpec>& spec) {
    std::vector<std::uint8_t> geo = flags_;
    n_walls_ = build_reg_walls(spec, geo, nx_, ny_, nz_, bc_nrm_, bc_tag_,
                               bc_unk_, bc_ext_, wall_u_, has_corners_);
    if (n_walls_ == 0) return;
    for (long n = 0; n < N_; ++n)
      if (bc_nrm_[std::size_t(n)] != NrmNone) flags_[std::size_t(n)] = RegWall;
    has_geometry_ = true;
    has_walls_ = true;
    bc_rho_.assign(std::size_t(N_), Real(1));
    if (has_corners_ && fd_corners_) bc_pi_.assign(std::size_t(6 * N_), Real(0));
  }
  void set_fd_corners(bool on) { fd_corners_ = on; }
  long wall_count() const { return n_walls_; }
  void set_magic(Real lambda) { omega_minus_ = omega_minus_for(omega_, lambda); }

  // Store f_i - w_i instead of f_i. Call BEFORE initialise_with; see core.cuh.
  void set_shifted(bool on) { shifted_ = on; }
  bool shifted() const { return shifted_; }
  Real omega_minus() const { return omega_minus_; }
  Real magic() const { return magic_parameter(omega_, omega_minus_); }
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
      Macro m = (flags_[std::size_t(n)] == Fluid_)
                    ? init(x, y, z)
                    : Macro{Real(1), Real(0), Real(0), Real(0)};
      // The seed is a DENSITY whichever storage is in use; `dens` follows.
      m.dens = shifted_ ? (m.rho - Real(1)) : m.rho;
      Real fl[27];
      for (int i = 0; i < 27; ++i) fl[i] = feq_of(i, m, shifted_);
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
                           std::vector<Real>& uy, std::vector<Real>& uz) {
    rho.resize(std::size_t(N_)); ux.resize(std::size_t(N_));
    uy.resize(std::size_t(N_));  uz.resize(std::size_t(N_));
    if (t_ % 2 == 0) macro_force<0>(rho, ux, uy, uz);
    else             macro_force<1>(rho, ux, uy, uz);
  }

  // The macro pass has to know the force, and which KIND it is: the physical
  // velocity of a forced scheme carries Guo's half shift. See macro_node.
  template <int P>
  void macro_force(std::vector<Real>& rho, std::vector<Real>& ux,
                   std::vector<Real>& uy, std::vector<Real>& uz) {
    const FluidParams p = params();
    for (long n = 0; n < N_; ++n) {
      if (fkind_ == ForceUniform)
        macro_node<P, ForceUniform>(p, N_, n, rho.data(), ux.data(), uy.data(), uz.data());
      else if (fkind_ == ForceBoussinesq)
        macro_node<P, ForceBoussinesq>(p, N_, n, rho.data(), ux.data(), uy.data(), uz.data());
      else if (fkind_ == ForceField)
        macro_node<P, ForceField>(p, N_, n, rho.data(), ux.data(), uy.data(), uz.data());
      else
        macro_node<P, ForceNone>(p, N_, n, rho.data(), ux.data(), uy.data(), uz.data());
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
    p.omega_minus = omega_minus_;
    p.shifted = shifted_;
    if (has_walls_) {
      p.bc_nrm = bc_nrm_.data();  p.bc_tag = bc_tag_.data();
      p.bc_unk = bc_unk_.data();  p.bc_ext = bc_ext_.data();
      p.wall_u = wall_u_.data();  p.bc_rho = bc_rho_.data();
      p.bc_pi = bc_pi_.empty() ? nullptr : bc_pi_.data();
    }
    p.bc_shear_omega = omega_;
    p.fd_corners = fd_corners_;
    return p;
  }

  template <int P> void dispatch_op() {
    if      (op_ == Op::BGK) dispatch_force<P, 0>();
    else if (op_ == Op::TRT) dispatch_force<P, 2>();
    else                     dispatch_force<P, 1>();
  }
  template <int P, int O> void dispatch_force() {
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
    const FluidParams p = params();
    // The wall pre-passes, in the same order the device launches them and for
    // the same reason: a corner's rho needs its straight-wall neighbours' rho to
    // exist first, and the FD gradient must run while nothing is writing
    // populations. Skipped entirely without corners.
    if (has_walls_ && has_corners_) {
      for (long n = 0; n < N_; ++n) wall_rho_node<P>(p, N_, n);
      for (long n = 0; n < N_; ++n) wall_corner_node<P>(p, N_, n);
    }
    if (has_walls_)
      for (long n = 0; n < N_; ++n) fluid_node_update<P, O, F, M, true, true>(p, N_, n);
    else if (has_geometry_)
      for (long n = 0; n < N_; ++n) fluid_node_update<P, O, F, M, true, false>(p, N_, n);
    else
      for (long n = 0; n < N_; ++n) fluid_node_update<P, O, F, M, false, false>(p, N_, n);
  }

  int nx_, ny_, nz_;
  long N_;
  Op op_;
  Real omega_, omega_bulk_, omega_minus_;
  std::vector<Real> f_;
  std::vector<std::uint8_t> flags_;
  std::vector<Real> ux_, uy_, uz_;
  const Real *Bx_ = nullptr, *By_ = nullptr, *Bz_ = nullptr;
  BodyForce force_{};
  int fkind_ = ForceNone;
  bool mhd_ = false;
  bool has_geometry_ = false;
  bool shifted_ = false;
  std::vector<std::uint8_t>  bc_nrm_, bc_ext_;
  std::vector<std::uint16_t> bc_tag_;
  std::vector<std::uint32_t> bc_unk_;
  std::vector<Real> bc_rho_, bc_pi_, wall_u_;
  long n_walls_ = 0;
  bool has_walls_ = false;
  bool has_corners_ = false;
  bool fd_corners_ = true;
  std::size_t t_ = 0;
};

//==============================================================================
//  Passive scalar.
//==============================================================================
class Scalar {
 public:
  Scalar(int nx, int ny, int nz, Real diffusivity, Real T_ref = Real(0),
         ScalarOp op = ScalarOp::BGK)
      : nx_(nx), ny_(ny), nz_(nz), T_ref_(T_ref), op_(op) {
    omega_ = omega_from_diffusivity<ScalarLattice>(diffusivity);
    N_ = long(nx) * ny * nz;
    h_.assign(std::size_t(ScalarLattice::Q * N_), Real(0));
    flags_.assign(std::size_t(N_), std::uint8_t(ScalarBulk));
    wall_.assign(std::size_t(N_), Real(0));
    T_.assign(std::size_t(N_), Real(0));
  }

  void set_geometry(const std::vector<std::uint8_t>& fl, const std::vector<Real>& wall) {
    flags_ = fl; wall_ = wall; has_geometry_ = true;
    long degenerate = 0;
    has_outflow_ = build_scalar_donors(flags_, nx_, ny_, nz_, donor_, degenerate) > 0;
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
    // The serial loop supplies for free the fence the device gets from the
    // kernel boundary: the outflow pass simply runs after the main one.
    if (t_ % 2 == 0) { run_step<0>(); if (has_outflow_) run_outflow<0>(); }
    else             { run_step<1>(); if (has_outflow_) run_outflow<1>(); }
    ++t_;
  }

  void compute_field() {
    const ScalarParams p = params();
    if (t_ % 2 == 0) {
      if (has_geometry_) for (long n = 0; n < N_; ++n) scalar_field_node<0, true>(p, N_, n);
      else               for (long n = 0; n < N_; ++n) scalar_field_node<0, false>(p, N_, n);
    } else {
      if (has_geometry_) for (long n = 0; n < N_; ++n) scalar_field_node<1, true>(p, N_, n);
      else               for (long n = 0; n < N_; ++n) scalar_field_node<1, false>(p, N_, n);
    }
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
  template <int P> void run_step() {
    const ScalarParams p = params();
    if (ux_) {
      if (has_outflow_)       for (long n = 0; n < N_; ++n) scalar_node_update<P, true, true, true>(p, N_, n);
      else if (has_geometry_) for (long n = 0; n < N_; ++n) scalar_node_update<P, true, true, false>(p, N_, n);
      else                    for (long n = 0; n < N_; ++n) scalar_node_update<P, true, false, false>(p, N_, n);
    } else {
      if (has_outflow_)       for (long n = 0; n < N_; ++n) scalar_node_update<P, false, true, true>(p, N_, n);
      else if (has_geometry_) for (long n = 0; n < N_; ++n) scalar_node_update<P, false, true, false>(p, N_, n);
      else                    for (long n = 0; n < N_; ++n) scalar_node_update<P, false, false, false>(p, N_, n);
    }
  }

  template <int P> void run_outflow() {
    const ScalarParams p = params();
    if (ux_) for (long n = 0; n < N_; ++n) scalar_outflow_node<P, true>(p, N_, n);
    else     for (long n = 0; n < N_; ++n) scalar_outflow_node<P, false>(p, N_, n);
  }

  ScalarParams params() {
    ScalarParams p;
    p.h = h_.data(); p.flags = flags_.data(); p.wall = wall_.data();
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.T_out = T_.data();
    p.donor = donor_.empty() ? nullptr : donor_.data();
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_; p.T_ref = T_ref_;
    p.regularised = (op_ == ScalarOp::Regularised);
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real T_ref_, omega_;
  ScalarOp op_ = ScalarOp::BGK;
  std::vector<Real> h_, wall_, T_;
  std::vector<std::uint8_t> flags_;
  std::vector<long> donor_;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  bool has_geometry_ = false;
  bool has_outflow_ = false;
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

  void set_geometry(const std::vector<std::uint8_t>& fl) { flags_ = fl; has_geometry_ = true; }

  // Magnetic walls. Same contract as the device class: set_geometry FIRST if
  // the run has any, since the unknown-direction mask is built from both.
  void set_walls(const std::vector<std::uint8_t>& kind,
                 const std::vector<Real>& wall_bx,
                 const std::vector<Real>& wall_by,
                 const std::vector<Real>& wall_bz) {
    mwall_ = kind;  wBx_ = wall_bx;  wBy_ = wall_by;  wBz_ = wall_bz;
    long blind = 0;
    has_walls_ = build_magnetic_walls(kind, has_geometry_ ? flags_
                                                          : std::vector<std::uint8_t>(),
                                      nx_, ny_, nz_, unk_, blind) > 0;
  }
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
    // With walls the collision reads B from the field array, not from the raw
    // population sum -- at a moment wall the sum is not the field. The flag
    // stops a coupled driver, which refreshes B itself before stepping the
    // fluid, from paying for that pass twice. A periodic run takes neither
    // branch. See magnetic.cuh.
    if (has_walls_ && !field_current_) compute_field();
    if (t_ % 2 == 0) run_step<0>();
    else             run_step<1>();
    field_current_ = false;
    ++t_;
  }

  void compute_field() {
    const MagneticParams p = params();
    if (t_ % 2 == 0) {
      if (has_walls_)         for (long n = 0; n < N_; ++n) magnetic_field_node<0, true, true>(p, N_, n);
      else if (has_geometry_) for (long n = 0; n < N_; ++n) magnetic_field_node<0, true, false>(p, N_, n);
      else                    for (long n = 0; n < N_; ++n) magnetic_field_node<0, false, false>(p, N_, n);
    } else {
      if (has_walls_)         for (long n = 0; n < N_; ++n) magnetic_field_node<1, true, true>(p, N_, n);
      else if (has_geometry_) for (long n = 0; n < N_; ++n) magnetic_field_node<1, true, false>(p, N_, n);
      else                    for (long n = 0; n < N_; ++n) magnetic_field_node<1, false, false>(p, N_, n);
    }
    field_current_ = true;
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
  template <int P> void run_step() {
    const MagneticParams p = params();
    if (ux_) {
      if (has_walls_)         for (long n = 0; n < N_; ++n) magnetic_node_update<P, true, true, true>(p, N_, n);
      else if (has_geometry_) for (long n = 0; n < N_; ++n) magnetic_node_update<P, true, true, false>(p, N_, n);
      else                    for (long n = 0; n < N_; ++n) magnetic_node_update<P, true, false, false>(p, N_, n);
    } else {
      if (has_walls_)         for (long n = 0; n < N_; ++n) magnetic_node_update<P, false, true, true>(p, N_, n);
      else if (has_geometry_) for (long n = 0; n < N_; ++n) magnetic_node_update<P, false, true, false>(p, N_, n);
      else                    for (long n = 0; n < N_; ++n) magnetic_node_update<P, false, false, false>(p, N_, n);
    }
  }

  MagneticParams params() {
    MagneticParams p;
    p.g = g_.data(); p.flags = flags_.data();
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.Bx = Bx_.data(); p.By = By_.data(); p.Bz = Bz_.data();
    if (has_walls_) {
      p.mwall = mwall_.data();  p.unknown = unk_.data();
      p.wBx = wBx_.data();  p.wBy = wBy_.data();  p.wBz = wBz_.data();
    }
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real omega_;
  std::vector<Real> g_, Bx_, By_, Bz_;
  std::vector<Real> wBx_, wBy_, wBz_;
  std::vector<std::uint8_t> flags_, mwall_, unk_;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  bool has_geometry_ = false;
  bool has_walls_ = false;
  bool field_current_ = false;
  std::size_t t_ = 0;
};

//==============================================================================
//  Colour-gradient two-component flow.
//
//  Three passes in the same order the device driver launches them, and the
//  serial loop supplies for free the fences the CUDA version gets from kernel
//  boundaries. refresh() must precede step(): the collision reads fields the
//  step cannot compute for itself, because the value it would compute is
//  consumed and overwritten in the same pass.
//==============================================================================
class Colour {
 public:
  Colour(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    N_ = long(nx) * ny * nz;
    fr_.assign(std::size_t(27 * N_), Real(0));
    fb_.assign(std::size_t(27 * N_), Real(0));
    fld_.assign(std::size_t(12 * N_), Real(0));
    // lbm::Fluid, not Fluid: inside namespace host the unqualified name binds to
    // the host::Fluid class above rather than to the CellType enumerator.
    flags_.assign(std::size_t(N_), std::uint8_t(lbm::Fluid));
  }

  ColourModel model;

  void set_geometry(const std::vector<std::uint8_t>& fl) {
    flags_ = fl; has_geometry_ = true;
  }

  // init(x, y, z, rho_r&, rho_b&)
  template <class Init>
  void initialise_with(Init init) {
    for (long n = 0; n < N_; ++n) {
      int x, y, z;
      coords(n, nx_, ny_, x, y, z);
      Real rr = Real(1), rb = Real(0);
      init(x, y, z, rr, rb);
      Real gr[27], gb[27];
      for (int i = 0; i < 27; ++i) {
        gr[i] = rr * ColourModel::phi_i(i, model.alpha_r);
        gb[i] = rb * ColourModel::phi_i(i, model.alpha_b);
      }
      init_scatter<0, ColourLattice>(fr_.data(), N_, x, y, z, nx_, ny_, nz_, gr);
      init_scatter<0, ColourLattice>(fb_.data(), N_, x, y, z, nx_, ny_, nz_, gb);
    }
    t_ = 0;
    refresh();
  }

  void refresh() {
    const ColourParams p = params();
    if (t_ % 2 == 0) {
      if (has_geometry_) for (long n = 0; n < N_; ++n) colour_fields_node<0, true>(p, N_, n);
      else               for (long n = 0; n < N_; ++n) colour_fields_node<0, false>(p, N_, n);
    } else {
      if (has_geometry_) for (long n = 0; n < N_; ++n) colour_fields_node<1, true>(p, N_, n);
      else               for (long n = 0; n < N_; ++n) colour_fields_node<1, false>(p, N_, n);
    }
    if (has_geometry_) for (long n = 0; n < N_; ++n) colour_gradient_node<true>(p, n);
    else               for (long n = 0; n < N_; ++n) colour_gradient_node<false>(p, n);
  }

  void step() {
    const ColourParams p = params();
    if (t_ % 2 == 0) {
      if (has_geometry_) for (long n = 0; n < N_; ++n) colour_node_update<0, true>(p, N_, n);
      else               for (long n = 0; n < N_; ++n) colour_node_update<0, false>(p, N_, n);
    } else {
      if (has_geometry_) for (long n = 0; n < N_; ++n) colour_node_update<1, true>(p, N_, n);
      else               for (long n = 0; n < N_; ++n) colour_node_update<1, false>(p, N_, n);
    }
    ++t_;
  }

  void field_to_host(const Real* src, std::vector<Real>& out) {
    out.assign(src, src + N_);
  }

  const Real* rho_red_device()  const { return &fld_[0]; }
  const Real* rho_blue_device() const { return &fld_[std::size_t(N_)]; }
  const Real* phi_device()      const { return &fld_[std::size_t(2 * N_)]; }
  const Real* ux_device()       const { return &fld_[std::size_t(3 * N_)]; }
  const Real* uy_device()       const { return &fld_[std::size_t(4 * N_)]; }
  const Real* uz_device()       const { return &fld_[std::size_t(5 * N_)]; }
  std::size_t timestep() const { return t_; }
  long nodes() const { return N_; }

  // Each colour over every slot, walls included -- the same argument
  // Scalar::total_population carries about populations in flight.
  double total_red()  const { double s = 0; for (Real v : fr_) s += double(v); return s; }
  double total_blue() const { double s = 0; for (Real v : fb_) s += double(v); return s; }

 private:
  ColourParams params() {
    ColourParams p;
    p.fr = fr_.data(); p.fb = fb_.data();
    p.rho_r = &fld_[0];
    p.rho_b = &fld_[std::size_t(N_)];
    p.phi   = &fld_[std::size_t(2 * N_)];
    p.ux    = &fld_[std::size_t(3 * N_)];
    p.uy    = &fld_[std::size_t(4 * N_)];
    p.uz    = &fld_[std::size_t(5 * N_)];
    p.gx    = &fld_[std::size_t(6 * N_)];
    p.gy    = &fld_[std::size_t(7 * N_)];
    p.gz    = &fld_[std::size_t(8 * N_)];
    p.rx    = &fld_[std::size_t(9 * N_)];
    p.ry    = &fld_[std::size_t(10 * N_)];
    p.rz    = &fld_[std::size_t(11 * N_)];
    p.flags = flags_.data();
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.m = model;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  std::vector<Real> fr_, fb_, fld_;
  std::vector<std::uint8_t> flags_;
  bool has_geometry_ = false;
  std::size_t t_ = 0;
};


//==============================================================================
//  Phase-field two-phase flow at a density ratio.
//
//  Six passes in the order the device driver launches them, and the serial loop
//  supplies for free the fences the CUDA version gets from kernel boundaries.
//  The ORDER is the point -- see the banner in phasefield.cuh -- so it lives
//  here, in step(), rather than in a driver.
//==============================================================================
template <class PL = DefaultPhaseLattice>
class PhaseField {
 public:
  using PhaseLat = PL;
  using Params   = PfParamsT<PL>;

  PhaseField(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    N_ = long(nx) * ny * nz;
    f_.assign(std::size_t(27 * N_), Real(0));
    h_.assign(std::size_t(PL::Q * N_), Real(0));
    fld_.assign(std::size_t(15 * N_), Real(0));
    pflags_.assign(std::size_t(N_), std::uint8_t(PhaseBulk));
    fflags_.assign(std::size_t(N_), std::uint8_t(lbm::Fluid));
  }

  PhaseModel      phase;
  MultiphaseModel fluid;

  void enable_viscous_force(bool on) { viscous_ = on; }

  // Same contract as the device class: independent operators, both BGK by
  // default, and the phase field's central-moment form refused on a lattice
  // that cannot carry it -- with a message, at setup.
  void set_phase_op(PhaseOp op) {
    if (op == PhaseOp::CentralMoments && PL::Q != 27) {
      std::fprintf(stderr,
                   "host::PhaseField: central moments need a product lattice; "
                   "this one is D3Q%d. Use host::PhaseField<D3Q27>.\n", PL::Q);
      std::exit(1);
    }
    phase_op_ = op;
  }
  void set_fluid_op(MultiOp op) { fluid_op_ = op; }

  // An external per-node force, owned by the caller -- a penalised solid, say.
  void couple_external_force(const Real* fx, const Real* fy, const Real* fz) {
    ex_ = fx;  ey_ = fy;  ez_ = fz;
  }
  void set_pre_fluid(std::function<void()> fn) { pre_fluid_ = std::move(fn); }
  PhaseOp phase_op() const { return phase_op_; }
  MultiOp fluid_op() const { return fluid_op_; }

  void set_mobility(Real m) { phase.omega = PhaseModel::omega_from_mobility<PL>(m); }
  Real mobility() const { return phase.template mobility<PL>(); }

  void set_geometry(const std::vector<std::uint8_t>& pf,
                    const std::vector<std::uint8_t>& ff) {
    pflags_ = pf;  fflags_ = ff;  has_geometry_ = true;
  }

  // init(x, y, z, phi&, p_tilde&)
  template <class Init>
  void initialise_with(Init init) {
    for (long n = 0; n < N_; ++n) {
      int x, y, z;
      coords(n, nx_, ny_, x, y, z);
      Real ph = Real(0), pt = Real(0);
      init(x, y, z, ph, pt);
      Real g[27];
      for (int i = 0; i < 27; ++i) g[i] = FluidLattice::w(i) * pt;
      init_scatter<0, FluidLattice>(f_.data(), N_, x, y, z, nx_, ny_, nz_, g);
      Real e[PL::Q];
      for (int i = 0; i < PL::Q; ++i) e[i] = PL::w(i) * ph;
      init_scatter<0, PL>(h_.data(), N_, x, y, z, nx_, ny_, nz_, e);
      fld_[std::size_t(n)] = ph;                       // phi
      fld_[std::size_t(8 * N_ + n)] = pt;              // p~
    }
    t_ = 0;
    derivatives();
  }

  void step() { if (t_ % 2 == 0) run<0>(); else run<1>(); ++t_; }

  void field_to_host(const Real* src, std::vector<Real>& out) {
    out.assign(src, src + N_);
  }

  const Real* phi_device() const { return &fld_[0]; }
  const Real* lap_device() const { return &fld_[std::size_t(4 * N_)]; }
  const Real* ux_device()  const { return &fld_[std::size_t(5 * N_)]; }
  const Real* uy_device()  const { return &fld_[std::size_t(6 * N_)]; }
  const Real* uz_device()  const { return &fld_[std::size_t(7 * N_)]; }
  const Real* pt_device()  const { return &fld_[std::size_t(8 * N_)]; }
  std::size_t timestep() const { return t_; }
  long nodes() const { return N_; }

  // Every phase population, wall slots included -- the conserved quantity, and
  // a sharp check: the Allen-Cahn form conserves phi EXACTLY, since
  // sum_i S_i = 0 and both streaming and collision are conservative. This is not
  // the same number as summing phi over the bulk, and the difference is not a
  // leak: under an in-place scheme a population in flight toward a wall spends a
  // step in a slot the wall owns.
  double total_phase() const {
    double s = 0;
    for (Real v : h_) s += double(v);
    return s;
  }

 private:
  void derivatives() {
    const Params p = params();
    if (has_geometry_) for (long n = 0; n < N_; ++n) pf_derivatives_node<true>(p, n);
    else               for (long n = 0; n < N_; ++n) pf_derivatives_node<false>(p, n);
  }

  template <int P> void run() {
    const Params p = params();
    if (has_geometry_) for (long n = 0; n < N_; ++n) pf_field_node<P, true>(p, N_, n);
    else               for (long n = 0; n < N_; ++n) pf_field_node<P, false>(p, N_, n);
    derivatives();
    if (viscous_) {
      if (has_geometry_) for (long n = 0; n < N_; ++n) pf_viscous_node<true>(p, n);
      else               for (long n = 0; n < N_; ++n) pf_viscous_node<false>(p, n);
    }
    if (fluid.constant_reference())
      for (long n = 0; n < N_; ++n) pf_pgrad_node(p, n);
    // PASS 4b, mirroring the device: the macroscopic field alone, so a
    // penalised body can be applied in the one window where u is current and
    // the collision has not happened yet. See phasefield.cuh's PASS 4b.
    if (pre_fluid_) {
      if (has_geometry_) for (long n = 0; n < N_; ++n) pf_macro_node<P, true>(p, N_, n);
      else               for (long n = 0; n < N_; ++n) pf_macro_node<P, false>(p, N_, n);
      pre_fluid_();
    }
    if (has_geometry_) for (long n = 0; n < N_; ++n) pf_fluid_node<P, true>(p, N_, n);
    else               for (long n = 0; n < N_; ++n) pf_fluid_node<P, false>(p, N_, n);
    if (has_geometry_) for (long n = 0; n < N_; ++n) pf_phase_node<P, true>(p, N_, n);
    else               for (long n = 0; n < N_; ++n) pf_phase_node<P, false>(p, N_, n);
  }

  Params params() {
    Params p;
    p.f = f_.data();  p.h = h_.data();
    p.phi = &fld_[0];
    p.gx = &fld_[std::size_t(N_)];
    p.gy = &fld_[std::size_t(2 * N_)];
    p.gz = &fld_[std::size_t(3 * N_)];
    p.lap = &fld_[std::size_t(4 * N_)];
    p.ux = &fld_[std::size_t(5 * N_)];
    p.uy = &fld_[std::size_t(6 * N_)];
    p.uz = &fld_[std::size_t(7 * N_)];
    p.pt = &fld_[std::size_t(8 * N_)];
    if (viscous_) {
      p.vx = &fld_[std::size_t(9 * N_)];
      p.vy = &fld_[std::size_t(10 * N_)];
      p.vz = &fld_[std::size_t(11 * N_)];
    }
    if (fluid.constant_reference()) {
      p.px = &fld_[std::size_t(12 * N_)];
      p.py = &fld_[std::size_t(13 * N_)];
      p.pz = &fld_[std::size_t(14 * N_)];
    }
    p.ex = ex_;  p.ey = ey_;  p.ez = ez_;
    p.pflags = pflags_.data();  p.fflags = fflags_.data();
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.pm = phase;  p.fm = fluid;
    p.pop = phase_op_;  p.fop = fluid_op_;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  std::vector<Real> f_, h_, fld_;
  std::vector<std::uint8_t> pflags_, fflags_;
  const Real *ex_ = nullptr, *ey_ = nullptr, *ez_ = nullptr;
  std::function<void()> pre_fluid_;
  bool has_geometry_ = false;
  bool viscous_ = false;
  PhaseOp phase_op_ = PhaseOp::BGK;
  MultiOp fluid_op_ = MultiOp::BGK;
  std::size_t t_ = 0;
};


//==============================================================================
//  A rigid body by volume penalisation -- host reference.
//
//  Same interface as PenalisedBody in body.cuh, running the same per-node
//  functions in a serial loop. The device version's whole difficulty is the
//  seven-accumulator reduction; here it is a running total, which is exactly
//  why building a body case on the host first is worth doing -- a wrong sign in
//  the roll equation looks identical on both and costs nothing to find here.
//==============================================================================
template <class Shape = Rect>
class Body {
 public:
  Body(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    N_ = long(nx) * ny * nz;
    fx_.assign(std::size_t(N_), Real(0));
    fy_.assign(std::size_t(N_), Real(0));
    fz_.assign(std::size_t(N_), Real(0));
  }

  Shape shape;
  BodyProperties props;
  Real vx = 0, vy = 0, vz = 0, omega = 0;

  void couple_velocity(const Real* ux, const Real* uy) { ux_ = ux; uy_ = uy; uz_ = nullptr; }
  void couple_velocity(const Real* ux, const Real* uy, const Real* uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }
  const Real* fx() const { return fx_.data(); }
  const Real* fy() const { return fy_.data(); }
  const Real* fz() const { return fz_.data(); }

  struct Moments { double area = 0, second = 0; };
  Moments indicator_moments() const {
    Moments m;
    for (long n = 0; n < N_; ++n) {
      int x, y, z;
      coords(n, nx_, ny_, x, y, z);
      const double c = double(shape.chi(Real(x), Real(y), Real(z)));
      const double rx = double(x) - double(shape.cx), ry = double(y) - double(shape.cy);
      m.area += c;
      m.second += c * (rx * rx + ry * ry);
    }
    return m;
  }
  void set_uniform_density(Real rho_b) {
    const Moments m = indicator_moments();
    props.mass    = Real(double(rho_b) * m.area);
    props.inertia = Real(double(rho_b) * m.second);
  }

  template <class DensityOf>
  BodyReaction refresh(DensityOf dens) { return refresh(dens, dens); }

  template <class LiquidOf, class LbmOf>
  BodyReaction refresh(LiquidOf dens, LbmOf ldens) {
    const BodySums q = probe(dens, ldens);
    double dux = 0, duy = 0, dw = 0, duz = 0;
    body_solve(props, q, dux, duy, dw, duz);
    if (props.free_translation) {
      vx += Real(dux); vy += Real(duy);
      if (uz_) vz += Real(duz);
    }
    if (props.free_rotation)    { omega += Real(dw); }
    apply(dens, ldens);
    return body_reaction(props, q, dux, duy, dw);
  }

  void advance() {
    shape.cx += vx;
    shape.cy += vy;
    advance_z(shape);
    shape.set_angle(shape.theta + omega);
  }

  //--------------------------------------------------------------------------
  //  THE SIX-DOF PATH, serial. The device version's difficulty is the sixteen
  //  accumulators; here it is a running total. Which is the point of having
  //  this: the 6x6 can be checked against the parent's validated 3x3 on a
  //  planar problem with no device in the loop at all.
  //--------------------------------------------------------------------------
  struct Reaction6 {
    Real fx = 0, fy = 0, fz = 0;
    Real tx = 0, ty = 0, tz = 0;
    double fluid_mass = 0;
    Real rx = 0, ry = 0, rz = 0;
  };

  Real wx = 0, wy = 0, wz = 0;     // angular velocity, world frame
  Mat3 inertia_body;               // I_b in the body frame

  BodySums6 moments6() const {
    BodySums6 q;
    for (long n = 0; n < N_; ++n) {
      int x, y, z;
      coords(n, nx_, ny_, x, y, z);
      const double c = double(shape.chi(Real(x), Real(y), Real(z)));
      if (c < 1e-6) continue;
      const double rx = double(x) - double(shape.cx);
      const double ry = double(y) - double(shape.cy);
      const double rz = double(z) - double(shape_cz(shape, 0));
      q.m += c;
      q.Jxx += c * rx * rx;  q.Jyy += c * ry * ry;  q.Jzz += c * rz * rz;
      q.Jxy += c * rx * ry;  q.Jxz += c * rx * rz;  q.Jyz += c * ry * rz;
    }
    return q;
  }

  double penalised_volume() const { return moments6().m; }

  // I_body = R^T I_world R. See body.cuh's copy for why that inverse is the
  // one that matters and why a cube cannot catch it being wrong.
  void set_uniform_density6(Real rho_b) {
    BodySums6 q = moments6();
    const double r = double(rho_b);
    q.m *= r;
    q.Jxx *= r; q.Jyy *= r; q.Jzz *= r;
    q.Jxy *= r; q.Jxz *= r; q.Jyz *= r;
    props.mass = Real(q.m);
    inertia_body = rotate_tensor(shape.Rm.transposed(), q.fluid_inertia());
  }

  template <class DensityOf>
  Reaction6 refresh6(DensityOf dens) { return refresh6(dens, dens); }

  template <class LiquidOf, class LbmOf>
  Reaction6 refresh6(LiquidOf dens, LbmOf ldens) {
    const BodySums6 q = probe6(dens, ldens);
    Body6Properties p;
    p.mass = props.mass;
    p.inertia_world = rotate_tensor(shape.Rm, inertia_body);
    // The current angular velocity, for the gyroscopic term.
    p.wx = wx;  p.wy = wy;  p.wz = wz;
    p.bx = props.bx;  p.by = props.by;  p.bz = props.bz;
    p.free_translation = props.free_translation;
    p.free_rotation = props.free_rotation;
    double dU[3], dW[3];
    body6_solve(p, q, dU, dW);
    if (props.free_translation) {
      vx += Real(dU[0]);  vy += Real(dU[1]);  vz += Real(dU[2]);
    }
    if (props.free_rotation) {
      wx += Real(dW[0]);  wy += Real(dW[1]);  wz += Real(dW[2]);
    }
    apply6(dens, ldens);

    const double S[3] = {q.Sx, q.Sy, q.Sz};
    const double WxS[3] = {dW[1] * S[2] - dW[2] * S[1],
                           dW[2] * S[0] - dW[0] * S[2],
                           dW[0] * S[1] - dW[1] * S[0]};
    const double SxU[3] = {S[1] * dU[2] - S[2] * dU[1],
                           S[2] * dU[0] - S[0] * dU[2],
                           S[0] * dU[1] - S[1] * dU[0]};
    const Mat3 If = q.fluid_inertia();
    double IfW[3];
    for (int i = 0; i < 3; ++i)
      IfW[i] = double(If(i, 0)) * dW[0] + double(If(i, 1)) * dW[1]
             + double(If(i, 2)) * dW[2];
    const double g[3] = {double(props.bx), double(props.by), double(props.bz)};
    Reaction6 out;
    out.fx = Real(2.0 * q.Px - 2.0 * (q.m * dU[0] + WxS[0]));
    out.fy = Real(2.0 * q.Py - 2.0 * (q.m * dU[1] + WxS[1]));
    out.fz = Real(2.0 * q.Pz - 2.0 * (q.m * dU[2] + WxS[2]));
    out.tx = Real(2.0 * q.Lx - 2.0 * (IfW[0] + SxU[0]));
    out.ty = Real(2.0 * q.Ly - 2.0 * (IfW[1] + SxU[1]));
    out.tz = Real(2.0 * q.Lz - 2.0 * (IfW[2] + SxU[2]));
    out.fluid_mass = q.m;
    out.rx = Real(-(S[1] * g[2] - S[2] * g[1]));
    out.ry = Real(-(S[2] * g[0] - S[0] * g[2]));
    out.rz = Real(-(S[0] * g[1] - S[1] * g[0]));
    return out;
  }

  void advance6() {
    shape.cx += vx;
    shape.cy += vy;
    shape.cz += vz;
    Quat q = shape.q;
    q.integrate(wx, wy, wz, Real(1));
    shape.set_orientation(q);
  }

  template <class LiquidOf, class LbmOf>
  BodySums6 probe6(LiquidOf dens, LbmOf ldens) const {
    const Real r2 = shape.reach() * shape.reach();
    const BodyState6 st = state6();
    BodySums6 q;
    for (long n = 0; n < N_; ++n)
      body_probe6_node(shape, st, ux_, uy_, uz_, fx_.data(), fy_.data(),
                       fz_.data(), dens, ldens, n, r2, q);
    return q;
  }

  template <class LiquidOf, class LbmOf>
  void apply6(LiquidOf dens, LbmOf ldens) {
    const Real r2 = shape.reach() * shape.reach();
    const BodyState6 st = state6();
    for (long n = 0; n < N_; ++n)
      body_apply6_node(shape, st, ux_, uy_, uz_, fx_.data(), fy_.data(),
                       fz_.data(), dens, ldens, n, r2);
  }

  template <class LiquidOf, class LbmOf>
  BodySums probe(LiquidOf dens, LbmOf ldens) const {
    const Real r2 = shape.reach() * shape.reach();
    const BodyState st = state();
    BodySums q;
    for (long n = 0; n < N_; ++n)
      body_probe_node(shape, st, ux_, uy_, uz_, fx_.data(), fy_.data(),
                      fz_.data(), dens, ldens, n, r2, q);
    return q;
  }

  template <class LiquidOf, class LbmOf>
  void apply(LiquidOf dens, LbmOf ldens) {
    const Real r2 = shape.reach() * shape.reach();
    const BodyState st = state();
    for (long n = 0; n < N_; ++n)
      body_apply_node(shape, st, ux_, uy_, uz_, fx_.data(), fy_.data(),
                      fz_.data(), dens, ldens, n, r2);
  }

 private:
  template <class S> static auto shape_cz(const S& sh, int) -> decltype(sh.cz) { return sh.cz; }
  template <class S> static Real shape_cz(const S&, long) { return Real(0); }
  template <class S> static auto bump_z(S& sh, Real dz, int) -> decltype(sh.cz, void()) { sh.cz += dz; }
  template <class S> static void bump_z(S&, Real, long) {}

  BodyState6 state6() const {
    BodyState6 st;
    st.cx = shape.cx;  st.cy = shape.cy;  st.cz = shape_cz(shape, 0);
    st.vx = vx;  st.vy = vy;  st.vz = vz;
    st.wx = wx;  st.wy = wy;  st.wz = wz;
    st.nx = nx_; st.ny = ny_;
    return st;
  }

  BodyState state() const {
    BodyState st;
    st.cx = shape.cx;  st.cy = shape.cy;  st.cz = shape_cz(shape, 0);
    st.vx = vx;  st.vy = vy;  st.vz = vz;  st.omega = omega;
    st.nx = nx_; st.ny = ny_;
    return st;
  }
  template <class S> void advance_z(S& sh) { bump_z(sh, vz, 0); }

  int nx_, ny_, nz_;
  long N_;
  std::vector<Real> fx_, fy_, fz_;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
};

//==============================================================================
//  Free surface -- host reference.
//
//  Same five passes in the same order, serially. The serial loop supplies for
//  free the fences the device gets from kernel boundaries, and every pass here
//  is a gather, so running them in a plain loop is not an approximation to what
//  the kernels do -- it is the same computation.
//
//  This is where a free-surface case should be built first. The failure mode
//  that matters is a mass ledger that does not balance, and that is visible at
//  32x32 in a second.
//==============================================================================
class FreeSurface {
 public:
  FreeSurface(int nx, int ny, int nz, Real nu)
      : nx_(nx), ny_(ny), nz_(nz), omega_(omega_from_viscosity(nu)) {
    N_ = long(nx) * ny * nz;
    fa_.assign(std::size_t(27 * N_), Real(0));
    fb_.assign(std::size_t(27 * N_), Real(0));
    fld_.assign(std::size_t(7 * N_), Real(0));
    flg_.assign(std::size_t(4 * N_), std::uint8_t(FsGas));
  }

  Real rho_G = Real(1);
  Real omega_bulk = Real(1);
  Real fill_offset = Real(1e-3);
  bool drop_detached = true;

  void set_gravity(Real ax, Real ay, Real az = Real(0)) { gx_ = ax; gy_ = ay; gz_ = az; }

  void set_geometry(const std::vector<std::uint8_t>& flags) {
    for (long n = 0; n < N_; ++n) flg_[std::size_t(n)] = flags[std::size_t(n)];
  }

  template <class Init>
  void initialise_with(Init init) {
    const FsParams p = params();
    for (long n = 0; n < N_; ++n) fs_init_node(p, N_, n, init);
    t_ = 0;
    close_interface();
  }

  void step() {
    const FsParams p = params();
    for (long n = 0; n < N_; ++n) fs_stream_collide_node(p, N_, n);
    for (long n = 0; n < N_; ++n) fs_mass_exchange_node(p, N_, n);
    for (long n = 0; n < N_; ++n) fs_classify_node(p, n);
    close_interface();
    fa_.swap(fb_);
    ++t_;
  }

  void field_to_host(const Real* src, std::vector<Real>& out) {
    out.assign(src, src + N_);
  }
  void flags_to_host(std::vector<std::uint8_t>& out) {
    out.assign(flg_.begin(), flg_.begin() + N_);
  }

  // The one property that can be silently lost. Summed in double even in an
  // FP32 build; see the banner.
  double total_mass() const {
    double s = 0;
    for (long n = 0; n < N_; ++n) s += double(fld_[std::size_t(n)]);
    return s;
  }

  const Real* mass_device() const { return &fld_[0]; }
  const Real* eps_device()  const { return &fld_[std::size_t(N_)]; }
  const Real* rho_device()  const { return &fld_[std::size_t(3 * N_)]; }
  const Real* ux_device()   const { return &fld_[std::size_t(4 * N_)]; }
  const Real* uy_device()   const { return &fld_[std::size_t(5 * N_)]; }
  const Real* uz_device()   const { return &fld_[std::size_t(6 * N_)]; }
  std::size_t timestep() const { return t_; }
  long nodes() const { return N_; }

 private:
  void close_interface() {
    const FsParams p = params();
    for (long n = 0; n < N_; ++n) fs_promote_node(p, n);
    for (long n = 0; n < N_; ++n) fs_settle_node(p, N_, n);
    for (long n = 0; n < N_; ++n) {
      flg_[std::size_t(n)] = flg_[std::size_t(2 * N_ + n)];
      flg_[std::size_t(N_ + n)] = flg_[std::size_t(2 * N_ + n)];
    }
  }

  FsParams params() {
    FsParams p;
    p.src = fa_.data();  p.dst = fb_.data();
    p.flags  = &flg_[0];
    p.newf   = &flg_[std::size_t(N_)];
    p.fin    = &flg_[std::size_t(2 * N_)];
    p.reinit = &flg_[std::size_t(3 * N_)];
    p.mass   = &fld_[0];
    p.eps    = &fld_[std::size_t(N_)];
    p.excess = &fld_[std::size_t(2 * N_)];
    p.rho    = &fld_[std::size_t(3 * N_)];
    p.ux     = &fld_[std::size_t(4 * N_)];
    p.uy     = &fld_[std::size_t(5 * N_)];
    p.uz     = &fld_[std::size_t(6 * N_)];
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_;  p.omega_bulk = omega_bulk;
    p.rho_G = rho_G;
    p.gx = gx_; p.gy = gy_; p.gz = gz_;
    p.fill_offset = fill_offset;
    p.drop_detached = drop_detached;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real omega_;
  Real gx_ = 0, gy_ = 0, gz_ = 0;
  std::vector<Real> fa_, fb_, fld_;
  std::vector<std::uint8_t> flg_;
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
