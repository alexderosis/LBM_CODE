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
#include "colour.cuh"
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
  void set_magic(Real lambda) { omega_minus_ = omega_minus_for(omega_, lambda); }
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
    p.omega_minus = omega_minus_;
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
    } else {
      run<P, O, ForceNone, false>();
    }
  }
  template <int P, int O, int F, bool M> void run() {
    const FluidParams p = params();
    if (has_geometry_)
      for (long n = 0; n < N_; ++n) fluid_node_update<P, O, F, M, true>(p, N_, n);
    else
      for (long n = 0; n < N_; ++n) fluid_node_update<P, O, F, M, false>(p, N_, n);
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
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real T_ref_, omega_;
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
  bool has_geometry_ = false;
  bool viscous_ = false;
  PhaseOp phase_op_ = PhaseOp::BGK;
  MultiOp fluid_op_ = MultiOp::BGK;
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
