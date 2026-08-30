#pragma once
//==============================================================================
//  Free-surface flow: a sharp interface, and only the liquid.
//
//  THE THIRD ROUTE TO A MULTIPHASE FLOW IN THIS CODE, and the one that changes
//  the bargain rather than the parameters. PhaseFieldSolver.hpp and
//  ColourGradientSolver.hpp both resolve BOTH fluids, and everything difficult
//  about them follows from that: the pressure-gauge conditioning at a density
//  ratio, the alpha interpolation, the recolouring. This one does not resolve
//  the gas at all. The gas is a VOID held at a prescribed pressure -- it has no
//  populations, no velocity and no dynamics -- and the liquid meets it across a
//  boundary condition rather than across a diffuse interface.
//
//  What that buys is the thing the other two fight for. There is no density
//  ratio here because there is no second density: the ratio is infinite by
//  construction, and it costs nothing, because the expensive half of the problem
//  was deleted rather than discretised. It is the standard tool for violent
//  free-surface flow -- dam break, sloshing, filling, splashing -- and it is
//  useless for anything where the gas does work: a rising bubble, a driven
//  gas-liquid shear layer, anything with a Weber number the gas participates in.
//
//      module            interface   fluids resolved   density ratio
//      phase field       diffuse     both              ~100, with the gauge problem
//      colour gradient   diffuse     both              1000 in principle
//      free surface      sharp       liquid only       infinite, by construction
//
//  PROVENANCE, STATED PLAINLY. This follows the standard formulation of Korner,
//  Thies, Hofmann, Theobald & Rude, J. Stat. Phys. 121, 179 (2005), and the
//  refinements in Thurey's and Rude's later work. Unlike the colour-gradient
//  module there is no paper in SomeRefs/ to check equation by equation, so
//  nothing here cites an equation number and every choice below is justified
//  from the mechanics rather than by reference. Where the literature offers a
//  refinement this code does not implement, it says so.
//
//==============================================================================
//  THE STATE. Every cell is Gas, Interface or Fluid (or a wall, or halo). Fluid
//  cells are ordinary LBM cells. Gas cells hold nothing. Interface cells are the
//  entire method: they carry an explicit MASS m alongside their populations, and
//  a fill level
//
//      epsilon = m / rho,
//
//  which is 1 when the cell is full and 0 when it is empty. The interface is a
//  closed, one-cell-thick shell separating Fluid from Gas, and keeping it closed
//  through arbitrary topology change is most of the work.
//
//  MASS IS ADVECTED, NOT A MOMENT, and that is the sharpest difference from
//  everything else in this tree. Elsewhere density is sum_i f_i and conservation
//  is a property of the collision operator, checkable by algebra. Here mass moves
//  because populations cross faces, and the exchange between two cells is
//
//      dm_i(x) = f_ibar(x + c_i) - f_i(x)          both cells Fluid or Interface,
//
//  scaled by the mean fill level when both are Interface, because a half-full
//  cell can only exchange half as much. Nothing enforces conservation but the
//  antisymmetry of that expression: what x sends to its neighbour is exactly
//  what the neighbour receives from x. That is why validation checks total mass
//  to round-off first and everything else second -- it is the one property that
//  can be silently lost.
//
//  THE FREE-SURFACE CONDITION reconstructs the populations that would have
//  arrived from the gas. For every direction i whose source cell x - c_i is Gas,
//
//      f_i(x) = f_i^eq(rho_G, u) + f_ibar^eq(rho_G, u) - f_ibar(x),
//
//  with rho_G fixing the gas pressure through p = rho_G cs^2. Its two moments
//  are what the free surface is: the normal stress equals the gas pressure, and
//  the tangential stress vanishes. f_ibar(x) here is the post-collision
//  population at x from the previous step, which the two-lattice scheme still
//  holds in its source array -- no extra storage is needed to find it.
//
//  THE VELOCITY IN THAT RECONSTRUCTION IS ONE STEP OLD. u is needed to build the
//  equilibria, and u comes from a sum over the populations, some of which are
//  the ones being reconstructed. The circle is broken with the previous step's
//  velocity, which is first-order in time and is what the literature does. It
//  matters where the interface accelerates hard; it is invisible in a standing
//  wave.
//
//==============================================================================
//  WHY TWO-LATTICE STREAMING, when everything else in this code uses Esoteric
//  Pull. Three requirements, and Esoteric Pull fails two of them outright.
//
//   1. Mass exchange needs f_i(x) and f_ibar(x + c_i) as POST-COLLISION values,
//      after the whole domain has collided. Under an in-place scheme those slots
//      hold a mixture of two time levels, and which one depends on parity and on
//      the direction's index. Under two-lattice they are simply the destination
//      array, complete and unambiguous, once the collide pass has fenced.
//   2. A cell that fills up turns its gas neighbours into interface cells, which
//      then need populations. In place, a cell's populations live in its
//      neighbours' slots -- initialising a neighbour means writing storage that
//      other cells are concurrently reading through their own aliases.
//   3. Reading a neighbour's post-collision state while writing one's own is the
//      normal case here and impossible there.
//
//  Requirement 2 is dodged rather than met: every pass below is a GATHER. A cell
//  that is about to become interface initialises ITSELF by averaging its
//  neighbours, and a cell with excess mass to give away publishes it in a field
//  that its neighbours collect. Nothing ever writes another cell's memory, so
//  there is no race to reason about and no atomics. That is a deliberate shape,
//  not an accident of the streaming choice -- it would be worth keeping even if
//  scatters were free.
//
//  The cost is two population arrays instead of one. For a method that deletes
//  an entire phase, that is not the expensive part.
//
//==============================================================================
//  WHAT THIS DOES NOT MODEL.
//
//   * NO SURFACE TENSION. rho_G is uniform, so the gas pressure carries no
//     curvature term. Adding it means p_G = p_atm - sigma kappa with kappa from
//     the fill-level field, and the machinery for a curvature is not here.
//     Everything at a Bond number where surface tension competes with gravity is
//     therefore out of scope, which includes most droplet problems and none of
//     the dam-break family.
//   * NO GAS DYNAMICS, and this is the method, not an omission. An enclosed
//     bubble does not compress, because its pressure is prescribed rather than
//     solved. Modelling that needs a volume-of-gas tracker with a separate
//     bubble-merging graph, which is a larger piece of work than this file.
//   * THE RECONSTRUCTION IS APPLIED TO GAS-FACING DIRECTIONS ONLY. The
//     literature also reconstructs directions with c_i . n > 0 for the interface
//     normal n, which stabilises interfaces that are nearly tangential to the
//     lattice. That refinement is not here, and if a case shows interface
//     roughening along a diagonal this is the first thing to add.
//   * EXCESS MASS IS REDISTRIBUTED UNIFORMLY among eligible neighbours rather
//     than weighted by the interface normal. Uniform redistribution conserves
//     mass exactly -- which is what matters most -- but is known to be more
//     diffusive at a sharply curved interface than the normal-weighted form.
//   * NO SUBGRID INTERFACE. The interface is resolved to one cell, so a film
//     thinner than a cell either survives as a one-cell layer or vanishes.
//   * A MOVING OBSTACLE IS NOT YET RELIABLE, and the reason is in this file
//     rather than in the caller. mass_exchange() conserves mass by antisymmetry
//     -- what one cell gives across a face, its neighbour takes -- and that
//     assumes BOTH cells agree on the mask for the whole pass. move_obstacle()
//     changes the mask between passes, so every cell the body sweeps breaks the
//     assumption. transfer_covered_mass() keeps the TOTAL right but does not
//     restore the per-face antisymmetry, and the difference shows up wherever
//     many mask cells change at once: a flat face landing on a flat surface, a
//     corner entering, a rotated rectangle's staircase flipping cells as it
//     turns. demonstrator/water_entry_fs.cpp measures all three. The fix is to
//     make the mass exchange itself aware of the moving boundary; bounce-back,
//     momentum exchange and refill are all in place and are not the problem.
//==============================================================================
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"
#include "memory/TwoLattice.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Cell states. A distinct enum from CellType and from PhaseCell, with values
// that deliberately do NOT line up with either: all three are stored as uint8
// and mixing them would compile silently.
//------------------------------------------------------------------------------
enum FsCell : std::uint8_t {
  FsGas       = 10,   // no liquid; not simulated, holds nothing
  FsInterface = 11,   // partially filled; carries mass and the free surface
  FsFluid     = 12,   // full; an ordinary LBM cell
  FsSolid     = 13,   // wall: halfway bounce-back
  FsExcluded  = 14,   // halo
};

//------------------------------------------------------------------------------
// CENTRAL MOMENTS, not BGK, and gravity through Guo.
//
// The collision is a policy exactly as it is for FluidSolver, and the default is
// the central-moment operator of MomentCollision.hpp. That is the same choice
// the Rayleigh-Taylor cases make and for the same reason: BGK relaxes every mode
// at omega, and a free-surface flow worth simulating -- a collapsing column, a
// breaking wave -- reaches the same high-Reynolds regime where that stops being
// survivable. It costs a moment transform per cell against a scheme that has
// already deleted an entire phase.
//
// The force is Guo's with a CONSTANT vector, which is gravity at rho_0 = 1. That
// is exact for this method rather than an approximation: the liquid is nearly
// incompressible, the gas has no weight because it has no populations, and an
// interface cell's rho is close to one whatever its fill level -- the fill level
// is carried by the mass, not by the density.
//------------------------------------------------------------------------------
template <class L, class Collision = CentralMoments<L, Guo, RawPopulations>>
class FreeSurfaceSolver {
 public:
  using Lattice   = L;
  using Streaming = TwoLattice<L>;
  static constexpr int Q  = L::Q;
  static constexpr int D  = L::D;
  static constexpr int NF = Streaming::nb_first;
  static constexpr int NS = Streaming::nb_stride;

  static_assert(L::supports_navier_stokes,
                "the free surface is a Navier-Stokes solver and needs a lattice "
                "with isotropic fourth-order moments.");
  // D3Q19 is deferred, not excluded on principle: the solver's own machinery is
  // lattice-generic and was measured on it, but the central-moment operator
  // reaches D3Q19 through the monomial basis rather than the product one, and
  // that pairing has not been exercised here. D2Q9 and D3Q27 are the product
  // lattices and are what this runs on.
  static_assert(ProductBasis<L>::enabled,
                "the free surface currently pairs with the product-basis central "
                "moment operator: D2Q9 or D3Q27.");

  explicit FreeSurfaceSolver(const Domain& dom)
      : dom_(dom), pop_(dom),
        flags_("fs_flags", dom.n_padded),
        newf_("fs_newflags", dom.n_padded),
        final_("fs_finalflags", dom.n_padded),
        reinit_("fs_reinit", dom.n_padded),
        wux_("fs_wux", dom.n_padded), wuy_("fs_wuy", dom.n_padded),
        wuz_("fs_wuz", dom.n_padded), obst_("fs_obst", dom.n_padded),
        was_body_("fs_wasbody", dom.n_padded),
        mass_("fs_mass", dom.n_padded),
        eps_("fs_eps", dom.n_padded),
        excess_("fs_excess", dom.n_padded),
        rho_("fs_rho", dom.n_padded),
        ux_("fs_ux", dom.n_padded), uy_("fs_uy", dom.n_padded),
        uz_("fs_uz", dom.n_padded) {
    h_flags_ = Kokkos::create_mirror_view(flags_);
    for (Index n = 0; n < dom.n_padded; ++n) {
      Index px, py, pz; dom_.coords(n, px, py, pz);
      h_flags_(n) = dom_.is_interior(px, py, pz) ? FsGas : FsExcluded;
    }
    Kokkos::deep_copy(flags_, h_flags_);
  }

  //---- parameters -------------------------------------------------------------
  Collision coll{};              // omega, omega_bulk and the force live here
  Real rho_G  = Real(1);         // gas density: p_G = rho_G cs^2
  // Conversion hysteresis. A cell is promoted only past 1 + fill_offset and
  // demoted only below -fill_offset, so a cell sitting exactly at a threshold
  // cannot convert back and forth every step. The literature's value; without
  // it, a quiescent interface flickers and the flicker radiates noise.
  Real fill_offset = Real(1e-3);
  bool drop_detached = true;   // isolated specks -> gas; see classify()

  static Real omega_from_viscosity(Real nu) {
    return Collision::omega_from_viscosity(nu);
  }
  static Real viscosity_from_omega(Real w) {
    return Collision::viscosity_from_omega(w);
  }
  // Gravity, as the constant Guo force it is.
  void set_gravity(Real ax, Real ay, Real az = Real(0)) {
    coll.forcing.fx = ax;  coll.forcing.fy = ay;  coll.forcing.fz = az;
  }

  //----------------------------------------------------------------------------
  // fn(x, y, z) -> FsCell, over interior coordinates. Call BEFORE initialize().
  //----------------------------------------------------------------------------
  template <class Fn>
  void set_geometry(Fn fn) {
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x)
          h_flags_(dom_.id(x, y, z)) = fn(x, y, z);
    Kokkos::deep_copy(flags_, h_flags_);
    Kokkos::deep_copy(newf_, flags_);
    Kokkos::deep_copy(final_, flags_);
  }

  //----------------------------------------------------------------------------
  // fn(n) -> Seed. `fill` in [0, 1] describes the liquid: 1 seeds Fluid, 0 seeds
  // Gas, anything between seeds an Interface cell, and the classification and
  // one-cell interface shell follow from it. `rho` seeds the pressure through
  // p = rho cs^2.
  //
  // THE DENSITY IS WORTH SETTING. Left at 1 the liquid starts with no pressure
  // gradient at all, so a column under gravity has to build its own hydrostatic
  // profile, and it does that by ringing: an acoustic transient that crosses the
  // domain in a few hundred steps and takes far longer to damp. That is harmless
  // in a dam break, where the flow is violent enough not to care, and ruinous in
  // a standing-wave measurement, where it is noise at the amplitude being
  // measured. rho = rho_G + g (y_surface - y) / cs^2 starts it quiet.
  //----------------------------------------------------------------------------
  struct Seed { Real fill = 0, rho = Real(1); };

  template <class Fn>
  void initialize(Fn fn) {
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_, newf = newf_;
    auto mass = mass_, eps = eps_, rho = rho_;
    auto ux = ux_, uy = uy_, uz = uz_;
    Kokkos::parallel_for("fs_init", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t was = flags(n);
      Real e = Real(0), r = Real(1);
      if (was != FsSolid && was != FsExcluded) {
        const Seed sd = fn(n);
        e = sd.fill;
        r = sd.rho;
        e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
        flags(n) = (e >= Real(1)) ? FsFluid : (e <= Real(0) ? FsGas : FsInterface);
      }
      newf(n) = flags(n);
      rho(n) = r;
      ux(n) = Real(0); uy(n) = Real(0); uz(n) = Real(0);
      eps(n)  = e;
      mass(n) = e * r;
      // Every cell is given populations, walls and gas included. A wall holds
      // in-transit populations exactly as it does elsewhere in this code, and a
      // gas cell that later becomes interface is initialised properly then --
      // but leaving either as raw zeros makes the first step read garbage.
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      for (int i = 0; i < Q; ++i) {
        const Real v = Collision::seed_value(i, r, Real(0), Real(0), Real(0));
        acc.src(n, i) = v;
        acc.dst(n, i) = v;
      }
    });
    Kokkos::fence();
    // The seeded fill levels need not describe a closed interface -- a caller
    // may hand over a sharp step. One repair pass turns it into one.
    close_interface();
  }

  //----------------------------------------------------------------------------
  // One step. Five fenced passes; see the banner for why they cannot be fewer.
  //----------------------------------------------------------------------------
  void step() {
    stream_collide();
    mass_exchange();
    classify();
    close_interface();
    pop_.end_of_step();
    ++t_;
  }

  //---- accessors --------------------------------------------------------------
  View1D<std::uint8_t> flags() const { return flags_; }
  //----------------------------------------------------------------------------
  // Non-zero where this cell's populations AND stored velocity were rebuilt from
  // its neighbours during the step just finished -- that is, where a gas cell
  // became an interface cell and had to be given a state from nothing.
  //
  // ANY MODULE THAT FORCED THE FLUID LAST STEP NEEDS THIS. A direct-forcing
  // scheme like PenalisedBody applies a force, then on the next step subtracts
  // its own contribution back out of the stored velocity. That subtraction is
  // only valid if the velocity still contains the contribution -- and in a
  // reinitialised cell it does not, because the average of the neighbours
  // replaced it. Subtracting a force whose effect was erased is the same
  // runaway, and it is not small: it is what a body touching a free surface
  // does to itself within a few hundred steps.
  //----------------------------------------------------------------------------
  View1D<std::uint8_t> reinitialised() const { return reinit_; }

  //----------------------------------------------------------------------------
  // The force and torque the fluid exerted on the obstacle during the last
  // step, by momentum exchange. Unlike PenalisedBody's reaction -- which is a
  // closed form derived from the solve -- this is a genuine measurement: it is
  // summed link by link over the boundary from the populations that actually
  // bounced.
  //----------------------------------------------------------------------------
  struct Exchange { Real fx = 0, fy = 0, torque = 0; };
  Exchange exchange() const { return Exchange{fx_, fy_, tz_}; }

  //----------------------------------------------------------------------------
  // Move the obstacle. `inside(n)` says whether cell n is now within the body;
  // `wall_vel(n, u[3])` gives the body's surface velocity there. Both are device
  // lambdas, so the solver needs no notion of a body.
  //
  // THIS RUNS BEFORE step(), and it must: a cell that is solid at the start of a
  // step can never be given a state during one -- promote() and settle() both
  // return early on FsSolid -- so refill has to happen first or not at all.
  //
  // MASS IS MOVED FROM THE FRONT OF THE BODY TO THE BACK, which is what a solid
  // sweeping through liquid physically does. The cells the body newly COVERS
  // hand their mass to the cells it newly UNCOVERS, in equal shares. Both sets
  // are the same size to within the body's motion in one step, so the transfer
  // is local, and it is exactly conservative by construction rather than by an
  // argument about fluxes. If nothing was uncovered -- a body that has stopped --
  // the covered mass goes into the ordinary excess machinery instead, and if
  // that has no takers either it is reported as lost rather than dropped
  // silently.
  //----------------------------------------------------------------------------
  template <class InsideFn, class WallFn>
  void move_obstacle(InsideFn inside, WallFn wall_vel, Real cx, Real cy) {
    bcx_ = cx;  bcy_ = cy;
    mark_obstacle(inside, wall_vel);
    transfer_covered_mass();
  }
  View1D<Real> fill() const { return eps_; }
  View1D<Real> mass() const { return mass_; }
  View1D<Real> rho()  const { return rho_; }
  View1D<Real> ux() const { return ux_; }
  View1D<Real> uy() const { return uy_; }
  View1D<Real> uz() const { return uz_; }
  const Domain& domain() const { return dom_; }
  std::size_t timestep() const { return t_; }

  // The conserved quantity, and the one that can be silently lost. Fluid cells
  // count their full density; interface cells count their explicit mass.
  Real total_mass() const {
    auto flags = flags_; auto mass = mass_; auto rho = rho_;
    Real s = 0;
    Kokkos::parallel_reduce("fs_mass", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc) {
        const std::uint8_t f = flags(n);
        if (f == FsFluid)          acc += rho(n);
        else if (f == FsInterface) acc += mass(n);
      }, s);
    return s;
  }

  // How many cells of each kind -- the cheapest way to see an interface that is
  // dissolving or a conversion storm.
  struct Census { Index gas = 0, interface_ = 0, fluid = 0; };
  Census census() const {
    auto flags = flags_;
    Index g = 0, i2 = 0, fl = 0;
    Kokkos::parallel_reduce("fs_census", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Index& a, Index& b, Index& c) {
        const std::uint8_t f = flags(n);
        if (f == FsGas) ++a; else if (f == FsInterface) ++b; else if (f == FsFluid) ++c;
      }, g, i2, fl);
    return Census{g, i2, fl};
  }

 public:
  //----------------------------------------------------------------------------
  // NVCC CONSTRAINT, NOT A DESIGN CHOICE -- the same note FluidSolver,
  // PhaseFieldSolver and ColourGradientSolver carry, for the same reason: CUDA
  // forbids an extended host-device lambda inside a private member function.
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // Edit the mask, and refill what the body uncovered.
  //
  // REFILL IS EQUILIBRIUM AT THE WALL VELOCITY PLUS THE DONOR'S NON-EQUILIBRIUM.
  // A cell that was solid holds no distribution -- its slots carry reflected
  // populations, whose moments are not a fluid state -- so all Q populations
  // have to be constructed. The cheap choice is equilibrium alone, which is what
  // settle() uses for a gas cell arriving from nothing; it is wrong here,
  // because it discards the local viscous stress and the very next momentum
  // exchange on this cell's links reports that stress as a force. Copying the
  // non-equilibrium part from ONE fluid neighbour keeps it, needs no collinear
  // stencil -- so it does not fail in a narrow gap or against a thin body, where
  // a three-point normal extrapolation has nowhere to stand -- and cannot
  // overshoot, because it adds a valid non-equilibrium to a valid equilibrium.
  //
  // The donor is the fluid neighbour most nearly along the outward normal, which
  // here is approximated by the direction away from the body centre. seed_value
  // rather than Equilibrium::eq, so the refill agrees with whatever operator is
  // plugged in and with its storage convention.
  //----------------------------------------------------------------------------
  template <class InsideFn, class WallFn>
  void mark_obstacle(InsideFn inside, WallFn wall_vel) {
    Kokkos::deep_copy(was_body_, obst_);     // the mask before this move
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_, newf = newf_, fin = final_, reinit = reinit_;
    auto mass = mass_, eps = eps_, rho = rho_, excess = excess_;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto wux = wux_, wuy = wuy_, wuz = wuz_;
    auto obst = obst_;
    const Index hx = dom_.hx, hy = dom_.hy;
    const Real cx = bcx_, cy = bcy_;

    // Pass A: the mask itself, plus the wall velocity the body's cells need.
    Kokkos::parallel_for("fs_mask", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Real uw[3] = {Real(0), Real(0), Real(0)};
      wall_vel(n, uw);
      wux(n) = uw[0]; wuy(n) = uw[1]; wuz(n) = uw[2];
      const std::uint8_t was = flags(n);
      const std::uint8_t was_body = obst(n);
      if (was == FsExcluded) { fin(n) = was; obst(n) = 0; return; }
      const bool in = inside(n);
      // Only the BODY's cells carry a wall velocity; a static wall is at rest
      // and must stay there.
      if (!in) { wux(n) = Real(0); wuy(n) = Real(0); wuz(n) = Real(0); }
      obst(n) = in ? 1 : 0;
      if (in) {
        fin(n) = FsSolid;
        if (!was_body && was != FsSolid) {
          excess(n) = mass(n); mass(n) = Real(0); eps(n) = Real(0);
        }
      } else {
        // Only a cell the BODY has left needs refilling; a static wall never
        // becomes anything.
        fin(n) = was_body ? FsInterface : was;
      }
    });
    Kokkos::fence();

    // Pass B: refill, reading the OLD flags for donors so nothing half-updated
    // is used as a source.
    auto obst_prev = was_body_;
    Kokkos::parallel_for("fs_refill", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (!(obst_prev(n) && fin(n) != FsSolid)) return;
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      const Real rx = Real(Index(n % 1) ) ;  (void)rx;
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real ox = Real(px - hx) - cx, oy = Real(py - hy) - cy;
      const Real om = Kokkos::sqrt(ox * ox + oy * oy);
      const Real nx2 = (om > Real(0)) ? ox / om : Real(0);
      const Real ny2 = (om > Real(0)) ? oy / om : Real(0);

      // The fluid neighbour most nearly along the outward normal.
      Index donor = n;  Real best = Real(-2);
      for (int i = 1; i < Q; ++i) {
        const Index j = nb.j[i];
        const std::uint8_t fj = flags(j);
        if (fj != FsFluid && fj != FsInterface) continue;
        const Real cxx = Real(cvel<L>(i, 0)), cyy = Real(cvel<L>(i, 1));
        const Real cm = Kokkos::sqrt(cxx * cxx + cyy * cyy);
        const Real dot = (cm > Real(0)) ? (cxx * nx2 + cyy * ny2) / cm : Real(-2);
        if (dot > best) { best = dot; donor = j; }
      }

      const Real uw[3] = {wux(n), wuy(n), wuz(n)};
      if (donor == n) {                       // no fluid neighbour at all
        for (int i = 0; i < Q; ++i)
          acc.dst(n, i) = Collision::seed_value(i, Real(1), uw[0], uw[1], uw[2]);
        rho(n) = Real(1);
      } else {
        const Real rd = rho(donor), ud = ux(donor), vd = uy(donor), wd = uz(donor);
        for (int i = 0; i < Q; ++i) {
          const Real fneq = acc.src(donor, i)
                          - Collision::seed_value(i, rd, ud, vd, wd);
          acc.dst(n, i) = Collision::seed_value(i, rd, uw[0], uw[1], uw[2]) + fneq;
        }
        rho(n) = rd;
      }
      // TwoLattice swaps at end_of_step, so a refill written to dst would only
      // be seen after the coming step's pull. Write both.
      for (int i = 0; i < Q; ++i) acc.src(n, i) = acc.dst(n, i);
      ux(n) = uw[0]; uy(n) = uw[1]; uz(n) = uw[2];
      mass(n) = Real(0); eps(n) = Real(0);
      reinit(n) = 1;
    });
    Kokkos::fence();
    Kokkos::deep_copy(flags_, final_);
    Kokkos::deep_copy(newf_, final_);
  }

  // The covered cells' mass, handed to the uncovered ones. Exactly conservative:
  // one reduction to find how much and how many, one pass to hand it over.
  void transfer_covered_mass() {
    auto flags = flags_; auto excess = excess_, mass = mass_, eps = eps_, rho = rho_;
    auto reinit = reinit_;
    Real m_cov = 0; Index n_unc = 0;
    Kokkos::parallel_reduce("fs_cover_sum", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc_m, Index& acc_n) {
        if (flags(n) == FsSolid) acc_m += excess(n);
        else if (reinit(n) && flags(n) == FsInterface) ++acc_n;
      }, m_cov, n_unc);
    Kokkos::fence();
    if (n_unc == 0) return;                  // leave it to the excess machinery
    const Real share = m_cov / Real(n_unc);
    Kokkos::parallel_for("fs_cover_give", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (flags(n) == FsSolid) { excess(n) = Real(0); return; }
      if (reinit(n) && flags(n) == FsInterface) {
        mass(n) = share;
        const Real r = rho(n);
        eps(n) = (r > Real(0)) ? share / r : Real(0);
      }
    });
    Kokkos::fence();
  }

  // Pass 1. Pull, reconstruct the gas-facing populations, collide, store.
  void stream_collide() {
    const auto acc = pop_.template access<0>();
    const auto cl  = coll;
    const Domain d = dom_;
    auto flags = flags_;
    auto rho = rho_, ux = ux_, uy = uy_, uz = uz_, mass = mass_, eps = eps_;
    auto wux = wux_, wuy = wuy_, wuz = wuz_;
    auto obst = obst_;
    const Real rg = rho_G;
    const Index hx = dom_.hx, hy = dom_.hy;
    const Real bcx = bcx_, bcy = bcy_;
    constexpr Real ics = inv_cs2<L, Real>();

    Real sfx = 0, sfy = 0, stz = 0;
    Kokkos::parallel_reduce("fs_stream_collide", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& afx, Real& afy, Real& atz) {
        const std::uint8_t fl = flags(n);
        if (fl != FsFluid && fl != FsInterface) return;

        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);

        // The velocity that goes into the reconstruction is the previous step's;
        // see the banner. rho_G, not rho(n), sets the gas pressure.
        const Real u0[3] = {ux(n), uy(n), uz(n)};

        Real f[Q];
        for (int i = 0; i < Q; ++i) {
          const Index from = (i == 0) ? n : nb.j[opp(i)];
          const std::uint8_t ff = flags(from);
          if (ff == FsGas) {
            // Free-surface condition: impose the gas pressure through the two
            // moments of an anti-bounce-back pair, against this cell's own
            // outgoing population in the opposite direction. The equilibrium is
            // the COLLISION's own, so the boundary condition and the operator
            // agree about what equilibrium means.
            f[i] = Collision::seed_value(i, rg, u0[0], u0[1], u0[2])
                 + Collision::seed_value(opp(i), rg, u0[0], u0[1], u0[2])
                 - acc.src(n, opp(i));
          } else if (ff == FsSolid || ff == FsExcluded) {
            // HALFWAY BOUNCE-BACK OFF A MOVING WALL, and the momentum it
            // delivers. `i` is the direction the population comes BACK along, so
            // the wall lies along c_opp(i) and the population that went into it
            // is acc.src(n, opp(i)).
            //
            // The correction is exactly 2 w_i rho (c_i . u_w) / cs^2. It is not
            // an expansion: it is the difference between the equilibria in
            // direction i and opp(i) at the wall velocity, whose even parts
            // cancel identically, and it is what makes a fluid moving WITH the
            // wall a fixed point of the boundary. The 1/cs^2 is written as
            // inv_cs2 and never folded into a literal 6 -- that is only right
            // for the cs^2 = 1/3 lattices.
            const Real fout = acc.src(n, opp(i));
            const Real cuw = Real(cvel<L>(i, 0)) * wux(from)
                           + Real(cvel<L>(i, 1)) * wuy(from)
                           + Real(cvel<L>(i, 2)) * wuz(from);
            const bool is_body = obst(from) != 0;
            const Real corr = is_body
                ? Real(2) * weight<L, Real>(i) * rho(n) * cuw * ics : Real(0);
            f[i] = fout + corr;
            // MOMENTUM EXCHANGE. What the link hands the wall is
            // c_q (f_out + f_back) with c_q = -c_i: the two travel in opposite
            // directions, so after the reversal their momenta ADD -- the ball
            // that bounces delivers 2mv, not mv. Summing the pair rather than
            // the expanded form keeps this correct if the closure ever changes.
            //
            // Only the liquid pushes: an interface cell a tenth full has a tenth
            // of the fluid on that link, so the exchange is scaled by the fill.
            if (is_body) {
              // THE GAS PRESSURE HAS TO BE PUT BACK. Momentum exchange delivers
              // the ABSOLUTE pressure force on each link. Around a fully wetted
              // closed body that cancels -- sum_links w_q c_q = 0 -- and only
              // the gradient survives, which is buoyancy. A partly submerged
              // body has no such cancellation: its dry side faces gas, and the
              // gas is a void with no populations, so it pushes back with
              // nothing. The body then feels a whole atmosphere on its wetted
              // face and none on top.
              //
              // Measured, that is not a small bias: a square entering at Mach
              // 0.07 reversed its velocity within forty steps under a force
              // twenty times the physical slamming load.
              //
              // The fix needs no extra pass. Since the dry links would carry
              // exactly 2 w_q rho_G each and the whole closed surface sums to
              // zero, subtracting that same constant from every WET link is
              // identical to adding it on every dry one. What is left is the
              // gauge pressure, which is what a surface force is.
              const Real wgt = (fl == FsInterface) ? eps(n) : Real(1);
              const Real p = wgt * (fout + f[i]
                                  - Real(2) * weight<L, Real>(i) * rg);
              const Real dx = -Real(cvel<L>(i, 0)) * p;
              const Real dy = -Real(cvel<L>(i, 1)) * p;
              afx += dx;  afy += dy;
              Index qx, qy, qz; d.coords(n, qx, qy, qz);
              atz += (Real(qx - hx) - bcx) * dy - (Real(qy - hy) - bcy) * dx;
            }
          } else {
            f[i] = acc.src(from, i);
          }
        }

        // The collision operator owns the macroscopics, the force and the
        // relaxation. Everything above this line is the free surface; everything
        // below it is the same operator the single-phase solver runs.
        const Macro mac = cl.macroscopic(f, n);
        const Real r = Collision::density(mac);
        cl.collide(f, mac, n);
        for (int i = 0; i < Q; ++i) acc.dst(n, i) = f[i];

        rho(n) = r;
        ux(n) = mac.ux; uy(n) = mac.uy; uz(n) = mac.uz;
        if (fl == FsFluid) mass(n) = r;    // a full cell's mass IS its density
      }, sfx, sfy, stz);
    Kokkos::fence();
    fx_ = sfx;  fy_ = sfy;  tz_ = stz;
  }

  // Pass 2. Mass across faces, from the post-collision state of both cells.
  void mass_exchange() {
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_; auto mass = mass_, eps = eps_;

    Kokkos::parallel_for("fs_mass_exchange", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        if (flags(n) != FsInterface) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        const Real e0 = eps(n);
        Real dm = Real(0);
        for (int i = 1; i < Q; ++i) {
          const Index j = nb.j[i];
          const std::uint8_t fj = flags(j);
          // What j sends me in the opposite direction, minus what I send j.
          const Real in  = acc.dst(j, opp(i));
          const Real out = acc.dst(n, i);
          if (fj == FsFluid) {
            dm += in - out;
          } else if (fj == FsInterface) {
            // Two partly filled cells can only exchange in proportion to how
            // much liquid there is to exchange. The mean of the two fill levels
            // is symmetric between them, which is what keeps the pair's total
            // mass unchanged.
            dm += Real(0.5) * (e0 + eps(j)) * (in - out);
          }
          // Gas and wall neighbours exchange nothing.
        }
        mass(n) += dm;
      });
    Kokkos::fence();
  }

  //----------------------------------------------------------------------------
  // Pass 3. Fill levels, and which cells want to convert.
  //
  // Two of the four conversion rules are about the fill level and two are about
  // the NEIGHBOURHOOD, and the second pair is not optional.
  //
  // A DETACHED INTERFACE CELL -- one with no liquid neighbour at all -- is a
  // speck of liquid with nothing touching it. Nothing can exert pressure on it,
  // so nothing balances gravity, and it accelerates at g for as long as the run
  // lasts. That is not a slow degradation: in a dam break the splash throws such
  // specks constantly, and each one is a free-fall trajectory that never ends.
  // Measured before this rule existed, max|u| grew LINEARLY at exactly g from
  // the moment the sheet first broke up, reaching Mach 1.2 by the end of an
  // otherwise healthy run, and the mass drift went with it.
  //
  // A BURIED INTERFACE CELL -- one with no gas neighbour -- looks like the
  // symmetric case and a rule for it was written, tested and REMOVED. Promoting
  // it to Fluid means declaring it full, and a cell that was thirty per cent
  // full then owes seventy per cent of a density to its neighbours through the
  // excess mechanism. Those neighbours are interface cells with little mass to
  // give; they go negative, convert to gas, expose more fluid, and the whole
  // surface unzips. Measured on the standing wave: the liquid was gone inside
  // six hundred steps, from a rule that fires on a handful of cells a step. The
  // detached rule alone leaves that case at 1.77%.
  //
  // The moral is that the two rules are not symmetric. Deleting a speck loses a
  // little mass and stops an unbounded velocity; filling a cell CREATES mass and
  // makes its neighbours pay, and there is no reason to think they can.
  //
  // The mass a detached cell takes with it is genuinely lost: by construction it
  // has no interface neighbour to hand it to. That is the documented
  // no-subgrid-interface limitation arriving in the accounts, and it is why the
  // demonstrators print mass drift every frame.
  //----------------------------------------------------------------------------
  void classify() {
    const Domain d = dom_;
    auto flags = flags_, newf = newf_;
    auto mass = mass_, eps = eps_, rho = rho_, excess = excess_;
    const Real off = fill_offset;
    const bool do_detach = drop_detached;

    Kokkos::parallel_for("fs_classify", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t fl = flags(n);
      excess(n) = Real(0);
      newf(n) = fl;
      if (fl == FsFluid)     { eps(n) = Real(1); return; }
      if (fl != FsInterface) { if (fl == FsGas) eps(n) = Real(0); return; }
      const Real r = rho(n);
      const Real e = (r > Real(0)) ? mass(n) / r : Real(0);
      eps(n) = e;

      if (e > Real(1) + off) {
        newf(n) = FsFluid;
        excess(n) = mass(n) - r;        // the part that will not fit
        return;
      }
      if (e < -off) {
        newf(n) = FsGas;
        excess(n) = mass(n);            // a debt, carried as a negative
        return;
      }
      // The neighbourhood rules. flags_ is stable through this pass -- nothing
      // here writes it -- so reading a neighbour's is safe.
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      bool has_gas = false, has_liquid = false;
      for (int i = 1; i < Q; ++i) {
        const std::uint8_t fj = flags(nb.j[i]);
        if (fj == FsGas) has_gas = true;
        else if (fj == FsFluid || fj == FsInterface) has_liquid = true;
        // A cell resting against a wall has something holding it, so it is not
        // the free-falling speck the rule is about. Without this an obstacle
        // sweeping past leaves a trail of cells that the detached rule deletes.
        else if (fj == FsSolid) has_liquid = true;
      }
      (void)has_gas;
      if (!has_liquid && do_detach) {   // a speck in free fall
        newf(n) = FsGas;
        excess(n) = mass(n);
      }
    });
    Kokkos::fence();
  }

  //----------------------------------------------------------------------------
  // Pass 4. Keep the interface closed, and settle the books.
  //
  // Three things at once, all as gathers:
  //   * a Gas cell beside a cell becoming Fluid must become Interface, or the
  //     new fluid cell would sit against a void with no boundary condition;
  //   * a Fluid cell beside a cell becoming Gas must become Interface, for the
  //     same reason from the other side;
  //   * the mass that would not fit in a converting cell is handed to the
  //     interface cells around it.
  //
  // A newly created Interface cell has no populations worth keeping, so it takes
  // the average density and velocity of its non-gas neighbours and is seeded at
  // equilibrium. That average is a gather, and it is why this can be one pass.
  //----------------------------------------------------------------------------
  void close_interface() {
    promote();
    settle();
  }

  //----------------------------------------------------------------------------
  // Pass 4a. Decide every cell's FINAL state, reading only the intents that
  // pass 3 published. It writes a THIRD array rather than editing the one it
  // reads, and that is not tidiness.
  //
  // The first version of this did both in one kernel: it computed a cell's new
  // flag from its neighbours' newf() and wrote its own newf() at the end. Every
  // cell was therefore reading an array other cells were concurrently writing,
  // so what a cell saw of its neighbour depended on thread scheduling. The
  // damage was not a subtly wrong interface -- it was mass creation, because the
  // redistribution below counts how many neighbours will take a share and that
  // count disagreed with how many actually did. Measured: total mass reached
  // 1e67 in three hundred steps.
  //----------------------------------------------------------------------------
  void promote() {
    const Domain d = dom_;
    auto flags = flags_, newf = newf_, fin = final_;
    Kokkos::parallel_for("fs_promote", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t was = flags(n);
      if (was == FsSolid || was == FsExcluded) { fin(n) = was; return; }
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      std::uint8_t now = newf(n);
      // A cell next to one that is about to be Fluid cannot be Gas: the new
      // fluid cell would face a void with no boundary condition to apply.
      if (now == FsGas) {
        for (int i = 1; i < Q; ++i)
          if (newf(nb.j[i]) == FsFluid) { now = FsInterface; break; }
      }
      // And symmetrically from the other side.
      if (now == FsFluid) {
        for (int i = 1; i < Q; ++i)
          if (newf(nb.j[i]) == FsGas) { now = FsInterface; break; }
      }
      fin(n) = now;
    });
    Kokkos::fence();
  }

  //----------------------------------------------------------------------------
  // Pass 4b. Settle the books against the final states.
  //
  // The excess a converting cell cannot hold is shared equally among the
  // neighbours that end up Interface. Each taker recomputes that neighbour's
  // taker COUNT itself rather than having the donor publish it -- which keeps
  // this a pure gather -- and the predicate it counts with must be exactly the
  // predicate under which it takes, or the shares do not sum to the excess.
  // Both now read final_, which nothing in this pass writes.
  //----------------------------------------------------------------------------
  void settle() {
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_, fin = final_;
    auto mass = mass_, eps = eps_, rho = rho_, excess = excess_;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto reinit = reinit_;

    Kokkos::parallel_for("fs_settle", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t was = flags(n);
      reinit(n) = 0;
      if (was == FsSolid || was == FsExcluded) return;
      const std::uint8_t now = fin(n);
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);

      if (now == FsInterface) {
        Real got = Real(0);
        for (int i = 1; i < Q; ++i) {
          const Index j = nb.j[i];
          const Real ex = excess(j);
          if (ex == Real(0)) continue;
          Neighbours<L> nj;
          d.template fill_neighbours<L, NF, NS>(j, nj);
          int takers = 0;
          for (int k = 1; k < Q; ++k)
            if (fin(nj.j[k]) == FsInterface) ++takers;
          if (takers > 0) got += ex / Real(takers);
        }

        if (was == FsGas) {
          // Arriving from nothing: take the mean state of the neighbours that
          // have one, and start at equilibrium.
          Real sr = Real(0), su[3] = {Real(0), Real(0), Real(0)};
          int cnt = 0;
          for (int i = 1; i < Q; ++i) {
            const Index j = nb.j[i];
            const std::uint8_t fj = flags(j);
            if (fj != FsFluid && fj != FsInterface) continue;
            sr += rho(j); su[0] += ux(j); su[1] += uy(j); su[2] += uz(j);
            ++cnt;
          }
          const Real inv = cnt ? Real(1) / Real(cnt) : Real(0);
          const Real r = cnt ? sr * inv : Real(1);
          const Real u[3] = {su[0] * inv, su[1] * inv, su[2] * inv};
          for (int i = 0; i < Q; ++i)
            acc.dst(n, i) = Collision::seed_value(i, r, u[0], u[1], u[2]);
          rho(n) = r;  ux(n) = u[0]; uy(n) = u[1]; uz(n) = u[2];
          mass(n) = Real(0);
          reinit(n) = 1;      // its state came from nowhere; see reinitialised()
        } else if (was == FsFluid) {
          mass(n) = rho(n);           // still full; it just has a surface now
        }
        mass(n) += got;
        const Real r = rho(n);
        eps(n) = (r > Real(0)) ? mass(n) / r : Real(0);
      } else if (now == FsFluid) {
        mass(n) = rho(n);  eps(n) = Real(1);
      } else {  // FsGas
        mass(n) = Real(0); eps(n) = Real(0);
      }
    });
    Kokkos::fence();
    Kokkos::deep_copy(flags_, final_);
    Kokkos::deep_copy(newf_, final_);
  }

 private:
  Domain dom_;
  Streaming pop_;
  View1D<std::uint8_t> flags_, newf_, final_, reinit_;
  HostView1D<std::uint8_t> h_flags_;
  View1D<Real> mass_, eps_, excess_;
  View1D<Real> wux_, wuy_, wuz_;      // wall velocity, meaningful in body cells
  // WHICH SOLID CELLS ARE THE BODY'S. A tank wall and a moving obstacle are both
  // FsSolid and must not be confused: the wall does not move, does not get
  // refilled when the body passes, and above all does not contribute to the
  // body's force. Conflating them was measured -- the tank floor and lid took
  // the body's velocity, pumped the fluid, converted themselves to interface
  // cells on the first call, and added their own links to the momentum exchange.
  View1D<std::uint8_t> obst_, was_body_;
  Real bcx_ = 0, bcy_ = 0;            // body centre, for the torque
  Real fx_ = 0, fy_ = 0, tz_ = 0;     // momentum-exchange force and torque
  View1D<Real> rho_, ux_, uy_, uz_;
  std::size_t t_ = 0;
};

}  // namespace lbm
