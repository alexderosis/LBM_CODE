#pragma once
//==============================================================================
//  Free-surface flow: a sharp interface, and only the liquid.
//
//  A port of ../src/solver/FreeSurfaceSolver.hpp, following the standard
//  formulation of Korner, Thies, Hofmann, Theobald & Rude, J. Stat. Phys. 121,
//  179 (2005). There is no paper in SomeRefs/ to check equation by equation, so
//  nothing here cites an equation number and every choice is justified from the
//  mechanics.
//
//  THE THIRD ROUTE TO A MULTIPHASE FLOW HERE, and the one that changes the
//  bargain rather than the parameters. phasefield.cuh and colour.cuh both
//  resolve BOTH fluids, and everything difficult about them follows from that.
//  This one does not resolve the gas at all: the gas is a VOID at a prescribed
//  pressure -- no populations, no velocity, no dynamics -- and the liquid meets
//  it across a boundary condition rather than a diffuse interface.
//
//  There is no density ratio here because there is no second density. The ratio
//  is infinite by construction and it costs nothing, because the expensive half
//  of the problem was deleted rather than discretised.
//
//      module            interface   fluids resolved   density ratio
//      phase field       diffuse     both              ~100, with the gauge problem
//      colour gradient   diffuse     both              1000 in principle
//      free surface      sharp       liquid only       infinite, by construction
//
//==============================================================================
//  THE STATE. Every cell is Gas, Interface, Fluid or Solid. Interface cells are
//  the entire method: they carry an explicit MASS alongside their populations,
//  and a fill level epsilon = m / rho, 1 when full and 0 when empty. The
//  interface is a closed, one-cell-thick shell between Fluid and Gas, and
//  keeping it closed through arbitrary topology change is most of the work.
//
//  MASS IS ADVECTED, NOT A MOMENT, and that is the sharpest difference from
//  everything else in this tree. Elsewhere density is sum_i f_i and conservation
//  is a property of the collision, checkable by algebra. Here mass moves because
//  populations cross faces:
//
//      dm_i(x) = f_ibar(x + c_i) - f_i(x),
//
//  scaled by the mean fill level when both cells are Interface, because a
//  half-full cell can only exchange half as much. NOTHING ENFORCES CONSERVATION
//  BUT THE ANTISYMMETRY OF THAT EXPRESSION -- what x sends is exactly what the
//  neighbour receives. That is why the test checks total mass to round-off
//  first and everything else second: it is the one property that can be
//  silently lost.
//
//  THE FREE-SURFACE CONDITION reconstructs the populations that would have
//  arrived from the gas. For every direction whose source cell is Gas,
//
//      f_i(x) = f_i^eq(rho_G, u) + f_ibar^eq(rho_G, u) - f_ibar(x),
//
//  with rho_G fixing the gas pressure through p = rho_G cs^2. Its two moments
//  ARE the free surface: normal stress equal to the gas pressure, tangential
//  stress zero. f_ibar(x) is the previous step's post-collision population at x,
//  which the two-lattice scheme still holds in its source array -- no extra
//  storage is needed to find it.
//
//  THE VELOCITY IN THAT RECONSTRUCTION IS ONE STEP OLD, because u comes from a
//  sum over populations, some of which are the ones being reconstructed. The
//  circle is broken with the previous step's velocity: first order in time, and
//  what the literature does. It matters where the interface accelerates hard and
//  is invisible in a standing wave.
//
//==============================================================================
//  WHY TWO-LATTICE STREAMING, when everything else in this tree uses Esoteric
//  Pull -- and note that this is the ONE module here that does not. Three
//  requirements, and Esoteric Pull fails two outright.
//
//   1. Mass exchange needs f_i(x) and f_ibar(x + c_i) as POST-COLLISION values,
//      after the whole domain has collided. In place, those slots hold a mixture
//      of two time levels, and which one depends on parity and on the
//      direction's index. Two-lattice makes them simply the destination array,
//      complete and unambiguous, once the collide pass has fenced.
//   2. A cell that fills up turns its gas neighbours into interface cells, which
//      then need populations. In place, a cell's populations live in its
//      NEIGHBOURS' slots, so initialising a neighbour means writing storage
//      other cells are concurrently reading through their own aliases.
//   3. Reading a neighbour's post-collision state while writing one's own is the
//      normal case here and impossible there.
//
//  Requirement 2 is dodged rather than met: EVERY PASS BELOW IS A GATHER. A cell
//  about to become interface initialises ITSELF by averaging its neighbours, and
//  a cell with excess mass publishes it in a field its neighbours collect.
//  Nothing ever writes another cell's memory, so there is no race to reason
//  about and no atomics -- a deliberate shape, worth keeping even if scatters
//  were free, and on a device more so.
//
//  The cost is two population arrays instead of one: 216 bytes per node at FP32
//  against Esoteric Pull's 108. For a method that deletes an entire phase, that
//  is not the expensive part.
//
//==============================================================================
//  WHAT THIS DOES NOT MODEL.
//
//   * NO SURFACE TENSION. rho_G is uniform, so the gas pressure carries no
//     curvature term. Everything at a Bond number where surface tension competes
//     with gravity is out of scope -- most droplet problems, and none of the
//     dam-break family.
//   * NO GAS DYNAMICS, and this is the method rather than an omission. An
//     enclosed bubble does not compress, because its pressure is prescribed
//     rather than solved.
//   * THE RECONSTRUCTION IS APPLIED TO GAS-FACING DIRECTIONS ONLY. The
//     literature also reconstructs directions with c_i . n > 0 for the interface
//     normal, which stabilises interfaces nearly tangential to the lattice. If a
//     case shows interface roughening along a diagonal, this is the first thing
//     to add.
//   * EXCESS MASS IS REDISTRIBUTED UNIFORMLY among eligible neighbours rather
//     than weighted by the interface normal. Uniform conserves mass exactly --
//     what matters most -- and is more diffusive at a sharply curved interface.
//   * NO SUBGRID INTERFACE: a film thinner than a cell either survives as a
//     one-cell layer or vanishes.
//   * NO MOVING OBSTACLE, AND THAT IS DELIBERATE. The parent has one and its own
//     banner says it is not reliable: two defects are identified, measured, and
//     LEFT IN, because fixing either makes the demonstrator fail sooner --
//     t* = 6.16 down to 3.39 for one, 2.54 with the other -- so its reach
//     depends on those errors cancelling something that has not been found.
//     Porting that would move a known-bad ledger into a second codebase and give
//     it a second place to be trusted by accident. Static walls (FsSolid) are
//     here and are ordinary halfway bounce-back; a body that MOVES is not.
//==============================================================================
#include "streaming.cuh"

namespace lbm {

using FsLattice = D3Q27;

//------------------------------------------------------------------------------
// Cell states. A distinct enum from CellType, ScalarCell, PhaseCell and MagCell,
// with values that deliberately do NOT line up with any of them: all are stored
// as uint8 and mixing them would compile silently.
//------------------------------------------------------------------------------
enum FsCell : std::uint8_t {
  FsGas       = 10,   // no liquid; not simulated, holds nothing
  FsInterface = 11,   // partially filled; carries mass and the free surface
  FsFluid     = 12,   // full; an ordinary LBM cell
  FsSolid     = 13,   // wall: halfway bounce-back
};

//------------------------------------------------------------------------------
// The product-form equilibrium, as populations.
//
// Reached by inverse-transforming k_eq = (rho, 0, ..., 0) rather than by
// evaluating a Hermite series -- cheaper, and exactly consistent with what the
// central-moment collision relaxes toward, so the free-surface boundary
// condition and the operator agree about what equilibrium means. Getting those
// two out of step puts a permanent source at every gas-facing link.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void cm_equilibrium(Real rho, const Real u[3], Real f[27]) {
  Real k[27];
  for (int n = 0; n < 27; ++n) k[n] = Real(0);
  k[mi(0, 0, 0)] = rho;
  to_populations(k, u, f);
}

//------------------------------------------------------------------------------
// Kernel parameters. One struct for all five passes.
//------------------------------------------------------------------------------
struct FsParams {
  Real* src = nullptr;              // previous step's post-collision populations
  Real* dst = nullptr;              // this step's
  std::uint8_t* flags = nullptr;
  std::uint8_t* newf = nullptr;     // intent, published by classify
  std::uint8_t* fin = nullptr;      // final state, decided by promote
  std::uint8_t* reinit = nullptr;   // rebuilt-from-nothing marker, for diagnostics
  Real* mass = nullptr;
  Real* eps = nullptr;
  Real* excess = nullptr;
  Real* rho = nullptr;
  Real* ux = nullptr;  Real* uy = nullptr;  Real* uz = nullptr;
  int nx = 0, ny = 0, nz = 0;
  Real omega = Real(1), omega_bulk = Real(1);
  Real rho_G = Real(1);
  Real gx = Real(0), gy = Real(0), gz = Real(0);
  // A cell converts when its fill leaves [-off, 1 + off]. The band stops a cell
  // hovering at exactly full from converting back and forth every step, which
  // costs a re-initialisation each time and roughens the interface.
  Real fill_offset = Real(1e-3);
  bool drop_detached = true;
};

LBM_HD LBM_INLINE Real fs_at(const Real* f, long N, long n, int i) {
  return f[long(i) * N + n];
}

// The seed a caller supplies per node. `fill` in [0,1] describes the liquid: 1
// seeds Fluid, 0 seeds Gas, anything between seeds an Interface cell, and the
// one-cell interface shell follows from it. `rho` seeds the pressure through
// p = rho cs^2.
//
// THE DENSITY IS WORTH SETTING. Left at 1 the liquid starts with no pressure
// gradient at all, so a column under gravity has to build its own hydrostatic
// profile -- and it does that by RINGING: an acoustic transient that crosses the
// domain in a few hundred steps and takes far longer to damp. Harmless in a dam
// break, ruinous in a standing-wave measurement where it is noise at the
// amplitude being measured. rho = rho_G + g (y_surface - y) / cs^2 starts it
// quiet.
struct FsSeed { Real fill = 0, rho = Real(1); };

template <class Init>
LBM_HD LBM_INLINE void fs_init_node(const FsParams& p, long N, long n, Init init) {
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real e = Real(0), r = Real(1);
  if (p.flags[n] != FsSolid) {
    const FsSeed sd = init(x, y, z);
    e = sd.fill;  r = sd.rho;
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    p.flags[n] = (e >= Real(1)) ? FsFluid : (e <= Real(0) ? FsGas : FsInterface);
  }
  p.newf[n] = p.flags[n];
  p.fin[n] = p.flags[n];
  p.rho[n] = r;
  p.ux[n] = Real(0);  p.uy[n] = Real(0);  p.uz[n] = Real(0);
  p.eps[n] = e;
  p.mass[n] = e * r;
  // EVERY cell is given populations, walls and gas included. A wall holds
  // in-transit populations exactly as elsewhere in this code, and a gas cell
  // that later becomes interface is initialised properly then -- but leaving
  // either as raw zeros makes the first step read garbage.
  const Real u[3] = {Real(0), Real(0), Real(0)};
  Real g[27];
  cm_equilibrium(r, u, g);
  for (int i = 0; i < 27; ++i) {
    p.dst[long(i) * N + n] = g[i];
    p.src[long(i) * N + n] = g[i];
  }
}

//------------------------------------------------------------------------------
// PASS 1. Stream, apply the free surface and the walls, collide.
//
// The gather is a PULL: f_i(x) comes from x - c_i, which is the neighbour in
// direction opp(i).
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void fs_stream_collide_node(const FsParams& p, long N, long n) {
  const std::uint8_t fl = p.flags[n];
  if (fl != FsFluid && fl != FsInterface) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  // The velocity in the reconstruction is the PREVIOUS step's; see the banner.
  const Real u0[3] = {p.ux[n], p.uy[n], p.uz[n]};
  Real geq[27];
  cm_equilibrium(p.rho_G, u0, geq);

  Real f[27];
  for (int i = 0; i < 27; ++i) {
    const long from = (i == 0) ? n
        : neighbour<FsLattice>(x, y, z, opp(i), p.nx, p.ny, p.nz);
    const std::uint8_t ff = p.flags[from];
    if (ff == FsGas) {
      // Free-surface condition: impose the gas pressure through the two moments
      // of an anti-bounce-back pair, against this cell's own outgoing
      // population in the opposite direction.
      f[i] = geq[i] + geq[opp(i)] - fs_at(p.src, N, n, opp(i));
    } else if (ff == FsSolid) {
      // Halfway bounce-back off a static wall: what went in comes back.
      f[i] = fs_at(p.src, N, n, opp(i));
    } else {
      f[i] = fs_at(p.src, N, from, i);
    }
  }

  // Everything above this line is the free surface; everything below it is the
  // same central-moment operator the single-phase solver runs, with gravity
  // through Guo. The force is a CONSTANT vector at rho_0 = 1, which is exact
  // for this method rather than an approximation: the liquid is nearly
  // incompressible, the gas has no weight because it has no populations, and an
  // interface cell's rho is close to one whatever its fill -- the fill is
  // carried by the MASS, not by the density.
  Macro m = macroscopic(f);
  Coupling cp;
  cp.F[0] = p.gx;  cp.F[1] = p.gy;  cp.F[2] = p.gz;
  shift_velocity(m, cp.F);
  collide_cm_gen<true, false>(f, m, p.omega, p.omega_bulk, cp);

  for (int i = 0; i < 27; ++i) p.dst[long(i) * N + n] = f[i];
  p.rho[n] = m.rho;
  p.ux[n] = m.ux;  p.uy[n] = m.uy;  p.uz[n] = m.uz;
  if (fl == FsFluid) p.mass[n] = m.rho;     // a full cell's mass IS its density
}

//------------------------------------------------------------------------------
// PASS 2. Mass across faces, from the post-collision state of BOTH cells.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void fs_mass_exchange_node(const FsParams& p, long N, long n) {
  if (p.flags[n] != FsInterface) return;
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  const Real e0 = p.eps[n];
  Real dm = Real(0);
  for (int i = 1; i < 27; ++i) {
    const long j = neighbour<FsLattice>(x, y, z, i, p.nx, p.ny, p.nz);
    const std::uint8_t fj = p.flags[j];
    const Real in  = fs_at(p.dst, N, j, opp(i));     // what j sends me
    const Real out = fs_at(p.dst, N, n, i);          // what I send j
    if (fj == FsFluid) {
      dm += in - out;
    } else if (fj == FsInterface) {
      // Two partly filled cells exchange in proportion to how much liquid there
      // is to exchange. The MEAN of the two fill levels is symmetric between
      // them, which is what keeps the pair's total mass unchanged -- and that
      // antisymmetry is the only thing conserving mass here.
      dm += Real(0.5) * (e0 + p.eps[j]) * (in - out);
    }
    // Gas and wall neighbours exchange nothing.
  }
  p.mass[n] += dm;
}

//------------------------------------------------------------------------------
// PASS 3. Fill levels, and which cells want to convert.
//
// Two of the rules are about the fill level and one is about the NEIGHBOURHOOD,
// and the third is not optional.
//
// A DETACHED INTERFACE CELL -- one with no liquid neighbour at all -- is a speck
// of liquid with nothing touching it. Nothing can exert pressure on it, so
// nothing balances gravity, and it accelerates at g for as long as the run
// lasts. In a dam break the splash throws such specks constantly and each one is
// a free-fall trajectory that never ends. Measured in the parent before the rule
// existed: max|u| grew LINEARLY at exactly g from the moment the sheet broke up,
// reaching Mach 1.2 by the end of an otherwise healthy run.
//
// A BURIED INTERFACE CELL -- one with no gas neighbour -- looks like the
// symmetric case, and a rule for it was written, tested and REMOVED in the
// parent. Promoting it to Fluid declares it full, so a cell that was thirty per
// cent full then owes seventy per cent of a density to neighbours who are
// interface cells with little to give; they go negative, convert to gas, expose
// more fluid, and the surface unzips. The two rules are NOT symmetric: deleting
// a speck loses a little mass and stops an unbounded velocity, while filling a
// cell CREATES mass and makes its neighbours pay, with no reason to think they
// can. So only the detached rule is here.
//
// The mass a detached cell takes with it is genuinely lost -- by construction it
// has no interface neighbour to hand it to. That is the no-subgrid-interface
// limitation arriving in the accounts.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void fs_classify_node(const FsParams& p, long n) {
  const std::uint8_t fl = p.flags[n];
  p.excess[n] = Real(0);
  p.newf[n] = fl;
  if (fl == FsFluid) { p.eps[n] = Real(1); return; }
  if (fl != FsInterface) { if (fl == FsGas) p.eps[n] = Real(0); return; }

  const Real r = p.rho[n];
  const Real e = (r > Real(0)) ? p.mass[n] / r : Real(0);
  p.eps[n] = e;

  if (e > Real(1) + p.fill_offset) {
    p.newf[n] = FsFluid;
    p.excess[n] = p.mass[n] - r;             // the part that will not fit
    return;
  }
  if (e < -p.fill_offset) {
    p.newf[n] = FsGas;
    p.excess[n] = p.mass[n];                 // a debt, carried as a negative
    return;
  }

  // flags is stable through this pass -- nothing here writes it -- so reading a
  // neighbour's is safe.
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  bool has_liquid = false;
  for (int i = 1; i < 27; ++i) {
    const std::uint8_t fj = p.flags[neighbour<FsLattice>(x, y, z, i, p.nx, p.ny, p.nz)];
    if (fj == FsFluid || fj == FsInterface) has_liquid = true;
    // A cell resting against a wall has something holding it, so it is not the
    // free-falling speck the rule is about.
    else if (fj == FsSolid) has_liquid = true;
  }
  if (!has_liquid && p.drop_detached) {
    p.newf[n] = FsGas;
    p.excess[n] = p.mass[n];
  }
}

//------------------------------------------------------------------------------
// PASS 4a. Decide every cell's FINAL state, reading only the intents pass 3
// published. It writes a THIRD array rather than editing the one it reads, and
// that is not tidiness.
//
// The parent's first version did both in one kernel: a cell computed its new
// flag from its neighbours' newf and wrote its own newf at the end, so every
// cell was reading an array other cells were concurrently writing and what it
// saw depended on scheduling. The damage was not a subtly wrong interface but
// MASS CREATION, because the redistribution below counts how many neighbours
// will take a share and that count disagreed with how many actually did.
// Measured: total mass reached 1e67 in three hundred steps.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void fs_promote_node(const FsParams& p, long n) {
  const std::uint8_t was = p.flags[n];
  if (was == FsSolid) { p.fin[n] = was; return; }
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  std::uint8_t now = p.newf[n];
  // A cell next to one about to be Fluid cannot be Gas: the new fluid cell would
  // face a void with no boundary condition to apply.
  if (now == FsGas) {
    for (int i = 1; i < 27; ++i)
      if (p.newf[neighbour<FsLattice>(x, y, z, i, p.nx, p.ny, p.nz)] == FsFluid) {
        now = FsInterface;  break;
      }
  }
  // And symmetrically from the other side.
  if (now == FsFluid) {
    for (int i = 1; i < 27; ++i)
      if (p.newf[neighbour<FsLattice>(x, y, z, i, p.nx, p.ny, p.nz)] == FsGas) {
        now = FsInterface;  break;
      }
  }
  p.fin[n] = now;
}

//------------------------------------------------------------------------------
// PASS 4b. Settle the books against the final states.
//
// The excess a converting cell cannot hold is shared equally among the
// neighbours that end up Interface. Each taker recomputes that neighbour's taker
// COUNT itself rather than having the donor publish it -- which keeps this a
// pure gather -- and the predicate it counts with must be EXACTLY the predicate
// under which it takes, or the shares do not sum to the excess. Both read fin,
// which nothing in this pass writes.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void fs_settle_node(const FsParams& p, long N, long n) {
  const std::uint8_t was = p.flags[n];
  p.reinit[n] = 0;
  if (was == FsSolid) return;
  const std::uint8_t now = p.fin[n];
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  if (now == FsInterface) {
    Real got = Real(0);
    for (int i = 1; i < 27; ++i) {
      const long j = neighbour<FsLattice>(x, y, z, i, p.nx, p.ny, p.nz);
      const Real ex = p.excess[j];
      if (ex == Real(0)) continue;
      int jx, jy, jz;
      coords(j, p.nx, p.ny, jx, jy, jz);
      int takers = 0;
      for (int k = 1; k < 27; ++k)
        if (p.fin[neighbour<FsLattice>(jx, jy, jz, k, p.nx, p.ny, p.nz)] == FsInterface)
          ++takers;
      if (takers > 0) got += ex / Real(takers);
    }

    if (was == FsGas) {
      // Arriving from nothing: take the mean state of the neighbours that have
      // one, and start at equilibrium.
      Real sr = Real(0), su[3] = {Real(0), Real(0), Real(0)};
      int cnt = 0;
      for (int i = 1; i < 27; ++i) {
        const long j = neighbour<FsLattice>(x, y, z, i, p.nx, p.ny, p.nz);
        const std::uint8_t fj = p.flags[j];
        if (fj != FsFluid && fj != FsInterface) continue;
        sr += p.rho[j];  su[0] += p.ux[j];  su[1] += p.uy[j];  su[2] += p.uz[j];
        ++cnt;
      }
      const Real inv = cnt ? Real(1) / Real(cnt) : Real(0);
      const Real r = cnt ? sr * inv : Real(1);
      const Real u[3] = {su[0] * inv, su[1] * inv, su[2] * inv};
      Real g[27];
      cm_equilibrium(r, u, g);
      for (int i = 0; i < 27; ++i) p.dst[long(i) * N + n] = g[i];
      p.rho[n] = r;  p.ux[n] = u[0];  p.uy[n] = u[1];  p.uz[n] = u[2];
      p.mass[n] = Real(0);
      p.reinit[n] = 1;                 // its state came from nowhere
    } else if (was == FsFluid) {
      p.mass[n] = p.rho[n];            // still full; it just has a surface now
    }
    p.mass[n] += got;
    const Real r = p.rho[n];
    p.eps[n] = (r > Real(0)) ? p.mass[n] / r : Real(0);
  } else if (now == FsFluid) {
    p.mass[n] = p.rho[n];  p.eps[n] = Real(1);
  } else {                             // FsGas
    p.mass[n] = Real(0);  p.eps[n] = Real(0);
  }
}

#if defined(__CUDACC__)

__global__ void fs_stream_collide_kernel(FsParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  fs_stream_collide_node(p, N, n);
}
__global__ void fs_mass_kernel(FsParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  fs_mass_exchange_node(p, N, n);
}
__global__ void fs_classify_kernel(FsParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  fs_classify_node(p, n);
}
__global__ void fs_promote_kernel(FsParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  fs_promote_node(p, n);
}
__global__ void fs_settle_kernel(FsParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  fs_settle_node(p, N, n);
}

// Seed. init(x, y, z) -> FsSeed. `fill` in [0,1] describes the liquid: 1 seeds
// Fluid, 0 seeds Gas, anything between seeds an Interface cell.
//
// THE DENSITY IS WORTH SETTING. Left at 1 the liquid starts with no pressure
// gradient at all, so a column under gravity has to build its own hydrostatic
// profile, and it does that by RINGING -- an acoustic transient that crosses the
// domain in a few hundred steps and takes far longer to damp. Harmless in a dam
// break, ruinous in a standing-wave measurement where it is noise at the
// amplitude being measured. rho = rho_G + g (y_surface - y) / cs^2 starts it
// quiet.
template <class Init>
__global__ void fs_initialise(FsParams p, long N, Init init) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  fs_init_node(p, N, n, init);
}

//==============================================================================
//  Host-side driver. Owns the two lattices and the five-pass order.
//==============================================================================
class FreeSurfaceSolver {
 public:
  FreeSurfaceSolver(int nx, int ny, int nz, Real nu)
      : nx_(nx), ny_(ny), nz_(nz), omega_(omega_from_viscosity(nu)) {
    N_ = long(nx) * ny * nz;
    const std::size_t pop = sizeof(Real) * 27 * std::size_t(N_);
    LBM_CUDA_CHECK(cudaMalloc(&fa_, pop));
    LBM_CUDA_CHECK(cudaMalloc(&fb_, pop));
    const std::size_t fld = sizeof(Real) * std::size_t(N_);
    for (int k = 0; k < NFIELD; ++k) LBM_CUDA_CHECK(cudaMalloc(&field_[k], fld));
    for (int k = 0; k < NFLAG; ++k)
      LBM_CUDA_CHECK(cudaMalloc(&flag_[k], sizeof(std::uint8_t) * std::size_t(N_)));
    LBM_CUDA_CHECK(cudaMemset(flag_[0], FsGas, sizeof(std::uint8_t) * std::size_t(N_)));
  }
  ~FreeSurfaceSolver() {
    cudaFree(fa_); cudaFree(fb_);
    for (int k = 0; k < NFIELD; ++k) cudaFree(field_[k]);
    for (int k = 0; k < NFLAG; ++k) cudaFree(flag_[k]);
  }
  FreeSurfaceSolver(const FreeSurfaceSolver&) = delete;
  FreeSurfaceSolver& operator=(const FreeSurfaceSolver&) = delete;

  Real rho_G = Real(1);
  Real omega_bulk = Real(1);
  Real fill_offset = Real(1e-3);
  bool drop_detached = true;

  void set_gravity(Real ax, Real ay, Real az = Real(0)) { gx_ = ax; gy_ = ay; gz_ = az; }

  // Cell roles, one per node. Call BEFORE initialise_with, so a wall is not
  // overwritten by the seed.
  void set_geometry(const std::vector<std::uint8_t>& flags) {
    LBM_CUDA_CHECK(cudaMemcpy(flag_[0], flags.data(),
                              sizeof(std::uint8_t) * std::size_t(N_),
                              cudaMemcpyHostToDevice));
  }

  template <class Init>
  void initialise_with(Init init) {
    const int B = 128, G = int((N_ + B - 1) / B);
    fs_initialise<<<G, B>>>(params(), N_, init);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    t_ = 0;
    // The seeded fill levels need not describe a closed interface -- a caller may
    // hand over a sharp step. One repair pass turns it into one.
    close_interface();
  }

  //--------------------------------------------------------------------------
  // One step. Five fenced passes; see the banner for why they cannot be fewer.
  // Kernel boundaries on one stream ARE the fences.
  //--------------------------------------------------------------------------
  void step() {
    const int B = 128, G = int((N_ + B - 1) / B);
    const FsParams p = params();
    fs_stream_collide_kernel<<<G, B>>>(p, N_);
    fs_mass_kernel<<<G, B>>>(p, N_);
    fs_classify_kernel<<<G, B>>>(p, N_);
    close_interface();
    LBM_CUDA_CHECK(cudaGetLastError());
    std::swap(fa_, fb_);                 // end of step: dst becomes src
    ++t_;
  }

  void field_to_host(const Real* src, std::vector<Real>& out) {
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    out.resize(std::size_t(N_));
    LBM_CUDA_CHECK(cudaMemcpy(out.data(), src, sizeof(Real) * std::size_t(N_),
                              cudaMemcpyDeviceToHost));
  }
  void flags_to_host(std::vector<std::uint8_t>& out) {
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    out.resize(std::size_t(N_));
    LBM_CUDA_CHECK(cudaMemcpy(out.data(), flag_[0],
                              sizeof(std::uint8_t) * std::size_t(N_),
                              cudaMemcpyDeviceToHost));
  }

  // TOTAL LIQUID MASS, the one property that can be silently lost -- see the
  // banner. Summed in double even in an FP32 build, for the same reason
  // reduce_population is.
  double total_mass() {
    std::vector<Real> m;
    field_to_host(field_[0], m);
    double s = 0;
    for (Real v : m) s += double(v);
    return s;
  }

  const Real* mass_device() const { return field_[0]; }
  const Real* eps_device()  const { return field_[1]; }
  const Real* rho_device()  const { return field_[3]; }
  const Real* ux_device()   const { return field_[4]; }
  const Real* uy_device()   const { return field_[5]; }
  const Real* uz_device()   const { return field_[6]; }
  std::size_t timestep() const { return t_; }
  long nodes() const { return N_; }

 private:
  static constexpr int NFIELD = 7;    // mass eps excess rho ux uy uz
  static constexpr int NFLAG  = 4;    // flags newf final reinit

  void close_interface() {
    const int B = 128, G = int((N_ + B - 1) / B);
    const FsParams p = params();
    fs_promote_kernel<<<G, B>>>(p, N_);
    fs_settle_kernel<<<G, B>>>(p, N_);
    LBM_CUDA_CHECK(cudaGetLastError());
    // final -> flags, and final -> newf, so the next classify starts consistent.
    LBM_CUDA_CHECK(cudaMemcpy(flag_[0], flag_[2],
                              sizeof(std::uint8_t) * std::size_t(N_),
                              cudaMemcpyDeviceToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(flag_[1], flag_[2],
                              sizeof(std::uint8_t) * std::size_t(N_),
                              cudaMemcpyDeviceToDevice));
  }

  FsParams params() const {
    FsParams p;
    p.src = fa_;  p.dst = fb_;
    p.flags = flag_[0];  p.newf = flag_[1];  p.fin = flag_[2];  p.reinit = flag_[3];
    p.mass = field_[0];  p.eps = field_[1];  p.excess = field_[2];
    p.rho = field_[3];
    p.ux = field_[4];  p.uy = field_[5];  p.uz = field_[6];
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
  Real* fa_ = nullptr;
  Real* fb_ = nullptr;
  Real* field_[NFIELD] = {};
  std::uint8_t* flag_[NFLAG] = {};
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
