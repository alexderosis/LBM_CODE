# M3LB — working notes for Claude

Lattice Boltzmann solver, C++20 on Kokkos. Read this before proposing changes or
writing a case; most of it is knowledge that is otherwise spread across
`doc/m3lb.pdf` (107 pages) and the banner comments at the top of each header.

**The banners are the documentation.** Every non-obvious decision in this tree is
argued at the top of the file that implements it, usually with the measurement
that settled it. When you are about to change something, read that file's banner
first — it frequently already says why the obvious change is wrong.

---

## Two independent codebases

| | what it is | when to use |
|---|---|---|
| `src/` + `validation/` | the main solver, Kokkos, five lattices, four collision operators, thermal + MHD + multiphase + free surface | default; anything on CPU; anything needing the full physics |
| `GPU/` | a second implementation written directly in CUDA, sharing **no headers** with the first | GPU runs, or cross-checking one implementation against the other |

They deliberately duplicate the physics. That is the point: they agree where they
overlap, and disagreement is a bug in one of them. Do not "de-duplicate" them.

`GPU/` compiles as plain C++ too (`-DLBM_HOST_ONLY=ON`), so every CUDA driver can
be built and run on a laptop before it touches a device. Do that first — it is
how wrong initial conditions and wrong diagnostics get found cheaply.

---

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_THREADS=ON
cmake --build build -j4
cd build && ctest --output-on-failure
```

**Pass `-DKokkos_ENABLE_THREADS=ON`.** Without it CMake prints *"no host-parallel
backend, tests run serial"* and everything runs single-threaded — several times
slower, and easy not to notice. Tests then get `--kokkos-num-threads=4`
automatically (`LBM_TEST_THREADS` in `CMakeLists.txt`); pass it by hand when
running an executable directly:

```bash
./build/validation/poiseuille --kokkos-num-threads=4
```

FP32 build: `-DLBM_PRECISION=float`. It cannot resolve the finest convergence
tests — that is documented, not a bug.

GPU (`GPU/` is its own CMake project, nvcc only, no Kokkos):

```bash
cd GPU && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=75
cmake --build build -j4          # 75 = T4/Turing, 80 = A100, 90 = Hopper
```

---

## Choosing a solver

| physics | use | notes |
|---|---|---|
| single-phase, single-component | `FluidSolver` | **the default — see the rule below** |
| + temperature / passive scalar | `ScalarSolver` alongside | own lattice, velocity is an input |
| + magnetic field | `MagneticSolver` | Dellar vector distribution |
| two-phase, diffuse interface | `PhaseFieldSolver` | conservative Allen–Cahn, prescribed interface width, density ratio ~100 |
| two-phase, diffuse, high ratio | `ColourGradientSolver` | no interface equation; width is an *outcome*; 1000 in the source paper's own static tests |
| liquid + void, sharp interface | `FreeSurfaceSolver` | gas not resolved; infinite ratio by construction |

**Standing rule: do not use a multiphase solver for single-phase or
single-component flow.** The multiphase path changes the equation of state — the
populations carry a normalised pressure rather than a density — and that
propagates through the macroscopic definitions, forcing, boundaries and the
initial condition. Use `FluidSolver` unless the problem genuinely has two phases
or two components.

The two immiscible models do not dominate each other. On a matched static droplet
the colour gradient is more accurate on Laplace's law at every density ratio
measured; the phase field carries a spurious current 3× to 118× smaller. That is
one static case, not a flow.

---

## From a physical problem to lattice units

Most cases here are dimensionless and need no conversion at all: they are
specified by Re, Ra, Pr or Ha and a resolution, and the answer is a
dimensionless number checked against a table. Conversion only bites when the
geometry carries a real length — `urban`, `height_field`, `aorta`.

**What the code converts, and what it does not.** Every operator turns a
transport coefficient into a relaxation rate, and most turn it back:

| operator | forward | inverse |
|---|---|---|
| `BGK`, `TRT`, `MomentCollision`, `MultiphaseBGK`, `MhdBGK`, `MhdCentralMoments` | `omega_from_viscosity(nu)` | `viscosity_from_omega(w)` |
| `ScalarBGK` | `omega_from_diffusivity(d)` | `diffusivity_from_omega(w)` |
| `MagneticBGK` | `omega_from_resistivity(eta)` | — |
| `PhaseFieldBGK` | `omega_from_mobility(m)` | — |
| `ColourGradient` | — | `viscosity_from_tau(tau)` |

They take the coefficient **already in lattice units**, and each reads its own
lattice's `cs2`, so they are the safe way to get ω. `ScalarBGK` also accepts a
*field* of rates, `omega_of`, for conjugate heat transfer — a solid inclusion in
a fluid. It reproduces a conductivity ratio only where ρc_p is uniform, since the
scheme transports the temperature rather than the enthalpy; where it applies, the
interface is exact rather than second order (`validation/zhou_thermal.cpp`
measures 1e-9 % at ratios from 0.1 to 100). Nothing in `src/` converts
metres and seconds — that arrow belongs to the case. The only worked example in
the tree is the local `struct Scaling` at `demonstrator/urban.cpp:123`, and it
is local on purpose: its banner argues one particular strategy (fix `dt` by
capping the fastest cell, let ω follow), which is right for an
advection-dominated problem and not in general.

### The dimensionless recipe

Three choices, and the fourth is not yours. Fixing the resolution, the Reynolds
number and the lattice velocity fixes the viscosity — you do not also get to
pick τ.

1. **Resolve the characteristic length**: `N` cells across `L`. Wall-bounded
   cases size the domain `ny = H + 2`, and the Reynolds length is `H`, not `ny`.
2. **Choose `u0` in lattice units.** It is a Mach number in disguise:
   `Ma = u0 / cs` with `cs = 1/sqrt(3) = 0.577`, and the compressibility error
   grows as `Ma^2`. Keep `u0 <= 0.05` (`Ma <= 0.087`) unless you have measured
   what a larger one costs.
3. **Viscosity follows**: `nu = u0 * N / Re`.
4. **Then** `coll.omega = Coll::omega_from_viscosity(nu)`.
5. **Read τ = 1/ω back before running.** τ → 1/2 is the stability floor. Since
   `nu = u0 * N / Re`, a larger Re is bought either with a bigger grid or a
   smaller `u0` — and a smaller `u0` costs steps (below).
6. **Pick the timescale deliberately.** One convective time is `N / u0` steps;
   the momentum-diffusion time is `N^2 / nu`. They differ by a factor of Re, and
   quoting a result after the wrong one is how a case looks converged when it is
   not: `examples/flow_past_square.cpp` records `Re_eff` falling from 49 to 26
   over 30,000 steps against a diffusive time of 576,000.

`examples/flow_past_square.cpp:92-118` is this recipe as running code, with the
τ floor check and the Mach print.

### When the geometry is physical

Set `dx = L_phys / N` metres per cell. One further choice fixes `dt`, and the
two strategies are not equivalent:

- **Cap the velocity.** Pick `u_lat_max`, then `dt = u_lat_max * dx / u_phys_max`.
  Bounds the Courant number; use it when advection dominates
  (`demonstrator/urban.cpp:256`).
- **Fix τ.** Pick τ, then `dt = (tau - 0.5) * cs2 * dx * dx / nu_phys`. Bounds
  the diffusive accuracy; use it when diffusion dominates
  (`demonstrator/urban.cpp:253`).

With `dx` and `dt` chosen, everything else follows:

```
u_lat  = u_phys  * dt / dx
nu_lat = nu_phys * dt / (dx * dx)          # and D_lat, eta_lat, the same way
a_lat  = a_phys  * dt * dt / dx            # accelerations, e.g. gravity
t_phys = steps * dt
```

### Traps

- **`cs2` is not 1/3 on every lattice.** D3Q7 is **1/4**; D2Q5 and the rest are
  1/3 (`src/lattice/Lattices.hpp`). So `tau = 3*nu + 1/2` is simply wrong on
  D3Q7. Use `omega_from_*`, which reads the lattice's own `cs2`; keep the closed
  form only for a printed diagnostic on a `cs2 = 1/3` lattice. The literal
  `0.25` in `demonstrator/urban.cpp:253` is this, in the open: it is correct
  only because that case is D3Q7, and copying the line to a D3Q19 or D3Q27
  scalar would be wrong by 4/3 without failing.
- **Halving `u0` at fixed Re and N hurts twice**: it doubles the steps to the
  same convective time *and* halves ν, pushing τ toward 1/2. Raising `N` instead
  raises ν and moves τ away from the floor; resolution is the expensive knob,
  not the dangerous one.
- Two entries under **Measurement discipline** below are units errors wearing
  another hat, and belong here too: match *kinematic* viscosity across a density
  ratio, not dynamic; and a wrong constant still gives a consistent simulation,
  so check the inputs against the source, not only against themselves.

---

## Invariants that break silently

These produce plausible, converged, wrong answers rather than crashes.

- **Direction ordering is a contract.** `opp(i)` is `i+1` for odd `i` and `i-1`
  for even (`src/lattice/Lattices.hpp`). Esoteric Pull depends on it. If you add
  or reorder a lattice's velocity set, opposite pairs must stay adjacent.
- **Storage × streaming pairings are not free.** Shifted populations centre the
  stored variable on `p̃ = 1`, which is exactly the pressure gauge that must be
  avoided at a density ratio — so the multiphase operators declare
  `RawPopulations`. `FreeSurfaceSolver` `static_assert`s *against* Esoteric Pull
  and needs `TwoLattice`, because it reads a neighbour's post-collision state
  while writing its own.
- **D3Q19 is not a product lattice.** It is D3Q27 minus its corners, so it uses a
  generated monomial basis. Neither `MultiphaseCentralMoments` nor
  `PhaseFieldCentralMoments` runs on it at all (both `static_assert` on
  `ProductBasis::enabled`, so it is a compile error rather than a wrong answer),
  and its MHD Maxwell sum leaves a ghost-mode residual. Prefer D3Q27 for
  anything above second order.
- **A published moment list belongs to a basis.** `ProductBasis` is *shifted*,
  phi_2 = C^2 - cs2; most papers tabulate *monomial* central moments, and the
  same physics occupies different slots in the two. De Rosis & Enan's Eq. (61)
  lists nine nonzero phase-field source entries; in the shifted basis six of
  them are identically zero, because the (a,a,b) slot gets
  `cs4 A_b - cs2*cs2 A_b = 0`. Transcribing such a list slot-for-slot
  double-counts; deleting terms from a monomial implementation loses them.
  Neither crashes. `tests/test_phase_field.cpp` block 8 pins both directions.
- **Wall conventions.** Wall-bounded cases size the domain as `ny = H + 2`: `H`
  fluid nodes plus one solid row each side. Getting this wrong shifts the
  Reynolds or Rayleigh number silently.
- **Coupling order belongs to the solver, not the driver.** For the phase field
  it lives in `step()`; refreshing φ late misplaces the interface rather than
  merely damping it.
- **`temperature()` is ZERO at an adiabatic scalar node**, because bounce-back
  puts the insulated plane at 0.5 and the node is a ghost outside the fluid
  (`ScalarSolver.hpp`'s `field_kernel`). Harmless when that node is `Solid` for
  the fluid — it never collides. *Not* harmless when it is a `RegWall`, which is
  a fluid node that does collide and does get forced: `BoussinesqGuo` then reads
  T = 0 against your `T0` and applies a body force along the whole wall. This is
  the concrete cost of mixing the two wall families, and it cost a benchmark run
  in `validation/zhou_thermal.cpp` (case 3.5, whose moving lid forces the fluid
  side to be on-node). Two defences: keep the temperature gauge symmetric about
  zero so that `field = 0` means *neutrally buoyant*, and use `ScalarOutflow` —
  which is on-node, zero-gradient, and reports the real temperature — where an
  on-node adiabatic wall is what you actually need.

---

## Adding a case

Two steps.

1. Write `validation/<name>.cpp` (a regression test with an exact answer) or
   `demonstrator/<name>.cpp` (something to look at). Copy the shape from a small
   existing one — `validation/tgv2d.cpp` is a compact fluid-only example,
   `validation/natural_convection.cpp` a coupled one at 56 lines.
2. Register it. `validation/CMakeLists.txt` has **two** `foreach` lists: the
   first is built *and run by `ctest`*, the second is analysis runs too slow to
   be tests. Put a regression test in the first, a sweep in the second.
   `demonstrator/CMakeLists.txt` has one list.

The skeleton — `validation/claudemd_skeleton_check.cpp` is this snippet as a
buildable file, so it stays true:

```cpp
#include "collision/BGK.hpp"       // BEFORE FluidSolver.hpp: it defines Macro,
#include "solver/FluidSolver.hpp"  // which FluidSolver.hpp uses but does not include
#include "memory/EsotericPull.hpp"

Kokkos::initialize(argc, argv);
{
  Domain d(nx, ny, nz, /*periodic x,y,z=*/true, true, true);
  Coll coll;  coll.omega = Coll::omega_from_viscosity(nu);
  FluidSolver<Lattice, EsotericPull<Lattice>, Coll> s(d, coll);
  s.initialize_field(KOKKOS_LAMBDA(Index n) { return FlowState{rho, ux, uy, uz}; });
  for (std::size_t t = 0; t < T; ++t) s.step();
}
Kokkos::finalize();
```

Headers are **not** self-contained: `FluidSolver.hpp` refers to `Macro`, which
`collision/BGK.hpp` defines, so alphabetical include order fails to compile. The
existing cases dodge this by including `validation/Campaign.hpp`, which pulls
things in the right order — use that in a validation case and you will not meet
the problem.

A validation case should compare against something with a known answer —
analytic, a published table, or a convergence rate — and print a PASS/FAIL. Cases
that merely run are demonstrators.

---

## Out of scope, and known unreliable

Do not spend time on these without saying so first; several are deliberate.

- **No MPI.** Single rank. `Domain` carries halo machinery but there is no
  exchange.
- **No contact line or wetting model** in the phase field; no open boundary for φ.
- **The free surface has no surface tension** (uniform gas pressure, no curvature
  term) and **no gas dynamics** — an enclosed bubble does not compress.
- **The free surface's moving obstacle is not reliable.** The cause is in
  `transfer_covered_mass()` and `settle()`, is written up in the module banner
  with measurements, and is not a caller error. Do not present a run with a
  moving obstacle as a result.
- **`GPU/` is D3Q27 only**, and that is now the main thing it does not share
  with `src/`. As of 2026-09-02 it also has TRT, shifted storage, central
  moments for *both* multiphase distributions (its phase field runs on D3Q7 or
  D3Q27, and the central-moment operator needs the latter), the open scalar
  boundary, Dellar's moment-based magnetic wall, the penalised rigid body with
  both shapes, and the free surface.
  regularised (on-node) velocity walls, and `hartmann` — so a wall-bounded MHD
  benchmark now runs there too.
  What it still lacks: an open boundary for the FLUID (the parent's `NrmOutXp` /
  `NrmOutFree`), D3Q19, raw MRT, and a moving obstacle in the free surface — the
  last deliberately, see that module's banner and the entry above.
  As of 2026-09-04 the D3Q7 scalar also has a **regularised collision**
  (`collide_scalar_regularised`, `ScalarOp::Regularised`) beside BGK: it relaxes
  the flux moments at ω and annihilates the three ghost moments rather than
  relaxing them. Identical to BGK at ω = 1, and it matters only near ω = 2 —
  where BGK becomes a reflection, so the ghosts invert every step and never
  damp. At Ra = 1e14 (ω_T = 1.99999905) BGK's near-wall temperature *rings*:
  Nu_bot swung 34.9 ↔ 93.6 over sixty free-fall times against an analytic ~100,
  bounded rather than diverging, so it averages into a plausible number.
  `rb_high_ra` defaults to the regularised operator and keeps `-sop bgk` to
  reproduce the ringing on demand.
  One property to know rather than a gap: **regularised walls are not mass
  conserving**, since BC3 overwrites populations. A closed box holds its mass
  exactly; a driven cavity leaks linearly and does not saturate (−1.7e-2 over
  20000 steps at 32²). Do not read an absolute pressure off a long cavity run.
- **The rigid body is of uniform density**, and there is no collision model, so
  bodies interpenetrate. It is no longer 2-D: `RigidBody3D.hpp` and
  `PenalisedBody`'s `refresh6`/`advance6` solve the full 6x6 with a rotating
  inertia tensor and a quaternion pose (`Box` shape, `demonstrator/cube_entry`).
  The two paths are **separate**, not layered: `Rect` and `Wedge` are prisms with
  `six_dof = false` and keep the validated 3x3, so calling `refresh()` on a Box
  or `refresh6()` on a Rect is a compile error rather than a wrong answer.
  `set_uniform_density6()` measures the inertia in the world frame and stores it
  in the body frame — using the 2-D `set_uniform_density()` on a 6-DOF body
  fills the mass and leaves the tensor at **zero**, which makes the angular half
  singular and produces a plausible tumble rather than a failure.

---

## Measurement discipline

This tree has produced several confident wrong conclusions. All of them came from
the same few mistakes, so they are worth naming.

- **Change one thing.** Two runs that differ in geometry *and* precision, or in
  viscosity *and* interface width, cannot attribute a difference. A comparison of
  two models at their own driver defaults is a comparison of drivers.
- **Match the right quantity.** For a density ratio, match *kinematic* viscosity,
  not dynamic. Matching μ across a ratio of 100 leaves the heavy phase at ν/100
  and drives ω to 1.994 against a limit of 2 — which reads as the model failing
  when it is the setup.
- **Check convergence before quoting.** Report a number only after the time
  series is flat, and say so if it is still creeping.
- **Agreement between a port and its host reference proves the port, not the
  physics.** Both run the same arithmetic. When a port sits several times worse
  than the code it came from, that gap is a defect until shown otherwise — do not
  write it up as a property of the model.
- **Check that a metric is reproducible before ranking with it.** The step at
  which an unstable run blows up moves by 2× with the Kokkos backend alone. It is
  not a measurement.
- **A wrong constant is still a consistent simulation.** Every test can pass, the
  device can match the host to every digit, and a quantity can converge and hold
  — while the case being solved is not the one intended. Verify inputs against
  the source paper and against the sibling implementation, not only against
  internal consistency.

---

## Conventions

- Match the surrounding code: banner comments that argue the decision, measured
  numbers rather than adjectives, and an explicit statement of what a piece does
  *not* do.
- When a limitation is found, write it into the module banner and into
  `doc/m3lb.tex`'s "Known limitations" — not only into a commit message.
- Results in `results/` are tracked reference data. Build trees and field dumps
  are ignored; `doc/m3lb.pdf` is tracked because it is the deliverable.
- Rebuild the document with `make -C doc` (needs a full TeX install).
