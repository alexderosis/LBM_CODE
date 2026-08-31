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
  generated monomial basis. The multiphase central-moment operator does not run
  on it at all, and its MHD Maxwell sum leaves a ghost-mode residual. Prefer
  D3Q27 for anything above second order.
- **Wall conventions.** Wall-bounded cases size the domain as `ny = H + 2`: `H`
  fluid nodes plus one solid row each side. Getting this wrong shifts the
  Reynolds or Rayleigh number silently.
- **Coupling order belongs to the solver, not the driver.** For the phase field
  it lives in `step()`; refreshing φ late misplaces the interface rather than
  merely damping it.

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
- **`GPU/` covers a fraction of `src/`**: D3Q27 only, raw storage only, no open
  scalar boundary, no physical magnetic wall (so every MHD case there is
  periodic — do not read a wall-bounded MHD result off `GPU/`), no free surface.
  The parent's `hartmann` case *is* wall-bounded MHD and is fine.
- **The rigid body is 2-D and of uniform density**; there is no collision model,
  so bodies interpenetrate.

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
