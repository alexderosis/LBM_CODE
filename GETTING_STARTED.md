# Getting started

For someone who has just been given access to this repository and wants to run
the multiphysics cases — thermal convection, magnetohydrodynamics, flow around
geometry — without first reading 65 pages.

There are **two independent codes** here and it matters which one you use:

| | what it is | use it when |
|---|---|---|
| `src/` + `validation/` | the main solver, C++20 on Kokkos. 29 validation cases, five lattices, four collision operators, thermal + MHD modules, two demonstrators | you want the full physics, or you are running on CPU |
| `GPU/` | a second implementation written directly in CUDA, deliberately sharing no code with the first | you want GPU speed, or you want to check one code against the other |

They agree where they overlap — Taylor–Green to 5.3e-04 in energy — which is the
point of having both.

---

## 1. Run some physics in five minutes, with no GPU

**Start here even if you have a GPU.** The whole numerical core of `GPU/` is
written so it compiles as plain C++ as well as CUDA, so you can run complete
simulations on a laptop before you touch a cluster. Nothing about this is a toy:
it is the same per-node functions the CUDA kernels call, driven by a for-loop.

```bash
git clone https://github.com/alexderosis/M3LB.git
cd M3LB/GPU
cmake -S . -B build-host -DLBM_HOST_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

That needs **no CUDA toolkit and no GPU**. It should take about a minute to
build and ten seconds to run, and print `100% tests passed`.

What you just ran:

* `host_check` — 47 unit checks. Velocity sets, the moment transforms, one
  collision at a time.
* `host_physics` — ten *whole simulations* against closed-form answers:
  Poiseuille flow between walls, conduction between fixed-temperature plates,
  a decaying sinusoid, uniform buoyancy, resistive decay, and the shear Alfvén
  wave. Each prints the measured value beside the exact one.

Read the output of `host_physics` before anything else. It is the fastest way to
see what the code claims to do and how closely it does it.

---

## 2. The multiphysics cases

Each is a standalone program printing a table. The host build above already made
all of them, prefixed `host_`; a CUDA build makes the same programs without the
prefix, from the same source files.

### Thermal convection — `rayleigh_benard`

A fluid layer heated from below between two rigid plates. Exercises everything at
once: solid walls for the momentum, fixed-temperature walls for the scalar, and
the buoyancy force that couples them.

```bash
./build-host/host_rayleigh_benard -h 12 -ra 5000    # convection
./build-host/host_rayleigh_benard -h 12 -ra 800     # no convection
```

**Why this case and not a lid-driven cavity:** it needs no reference table.
Linear stability puts the onset of convection at **Ra_c = 1707.762**, a constant
that does not depend on the fluid or anything else. Above it convection must
sustain; below it must die.

Those two commands take about a minute each on a laptop and end with

```
  final Nu = 2.065484   max |u| = 2.9204e-02
  AS EXPECTED: at Ra = 5000.0 > Ra_c = 1707.762, convection is sustained
```

and, for Ra = 800, `Nu = 0.995366` with `max |u| = 4.27e-05` — a factor of nearly
700 in the velocity across the threshold. On a GPU at H = 24 the same pair gives
2.101 and 0.998848.

Useful flags: `-h H` layer depth in cells, `-ra`, `-pr` Prandtl number,
`-op bgk|cm` collision operator, `-steps N`, `-amp` size of the seeded
perturbation.

**Cost warning.** The run length scales as H², because that is the thermal
diffusion time — so doubling the resolution costs four times as many steps on
four times as many cells. H = 12 is about a minute on a laptop; H = 96 is
*fourteen minutes on a T4*. Start small, and use `-steps` to cut a run short
while you are finding your way.

One result worth knowing before you quote a Nusselt number: at H = 96 and
Ra = 5000, Nu was **still rising** at 15 thermal diffusion times (2.1201 → 2.1223
over the last four probes). Converged-looking is not converged; check the last
few rows of the table, not just the final line.

### Magnetohydrodynamics — `alfven` and `orszag_tang`

```bash
./build-host/host_alfven -l 64 -op cm
```

The shear Alfvén wave: an **exact solution of the full nonlinear** incompressible
MHD equations. It reports two numbers, and they fail differently, which is the
whole value of the case:

* the **wave speed** is wrong if the Lorentz coupling is wrong — visible
  immediately, at any resolution;
* the **damping rate** is wrong if the two-way coupling is evaluated one step out
  of date — and that error only shows up under refinement. `-sweep` runs three
  resolutions to expose it. **Run the sweep in FP64** (`-DLBM_DOUBLE=ON`), or the
  FP32 noise floor will look exactly like the bug it is meant to find.

```bash
./build-host/host_orszag_tang -m 24 -tmax 1.0
```

Orszag–Tang is the only case here that genuinely tests **div B**: in the wave
cases the divergence is structurally zero and would report round-off whatever the
scheme did. On a GPU at M = 128 it reaches 2.0e-02 and converges at second order.

### Flow around geometry — `channel`

```bash
./build-host/host_channel -h 16 -op cm
./build-host/host_channel -h 32 -op cm
```

Force-driven flow between two walls, against the exact parabola. Run **two**
resolutions and look at the printed `error × H²`: a wall in the wrong place gives
an error going like 1/H, so that product would grow. If it stays put, the wall is
where it should be. In FP64 it is 0.333 at H = 16, 32 and 64.

---

## 3. When you have a GPU

Same source, add the CUDA compiler:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=75
cmake --build build -j
./build/host_check && ./build/host_physics    # still worth running first
./build/rayleigh_benard -h 24 -ra 5000
./build/bench
```

`-DLBM_GPU_ARCH=` is the compute capability without the dot: 75 for a T4, 80 for
an A100, 86 for RTX 30xx, 89 for an L4 or RTX 40xx, 90 for H100/H200. Usually it
is the only thing you change.

Two things that will otherwise cost you a day:

* **FP32 is the default and should stay that way for production.** Consumer and
  data-centre-inference NVIDIA parts run FP64 at 1/32 to 1/64 of the FP32 rate.
  Build FP64 separately, for checking arithmetic, not for speed.
* **Size the problem to the card.** At 128³ this code is memory-bandwidth-bound,
  which is what a correct LBM kernel should be. Below about 64³ you are measuring
  launch overhead instead.

---

## 3b. The 3D MHD vortex on an H200

The Orszag–Tang vortex is the case that actually tests div&nbsp;B, and an H200 is
where it can be run at a resolution worth quoting.

```bash
git clone https://github.com/alexderosis/M3LB.git
cd M3LB/GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=90
cmake --build build -j

./build/host_check && ./build/host_physics      # seconds, no GPU needed
./build/orszag_tang -m 256 -tmax 4.0 -op cm     # about a minute
./build/orszag_tang -m 512 -tmax 4.0 -op cm -probes 8
```

`-DLBM_GPU_ARCH=90` is Hopper — that flag is the only thing that changes between
cards. **Run `host_check` and `host_physics` first**: they take ten seconds, need
no GPU, and if they pass then any later problem is in the setup rather than the
arithmetic.

**Sizing.** The solver holds 27 fluid populations, 21 magnetic ones and six
fields per node, so about 217 bytes in FP32:

| M | nodes | memory | steps to t = 4 | expected wall time |
|---|---|---|---|---|
| 256 | 16.8M | 3.6 GB | 23,483 | ~1 min |
| 384 | 56.6M | 12.3 GB | 35,224 | ~5 min |
| 512 | 134M | 29 GB | 46,966 | ~15 min |

Times are extrapolated from a T4, where the coupled step runs at 41% of the
uncoupled fluid rate; they have not been measured on an H200 and could be out by
a factor of two either way. Memory is comfortable in all three cases on 141 GB.

**Use `-probes` on the large grids.** Each probe copies six full fields to the
host and walks every node there to form the curl and the divergence. At 512³ that
is 3.2 GB and 134 million host iterations per sample, which can cost more than
the simulation itself. Eight probes still resolve the history; the summary lines
do not depend on how many were taken.

**What to check in the output**, in order of what would be most alarming:

1. `largest rise in E_u + E_b` must be `0.000e+00`. Ideal MHD cannot gain energy.
   It was zero at M = 64, 96 and 128 on a T4.
2. `worst max|div B| / k|B|` should keep falling at second order. Measured
   7.177e-02, 3.461e-02, 2.002e-02 at M = 64, 96, 128 — so **M = 256 should land
   near 5.0e-03 and M = 512 near 1.3e-03**. If it stalls instead, that is a real
   result and worth reporting rather than tuning away.
3. The energy history: magnetic energy should sit nearly flat to t ≈ 0.8 while
   kinetic energy falls by about a third — the flow winding up the field — and
   only then should the two decay together.

**Two things not to do.** Do not build FP64 (`-DLBM_DOUBLE=ON`) for this: it is
for checking arithmetic, and it costs a large factor on any card. And do not read
a wall-bounded MHD result off this code — a non-fluid cell is skipped, which is
bounce-back on the induction distribution and is not a physical magnetic wall.
Every MHD case here is periodic, by design.

## 3c. Throughput: the cubic box, for comparing MLUPS with other codes

The simplest thing this code runs, and the one to quote against other GPU LBM
codes: a cube, no-slip walls on all six faces, no forcing, no coupled fields.

```bash
./build/box                        # BGK and CM, 520^3 / 720^3 / 1390^3
./build/box -n 720 -op bgk         # one size, one operator
./build/box -n 720 -periodic       # walls off, for codes that time an empty box
```

FP32 is the default and is what you want here. The initial condition is a
divergence-free vortex built from a vector potential, `u = curl A`, chosen so
that all three velocity components vanish on all six wall planes — so the run is
stable and conserves mass rather than merely being fast. The `mass drift` column
is the check: it sums every slot in the lattice, wall slots included, which is
the quantity Esoteric Pull conserves exactly. If it is not ~1e-7, the MLUPS
figure beside it means nothing.

### Read the two caveats before quoting a number

**1. `1390^3` does not fit on an H200, and the tool will tell you so.**
This code stores D3Q27 in FP32: 27 × 4 = 108 B/node, plus one geometry byte.

| grid | nodes | memory | on a 141 GB H200 |
|---|---|---|---|
| 520³ | 1.41 × 10⁸ | 15.3 GB | fits easily |
| 720³ | 3.73 × 10⁸ | 40.7 GB | fits |
| 1076³ | 1.25 × 10⁹ | 136 GB | the largest cube that fits |
| 1390³ | 2.69 × 10⁹ | **293 GB** | **needs 2.1 H200s** |

There is exactly one lattice and nothing streams from host memory or splits
across devices, so capacity alone fixes the ceiling. `box` prints the largest
cube for whatever card it finds and skips the rest instead of dying inside
`cudaMalloc`. If `1390^3` came from another code's table, note what it implies:
2.69 × 10⁹ nodes in 141 GB is a budget of **52 B/node**, which is half-precision
storage (D3Q19 FP16 is 38 B, D3Q27 FP16 is 54 B) — not an FP32 comparison.

**2. MLUPS is not comparable across velocity sets.** An LBM step is
bandwidth-bound, so MLUPS ≈ achieved bandwidth ÷ bytes per node. D3Q27 FP32
moves 108 B where D3Q19 FP32 moves 76 and a D3Q19 FP16 code moves 38 — so a
D3Q19 FP16 code can report 2.8× our MLUPS while using the machine no better.
`box` therefore prints **GB/s and % of theoretical pin bandwidth** beside every
MLUPS figure. That percentage is the portable number, and it is the one to put
in front of somebody comparing implementations.

Expect the three grids to give nearly the same MLUPS. Past the L2 cache the rate
is set by bytes per node and by the memory system, not by how many nodes there
are; a sweep that climbs steeply with size is usually measuring launch overhead
at the small end.

### What it actually measures — one T4, 256³, FP32, 200 steps

| run | BGK | central moments | % of 320 GB/s peak |
|---|---|---|---|
| at rest, walls | 908.6 | 912.3 | 61.3 / 61.6 |
| decaying vortex, walls | 917.3 | 919.9 | 61.9 / 62.1 |
| at rest, `-periodic` | 932.4 | 933.8 | 62.9 / 63.0 |

Three things worth reading off that table:

* **The initial state does not change the rate.** At rest and stirred differ by
  ~1%, i.e. run-to-run noise, which is what a kernel with no data-dependent
  branch should do. So starting at rest costs nothing in realism of the number.
* **Central moments is not slower than BGK.** 912 against 909 — a 27-moment
  collision for free, because the step is bandwidth-bound and both move the same
  108 B/node. (The Kokkos code in the parent directory collapses here instead;
  that is an artefact of that implementation, not of the scheme.)
* **Walls cost about 2.6%** on this card, 932 → 909. That is the templated
  geometry path; loading the flags unconditionally cost twice as much. See the
  note at the top of `include/lbm/streaming.cuh`.

A T4 reaches ~62% of its theoretical pin bandwidth here. That percentage, not
the MLUPS, is what should carry across to an H200.

## 4. The larger validation suite

`validation/` in the parent directory has 29 cases the CUDA code does not — the
lid-driven cavity against Ghia (1982), natural convection against de Vahl Davis
(1983), Hartmann flow, inlet-driven Poiseuille, the whole De Rosis & Coreixas
(2020) campaign, grid-convergence sweeps over every lattice × operator
combination.

```bash
cd ..                 # repo root
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_THREADS=ON
cmake --build build -j
cd build && ctest --output-on-failure
```

**The Threads backend defaults to ONE thread and ignores `OMP_NUM_THREADS`.**
Pass `--kokkos-num-threads=N` on the command line or set `KOKKOS_NUM_THREADS`.
Forgetting this is the single most common way to conclude the code is slow.

---

## 5. Writing your own case

Everything above runs cases that are already here. This section is about your
problem.

Start from **`examples/flow_past_square.cpp`** — flow past a square cylinder in a
channel. It is a template rather than a result: heavily commented, split into
four blocks you change, and it exercises every piece a real case needs.

```bash
cmake --build build -j4 --target flow_past_square
./build/examples/flow_past_square --kokkos-num-threads=4
```

Converged output, about 15,000 steps:

```
  384x96   side = 16   Re = 40   u0 = 0.040 (Ma = 0.069)
  nu = 0.016000   tau = 0.5480

  step      max|u|       mean u       Re_eff     mass drift
  16000     7.0882e-02   4.0170e-02   40.2       +4.077e-07
```

and a `profile.csv` of the wake, which shows *negative* streamwise velocity on
the centreline — the recirculation bubble behind the body.

### The four blocks

1. **The discretisation.** Three type aliases fix the lattice, the collision
   operator and the streaming scheme. D2Q9 in 2D, D3Q27 in 3D. BGK is the right
   default; swap the alias for a central-moment operator and nothing else in the
   file changes.
2. **The domain and the geometry.** `set_geometry`'s callback runs **on the
   host**, once, and returns a `CellType` per cell — `Fluid`, `Solid` (halfway
   bounce-back), `RegWall`, `Excluded`. Being on the host is the point: you can
   read a mesh, an image or a height field here with no device-side machinery.
   A circle, an aerofoil, a city — anything you can answer per cell.
3. **The initial condition.** This callback runs **on the device**
   (`KOKKOS_LAMBDA`) and returns a `FlowState{rho, ux, uy, uz}`. Solid cells are
   skipped for you.
4. **The time loop and your diagnostic.** `s.step()` advances one step. The
   macroscopic fields are computed only when you ask — `compute_macroscopic()`
   or `step(true)` — because on a GPU that is a separate pass over memory.

### The two things that catch people

**Lattice units, and the Reynolds trade.** Everything is dimensionless: `dx = dt
= 1`. You pick `u0` and a length, and the viscosity follows from the Reynolds
number you want, `nu = u0 * L / Re`. Two constraints then squeeze you from both
sides:

* `u0` is a Mach number in disguise — `Ma = u0*sqrt(3)`, and compressibility
  error grows as `Ma^2`. Keep `u0 <= 0.05`.
* `tau = 3*nu + 0.5` must stay clear of `1/2`, or the scheme rings and diverges.

So a higher Reynolds number is bought with a bigger grid or a smaller `u0`, never
for free. The template prints `tau` before the run and warns below `0.51`,
because "it diverged" almost always means `tau` came out too small.

**A fixed body force does not give you the Reynolds number you asked for.** Held
constant, the force drives the flow until it balances the *total* drag — walls
plus your obstacle — and reaching that balance takes the momentum-diffusion time
`H^2/nu`, which for the template's defaults is 576,000 steps. Measured, with the
force held fixed: `Re_eff` fell from 49 to 26 over 30,000 steps and was still
falling. The template therefore adjusts the force to hold the mean velocity, in
about ten lines, which is what makes `-re 40` actually run at 40. If you drive
your case with a force, do the same.

### Where to put the file

Three directories, and the distinction is worth keeping:

| directory | what belongs there | built | run by `ctest` |
|---|---|---|---|
| `validation/` | has a **known answer** — analytic, a published table, a convergence rate — and prints PASS/FAIL | yes | yes, if in the first `foreach` list |
| `demonstrator/` | produces something to look at; no exact answer | yes | no |
| `examples/` | templates to read and copy | yes | no |

Add your file's name to the `foreach` list in that directory's
`CMakeLists.txt`. A validation case goes in the **first** list in
`validation/CMakeLists.txt` (the one that calls `add_test`); the second list is
for analysis runs too slow to be tests.

### If you are using Claude Code

`CLAUDE.md` in the repository root is loaded automatically and carries the
conventions, the invariants that break silently, what is out of scope, and the
measurement traps. You should not need to paste any of it.

**You do not need a terminal for this.** Claude Code runs as a VS Code or
JetBrains extension, as a desktop app on Mac and Windows, and in the browser at
`claude.ai/code`, as well as from the command line. `CLAUDE.md` is read from the
project directory, so it behaves the same in all of them — if you already have
the repository open in an editor, the extension is the shortest path.

---

## 6. Where to read next

* [`doc/m3lb.pdf`](doc/m3lb.pdf) — 73 pages: every scheme with its
  equations, why each design decision was made, the complete validation record
  with numbers, and a **list of known limitations**. Read the limitations first;
  they will save you more time than the rest.
* [`GPU/README.md`](GPU/README.md) — the CUDA code, its measured performance, and
  what it deliberately does not implement.
* The header comment of any case file. They are written to explain what the case
  tests and how it fails, not just what it computes.

## 7. Things that are absent, not merely untested

So you do not spend a week looking for them.

* `GPU/` has **no open (outflow) boundary for the scalar** and **no physical
  magnetic wall condition** — every MHD case there is periodic. Skipping a cell
  is bounce-back on the induction distribution, which is neither the perfectly
  conducting nor the insulating condition.
* `GPU/` stores raw populations, no shifted storage, and it costs FP32 accuracy
  at fine resolution.
* In the Kokkos code, the **central-moment operators are unusable on a GPU** —
  two configurations did not finish in seventeen minutes at 64³ where BGK took
  0.03 s. Unexplained; register spilling is ruled out. Use `GPU/` if you need
  central moments on a device.
* D3Q19 is excluded from current validation work.

## 8. If something looks wrong

Run `host_check` and `host_physics` first. If those pass, the arithmetic is fine
and the problem is in the setup, the boundary conditions or the run length. If
they fail, the failing line names the quantity and prints the measured value
beside the exact one.
