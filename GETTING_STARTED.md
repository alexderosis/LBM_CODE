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

## 5. Where to read next

* [`doc/m3lb.pdf`](doc/m3lb.pdf) — 73 pages: every scheme with its
  equations, why each design decision was made, the complete validation record
  with numbers, and a **list of known limitations**. Read the limitations first;
  they will save you more time than the rest.
* [`GPU/README.md`](GPU/README.md) — the CUDA code, its measured performance, and
  what it deliberately does not implement.
* The header comment of any case file. They are written to explain what the case
  tests and how it fails, not just what it computes.

## 6. Things that are absent, not merely untested

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

## 7. If something looks wrong

Run `host_check` and `host_physics` first. If those pass, the arithmetic is fine
and the problem is in the setup, the boundary conditions or the run length. If
they fail, the failing line names the quantity and prints the measured value
beside the exact one.
