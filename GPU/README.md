# Native CUDA lattice Boltzmann

A second implementation, independent of the Kokkos code in the parent directory.
It shares no headers with `../src`, is not built by the parent's CMake, and is
not a port of it — only the physics and the output format are deliberately the
same, so the two can be compared.

**Status: core complete, built and validated on a Tesla T4. Physics coverage is
a fraction of the parent's.** Read the scope table before assuming anything works.

## Measured, on a Tesla T4 (sm_75, CUDA 12.8, FP32, 64^3)

| operator | MLUPS | GB/s | mass drift |
|---|---|---|---|
| BGK | 943.1 | 203.7 | 0.000e+00 |
| central moments | 933.0 | 201.5 | 0.000e+00 |

Central moments runs at **99% of BGK speed**, and 203.7 GB/s is **64% of the
T4's 320 GB/s peak** — i.e. the kernel is memory-bound, which is what a
correctly written LBM kernel should be. This matters beyond this code: the
parent Kokkos implementation could not complete two central-moment
configurations at the same size in seventeen minutes, and `-Xptxas -v` ruled out
register spilling as the cause. The same algorithm at BGK speed here shows the
collapse there is **not** intrinsic to the operator.

**Taylor–Green at Re=1600, 64^3, tau=0.502400, central moments**, against the
parent's committed FP64 reference `../results/E_tgv3d/tgv3d_re1600_d3q27_cm.dat`:

| | worst relative difference |
|---|---|
| kinetic energy E/E0 | 5.3e-04 |
| enstrophy Psi/Psi0 | 2.1e-03 |

Enstrophy peak 2.1292 at t\*=1.000 against the parent's 2.1305 at t\*=1.000.
Two independent implementations, FP32 against FP64 — this is the agreement
FP32 round-off alone would predict. 32,000 steps in 9.12 s (920 MLUPS).

## Why it is written this way

Every function in `include/lbm/core.cuh` and the indexing half of
`include/lbm/solver.cuh` is marked `LBM_HD`, which expands to
`__host__ __device__` under nvcc and to *nothing* under a plain C++ compiler.
That is not portability for its own sake. It means the whole numerical core —
equilibrium, the central-moment transform, the collision, the streaming
indexing — compiles and runs on a machine with **no GPU at all**, and
`test/host_check.cpp` exercises it there.

This is not decoration. Writing the core this way caught a real bug before a
single kernel was launched: `eq_factors` added a `cs^2` that `fwd1d` already
subtracts, so equilibrium was not a fixed point of the collision. The flow would
have decayed *plausibly but wrongly*, which is close to the worst possible
failure to have to diagnose on a remote GPU through a notebook.

Run the host checks first. Always.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=75
cmake --build build -j
./build/host_check      # no GPU needed
./build/tgv3d -d 64 -op cm
./build/bench
```

## What is implemented

| | this code | parent (Kokkos) |
|---|---|---|
| lattices | D3Q27 | D2Q9, D2Q5, D3Q7, D3Q19, D3Q27 |
| collision | BGK, central moments | BGK, TRT, raw MRT, central moments |
| streaming | Esoteric Pull | Esoteric Pull, two-lattice |
| storage | raw | raw, shifted |
| boundaries | **periodic only** | bounce-back, regularised, outflow, moment-based |
| forcing | **none** | Guo, high-order Hermite |
| thermal / MHD | **none** | both |
| geometry | **none** | arbitrary voxel input |
| cases | Taylor–Green, benchmark | ~20 validation cases |

Everything in bold is absent, not merely untested. This code cannot run the
aorta, cannot run Hartmann flow, cannot run a channel, and has no way to impose
a wall.

## Verification

`host_check` runs 20 checks with no GPU: the velocity set and its moments, the
`opp(i) == i+1` pairing that Esoteric Pull depends on, the direction table,
equilibrium moments, the central-moment transform round trip, equilibrium in
central-moment space, mass and momentum conservation for both operators,
equilibrium as a fixed point, `gather(init_scatter(x)) == x`, and — the one the
scheme lives or dies on — that a single streaming step transports each of the 26
directions by exactly `c_i` and nowhere else.

The Taylor–Green case prints the same columns as the parent's
`validation/tgv3d.cpp`, whose committed reference is
`../results/E_tgv3d/tgv3d_re1600_d3q27_cm.dat` at D=64, Re=1600, τ=0.502400.
The two will **not** agree bit for bit — this code is FP32 by default where the
parent is FP64 — but the energy and enstrophy curves should agree closely.
A visible discrepancy is a port bug, and matching the output format is what
makes that easy to see.

## Notes for anyone extending this

**Do not use lambdas for kernel bodies or initial conditions.** Use a struct
with `operator()`. nvcc forbids extended `__host__ __device__` lambdas inside
generic lambdas, forbids function-local types in their template arguments, and
forbids first-capturing a variable inside an `if constexpr`. Taking the parent
implementation to a GPU cost six separate fixes for exactly these rules, none of
which the CPU backend can see. Structs sidestep all of them.

**`init_scatter` is not `scatter`.** `scatter` writes post-collision populations
into the slots the *next* parity reads — that is the streaming step. Using it to
lay down an initial condition shifts every population by one cell before the
first step. `init_scatter` is the exact inverse of `gather`. `host_check`
asserts the round trip.

**Keep the layout SoA.** `f[i * n_nodes + node]`, so consecutive threads touch
consecutive addresses. This is the single most important decision in a GPU LBM
code, and the reason it is written this way round rather than AoS.

**Watch registers before blaming arithmetic.** Build with
`-DLBM_PTXAS_VERBOSE=ON` and read the spill columns. In the parent
implementation the central-moment operator ran orders of magnitude slower than
BGK on a T4; register spilling was the obvious hypothesis and `-Xptxas -v`
**ruled it out** (zero spill bytes, 116 registers). That question is still open,
and it is the main reason this code exists — so measure before theorising.
