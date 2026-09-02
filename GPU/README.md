# Native CUDA lattice Boltzmann

A second implementation, independent of the Kokkos code in the parent directory.
It shares no headers with `../src`, is not built by the parent's CMake, and is
not a port of it — only the physics and the output format are deliberately the
same, so the two can be compared.

**Status, in two halves.**

*The D3Q27 fluid core* is built and validated on a Tesla T4 and an H200. Numbers
below are measured on those devices.

*Thermal transport, MHD and geometry* were added afterwards. They were written
and verified on a machine with no GPU, using the reference driver in
`include/lbm/hostsim.hpp`, and then **built with nvcc and run on a T4** — see the
next section. The build was clean first try and every physics result reproduced
what the host had predicted, including one it predicted would appear only at
higher resolution. Read the scope table before assuming anything works.

*TRT, central moments for both multiphase distributions, the open scalar
boundary, the magnetic wall, the penalised rigid body, the free surface and
shifted storage* were added last, closing every gap the parent's `CLAUDE.md`
listed against this tree bar one (regularised walls). They were written the same
way — host first, `hostsim.hpp`, six test binaries — and the host build is what
the numbers quoted for them come from unless a section says otherwise.

That work also found a **pre-existing bug in this code**: `compute_macro` did
not apply Guo's half-force shift while the step kernel did, so every forced run
reported a velocity uniformly low by `F/(2ρ)` — exactly `G/2` at every node, to
eight digits. It is fixed. A constant offset is invisible in a profile *shape*,
which is how it survived; what it moved was the fitted wall position, and it had
led to a wrong conclusion about TRT that is retracted below.

## Measured, on a Tesla T4 (sm_75, CUDA 12.8, FP32)

| operator | grid | MLUPS | GB/s | mass drift |
|---|---|---|---|---|
| BGK | 64^3 | 943.1 | 203.7 | 0.000e+00 |
| central moments | 64^3 | 933.0 | 201.5 | 0.000e+00 |
| BGK | 128^3 | 980.3 | 211.8 | 0.000e+00 |
| central moments | 128^3 | 973.3 | 210.2 | 0.000e+00 |

These predate geometry. The current figures at 128^3 are 950.16 (BGK) and 950.37
(central moments) — a 2.9% cost that survives after the geometry check was moved
behind a template. Measured A/B on one card; see "Geometry cost 5.6%" below.

## Measured, on an NVIDIA H200 (sm_90, CUDA 12.8, FP32, 512^3)

| operator | MLUPS | GB/s | % of 4814 GB/s peak | mass drift |
|---|---|---|---|---|
| BGK | 17233.4 | 3722.4 | 77.3% | 0.000e+00 |
| central moments | 17171.4 | 3709.0 | 77.0% | 0.000e+00 |

Central moments at **99.6% of BGK**, now on two GPU generations, which closes
the question this code was written to answer. The efficiency is *higher* than
the T4's 66%, so the H200 delivers 17.6x the T4's throughput against a 15.0x
bandwidth ratio. Taylor-Green at 512^3 -- 134M nodes, 256000 steps -- runs in
2046 s at 16792 MLUPS with zero mass drift.

Central moments runs at **99% of BGK speed**, and 211.8 GB/s is **66% of the
T4's 320 GB/s peak** — i.e. the kernel is memory-bound, which is what a
correctly written LBM kernel should be. Going from 64^3 to 128^3 buys only ~4%,
which says 64^3 already came close to saturating this card; on a larger device
the useful sizes start well above that. This matters beyond this code: the
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

**And it converges with resolution.** The same case at 128^3 (tau = 0.504800,
64,000 steps in 138 s, 972.8 MLUPS):

| | 64^3 | 128^3 |
|---|---|---|
| enstrophy peak | 2.1292 | 2.2746 |
| E/E0 at t\*=10 | 0.014462 | 0.014929 |

The peak rises 6.8% and less energy is lost, both being the under-resolution
signature lifting: at 64^3 the cascade reaches the grid scale early and
dissipates energy prematurely, damping the vorticity peak.

**This case is NOT the published Taylor-Green benchmark, and cannot be compared
to it.** An earlier version of this file said 128^3 was "far short of the
256^3-512^3 the published DNS uses", which implies the same problem at lower
resolution. It is a different problem, for two independent reasons. The initial
condition here has three non-zero velocity components with factors of 1/2, where
the benchmark has w = 0. And Re is defined as u0*D/nu, where the benchmark uses
Re = V0*L/nu on a box of side 2*pi*L -- so L = D/(2 pi) and the two differ by a
factor of 2 pi: what this case calls Re = 1600 is Re = 255 in benchmark terms.

`src/tgv3d_bench.cu` runs the benchmark configuration proper, with the w = 0
initial condition, Re on L, and the dissipation rate reported both ways so the
gap between them measures resolution. Its Kokkos twin is
`../validation/tgv3d_bench.cpp`; the two share the initial condition, the
viscosity and the non-dimensionalisation line for line.

## Thermal, MHD and geometry — confirmed on a T4

Written and debugged with no GPU, then run on one. Tesla T4, sm_75, CUDA 12.8,
FP32 unless stated.

**The build was clean on the first attempt** — eight targets, `cmake` plus
`nvcc`, 1m03s, zero errors and zero warnings. That is the whole return on
writing the core as `LBM_HD` and using structs rather than lambdas: the six nvcc
restrictions the Kokkos port had to discover one at a time never came up, because
the code was written already knowing them.

| test | on the device | what the host predicted |
|---|---|---|
| `host_check`, 47 unit checks | ALL PASSED | ALL PASSED |
| Poiseuille, error × H² at H = 16 | 0.339 | 0.336 (FP32), 0.333 (FP64) |
| Rayleigh–Bénard, Ra = 5000, H = 24 | Nu = 2.100998 | 2.0437 at H = 12 |
| Rayleigh–Bénard, Ra = 800, H = 24 | Nu = 0.998847, max&#124;u&#124; 5.6e-06 | 0.9954 at H = 12 |
| Alfvén wave, L = 64, speed | +1.288e-04 | −5.280e-05 |
| Alfvén wave, L = 64, damping | +1.623e-03 | +1.679e-03 |
| Orszag–Tang, M = 64, max&#124;div B&#124;/k&#124;B&#124; | 7.336e-02 | 1.091e-01 at M = 32 |
| Orszag–Tang, M = 64, energy rise | 0.000e+00 | 0 from M = 32 |

Two of those are worth pausing on.

**The Rayleigh–Bénard offset below onset refined exactly as predicted.** The host
measured Nu = 0.9954 at H = 12 and 0.99873 at H = 24, and called the deviation a
second-order artefact. The device, independently, at H = 24: 0.998847. The
prediction was made before any of this ran on a GPU.

**Orszag–Tang's energy budget cleaned up exactly where it was predicted to.** The
host found a spurious energy rise of 1.10e-02 at M = 12 falling to zero by
M = 32, and argued it was under-resolution rather than a defect. At M = 64 on the
device the rise is 0.000e+00, and div B has fallen to 7.34e-02 from 1.09e-01 at
M = 32 — still converging.

**The FP64 Alfvén sweep on the device reproduced the host to every digit printed:**

| L | speed error | damping error |
|---|---|---|
| 32 | +2.222e-04 | +2.080e-03 |
| 64 | +5.570e-05 | +4.972e-04 |
| 128 | +1.397e-05 | +1.249e-04 |

Identical values from a serial host loop and from 128-thread CUDA blocks. In
FP64 the two execution paths agree to the printed precision, which is about as
strong a statement as this arrangement can make.

### Geometry cost 5.6%, not the half a per cent it was written down as

The flags array was, at first, always loaded — even for a periodic box with no
geometry at all — because one byte per node against 216 of population traffic is
half a per cent, and that is what `streaming.cuh` called it: "a rounding error in
bandwidth". Three points on one T4 at 128³:

| | BGK | central moments | |
|---|---|---|---|
| A · `cf19ebe`, before geometry existed | 978.15 | 977.35 | |
| B · `7f84999`, flags always loaded | 923.99 | 919.81 | −5.5%, −5.9% |
| C · `68e84e2`, geometry templated | 950.16 | 950.37 | −2.9%, −2.8% |

The byte-count estimate was out by a factor of ten. What a bandwidth-bound kernel
pays for is not the byte but the extra memory **stream** — more transactions,
more cache pressure, another address in flight.

**The template recovers about half of it.** `HasGeometry` is now a compile-time
parameter of the fluid, scalar and magnetic node updates, and the short-circuit
`if (HasGeometry && p.flags[n] != Fluid)` means a periodic run never emits the
load at all. Worth 2.8–3.3%.

**The remaining 2.9% is not explained, and it is not the obvious candidates.**
`-Xptxas -v` on the same two builds: the old `stream_collide` used 96 registers
across its 4 instantiations, the new `fluid_kernel` uses 63–80 across its 40, and
**both spill zero bytes**. So it is neither register pressure nor spilling —
registers went *down*. What else differs is a much larger kernel parameter struct
and a runtime `if (p.ux_out)` for the coupled-velocity write. `ncu --set full` on
one kernel would settle it. Measure before theorising; that is what this file
says elsewhere and it applies here too.

Mass drift stays exactly 0.000e+00 at every point above, and central moments
stays level with BGK, so nothing about the operator changed.

## The host verification that predicted all of the above

Everything in this section was produced by `include/lbm/hostsim.hpp`, on a laptop
with no GPU, before any of the device runs above.

It is worth more than it sounds, because it is not a second implementation. The per-node update is a plain `LBM_HD` function —
`fluid_node_update`, `scalar_node_update`, `magnetic_node_update` — and the CUDA
kernel is three lines of index arithmetic around it. The host driver calls the
same function in a serial loop, which is legitimate because under Esoteric Pull
each storage slot has exactly one writer per step: the nodes of one step commute,
so a for-loop is not an approximation to the launch, it is the same computation.
What remains unverified is the launch itself — grid configuration, register
pressure, transfers.

`test/host_physics.cpp` runs these and passes in **both** precisions:

| case | what it pins down | FP32 | FP64 |
|---|---|---|---|
| Poiseuille, bounce-back walls | wall placement, Guo forcing | 1.4e-03 | 1.3e-03 |
| closed box | every slot written exactly once | 1.6e-06 | 1.4e-14 |
| insulating box | the scalar is conserved | 9.7e-09 | 1.0e-13 |
| conduction, Dirichlet walls | anti-bounce-back, plane placement | 1.4e-06 | 2.5e-15 |
| decaying sinusoid | the D3Q7 diffusivity, cs² = 1/4 | 8.1e-04 | 8.1e-04 |
| advected sinusoid | the advective flux | 4.4e-04 | 4.4e-04 |
| uniform buoyancy | the whole Boussinesq path | 6.7e-05 | 2.9e-13 |
| resistive decay | induction alone, no flow | 4.1e-03 | 4.1e-03 |
| shear Alfvén wave, BGK | Lorentz coupling + induction + order | 8.1e-04 | 3.3e-04 |
| shear Alfvén wave, CM | the same, central moments | 8.4e-05 | 3.8e-04 |

Figures are worst relative error against the closed-form answer. Where FP32 and
FP64 agree, the number is discretisation error and not round-off.

**The wall is exactly where it should be.** For Poiseuille the interesting
quantity is not the error but how it scales: a wall in the wrong place gives an
error going like 1/H, a correctly placed one leaves the 1/H² truncation error.
In FP64, `channel` gives

| H | amplitude error | × H² |
|---|---|---|
| 16 | −1.631e-03 | 0.333 |
| 32 | −4.071e-04 | 0.333 |
| 64 | −1.017e-04 | 0.333 |

— constant to three digits over two doublings. In FP32 the same run reads 0.336
at H = 16 and 0.498 at H = 32: the raw-population storage runs out of mantissa
before the discretisation error does. That is the cost of not having shifted
storage, and it is why the parent has it.

**Conduction confirms where the thermal plane sits.** With Dirichlet layers at
y = 0 and y = H+1, the measured gradient is 0.062500 = 1/16 exactly, which is the
distance between the two PLANES; measuring between the NODES would give 1/15 =
0.066667. Both walls — no-slip and isothermal — land on the same two planes,
which they must, or H would mean two different things in the Rayleigh number.

**The D3Q7 sound speed is cs² = 1/4, and the sinusoid proves it.** The decay-rate
error is −0.327% at L = 16 and −0.081% at L = 32, a ratio of 4.05. A wrong cs²
would be an offset that does not converge; this converges at k².

### Rayleigh–Bénard brackets a constant of nature

`rayleigh_benard` needs no reference table. Linear stability puts the onset of
convection between two rigid plates at **Ra_c = 1707.762**, independent of Pr and
of everything else. Central moments, 15 thermal diffusion times, on a T4:

| H | nodes | Ra = 800 (below) | Ra = 5000 (above) | steps | MLUPS |
|---|---|---|---|---|---|
| 24 | 4,992 | Nu = 0.998848, max&#124;u&#124; 5.60e-06 | Nu = 2.100998, max&#124;u&#124; 1.51e-02 | 432,000 | 510 |
| 48 | 19,200 | Nu = 0.999703, max&#124;u&#124; 1.00e-06 | Nu = 2.109297, max&#124;u&#124; 7.60e-03 | 1,728,000 | 1099 |
| 96 | 75,264 | Nu = 0.999912, max&#124;u&#124; 8.42e-07 | Nu = 2.122282, max&#124;u&#124; 3.85e-03 | 6,912,000 | 612 |

Four orders of magnitude in the velocity across the threshold at H = 96. This is
the case that exercises everything added here at once — solid walls for the
momentum, Dirichlet walls for the scalar, and the Boussinesq force that couples
them.

**Below onset the prediction held exactly.** The exact answer is Nu = 1, and the
deviation was argued from host runs to be a second-order artefact of a spurious
wall-driven flow. Measured: Nu − 1 = −1.152e-03, −2.97e-04, −8.8e-05 at
H = 24, 48, 96, which are convergence exponents of 1.95 and 1.76. The residual
velocity is *independent of the seeded perturbation* (4.263e-05 with no seed at
all against 4.271e-05 with a seed a hundred times larger, at H = 12) and linear
in Ra, so it is a steady spurious flow the body force drives near the walls, and
it refines away at the scheme's design order.

**Above onset, do NOT read those three numbers as grid convergence.** They rise
by increasing amounts — +0.0083 then +0.0130 — which is the opposite of a
converging sequence, and the reason is that the H = 96 run has not reached steady
state. Its last five probes:

| t / t_diff | 12.00 | 12.75 | 13.50 | 14.25 | 15.00 |
|---|---|---|---|---|---|
| Nu | 2.120058 | 2.120604 | 2.121189 | 2.121720 | 2.122282 |

still climbing at about 7e-04 per diffusion time. **Nu = 2.1223 is a lower bound,
not a converged value,** and a proper number needs a run several times longer —
which at H = 96 costs 14 minutes per 15 diffusion times on a T4. What the three
rows do establish is the two-sided bracket of Ra_c, which is what the case was
brought in for.

**max|u| halves when H doubles**, 1.51e-02 → 7.60e-03 → 3.85e-03, which is
exactly right: with Ra and D fixed in lattice units the velocity scale is D/H, so
diffusive scaling demands u ∝ 1/H. A quantity that did not do that would mean the
non-dimensionalisation was wrong.

**The throughput curve is not monotonic, and the reason is the L2 cache.** At
H = 24 the grid is 4,992 nodes — 39 blocks against 40 SMs, so barely one block
each and the kernel is occupancy-bound at 510 MLUPS. At H = 48 the whole lattice
is 19,200 × 27 × 4 = 2.1 MB, which fits inside the T4's 4 MB L2: 1099 MLUPS,
*faster than the 950 the bandwidth-bound 128³ benchmark reaches*. At H = 96 the
lattice is 8.1 MB, L2 no longer holds it, and it drops back to 612.

### The Alfvén wave is the case that earns its keep

An exact solution of the full **nonlinear** incompressible MHD equations: u is
perpendicular to B₀ and everything depends only on x, so (u·∇)u vanishes
identically while (B·∇)B does not. Both the Lorentz coupling and the induction
equation are driven, and both must be right.

The two measurements fail differently, which is the whole point. An error in the
**Lorentz coupling** shows up as the wrong wave *speed*, at any resolution. An
error in the **coupling order** shows up only in the *damping*, and only under
refinement — stepping the fluid against a field from the previous step is a
first-order splitting error, and under diffusive scaling it appears as a damping
offset that survives every grid refinement. The parent implementation found
exactly that: its damping error *grew*, 1.55e-2 → 2.79e-2 → 3.16e-2, while the
phase speed converged cleanly.

`alfven -sweep`, FP64, diffusive scaling (v_A ~ 1/L at fixed ν and η, so the
Reynolds and Lundquist numbers are the same on every grid):

| L | speed error | ratio | damping error | ratio |
|---|---|---|---|---|
| 32 | +2.222e-04 | | +2.080e-03 | |
| 64 | +5.570e-05 | 3.99 | +4.972e-04 | 4.18 |
| 128 | +1.397e-05 | 3.99 | +1.249e-04 | 3.98 |

**Both converge at second order.** The coupling is simultaneous, not lagged.

The same sweep in FP32 reads +2.9e-04 → −4.9e-04 → −1.1e-03 on the speed and
+2.7e-03 → +3.8e-03 → +5.1e-02 on the damping — errors that *grow*, which is the
signature of the bug the sweep exists to find. They are not. Under diffusive
scaling v_A goes like 1/L, so at L = 128 the perturbation is 5e-04 on populations
of order 0.3, which leaves about three decimal digits above FP32 noise. **Run the
sweep in FP64 or do not run it**, and this is a fair warning about how easy it is
to read a precision failure as a physics failure.

### Orszag–Tang: the only case here that tests div B, run properly

In both wave cases div B is structurally zero and reports round-off whatever the
scheme does. Orszag–Tang's nonlinear dynamics makes every component depend on
every coordinate, so a scheme that generates monopoles shows it.

Re = 100, Ma = 0.034, Pr_m = 1, central moments, to t = 4 — the same physical
problem on three grids, on a T4:

| M | nodes | τ | steps | max &#124;div B&#124;/k&#124;B&#124; | energy rise | MLUPS |
|---|---|---|---|---|---|---|
| 64 | 262,144 | 0.513325 | 5,870 | 7.177e-02 | 0.000e+00 | 359.2 |
| 96 | 884,736 | 0.519988 | 8,805 | 3.461e-02 | 0.000e+00 | 378.4 |
| 128 | 2,097,152 | 0.526650 | 11,741 | 2.002e-02 | 0.000e+00 | 391.6 |

**div B converges at second order** — the ratios are 1.80 and 1.90 in the
exponent, against 1.5× and 1.33× refinements. (The coarse host runs at M = 16–32
suggested only first order; they were deep in the under-resolved regime and the
estimate was too pessimistic.) The antisymmetry of the induction equilibrium's
first moment is what keeps this bounded, and `host_check` asserts that
antisymmetry directly.

**The energy budget is clean at every resolution here.** Ideal incompressible MHD
has dE/dt = −ν|∇u|² − η|∇B|², so E_u + E_b must never rise, and it does not:
0.000e+00 at all three M. The spurious rise the host reported (1.10e-02 at
M = 12) was under-resolution, and it is gone by M = 32.

M = 128, 11,741 steps in 62.9 s. The history is the physics:

| t | E/E0 | E_u/E0 | E_b/E0 | max&#124;div B&#124;/k&#124;B&#124; | J_max |
|---|---|---|---|---|---|
| 0.0 | 1.000000 | 0.510204 | 0.489796 | 2.238e-06 | — |
| 0.4 | 0.918199 | 0.454361 | 0.463839 | 5.113e-03 | 3.007e-03 |
| 0.8 | 0.798871 | 0.331558 | 0.467313 | 9.537e-03 | 4.821e-03 |
| 1.2 | 0.656237 | 0.242583 | 0.413654 | 1.458e-02 | 6.962e-03 |
| 1.6 | 0.527835 | 0.203742 | 0.324093 | 2.002e-02 | 4.449e-03 |
| 4.0 | 0.170094 | 0.070211 | 0.099883 | 4.002e-03 | 1.354e-03 |

Two things to read off it. Up to t ≈ 0.8 the **magnetic energy barely moves**
(0.4898 → 0.4673) while the kinetic energy falls by a third (0.5102 → 0.3316):
the flow is winding up the field, and the transfer is nearly lossless. After that
both decay together, which is resistive dissipation in the current sheets taking
over. And div B is largest exactly where the sheets are sharpest — it peaks at
t ≈ 1.6 and then falls back by a factor of five.

**J_max peaks between t = 1.0 and t = 1.4** on this Δt = 0.2 sampling, which is
consistent with the paper's statement that the current "shows a maximum at about
t = 1" — but the sampling cannot say more than that, and the driver deliberately
asserts nothing about it. Figure 6 there is a plot against a pseudospectral run
whose values are not available numerically, so no per-point comparison is
possible and none is claimed.

**Not claimed:** when J_max peaks. That is a comparison against a pseudospectral
reference whose values are not available numerically, so the driver prints the
history and asserts nothing about it.

## Running on another card

`-DLBM_GPU_ARCH=` takes the compute capability without the dot: 70 Volta,
75 Turing (T4), 80 A100, 86 RTX 30xx, 89 L4 / RTX 40xx, **90 Hopper (H100,
H200)**. That flag is normally the only thing to change.

Two notes for large or modern devices. **Size the problem to the card** — 512^3
is 134M nodes and 14.5 GB in FP32, which is nothing on an H200's 141 GB but is
where such a card starts to be used properly; the 128^3 figures above would
badly understate it. And **FP64 is worth building separately** (`-DLBM_DOUBLE=ON`)
on parts with full-rate FP64, where it allows a direct comparison against the
parent's FP64 references rather than the 5e-4 agreement FP32 gives. Strip
`--use_fast_math` from CMakeLists.txt if strict IEEE is wanted for that.

`src/bench.cu` reports peak bandwidth via `cudaDeviceProp::memoryClockRate`,
deprecated in CUDA 12 and possibly absent in 13. If it will not compile, delete
that one `printf`; nothing else uses it.

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

**That property has since been extended to whole simulations.** Every driver in
`src/` is written against the aliases in `include/lbm/backend.cuh`, which resolve
to the CUDA solvers under nvcc and to the host reference drivers under a plain
C++ compiler. The two sets of classes expose the same interface, so one source
builds and runs both ways with no `#ifdef` of its own — and a wrong initial
condition, a wrong Rayleigh number, a wrong diagnostic or a wrong coupling order
can be found on a laptop before anything reaches a device.

Run the host checks first. Always.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=75
cmake --build build -j
./build/host_check         # no GPU needed, seconds
./build/host_physics       # no GPU needed, under ten seconds
./build/tgv3d -d 64 -op cm
./build/bench
```

On a machine with no CUDA toolkit at all, configure without it. Every check and
every driver still builds, as plain C++17:

```bash
cmake -S . -B build-host -DLBM_HOST_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
./build-host/host_rayleigh_benard -h 12 -ra 5000
```

The drivers, and what each is for:

| driver | case | checks |
|---|---|---|
| `tgv3d` | Taylor–Green vortex | the fluid core, against the parent's reference |
| `tgv3d_bench` | the published DNS benchmark | resolution |
| `bench` | throughput | MLUPS and GB/s |
| `channel` | force-driven Poiseuille | geometry, forcing, where the wall is |
| `rayleigh_benard` | convection between plates | thermal + walls + buoyancy, against Ra_c |
| `alfven` | shear Alfvén wave | the Lorentz coupling, and the coupling order |
| `orszag_tang` | Orszag–Tang 3D | div B, the energy budget |

Add `-DLBM_HOST_APPS=ON` on a machine that has CUDA to get `host_*` twins of
every driver alongside the device ones.

## What is implemented

| | this code | parent (Kokkos) |
|---|---|---|
| lattices | D3Q27 fluid, D3Q7 or D3Q27 phase, D3Q7 scalar and field | D2Q9, D2Q5, D3Q7, D3Q19, D3Q27 |
| collision | BGK, TRT, central moments | BGK, TRT, raw MRT, central moments |
| streaming | Esoteric Pull; two-lattice for the free surface | Esoteric Pull, two-lattice |
| storage | raw, shifted | raw, shifted |
| boundaries | periodic, bounce-back, scalar adiabatic / Dirichlet / outflow, magnetic moment-based | + regularised |
| forcing | Guo: uniform, Boussinesq, arbitrary per-node field | Guo, high-order Hermite |
| thermal | advection–diffusion + Boussinesq | same |
| MHD | Dellar induction + Maxwell stress, BGK and central moments, moment-based walls | + the published D2Q9 scheme |
| multiphase | colour gradient + phase field, each with BGK **and** central moments; free surface | same, + D2Q9/D3Q19 |
| rigid body | volume penalisation, Rect and Wedge | same, + a moving obstacle in the free surface |
| geometry | arbitrary voxel input | arbitrary voxel input |
| cases | 8 drivers | ~20 validation cases |

Still absent, not merely untested:

* **regularised (on-node) velocity walls.** This has halfway bounce-back only,
  which puts the no-slip plane half a cell outside the last fluid node — while
  the magnetic moment condition puts B exactly *on* it. Mixing them is a
  half-cell disagreement about the channel width, which is precisely the error
  the moment condition exists to avoid, so **a wall-bounded MHD benchmark such
  as Hartmann still wants the parent.** The magnetic wall itself is here and
  measured; what is missing is its velocity counterpart.
* **a moving obstacle in the free surface.** Deliberately not ported: the
  parent's own banner records two measured defects that are left in because
  fixing either makes its demonstrator fail *sooner*, so its reach depends on
  those errors cancelling something unidentified. Static walls are here.
* **surface tension and gas dynamics in the free surface**, which is the method
  rather than an omission — see the banner in `freesurface.cuh`.
* **conducting and insulating magnetic walls.** `MagDirichlet` imposes a
  prescribed external field and `MagOutflow` a zero gradient; neither is a wall
  coupled to a wall current or matched onto an exterior vacuum field.
* **wetting / contact angle** for the phase field, and any open boundary for φ.
* **D3Q19, raw MRT, the aorta, height-field input.**

## The colour gradient — confirmed on a T4

Written and debugged with no GPU, then run on one. Tesla T4, sm_75, CUDA 12.8.
The build was clean first try, nine targets, 6m38s, and the FP64 build of the
same driver is clean too.

> **The numbers in this section were re-measured after a defect was found in the
> driver.** `src/droplet.cu` seeded `tanh[(r-R)/W]` where Saito et al. Eqs.
> (41)–(42), the parent's `validation/static_droplet.cpp` and `src/bubble.cu`
> all seed `tanh[2(r-R)/W]` — so every colour-gradient case here ran at twice
> the intended interface width. Fixed; what it cost is at the end of this
> section.

**The device reproduced the host reference to every digit printed**, on
everything except one noisy diagnostic. Same case, 40³ with R = 10 and W = 2,
3000 steps, both in FP64 — one on a laptop through the serial driver in
`hostsim.hpp`, one in CUDA kernels on the card:

| | sigma | error | R_eff | spurious current | blue in the core |
|---|---|---|---|---|---|
| host reference, FP64 | 1.047132e-03 | +4.71% | 10.082 | 1.654e-05 | 1.720e-04 |
| T4, FP64 | 1.047132e-03 | +4.71% | 10.082 | 1.575e-05 | 1.720e-04 |
| T4, FP32 | 1.041813e-03 | +4.18% | 10.082 | 1.650e-05 | 1.720e-04 |

That is the whole return on writing the numerical core as `LBM_HD`: the kernels
and the reference driver are not similar, they are the same arithmetic, and FP32
costs half a percentage point on this measurement.

**The spurious current is the exception, and it is the diagnostic rather than
the port.** It is a pointwise *maximum* of a quantity near 1.6e-05, and running
the device out to 24000 steps shows it wandering — 1.575e-05, 1.617e-05,
1.610e-05, 1.609e-05, 1.611e-05 — while sigma sits at +4.71% unchanged to three
decimals from step 3000 onward. A 5% gap between host and device on a fluctuating
max, with every converged quantity matching to seven digits, is that fluctuation
and not a disagreement about the answer.

**Laplace's law at the resolution the parent uses**, 64³ with R = 16 and W = 4,
window 3W, nu = 0.1 in both phases, run to 60000 steps (FP64 row to 20000):

| gamma | precision | sigma measured | asked | error | spurious current |
|---|---|---|---|---|---|
| 1 | FP32 | 1.009376e-03 | 1e-03 | **+0.94%** | 1.984e-05 |
| 1 | FP64 | 1.009339e-03 | 1e-03 | **+0.93%** | 1.977e-05 |
| 10 | FP32 | 1.006689e-03 | 1e-03 | **+0.67%** | 1.215e-05 |
| 100 | FP32 | 9.663819e-04 | 1e-03 | −3.36% | 3.511e-05 |

The parent measures gamma = 1 within 0.87% and gamma = 10 within 0.77% on the
same model. The port now agrees with it to about a tenth of a point at both
ratios, in both precisions.

**A claim this section used to make, and which the fix removed.** It read: *the
sign of the error is geometry, not precision and not the port* — +2.5% at
R = 10, W = 2 against −2.6% at R = 16, W = 4, with the two constituent errors
"evidently crossing somewhere between those two interface resolutions." There is
no crossing. At the corrected seed and fixed gamma = 10 the error is **+4.71%**
at R = 10, W = 2 and **+0.67%** at R = 16, W = 4: same sign, monotonically
smaller as the interface is better resolved, which is what convergence is
supposed to look like. The sign flip was one configuration being run at twice
the interface width of the other.

### What the wrong seed cost

| gamma | parent | 2x-wide seed | fixed |
|---|---|---|---|
| 1 | 0.87% | −2.48% | **+0.94%** |
| 10 | 0.77% | −3.02% | **+0.67%** |
| 100 | — | −13.61% | **−3.36%** |

It hid because a wrong interface width is still a *consistent* simulation. The
device matched the host to every digit, `host_colour`'s 51 algebraic identities
all held, sigma converged to a steady value and stayed there. Nothing was
inconsistent — only wrong. The one number that could have shown it was the
threefold gap against the parent at gamma = 1, and that gap was written down
here as an open question about the model.

It was found by building the controlled comparison in the next section: at
`-w 4` the two drivers reported effective radii of 16.78 and 16.20 for the same
seeded R, which is a tail-volume signature of two different profiles. Both now
read 16.203. The fix is also self-checking — `droplet -w 2 -window 6` on the old
code and `droplet -w 4 -window 3` on the new one are the same physical
configuration, and they agree to every digit printed: +0.94%, +0.67%, −3.36%,
with spurious currents 1.984e-05, 1.215e-05 and 3.511e-05 in both.

### The 47x, found and fixed

The first version ran at **20.2 MLUPS** at 64³ in FP32 (5000 steps in 64.88 s)
against the single-phase core's 950. A factor of 47 is far more than the extra
arithmetic explains, and `-DLBM_PTXAS_VERBOSE=ON` — which exists for exactly this
suspicion — said why: zero spills, but a **216-byte stack frame**, two 27-element
arrays that did not fit in registers and lived in local memory, read and written
at every node. The fluid kernel has none.

The cause was in `colour.cuh`'s own banner: the equilibrium was built as
POPULATIONS and then transformed, and "a closed form could be derived; until it
is measured to matter, it would be a second thing to keep correct." That was the
measurement, so the closed form was derived — see the banner in `colour.cuh` for
the three pieces and why none of them may be lifted from `core.cuh`'s
`eq_moment()`. One transform in, one out; `ke` and `kp` are never arrays.

| | before | after |
|---|---|---|
| stack frame | 216 bytes | **0 bytes** |
| spill stores / loads | 0 / 0 | 0 / 0 |
| registers | 119 | 124 |
| **throughput, 64³ FP32** | **20.2 MLUPS** | **252.5 MLUPS** |

**12.5x**, and 252.5 against the single-phase core's 950 means the colour
gradient now costs 3.8x a single-phase step — for two distributions, three
kernels and a heavier collision, which is about what the work implies.

**FP64 is unchanged to every digit printed**, on the device and on the host:
sigma 1.025048e-03, +2.50%, R_eff 10.319, spurious current 1.835e-05, mean core
blue 1.684e-04. So this is the same operator, not a cheaper approximation of it.
`test/host_colour.cpp` keeps the old path as `reference_collide()` and asserts
the two agree over 60 states — density ratios 1, 10 and 100, a viscosity ratio,
five colour-gradient orientations including zero, four velocities, a body force
and a non-zero `rho_ref` — to 5.9e-16 in FP64 and 2.1e-07 in FP32, about one ulp.

In FP32 the *results* move within their own noise, which is worth stating
plainly rather than presenting as an improvement. Different arithmetic order,
same operator. (The absolute values quoted for that comparison at the time were
taken at the 2x-wide seed; the closed form and the seed are independent changes,
and the equivalence assertion in `test/host_colour.cpp` does not depend on any
initial condition.)

## The phase field — confirmed on a T4

Clean build first try. `host_phasefield` passes on the card's machine. Laplace's
law at 64³ with R = 16 and W = 4, FP32, run to 40000 steps and converged (the
last two report rows agree to the digit):

| gamma | viscosity | sigma measured | error | spurious current |
|---|---|---|---|---|
| 1 | mu matched | 9.382817e-04 | −6.17% | 1.235e-05 |
| 10 | mu matched | 9.573443e-04 | −4.27% | 1.162e-05 |
| 100 | mu matched | — | **diverged** | — |
| 10 | nu matched | 9.569894e-04 | −4.30% | 2.249e-06 |
| 100 | nu matched | 9.625386e-04 | −3.75% | 5.685e-07 |

**The divergence at a ratio of 100 was the viscosity choice, not the model.**
Matching the DYNAMIC viscosity across a ratio of 100 leaves the heavy phase with
a hundred-fold smaller kinematic viscosity, and
`omega = 1/(mu/(rho cs^2) + 1/2)` is then 1.994 — against a stability limit of 2.
Matching the KINEMATIC viscosity instead (`-muh 5.0`) runs the same case to
−3.75% with a spurious current twenty times smaller. Worth stating as a
parameter trap rather than a capability: nothing in the model objects to a ratio
of 100, and `omega` is what to check first when one of these diverges.

**Throughput is 354.6 MLUPS at 64³ in FP32** — faster than the colour gradient's
252.5, and 37% of the single-phase core's 950, for six passes and two
distributions. `-DLBM_PTXAS_VERBOSE=ON` reports **0 bytes stack frame and no
spills** on every kernel (43, 64 and 72 registers for the phase, fluid and
derivative passes), which is the colour gradient's lesson applied by
construction rather than after a measurement.

### The two engines on the same measurement

Tesla T4, FP32, 64³, R = 16, W = 4, sigma asked 1e−3, **nu = 0.1 in both phases
of both models**, averaging gap 3W in both, 60000 steps.

| gamma | colour gradient | its spurious current | phase field | its spurious current |
|---|---|---|---|---|
| 1 | **+0.94%** | 1.98e−05 | −6.17% | 6.65e−06 |
| 10 | **+0.67%** | 1.22e−05 | −4.32% | 1.11e−06 |
| 100 | −3.36% | 3.51e−05 | **−3.77%** | 2.99e−07 |

The colour gradient is the more accurate of the two on Laplace's law at every
ratio tested, and the phase field carries the smaller spurious current at every
ratio — by a factor of 3 at gamma = 1 and of 118 at gamma = 100, where its
current *falls* with the ratio while the colour gradient's does not. The
accuracy gap closes as the ratio rises: at gamma = 100 the two are within half a
percentage point of each other, and the colour gradient's case there is the one
still creeping at 60000 steps (−2.44% at 30000, −3.36% at 60000), so read it as
near-converged rather than converged. Every other case is flat to the digits
shown from 30000 on.

**Building this table is what found the seed defect above**, and that is the
argument for building it. Both engines had been measured, both agreed with their
host references, and neither had anything visibly wrong. It was putting them
side by side on one case that produced a number which could not be right —
two effective radii for one seeded R — and the first version of the table, run
before the fix, inverted the conclusion: it read −2.50%, −3.54%, −13.61% for the
colour gradient, a model apparently collapsing under density ratio and losing to
the phase field at gamma = 100. Ten percentage points at gamma = 100 were the
seeded interface width.

Two smaller things the sweep settled. The phase field is **window-independent to
five digits** — 2W and 3W give the same answer at every ratio — while the colour
gradient moves four percentage points at gamma = 100 (−13.63% at 3W against
−9.65% at 2W, on the pre-fix seed), exactly where its own banner predicted, with
the mechanism measurable: mean light-phase density in the core is 2.3e−08 at 3W
against 2.8e−06 at 2W, two orders of magnitude more contamination, weighted
gamma-fold because cs_b² = gamma cs_r². And **neither droplet changes size**:
both models conserve their own indicator, so the effective radius is pinned at
its seeded value from step 0 and diagnoses the seeded profile rather than any
relaxation. That is what made it a usable check on the seed.

What is still *not* controlled here: the colour gradient has no mobility and the
phase field has no recolouring, so the mechanism maintaining the interface
differs by construction. That is a difference between the models, not a
parameter left unmatched.

## Verification

`host_check` runs 47 checks with no GPU, in either precision, `host_phasefield`
22 more, and `host_colour` 51 more for the colour-gradient module — in both halves of the word. The
physics half re-derives the model's invariants from the code as written, because
the failure a port invites is silently reverting one of the three readings that
had to be worked out rather than copied from the paper: the per-colour rest term,
the perturbation coefficient A rather than A/2, and the fact that cs² means two
different numbers in the same operator. The integration half runs a real droplet
through all three passes, which is what actually exercises the Esoteric Pull
parity, the neighbour wrap and the gather/scatter pairing — where a port fails if
it fails at all. Both colours are conserved to machine precision over fifty
steps.

`host_check` runs 47 checks with no GPU, in either precision. The original
twenty: the velocity set and its moments, the `opp(i) == i+1` pairing that
Esoteric Pull depends on, the direction table, equilibrium moments, the
central-moment transform round trip, equilibrium in central-moment space, mass
and momentum conservation for both operators, equilibrium as a fixed point,
`gather(init_scatter(x)) == x`, and — the one the scheme lives or dies on — that
a single streaming step transports each of the 26 directions by exactly `c_i`
and nowhere else.

And, for the physics added since: the D3Q7 velocity set and its cs² = 1/4, that
both lattices obey the same pairing contract, the scalar equilibrium's two
moments, the induction equilibrium's first moment **and that it is exactly
antisymmetric** — the property that keeps div B from being generated — the
Maxwell stress carrying no mass and no momentum while delivering exactly M_ab in
the second, its central moments against the closed forms the parent verified
symbolically, the Guo source's three moments, that a forced collision adds
*exactly* F to the momentum under both operators, that the MHD equilibrium is a
fixed point of both, and the streaming round trip again on D3Q7.

**One of the original twenty was asserting the wrong thing, and only FP64 could
tell.** It fed the second-order equilibrium `feq` to *both* operators and asked
that neither move it. That is true of BGK and false of central moments, whose
fixed point is the Maxwellian one — the two differ by the Galilean defects of the
truncation, which are O(u³). At u ≈ 0.02 that is 2.5e-06: comfortably inside the
FP32 tolerance and nowhere near the FP64 one. The operator was never wrong; the
check was, and the looser precision hid it. Both now assert against their own
equilibrium, and both pass in FP32 and FP64.

`host_physics` runs whole simulations against closed-form solutions; see the
table above. It found three faults in its own first draft and none in the solver:
a decay measured for so long that the mode had fallen into round-off, a peak
velocity compared against a continuous maximum that falls between nodes, and a
magnetic conservation test that fed the operator a field which was not the moment
of the populations being collided.

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

**Refresh the coupled fields BEFORE stepping the fluid.** Two-way coupling must
be evaluated simultaneously. Stepping the fluid against a temperature or a
magnetic field from the previous step is a first-order splitting error, and under
diffusive scaling the ratio ω²Δt/(νk²) is independent of N — so it appears as a
damping offset that survives every grid refinement. A non-converging error
sitting beside a converging one is the signature, and `alfven -sweep` is there to
show it. The order is `compute_field()`, then `fluid.step()`, then the coupled
solvers' own `step()`. On the device those are launches on the default stream and
are already ordered; no fence is needed.

It is also the cheaper of the two mirror images. Recovering u for the coupled
fields instead would need a separate pass over 27 populations, where refreshing T
costs 7 and B costs 21 — and the fluid writes u as a by-product of a gather it
was doing anyway.

**Write the driver against `backend::`, not against `Solver`.** The aliases in
`include/lbm/backend.cuh` resolve to the CUDA classes under nvcc and to the host
reference drivers otherwise, so the same source runs both ways. Every case here
was debugged on a laptop before it was ever a kernel launch; the Rayleigh–Bénard
bracket, the Alfvén convergence sweep and the div B history in this README were
all measured that way.

**A gap between the total and the fluid-cell sum is not a leak.** Under Esoteric
Pull a population in flight toward a wall spends a step in a slot the WALL owns,
so a sum over fluid cells always undercounts — 4% in the insulating box here, 13%
on an urban geometry in the parent. `host::Scalar::total_population()` and
`host::Fluid::total_mass()` sum every slot, and those are the conserved
quantities. The macroscopic field is what is missing the difference, not the
solver.

**Watch registers before blaming arithmetic.** Build with
`-DLBM_PTXAS_VERBOSE=ON` and read the spill columns. In the parent
implementation the central-moment operator ran orders of magnitude slower than
BGK on a T4; register spilling was the obvious hypothesis and `-Xptxas -v`
**ruled it out** (zero spill bytes, 116 registers). That question is still open,
and it is the main reason this code exists — so measure before theorising.
