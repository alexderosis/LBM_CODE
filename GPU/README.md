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
of everything else. At H = 12, BGK:

| Ra | Nu | max &#124;u&#124; | verdict |
|---|---|---|---|
| 5000 | 2.043700 | 2.88e-02 | convection sustained |
| 800 | 0.995366 | 4.27e-05 | convection died |

A factor of 670 in the velocity across the threshold. This is the case that
exercises everything added here at once — solid walls for the momentum, Dirichlet
walls for the scalar, and the Boussinesq force that couples them.

**One honest caveat.** Below onset the exact answer is Nu = 1 and the fluid at
rest; the run gives Nu = 0.9954 and a residual max|u| of 4.3e-05. That residual
is *independent of the seeded perturbation* — 4.263e-05 with no seed at all
against 4.271e-05 with a seed a hundred times larger — so it is not a slowly
decaying mode but a steady spurious flow the body force drives near the walls. It
is linear in Ra (2.15, 4.26, 8.50 × 10⁻⁵ at Ra = 400, 800, 1600) and the Nu
offset converges at second order in H, −0.46% at H = 12 against −0.13% at H = 24.
A discretisation artefact at the scheme's design order, then — but quote Nu from
two resolutions, not one.

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
| lattices | D3Q27 fluid, D3Q7 scalar and field | D2Q9, D2Q5, D3Q7, D3Q19, D3Q27 |
| collision | BGK, central moments | BGK, TRT, raw MRT, central moments |
| streaming | Esoteric Pull | Esoteric Pull, two-lattice |
| storage | raw | raw, shifted |
| boundaries | periodic, bounce-back, scalar adiabatic and Dirichlet | + regularised, outflow, moment-based |
| forcing | Guo: uniform and Boussinesq | Guo, high-order Hermite |
| thermal | advection–diffusion + Boussinesq | same |
| MHD | Dellar induction + Maxwell stress, BGK and central moments | + the published D2Q9 scheme |
| geometry | arbitrary voxel input | arbitrary voxel input |
| cases | 6 drivers | ~20 validation cases |

Still absent, not merely untested:

* **shifted storage.** The fluid stores raw populations, and it costs accuracy in
  FP32 — see the Poiseuille row below, where FP32 loses the clean second-order
  convergence FP64 shows at H = 32.
* **the scalar's open boundary.** Outflow needs a donor map and a second kernel
  after a fence; reading a donor inside the main kernel is a genuine race under
  Esoteric Pull, because the two slots a node reads are the two it writes.
* **magnetic wall conditions.** A non-fluid cell is skipped, which on this
  storage means bounce-back on the induction distribution — and that is neither
  the perfectly conducting nor the insulating condition. Dellar's moment-based
  wall is what those need. Every MHD case here is periodic. Do not read a
  wall-bounded MHD result off this code.
* **D3Q19, TRT, raw MRT, regularised walls, the aorta, height-field input.**

## Verification

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
