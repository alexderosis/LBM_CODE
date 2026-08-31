# M3LB

**M**anchester **M**ultiphysics **M**odelling &mdash; **L**attice-**B**oltzmann.

Write it **`M3LB`** everywhere a machine reads it: repository, filenames,
commands, BibTeX keys, DOIs, citations. **M³LB** is the typeset form, for the
document cover, slides and figures. One ASCII spelling means the name stays
searchable and citable.

Modular lattice Boltzmann solver in C++20 on Kokkos.

**Status: Milestone 8.** D2Q9/D3Q19/D3Q27 BGK fluid solver, **Esoteric Pull**
in-place streaming (one population set) cross-checked bit-for-bit against the
two-lattice reference, **shifted populations** for usable FP32, halfway
bounce-back, Guo forcing. Validated against analytic Poiseuille flow to machine
precision and against decaying Taylor-Green / off-axis shear waves at
second-order convergence. Collision operators: **BGK, TRT, raw MRT and central
moments**, the last two sharing one factorised product-basis implementation.
A **thermal module** adds a second distribution set on its own lattice, coupled
to the flow by Boussinesq buoyancy; an **MHD module** adds a vector-valued
magnetic distribution (Dellar) coupled through the Maxwell stress. Targets NVIDIA GPUs and CPU clusters;
Serial + Threads locally.

All three MATLAB generator questions are resolved and the scripts under
`MATLAB/` are corrected (originals kept in `MATLAB/original/`).

## Writing your own case

`examples/flow_past_square.cpp` is a commented template — flow past a square
cylinder in a channel — split into four blocks you change: the discretisation,
the geometry, the initial condition, and the time loop with your diagnostic.
[`GETTING_STARTED.md`](GETTING_STARTED.md#5-writing-your-own-case) walks through
it, including the two things that catch people: the Mach/`tau` trade that decides
what Reynolds number a given grid can reach, and why a *fixed* body force does not
give you the Reynolds number you asked for.

```bash
cmake --build build -j4 --target flow_past_square
./build/examples/flow_past_square --kokkos-num-threads=4
```

## Documentation

[`doc/m3lb.pdf`](doc/m3lb.pdf) is the release document: every scheme with
its equations, the methodology behind each design decision, the complete
validation record with numbers, the performance figures and their caveats, and a
list of known limitations. Rebuild with `make -C doc` (needs a LaTeX install).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_THREADS=ON
cmake --build build -j8
cd build && ctest --output-on-failure
```

Precision is a configure-time switch (`double` is the default):

```bash
cmake -S . -B build-fp32 -DLBM_PRECISION=float ...
```

### GPU

**This code has been run on a GPU, and the result is mixed — read this before
planning around it.** On a Tesla T4 (sm_75, CUDA 12.8, FP32), seven validation
executables build and run after six separate fixes for nvcc restrictions the CPU
backends cannot see (all committed, all commented). BGK and TRT reach ~420–447
MLUPS at 64³ with **mass drift exactly 0.000e+00**.

**But the two central-moment configurations did not finish in seventeen minutes**
at the same size, against ~0.03 s for BGK — four orders of magnitude, and
`-Xptxas -v` rules out register spilling (zero spill bytes, 116 registers). That
matters because the aorta and the whole validation campaign run central moments,
so the 447 MLUPS is a **BGK-only** figure and must not be quoted as this code's
GPU throughput. The next step is measurement (`ncu --set full`), not another
hypothesis.

Every number in `doc/` and elsewhere in this README is the Threads backend.

**`GPU/` is a second, independent implementation** written directly in CUDA to
find out whether that collapse was the algorithm or the port. It is not: the same
27-moment scheme runs at **99.6% of BGK speed** there, 17.2 GLUPS on an H200.
See [`GPU/README.md`](GPU/README.md), and [`GETTING_STARTED.md`](GETTING_STARTED.md)
if you are new here.

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
      -DLBM_GPU_BACKEND=cuda -DLBM_GPU_ARCH=ADA89 -DLBM_PRECISION=float
cmake --build build-gpu -j8
cd build-gpu && ctest --output-on-failure
```

`LBM_GPU_BACKEND` is one of `none|cuda|hip|sycl`; `LBM_GPU_ARCH` is the Kokkos
arch macro without its `Kokkos_ARCH_` prefix (`ADA89`, `AMPERE86`, `VEGA90A`,
…). Both only apply when Kokkos is built here; against an installed Kokkos that
installation's backends win. You can still configure Kokkos by hand instead.

Four things worth knowing before starting:

* **Use FP32.** Consumer NVIDIA cards run FP64 at 1/32 to 1/64 of their FP32
  rate, so a default `double` build on a laptop GPU is routinely *slower than
  the CPU*. The FP32 caveat in Known Limitations then applies — it cannot
  resolve the finest convergence tests.
* **Run `ctest` on the device first.** If `poiseuille`, `decaying_flows` and
  `galilean` pass, the port is essentially sound. Debugging a 3D turbulence case
  first is miserable.
* **Compare to tolerance, not bitwise.** CPU results are bitwise reproducible
  across thread counts, so they make an exact reference — but GPU reductions sum
  in a different order and will not match to the last bit.
* **Single device only.** There is no MPI (see Known Limitations), so the
  problem has to fit in one GPU's memory. D3Q27 in FP32 is ~108 bytes/node:
  128³ is 226 MB, 192³ is 764 MB, 256³ is 1.8 GB, before the macroscopic fields.

There is no Metal backend in Kokkos, so an Apple GPU is not a target.

`colab_gpu_test.ipynb` in the repository root does all of the above on a free
Colab GPU: detects the device and maps its compute capability to a
`Kokkos_ARCH_*` macro, installs Kokkos (optionally cached to Drive so a
disconnect does not cost the whole build), builds FP32, runs `ctest` on the
device, then runs 3D Taylor-Green on both backends and compares the energy and
enstrophy decay. Roughly 25-40 minutes, nearly all of it compiling.

## Layout

| path | contents |
|---|---|
| `src/lattice/`     | D2Q5, D2Q9, D3Q7, D3Q19, D3Q27 descriptors; lattice identities checked at compile time |
| `src/core/`        | scalar types, Kokkos aliases, layout policy |
| `src/grid/`        | `Domain`: padded Cartesian block with halo |
| `src/memory/`      | streaming schemes (`TwoLattice`, `EsotericPull`) and the raw/shifted storage tags |
| `src/equilibrium/` | second-order and product-form equilibria |
| `src/collision/`   | `BGK`, `TRT`, `MomentCollision` (raw MRT / central moments), `ScalarBGK`, `MagneticBGK`, `MhdBGK`, `MhdCentralMoments` |
| `src/forcing/`     | `NoForcing`, `Guo`, `BoussinesqGuo` |
| `src/boundary/`    | cell flags; bounce-back lives in the fused kernel |
| `src/solver/`      | `FluidSolver`, `ScalarSolver`, `MagneticSolver` |
| `src/io/`          | `.vti` writer |
| `tests/`           | unit tests |
| `validation/`      | Poiseuille (walls); Taylor-Green, ABC/Beltrami, shear wave (bulk); Galilean invariance |
| `MATLAB/`          | symbolic central-moment generators + `esopull_ordering.m` |

## Design rules

1. **No virtual functions in device code.** Lattice, streaming, equilibrium,
   forcing and collision are compile-time policies that inline into one kernel.
   Dynamic dispatch is allowed only in the host-side factory, above `FluidSolver`.
2. **Never pin the Kokkos layout.** Population fields are `View<Real**>` with
   extents `(node, q)` and the *default* layout, which is SoA on GPU and AoS on
   CPU. See `src/core/Types.hpp`.
3. **Direction-ordering contract.** Rest at index 0, opposite directions in
   adjacent pairs, so `opp(i) = i+1` (odd `i`) / `i-1` (even `i`). Esoteric Pull
   requires it; every moment matrix is generated against it. `Lattices.hpp`
   static_asserts it.
4. **Streaming and collision are one fused kernel.** In-place schemes cannot
   have a separate streaming pass, so nothing is written that would assume one.
5. **Parity is a compile-time parameter, never a runtime branch.** Esoteric Pull
   selects different storage slots on even and odd steps; `FluidSolver::step`
   dispatches to `run_step<0>` / `run_step<1>` so that decision folds away.
6. **Periodicity is index wrapping, not halo copying.** Esoteric Pull writes into
   its neighbours' storage, so a halo-copy scheme would need a two-way exchange
   whose slot set depends on parity. A halo is allocated only in non-periodic
   directions; MPI will reintroduce one on partitioned directions.
7. **Keep the reference implementation.** `TwoLattice` stays in the tree
   permanently as the thing to bisect against, and the test suite cross-checks
   Esoteric Pull against it after every step.

## Validation

`validation/poiseuille` runs force-driven plane Poiseuille flow. With halfway
bounce-back the effective wall position is viscosity-dependent for BGK, and the
slip vanishes exactly at the magic parameter

    Lambda = (1/omega - 1/2)^2 = 3/16   ->   tau = 0.5 + sqrt(3)/4 = 0.9330127

so at that tau the discrete solution is exact to round-off. Measured (FP64,
Esoteric Pull + shifted):

| tau | H | L2 rel err | wall slip |
|---|---|---|---|
| 0.6000000 | 32 | 1.27e-03 | -7.40e-03 |
| 0.8000000 | 32 | 6.95e-04 | -4.06e-03 |
| 0.9330127 | 16 | 3.54e-14 | +1.93e-14 |
| 0.9330127 | 32 | 1.63e-14 | +3.35e-14 |
| 0.9330127 | 64 | 6.52e-13 | +4.66e-13 |
| 1.2000000 | 32 | 2.16e-03 | +1.26e-02 |

The slip changes sign across the magic tau, as it must.

All four streaming/storage combinations are then run at the magic tau and must
agree; the two streaming schemes share no storage code, so they are required to
match bit for bit, not approximately:

| streaming | storage | L2 (FP64) | L2 (FP32) |
|---|---|---|---|
| TwoLattice   | raw     | 1.433e-13 | 5.969e-04 |
| TwoLattice   | shifted | 1.626e-14 | 2.778e-05 |
| EsotericPull | raw     | 1.433e-13 | 5.969e-04 |
| EsotericPull | shifted | 1.626e-14 | 2.778e-05 |

The shift buys **21.5x** in FP32 and 8.8x in FP64, and the acceptance test
asserts that gain rather than merely tolerating its absence.

## Collision operators

| operator | header | notes |
|---|---|---|
| `BGK`             | [BGK.hpp](src/collision/BGK.hpp) | single relaxation time |
| `TRT`             | [TRT.hpp](src/collision/TRT.hpp) | symmetric/antisymmetric split, magic parameter |
| `MRT` / `CentralMoments` | [MomentCollision.hpp](src/collision/MomentCollision.hpp) | one implementation, `Central` toggle; D2Q9, D3Q19, D3Q27 |

`MRT` and `CentralMoments` differ only in the velocity the moment basis is built
at -- `u_b = 0` for raw moments, `u_b = u` for central -- which is exactly the
`central_moments` toggle in `MATLAB/*_CM.m`.

**No Q x Q matrices anywhere.** The product basis `{1, C, C^2 - cs2}` factorises
into D successive 1D passes (`ProductBasis`), which is 4.5x cheaper than a dense
27x27 contraction and was checked against it in exact rational arithmetic. The
same basis diagonalises the Maxwellian: the equilibrium is `rho` in the zeroth
moment and **exactly zero in all 26 others**, so no equilibrium populations are
ever formed and no `K_eq` vector is stored. Shifted storage is handled by
subtracting the moments of `w`, which factorise too.

**D3Q19 is not a product lattice** (it is D3Q27 minus its eight corners), so it
uses a generated 19-monomial basis instead --
[MonomialBasis.hpp](src/collision/MonomialBasis.hpp). `SelectBasis` picks the
right one per lattice and the operator itself is basis-generic. Verified in
exact rational arithmetic before the header was generated: M is invertible on
D3Q19, the monomial set is downward closed so the binomial central-moment shift
needs nothing outside the basis, the shift matches direct central summation and
round-trips exactly, and -- the useful surprise -- the equilibrium raw moments
come out **exactly product-form**, `m_eq = rho P(p,ux)P(q,uy)P(r,uz)` with
`P = {1, u, cs2+u^2}`, so the equilibrium central moments are the Maxwellian ones
on all 19 representable monomials, the same structure as D3Q27. (D3Q19's known
equilibrium defects live at monomials such as (300) and (111), which the lattice
cannot represent at all and which are not in the basis.)

Also settled: `D3Q19_CM.m`'s **ortho and non-ortho branches define the identical
operator**. Both relax positions 6..10 at omega and everything else at 1; in
either basis those five span the deviatoric second-order subspace and position 5
spans the trace, and sending independent combinations to equilibrium is the same
as sending the subspace to equilibrium. There is no choice to make.

### The magic parameter, and what each operator fixes

`Lambda = (1/w_even - 1/2)(1/w_odd - 1/2) = 3/16` removes the halfway
bounce-back wall slip. Which rates play those roles depends on the operator, so
each has its own magic point -- and TRT can hit it at *every* viscosity, which
is the whole reason it exists. Measured wall slip (D2Q9, H = 32):

| tau | BGK | TRT(3/16) | MRT | CentralMoments |
|---|---|---|---|---|
| 0.6000000 | -7.40e-03 | +2.51e-14 | -5.73e-03 | -5.73e-03 |
| 0.8000000 | -4.06e-03 | +7.53e-14 | -1.56e-03 | -1.56e-03 |
| **0.8750000** | -1.95e-03 | -1.92e-14 | **+4.11e-14** | **+7.39e-14** |
| **0.9330127** | **-1.08e-14** | +5.35e-14 | +1.21e-03 | +1.21e-03 |
| 1.2000000 | +1.26e-02 | +3.55e-15 | +6.77e-03 | +6.77e-03 |

The moment operators relax third-order modes at 1, so their magic point is
`(1/tau - 1/2)(1/2) = 3/16`, i.e. `tau = 7/8` exactly -- predicted before the
run and confirmed to 7e-14.

### Resolution behaviour at the wall

At its magic point an operator is exact at *every* resolution, so there is no
order to fit -- the assertion is that the error stays at round-off as the grid
refines. Away from it (FP64, D2Q9):

| operator @ tau | L2 H=16 | L2 H=32 | L2 H=64 | ord(L2) | ord(slip) |
|---|---|---|---|---|---|
| BGK @ 0.9330127 | 4.2e-14 | 1.2e-14 | 5.8e-13 | exact | exact |
| TRT(3/16) @ 0.6 | 4.6e-14 | 1.1e-13 | 1.6e-13 | exact | exact |
| MRT @ 0.875 | 3.2e-14 | 9.3e-14 | 3.1e-13 | exact | exact |
| CentralMoments @ 0.875 | 3.6e-14 | 8.4e-14 | 3.2e-13 | exact | exact |
| BGK @ 0.6 (off-magic) | 5.1e-03 | 1.3e-03 | 3.2e-04 | **2.00** | **1.00** |

Off-magic bounce-back is still **second** order -- a lattice-aligned wall does
not degrade the scheme. The pair of orders is what identifies the defect: the
discrete error is a constant velocity offset `C ~ (Lambda - 3/16) G / nu`, which
at fixed peak velocity scales as `1/H^2`, while the *fitted wall position* moves
only as `1/H` because the parabola's curvature shrinks with it. Seeing 2 and 1
together says "offset", not "displaced wall".

These two assertions are FP64-only. Refining to H=64 drives the discretisation
error to 3e-4, which in FP32 sits below the accumulated round-off floor of about
1e-3; loosening the tolerance would not fix that, it would only stop the test
measuring anything.

### Galilean invariance

The test central moments exist for. Superimposing a uniform velocity `U0` only
translates the solution, so the recovered viscosity must not depend on it.
ABC/Beltrami flow, D3Q27, N = 32, tau = 0.8:

| lattice | operator | U0=0 | U0=0.05 | U0=0.10 | U0=0.15 | drift | vs BGK |
|---|---|---|---|---|---|---|---|
| D3Q27 | BGK             | 0.100620 | 0.099864 | 0.097595 | 0.093812 | 6.77e-02 | - |
| D3Q27 | TRT(3/16)       | 0.100368 | 0.099618 | 0.097368 | 0.093617 | 6.73e-02 | 1x |
| D3Q27 | MRT (raw)       | 0.100484 | 0.100485 | 0.100486 | 0.100490 | 5.87e-05 | **1153x** |
| D3Q27 | CentralMoments  | 0.100484 | 0.100484 | 0.100482 | 0.100479 | 4.70e-05 | **1439x** |
| D3Q19 | BGK             | 0.100620 | 0.099864 | 0.097595 | 0.093812 | 6.77e-02 | 1x |
| D3Q19 | MRT (raw)       | 0.100484 | 0.100484 | 0.100486 | 0.100489 | 5.33e-05 | **1268x** |
| D3Q19 | CentralMoments  | 0.100484 | 0.100484 | 0.100482 | 0.100479 | 5.24e-05 | **1291x** |

The gain is essentially lattice-independent: D3Q19 recovers 1291x against
D3Q27's 1439x. Whatever central moments are buying here, it is not something
that needs the extra eight velocities.

Two things worth reading off this table:

- **TRT is no better than BGK here.** The magic parameter fixes where the *wall*
  sits, not which frame the collision prefers. These are independent defects and
  it takes both operators to address both.
- **Most of the gain is the complete equilibrium, not the co-moving frame.** Raw
  MRT already recovers 1153x because this implementation gives it the full
  product-form equilibrium; moving the basis to `u` adds a further 25%. Worth
  knowing before attributing everything to centring.

Every operator's residual scales as `U0^2` (fitted powers 2.00-2.10), i.e. what
is left is the weakly-compressible `O(Ma^2)` error of the method itself, not a
frame dependence of the collision.

## Bulk validation

`validation/decaying_flows` runs three periodic flows with exact Navier-Stokes
solutions. Being periodic, they isolate the bulk collision operator from the
wall treatment that `poiseuille` covers.

| case | character | what it is for |
|---|---|---|
| **Taylor-Green** | nonlinear, z-independent | cross-lattice consistency |
| **ABC / Beltrami** | nonlinear, genuinely 3D | 3D convergence |
| **Diagonal shear wave** | linear, off-axis | lattice isotropy |

Taylor-Green's field does not depend on z, so it is the *same 2D problem*
whichever lattice runs it -- useful as a consistency check but not a 3D test.
The ABC flow `u = U(sin kz + cos ky, sin kx + cos kz, sin ky + cos kx)` is the 3D
one: `curl u = k u`, so `u x (curl u) = 0` and the nonlinear term collapses to
`grad(|u|^2/2)`, absorbed by the pressure, giving `u(t) = u_0 exp(-nu k^2 t)`
exactly. Every component varies in every direction.

The effective viscosity does **not** equal `nu` at finite resolution: a sinusoid
decays at the rate set by the discrete Laplacian, which differs at O(k^2). What
is asserted is that the error converges away at second order. Diffusive scaling
`U ~ 1/N` keeps the Mach-number error second order too.

Orders are fitted for **every collision operator**, not just BGK -- a moment
operator can be conservative, pass every algebraic check and be exact at its
magic point while still carrying a degraded order in a mode a single-resolution
test never excites. 11 studies x 2 quantities x 2 resolution pairs = 44 fitted
orders, all 2.00 +- 0.11 (FP64):

| case | lattice | operator | ladder | nu err | L2(u) |
|---|---|---|---|---|---|
| Taylor-Green | D2Q9 | BGK | 16/32/64 | 2.02, 2.02 | 2.00, 2.00 |
| Taylor-Green | D2Q9 | TRT | 16/32/64 | 2.02, 2.02 | 1.99, 2.00 |
| Taylor-Green | D2Q9 | MRT | 16/32/64 | 2.02, 2.02 | 2.00, 2.00 |
| Taylor-Green | D2Q9 | CM  | 16/32/64 | 2.02, 2.02 | 2.00, 2.00 |
| ABC/Beltrami | D3Q27 | BGK | 16/32/48 | 2.01, 2.01 | 2.00, 2.00 |
| ABC/Beltrami | D3Q27 | TRT | 16/32/48 | 2.01, 2.01 | 2.00, 2.04 |
| ABC/Beltrami | D3Q27 | MRT | 16/32/48 | 2.01, 2.01 | 2.00, 2.02 |
| ABC/Beltrami | D3Q27 | CM  | 16/32/48 | 2.01, 2.01 | 2.00, 2.02 |
| ABC/Beltrami | D3Q19 | BGK | 16/32/48 | 2.01, 2.01 | 2.00, 2.01 |
| ABC/Beltrami | D3Q19 | MRT | 16/32/48 | 2.01, 2.01 | 2.00, 2.02 |
| ABC/Beltrami | D3Q19 | CM  | 16/32/48 | 2.01, 2.01 | 2.00, 2.02 |
| shear wave | D3Q19 | BGK | 16/32/64 | 2.09, 2.01 | 1.97, 2.00 |
| shear wave | D3Q19 | CM  | 16/32/64 | 2.10, 2.00 | 1.98, 1.99 |
| shear wave | D3Q27 | BGK | 16/32/64 | 2.10, 2.01 | 1.98, 2.00 |
| shear wave | D3Q27 | CM  | 16/32/64 | 2.11, 2.00 | 1.99, 2.00 |

All four operators are second order; what differs is the error *amplitude*. On
the ABC flow at N=48 the nu error runs TRT 1.63e-3 < MRT 2.15e-3 ~ CM 2.15e-3 <
BGK 2.75e-3 -- so in a periodic bulk flow TRT with Lambda = 3/16 is the most
accurate of the four, because the magic parameter tunes the third-order modes
whereas the moment operators send them straight to equilibrium. On the off-axis
shear wave the ordering flips and CM leads. Viscosity error there at N=64,
relative to D3Q19+BGK:

| D3Q19 BGK | D3Q19 CM | D3Q27 BGK | D3Q27 CM |
|---|---|---|---|
| 1.00x | **1.52x** | 1.12x | 1.38x |

**D3Q19 with central moments is the most isotropic combination of the four** --
better than D3Q27 with central moments, and 1.7x faster. That inverts the usual
assumption that more velocities means more isotropy: D3Q27's central-moment
operator has 20 higher moments forced to equilibrium at rate 1 against D3Q19's 9,
and each of those carries its own error. Measured on one case (diagonal shear
wave, tau = 0.8) and not to be over-generalised, but it does mean D3Q19 + CM
deserves to be on the table rather than dismissed.

Neither operator dominates everywhere; which one wins depends on the flow.

Plus: D2Q9/D3Q19/D3Q27 agree on the same 2D field to 3.9e-14, and all four
streaming/storage combinations agree to 1.8e-14 on the ABC flow.

## Thermal module

A passive scalar on its own lattice — [ScalarSolver.hpp](src/solver/ScalarSolver.hpp)
and [ScalarBGK.hpp](src/collision/ScalarBGK.hpp) — coupled to the flow both ways:
buoyancy from the scalar into the fluid, advection from the fluid into the scalar.

This was the first real test of the module composition, and it passed: D3Q7 has
seven populations to the fluid's nineteen, a different speed of sound (1/4 against
1/3), a different collision concept (velocity is an *input*, not recovered from
its own populations) and different boundary conditions. Nothing in `FluidSolver`,
the lattices, the streaming schemes or the moment operators had to change. The one
interface addition was a node index on the collision, so a body force can vary in
space — buoyancy needs it now, the Lorentz force will need it for MHD.

Boundary conditions are again alternative collisions on marked cells, but note the
asymmetry with the fluid: bounce-back (adiabatic) is the identity on Esoteric
Pull's storage so those cells are skipped, while anti-bounce-back (Dirichlet)
flips a sign and adds a source, so those cells are processed every step. Both
still write back into the slots they read.

**Scalar transport**, against exact solutions of the advection-diffusion equation.
Orders at N = 16 -> 32 -> 64, diffusive scaling in both the amplitude and the
Peclet number:

| case | order (FP64) | order (FP32) | D_eff rel err | phase err |
|---|---|---|---|---|
| D2Q5 diffusion | 2.00 | 2.03 | 1.5e-03 | 4.5e-14 |
| D3Q7 diffusion | 2.00 | 2.00 | 1.3e-03 | 2.8e-14 |
| D2Q5 advection-diffusion | 2.00 | 2.02 | 1.1e-03 | 2.5e-04 |
| D3Q7 advection-diffusion | 1.99 | 2.00 | 7.2e-04 | 2.3e-04 |

The phase is measured as well as the amplitude: an error in the coupling term
puts the wave in the wrong place, which an amplitude check alone would miss.

**Natural convection** in a differentially heated square cavity, against
de Vahl Davis (1983), Pr = 0.71:

| Ra | N | operator | Nu | benchmark | deviation | u_max | reference |
|---|---|---|---|---|---|---|---|
| 1e3 | 64 | BGK | 1.1176 | 1.118 | −0.04% | 3.65 | 3.649 |
| 1e3 | 64 | TRT | 1.1179 | 1.118 | −0.01% | 3.65 | |
| 1e4 | 64 | BGK | 2.2431 | 2.243 | +0.00% | 16.14 | 16.178 |
| 1e4 | 64 | TRT | 2.2479 | 2.243 | +0.22% | 16.14 | |
| 1e5 | 96 | BGK | 4.5169 | 4.519 | −0.05% | 34.47 | 34.730 |
| 1e5 | 96 | TRT | 4.5323 | 4.519 | +0.29% | 34.52 | |

Within 0.3% everywhere. Worth noting that BGK edges TRT here, against the
expectation that TRT's magic parameter would help — the wall placement it fixes
is a momentum boundary, while what limits this case is the thermal boundary
layer, resolved by the scalar lattice.

## MHD module

The magnetic field is carried by a **vector-valued distribution** (Dellar 2002) --
[MagneticBGK.hpp](src/collision/MagneticBGK.hpp),
[MagneticSolver.hpp](src/solver/MagneticSolver.hpp) -- with `B_a = sum_i g_i^a` and
an equilibrium whose first moment is the antisymmetric induction flux
`u_b B_a - B_b u_a`. Only that moment is needed, so the magnetic field runs on
D2Q5 while the flow runs on D2Q9, on the same Domain, at the same time.

The Lorentz force is *not* applied as a body force. Instead the fluid equilibrium
is given the right second moment directly ([MhdBGK.hpp](src/collision/MhdBGK.hpp)),
which avoids taking derivatives of B and losing an order.

### Validation

Two exact solutions before the vortex:

| case | order | recovered | error |
|---|---|---|---|
| resistive decay, L2 | 2.00 | eta = 0.050110 | 2.2e-03 |
| shear Alfven wave, phase speed | 2.00 | v_A = 0.050017 | 3.4e-04 |
| shear Alfven wave, damping | - | (nu+eta)/2 | 5.3e-03 |

The shear Alfven wave is an exact solution of the **full nonlinear** equations,
not just the linearised ones: with u perpendicular to B0 and everything depending
only on x, `(u.grad)u` vanishes identically while `(B.grad)B` does not, so the
Lorentz coupling and the induction equation are both driven and both have to be
right. Phase is measured as well as amplitude, because an error in the coupling
shows up as the wrong wave speed rather than the wrong amplitude.

### Orszag-Tang vortex

`u = u0(-sin 2pi y/L, sin 2pi x/L)`, `B = B0(-sin 2pi y/L, sin 4pi x/L)`,
Re = Rm = 100, to `t* = t u0/L = 0.5`:

| N | div B (final) | div B (max) | E_mag/E_kin | E_tot loss | steps |
|---|---|---|---|---|---|
| 32 | 2.98e-02 | 9.34e-02 | 1.5618 | 0.6239 | 640 |
| 64 | 9.08e-03 | 2.98e-02 | 1.5727 | 0.6227 | 2560 |

There is no closed-form solution, so this is not validated against a formula.
What it tests is what a formula cannot: divergence preservation through a long
nonlinear run, the energy budget, and behaviour under refinement.

**div B is not preserved to machine precision.** Measured against the field's own
gradient scale `k|B|` it sits at truncation level and converges away at order
1.65. Stating that plainly matters, because the two wave cases above *do* report
round-off — and that is meaningless there: in both, `B_x` depends only on `y` and
`B_y` only on `x`, so their divergence is structurally zero whatever the scheme
does. Orszag-Tang is the only case here that exercises the property at all.

### Against De Rosis, Leveque & Chahine (2018), Table 1

Orszag-Tang at Re ~ 628, Pr_m = 1, on the paper's own setup: L = 2 pi m,
u0 = b0 = 2, rho = 1, N = 1024, dt = 5e-5 s. The derived lattice parameters
reproduce the paper's quoted values exactly (u0_lat = 1.6297e-2, Ma = 0.0282),
and Re is the paper's lattice definition u0 N / nu. Reference column [13] is the
high-resolution pseudo-spectral data quoted there.

| | t (s) | spectral [13] | paper LB | here, BGK | here, hybrid CM |
|---|---|---|---|---|---|
| j_max | 0.5 | 18.24 | 18.24 | 18.23 | 18.23 |
| j_max | 1.0 | 46.59 | 46.65 | 46.55 | 46.55 |
| zeta_max | 0.5 | 6.758 | 6.756 | 6.756 | **6.756** |
| zeta_max | 1.0 | 14.20 | 14.18 | 14.182 | **14.180** |

Against the spectral reference the errors are 0.07 / 0.09 / 0.03 / 0.14 %, where
the paper's own are 0 / 0.13 / 0.03 / 0.14 % -- the same accuracy from an
independent implementation, and the hybrid scheme lands on the paper's published
vorticity values to every quoted digit.

The one residual is j_max at t = 1, 0.22% below the paper. That is the quantity
to expect it in: the paper computes the current locally from the distributions
while this uses second-order central differences, and j is the more
gradient-sensitive of the two peaks. Vorticity agreeing exactly while the current
is 0.2% low points at the derivative operator rather than the scheme.

Convergence toward the reference is second order: j_max at t = 1 goes -1.92%
(N=256), -0.46% (N=512), -0.09% (N=1024).

Both collision operators are available for this case:
`orszag_tang -op bgk` runs the Dellar baseline (the paper's ref [13]) and
`-op cm` runs the paper's hybrid central-moment scheme,
[MhdCentralMoments.hpp](src/collision/MhdCentralMoments.hpp), Equations (7)-(13):
the non-orthogonal basis of Eq. (8), the equilibria of Eq. (11),
`omega_4 = omega_5 = omega_v`, `omega_3` for bulk, `omega_6 = omega_7 = omega_8 = 1`,
with BGK for the magnetic field.

Eq. (11) was not taken on trust. `test_moments` builds the paper's Eq. (1)
equilibrium, contracts its central moments directly, and compares: all nine agree
to 1.1e-16, which also confirms the sign convention on b. Note those equilibria
are the central moments of the SECOND-ORDER truncated equilibrium, so their
hydrodynamic parts are deliberately non-Maxwellian -- `-rho ux^2 uy` in k6 and
`3 rho ux^2 uy^2` in k8 are the Galilean defects of the truncation, and
reproducing the paper means reproducing them rather than improving them away.

At Re = 628 the two operators agree to about 0.1% at every resolution, which is
the expected result rather than a null one: the paper's case for the hybrid
scheme is stability at high Re (its Figure 2), not accuracy at a Reynolds number
where its Figure 1 shows BGK is comfortably stable.

### Quasi-2D reduction: 3D lattices with one cell in z

Orszag-Tang is a 2D problem, so running it on a 3D lattice with `nz = 1` is a
reduction test. It needs no code change: with periodic z and a single cell the
neighbour wrap sends the out-of-plane neighbour back to the node itself, and
Esoteric Pull's pair swap handles that correctly.

**D3Q27 reduces to D2Q9 exactly.** Two conditions have to hold and both do, in
exact arithmetic: the D3Q27 weights sum over z onto the D2Q9 weights term for
term (8/27 + 2*2/27 = 4/9, and so on), and the out-of-plane Maxwell-stress
contribution `sum_z w_i (c_z^2 - cs2)` cancels within every in-plane group.

**D3Q19 does not.** Its weights project correctly but the Maxwell sum leaves a
residual (-1/27 on the rest group, +1/54 on the axes, -1/108 on the diagonals).
That residual has zero 0th, 1st *and* 2nd moments, so it excites only ghost modes
and leaves the hydrodynamics untouched -- which is why the difference is a few
tenths of a percent rather than an error.

The lattice-to-lattice spread converges away at second order (0.13/0.26/0.42
percentage points at N=256 becoming 0.03/0.07/0.10 at N=512), so all three
converge to the same answer. The magnetic lattice matters more than the fluid
one: swapping D2Q5 for D3Q7 moves j_max by ~0.25 points, unsurprising given
cs2 = 1/3 against 1/4. Select with `orszag_tang -lat d3q19|d3q27 [-maglat d2q5]`.

### A coupling error that does not refine away

The first version stepped the fluid before refreshing B, so the fluid collided
against `B(t-1)`. That is a first-order splitting error, and under diffusive
scaling the ratio `(omega^2 dt) / (nu k^2)` is **independent of N** -- so it
appears as a damping offset that survives every refinement. It cost 3% of the
Alfven damping rate and showed up as a beat in the amplitude history, with the
error *growing* from 1.55e-2 to 3.16e-2 as the grid refined. Refreshing B first
removed both. `MagneticSolver::step` takes a `field_is_current` flag so a coupled
driver does not pay for the field pass twice.

## Performance

Apple M1, Threads backend, D3Q19/D3Q27, 96^3. Run-to-run spread under 2%.

Streaming and storage (BGK, D3Q19):

| streaming | storage | B/node FP64 | MLUPS FP64 | MLUPS FP32 |
|---|---|---|---|---|
| TwoLattice   | raw     | 304 | 20.1 | 24.2 |
| EsotericPull | raw     | 152 | **29.7** | **35.5** |
| EsotericPull | shifted | 152 | 29.5 | 35.1 |

Collision operator (Esoteric Pull + shifted, FP64):

| lattice | BGK | TRT | MRT | CentralMoments |
|---|---|---|---|---|
| D3Q19 | 29.9 | 15.7 | **12.3** | **9.3** |
| D3Q27 | 14.2 | 11.4 | 5.9 | 5.9 |

Central moments cost 3.2x BGK on D3Q19 and 2.4x on D3Q27, in line with the extra
arithmetic. **D3Q19 + CM is 1.6x faster than D3Q27 + CM** -- its transform is
sparser (M has 127/361 nonzeros, M^-1 91/361, plus 72 shift terms, about 290
operations against D3Q27's 648) and it moves 30% fewer bytes. Combined with the
isotropy result above, D3Q19 + CM is the combination to try first for 3D. **Treat these ratios with care**: this machine is nowhere near
memory-bound at these rates (~1.4 GB/s against ~68 GB/s available), so the
collision cost shows in full. On a GPU, where the kernel is bandwidth-bound,
most of it should hide behind the memory traffic. Re-run `lbm_app` on the target
hardware before choosing an operator on cost grounds.

Three optimisations found through this harness, all measured rather than assumed:

- Restructuring the streaming hot loop to work on opposite **pairs** rather than
  per index removed a runtime `i & 1` test: D3Q27 Esoteric Pull 13.5 -> 14.0 MLUPS,
  putting it ahead of two-lattice.
- `ProductBasis::pi()` was calling a `constexpr` linear search over the velocity
  set with *loop-variable* arguments, so it ran live: 2*Q*Q comparisons per node.
  Replacing it with a precomputed table took the moment operators from 1.8 to
  6.6 MLUPS -- a **3.7x** defect, not a design cost.
- Hoisting the equilibrium evaluation out of TRT's pair loop (which does not
  vectorise) gained 37% on D3Q27 and 45% on D3Q19.

- The same class of defect reappeared in the basis-generic refactor: `p_of(n)`
  computed as `n / 9`, `(n / 3) % 3` costs three integer divisions per moment with
  a runtime index. Table lookups recovered 9%.

Compacting the neighbour list to the odd half was also tried, made no measurable
difference, and was reverted. Making the operator basis-generic (so one
implementation serves the product basis and D3Q19's monomial basis) costs about
7% on D3Q27 against the previous hardcoded version -- a fair trade for not
maintaining two copies of the relaxation.

## Roadmap

- [x] **0** build system, lattices, precision switch
- [x] **1** D2Q9 BGK, two-lattice streaming, bounce-back, Guo forcing, Poiseuille
- [x] **2** Esoteric Pull, verified against `TwoLattice`
- [x] **5a** shifted populations for FP32 (pulled forward)
- [x] **3** D3Q19/D3Q27 validation, Taylor-Green, benchmark harness
- [x] **4** TRT -> MRT -> central moments (from `MATLAB/`)
- [ ] **5b** performance: boundary index lists, occupancy tuning
- [ ] **6** MPI
- [x] **7** thermal module (D2Q5/D3Q7 + Boussinesq)
- [x] **8** MHD (Dellar vector distribution, Orszag-Tang)

## MATLAB generators

`MATLAB/` holds the symbolic central-moment generators. All three questions they
raised are resolved and the scripts corrected in place; `MATLAB/original/` keeps
them as they were.

1. **D3Q19 defined the equilibrium twice.** The `if/elseif` block was silently
   overwritten by a later `nnz` chain. The `nnz` chain is removed and the first
   block kept: comparing equilibrium central moments against the continuous
   Maxwellian, it deviates in 10 of 35 moments (to 4th order) versus 25 for the
   one that was winning.
2. **D3Q27 dropped `(1-omega) k_pre` on three second-order moments.** `L` relaxes
   1-based positions 5..10 but `K_pre` was assigned only at 1, 8, 9, 10, so the
   three normal second-order central moments were driven to equilibrium at rate 1
   while the shears relaxed at omega -- direction-dependent viscosity. Confirmed
   as a bug and fixed for both basis branches.
3. **Basis choice**: `ortho` is the production basis for D3Q27.

All three scripts now use the Esoteric Pull direction ordering, so their output
drops straight into `src/lattice/Lattices.hpp`'s convention. `esopull_ordering.m`
carries the permutation vectors for converting previously generated output.

Bulk viscosity is now an explicit `omega_bulk` on `MomentCollision`, defaulting
to `omega` (D3Q27's ortho convention); set it to 1 for the D2Q9/D3Q19 convention.
It is no longer an accident of which basis toggle is set.
