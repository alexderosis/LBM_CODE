# Demonstrator cases

A **validation** case has something to be right against and fails if it misses
it. A **demonstrator** has none of that: it shows the solver running on a
problem the rest of the suite cannot express. Nothing here is evidence of
accuracy, and nothing here is registered with `add_test`.

## urban

Passive-scalar dispersion through a voxelised city, D3Q7 advection-diffusion on
a prescribed logarithmic wind. Written up as the urban section of
`doc/m3lb.tex`. Two cities have been run, from the same OSM-derived height
fields: Manchester city centre (400x400x60 at 5 m, 1.79% solid) and Midtown
Manhattan (400x400x100 at 5 m, 6.87% solid, towers to 472 m).

| file | what it is |
|---|---|
| `urban.cpp` | the case: height-field ingest, log-law wind, physical-units scaling, source injection, div(u) and stability diagnostics, legacy-VTK output |
| `vol_urban.cpp` | volume renderer — nearest-neighbour buildings, three translucent concentration shells, front-to-back compositing onto a light ground |
| `../doc/fig/make_urban_anim.sh` | the collage pipeline: three cameras per frame, SVG overlay, ImageMagick, ffmpeg |
| `../doc/fig/plume_stats.py` | measures the plume — crosswind spread below and above the rooflines, and the shell levels by enclosed mass |
| `../doc/fig/cams/*.env` | camera presets, one per domain shape |

### Running

```
build/demonstrator/urban -geom <prefix> -out <dir> -diff 35 -bearing 208 \
  -src 100 29 1 -minutes 15 -out-every 18 --kokkos-num-threads=4
```

`<prefix>` names `<prefix>_heights.npy` and `<prefix>_meta.json`. `-bearing` is
meteorological, the direction the wind comes FROM, so 270 is a westerly blowing
toward +x. `-diffusion-only` drops the wind entirely and takes its time step
from the relaxation rate instead. `-top open` holds the top at C = 0 rather
than treating it as a lid. `-mode solved` solves the wind with D3Q27 TRT instead
of prescribing it, and is **interior-only**: the lateral boundaries jet.

**Pass `--kokkos-num-threads`.** This build uses the Threads backend, which
defaults to one thread and ignores `OMP_NUM_THREADS`. Four is the whole gain on
an M1 — D3Q7 is memory bound and the efficiency cores add nothing. 84 MLUPS
serial, 173 at four threads, on the Manhattan grid.

**Read the stability margin before letting a run go long.** The prescribed wind
is not divergence-free, which puts a spurious `-C div(u)` source in the
transport term, and below about 8x the damping it wins. The failure does not
look like a failure: mass tracks injection while the undershoot grows
geometrically. The `min C` column is the early warning, and the run now stops
itself when the undershoot passes half the peak. A larger `-diff` is the fix; a
shorter time step is not, and neither is `-u-lat`.

**`retained` is a real conservation statement now** — it sums every population
in the lattice, not the macroscopic field over fluid cells. The `in fluid`
column beside it is how much of that total the plotted field sees; the rest is
in slots owned by walls, in flight for one step. Over Manhattan that is 9%. See
`validation/scalar_mass.cpp`.

### Rendering

```
CAMS=doc/fig/cams/manhattan.env doc/fig/make_urban_anim.sh \
  <vtk_dir> <run.log> <work_dir> out.mp4 "$(python3 doc/fig/plume_stats.py \
  <vtk_dir>/conc_0049.vtk --log <run.log> --levels)"
```

The run log is an input, not a convenience: frame times, the city, the grid, the
wind and the fetch are all read out of it, so a caption cannot disagree with the
run it is captioning. Shell levels are measured from the last frame — the same
release spread over 2 km carries an order of magnitude less than it does over
200 m, and levels carried over from another run mean nothing.

## aorta

Flow through a voxelised patient-specific aorta (SimVascular case
`0074_H_AO_H`), D3Q27 central moments. Written up as §12 of `doc/m3lb.tex`.

| file | what it is |
|---|---|
| `aorta.cpp` | the case: geometry ingest, boundary conditions, conserving-outflow controller, steady and pulsatile drive |
| `vol_aorta.cpp` | volume renderer for the dumps — translucent vessel shell, speed-coloured interior, streamlines, cardiac-phase inset |
| `make_aorta_anim.py` | driver: shared colour scale across frames, then ffmpeg |

Two things it depends on that deliberately live elsewhere:

- **`src/io/VoxelGeometry.hpp`** is the voxel-geometry reader. It is a *solver
  capability* — `set_geometry` takes an arbitrary predicate, so any voxelised
  geometry can drive the solver — and the aorta is one use of it, not its owner.
  It stays in `src/` for the same reason the lattices do.
- **`validation/FieldDump.hpp`** is the diagnostic dump helper, shared with five
  validation cases. It is not part of the solver.

Generated figures land in `doc/fig/` with every other figure in the document,
since that is where `\graphicspath` points.

### Running

```
build/demonstrator/aorta -re 50 -u 0.02 -steps 13500
build/demonstrator/aorta -re 50 -u 0.02 -pulse -period 2000 -ramp 800 -steps 12800
```

`FIGVOL=1` dumps speed volumes; `FIGVEC=1` dumps the three velocity components,
which is what streamline integration needs. `-dumpfrom N` restricts dumping to a
window at the end of a run — vector dumps are 3x the size, and a long run needs
to converge but only its last beats need rendering.

```
python3 demonstrator/make_aorta_anim.py <dump_dir> out.mp4 15 --period 2000 --probe 100
```

## rayleigh_taylor

Heavy fluid over light in a box W x 4W, periodic in x, no-slip top and bottom,
with the interface given a single-mode perturbation of amplitude 0.1 W. The
multiphase module doing what it was built for: an interface that rolls up,
reconnects and keeps going. Uses the pressure-based operator, which is the only
one here that reaches a density ratio at all.

| file | what it is |
|---|---|
| `rayleigh_taylor.cpp` | the case: hydrostatic initialisation, front tracking, raw field dumps |
| `render_rt.cpp` | frame renderer — phase field and vorticity, three palettes |

### Running

```
build/demonstrator/rayleigh_taylor -w 192 -at 0.1 -re 30000 -op cm \
  -nframes 180 -dump <dir> --kokkos-num-threads=4
build/demonstrator/render_rt -in <dir> -out <frames> -n 181 -pal aurora -up 2 -crop 350,860
ffmpeg -framerate 24 -i <frames>/rt_%04d.ppm -c:v libx264 -pix_fmt yuv420p out.mp4
```

`-at` is the Atwood number, so `rho_H/rho_L = (1+At)/(1-At)`; `-re` is built on
the free-fall velocity `sqrt(gW)`; time is reported as `t* = t / sqrt(W/(g At))`
so runs at the same At compare regardless of the lattice numbers underneath.

**THE RENDERER IS SEPARATE ON PURPOSE**, for the reason `vol_aorta` and
`vol_urban` are: welded to the simulation, every change of colour map costs a
full re-run — eight and a half minutes at W = 192 to alter a hue. Fields go out
raw in `FieldDump.hpp`'s format and `render_rt` turns 181 frames into pictures in
2.6 seconds. Palettes are `aurora` (default), `paper` (light ground, for slides
on white) and `neon`.

**The vorticity scale calibrates on the LAST frame, not the first.** The first
frame of this case is a fluid at rest, so its 99.9th-percentile vorticity is
round-off — 1.4e-07 against the 6.2e-03 the developed flow reaches — and
calibrating there puts the whole film at full saturation. A percentile rather
than a maximum, so one hot cell cannot flatten everything else.

**Use `-op cm` at any serious Reynolds number.** BGK relaxes every mode at
omega, and at Re = 30000 omega is 1.99795; it diverges at `t* = 1.5` where the
central-moment operator completes `t* = 3`.

### The three clips in `anim/`

| file | what it shows |
|---|---|
| `rayleigh_taylor.mp4` | At = 0.5, Re = 256 — the reference case, sound |
| `rayleigh_taylor_re30000.mp4` | At = 0.1, Re = 30000 — the high-Re case, sound |
| `rayleigh_taylor_at998.mp4` | At = 0.998 under **BGK**, and a KNOWN ARTEFACT |

The third is kept as evidence, not as a showcase. Its phase field is smooth and
physically sensible but its velocity field is dominated by a one-cell
alternating mode, measured 70x stronger than the same render at a density ratio
of 3 — grid-scale oscillation that BGK does not damp, which is the measurement
that motivated the central-moment operator. Do not show it as a result.

## water_entry

A square dropped into a free water surface: approach, impact, cavity, splash-up
jets. The multiphase module carrying a moving rigid body, coupled by volume
penalisation rather than an immersed boundary — see `PenalisedBody.hpp` for why.

| file | what it is |
|---|---|
| `water_entry.cpp` | the case: diffuse free surface, hydrostatic seed through the interface, a falling body with its reaction fed back into Newton |

### Running

```
build/demonstrator/water_entry -l 48 -ratio 50 -rhob 2 -theta 0 \
  -tmax 6 -nframes 150 -dump <dir> --kokkos-num-threads=4
build/demonstrator/render_rt -in <dir> -out <frames> -n 151 -body -pal aurora
```

`-rhob` is the body's density as a multiple of the water's, `-drop` its release
height in body widths, and `-theta` its release tilt in degrees. `-l` sets the
side of the square and everything scales off it.

**IT FALLS RATHER THAN BEING PUSHED.** De Rosis & Enan run this problem with a
prescribed constant entry velocity; here the reaction closes Newton's equations,
so the deceleration on impact is a result. `-rhob` below 1 floats: the square
enters, stops, reverses and bobs — measured at `-rhob 0.6`, reversing at
`t U/L = 5.0` and rising through 0.56 L before falling back. That case is the
one the classical Uhlmann fictitious-mass correction cannot express at all,
since its denominator `m_b - m_f` changes sign there.

**`-theta` makes it roll.** An off-axis square strikes one corner first and
slaps flat, which is not available to a translation-only body. The rigid-body
solve is a coupled 3x3 in sway, heave and roll; the hydrostatics it rests on is
checked against Archimedes and metacentric theory in
`validation/floating_body.cpp`, which is where the numbers are.

**There is no exact answer here**, which is why this is a demonstrator. Wagner's
slamming theory covers a wedge; a flat-bottomed square has a singular impact
pressure and no closed form. There is also no contact-line model, so read the
splash, not the meniscus.
