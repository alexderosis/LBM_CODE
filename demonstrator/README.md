# Demonstrator cases

A **validation** case has something to be right against and fails if it misses
it. A **demonstrator** has none of that: it shows the solver running on a
problem the rest of the suite cannot express. Nothing here is evidence of
accuracy, and nothing here is registered with `add_test`.

## aorta

Flow through a voxelised patient-specific aorta (SimVascular case
`0074_H_AO_H`), D3Q27 central moments. Written up as §12 of `doc/lbm_code.tex`.

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
