# Running M3LB on CSF3

CSF3 is Manchester's Computational Shared Facility. It runs **SLURM** (it used
to be SGE — anything you find mentioning `qsub` is stale).

Host `csf3.itservices.manchester.ac.uk`, your **IT username, not your email**,
Duo 2FA (enter `1` for a push, or type a fob passcode). Off campus you need the
GlobalProtect VPN first.

---

## GPU partitions, and which one to use

| partition | GPU | per node | access | max wallclock |
|---|---|---|---|---|
| `gpuA` | A100 80 GB | 4 | **open to all** | 4 days |
| `gpuH` | H200 141 GB | 8 | **restricted — request it** | 4 days |
| `gpuH_short` | H200 141 GB | 8 | restricted | 1 day, interactive allowed |
| `gpuL` | L40S 48 GB | 4 | open to all | 4 days |
| `gpuA40GB` | A100 40 GB | 4 | very restricted | 4 days |
| `gpuV` | V100 | — | **withdrawn Oct 2025** | — |

**Use `gpuA`. Do not use `gpuL` for anything in this tree.** The L40S is Ada:
FP64 runs at 1/64 of FP32, about 1.4 TFLOPS. Every convergence test here is
FP64 and the central-moment collision is arithmetic-heavy — which is exactly
the regime that made a T4 give only 120 MLUPS FP64 at H = 1000, ALU-bound
rather than bandwidth-bound. The A100 is 1:2 FP64 at 9.7 TFLOPS, the H200
34 TFLOPS.

`gpuH` is worth requesting but do not wait for it; `gpuA` is open and enough.

---

## One-time setup

`~/.ssh/config` on your laptop, so login is one word and Duo asks once per ten
minutes rather than once per command:

```
Host csf3
    HostName csf3.itservices.manchester.ac.uk
    User <your-IT-username>
    ServerAliveInterval 60
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Then, on CSF3 — build in scratch, because the field dumps are GB-scale and home
has a quota. Scratch is **not backed up** and files unused for three months can
be deleted, so copy anything that matters back to home or RDS.

```bash
ssh csf3
cd ~/scratch && git clone https://github.com/alexderosis/M3LB.git && cd M3LB
module load libs/cuda/12.8.1        # check `module avail cuda`

# GPU/ -- its own CMake project, nvcc only, no Kokkos
cd GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=80 -DLBM_DOUBLE=ON
cmake --build build -j4
```

`LBM_GPU_ARCH` is **80** for the A100, **90** for an H200, 89 for an L40S.

For `cmbench` you also need the *parent* Kokkos tree, which is a separate CMake
project — see the header of `cmbench.sub`.

---

## The two jobs here

**Submit from the repo root.** `sbatch` copies the script, so it cannot locate
itself and the submit directory is the only handle it has. Either submit from
the root or set `M3LB_ROOT`; both scripts check and fail with the fix printed.

```bash
cd ~/scratch/M3LB
sbatch GPU/csf3/cmbench.sub                          # first: under a minute
sbatch --array=0 GPU/csf3/rb_cold.sub                # H = 498,  ~5 min
sbatch --array=1 GPU/csf3/rb_cold.sub                # H = 998,  ~45 min
sbatch --array=2 --time=12:00:00 GPU/csf3/rb_cold.sub  # H = 1998, ~6 h
```

**Ask for the wallclock you need, not the maximum.** SLURM backfills short jobs
into gaps ahead of long ones, so a 1-day request can only start when a 1-day
hole opens. Submitting the 45-minute H = 998 element with `-t 1-0` got it
scheduled to start **sixteen hours later** with reason `(Priority)`; at
`--time=2:00:00` it schedules against a two-hour hole instead. The script now
defaults to 2 h, which covers elements 0 and 1; element 2 must override.

### Checking on it

An array job gets a compound id, `<jobid>_<index>` — `12345_0` for `--array=0`.

```bash
squeue --me                     # queued / running; ST is PD pending, R running
squeue --me --start             # SLURM's estimate of when a pending job starts
tail -f rbcold-12345_0.log      # live progress, one row per output interval
sacct -j 12345 --format=JobID,State,Elapsed,MaxRSS,ExitCode   # after it ends
scancel 12345_0                 # kill one array element
scancel 12345                   # kill the whole array
```

`squeue` showing nothing means the job has finished (or never started) — look at
the log. A job that died in the first seconds almost always says why in the
first ten lines: a missing module, or the `ERROR: no executable at ...` check.

The log lands in the directory you submitted from; the **field dumps go to
`runs/rb_cold_h<H>/`**, deliberately, since at `-out 1` they are hundreds of
megabytes and have no business next to the source.

**`cmbench.sub` first.** It settles the Kokkos central-moment collapse, which
has been open since the port and is the only thing in this tree that genuinely
cannot be diagnosed without a GPU. It is bounded by construction and cannot
hang. Its header says what each possible answer means.

**`rb_cold.sub`** is Rayleigh–Bénard at Ra = 10¹¹ with the reference's cold
start, at H = 498 / 998 / 1998. The H = 498 element reproduces a CPU run that
**failed** — halted on the maximum principle at t = 25 with T = 1.19 against a
physical maximum of 0.5 — so it is a cross-code check of a known failure, and
the other two are the refinement that failure calls for.

Rough cost on one A100 at ~1.5 GLUPS FP64, 100 free-fall times:

| H | grid | cells | steps | est. |
|---|---|---|---|---|
| 498 | 1000 × 500 | 5.0e5 | 1.0e6 | ~5 min |
| 998 | 2004 × 1000 | 2.0e6 | 2.0e6 | ~45 min |
| 1998 | 4012 × 2000 | 8.0e6 | 4.0e6 | ~6 h |

For scale, H = 498 took **3 hours on four CPU threads** before it died.

---

## Reading the output

In this order, and the first one is not optional:

1. **`T_min` / `T_max`.** Advection–diffusion with Dirichlet data obeys a
   maximum principle, so T outside `[0, 1]` is the scheme failing and nothing
   else. Out-of-bounds lines are marked `!`; the run halts at twice the range.
   The **last line of every run** states the worst excursion, when it happened,
   and whether it recovered — a run that *completed* is not thereby a run that
   *stayed in bounds*, and a Nusselt number measured outside them is not a
   measurement.
2. **`Nu_bot` against `Nu_top`.** They must agree with each other. Their
   disagreement is the honest error bar. The failed CPU run had 80.4 against
   57.4.
3. **`Nu_vol`** only when it is far above `Nu_floor` *and* not flipping sign.
   It carries a factor H/α — 5.3e6 at these parameters — so it is mostly
   amplified noise. In the CPU run it flipped sign every output row for the
   first twenty free-fall times while both plate estimators sat correctly at 1.
4. `Nu_ref` is the D3Q19 reference's own normalisation, printed only so the two
   codes can go in one table. It is the raw correlation divided by `nx-1`, and
   it is 3% high by construction. Do not mix it with `Nu_vol`.

## Post-processing

Field dumps are `<prefix>_T_*.bin` (temperature) and `<prefix>_u_*.bin` (speed),
`nx × ny` float32 behind two `int32` of header, one pair per output interval.

`doc/fig/mkpng.py` renders them and is **pure stdlib** — no numpy, which matters
because CSF3's default python may not have it:

```bash
cd runs/rb_cold_h498
python3 ../../doc/fig/mkpng.py seq rb_cold_h498_T_0050.bin T.png 0 1
```

**Pin the range** (`0 1` above, the physical range of T). Without it every frame
gets its own scale and the colours move when the field does not; pinned, a field
that leaves its physical range saturates visibly and the tool prints `CLIPPED` —
which is how an out-of-bounds run looks in a picture rather than in a column.

A whole sequence, then a video (`module load apps/binapps/ffmpeg` if `ffmpeg` is
not already on PATH):

```bash
mkdir -p png
for f in rb_cold_h498_T_*.bin; do
  n=${f##*_T_}; n=${n%.bin}
  python3 ../../doc/fig/mkpng.py seq "$f" "png/T_$n.png" 0 1
done
ffmpeg -y -framerate 10 -pattern_type glob -i 'png/T_*.png' \
       -vf vflip -c:v libx264 -pix_fmt yuv420p -crf 18 rb_cold_h498.mp4
```

**`-vf vflip` is not optional.** The dump writes row y = 0 first and PNG puts the
first row at the top, so the raw image has the HOT plate at the top and the
plumes falling. Without the flip the physics reads upside down.

### ParaView

`doc/fig/bin2vtk.py` converts the dumps to legacy VTK, which ParaView opens
directly. Pure stdlib, and the output is the same size as the input rather than
the ~5x an ASCII VTK costs:

```bash
cd runs/rb_cold_h998
python3 ../../doc/fig/bin2vtk.py --glob 'rb_cold_h998_T_*.bin' \
        --out h998 --pair rb_cold_h998_u
```

That writes `h998_0000.vtk .. h998_00NN.vtk`, each carrying **both** scalars --
`Temperature` and `Speed` -- so you can switch fields in ParaView without
reloading. Open the collapsed name (`h998_..vtk`) in the file dialog and
ParaView groups them as a time series with the animation controls live.

Two things the script exists to get right. Legacy VTK binary is specified
**big-endian** whatever machine wrote it; native little-endian floats produce a
file ParaView opens without complaint and renders as values around 1e-40 and
1e38, which looks like a diverged simulation rather than a byte-order mistake.
And the dumps carry a two-`int32` header that a raw reader has to be told to
skip.

If you would rather not convert, ParaView's **Raw (binary) Files** reader can
open a `.bin` directly: Data Extent `0 nx-1  0 ny-1  0 0`, Scalar Type `float`,
Byte Order `LittleEndian`, **Header Size `8`**. That last one is the two int32s;
without it the first two values are read as floats and the field is shifted by
two cells.

The driver's own `-vtk` flag also writes real VTK, but ASCII and only if you
decide before the run: ~40 MB per frame at 2004 x 1000 against 8 MB for the
binary dump.

For the linear phase the raw field is useless — the perturbation is four orders
below the mean profile. Render the departure from the horizontal average
instead, `T'(x,y) = T(x,y) − ⟨T⟩ₓ(y)`, with the diverging map:

```bash
python3 - <<'EOF'
import struct, sys
src, dst = 'rb_cold_h498_T_0010.bin', 'tp.bin'
d = open(src,'rb').read(); nx, ny = struct.unpack('<ii', d[:8])
v = list(struct.unpack('<%df' % (nx*ny), d[8:8+4*nx*ny]))
for y in range(ny):
    row = v[y*nx:(y+1)*nx]; m = sum(row)/nx
    for x in range(nx): v[y*nx+x] = row[x] - m
open(dst,'wb').write(struct.pack('<ii', nx, ny) + struct.pack('<%df' % (nx*ny), *v))
EOF
python3 ../../doc/fig/mkpng.py div tp.bin Tpert.png 1.0
```

That is what showed the mode structure on the CPU run while every Nusselt
column still read 1.0.

Copy anything worth keeping back to home or RDS — **scratch is not backed up**
and files unused for three months can be deleted.

---

## Notes

- **One GPU.** `GPU/` has no MPI and no multi-GPU, so `-G 1` always. Asking for
  a whole node idles three A100s and queues far longer. Multi-GPU would need a
  two-way, parity-dependent halo exchange because of Esoteric Pull — see the
  known-limitations section of `doc/m3lb.pdf`.
- **No restart.** A run that outlives its wallclock is lost, so size `-t`
  generously; you have 4 days on `gpuA` and the runs above need hours.
- **Check the module versions.** `libs/cuda/12.8.1` is what CSF3's own GPU
  example uses, but run `module avail cuda` and `module avail gcc` — nvcc needs
  a C++20 host compiler and the parent Kokkos tree will fail first if the
  default `gcc` is too old.
