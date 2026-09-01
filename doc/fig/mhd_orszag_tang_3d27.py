#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python has
# neither. python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
"""Orszag-Tang vortex on D3Q27 + D3Q7 with a single cell in z, against the
native D2Q9 + D2Q5 pair.

Running a two-dimensional problem on a 3-D lattice with nz = 1 and periodic z
is a reduction test: the wrap sends the out-of-plane neighbour back to the node
itself, so the extra populations stream in place and the answer must land on the
2-D scheme's. D3Q27 is a product lattice, so that reduction is exact; the panel
of differences is what says whether it happened.

Reads the raw float32 field dumps orszag_tang -dump writes:
  int32 nx, int32 ny, then nx*ny float32, x fastest.
Usage: mhd_orszag_tang_3d27.py out.png <dir-with-27> <dir-with-9>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "doc/fig/mhd_orszag_tang_3d27.png"
d27 = sys.argv[2] if len(sys.argv) > 2 else "."
d9  = sys.argv[3] if len(sys.argv) > 3 else "."


def raw(path):
    with open(path, "rb") as fh:
        nx, ny = np.fromfile(fh, dtype=np.int32, count=2)
        return np.fromfile(fh, dtype=np.float32, count=nx * ny).reshape(ny, nx)


plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(2, 3, figsize=(13.6, 8.6))

for row, (what, sym) in enumerate((("j", r"$j = \nabla\times b$"),
                                   ("zeta", r"$\zeta = \nabla\times u$"))):
    for col, t in enumerate(("t05", "t1")):
        f27 = os.path.join(d27, f"ot_{what}_{t}.bin")
        f9 = os.path.join(d9, f"ot_{what}_{t}.bin")
        if not os.path.exists(f27):
            ax[row, col].set_axis_off(); continue
        A = raw(f27)
        lim = np.abs(A).max()
        im = ax[row, col].imshow(A, origin="lower", cmap="RdBu_r",
                                 vmin=-lim, vmax=lim,
                                 extent=(0, 1, 0, 1), aspect="equal")
        ax[row, col].set_title(f"{sym},  $t$ = {'0.5' if t=='t05' else '1.0'} s\n"
                               f"D3Q27 + D3Q7, $n_z$ = 1   "
                               f"$\\|\\cdot\\|_\\infty$ = {lim:.4g}", fontsize=9.5)
        ax[row, col].set_xlabel("$x/L$"); ax[row, col].set_ylabel("$y/L$")
        fig.colorbar(im, ax=ax[row, col], fraction=0.046, pad=0.03)

    # third column: the difference against the native 2-D pair at t = 1
    f27 = os.path.join(d27, f"ot_{what}_t1.bin")
    f9 = os.path.join(d9, f"ot_{what}_t1.bin")
    a = ax[row, 2]
    if os.path.exists(f27) and os.path.exists(f9):
        A, B = raw(f27), raw(f9)
        Dif = A - B
        lim = max(np.abs(Dif).max(), 1e-300)
        im = a.imshow(Dif, origin="lower", cmap="PuOr", vmin=-lim, vmax=lim,
                      extent=(0, 1, 0, 1), aspect="equal")
        rel = np.abs(Dif).max() / np.abs(B).max()
        a.set_title(f"{sym}:  D3Q27 $-$ D2Q9 at $t$ = 1 s\n"
                    f"max $|\\Delta|$ = {np.abs(Dif).max():.3g}  "
                    f"({100*rel:.2g}% of peak)", fontsize=9.5)
        a.set_xlabel("$x/L$"); a.set_ylabel("$y/L$")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    else:
        a.set_axis_off()

fig.suptitle("Orszag-Tang vortex, Re = 628, $N$ = 512 -- M3LB D3Q27 fluid + "
             "D3Q7 magnetic field with a single cell in $z$.\nThe right column "
             "is the difference against the native D2Q9 + D2Q5 pair: the "
             "reduction on a product lattice should be exact.", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out, dpi=150)
print("wrote", out)
