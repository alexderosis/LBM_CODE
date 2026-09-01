#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python on the
# development machine has neither. A throwaway venv is enough:
#     python3 -m venv /tmp/v3 && /tmp/v3/bin/pip install numpy matplotlib
#     /tmp/v3/bin/python doc/fig/enan_rt_phi.py fig/enan_rt_phi.png
"""The order parameter through the Rayleigh-Taylor instability, against the
snapshots De Rosis & Enan show in their Figs. 12, 13 and 15.

One row per case, one column per tabulated instant t/t0 = 0, 0.5, ... 3. The
heavy fluid is dark. Rows one and two are their Figs. 12 and 13, 2-D at
Re = 256 and Re = 3000; row three is the plane z = nz/2 of the 3-D Re = 256
case, which is the diagonal the finger forms on and is the case that
disagrees with their Table IX.

Reads doc/fig/rtphi_<tag>_t<i>.bin, written by validation/enan_rt -field: two
int32 dimensions then nx*ny float32, x fastest. Those are gitignored --
regenerable output, not tracked data -- so run the driver before this script.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "enan_rt_phi.png"
HERE = os.path.dirname(os.path.abspath(__file__))
TSTAR = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

ROWS = [("rt2d_w256_re256_iw3",  r"2-D, $Re=256$"        + "\n(their Fig. 12)"),
        ("rt2d_w256_re3000_iw3", r"2-D, $Re=3000$"       + "\n(their Fig. 13)"),
        ("rt3d_w64_re256_iw3",   r"3-D, $Re=256$, $z=W/2$" + "\n(their Fig. 15)")]

def load(tag, i):
    p = os.path.join(HERE, f"rtphi_{tag}_t{i}.bin")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        nx, ny = np.fromfile(f, dtype=np.int32, count=2)
        a = np.fromfile(f, dtype=np.float32, count=int(nx) * int(ny))
    if a.size != int(nx) * int(ny):
        return None
    return a.reshape(int(ny), int(nx))

have = [(t, l) for t, l in ROWS if load(t, 0) is not None]
if not have:
    sys.exit("no snapshots found -- run validation/enan_rt -field first")

fig, ax = plt.subplots(len(have), len(TSTAR),
                       figsize=(1.55 * len(TSTAR), 4.4 * len(have)))
ax = np.atleast_2d(ax)

for r, (tag, label) in enumerate(have):
    for c, ts in enumerate(TSTAR):
        a = ax[r, c]
        phi = load(tag, c)
        a.set_xticks([]); a.set_yticks([])
        if phi is None:
            a.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=7,
                   color="#999999", transform=a.transAxes)
            for sp in a.spines.values():
                sp.set_visible(False)
            continue
        # Heavy fluid dark. The interface is phi = 1/2, drawn on top so the
        # diffuse profile does not hide where it actually is.
        a.imshow(phi, origin="lower", cmap="BuPu", vmin=0.0, vmax=1.0,
                 aspect="auto", interpolation="bilinear")
        a.contour(phi, levels=[0.5], colors="k", linewidths=0.6)
        if r == 0:
            a.set_title(f"$t/t_0={ts:g}$", fontsize=9)
        if c == 0:
            a.set_ylabel(label, fontsize=8.5)

fig.tight_layout(pad=0.35)
fig.savefig(OUT, dpi=165)
print("wrote", OUT, f"({len(have)} rows)")
