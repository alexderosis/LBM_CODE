#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. numpy, matplotlib and scikit-image; the SYSTEM python
# here has none of them:
#     python3 -m venv /tmp/v3 && /tmp/v3/bin/pip install numpy matplotlib scikit-image
#     /tmp/v3/bin/python doc/fig/enan_rt3d_render.py fig/enan_rt3d.png
"""The 3-D Rayleigh-Taylor interface, as their Fig. 15 shows it.

The phi = 1/2 isosurface at each tabulated instant, extracted by marching
cubes from the volumes validation/enan_rt writes with -field -vol. A mid-plane
SLICE is not the same picture once the spike rolls up -- the mushroom cap and
the saddle points between the four sides are exactly what a single cut misses,
and they are what their Fig. 15 is showing.

The domain is W x 4W x W at W = 64 with gravity along y. Only the middle of
the column is drawn: the spike and bubble stay well inside it and the full
4W aspect ratio wastes most of the frame.

Reads doc/fig/rtphi_<tag>_t<i>.bin -- three int32 dimensions then nx*ny*nz
float32 with x fastest, gitignored as regenerable output.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

OUT = sys.argv[1] if len(sys.argv) > 1 else "enan_rt3d.png"
TAG = sys.argv[2] if len(sys.argv) > 2 else "rt3d_w64_re256_iw5"
HERE = os.path.dirname(os.path.abspath(__file__))
TSTAR = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

def load(i):
    p = os.path.join(HERE, f"rtphi_{TAG}_t{i}.bin")
    if not os.path.exists(p): return None
    with open(p, "rb") as f:
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        a = np.fromfile(f, dtype=np.float32, count=int(nx)*int(ny)*int(nz))
    if a.size != int(nx)*int(ny)*int(nz) or nz < 2:
        return None                                  # a slice, not a volume
    return a.reshape(int(nz), int(ny), int(nx))      # z, y, x

have = [i for i in range(len(TSTAR)) if load(i) is not None]
if not have:
    sys.exit("no 3-D volumes -- run enan_rt -3d -field -vol first")

n = len(have)
fig = plt.figure(figsize=(max(7.0, 2.3*n), 4.0))
for k, i in enumerate(have):
    v = load(i)                                       # (z, y, x)
    nz, ny, nx = v.shape
    # keep the middle half of the long axis: the interface never leaves it
    ylo, yhi = ny//4, (3*ny)//4
    sub = v[:, ylo:yhi, :]
    ax = fig.add_subplot(1, n, k+1, projection="3d")
    try:
        verts, faces, _, _ = measure.marching_cubes(np.ascontiguousarray(sub),
                                                    level=0.5)
    except (ValueError, RuntimeError):
        ax.set_axis_off(); ax.set_title(f"$t/t_0={TSTAR[i]:g}$", fontsize=9)
        continue
    # verts are (z, y, x); draw with y up
    mesh = Poly3DCollection(verts[faces][:, :, [2, 0, 1]], alpha=0.9)
    mesh.set_facecolor("#7a3b52")
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    ax.set_xlim(0, nx); ax.set_ylim(0, nz); ax.set_zlim(0, yhi-ylo)
    ax.set_box_aspect((1, 1, 2))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-58)
    # No box, no panes, no grid: the surface is the whole content and the
    # frame only competes with it across seven small panels.
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)
        pane.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_title(f"$t/t_0={TSTAR[i]:g}$", fontsize=9)

fig.suptitle("3-D Rayleigh-Taylor, $Re=256$, $W=64$: the $\\phi=1/2$ surface, "
             "as their Fig. 15 shows it", fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.0)
fig.savefig(OUT, dpi=170)
print("wrote", OUT, f"({n} instants)")
