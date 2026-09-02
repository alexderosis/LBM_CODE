#!/usr/bin/env python3
# NOTE ON DEPENDENCIES, same as enan_rt3d_render.py: numpy, matplotlib and
# scikit-image, which the SYSTEM python here does not have.
#     python3 -m venv /tmp/v3 && /tmp/v3/bin/pip install numpy matplotlib scikit-image
#     /tmp/v3/bin/python doc/fig/rt3d_isoanim.py '/tmp/cudavol/v_*.bin' out.mp4
# ffmpeg is used for the encode and must be on PATH.
"""An ANIMATION of the phi = 1/2 surface through a 3-D Rayleigh-Taylor run.

enan_rt3d_render.py draws the seven tabulated instants side by side, which is
the figure the paper's Fig. 15 is. This draws every frame of a dense dump and
encodes them, which is the thing you actually want when the question is what
the interface DOES rather than where the spike is at seven moments.

Input is whatever GPU/src/rti3d.cu -vol writes, or validation/enan_rt -field
-vol: three int32 dimensions, then nx*ny*nz float32 with x fastest.

WHY THE MIDDLE HALF OF THE COLUMN ONLY. The domain is W x 4W x W and the
interface never leaves the middle band; drawing the full aspect ratio wastes
three quarters of every frame on undisturbed fluid. The crop is fixed across
frames so the surface is not silently rescaled as it grows -- an animation that
rescales its own axes makes a spike that falls look like a spike that does not.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def load_volume(path):
    """Three int32 dimensions, then nx*ny*nz float32. Returns (z, y, x)."""
    nx, ny, nz = np.fromfile(path, dtype=np.int32, count=3)
    nx, ny, nz = int(nx), int(ny), int(nz)
    if nz < 2:
        raise ValueError("%s has nz = %d: that is a -field plane, not a -vol "
                         "volume" % (os.path.basename(path), nz))
    a = np.fromfile(path, dtype=np.float32, offset=12)
    if a.size != nx * ny * nz:
        raise ValueError("%s: %d floats for a %dx%dx%d volume"
                         % (os.path.basename(path), a.size, nx, ny, nz))
    return a.reshape(nz, ny, nx)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cudavol/v_*.bin"
    out = sys.argv[2] if len(sys.argv) > 2 else "rt3d_iso.mp4"
    label = sys.argv[3] if len(sys.argv) > 3 else "GPU/ code (host build)"
    tref = float(sys.argv[4]) if len(sys.argv) > 4 else 1600.0
    nsteps = float(sys.argv[5]) if len(sys.argv) > 5 else 4800.0

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("no volumes matching " + pattern)
    every = nsteps / max(1, len(paths) - 1)

    tmp = os.path.join(os.path.dirname(os.path.abspath(out)) or ".", "_isoframes")
    os.makedirs(tmp, exist_ok=True)

    for k, p in enumerate(paths):
        v = load_volume(p)
        nz, ny, nx = v.shape
        ylo, yhi = ny // 4, 3 * ny // 4
        sub = np.ascontiguousarray(v[:, ylo:yhi, :])
        fig = plt.figure(figsize=(3.2, 5.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        try:
            verts, faces, _, _ = measure.marching_cubes(sub, level=0.5)
            mesh = Poly3DCollection(verts[faces][:, :, [2, 0, 1]], alpha=0.95)
            mesh.set_facecolor("#7a3b52")
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
        except (ValueError, RuntimeError):
            pass                       # a frame with no interface is still a frame
        ax.set_xlim(0, nx)
        ax.set_ylim(0, nz)
        ax.set_zlim(0, yhi - ylo)
        ax.set_box_aspect((1, 1, 2))
        # elev 22 rather than a near-edge-on 14: at 14 the cap of the mushroom
        # hides the four side spikes behind it, which is most of what the
        # surface is doing.
        ax.view_init(elev=22, azim=-60)
        ax.set_axis_off()
        # The 3-D axes are given the whole figure and the labels are drawn on
        # top of it; the default margins leave a third of the frame empty.
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig.text(0.5, 0.955, "$t/t_0 = %.2f$" % (k * every / tref),
                 ha="center", fontsize=11)
        fig.text(0.5, 0.02, label, ha="center", fontsize=7, color="#555555")
        fig.savefig(os.path.join(tmp, "f%04d.png" % k), dpi=120)
        plt.close(fig)
        if (k + 1) % 10 == 0 or k + 1 == len(paths):
            print("  rendered %d/%d" % (k + 1, len(paths)), flush=True)

    cmd = ["ffmpeg", "-y", "-framerate", "10", "-i", os.path.join(tmp, "f%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=460:-2", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print("wrote", out, "(%d frames, %.0f kB)" % (len(paths),
                                                  os.path.getsize(out) / 1024))


if __name__ == "__main__":
    main()
