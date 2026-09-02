#!/usr/bin/env python3
# NOTE ON DEPENDENCIES: numpy and matplotlib, plus ffmpeg on PATH. Colab has all
# three already; the SYSTEM python here has none of them, so locally use a venv:
#     python3 -m venv /tmp/v3 && /tmp/v3/bin/pip install numpy matplotlib
#
#     python3 doc/fig/rt2d_anim.py '/content/rt2d/f_*.bin' rt2d.mp4 10.0
"""The 2-D Rayleigh-Taylor order parameter, as an animation.

Reads the PLANE dumps that GPU/src/rti3d.cu writes without -vol: two int32
dimensions (nx, ny) then nx*ny float32, x fastest. That is the same layout
doc/fig/enan_rt3d_compare.py reads, and it is NOT the -vol layout, which puts
three int32 in front -- a volume read as a plane produces a picture rather than
an error, so the header length is checked rather than assumed.

WHY THE FRAME IS ROTATED. The domain is W x 4W with gravity along y, so drawn
honestly it is a 1:4 portrait -- at W = 500 that is a 500 x 2000 video, which no
player shows usefully. Gravity is drawn pointing LEFT instead of down: the
aspect is preserved exactly, nothing is stretched, and the result fits a screen.
The gravity arrow in the corner says which way is down.

The colour scale is pinned to [0, 1] for every frame. Letting it track the
per-frame range would hide exactly the thing worth seeing -- the order parameter
overshooting its bounds when the scheme is marginal, which at Re = 30000 it is.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Pale cool grey for the light phase, the plum of the isosurface figures for the
# heavy one, so the two Rayleigh-Taylor figures in doc/ read as one family.
CMAP = LinearSegmentedColormap.from_list(
    "m3lb_rt", ["#eef2f5", "#c8d3dc", "#9d7f92", "#7a3b52", "#4a2138"])


def load_plane(path):
    """Two int32 (nx, ny) then nx*ny float32, x fastest. Returns (ny, nx)."""
    size = os.path.getsize(path)
    nx, ny = (int(v) for v in np.fromfile(path, dtype=np.int32, count=2))
    if 8 + 4 * nx * ny != size:
        raise ValueError(
            "%s is %d bytes, not the %d a %dx%d plane needs -- it is probably a "
            "-vol dump, which has a THREE-int32 header"
            % (os.path.basename(path), size, 8 + 4 * nx * ny, nx, ny))
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(ny, nx)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "/content/rt2d/f_*.bin"
    out = sys.argv[2] if len(sys.argv) > 2 else "rt2d.mp4"
    tmax = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    # A DIRECTORY IS ACCEPTED AS WELL AS A GLOB, and that is not laziness: in a
    # Colab cell a quoted wildcard is exactly the thing the editor's bracket
    # auto-pairing corrupts, and an unquoted one is expanded by the shell so
    # only the FIRST frame ever reaches argv. Passing the directory sidesteps
    # both.
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*.bin")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("no frames matching " + pattern)
    dt = tmax / max(1, len(paths) - 1)

    tmp = os.path.join(os.path.dirname(os.path.abspath(out)) or ".", "_rt2dframes")
    os.makedirs(tmp, exist_ok=True)

    lo, hi = 0.0, 0.0
    for k, p in enumerate(paths):
        phi = load_plane(p)
        lo, hi = min(lo, float(phi.min())), max(hi, float(phi.max()))
        ny, nx = phi.shape
        # Rotate so the long axis is horizontal; gravity then points left.
        img = np.rot90(phi, k=1)
        fig_w = 12.0
        fig_h = fig_w * nx / float(ny)
        fig = plt.figure(figsize=(fig_w, fig_h + 0.5), dpi=100)
        ax = fig.add_axes([0.0, 0.0, 1.0, fig_h / (fig_h + 0.5)])
        ax.imshow(img, origin="lower", cmap=CMAP, vmin=0.0, vmax=1.0,
                  aspect="equal", interpolation="bilinear")
        ax.set_axis_off()
        # Anchored to the very top of the figure, NOT to the axes: the label
        # band is the 0.5 in of figure height the axes does not occupy, and
        # placing the text relative to the axes puts it back inside the image.
        fig.text(0.5, 0.995, "$t/t_0 = %.2f$" % (k * dt), ha="center",
                 va="top", fontsize=13, color="#222222")
        fig.text(0.012, 0.995, r"$g \leftarrow$", ha="left", va="top",
                 fontsize=11, color="#666666")
        fig.savefig(os.path.join(tmp, "f%04d.png" % k), dpi=100,
                    facecolor="white")
        plt.close(fig)
        if (k + 1) % 20 == 0 or k + 1 == len(paths):
            print("  rendered %d/%d" % (k + 1, len(paths)), flush=True)

    # libx264 needs even dimensions; scale to a fixed width and round the height.
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(tmp, "f%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-vf", "scale=1200:-2", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print("wrote %s  (%d frames, %.1f MB)"
          % (out, len(paths), os.path.getsize(out) / 1e6))
    # Reported because the colour scale is clamped and would otherwise hide it:
    # phi outside [0, 1] is the scheme running out of margin, not a rendering
    # choice, and at Re = 30000 tau sits at 0.502.
    print("  phi over the whole run stayed within [%.4f, %.4f]" % (lo, hi))
    if lo < -0.01 or hi > 1.01:
        print("  NOTE: phi left [0, 1] by more than 1e-2 -- the run is marginal,"
              " and the colour scale is clamped so the frames do not show it")


if __name__ == "__main__":
    main()
