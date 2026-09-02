#!/usr/bin/env python3
# NOTE ON DEPENDENCIES: numpy, matplotlib and scikit-image, plus ffmpeg. The
# SYSTEM python here has none of the three:
#     python3 -m venv /tmp/v3 && /tmp/v3/bin/pip install numpy matplotlib scikit-image
#     /tmp/v3/bin/python doc/fig/impact3d_anim.py /tmp/sp24 out.mp4 6.0
"""A 3-D view of a sphere entering water: the phi = 1/2 surface and the body.

Reads what GPU/src/impact.cu -vol writes -- three int32 dimensions then
nx*ny*nz float32 with x fastest -- plus the <prefix>_body.dat trajectory, one
line of `frame cx cy cz R` per frame.

WHY THE BODY IS REBUILT RATHER THAN READ. chi is analytic, so four numbers
reproduce the sphere exactly and a chi volume would double the driver's output
to carry information that fits on one line. It also means the body is drawn as
the sphere it IS rather than as a marching-cubes approximation of its own
smoothed indicator, which would blur the one object in the picture whose shape
is known.

WHY THE WATER IS CUT AT THE BODY'S CENTRE PLANE. Drawn whole, the free surface
is a lid: it is flat and horizontal everywhere except right at the impact, so
from any viewpoint above it the cavity and the sphere are behind it and the
picture shows nothing. Cutting the volume at z = cz turns the render into a
section-in-3-D -- the cavity's profile is visible on the cut face, the crown
still stands proud of it, and the sphere sits in the middle of both. The cut
follows the body rather than the domain so it stays through the interesting
plane if the sphere ever drifts off axis.

The y band is cropped and FIXED across frames: the domain is twice as tall as
it is wide and the entry happens in the middle third of it, so most of the
height is undisturbed water. A crop that tracked the body would make a sinking
sphere look stationary.
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

WATER = "#7a3b52"      # the plum the 2-D renders use for phi = 1
BODY = "#e08a2e"       # the amber the 2-D renders use for the solid


def load_volume(path):
    """Three int32 dimensions then nx*ny*nz float32. Returns (z, y, x)."""
    nx, ny, nz = (int(v) for v in np.fromfile(path, dtype=np.int32, count=3))
    if nz < 2:
        raise ValueError("%s has nz = %d -- that is a plane, not a -vol volume"
                         % (os.path.basename(path), nz))
    a = np.fromfile(path, dtype=np.float32, offset=12)
    if a.size != nx * ny * nz:
        raise ValueError("%s: %d floats for a %dx%dx%d volume"
                         % (os.path.basename(path), a.size, nx, ny, nz))
    return a.reshape(nz, ny, nx)


def load_body(prefix):
    """{frame: (cx, cy, cz, R)} from <prefix>_body.dat, or {} if absent."""
    p = prefix + "_body.dat"
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        f = line.split()
        if len(f) == 5:
            out[int(f[0])] = tuple(float(v) for v in f[1:])
    return out


def sphere_mesh(cx, cy, cz, R, n=26):
    """A parametric sphere as a quad mesh, in the plot's (x, z, y) axes."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2 + 1)
    xs = cx + R * np.outer(np.cos(u), np.sin(v))
    zs = cz + R * np.outer(np.sin(u), np.sin(v))
    ys = cy + R * np.outer(np.ones_like(u), np.cos(v))
    return xs, zs, ys


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sp24"
    out = sys.argv[2] if len(sys.argv) > 2 else "impact3d.mp4"
    tmax = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 12

    prefix = src if not os.path.isdir(src) else os.path.join(src, "f")
    paths = sorted(glob.glob(prefix + "_vol_*.bin"))
    if not paths:
        raise SystemExit("no volumes matching " + prefix + "_vol_*.bin")
    body = load_body(prefix)
    dt = tmax / max(1, len(paths) - 1)

    tmp = os.path.join(os.path.dirname(os.path.abspath(out)) or ".", "_i3dframes")
    os.makedirs(tmp, exist_ok=True)

    # One crop for every frame, from the first volume's shape. See the docstring
    # on why this is fixed rather than tracking the body.
    v0 = load_volume(paths[0])
    nz0, ny0, nx0 = v0.shape
    ylo, yhi = ny0 // 4, (3 * ny0) // 4

    for k, p in enumerate(paths):
        v = load_volume(p)
        nz, ny, nx = v.shape
        pose = body.get(k)
        zcut = int(round(pose[2])) if pose else nz // 2
        zcut = max(2, min(nz - 1, zcut))
        sub = np.ascontiguousarray(v[:zcut, ylo:yhi, :])

        fig = plt.figure(figsize=(5.2, 5.6), dpi=115)
        ax = fig.add_subplot(111, projection="3d")
        try:
            verts, faces, _, _ = measure.marching_cubes(sub, level=0.5)
            # verts are (z, y, x); the plot's axes are (x, z, y).
            mesh = Poly3DCollection(verts[faces][:, :, [2, 0, 1]], alpha=1.0)
            mesh.set_facecolor(WATER)
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
        except (ValueError, RuntimeError):
            pass                      # a frame with no interface is still a frame

        if pose:
            cx, cy, cz, R = pose
            xs, zs, ys = sphere_mesh(cx, cz, cy - ylo, R)
            ax.plot_surface(xs, zs, ys, color=BODY, shade=True,
                            linewidth=0, antialiased=False)

        ax.set_xlim(0, nx)
        ax.set_ylim(0, nz)
        ax.set_zlim(0, yhi - ylo)
        ax.set_box_aspect((1, 1, (yhi - ylo) / float(nx)))
        # 24 rather than 16: at a shallower angle the free surface is nearly
        # edge-on and the cavity it makes reads as a line rather than a bowl.
        ax.view_init(elev=24, azim=-62)
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig.text(0.5, 0.985, "$t/t_0 = %.2f$" % (k * dt), ha="center",
                 va="top", fontsize=12)
        fig.text(0.015, 0.985, r"$g \downarrow$", ha="left", va="top",
                 fontsize=10, color="#666666")
        fig.text(0.5, 0.015, "water cut at the body's centre plane; "
                 "sphere drawn from its pose", ha="center", fontsize=7,
                 color="#777777")
        fig.savefig(os.path.join(tmp, "f%04d.png" % k), dpi=115,
                    facecolor="white")
        plt.close(fig)
        if (k + 1) % 8 == 0 or k + 1 == len(paths):
            print("  rendered %d/%d" % (k + 1, len(paths)), flush=True)

    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(tmp, "f%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-vf", "scale=600:-2", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print("wrote %s  (%d frames, %.1f MB)"
          % (out, len(paths), os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
