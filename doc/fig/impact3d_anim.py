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

# A DARK GROUND, AND THAT IS NOT DECORATION. The water is drawn translucent
# (see below on why it has to be), and translucency desaturates it toward
# whatever lies behind: on white, a plum at alpha 0.4 comes out pink and an
# amber body comes out muddy brown. Against near-black the same alpha reads as
# glass, and the cavity's curvature is legible because the shading has somewhere
# dark to fall off to. The 2-D renders keep the plum-on-white palette, which is
# right there -- they are opaque.
GROUND = "#12161c"     # near-black, slightly blue
WATER = "#5b8fb9"      # steel blue
BODY = "#f0e3c8"       # ivory, the one value that stays clear of the water
INK = "#c9d3de"        # labels, light enough to read on the ground


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
    """A parametric sphere as a quad mesh, returned in PLOT order (x, z, y).

    The arguments are in DOMAIN order -- cy is the height, along gravity -- and
    the return is in plot order, because the 3-D axes here put the domain's y on
    the vertical. Those two orders differing is exactly the trap: calling this
    as (cx, cz, cy, R) puts the body's height on the z axis and its z on the
    height, and the two are numerically close enough on a loose crop that the
    sphere still looks roughly right. It took a tighter crop, where the two
    differ by 17 cells, for the sphere to be visibly outside its own cavity.
    """
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

    # ONE CROP FOR EVERY FRAME, sized from the body's whole trajectory rather
    # than as a fixed fraction of the domain. A quarter-to-three-quarters band
    # left a third of the frame as undisturbed water below and still air above;
    # taking the travel plus a couple of radii of margin fills the picture
    # without ever clipping the body or the crown. Still FIXED across frames --
    # a crop that tracked the body would make a sinking sphere look stationary.
    v0 = load_volume(paths[0])
    nz0, ny0, nx0 = v0.shape
    if body:
        cys = [p_[1] for p_ in body.values()]
        rr = max(p_[3] for p_ in body.values())
        ylo = int(max(0, min(cys) - 4.0 * rr))
        yhi = int(min(ny0, max(cys) + 2.5 * rr))
    else:
        ylo, yhi = ny0 // 4, (3 * ny0) // 4

    for k, p in enumerate(paths):
        v = load_volume(p)
        nz, ny, nx = v.shape
        pose = body.get(k)
        zcut = int(round(pose[2])) if pose else nz // 2
        zcut = max(2, min(nz - 1, zcut))
        # THE FAR HALF IS KEPT, NOT THE NEAR ONE, and getting this backwards
        # produces a render that looks right and shows nothing: matplotlib does
        # not depth-sort a Poly3DCollection against a plot_surface, so whichever
        # half of the water is between the camera and the body draws OVER it and
        # the sphere disappears behind its own cavity. At azim = -62 the near
        # half is z < cz, so that is the half to remove.
        sub = np.ascontiguousarray(v[zcut:, ylo:yhi, :])
        # SEAL THE BLOCK, or the water is a membrane. Marching cubes on the
        # cropped array finds only the phi = 1/2 level set, which inside the
        # crop is the free surface and the cavity and nothing else -- no bottom,
        # no sides, no face on the plane we cut. Rendered, that is a sheet
        # floating in space with the sphere hanging below it, and it does not
        # read as water at all. Padding one layer of phi = 0 around the crop
        # makes the level set CLOSE: the free surface stays where it is, and
        # flat caps appear wherever the water meets the edge of the block -- the
        # cut plane included. That is what a cutaway of a solid looks like.
        #
        # The caps are an artefact of the crop and are honest ones: the block
        # genuinely is a sub-volume, and the label says so.
        #
        # BUT NOT ON THE CUT FACE. Sealing all six sides makes the near face a
        # filled plane pointing at the camera, and then the picture is a solid
        # slab with the cavity and the body hidden inside it -- the opposite
        # failure to the membrane. The cut plane is the first z index of the
        # kept block, so it is padded on the far side only: bottom and sides
        # closed, the face we cut left OPEN to look in through.
        sub = np.pad(sub, ((0, 1), (1, 1), (1, 1)), mode="constant",
                     constant_values=0.0)

        fig = plt.figure(figsize=(5.2, 5.6), dpi=115, facecolor=GROUND)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(GROUND)
        try:
            verts, faces, _, _ = measure.marching_cubes(sub, level=0.5)
            verts[:, 1] -= 1.0           # undo the pad on y ...
            verts[:, 2] -= 1.0           # ... and on x; axis 0 had none below
            verts[:, 0] += zcut          # the slice started at zcut
            # verts are (z, y, x); the plot's axes are (x, z, y).
            # TRANSLUCENT, because cutting alone cannot expose the body: once
            # the cavity closes it is a POCKET around the sphere, so its far
            # wall faces inward and occludes the ball from every angle. Alpha
            # lets the cavity keep its shape while the body shows through it.
            mesh = Poly3DCollection(verts[faces][:, :, [2, 0, 1]], alpha=0.42)
            mesh.set_facecolor(WATER)
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
        except (ValueError, RuntimeError):
            pass                      # a frame with no interface is still a frame

        if pose:
            cx, cy, cz, R = pose
            xs, zs, ys = sphere_mesh(cx, cy - ylo, cz, R)
            ax.plot_surface(xs, zs, ys, color=BODY, shade=True,
                            linewidth=0, antialiased=False)

        ax.set_xlim(0, nx)
        ax.set_ylim(0, nz)
        ax.set_zlim(0, yhi - ylo)
        # zoom, because a 3-D axes leaves a wide margin by default and the
        # subject was occupying about a third of the frame.
        ax.set_box_aspect((1, 1, (yhi - ylo) / float(nx)), zoom=1.45)
        # 24 rather than 16: at a shallower angle the free surface is nearly
        # edge-on and the cavity it makes reads as a line rather than a bowl.
        ax.view_init(elev=24, azim=-62)
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig.text(0.5, 0.985, "$t/t_0 = %.2f$" % (k * dt), ha="center",
                 va="top", fontsize=12, color=INK)
        fig.text(0.015, 0.985, r"$g \downarrow$", ha="left", va="top",
                 fontsize=10, color="#7c8896")
        fig.text(0.5, 0.015, "a cropped block, cut at the body's centre plane; "
                 "flat faces are the crop, not the tank", ha="center",
                 fontsize=7, color="#6b7684")
        fig.savefig(os.path.join(tmp, "f%04d.png" % k), dpi=115,
                    facecolor=GROUND)
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
