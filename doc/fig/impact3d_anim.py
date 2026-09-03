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
    """{frame: pose} from <prefix>_body.dat, or {} if absent.

    TWO ARITIES, TOLD APART BY THE FIELD COUNT, because the drivers write two
    different bodies and neither should be able to be read as the other. Five
    fields is sphere_entry.cpp / impact.cu's `frame cx cy cz R`; eleven is
    cube_entry.cpp's `frame cx cy cz hx hy hz qw qx qy qz`. A line of any other
    length is skipped rather than guessed at -- a sphere renderer that silently
    read the first four numbers of a cube line would draw a ball of radius hx
    where the cube is, and it would look plausible.
    """
    p = prefix + "_body.dat"
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        f = line.split()
        if len(f) == 5:
            out[int(f[0])] = ("sphere",) + tuple(float(v) for v in f[1:])
        elif len(f) == 10:
            out[int(f[0])] = ("disc",) + tuple(float(v) for v in f[1:])
        elif len(f) == 11:
            out[int(f[0])] = ("box",) + tuple(float(v) for v in f[1:])
    return out


def quat_matrix(w, x, y, z):
    """The 3x3 rotation of a unit quaternion, matching Quat::matrix()."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# The 8 corners of the unit cube and the 6 faces, wound so that the cross
# product of the first two edges points OUT. The winding is what makes the
# shading below mean anything.
_CUBE_V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                    for sz in (-1, 1)], dtype=float)
_CUBE_F = [(0, 1, 3, 2), (4, 6, 7, 5),      # -x, +x
           (0, 4, 5, 1), (2, 3, 7, 6),      # -y, +y
           (0, 2, 6, 4), (1, 5, 7, 3)]      # -z, +z


def disc_faces(cx, cy, cz, R, hy, Rm, n=28):
    """A flat cylinder as quads and two caps, in PLOT order (x, z, y).

    The symmetry axis is BODY y, matching Disc in both codebases, so the disc is
    built as (R cos t, +/-hy, R sin t) and then rotated. Two caps plus a rim
    band -- the rim is what makes an attack angle legible, because a disc drawn
    as two flat circles has no thickness to read the tilt from.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([R * np.cos(t), np.zeros_like(t), R * np.sin(t)], axis=1)
    lo = ring + np.array([0.0, -hy, 0.0])
    hi = ring + np.array([0.0, +hy, 0.0])
    c = np.array([cx, cy, cz])
    lo = lo @ Rm.T + c
    hi = hi @ Rm.T + c

    light = np.array([0.4, 0.75, 0.53])
    light /= np.linalg.norm(light)
    quads, shades = [], []

    def push(poly):
        nrm = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        m = np.linalg.norm(nrm)
        shades.append(0.55 + 0.45 * abs(float(nrm @ light) / m) if m > 0 else 1.0)
        quads.append(poly[:, [0, 2, 1]])        # (x, y, z) -> (x, z, y)

    for i in range(n):
        j = (i + 1) % n
        push(np.array([lo[i], lo[j], hi[j], hi[i]]))
    push(hi)                                    # the two caps, as n-gons
    push(lo[::-1])
    return quads, shades


def box_faces(cx, cy, cz, hx, hy, hz, R):
    """Six quads in PLOT order (x, z, y), with a shade factor for each.

    A cube drawn in one flat colour reads as a hexagon: the three faces visible
    from any viewpoint have no edge between them that the eye can find. Each
    face is therefore shaded by |n . l| against a fixed light, which is the
    minimum needed for the three to separate -- and it also makes the TUMBLE
    legible, since the shading changes as the body turns even when its
    silhouette barely does.
    """
    v = (_CUBE_V * np.array([hx, hy, hz])) @ R.T + np.array([cx, cy, cz])
    light = np.array([0.4, 0.75, 0.53])
    light /= np.linalg.norm(light)
    quads, shades = [], []
    for f in _CUBE_F:
        q = v[list(f)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        nn = np.linalg.norm(n)
        shades.append(0.55 + 0.45 * abs(float(n @ light) / nn) if nn > 0 else 1.0)
        quads.append(q[:, [0, 2, 1]])       # (x, y, z) -> (x, z, y)
    return quads, shades


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
    # --key=value options after the four positionals. THE VIEW IS A PARAMETER
    # BECAUSE THE RIGHT VIEW DEPENDS ON THE BODY, not on taste.
    #
    # A SPHERE MUST BE CUT TO AND VIEWED FROM LOW DOWN: once its cavity closes
    # it is a pocket, so from any angle the far wall hides the ball, and only a
    # section shows the cavity's profile at all.
    #
    # AND FOR A SHALLOW SURFACE DEFORMATION, PREFER rt2d_anim.py ENTIRELY. The
    # stone's entry crater is 9 cells deep and about 10 across in a 256-wide
    # block -- 4 per cent of the frame -- and it sits 1.7 diameters BEHIND the
    # body, which has planed on past it. No 3-D view of an isosurface makes that
    # legible; the mid-plane section does, immediately, because it plots the
    # surface as a profile rather than as a nearly-flat plane seen obliquely.
    # Use this renderer for where the body is and that one for what the water
    # did.
    #
    # A SKIPPING DISC IS THE OPPOSITE CASE. It rides ON the surface rather than
    # inside a cavity, so nothing occludes it -- and the thing worth seeing is
    # the DEPRESSION it planes on, which lies in the surface plane. Cutting at
    # the body's centre throws away half of that depression, and a 24 degree
    # elevation puts the surface nearly edge-on so the rest of it reads as a
    # kink in a line. Hence --cut=none and a higher --elev for a skip.
    opts = {}
    pos = []
    for a in sys.argv[1:]:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
        else:
            pos.append(a)
    src = pos[0] if len(pos) > 0 else "/tmp/sp24"
    out = pos[1] if len(pos) > 1 else "impact3d.mp4"
    tmax = float(pos[2]) if len(pos) > 2 else 6.0
    fps = int(pos[3]) if len(pos) > 3 else 12
    elev = float(opts.get("elev", 24))
    azim = float(opts.get("azim", -62))
    cut_mode = opts.get("cut", "body")          # "body" or "none"
    ylo_opt = opts.get("ylo")
    yhi_opt = opts.get("yhi")
    zoom = float(opts.get("zoom", 1.45))

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
        cys = [p_[2] for p_ in body.values()]
        # The body's own scale: a sphere's radius, or a box's largest half
        # extent times sqrt(3) -- the distance to its CORNER, which is what a
        # tumbling cube can reach and a half extent is not.
        rr = max(p_[4] if p_[0] in ("sphere", "disc")
                 else 1.7320508 * max(p_[4], p_[5], p_[6])
                 for p_ in body.values())
        ylo = int(max(0, min(cys) - 4.0 * rr))
        yhi = int(min(ny0, max(cys) + 2.5 * rr))
    else:
        ylo, yhi = ny0 // 4, (3 * ny0) // 4
    # An explicit band overrides the trajectory-sized one. For a skip the
    # disturbance is a few cells deep while the flight spans many, so a crop
    # sized from the trajectory is almost all undisturbed water.
    if ylo_opt is not None:
        ylo = int(max(0, float(ylo_opt)))
    if yhi_opt is not None:
        yhi = int(min(ny0, float(yhi_opt)))

    for k, p in enumerate(paths):
        v = load_volume(p)
        nz, ny, nx = v.shape
        pose = body.get(k)
        if cut_mode == "none":
            zcut = 0                     # keep the whole span; nothing to hide
        else:
            zcut = int(round(pose[3])) if pose else nz // 2
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
        # With a cut, pad the FAR side only so the cut face stays open to look
        # in through. With no cut there is no face to look through and the
        # block is sealed on all six sides, which is what a solid volume of
        # water should look like.
        zpad = (1, 1) if cut_mode == "none" else (0, 1)
        sub = np.pad(sub, (zpad, (1, 1), (1, 1)), mode="constant",
                     constant_values=0.0)

        fig = plt.figure(figsize=(5.2, 5.6), dpi=115, facecolor=GROUND)
        # computed_zorder OFF, and this is the fix for a real defect rather than
        # a preference. mplot3d orders whole ARTISTS by their average depth, not
        # faces across artists, so the water block -- which spans the far half
        # of the domain and averages nearer than the body sitting inside it --
        # was winning and drawing its alpha 0.42 blue OVER the cube. The effect
        # was not a missing body but a discoloured one: ivory under 42 % blue
        # comes out pale blue-grey, and only two of the cube's three visible
        # faces were affected, so it read as odd shading rather than as a
        # painter's-order bug. With this off the zorders below are obeyed
        # literally: water first, body always on top of it.
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.set_facecolor(GROUND)
        try:
            verts, faces, _, _ = measure.marching_cubes(sub, level=0.5)
            verts[:, 1] -= 1.0           # undo the pad on y ...
            verts[:, 2] -= 1.0           # ... and on x; axis 0 had none below
            if cut_mode == "none":
                verts[:, 0] -= 1.0       # sealed on z as well, so undo that pad
            verts[:, 0] += zcut          # the slice started at zcut
            # verts are (z, y, x); the plot's axes are (x, z, y).
            # TRANSLUCENT, because cutting alone cannot expose the body: once
            # the cavity closes it is a POCKET around the sphere, so its far
            # wall faces inward and occludes the ball from every angle. Alpha
            # lets the cavity keep its shape while the body shows through it.
            mesh = Poly3DCollection(verts[faces][:, :, [2, 0, 1]], alpha=0.42,
                                    zorder=1)
            mesh.set_facecolor(WATER)
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
        except (ValueError, RuntimeError):
            pass                      # a frame with no interface is still a frame

        if pose and pose[0] == "sphere":
            _, cx, cy, cz, R = pose
            xs, zs, ys = sphere_mesh(cx, cy - ylo, cz, R)
            ax.plot_surface(xs, zs, ys, color=BODY, shade=True,
                            linewidth=0, antialiased=False, zorder=2)
        elif pose and pose[0] == "disc":
            _, cx, cy, cz, R_, hy_, qw, qx, qy, qz = pose
            quads, shades = disc_faces(cx, cy - ylo, cz, R_, hy_,
                                       quat_matrix(qw, qx, qy, qz))
            base = np.array(matplotlib.colors.to_rgb(BODY))
            body_mesh = Poly3DCollection(quads, alpha=1.0, zorder=2)
            body_mesh.set_facecolor([tuple(np.clip(base * sh, 0, 1))
                                     for sh in shades])
            body_mesh.set_edgecolor("#6f6552")
            body_mesh.set_linewidth(0.4)
            ax.add_collection3d(body_mesh)
        elif pose and pose[0] == "box":
            _, cx, cy, cz, hx_, hy_, hz_, qw, qx, qy, qz = pose
            quads, shades = box_faces(cx, cy - ylo, cz, hx_, hy_, hz_,
                                      quat_matrix(qw, qx, qy, qz))
            base = np.array(matplotlib.colors.to_rgb(BODY))
            # ONE COLLECTION, NOT SIX. Separate collections are drawn in the
            # order they were added and matplotlib does not depth-sort between
            # them, so six of them means a back face can draw over a front one
            # and the cube turns inside out as it tumbles. Inside a single
            # Poly3DCollection the faces ARE sorted.
            body_mesh = Poly3DCollection(quads, alpha=1.0, zorder=2)
            body_mesh.set_facecolor([tuple(np.clip(base * sh, 0, 1))
                                     for sh in shades])
            body_mesh.set_edgecolor("#6f6552")
            body_mesh.set_linewidth(0.5)
            ax.add_collection3d(body_mesh)

        ax.set_xlim(0, nx)
        ax.set_ylim(0, nz)
        ax.set_zlim(0, yhi - ylo)
        # zoom, because a 3-D axes leaves a wide margin by default and the
        # subject was occupying about a third of the frame.
        ax.set_box_aspect((1, 1, (yhi - ylo) / float(nx)), zoom=zoom)
        # 24 rather than 16: at a shallower angle the free surface is nearly
        # edge-on and the cavity it makes reads as a line rather than a bowl.
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig.text(0.5, 0.985, "$t/t_0 = %.2f$" % (k * dt), ha="center",
                 va="top", fontsize=12, color=INK)
        fig.text(0.015, 0.985, r"$g \downarrow$", ha="left", va="top",
                 fontsize=10, color="#7c8896")
        fig.text(0.5, 0.015,
                 ("a cropped block; flat faces are the crop, not the tank"
                  if cut_mode == "none" else
                  "a cropped block, cut at the body's centre plane; "
                  "flat faces are the crop, not the tank"),
                 ha="center", fontsize=7, color="#6b7684")
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
