#!/usr/bin/env python3
"""Rayleigh-Taylor 3-D: the Kokkos run's order parameter at the paper's instants.

Reads the mid-z phi planes that validation/enan_rt.cpp -field writes into
doc/fig, and draws them as a contact sheet at t/t0 = 0, 0.5, ..., 3.

WHY A PLANE AND NOT THE VOLUME. The finger forms on the diagonal of the box and
the plane at z = nz/2 cuts through it, so a slice carries the shape that the
spike position is measured from. The isosurface render (enan_rt3d_render.py) is
the other view and needs the full volume, which is 4 MB a frame.

The companion figure from GPU/src/rti3d.cu is generated the same way from the
same instants, so the two are directly comparable panel by panel.
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
T_STAR = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_plane(path):
    """A -field dump: int32 nx, ny then nx*ny float32, row-major in y.

    The -vol dumps from an earlier run have three int32 in front and are far
    larger; they are a different product and are rejected rather than
    misinterpreted, because a volume read as a plane produces a picture rather
    than an error.
    """
    size = os.path.getsize(path)
    hdr = np.fromfile(path, dtype=np.int32, count=3)
    nx, ny = int(hdr[0]), int(hdr[1])
    if 8 + 4 * nx * ny != size:
        raise ValueError(
            "%s is %d bytes, not the %d a %dx%d plane needs -- it is probably a "
            "-vol dump from another run" % (os.path.basename(path), size,
                                            8 + 4 * nx * ny, nx, ny))
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(ny, nx)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "rt3d_w64_re256_iw5"
    paths = [os.path.join(HERE, "rtphi_%s_t%d.bin" % (tag, i)) for i in range(7)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing: " + ", ".join(os.path.basename(p) for p in missing))

    planes = [load_plane(p) for p in paths]
    ny, nx = planes[0].shape
    W = nx
    y0, y1 = W // 2, 4 * W - W // 2          # the band the interface lives in

    fig, axs = plt.subplots(1, 7, figsize=(11.0, 4.2), dpi=120)
    for ax, t, p in zip(axs, T_STAR, planes):
        ax.imshow(p[y0:y1], origin="lower", cmap="RdBu_r", vmin=0.0, vmax=1.0,
                  aspect="equal")
        ax.set_axis_off()
        ax.set_title("t/t0 = %.1f" % t, fontsize=9)
    fig.suptitle("Kokkos M3LB, FP64 -- phase field CM + fluid CM, D3Q27, "
                 "W=%d Re=256 At=0.5 Ca=960 Pe=1024 xi=5" % W, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.06)
    out = os.path.join(HERE, "enan_rt3d_kokkos_sheet.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)

    # The spike, recomputed from the SAME planes the picture is drawn from, so
    # the figure and the number cannot disagree. Note this is the mid-z plane
    # only -- the driver's table is over the whole volume -- so it is a check on
    # the slice, not a replacement for the table.
    print("\n  t/t0   lowest y with phi>0.5 on this plane, over W")
    for t, p in zip(T_STAR, planes):
        rows = np.where((p > 0.5).any(axis=1))[0]
        print("  %4.1f   %.4f" % (t, rows.min() / W if len(rows) else float("nan")))


if __name__ == "__main__":
    main()
