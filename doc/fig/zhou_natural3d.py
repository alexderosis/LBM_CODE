#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python has
# neither. python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
"""Zhou et al. (2026) Sec. 3.6: natural convection in a cubic cavity, Pr = 0.71.

The paper gives no table for this case -- it compares centreline profiles on the
z = L/2 plane against Fusegi et al. (1991), plotted as symbols in its Figs. 27
and 29. So the velocity panel here uses the paper's own normalisation,
u_ref = sqrt(g beta h dT), which is exactly the u_c these runs are pinned with,
and marks the peaks read off those figures. Those marks carry the precision of a
plot: about 5 percent.

Every curve is plotted against POSITION on the shared abscissa. Putting velocity
on one axis and position on the other in the same frame is what the first
version of this script did, and it produced a cross.

Reads results/J_zhou_thermal/nat3d_N*_ra<Ra>_<op>_* and the matching nat2d files.
"""
import glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "doc/fig/zhou_natural3d.png"
D = "results/J_zhou_thermal"
PR = 0.71

# Peaks read off Zhou et al. Figs. 27 and 29 (Fusegi et al. plotted as symbols).
FUSEGI = {"1e+05": dict(ux=0.14, yux=0.85, uy=0.22, xuy=0.075),
          "1e+06": dict(ux=0.11, yux=0.88, uy=0.26, xuy=0.055)}


def raw(path):
    with open(path, "rb") as fh:
        nx, ny = np.fromfile(fh, dtype=np.int32, count=2)
        return np.fromfile(fh, dtype=np.float32, count=nx * ny).reshape(ny, nx)


def newest(pat):
    f = sorted(glob.glob(pat))
    return f[-1] if f else None


ras = []
for f in sorted(glob.glob(f"{D}/nat3d_N*_centrelines.dat")):
    tag = f.split("_ra")[1].split("_")[0]
    if tag not in ras:
        ras.append(tag)
if not ras:
    sys.exit("no 3-D centreline dumps found")

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(len(ras), 3, figsize=(14.0, 4.4 * len(ras)), squeeze=False)

for i, ra in enumerate(ras):
    c3 = newest(f"{D}/nat3d_N*_ra{ra}_*_centrelines.dat")
    c2 = newest(f"{D}/nat2d_N*_ra{ra}_*_centrelines.dat")
    d3 = np.loadtxt(c3)
    s, th_h, th_v, ux3, uy3 = d3[:, 0], d3[:, 1], d3[:, 2], d3[:, 3], d3[:, 4]
    uref = np.sqrt(float(ra) * PR)          # (u H/alpha) / sqrt(Ra Pr) = u/u_ref
    d2 = np.loadtxt(c2) if c2 else None

    a = ax[i][0]
    a.plot(s, th_h + 0.5, "-", lw=1.8, color="tab:red",
           label=r"$\theta$ vs $x/L$ at $y/L$=0.5")
    a.plot(s, th_v + 0.5, "-", lw=1.8, color="tab:blue",
           label=r"$\theta$ vs $y/L$ at $x/L$=0.5")
    if d2 is not None:
        a.plot(d2[:, 0], d2[:, 1] + 0.5, "--", lw=1.1, color="0.35",
               label="2-D, same $Ra$")
        a.plot(d2[:, 0], d2[:, 2] + 0.5, "--", lw=1.1, color="0.35")
    a.set_xlabel("position along the centreline")
    a.set_ylabel(r"$\theta$")
    a.set_title(f"Ra = {float(ra):.0e}   centreline temperature", fontsize=10)
    a.set_xlim(0, 1)
    a.legend(fontsize=8); a.grid(alpha=0.3)

    b = ax[i][1]
    fu = FUSEGI.get(ra)
    b.plot(s, ux3 / uref, "-", lw=1.8, color="tab:blue",
           label=r"$u_x/u_{ref}$ vs $y/L$ at $x/L$=0.5")
    b.plot(s, uy3 / uref, "-", lw=1.8, color="tab:red",
           label=r"$u_y/u_{ref}$ vs $x/L$ at $y/L$=0.5")
    if d2 is not None:
        b.plot(d2[:, 0], d2[:, 3] / uref, "--", lw=1.1, color="tab:blue", alpha=0.55)
        b.plot(d2[:, 0], d2[:, 4] / uref, "--", lw=1.1, color="tab:red", alpha=0.55)
    if fu:
        b.plot([fu["yux"]], [fu["ux"]], "o", mfc="none", mec="tab:blue", ms=10,
               mew=1.7, label=f"Fusegi (read off Fig.): {fu['ux']} @ {fu['yux']}")
        b.plot([fu["xuy"]], [fu["uy"]], "s", mfc="none", mec="tab:red", ms=10,
               mew=1.7, label=f"Fusegi (read off Fig.): {fu['uy']} @ {fu['xuy']}")
    b.axhline(0, color="0.7", lw=0.8)
    b.set_xlabel("position along the centreline")
    b.set_ylabel(r"$u/u_{ref}$,  $u_{ref}=\sqrt{g\beta h\Delta T}$")
    b.set_title(f"Ra = {float(ra):.0e}   centreline velocity\n"
                "solid 3-D, dashed 2-D at the same Ra", fontsize=10)
    b.set_xlim(0, 1)
    b.legend(fontsize=7.5, loc="lower center"); b.grid(alpha=0.3)

    tf = newest(f"{D}/nat3d_N*_ra{ra}_*_temp.dat")
    e = ax[i][2]
    if tf is not None:
        T = raw(tf)[1:-1, 1:-1]
        H = T.shape[0]
        sc = (np.arange(H) + 0.5) / H
        im = e.imshow(T + 0.5, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                      vmin=0, vmax=1, aspect="equal")
        e.contour(sc, sc, T + 0.5, levels=np.arange(0.1, 1.0, 0.1), colors="k",
                  linewidths=0.6)
        fig.colorbar(im, ax=e, fraction=0.046, pad=0.03)
    e.set_xlabel("$x/L$"); e.set_ylabel("$y/L$")
    e.set_title(f"Ra = {float(ra):.0e}   " r"$\theta$ on $z=L/2$, "
                r"isotherms at $\Delta\theta$=0.1", fontsize=10)

fig.suptitle("Zhou et al. (2026) Sec. 3.6 -- cubic cavity, M3LB D3Q27 central "
             "moments + D3Q7 BGK.  Circles and squares are peaks read off the\n"
             "paper's Figs. 27 and 29, where Fusegi et al. (1991) appear as "
             "symbols; treat them as good to about 5 per cent.", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.94 if len(ras) > 1 else 0.88))
fig.savefig(out, dpi=150)
print("wrote", out)
