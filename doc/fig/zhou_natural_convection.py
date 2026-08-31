#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python on the
# development machine has neither. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/zhou_natural_convection.py out.png
"""Zhou et al. (2026) Sec. 3.2: natural convection in a differentially heated
square cavity at Ra = 1e5 and 1e6, against de Vahl Davis (1983).

Row 1  isotherms over the temperature field, and streamlines over the speed.
Row 2  the two centreline velocity profiles and the local Nusselt number on the
       hot wall, with de Vahl Davis's tabulated peaks marked.

Reads results/J_zhou_thermal/nat2d_N*_ra<Ra>_<op>_{temp,ux,uy}.dat (figdump raw:
int32 nx, int32 ny, then nx*ny float32, x fastest) and *_centrelines.dat.
"""
import glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "doc/fig/zhou_natural_convection.png"
D = "results/J_zhou_thermal"

def raw(path):
    with open(path, "rb") as fh:
        nx, ny = np.fromfile(fh, dtype=np.int32, count=2)
        return np.fromfile(fh, dtype=np.float32, count=nx * ny).reshape(ny, nx)

def newest(pat):
    f = sorted(glob.glob(pat))
    return f[-1] if f else None

# de Vahl Davis (1983), Richardson-extrapolated benchmark
DAVIS = {"1e+05": dict(nu=4.519, umax=34.73, yu=0.855, vmax=68.59, xv=0.066,
                       numax=7.717, ynumax=0.081),
         "1e+06": dict(nu=8.800, umax=64.63, yu=0.850, vmax=219.36, xv=0.0379,
                       numax=17.925, ynumax=0.0378)}

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(14.2, 8.0))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.30, wspace=0.42)

for col, ra in enumerate(("1e+05", "1e+06")):
    tf = newest(f"{D}/nat2d_N*_ra{ra}_*_temp.dat")
    cf = newest(f"{D}/nat2d_N*_ra{ra}_*_centrelines.dat")
    if tf is None:
        continue
    T = raw(tf)
    U = raw(tf.replace("_temp", "_ux"))
    V = raw(tf.replace("_temp", "_uy"))
    ny, nx = T.shape
    # interior only: row/column 0 and -1 are the wall nodes
    Ti, Ui, Vi = T[1:-1, 1:-1], U[1:-1, 1:-1], V[1:-1, 1:-1]
    H = Ti.shape[0]
    s = (np.arange(H) + 0.5) / H

    a = fig.add_subplot(gs[0, 2 * col])
    im = a.imshow(Ti, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                  vmin=-0.5, vmax=0.5, aspect="equal")
    a.contour(s, s, Ti, levels=np.linspace(-0.45, 0.45, 10), colors="k",
              linewidths=0.6)
    a.set_title(f"Ra = {float(ra):.0e}   $\\theta$ + isotherms", fontsize=10)
    a.set_xlabel("$x/L$"); a.set_ylabel("$y/L$")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)

    b = fig.add_subplot(gs[0, 2 * col + 1])
    spd = np.hypot(Ui, Vi)
    im2 = b.imshow(spd, origin="lower", extent=(0, 1, 0, 1), cmap="viridis",
                   aspect="equal")
    b.streamplot(s, s, Ui, Vi, color="w", density=1.05, linewidth=0.55,
                 arrowsize=0.6)
    b.set_title(f"Ra = {float(ra):.0e}   $|u|H/\\alpha$ + streamlines", fontsize=10)
    b.set_xlabel("$x/L$"); b.set_ylabel("$y/L$")
    fig.colorbar(im2, ax=b, fraction=0.046, pad=0.03)

    if cf is None:
        continue
    d = np.loadtxt(cf)
    sc, ux, uy, nul = d[:, 0], d[:, 3], d[:, 4], d[:, 5]
    dv = DAVIS[ra]

    # Both curves against POSITION on the shared abscissa: u_x sampled down the
    # vertical centreline, u_y across the horizontal one. Plotting one of them
    # transposed puts velocity on one axis and position on the other in the same
    # frame, which is what the first version of this script did.
    c = fig.add_subplot(gs[1, 2 * col])
    c.plot(sc, ux, "-", lw=1.6, color="tab:blue",
           label="$u_xH/\\alpha$ vs $y/L$ at $x/L$=0.5")
    c.plot(sc, uy, "-", lw=1.6, color="tab:red",
           label="$u_yH/\\alpha$ vs $x/L$ at $y/L$=0.5")
    c.plot([dv["yu"]], [dv["umax"]], "o", mfc="none", mec="tab:blue", ms=9,
           mew=1.6, label=f"Davis $u_{{max}}$ = {dv['umax']} @ {dv['yu']}")
    c.plot([dv["xv"]], [dv["vmax"]], "s", mfc="none", mec="tab:red", ms=9,
           mew=1.6, label=f"Davis $v_{{max}}$ = {dv['vmax']} @ {dv['xv']}")
    c.axhline(0, color="0.7", lw=0.8)
    c.set_xlabel("position along the centreline")
    c.set_ylabel("velocity $\\times H/\\alpha$")
    c.set_xlim(0, 1)
    c.set_title(f"Ra = {float(ra):.0e}   centreline velocities", fontsize=10)
    c.legend(fontsize=7.5, loc="best"); c.grid(alpha=0.3)

    e = fig.add_subplot(gs[1, 2 * col + 1])
    e.plot(nul, sc, "-", lw=1.7, color="tab:green")
    e.plot([dv["numax"]], [dv["ynumax"]], "o", mfc="none", mec="k", ms=9, mew=1.6)
    e.annotate(f"Davis $Nu_{{max}}$ = {dv['numax']}\nat $y/L$ = {dv['ynumax']}",
               xy=(dv["numax"], dv["ynumax"]), xytext=(0.30, 0.30),
               textcoords="axes fraction", fontsize=7.5,
               arrowprops=dict(arrowstyle="->", lw=0.8))
    e.axvline(nul.mean(), color="0.4", ls="--", lw=1.0)
    e.text(nul.mean(), 0.63, f"  $\\overline{{Nu}}$ = {nul.mean():.4f}\n"
           f"  (Davis {dv['nu']})", fontsize=8, color="0.25")
    e.set_xlabel("local $Nu$ on the hot wall"); e.set_ylabel("$y/L$")
    e.set_title(f"Ra = {float(ra):.0e}   local Nusselt", fontsize=10)
    e.grid(alpha=0.3); e.set_ylim(0, 1)

fig.suptitle("Zhou et al. (2026) Sec. 3.2 -- M3LB D3Q27 central moments + D3Q7 "
             "BGK, midway wall family, against de Vahl Davis (1983)", fontsize=11)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
