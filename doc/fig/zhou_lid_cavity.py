#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python has
# neither. python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
"""Zhou et al. (2026) Sec. 3.5: thermal lid-driven cavity, heated from below,
Gr = 1e6, Pr = 0.71, Ri = 10, 1, 0.1.

These ARE steady states -- the Nusselt fluctuation over the averaging window is
0.05 to 0.5 per cent of the mean -- but only once the domain-mean temperature has
equilibrated, which takes 1e5 to 1e6 steps and which the field residual cannot
see. Runs stopped earlier look quasi-steady at quite different Nusselt numbers;
see the section text.

The velocity scale is max(U_lid, u_c), which is what the driver writes into the
dumps, because the lid-stopped control has U = 0.

Reads results/J_zhou_thermal/lid_N*_ri<Ri>_<op>_{temp,ux,uy}.dat and, if present,
lid.dat for the time-averaged Nusselt numbers.
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "doc/fig/zhou_lid_cavity.png"
D = "results/J_zhou_thermal"

# Cheng & Liu (2010) and Bettaibi et al. (2014), via Zhou et al. Table 10.
REF = {"10": (4.860, 4.848, 4.588), "1": (5.750, 5.739, 5.402),
       "0.1": (12.161, 12.138, 12.610)}


def raw(path):
    with open(path, "rb") as fh:
        nx, ny = np.fromfile(fh, dtype=np.int32, count=2)
        return np.fromfile(fh, dtype=np.float32, count=nx * ny).reshape(ny, nx)


def newest(pat):
    f = sorted(glob.glob(pat))
    return f[-1] if f else None


# Time-averaged Nusselt from the tracked table. Keyed by (Ri, u_c), not by Ri
# alone: lid.dat also holds the u_c control, which is Ri = 10 at a second u_c,
# and keying on Ri would silently let it overwrite the production row.
PROD_UC = {"10": 0.15, "1": 0.08, "0.1": 0.03}
ours = {}
if os.path.exists(f"{D}/lid.dat"):
    cols = None
    for ln in open(f"{D}/lid.dat"):
        if ln.startswith("#"):
            cols = ln.lstrip("# ").split()
            continue
        if not ln.strip():
            continue
        r = dict(zip(cols, ln.split()))
        ri = r.get("Ri")
        if r.get("lid") != "on" or ri not in PROD_UC:
            continue
        if abs(float(r["uc"]) - PROD_UC[ri]) > 1e-9:
            continue
        ours[ri] = (float(r["Nu_hot"]), float(r["Nu_cold"]), float(r["Nu_pkpk"]))

RIS = ("10", "1", "0.1")
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(2, 3, figsize=(13.4, 8.6))

for j, ri in enumerate(RIS):
    tf = newest(f"{D}/lid_N*_ri{ri}_*_temp.dat")
    if tf is None:
        for i in (0, 1):
            ax[i, j].text(0.5, 0.5, f"Ri = {ri}\nno data", ha="center",
                          va="center", transform=ax[i, j].transAxes)
            ax[i, j].set_axis_off()
        continue
    T = raw(tf)
    U = raw(tf.replace("_temp", "_ux"))
    V = raw(tf.replace("_temp", "_uy"))
    n = T.shape[0]
    s = np.arange(n) / (n - 1.0)              # on-node walls: L = N-1

    a = ax[0, j]
    im = a.imshow(T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                  vmin=0, vmax=1, aspect="equal")
    a.contour(s, s, T, levels=np.arange(0.1, 1.0, 0.1), colors="k", linewidths=0.6)
    ttl = f"Ri = {ri}   (Re = {np.sqrt(1e6/float(ri)):.0f})"
    if ri in ours:
        h, c, pk = ours[ri]
        cl, bt, zh = REF[ri]
        ttl += (f"\n$\\overline{{Nu}}$ = {0.5*(h+c):.2f} $\\pm$ {0.5*abs(h-c):.2f};"
                f"  Cheng {cl}, Zhou {zh}")
    else:
        ttl += r"   $\theta$, isotherms at $\Delta\theta$ = 0.1"
    a.set_title(ttl, fontsize=9.5)
    a.set_xlabel("$x/L$"); a.set_ylabel("$y/L$")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)

    b = ax[1, j]
    spd = np.hypot(U, V)
    im2 = b.imshow(spd, origin="lower", extent=(0, 1, 0, 1), cmap="magma",
                   aspect="equal")
    b.streamplot(s, s, U, V, color="w", density=1.3, linewidth=0.55, arrowsize=0.6)
    b.set_title(f"Ri = {ri}   speed / max($U_{{lid}}$, $u_c$) + streamlines",
                fontsize=9.5)
    b.set_xlabel("$x/L$"); b.set_ylabel("$y/L$")
    fig.colorbar(im2, ax=b, fraction=0.046, pad=0.03)

fig.suptitle("Zhou et al. (2026) Sec. 3.5 -- thermal lid-driven cavity, heated "
             "from below, Gr = 1e6, Pr = 0.71.  M3LB D3Q27 CM + D3Q7 BGK, "
             "on-node wall family.\nNusselt numbers are means over the last 60 "
             "probe intervals; the quoted spread is half the hot/cold gap.  "
             "Compare the streamlines with the paper's Fig. 23.", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out, dpi=150)
print("wrote", out)
