#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. This script needs numpy and matplotlib, which the SYSTEM
# python on the development machine does not have -- that is why the older
# doc/fig scripts are pure-python-plus-zlib. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/cavity_ghia.py ...
"""Lid-driven cavity: LBM against Ghia, Ghia & Shin (1982) Tables I and II.

Reads results/C_cavity/cavity_re<Re>_<lat>_<op>.dat, whose columns are
  y  u/U(LBM)  x  v/U(LBM)  u/U(Ghia)  v/U(Ghia)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
runs = sys.argv[2:]                      # "Re:path" pairs

cols = {"100": "tab:blue", "400": "tab:orange", "1000": "tab:red",
        "3200": "tab:green"}
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.9))

summary = []
for spec in runs:
    parts = spec.split(":")
    re_, path = parts[0], parts[1]
    full = parts[2] if len(parts) > 2 else None
    d = np.loadtxt(path)
    y, u, x, v, ug, vg = (d[:, i] for i in range(6))
    c = cols.get(re_, "k")
    if full:                      # smooth curve from all N points
        F = np.loadtxt(full)
        ax[0].plot(F[:, 1], F[:, 0], "-", color=c, lw=1.9, label=f"LBM, Re = {re_}")
        ax[1].plot(F[:, 0], F[:, 2], "-", color=c, lw=1.9, label=f"LBM, Re = {re_}")
    else:
        ax[0].plot(u, y, "-", color=c, lw=1.9, label=f"LBM, Re = {re_}")
        ax[1].plot(x, v, "-", color=c, lw=1.9, label=f"LBM, Re = {re_}")
    ax[0].plot(ug, y, "o", color=c, ms=7, mfc="none", mew=1.7)
    # Ghia's PUBLISHED v at Re = 400, x = 0.9063 is anomalous -- his own extremum
    # table gives min v = -0.44993 at x = 0.8594, and this station then recovers
    # almost the whole way in one interval and goes flat. Every other station at
    # Re = 400 agrees to 0.0072 or better; this one is off by 0.14, a factor of
    # twenty. It is plotted, because it is what the paper prints, but marked and
    # left out of the score.
    ANOM = 11 if re_ == "400" else None
    keep = np.ones_like(x, dtype=bool)
    if ANOM is not None:
        keep[ANOM] = False
        ax[1].plot(x[ANOM], vg[ANOM], "x", color=c, ms=11, mew=2.4, zorder=5)
        ax[1].annotate("published\nanomaly", (x[ANOM], vg[ANOM]),
                       textcoords="offset points", xytext=(30, 12),
                       fontsize=8.5, color=c, ha="center")
    ax[1].plot(x[keep], vg[keep], "o", color=c, ms=7, mfc="none", mew=1.7)
    du = np.abs(u - ug); dv = np.abs(v - vg)
    dvk = dv[keep]; xk = x[keep]
    summary.append((re_, du.max(), y[du.argmax()], dvk.max(), xk[dvk.argmax()]))

ax[0].plot([], [], "ko", ms=7, mfc="none", mew=1.7, label="Ghia et al. (1982)")
ax[0].set_xlabel(r"$u/U_{lid}$  on the vertical centreline  $x = 0.5$")
ax[0].set_ylabel("y")
ax[0].set_title("Table I:  u along the vertical centreline", fontsize=12.5)
ax[0].set_ylim(0, 1); ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=9.5, loc="upper left", framealpha=0.95)

ax[1].axhline(0, color="0.6", lw=0.9)
ax[1].set_xlabel("x")
ax[1].set_ylabel(r"$v/U_{lid}$  on the horizontal centreline  $y = 0.5$")
ax[1].set_title("Table II:  v along the horizontal centreline", fontsize=12.5)
ax[1].set_xlim(0, 1); ax[1].grid(alpha=0.3)
ax[1].legend(fontsize=9.5, loc="lower left", framealpha=0.95)

fig.suptitle("Lid-driven cavity, D3Q27 with one point in $z$, central moments, "
             r"$129\times129$,  $u_{lid}=0.02$", fontsize=13.5, y=0.99)

lines = ["Re      max|Δu|   at y      max|Δv|   at x     (Δv excludes Ghia's",
         "                                                  anomalous Re=400 point)"]
for r, du, ay, dv, axx in summary:
    lines.append(f"{r:>5}   {du:7.4f}   {ay:5.3f}    {dv:7.4f}   {axx:5.3f}")
fig.text(0.5, -0.045, "\n".join(lines), ha="center", va="top",
         family="monospace", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="0.7"))
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)
for r, du, ay, dv, axx in summary:
    print(f"  Re={r:>5}: max|du|={du:.5f} at y={ay:.4f}   max|dv|={dv:.5f} at x={axx:.4f}")
