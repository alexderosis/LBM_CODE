#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. This script needs numpy and matplotlib, which the SYSTEM
# python on the development machine does not have -- that is why the older
# doc/fig scripts are pure-python-plus-zlib. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/channel_validation.py ...
"""Analytical vs numerical figure for the inlet-driven channel, for a report."""
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tag, out = sys.argv[1], sys.argv[2]
prof = np.loadtxt(tag + "_profile.dat")
cl   = np.loadtxt(tag + "_centreline.dat")

# H from symmetry of the sampled stations (eta_first + eta_last = 1); U from the
# analytic column, which agrees at every station.
y   = prof[:, 0]
H   = y[0] + y[-1]
et  = y / H
U   = float(np.median(prof[:, 3] / (4.0 * et * (1.0 - et))))
un  = prof[:, 2]                      # numerical, m/s
ua  = prof[:, 3]                      # analytical at the same station, m/s
err = (un - ua) / U * 100.0

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
fig = plt.figure(figsize=(12.4, 7.4))
gs  = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.5, 1.0],
                       hspace=0.42, wspace=0.26)

# --- profile ---------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ys = np.linspace(0, H, 400)
ax.plot(U * 4.0 * (ys / H) * (1 - ys / H), ys, "-", color="0.2", lw=2.4,
        label=r"analytical  $u_x = 4U\,\eta(1-\eta)$", zorder=2)
ax.plot(un, y, "o", ms=8, mfc="none", mec="crimson", mew=1.9,
        label="LBM (D3Q27, central moments)", zorder=3)
ax.axhline(0, color="0.55", lw=3); ax.axhline(H, color="0.55", lw=3)
ax.text(0.985 * U, 0.035, "wall", color="0.4", fontsize=9, ha="right")
ax.set_xlabel(r"$u_x$  at the mid-section $x = 2.5$ m   [m/s]")
ax.set_ylabel("y   [m]")
ax.set_title("Fully developed velocity profile", fontsize=12.5)
ax.set_ylim(-0.02 * H, 1.02 * H)
ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="center left", framealpha=0.96)

# --- error -----------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.plot(err, y, "s-", color="teal", ms=6, lw=1.5)
ax.axvline(0, color="0.4", lw=1.0)
ax.set_xlabel(r"($u_{LBM}-u_{exact}$) / $U$   [%]")
ax.set_ylabel("y   [m]")
ax.set_title("Pointwise error", fontsize=12.5)
ax.set_ylim(-0.02 * H, 1.02 * H); ax.grid(alpha=0.3)

# --- centreline pressure ---------------------------------------------------
ax = fig.add_subplot(gs[1, :])
p = cl[:, 3] - cl[:, 3][cl.shape[0] // 2]
co = np.polyfit(cl[2:-2, 0], p[2:-2], 1)
ax.plot(cl[:, 0], p, "o", ms=4.2, color="navy", label="LBM centreline")
ax.plot(cl[:, 0], np.polyval(co, cl[:, 0]), "--", color="darkorange", lw=1.8,
        label=f"linear fit:  d$p$/d$x$ = {co[0]:.4g} Pa/m")
ax.set_xlabel("x   [m]"); ax.set_ylabel(r"$p - p_{mid}$   [Pa]")
ax.set_title("Streamwise pressure along the channel centreline", fontsize=12.5)
ax.grid(alpha=0.3); ax.legend(fontsize=10)

# --- numbers ---------------------------------------------------------------
nu, rho = 1.0e-6, 1000.0
dpdx_ex = -8.0 * rho * nu * U / H**2
L2 = float(np.sqrt(np.sum((un - ua)**2) / np.sum(ua**2)))
txt = (f"water:  $\\nu$ = {nu:.3g} m²/s,  $\\rho$ = {rho:.0f} kg/m³\n"
       f"box 5 × 1 × 1 m,  grid 66 × 15 × 13,  $\\Delta x$ = {H/ (len(y)):.4f} m\n"
       f"$U_{{max}}$ = {U:.4g} m/s,  Re = $U H/\\nu$ = {U*H/nu:.0f},  "
       f"$u_{{lat}}$ = 0.01,  Ma = 0.017,  $\\tau$ = 0.539\n"
       f"walls: halfway bounce-back    inlet: parabolic    outlet: back-pressure    z: periodic\n"
       f"relative $L_2$ error = {L2:.3e}      peak error = "
       f"{100*(un.max()-ua.max())/U:+.3f} %\n"
       f"d$p$/d$x$: {co[0]:.5g} vs {dpdx_ex:.5g} Pa/m exact  "
       f"({100*(co[0]-dpdx_ex)/dpdx_ex:+.2f} %)")
fig.text(0.5, 0.012, txt, ha="center", va="top", fontsize=9.6, family="monospace",
         bbox=dict(boxstyle="round,pad=0.55", fc="#f4f4f4", ec="0.7"))

fig.suptitle("Validation: plane Poiseuille flow of water, inlet driven", fontsize=14, y=0.995)
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)
print(f"  U={U:.6g} m/s  H={H:.6g} m  Re={U*H/nu:.1f}")
print(f"  rel L2 = {L2:.6e}   peak err = {100*(un.max()-ua.max())/U:+.4f}%")
print(f"  dp/dx  = {co[0]:.6g} vs {dpdx_ex:.6g} Pa/m  ({100*(co[0]-dpdx_ex)/dpdx_ex:+.3f}%)")
