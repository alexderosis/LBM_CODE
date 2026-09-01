#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python on the
# development machine has neither. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/mhd_hartmann3d.py out.png
"""Hartmann flow in 3-D, D3Q27 fluid + D3Q7 magnetic field, against Dellar's
Eq. (14).

Left   the velocity profile at several Hartmann numbers, showing the flat core
       and the wall layers of thickness L/Ha that the field produces.
Right  the induced streamwise field b_x at the same Hartmann numbers.
Bottom the grid-convergence ladder, and the field-free row at two viscosities --
       omega = 1 is where the regularised wall's curvature slip vanishes, and it
       is the row that shows the driver itself is exact.

Reads results/K_hartmann3d/prof_*.dat (y/L, u_num, u_exact, b_num, b_exact) and
hartmann3d.dat for the ladder.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "doc/fig/mhd_hartmann3d.png"
D = "results/K_hartmann3d"

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(13.6, 8.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.34, wspace=0.26)

# ------------------------------------------------------------------ profiles
HA = ["1", "3", "10", "30", "100"]
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(HA)))
ax_u = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
for c, ha in zip(cols, HA):
    f = f"{D}/prof_ha{ha}_ny129.dat"
    if not os.path.exists(f):
        continue
    d = np.loadtxt(f)
    y, un, ue, bn, be = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]
    ax_u.plot(un / ue.max(), y, "-", color=c, lw=1.8, label=f"Ha = {ha}")
    ax_u.plot(ue[::6] / ue.max(), y[::6], "o", color=c, ms=4.5, mfc="none")
    s = np.abs(be).max()
    if s > 0:
        ax_b.plot(bn / s, y, "-", color=c, lw=1.8, label=f"Ha = {ha}")
        ax_b.plot(be[::6] / s, y[::6], "o", color=c, ms=4.5, mfc="none")
for a, lab, ttl in ((ax_u, "$u_x/u_{\\max}$", "velocity"),
                    (ax_b, "$b_x/|b|_{\\max}$", "induced field")):
    a.set_xlabel(lab); a.set_ylabel("$y/L$")
    a.set_title(f"{ttl}, $n_y$ = 129\nlines: D3Q27 + D3Q7   circles: Dellar Eq. (14)",
                fontsize=10)
    a.axhline(1, color="0.6", lw=0.8); a.axhline(-1, color="0.6", lw=0.8)
    a.grid(alpha=0.3); a.legend(fontsize=8.5, loc="best")
    a.set_ylim(-1.05, 1.05)

# ------------------------------------------------------------------ ladders
cols_t, rows = None, []
for ln in open(f"{D}/hartmann3d.dat"):
    if ln.startswith("#"):
        cols_t = ln.lstrip("# ").split(); continue
    if ln.strip():
        rows.append(dict(zip(cols_t, ln.split())))

ax_c = fig.add_subplot(gs[1, 0])
lad = sorted([r for r in rows if float(r["ha"]) == 10.0 and int(r["nx"]) == 8],
             key=lambda r: int(r["ny"]))
if lad:
    n = np.array([int(r["ny"]) - 1 for r in lad], float)
    eu = np.array([float(r["l2u"]) for r in lad])
    eb = np.array([float(r["l2b"]) for r in lad])
    ou = np.polyfit(np.log(n), np.log(eu), 1)[0]
    ob = np.polyfit(np.log(n), np.log(eb), 1)[0]
    ax_c.loglog(n, eu, "o-", ms=6, lw=1.5, color="tab:blue",
                label=f"$u_x$   (order {-ou:.2f})")
    ax_c.loglog(n, eb, "s-", ms=6, lw=1.5, color="tab:red",
                label=f"$b_x$   (order {-ob:.2f})")
    ax_c.loglog(n, eu[0] * (n / n[0]) ** -2.0, "k:", lw=1.1, label="$h^{-2}$")
    for r, x, y in zip(lad, n, eu):
        ax_c.annotate(f"$L/Ha$={float(r['delta']):.1f}", (x, y), fontsize=7,
                      textcoords="offset points", xytext=(4, -11), color="0.35")
ax_c.set_xlabel("$n_y - 1$ (cells across the channel)")
ax_c.set_ylabel("relative $l_2$")
ax_c.set_title("grid convergence at Ha = 10\nthe annotation is the Hartmann "
               "layer in cells", fontsize=10)
ax_c.grid(alpha=0.3, which="both"); ax_c.legend(fontsize=9)

ax_f = fig.add_subplot(gs[1, 1])
for nuv, c, lab in ((0.1, "tab:orange", r"$\nu = 0.1$  ($\omega = 1.25$)"),
                    (1.0 / 6.0, "tab:green", r"$\nu = 1/6$  ($\omega = 1$)")):
    ff = sorted([r for r in rows
                 if float(r["ha"]) == 0.0 and abs(float(r["nu"]) - nuv) < 1e-4],
                key=lambda r: int(r["ny"]))
    if not ff:
        continue
    n = np.array([int(r["ny"]) - 1 for r in ff], float)
    e = np.array([float(r["l2u"]) for r in ff])
    o = np.polyfit(np.log(n), np.log(e), 1)[0]
    ax_f.loglog(n, e, "o-", ms=6, lw=1.5, color=c,
                label=lab + (f"   (order {-o:.2f})" if e.max() > 1e-8 else "   (round-off)"))
ax_f.axhline(1e-10, color="0.6", ls="--", lw=1.0)
ax_f.text(20, 1.4e-10, "round-off", fontsize=8, color="0.4")
ax_f.set_xlabel("$n_y - 1$")
ax_f.set_ylabel("relative $l_2$")
ax_f.set_title("Ha = 0: the field-free parabola\nthe wall slip is "
               r"$\propto(\omega-1)/\omega^2$ and vanishes at $\omega=1$",
               fontsize=10)
ax_f.grid(alpha=0.3, which="both"); ax_f.legend(fontsize=8.5, loc="lower left")

fig.suptitle("Hartmann flow in three dimensions -- M3LB D3Q27 MhdBGK fluid + "
             "D3Q7 MagneticBGK, regularised velocity walls and Dellar's moment\n"
             "condition for $B$, both ON the node.  Periodic along and across "
             "the channel; $B_0$ normal to the walls.", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out, dpi=150)
print("wrote", out)
