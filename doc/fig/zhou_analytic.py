#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib, which the SYSTEM python on
# the development machine does not have -- that is why the older doc/fig scripts
# are pure-python-plus-zlib. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/zhou_analytic.py out.png
"""Zhou et al. (2026) Sections 3.1 and 3.4: the two cases with exact solutions.

Left  -- conjugate conduction, four conductivity ratios, against the piecewise
         linear steady solution. The kink at x/L = 0.2 is the material interface.
Right -- the normal-plate advection-diffusion profile at Pe = 1, 10, 100 against
         exp(Pe y)/(exp(Pe)-1), plus the grid-convergence inset at Pe = 100.

Reads results/J_zhou_thermal/{cond,plate}_*.dat: x/L, theta_num, theta_ana.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "doc/fig/zhou_analytic.png"
D = "results/J_zhou_thermal"

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(15.6, 4.9))

# ---------------------------------------------------------------- 3.1
kcol = {"0.1": "tab:purple", "1": "tab:red", "10": "tab:green", "100": "tab:blue"}
for k in ("0.1", "1", "10", "100"):
    f = sorted(glob.glob(f"{D}/cond_L*_k{k}.dat"))
    if not f:
        continue
    d = np.loadtxt(f[-1])
    x, num, ana = d[:, 0], d[:, 1], d[:, 2]
    err = np.linalg.norm(num - ana) / np.linalg.norm(ana) * 100.0
    ax[0].plot(x, num, "-", color=kcol[k], lw=1.8,
               label=rf"$\kappa$ = {k}   ($\epsilon$ = {err:.1e} %)")
    ax[0].plot(x[::12], ana[::12], "o", color=kcol[k], ms=5, mfc="none")
ax[0].axvline(0.2, color="0.55", ls=":", lw=1.2)
ax[0].text(0.205, 0.06, "solid | fluid", color="0.35", fontsize=9)
ax[0].set_xlabel("$x/L$")
ax[0].set_ylabel(r"$\theta$")
ax[0].set_title("3.1  conjugate conduction, $L_s$ = 0.2$L$\n"
                "lines: D3Q7 BGK, per-node $\\omega$   circles: exact",
                fontsize=10)
ax[0].legend(fontsize=8.5, loc="upper right")
ax[0].grid(alpha=0.3)
ax[0].set_xlim(0, 1)

# ---------------------------------------------------------------- 3.4
pcol = {"1": "tab:blue", "10": "tab:green", "100": "tab:red"}
eps = {}
for pe in ("1", "10", "100"):
    f = sorted(glob.glob(f"{D}/plate_H*_Pe{pe}_mom.dat"))
    if not f:
        continue
    d = np.loadtxt(f[-1])
    y, num, ana = d[:, 0], d[:, 1], d[:, 2]
    eps[pe] = np.linalg.norm(num - ana) / np.linalg.norm(ana) * 100.0
    ax[1].plot(num, y, "-", color=pcol[pe], lw=1.8,
               label=rf"Pe = {pe}   ($\epsilon$ = {eps[pe]:.2e} %)")
    ax[1].plot(ana[::10], y[::10], "o", color=pcol[pe], ms=5, mfc="none")
ax[1].set_ylabel("$y/H$")
ax[1].set_xlabel(r"$\theta$")
ax[1].set_title("3.4  normal plate velocity, $H$ = 200\n"
                "lines: D3Q7 BGK, moment walls   circles: Eq. (44)", fontsize=10)
ax[1].legend(fontsize=8.5, loc="upper left")
ax[1].grid(alpha=0.3)
ax[1].set_ylim(0, 1)

# At Pe = 100 the whole profile lives in the top 5 % of the layer -- two cells
# of boundary layer at H = 200 -- so it needs its own scale to be visible at all.
f = sorted(glob.glob(f"{D}/plate_H*_Pe100_mom.dat"))
if f:
    d = np.loadtxt(f[-1])
    ins = ax[1].inset_axes([0.46, 0.10, 0.50, 0.36])
    m = d[:, 0] > 0.94
    ins.plot(d[m, 1], d[m, 0], "-", color="tab:red", lw=1.6)
    ins.plot(d[m, 2], d[m, 0], "o", color="tab:red", ms=4, mfc="none")
    ins.set_title("Pe = 100, top 6 %", fontsize=8, pad=2)
    ins.set_xlabel(r"$\theta$", fontsize=8, labelpad=1)
    ins.set_ylabel("$y/H$", fontsize=8, labelpad=1)
    ins.tick_params(labelsize=7)
    ins.grid(alpha=0.3)

# ---------------------------------------------------------------- 3.4 ladder
# From the driver's -n ladder at Pe = 100 (results/J_zhou_thermal/plate.dat).
lad_H = np.array([100, 200, 400, 800, 1600], float)
lad_a = np.array([9.3534, 2.3313, 0.5824, 0.1456, 0.0364])
lad_m = np.array([11.2667, 1.9708, 0.4218, 0.0983, 0.0238])
oa = np.polyfit(np.log(lad_H[1:]), np.log(lad_a[1:]), 1)[0]
om = np.polyfit(np.log(lad_H[1:]), np.log(lad_m[1:]), 1)[0]
ax[2].loglog(lad_H, lad_a, "s-", ms=6, lw=1.4, color="0.35",
             label=f"anti-bounce-back   (order {-oa:.2f})")
ax[2].loglog(lad_H, lad_m, "o-", ms=6, lw=1.4, color="tab:red",
             label=f"Dellar moment   (order {-om:.2f})")
ax[2].loglog(lad_H, lad_m[1] * (lad_H / lad_H[1]) ** -2.0, "k:", lw=1.1,
             label="$H^{-2}$")
ax[2].axhline(3.65, color="tab:purple", ls="-.", lw=1.2)
ax[2].text(1.05e2, 2.55, "Zhou et al. Table 9,\ncoupled at $H$ = 200: 3.65 %",
           fontsize=8, color="tab:purple")
ax[2].set_xlabel("$H$")
ax[2].set_ylabel(r"$\epsilon$ [%]")
ax[2].set_title("3.4  grid convergence at Pe = 100\n"
                "the error is boundary-layer resolution, not $u$", fontsize=10)
ax[2].legend(fontsize=8.5, loc="lower left")
ax[2].grid(alpha=0.3, which="both")

fig.suptitle("Zhou, De Rosis & Revell (2026) Sec. 3.1 and 3.4 -- M3LB D3Q7 BGK "
             "temperature field against the exact solutions", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out, dpi=150)
print("wrote", out)
