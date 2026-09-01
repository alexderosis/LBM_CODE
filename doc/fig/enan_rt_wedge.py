#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python on the
# development machine has neither. A throwaway venv is enough:
#     python3 -m venv /tmp/v3 && /tmp/v3/bin/pip install numpy matplotlib
#     /tmp/v3/bin/python doc/fig/enan_rt_wedge.py fig/enan_rt_wedge.png
"""Rayleigh-Taylor and wedge entry against De Rosis & Enan Sec. III.G-III.I.

Top left     the 2-D spike trajectory at three Reynolds numbers against their
             Table VIII. This is the pass.
Top right    the two 3-D cases against their Tables IX and X, with the band
             their Table IX collects from eight published models. This is the
             failure, and the band is drawn so that being outside it is visible
             rather than asserted.
Bottom left  wedge entry: F* against tau at three deadrise angles, with
             Wagner's straight line for each and the fit window shaded. Wagner
             is asymptotic in small phi, so the lines should be approached from
             either side as phi falls, not matched exactly.
Bottom right the drop-height artefact. Same impact, same window; released 20
             cells above the surface the force collapses mid-window because the
             pulse from the impulsive release comes back around the periodic
             domain, and at 60 cells it does not.

Reads results/L_enan/enan_rt.dat and results/M_wedge/wedge_phi*.dat, plus the
two drop-height series recorded in enan_wedge.cpp's banner.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "enan_rt_wedge.png"
TSTAR = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
RT_COLS = ("dim W Re At Ca Pe xi nu tau sigma M t_ref steps finite "
           "y0 y05 y1 y15 y2 y25 y3").split()
YS = ["y0", "y05", "y1", "y15", "y2", "y25", "y3"]

T8   = [1.900, 1.829, 1.620, 1.365, 1.118, 0.863, 0.575]
T9   = [1.898, 1.858, 1.741, 1.553, 1.304, 1.001, 0.648]
T10  = [1.898, 1.848, 1.680, 1.384, 0.964, 0.436, 0.000]
T9LO = [1.887, 1.839, 1.711, 1.504, 1.256, 0.988, 0.648]
T9HI = [1.904, 1.897, 1.776, 1.618, 1.396, 1.149, 0.863]

rt = {}
for line in open("results/L_enan/enan_rt.dat"):
    if line.startswith("#") or not line.strip():
        continue
    r = dict(zip(RT_COLS, line.split()))
    rt[(r["dim"], float(r["W"]), float(r["Re"]), float(r["Ca"]))] = \
        [float(r[y]) for y in YS]

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))

# ---- (a) 2-D ---------------------------------------------------------------
a = ax[0, 0]
for re, col, mk in ((256.0, "#3b6ea5", "o"), (3000.0, "#4a9a4a", "^"),
                    (30000.0, "#a53b4a", "s")):
    y = rt.get(("2", 256.0, re, 0.26))
    if y:
        a.plot(TSTAR, y, mk + "-", color=col, ms=5, lw=1.3,
               label=f"$Re={re:g}$")
a.plot(TSTAR, T8, "k--", lw=1.6, label="their Table VIII\n($Re=30\\,000$)")
a.set_xlabel(r"$t/t_0$"); a.set_ylabel(r"spike position $y^\dagger = y/W$")
a.set_title("(a) two dimensions, $W=256$: worst deviation 2.7 per cent", fontsize=10)
a.legend(fontsize=8); a.grid(alpha=0.3)

# ---- (b) 3-D, with the eight-model band -----------------------------------
a = ax[0, 1]
a.fill_between(TSTAR, T9LO, T9HI, color="#cccccc", alpha=0.75,
               label="their Table IX: eight\npublished models")
a.plot(TSTAR, T9,  "k--", lw=1.6, label="their Table IX ($Re=256$)")
a.plot(TSTAR, T10, "k:",  lw=1.6, label="their Table X ($Re=30\\,000$)")
for re, col, mk in ((256.0, "#3b6ea5", "o"), (30000.0, "#a53b4a", "s")):
    y = rt.get(("3", 64.0, re, 960.0))
    if y:
        a.plot(TSTAR, y, mk + "-", color=col, ms=5, lw=1.3,
               label=f"here, $Re={re:g}$")
a.set_xlabel(r"$t/t_0$"); a.set_ylabel(r"$y^\dagger$")
a.set_title("(b) three dimensions, $W=64$: outside the band from $t/t_0=1$",
            fontsize=10)
a.legend(fontsize=7); a.grid(alpha=0.3)

# ---- (c) wedge, F* against tau -------------------------------------------
a = ax[1, 0]
for deg, col, mk in ((10, "#3b6ea5", "o"), (20, "#a53b4a", "s"),
                     (30, "#4a9a4a", "^")):
    f = f"results/M_wedge/wedge_phi{deg}_ratio100.dat"
    if not os.path.exists(f):
        continue
    d = np.loadtxt(f)
    tv = 2.0 * np.tan(np.radians(deg)) / np.pi
    wag = np.pi ** 3 / (8.0 * np.tan(np.radians(deg)) ** 2)
    keep = d[:, 0] <= 1.05 * tv
    a.plot(d[keep, 0] / tv, d[keep, 1], mk + "-", color=col, ms=3.5, lw=1.1,
           label=f"${deg}^\\circ$")
    t = np.linspace(0, 1.05, 20)
    a.plot(t, wag * t * tv, "--", color=col, lw=1.1, alpha=0.65)
a.axvspan(0.25, 0.70, color="#f0e0a0", alpha=0.45, zorder=0)
a.text(0.475, 0.4, "fit window", fontsize=7.5, ha="center", color="#8a7020")
a.set_xlabel(r"$\tau/\tau_{\rm valid}$,   $\tau_{\rm valid}=2\tan\varphi/\pi$")
a.set_ylabel(r"$F^* = F/(\rho_H v^2 b)$")
a.set_title("(c) wedge entry; dashed is Wagner for each angle", fontsize=10)
a.legend(fontsize=8, title="deadrise", title_fontsize=8); a.grid(alpha=0.3)
a.set_ylim(0, 12)

# ---- (d) the drop-height artefact ----------------------------------------
# Recorded in enan_wedge.cpp's banner: same impact, only the release height
# differs, and the short drop leaves the startup pulse inside the fit window.
a = ax[1, 1]
tv20 = 2.0 * np.tan(np.radians(20.0)) / np.pi
for D, col, mk, lab in ((20, "#a53b4a", "s", "20 cells: slope $6.51$, $R^2\\,0.03$"),
                        (60, "#3b6ea5", "o", "60 cells: slope $29.55$, $R^2\\,0.997$")):
    f = f"/private/tmp/claude-501/-Users-alessandroderosis-Desktop-M3LB/0161c0d6-6071-4b55-b22c-a6315ebb30db/scratchpad/runs/wedge_drop{D}.dat"
    if not os.path.exists(f):
        continue
    d = np.loadtxt(f)
    keep = d[:, 0] <= 1.05 * tv20
    a.plot(d[keep, 0], d[keep, 1], mk + "-", color=col, ms=3.5, lw=1.2, label=lab)
wag20 = np.pi ** 3 / (8.0 * np.tan(np.radians(20.0)) ** 2)
t = np.linspace(0, 1.05 * tv20, 20)
a.plot(t, wag20 * t, "k--", lw=1.5, label="Wagner")
a.axvspan(0.25 * tv20, 0.70 * tv20, color="#f0e0a0", alpha=0.45, zorder=0)
a.set_xlabel(r"$\tau$"); a.set_ylabel(r"$F^*$")
a.set_title("(d) release height: an artefact that reads as physics", fontsize=10)
a.legend(fontsize=7.5); a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
