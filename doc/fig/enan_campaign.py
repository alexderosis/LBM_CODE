#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python here has
# neither. A throwaway venv is enough:
#     python3 -m venv /tmp/v3 && /tmp/v3/bin/pip install numpy matplotlib
#     /tmp/v3/bin/python doc/fig/enan_campaign.py fig/enan_campaign.png
"""The De Rosis & Enan campaign, central moments, at THEIR parameters.

(a) interface capture, their III.A-III.F, against the published column. Log
    scale: the cases span an order of magnitude in error.
(b) the Zalesak Peclet sweep, with omega marked. Their mobility is
    M = U d / Pe with d the DOMAIN, so omega rises to 1.988 and no further --
    an earlier version of this campaign read Pe as U xi / M, put omega at
    1.9998, and drew conclusions from a regime the paper never visits.
(c) the 2-D Rayleigh-Taylor spike against their Table VIII. The pass.
(d) the two 3-D cases against Tables IX and X, with the band their Table IX
    collects from eight published models.

Reads results/L_enan/enan_interface.dat and enan_rt.dat.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "enan_campaign.png"
IC = ("case lat L0 R xi U0 Pe M omega cycles steps e ref_fd "
      "phi_min phi_max mass_drift").split()
RT = ("dim W Re At Ca Pe xi nu tau sigma M t_ref steps finite "
      "y0 y05 y1 y15 y2 y25 y3").split()
YS = ["y0","y05","y1","y15","y2","y25","y3"]
TS = np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0])
T8 = [1.900,1.829,1.620,1.365,1.118,0.863,0.575]
T9 = [1.898,1.858,1.741,1.553,1.304,1.001,0.648]
T10= [1.898,1.848,1.680,1.384,0.964,0.436,0.000]
LO = [1.887,1.839,1.711,1.504,1.256,0.988,0.648]
HI = [1.904,1.897,1.776,1.618,1.396,1.149,0.863]

def load(path, cols):
    out=[]
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        out.append(dict(zip(cols, line.split())))
    return out
ic = load("results/L_enan/enan_interface.dat", IC)
rt = {}
for r in load("results/L_enan/enan_rt.dat", RT):
    rt[(r["dim"], float(r["Re"]))] = [float(r[y]) for y in YS]

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))

# ---- (a) the cases ---------------------------------------------------------
NAMES = [("translate41","Translation\n(II)"), ("zalesak","Zalesak\n(III)"),
         ("shear2d","Shear 2-D\n(IV)"), ("smooth2d","Smoothed\n(V)"),
         ("sphere3d","Sphere 3-D\n(VI)"), ("swirl3d","Swirl 3-D\n(VII)")]
a = ax[0,0]; x = np.arange(len(NAMES)); wdt = 0.36
mine=[]; theirs=[]
for c,_ in NAMES:
    rows=[r for r in ic if r["case"]==c]
    r = min(rows, key=lambda r: float(r["Pe"]))       # lowest Pe as the entry
    mine.append(float(r["e"])); theirs.append(float(r["ref_fd"]))
a.bar(x-wdt/2, mine,   wdt, label="M3LB central moments", color="#a53b4a")
a.bar(x+wdt/2, theirs, wdt, label="their published column", color="#999999")
# the D3Q7 column their Tables VI and VII also give
a.plot([4,5],[0.0754,0.1726],"kv",ms=6,label="their D3Q7 (VI, VII only)")
a.set_xticks(x); a.set_xticklabels([n for _,n in NAMES], fontsize=7.5)
a.set_yscale("log"); a.set_ylabel(r"relative error $e$, their Eq. (71)")
a.set_title("(a) interface capture at their mobilities", fontsize=10)
a.legend(fontsize=7.5); a.grid(axis="y", alpha=0.3, which="both")

# ---- (b) the Zalesak sweep -------------------------------------------------
a = ax[0,1]
z = sorted([(float(r["Pe"]), float(r["e"]), float(r["ref_fd"]), float(r["omega"]))
            for r in ic if r["case"]=="zalesak"])
pe=[q[0] for q in z]
a.plot(pe,[q[1] for q in z],"s-",color="#a53b4a",ms=6,label="M3LB central moments")
a.plot(pe,[q[2] for q in z],"o--",color="#999999",ms=5,label="theirs")
for p,e,_,om in z:
    a.annotate(rf"$\omega={om:.3f}$",(p,e),textcoords="offset points",
               xytext=(0,9),fontsize=7,ha="center",color="#a53b4a")
a.axhline(0,color="k",lw=0)
a.set_xscale("log"); a.set_xlabel(r"$Pe = U_{\rm ref}\,d/M$  (their definition)")
a.set_ylabel(r"$e$"); a.set_ylim(0.03,0.085)
a.set_title("(b) Zalesak: their ceiling is $\\omega=1.988$", fontsize=10)
a.legend(fontsize=8); a.grid(alpha=0.3)

# ---- (c) 2-D RTI -----------------------------------------------------------
a = ax[1,0]
y = rt.get(("2",30000.0))
if y: a.plot(TS,y,"s-",color="#a53b4a",ms=5,label="M3LB")
a.plot(TS,T8,"k--",lw=1.6,label="their Table VIII")
a.set_xlabel(r"$t/t_0$"); a.set_ylabel(r"spike $y^\dagger=y/W$")
a.set_title("(c) 2-D Rayleigh-Taylor, $Re=30\\,000$: 2.6 per cent worst", fontsize=10)
a.legend(fontsize=8); a.grid(alpha=0.3)

# ---- (d) 3-D RTI -----------------------------------------------------------
a = ax[1,1]
a.fill_between(TS,LO,HI,color="#cccccc",alpha=0.75,
               label="their Table IX:\neight published models")
a.plot(TS,T9,"k--",lw=1.5,label="their Table IX ($Re=256$)")
a.plot(TS,T10,"k:",lw=1.5,label="their Table X ($Re=30\\,000$)")
for re,col,mk in ((256.0,"#3b6ea5","o"),(30000.0,"#a53b4a","s")):
    y = rt.get(("3",re))
    if y: a.plot(TS,y,mk+"-",color=col,ms=5,label=f"M3LB, $Re={re:g}$")
a.set_xlabel(r"$t/t_0$"); a.set_ylabel(r"$y^\dagger$")
a.set_title("(d) 3-D: too slow now, having been too fast", fontsize=10)
a.legend(fontsize=7); a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
