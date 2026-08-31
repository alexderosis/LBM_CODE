#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. This script needs numpy and matplotlib, which the SYSTEM
# python on the development machine does not have -- that is why the older
# doc/fig scripts are pure-python-plus-zlib. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/tgv2d_convergence.py ...
"""Parse tgv.log, fit convergence rates, and plot."""
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = open(sys.argv[1]).read()
out = sys.argv[2] if len(sys.argv) > 2 else "tgv.png"

runs, cur = [], None
for line in log.splitlines():
    m = re.match(r"\s*III\.A Taylor-Green vortex\s+lattice (\S+)\s+operator (\S+)", line)
    if m:
        cur = {"label": f"{m.group(1)}, {m.group(2)}", "N": [], "err": [], "nue": [], "nui": []}
        runs.append(cur); continue
    if cur is None: continue
    t = line.split()
    # N tau steps L2 order [nu_imp nu_fit err%]
    if len(t) >= 5 and re.fullmatch(r"\d+", t[0]):
        try:
            cur["N"].append(int(t[0])); cur["err"].append(float(t[3]))
            if len(t) >= 8:
                cur["nui"].append(float(t[5])); cur["nue"].append(float(t[6]))
        except (ValueError, IndexError):
            pass

runs = [r for r in runs if len(r["N"]) >= 3]
print(f"{'run':<22} {'pts':>4} {'LSQ rate':>9} {'finest err':>12} {'nu err @ finest':>16}")
print("-" * 70)
for r in runs:
    N = np.array(r["N"], float); e = np.array(r["err"], float)
    rate = -np.polyfit(np.log(N), np.log(e), 1)[0]
    nue = ""
    if r["nue"]:
        nue = f"{100*(r['nue'][-1]-r['nui'][-1])/r['nui'][-1]:+.4f}%"
    print(f"{r['label']:<22} {len(N):4d} {rate:9.3f} {e[-1]:12.4e} {nue:>16}")

fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.4))
cols = plt.cm.tab10(np.linspace(0, 1, 10))
for k, r in enumerate(runs):
    N = np.array(r["N"], float); e = np.array(r["err"], float)
    ax[0].loglog(N, e, "o-", color=cols[k], lw=1.7, ms=6.5, label=r["label"])
    if r["nue"]:
        Nn = np.array(r["N"][:len(r["nue"])], float)
        rel = np.abs((np.array(r["nue"]) - np.array(r["nui"])) / np.array(r["nui"]))
        ax[1].loglog(Nn, rel, "s-", color=cols[k], lw=1.7, ms=6, label=r["label"])
n0 = np.array([7.0, 300.0])
for a, anchor in ((ax[0], 8.0), (ax[1], 2.5e-2)):
    ref = anchor * (n0 / n0[0]) ** (-2.0)
    a.loglog(n0, ref, ":", color="0.35", lw=1.4)
    a.text(n0[1], ref[1], "  slope $-2$", color="0.35", va="center", fontsize=9)
ax[0].set_xlabel("N   (points per side)")
ax[0].set_ylabel(r"relative $L_2$ error of $(u,v)$ at $t=T$")
ax[0].set_title("Convergence to the analytical Taylor-Green field", fontsize=12)
ax[1].set_xlabel("N   (points per side)")
ax[1].set_ylabel(r"$|\nu_{fit}-\nu_{imposed}|\,/\,\nu_{imposed}$")
ax[1].set_title(r"Effective viscosity from the energy decay", fontsize=12)
for a in ax:
    a.grid(which="both", alpha=0.3); a.legend(fontsize=9, loc="lower left")
fig.suptitle(r"Taylor-Green vortex decay,  Re $=1000$,  $u_0=0.02$,  one point in $z$",
             fontsize=13, y=1.01)
fig.savefig(out, dpi=160, bbox_inches="tight")
print("\nwrote", out)
