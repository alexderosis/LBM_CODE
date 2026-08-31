#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. This script needs numpy and matplotlib, which the SYSTEM
# python on the development machine does not have -- that is why the older
# doc/fig scripts are pure-python-plus-zlib. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/channel_convergence.py ...
"""Parse converge.log, fit orders, and plot.

Two fits per ladder:
  * least squares on log(err) vs log(N)  -> a single fitted rate
  * err = A/N^2 + B                      -> separates the second-order part from
    the NON-REFINING floor B, which is the number that says whether the ladder
    is measuring the scheme or measuring the Mach error.
"""
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = open(sys.argv[1]).read()
out = sys.argv[2] if len(sys.argv) > 2 else "converge.png"

ladders = []
cur = None
for line in log.splitlines():
    if line.startswith("--- "):
        cur = {"label": line[4:].strip(), "N": [], "err": [], "tau": [], "u0": []}
        ladders.append(cur)
        continue
    if cur is None:
        continue
    t = line.split()
    if len(t) == 8 and re.fullmatch(r"\d+", t[0]):
        try:
            cur["N"].append(int(t[0])); cur["u0"].append(float(t[1]))
            cur["tau"].append(float(t[3])); cur["err"].append(float(t[4]))
        except ValueError:
            pass

ladders = [L for L in ladders if len(L["N"]) >= 3]

print(f"{'ladder':<44} {'LSQ rate':>9} {'A':>11} {'floor B':>11} {'B/err_fine':>11}")
print("-" * 92)
for L in ladders:
    N = np.array(L["N"], float); e = np.array(L["err"], float)
    rate = -np.polyfit(np.log(N), np.log(e), 1)[0]
    # err = A/N^2 + B, linear least squares in (A, B)
    M = np.vstack([1.0 / N**2, np.ones_like(N)]).T
    (A, B), *_ = np.linalg.lstsq(M, e, rcond=None)
    tag = L["label"].split("  ")[0][:43]
    print(f"{tag:<44} {rate:9.3f} {A:11.4e} {B:11.3e} {B/e[-1]:11.2f}")

fig, ax = plt.subplots(figsize=(8.2, 6.0))
cols = plt.cm.tab10(np.linspace(0, 1, 10))
for k, L in enumerate(ladders):
    N = np.array(L["N"], float); e = np.array(L["err"], float)
    ax.loglog(N, e, "o-", color=cols[k % 10], lw=1.6, ms=6,
              label=L["label"].split("  ")[0])
n0 = np.array([8.0, 80.0])
for p, sty in ((1, "--"), (2, ":")):
    ref = 3e-2 * (n0 / n0[0]) ** (-p)
    ax.loglog(n0, ref, sty, color="0.35", lw=1.3)
    ax.text(n0[1], ref[1], f"  slope $-{p}$", color="0.35", va="center", fontsize=9)
ax.set_xlabel("N   (cells across the channel height H)")
ax.set_ylabel("relative $L_2$ error, mid-section")
ax.set_title("channel3d: grid convergence under two scalings", fontsize=12)
ax.grid(which="both", alpha=0.3)
ax.legend(fontsize=8.5, loc="lower left")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nwrote", out)
