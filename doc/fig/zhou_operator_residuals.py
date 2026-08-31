#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python has
# neither. python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
"""The paper's Figs. 13 and 14 claim: a central-moments collision keeps driving
the residual down where BGK stalls. This is that claim, re-measured.

One case (Sec. 3.2 natural convection), one resolution, one u_c; the D3Q7 BGK
temperature field is common to both runs, so the only difference is the FLUID
collision operator. Reads the -v residual history from the driver's stdout.

Usage: zhou_operator_residuals.py out.png label=file.txt [label=file.txt ...]
"""
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
runs = [a.split("=", 1) for a in sys.argv[2:]]

LINE = re.compile(r"t = (\d+)\s+res ([\d.eE+-]+)\s+drift ([\d.eE+-]+)"
                  r"\s+Nu_hot ([\d.eE+-]+)")
# BGK is drawn wide and pale UNDERNEATH, CM thin and dark on top, because the
# two histories coincide: a same-width dashed line simply vanishes under the
# solid one and the reader cannot tell whether it was plotted at all.
style = {"bgk": ("tab:orange", "-", 5.0, 0.55), "cm": ("tab:blue", "-", 1.5, 1.0),
         "mrt": ("tab:green", ":", 1.5, 1.0)}

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0))

for label, path in runs:
    op, ra = label.split("@")
    t, res, nu = [], [], []
    for ln in open(path):
        m = LINE.search(ln)
        if m:
            t.append(int(m.group(1))); res.append(float(m.group(2)))
            nu.append(float(m.group(4)))
    if not t:
        continue
    c, ls, lw, al = style.get(op, ("k", "-", 1.5, 1.0))
    a = ax[0] if ra == "1e5" else ax[1]
    dev = 100.0 * (nu[-1] / (4.519 if ra == "1e5" else 8.800) - 1.0)
    a.semilogy(np.array(t) / 1000.0, res, ls, color=c, lw=lw, alpha=al,
               label=f"{op.upper()}   Nu = {nu[-1]:.4f} ({dev:+.2f}%),"
                     f" {t[-1]//1000}k steps")

for a, ra, davis in ((ax[0], "1e5", 4.519), (ax[1], "1e6", 8.800)):
    a.text(0.03, 0.06, "the two histories coincide: neither operator has a\n"
                       "residual floor above round-off in this solver",
           transform=a.transAxes, fontsize=8.5, color="0.3")
    a.set_xlabel("time step / 1000")
    a.set_ylabel(r"$\|q(t{+}\Delta) - q(t)\|_2 / \|q(t{+}\Delta)\|_2$")
    a.set_title(f"Ra = {ra}   whole-field interval residual\n"
                f"(de Vahl Davis $\\overline{{Nu}}$ = {davis})", fontsize=10)
    a.legend(fontsize=9); a.grid(alpha=0.3, which="both")

# N is read from the run rather than written into the caption, because the first
# version of this script carried a resolution the runs had not used.
N = "?"
for _, path in runs:
    for ln in open(path):
        if "(H =" in ln:
            N = ln.split("N = ")[1].split(" ")[0]
            break
    if N != "?":
        break
fig.suptitle("Zhou et al. (2026) Figs. 13-14 re-measured: fluid collision "
             "operator, everything else identical.\nM3LB D3Q27 fluid + D3Q7 BGK "
             f"temperature, N = {N}, $u_c$ = 0.1, natural convection cavity, "
             "drift tolerance $10^{-13}$.", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(out, dpi=150)
print("wrote", out)
