#!/usr/bin/env python3
"""Emit the De Rosis & Enan Rayleigh-Taylor tables as LaTeX.

Reads results/L_enan/enan_rt.dat. Written as a generator rather than
transcribed by hand so the document cannot drift from the data, the same way
enan_interface_table.py is.

  (no argument)  the tabulated cases against their Tables VIII, IX and X
  --resolution   the W ladder at Re = 30000, and the Ca pair at each W
"""
import sys

SRC = "results/L_enan/enan_rt.dat"
COLS = ("dim W Re At Ca Pe xi nu tau sigma M t_ref steps finite "
        "y0 y05 y1 y15 y2 y25 y3").split()
YS = ["y0", "y05", "y1", "y15", "y2", "y25", "y3"]
TSTAR = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Their tabulated columns.
REF = {
    ("2", 30000.0): ("VIII", [1.900, 1.829, 1.620, 1.365, 1.118, 0.863, 0.575]),
    ("3", 256.0):   ("IX",   [1.898, 1.858, 1.741, 1.553, 1.304, 1.001, 0.648]),
    ("3", 30000.0): ("X",    [1.898, 1.848, 1.680, 1.384, 0.964, 0.436, 0.000]),
}

rows = []
for line in open(SRC):
    if line.startswith("#") or not line.strip():
        continue
    r = dict(zip(COLS, line.split()))
    rows.append(r)

def key(r):
    return (r["dim"], float(r["W"]), float(r["Re"]), float(r["Ca"]))

uniq = {}
for r in rows:
    uniq[key(r)] = r                      # last run of a configuration wins

def worst(r, ref):
    w = 0.0
    for i, k in enumerate(YS):
        if abs(ref[i]) < 1e-9:            # Table X ends at exactly zero
            continue
        w = max(w, abs(100.0 * (float(r[k]) / ref[i] - 1.0)))
    return w

if len(sys.argv) > 1 and sys.argv[1] == "--resolution":
    out = [r"\begin{tabular}{rrrrl}", r"\toprule",
           r"$W$ & $Ca$ & Worst dev. & Dev. at $t/t_0=3$ & \\", r"\midrule"]
    sel = [(64, 0.26), (128, 0.26), (256, 0.26), (64, 960.0), (256, 960.0)]
    for W, Ca in sel:
        r = uniq.get(("2", float(W), 30000.0, Ca))
        if r is None:
            continue
        ref = REF[("2", 30000.0)][1]
        last = 100.0 * (float(r["y3"]) / ref[6] - 1.0)
        note = "" if Ca == 0.26 else r"\textit{near-zero $\sigma$}"
        out.append(f"{W} & {Ca:g} & {worst(r, ref):.1f}\\% & "
                   f"{last:+.1f}\\% & {note} " + r"\\")
        if W == 256 and Ca == 0.26:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    sys.stdout.write("\n".join(out) + "\n")
    raise SystemExit

out = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
       r"Case & Table & " + " & ".join(f"$t/t_0={t:g}$" for t in TSTAR)
       + r" & Worst \\", r"\midrule"]
order = [("2", 256.0, 30000.0, 0.26), ("3", 64.0, 256.0, 960.0),
         ("3", 64.0, 30000.0, 960.0)]
for k in order:
    r = uniq.get(k)
    if r is None:
        continue
    tab, ref = REF[(k[0], k[2])]
    name = f"{k[0]}-D, $Re={k[2]:g}$"
    out.append(f"{name} & {tab} & "
               + " & ".join(f"{float(r[y]):.3f}" for y in YS)
               + f" & {worst(r, ref):.1f}\\% " + r"\\")
    out.append(r"\quad theirs & & "
               + " & ".join(f"\\textit{{{v:.3f}}}" for v in ref) + r" & \\")
out += [r"\bottomrule", r"\end{tabular}"]
sys.stdout.write("\n".join(out) + "\n")
