#!/usr/bin/env python3
"""Emit the De Rosis & Enan interface-capture table as LaTeX.

Reads results/L_enan/enan_interface.dat, which carries one row per (case,
operator, Pe). The operator is in the lat column: d3q27cm is the central-moment
collision, anything else is BGK on the named lattice. Written as a generator
rather than transcribed by hand so the document cannot drift from the data.
"""
import collections, os, sys

SRC = "results/L_enan/enan_interface.dat"
COLS = "case lat L0 R xi U0 Pe M omega cycles steps e ref_fd phi_min phi_max mass_drift".split()
# case -> (their table, a human label)
TAB = {
    "translate":   ("I",   "Diagonal translation, their III.A"),
    "translate41": ("II",  "Diagonal translation, Geier setup"),
    "zalesak":     ("III", "Zalesak disk"),
    "shear2d":     ("IV",  "Circular interface in shear"),
    "smooth2d":    ("V",   "Circular interface, smoothed shear"),
    "sphere3d":    ("VI",  "Spherical interface in shear"),
    "swirl3d":     ("VII", "Swirling deformation of a sphere"),
}
ORDER = ["translate41", "translate", "zalesak", "shear2d", "smooth2d", "sphere3d", "swirl3d"]

def num(x):
    try: return float(x)
    except ValueError: return float("nan")

rows = collections.defaultdict(dict)   # (case, Pe) -> {op: row}
for line in open(SRC):
    if line.startswith("#") or not line.strip(): continue
    r = dict(zip(COLS, line.split()))
    op = "cm" if r["lat"].endswith("cm") else "bgk"
    rows[(r["case"], round(num(r["Pe"]), 6))][op] = r

def fmt(r, key="e"):
    if r is None: return "---"
    v = num(r[key])
    if v != v: return r"\textit{diverged}"
    return f"{v:.4f}"

out = [r"\begin{tabular}{llrrrr}", r"\toprule",
       r"Case & Table & $Pe$ & $\omega_\phi$ & CM & Theirs \\", r"\midrule"]
last = None
for case in ORDER:
    keys = sorted([k for k in rows if k[0] == case], key=lambda k: k[1])
    for k in keys:
        d = rows[k]
        r = d.get("cm") or d.get("bgk")
        if r is None: continue
        tab, label = TAB[case]
        name = label if case != last else ""
        last = case
        out.append(f"{name} & {tab if name else ''} & {k[1]:.0f} & "
                   f"{num(r['omega']):.3f} & {fmt(r)} & "
                   f"{num(r['ref_fd']):.4f} " + r"\\")
out += [r"\bottomrule", r"\end{tabular}"]
sys.stdout.write("\n".join(out) + "\n")

# ---------------------------------------------------------------------------
# Second table, written to a separate file: the CM/BGK ratio against run
# length. Generated rather than transcribed for the same reason as the first,
# and kept separate because it makes a weaker claim -- it is the evidence that
# no single variable orders the two operators.
# ---------------------------------------------------------------------------
if len(sys.argv) > 1 and sys.argv[1] == "--ratio":
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Case & Steps & BGK & CM & CM/BGK \\", r"\midrule"]
    got = []
    for case in ORDER:
        for k in sorted([k for k in rows if k[0] == case], key=lambda k: k[1]):
            d = rows[k]
            if "bgk" not in d or "cm" not in d: continue
            b, c = num(d["bgk"]["e"]), num(d["cm"]["e"])
            if b != b or c != c or b == 0: continue
            got.append((int(num(d["cm"]["steps"])), TAB[case][1], b, c, c / b))
    for steps, label, b, c, r in sorted(got):
        st = f"{steps:,}".replace(",", r"\,")
        lines.append(f"{label} & {st} & {b:.4f} & {c:.4f} & "
                     f"{r:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    sys.stdout.write("\n".join(lines) + "\n")
