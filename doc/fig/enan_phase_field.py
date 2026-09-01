#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python on the
# development machine has neither. A throwaway venv is enough:
#     python3 -m venv /tmp/v3 && /tmp/v3/bin/pip install numpy matplotlib
#     /tmp/v3/bin/python doc/fig/enan_phase_field.py fig/enan_phase_field.png
"""The phase-field campaign against De Rosis & Enan, Phys. Fluids 33, 043315
(2021), Sec. III -- interface capture and the two collision operators.

Top left     the six interface-capture cases, BGK and central moments against
             the paper's finite-difference column. Log scale, because the cases
             span an order of magnitude in error.
Top right    the two Peclet sweeps, their Tables I and III. The point of the
             panel is the three Zalesak rows where BGK reaches 1e300 and the
             central-moment operator does not.
Bottom left  the cycle scan: BGK's translation error saturates within one cycle
             and the central-moment error compounds. This is the measurement
             that killed "the crossover is in omega".
Bottom right the mobility ladder at fixed run length, which is what made the
             omega story look right in the first place.

Reads results/L_enan/enan_interface.dat. The two lower panels are measurements
recorded in the operator banner rather than in that file, because they are
diagnostics at Peclet numbers no table covers and the driver deliberately
refuses to append those.
"""
import sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "enan_phase_field.png"
SRC = "results/L_enan/enan_interface.dat"
COLS = ("case lat L0 R xi U0 Pe M omega cycles steps e ref_fd "
        "phi_min phi_max mass_drift").split()

rows = []
for line in open(SRC):
    if line.startswith("#") or not line.strip():
        continue
    rows.append(dict(zip(COLS, line.split())))

def num(x):
    try:    return float(x)
    except ValueError: return float("nan")

def get(case, op, pe=None):
    for r in rows:
        is_cm = r["lat"].endswith("cm")
        if r["case"] != case:            continue
        if (op == "cm") != is_cm:        continue
        if pe is not None and abs(num(r["Pe"]) - pe) > 1e-6: continue
        return r
    return None

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))

# ---- (a) the six cases ------------------------------------------------------
CASES = [("translate41", "Translation\n(Table II)", 60),
         ("zalesak",     "Zalesak\n(III)",          80),
         ("shear2d",     "Shear 2-D\n(IV)",         60),
         ("smooth2d",    "Smoothed\n(V)",           40),
         ("sphere3d",    "Sphere 3-D\n(VI)",        60),
         ("swirl3d",     "Swirl 3-D\n(VII)",        60)]
x = np.arange(len(CASES)); w = 0.27
bgk = [num(get(c, "bgk", p)["e"]) if get(c, "bgk", p) else np.nan for c, _, p in CASES]
cm  = [num(get(c, "cm",  p)["e"]) if get(c, "cm",  p) else np.nan for c, _, p in CASES]
ref = [num(get(c, "bgk", p)["ref_fd"]) if get(c, "bgk", p) else np.nan
       for c, _, p in CASES]
a = ax[0, 0]
a.bar(x - w, bgk, w, label="M3LB BGK", color="#3b6ea5")
a.bar(x,     cm,  w, label="M3LB central moments", color="#a53b4a")
a.bar(x + w, ref, w, label="their FD column", color="#999999")
a.set_xticks(x); a.set_xticklabels([n for _, n, _ in CASES], fontsize=7.5)
a.set_yscale("log"); a.set_ylabel(r"relative error $e$, their Eq. (71)")
a.set_title("(a) interface capture, their III.A--III.F", fontsize=10)
a.legend(fontsize=8); a.grid(axis="y", alpha=0.3, which="both")

# ---- (b) the two Peclet sweeps ---------------------------------------------
a = ax[0, 1]
for case, style, name in (("translate", "o-", "Translation, Table I"),
                          ("zalesak",   "s-", "Zalesak, Table III")):
    for op, col in (("bgk", "#3b6ea5"), ("cm", "#a53b4a")):
        pts = sorted([(num(r["Pe"]), num(r["e"]))
                      for r in rows
                      if r["case"] == case
                      and (r["lat"].endswith("cm")) == (op == "cm")])
        if not pts: continue
        pe = [p for p, _ in pts]; er = [e for _, e in pts]
        good = [(p, e) for p, e in zip(pe, er) if e == e]
        bad  = [p for p, e in zip(pe, er) if e != e]
        if good:
            a.plot([p for p, _ in good], [e for _, e in good], style, color=col,
                   ms=5, lw=1.2,
                   label=f"{name}, {'CM' if op=='cm' else 'BGK'}")
        for p in bad:                                  # 1e300: mark the ceiling
            a.plot([p], [0.5], "x", color=col, ms=11, mew=2.2)
    pts = sorted({(num(r["Pe"]), num(r["ref_fd"])) for r in rows
                  if r["case"] == case})
    a.plot([p for p, _ in pts], [e for _, e in pts], "--", color="#999999",
           lw=1.2, label=f"{name}, theirs")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$Pe = U_0\xi/M$"); a.set_ylabel(r"$e$")
a.set_title("(b) the Peclet sweeps; $\\times$ is a run that reached $10^{300}$",
            fontsize=10)
a.legend(fontsize=6.6, ncol=2); a.grid(alpha=0.3, which="both")

# ---- (c) the cycle scan ----------------------------------------------------
# Measured with -cycles on their Table II translation; see the operator banner.
cyc      = [1, 2, 5, 10]
cyc_bgk  = [0.00814, 0.00803, 0.00804, 0.00817]
cyc_cm   = [0.01593, 0.01671, 0.02276, 0.03968]
a = ax[1, 0]
a.plot(cyc, cyc_bgk, "o-", color="#3b6ea5", label="BGK")
a.plot(cyc, cyc_cm,  "s-", color="#a53b4a", label="central moments")
a.set_xlabel("cycles (one cycle is 5000 steps)"); a.set_ylabel(r"$e$")
a.set_title("(c) BGK saturates, CM compounds", fontsize=10)
a.legend(fontsize=8); a.grid(alpha=0.3); a.set_ylim(0, 0.045)

# ---- (d) the mobility ladder ----------------------------------------------
mob      = [0.2, 0.05, 0.001]
om       = [0.909, 1.539, 1.988]
lad_bgk  = [0.0609, 0.0097, 0.0082]
lad_cm   = [0.0112, 0.0066, 0.0397]
a = ax[1, 1]
a.plot(om, lad_bgk, "o-", color="#3b6ea5", label="BGK")
a.plot(om, lad_cm,  "s-", color="#a53b4a", label="central moments")
for o, m in zip(om, mob):
    a.annotate(f"$M={m:g}$", (o, 0.0025), fontsize=7, ha="center",
               color="#555555")
a.axvline(2.0, color="#cc0000", ls=":", lw=1.2)
a.text(1.985, 0.05, "BGK stability edge", rotation=90, fontsize=7,
       color="#cc0000", va="top", ha="right")
a.set_xlabel(r"$\omega_\phi$"); a.set_ylabel(r"$e$")
a.set_title("(d) the same at fixed run length, varying $M$", fontsize=10)
a.set_yscale("log"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
