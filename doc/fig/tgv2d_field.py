#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. This script needs numpy and matplotlib, which the SYSTEM
# python on the development machine does not have -- that is why the older
# doc/fig scripts are pure-python-plus-zlib. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/tgv2d_field.py ...
"""Taylor-Green velocity-field snapshot: numerical vs analytical at t = T."""
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base, out = sys.argv[1], sys.argv[2]     # base = ..._N128, out = png

def load(fn):
    h = dict(re.findall(r"(\w+)=([-\d.e+]+)", open(fn).readline()))
    N, xi, u0 = int(h["N"]), float(h["xi"]), float(h["u0"])
    d = np.loadtxt(fn)
    ux = d[:, 2].reshape(N, N); uy = d[:, 3].reshape(N, N)
    return N, xi, u0, ux, uy

N, xi, u0, ux0, uy0 = load(base + "_t0.dat")
_, _, _, uxT, uyT   = load(base + "_tT.dat")

ix = np.arange(N)
X, Y = np.meshgrid(ix, ix, indexing="xy")          # X varies along axis 1
dec  = np.exp(-1.0)                                # t = T
axa  = -u0 * np.cos(xi * X) * np.sin(xi * Y) * dec
aya  =  u0 * np.sin(xi * X) * np.cos(xi * Y) * dec

m0 = np.hypot(ux0, uy0) / u0
mT = np.hypot(uxT, uyT) / u0
ma = np.hypot(axa, aya) / u0
dif = (mT - ma)

plt.rcParams.update({"font.size": 10.5})
fig, axs = plt.subplots(2, 2, figsize=(11.4, 9.6))
ext = [0, 2 * np.pi, 0, 2 * np.pi]

def show(ax, f, ttl, vmax, cmap="viridis", sym=False):
    kw = dict(origin="lower", extent=ext, cmap=cmap, interpolation="bilinear",
              aspect="equal")
    im = ax.imshow(f, vmin=(-vmax if sym else 0), vmax=vmax, **kw)
    ax.set_title(ttl, fontsize=11.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xticks([0, np.pi, 2*np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0, np.pi, 2*np.pi]); ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    return im

im = show(axs[0, 0], m0, r"$|u|/u_0$ at $t=0$   (initial condition)", 1.0)
fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.02)
# streamlines on the numerical field at t = T
im = show(axs[0, 1], mT, r"$|u|/u_0$ at $t=T$   LBM", 1.0)
xs = np.linspace(0, 2*np.pi, N)
axs[0, 1].streamplot(xs, xs, uxT, uyT, color="w", density=0.85, linewidth=0.6,
                     arrowsize=0.7)
fig.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.02)

im = show(axs[1, 0], ma, r"$|u|/u_0$ at $t=T$   analytical", 1.0)
fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.02)

v = float(np.abs(dif).max())
im = show(axs[1, 1], dif, r"($|u|_{LBM} - |u|_{exact})/u_0$", v, cmap="RdBu_r", sym=True)
fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.02)

L2 = float(np.sqrt(np.sum((uxT-axa)**2 + (uyT-aya)**2) /
                   np.sum(axa**2 + aya**2)))
fig.suptitle("Taylor-Green vortex decay on D3Q27 (one point in $z$), "
             f"$N={N}$, Re$\\,=1000$, $u_0={u0}$\n"
             f"amplitude at $t=T$: $e^{{-1}}$ = {dec:.5f} exact,  "
             f"{mT.max():.5f} measured    |    relative $L_2$ = {L2:.3e}    "
             f"max local deviation = {v:.2e}$\\,u_0$",
             fontsize=12, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
print(f"  peak |u|/u0 at t=T: measured {mT.max():.6f}, exact {dec:.6f} "
      f"({100*(mT.max()-dec)/dec:+.3f}%)")
print(f"  relative L2 = {L2:.6e}   max local dev = {v:.4e} u0")
