#!/usr/bin/env python3
# NOTE ON DEPENDENCIES. This script needs numpy and matplotlib, which the SYSTEM
# python on the development machine does not have -- that is why the older
# doc/fig scripts are pure-python-plus-zlib. A throwaway venv is enough:
#     python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib
#     /tmp/v/bin/python doc/fig/channel_transient.py ...
"""Animate the start-up transient of the inlet-driven channel.

Frames come from validation/channel3d run with -fevery; each holds the mid
z-plane as rows of (ux_lat, rho), y outer and x inner, with a header giving the
step, the physical time, the grid and the lattice->m/s scale.
"""
import glob, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

pat, out = sys.argv[1], sys.argv[2]
H, L, UMAX = 1.0, 5.0, 1.0e-5          # m, m, m/s
T_DIFF_S = 1.3e4 * 76.9230769          # diffusive time in seconds

files = sorted(glob.glob(pat + "_frame*.dat"))
if not files:
    sys.exit("no frames matched " + pat)

frames = []
for fn in files:
    hdr = open(fn).readline()
    g = dict(re.findall(r"(\w+)=([-\d.e+]+)", hdr))
    nx, ny = int(g["nx"]), int(g["ny"])
    d = np.loadtxt(fn)
    frames.append(dict(t=float(g["t_s"]), step=int(g["step"]),
                       ux=d[:, 0].reshape(ny, nx) * float(g["uscale"]),
                       nx=nx, ny=ny, dx=float(g["dx"])))

nx, ny, dx = frames[0]["nx"], frames[0]["ny"], frames[0]["dx"]
# Cell centres: with halfway bounce-back the walls sit half a cell outside the
# first and last fluid rows, so the fluid spans dx/2 .. H-dx/2.
yc = (np.arange(ny) + 0.5) * dx
xc = (np.arange(nx)) * dx
vmax = UMAX * 1.06

fig = plt.figure(figsize=(10.2, 7.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.42)

axc = fig.add_subplot(gs[0])
im = axc.imshow(frames[0]["ux"], origin="lower", aspect="equal",
                extent=[0, L, 0, H], cmap="viridis", vmin=0, vmax=vmax,
                interpolation="bilinear")
axc.set_xlabel("x  [m]"); axc.set_ylabel("y  [m]")
cb = fig.colorbar(im, ax=axc, pad=0.012, fraction=0.024)
cb.set_label(r"$u_x$  [m/s]")
axc.axvline(L / 2, color="crimson", ls="--", lw=1.1)
axc.text(L / 2 + 0.06, 0.5, "mid-section", color="crimson", fontsize=8.5, va="center")
ttl = axc.set_title("", fontsize=11)

axp = fig.add_subplot(gs[1])
ys = np.linspace(0, H, 400)
axp.plot(UMAX * 4.0 * (ys / H) * (1 - ys / H), ys, "-", color="0.25", lw=2.2,
         label="analytical, fully developed", zorder=2)
(ln,) = axp.plot([], [], "o-", color="crimson", ms=5.5, lw=1.4, mfc="none", mew=1.5,
                 label="LBM, mid-section", zorder=3)
axp.set_xlim(-0.04 * UMAX, 1.12 * UMAX); axp.set_ylim(0, H)
axp.set_xlabel(r"$u_x$ at $x = 2.5$ m  [m/s]"); axp.set_ylabel("y  [m]")
axp.grid(alpha=0.3); axp.legend(fontsize=9, loc="lower right", framealpha=0.95)
axp.axhline(0, color="0.6", lw=2.5); axp.axhline(H, color="0.6", lw=2.5)

fig.suptitle("Start-up transient, plane Poiseuille flow of water\n"
             r"5 $\times$ 1 $\times$ 1 m,  Re $= U_{max}H/\nu = 10$,  "
             r"$U_{max}=10^{-5}$ m/s,  D3Q27 central moments,  66$\times$15$\times$13",
             fontsize=11.5, y=1.005)

def upd(k):
    f = frames[k]
    im.set_data(f["ux"])
    ln.set_data(f["ux"][:, nx // 2], yc)
    ttl.set_text(f"t = {f['t']/86400:.1f} days     "
                 f"$t/t_{{diff}}$ = {f['t']/T_DIFF_S:.2f}     step {f['step']}")
    return im, ln, ttl

ani = FuncAnimation(fig, upd, frames=len(frames), blit=False, interval=100)
try:
    ani.save(out, writer=FFMpegWriter(fps=10, bitrate=2600))
    print("wrote", out)
except Exception as e:
    print("ffmpeg failed:", e)
gifout = out.rsplit(".", 1)[0] + ".gif"
ani.save(gifout, writer=PillowWriter(fps=10))
print("wrote", gifout)
print(f"{len(frames)} frames, t = 0 .. {frames[-1]['t']/86400:.1f} days "
      f"({frames[-1]['t']/T_DIFF_S:.2f} diffusive times)")
