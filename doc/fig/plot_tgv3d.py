"""Three-panel summary of a 3D Taylor-Green run: energy, enstrophy and the
dissipation rate, from the .dat that tgv3d writes into results/E_tgv3d/.

    python3 doc/fig/plot_tgv3d.py results/E_tgv3d/<file>.dat doc/fig/out.png

REQUIRES numpy AND matplotlib, unlike everything else in this directory.
mkpng.py, vol3d.py and make_anim.py are deliberately dependency-free -- pure
Python plus zlib -- because the target machine has no numpy. This one is not:
line plots with real axes and labelled ticks are not worth hand-rolling, and
this figure is a convenience rather than part of the document build. If you have
no matplotlib, one is available in the aorta project's venv.

NORMALISATION. t* here is t*u0/D, the code's own convention. The literature
normalises by L = D/2pi (the box is 2*pi*L), so the upper axis shows
t*u0/L = 2*pi*t*, which is what to compare against published Taylor-Green data.
The dissipation rate is -d(E/E0)/dt*: for incompressible flow it is proportional
to the enstrophy integral, so the two right-hand panels should peak together --
a useful internal check on the solver.
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.loadtxt(sys.argv[1])
t, E, Z = d[:, 0], d[:, 1], d[:, 2]
TWOPI = 2 * np.pi                      # t_std = 2*pi*t*  (see note below)
dEdt = -np.gradient(E, t)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a in ax:
    a.grid(alpha=.3); a.set_xlabel(r"$t^{*}=t\,u_0/D$")
    sec = a.secondary_xaxis("top", functions=(lambda x: x * TWOPI,
                                              lambda x: x / TWOPI))
    sec.set_xlabel(r"$t\,u_0/L$   (literature normalisation, $L=D/2\pi$)")

ax[0].semilogy(t, E, "o-", color="#1f4e9c", mfc="none", lw=1.8)
ax[0].set_ylabel(r"$E/E_0$"); ax[0].set_title("kinetic energy")

ax[1].plot(t, Z, "o-", color="#c1121f", mfc="none", lw=1.8)
ax[1].set_ylabel(r"$\Psi/\Psi_0$"); ax[1].set_title("enstrophy")
k = int(np.argmax(Z))
ax[1].axvline(t[k], color="#c1121f", ls="--", lw=1, alpha=.7)
ax[1].annotate(r"peak $t^{*}=%.2f$" "\n" r"$(t\,u_0/L=%.1f)$" % (t[k], t[k]*TWOPI),
               (t[k], Z[k]), textcoords="offset points", xytext=(10, -6),
               color="#c1121f", fontsize=9)

ax[2].plot(t, dEdt, "o-", color="#0b7a3b", mfc="none", lw=1.8)
ax[2].set_ylabel(r"$-\,\mathrm{d}(E/E_0)/\mathrm{d}t^{*}$")
ax[2].set_title("dissipation rate")
j = int(np.argmax(dEdt))
ax[2].axvline(t[j], color="#0b7a3b", ls="--", lw=1, alpha=.7)
ax[2].annotate(r"peak $t^{*}=%.2f$" "\n" r"$(t\,u_0/L=%.1f)$" % (t[j], t[j]*TWOPI),
               (t[j], dEdt[j]), textcoords="offset points", xytext=(10, -6),
               color="#0b7a3b", fontsize=9)

fig.suptitle(r"3D Taylor$-$Green vortex $\cdot$ D3Q27 central moments $\cdot$ "
             r"$\mathrm{Re}=1600$, $64^3$, $\tau=0.502400$, $\mathrm{Ma}=0.035$",
             y=1.06, fontsize=12)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=150, bbox_inches="tight")
print("  final E/E0 = %.5f at t*=%.1f" % (E[-1], t[-1]))
print("  enstrophy peak   %.4f  at t* = %.2f  (t u0/L = %.2f)" % (Z[k], t[k], t[k]*TWOPI))
print("  dissipation peak %.5f  at t* = %.2f  (t u0/L = %.2f)" % (dEdt[j], t[j], t[j]*TWOPI))
