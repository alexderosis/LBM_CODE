#!/usr/bin/env python3
"""Measure a plume instead of describing it.

The collage caption for the Manchester run said the lateral spread was
isotropic and that the streets did not steer it. That was true there and it is
an ASSERTION, not a measurement -- and Manhattan is exactly the geometry where
it might stop being true: a regular grid of 5-cell-wide canyons between 70 m
walls, with the wind aimed along one of the two axes. So measure it.

What is measured, at the last frame:

  * downwind distance s and crosswind offset n, both from the SOURCE along the
    wind's own direction, not along the grid axes -- those coincide for this
    configuration and would hide a real difference;
  * the concentration-weighted crosswind spread sigma_n and vertical centroid,
    binned in s;
  * the same split into a street-level slab and an aloft slab, because that is
    the comparison that separates "confined by the canyon" from "spread by
    diffusion". If the streets channel the plume, sigma_n below the rooflines is
    smaller than sigma_n above them at the same s. If they do not, the two agree.

Pure standard library: no numpy on this machine. 16M cells as array('f') is
64 MB; the loop below is the slow part and takes about a minute.

  usage: plume_stats.py <conc.vtk> --log <run.log> [--roof 60] [--bins 8]
         plume_stats.py ... --brief      one line, for a figure caption
         plume_stats.py ... --levels     three shell levels, by enclosed mass
"""
import array, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urban_overlay import Run, compass


def read_vtk(path, n):
    d = open(path, "rb").read()
    key = b"LOOKUP_TABLE default\n"
    a = array.array("f")
    a.frombytes(d[d.index(key) + len(key):][: n * 4])
    a.byteswap()                       # legacy VTK binary is big-endian
    return a


class Bin:
    __slots__ = ("m", "n", "nn", "z", "zz", "cells")
    def __init__(self): self.m = self.n = self.nn = self.z = self.zz = 0.0; self.cells = 0
    def add(self, w, n, z):
        self.m += w; self.n += w * n; self.nn += w * n * n
        self.z += w * z; self.zz += w * z * z; self.cells += 1
    def sigma_n(self):
        if self.m <= 0: return None
        v = self.nn / self.m - (self.n / self.m) ** 2
        return math.sqrt(max(v, 0.0))
    def mean_z(self):
        return self.z / self.m if self.m > 0 else None


def main():
    argv = sys.argv[1:]
    def opt(name, d=None): return argv[argv.index(name) + 1] if name in argv else d
    vtk = argv[0]
    run = Run(opt("--log")).parse()
    roof = float(opt("--roof", "60"))
    nbins = int(opt("--bins", "8"))

    nx, ny, nz, dx = run.nx, run.ny, run.nz, run.dx
    C = read_vtk(vtk, nx * ny * nz)

    r = math.radians(run.bearing)
    ex, ey = -math.sin(r), -math.cos(r)          # downwind
    px, py = ey, -ex                             # crosswind, right of downwind
    sx, sy = (run.src[0] + 0.5) * dx, (run.src[1] + 0.5) * dx

    # Bin width from the fetch, so the table covers the run whatever it is.
    x, y, fetch = run.src[0] + 0.5, run.src[1] + 0.5, 0
    while 0 <= x < nx and 0 <= y < ny:
        x += ex; y += ey; fetch += 1
    width = fetch * dx / nbins

    low = [Bin() for _ in range(nbins)]
    high = [Bin() for _ in range(nbins)]
    total = 0.0
    neg = 0.0
    for k in range(nz):
        z = (k + 0.5) * dx
        band = low if z < roof else high
        base = k * ny
        for j in range(ny):
            row = (base + j) * nx
            yy = (j + 0.5) * dx - sy
            for i in range(nx):
                c = C[row + i]
                # Solid is written as exactly -1; the undershoot is small and
                # negative and must not be counted as plume, but it IS reported,
                # because a spread computed from a field with negative lobes is
                # only as trustworthy as those lobes are small.
                if c <= 0.0:
                    if c > -0.5: neg -= c
                    continue
                xx = (i + 0.5) * dx - sx
                s = xx * ex + yy * ey
                if s < 0: continue
                b = int(s / width)
                if b >= nbins: continue
                band[b].add(c, xx * px + yy * py, z)
                total += c

    # Transfer-function levels, chosen from the field rather than guessed. The
    # renderer draws three shells; putting them at the concentrations that
    # enclose 95%, 80% and 50% of the plume's mass makes the outer shell mean
    # "almost all of it is inside here" and the inner one "half of it is in
    # here", which is a statement about the release. Levels carried over from
    # another run mean nothing at all: this plume's 50% level moves by an order
    # of magnitude between one minute and fifteen as it spreads.
    if "--levels" in argv:
        v = sorted(c for c in C if c > 1e-12)
        tot = sum(v)
        # Integrating from the highest concentration down, the 50% level is
        # reached FIRST and is the highest of the three; the renderer wants them
        # ascending, so the list is reversed on the way out.
        want, out, run = [0.50, 0.80, 0.95], [], 0.0
        for x in reversed(v):
            run += x
            while want and run >= want[0] * tot:
                out.append(x); want.pop(0)
            if not want: break
        print(",".join(f"{x:.3g}" for x in reversed(out)))
        return

    # One line, for a caption that then cannot disagree with the picture.
    # Averaged over the bins that actually carry street-level material: a ratio
    # from a bin holding 0.1% of the plume below the rooflines is noise, and
    # including it is how a caption ends up quoting a number nothing supports.
    if "--brief" in argv:
        num = den = 0.0
        for b in range(nbins):
            lo, hi = low[b], high[b]
            sl, sh = lo.sigma_n(), hi.sigma_n()
            if not (sl and sh and sh > 0): continue
            if lo.m + hi.m <= 0 or lo.m / (lo.m + hi.m) < 0.05: continue
            num += lo.m * (sl / sh); den += lo.m
        if den <= 0:
            print("no street-level material to measure")
        elif num / den < 0.95:
            print(f"canyon-confined — crosswind spread below the rooflines is "
                  f"{num / den:.2f}\u00d7 that above")
        else:
            print(f"the canyons do not hold it — spread below the rooflines is "
                  f"{num / den:.2f}\u00d7 that above")
        return

    print(f"\n  {os.path.basename(vtk)}   {run.place}")
    print(f"  wind toward {compass(run.bearing - 180)} ({run.bearing - 180:.0f} deg), "
          f"source at ({run.src[0]},{run.src[1]}) z = {run.src_z:g} m, "
          f"fetch {fetch * dx / 1000:.2f} km")
    print(f"  roofline split at {roof:g} m;  undershoot mass "
          f"{neg:.3e} = {100 * neg / max(total, 1e-30):.2f}% of the plume\n")
    print(f"  {'s (m)':>10} {'street sig_n':>13} {'aloft sig_n':>12} {'ratio':>7}"
          f" {'street <z>':>11} {'aloft <z>':>10} {'street frac':>12}")
    print("  " + "-" * 80)
    for b in range(nbins):
        lo, hi = low[b], high[b]
        sl, sh = lo.sigma_n(), hi.sigma_n()
        if lo.m + hi.m <= 0: continue
        ratio = f"{sl / sh:.2f}" if (sl and sh and sh > 0) else "  -"
        print(f"  {(b + 0.5) * width:10.0f} "
              f"{(f'{sl:.1f}' if sl else '-'):>13} {(f'{sh:.1f}' if sh else '-'):>12} "
              f"{ratio:>7} {(f'{lo.mean_z():.1f}' if lo.m else '-'):>11} "
              f"{(f'{hi.mean_z():.1f}' if hi.m else '-'):>10} "
              f"{100 * lo.m / (lo.m + hi.m):11.1f}%")
    print("\n  sig_n is the concentration-weighted crosswind standard deviation about\n"
          "  the plume's own axis. A ratio below 1 means the plume is NARROWER below\n"
          "  the rooflines than above them -- the canyons holding it in. A ratio at or\n"
          "  above 1 means they are not.\n")


if __name__ == "__main__":
    main()
