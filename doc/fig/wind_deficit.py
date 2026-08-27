#!/usr/bin/env python3
"""Convert a wind-speed field into a WAKE DEFICIT field, for 3D rendering.

Speed itself renders badly as isosurfaces. vol_urban bands three Gaussians in
log space a decade apart, and urban wind speed spans a factor of a few, not
decades -- the three shells collapse onto each other. Worse, shells of speed
wrap the FAST air and leave the wakes as holes, which is backwards: the wake is
the structure of interest.

The deficit inverts both problems. Subtracting the undisturbed logarithmic
profile at each height leaves ~0 in free stream and several m/s inside a wake,
so a shell at a given deficit IS a wake envelope, and the field spans from
round-off to metres per second -- decades, which is what the renderer wants.

  deficit(x,y,z) = max(0, u_log(z) - |u(x,y,z)|)

Solid cells keep the -1 sentinel so the renderer masks them identically.

  usage: wind_deficit.py <wind.vtk> <out.vtk> <nx> <ny> <nz>
                         [--ref plane|log] [--dx 5] [--ustar 0.855]
                         [--z0 1.0] [--kappa 0.41]
"""
import array, math, struct, sys


def main():
    a = sys.argv[1:]
    def opt(n, d): return float(a[a.index(n) + 1]) if n in a else d
    src, out = a[0], a[1]
    nx, ny, nz = int(a[2]), int(a[3]), int(a[4])
    dx    = opt('--dx', 5.0)
    ustar = opt('--ustar', 0.855)
    z0    = opt('--z0', 1.0)
    kappa = opt('--kappa', 0.41)

    raw = open(src, 'rb').read()
    key = b'LOOKUP_TABLE default\n'
    pos = raw.index(key) + len(key)
    head = raw[:pos]
    F = array.array('f'); F.frombytes(raw[pos:][:nx * ny * nz * 4]); F.byteswap()

    # REFERENCE: the log profile, or the plane mean at each height.
    #
    # The log profile is the obvious choice and it is the wrong one. Every cell
    # inside the canopy is slower than the undisturbed profile -- that is what a
    # city does -- so the deficit is large everywhere below roof height and the
    # render is one opaque slab with no structure in it. Referencing the MEAN AT
    # THAT HEIGHT instead asks a different and more useful question: which cells
    # are slow compared with what is normal at their own level. Wakes stand out;
    # the canopy-wide slowdown, which is real but is not structure, cancels.
    ref = 'plane'
    if '--ref' in a: ref = a[a.index('--ref') + 1]
    if ref == 'log':
        prof = [ustar / kappa * math.log(((k + 0.5) * dx + z0) / z0)
                for k in range(nz)]
    elif ref == 'plane':
        prof = []
        for k in range(nz):
            base = k * ny; tot = 0.0; n = 0
            for y in range(ny):
                row = (base + y) * nx
                for x in range(nx):
                    v = F[row + x]
                    if v >= -0.5: tot += v; n += 1
            prof.append(tot / n if n else 0.0)
    else:
        sys.exit("wind_deficit: --ref takes 'plane' or 'log'")
    peak = 0.0
    for k in range(nz):
        u = prof[k]
        base = k * ny
        for y in range(ny):
            row = (base + y) * nx
            for x in range(nx):
                v = F[row + x]
                if v < -0.5:
                    continue                      # building: keep the sentinel
                d = u - v
                F[row + x] = d if d > 0.0 else 0.0
                if d > peak: peak = d

    F.byteswap()
    # The header carries the field NAME; rewrite it so a reader cannot mistake
    # a deficit for a speed.
    head = head.replace(b'SCALARS speed float 1', b'SCALARS deficit float 1')
    open(out, 'wb').write(head + F.tobytes())
    print("  %s  ref=%s  peak deficit %.3f m/s  (reference at top %.2f m/s)"
          % (out.split('/')[-1], ref, peak, prof[-1]))


main()
