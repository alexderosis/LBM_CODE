#!/usr/bin/env python3
"""Horizontal slice of a scalar field written by demonstrator/urban.

For the WIND this is the right plot and a volume render is not. The interesting
structure in an unsteady urban wind is the wake: a low-speed region shed behind
a building, meandering downstream. Isosurfaces of speed put shells around the
FAST air and render the wakes as holes -- exactly backwards -- while a slice at
one height shows the wake as what it is, a dark lobe in a bright field.

Buildings are drawn from the -1 sentinel that travels in the same file, so
geometry and field can never disagree about which cell is a wall.

Two decisions worth stating. The colour scale is fixed by --vmax rather than by
each frame's own maximum, because a per-frame scale makes a steady field look
like it is pulsing. And the ramp is linear, not logarithmic: wind speed spans a
factor of a few, not decades, and a log ramp would flatten precisely the
contrast this plot exists to show.

  usage: plot_slice.py <field.vtk> <nx> <ny> <nz> <k> <out.png>
                       [--vmax V | --pct 98] [--dx 5]
"""
import array, math, struct, sys, zlib


def read_vtk(path, n):
    d = open(path, 'rb').read()
    key = b'LOOKUP_TABLE default\n'
    a = array.array('f')
    a.frombytes(d[d.index(key) + len(key):][:n * 4])
    a.byteswap()                       # legacy VTK binary is big-endian
    return a


def png(path, w, h, buf):
    def chunk(t, d):
        return (struct.pack('>I', len(d)) + t + d +
                struct.pack('>I', zlib.crc32(t + d) & 0xffffffff))
    raw = b''.join(b'\x00' + bytes(buf[y*w*3:(y+1)*w*3]) for y in range(h))
    open(path, 'wb').write(
        b'\x89PNG\r\n\x1a\n' +
        chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
        chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def ramp(t):
    """Deep blue (still air) through teal and amber to white (fastest).
    Dark at the bottom so wakes read as voids against the free stream."""
    t = max(0.0, min(1.0, t))
    stops = [(0.00, (18, 24, 46)), (0.25, (26, 82, 118)),
             (0.50, (38, 152, 146)), (0.72, (232, 168, 56)),
             (1.00, (255, 248, 226))]
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0])
            return tuple(int(a[1][k] + f * (b[1][k] - a[1][k])) for k in range(3))
    return stops[-1][1]


def main():
    a = sys.argv[1:]
    def opt(name, d=None):
        return a[a.index(name) + 1] if name in a else d
    src, nx, ny, nz, k, out = a[0], int(a[1]), int(a[2]), int(a[3]), int(a[4]), a[5]
    dx = float(opt('--dx', '5'))
    F = read_vtk(src, nx * ny * nz)
    at = lambda x, y: F[(k * ny + y) * nx + x]

    vals = [at(x, y) for y in range(ny) for x in range(nx) if at(x, y) > -0.5]
    # SCALE TO A PERCENTILE, NOT THE MAXIMUM. The outlet/top corner artefact of
    # sec:urbansolvedcity reaches several times the free stream, and scaling to
    # it renders the entire city in the bottom quarter of the ramp -- the same
    # trap the aorta volume render hit, where p100 was 2.7x p99.9. The artefact
    # is left visible rather than cropped; it just does not get to set the scale.
    if opt('--vmax'):
        vmax = float(opt('--vmax'))
    elif vals:
        q = float(opt('--pct', '98'))
        v = sorted(vals)
        vmax = v[min(len(v) - 1, int(q / 100.0 * len(v)))]
    else:
        vmax = 1.0

    pad = 14
    W, H = nx + 2 * pad, ny + 2 * pad + 22
    buf = bytearray([12, 13, 20] * (W * H))

    def put(px, py, rgb):
        if 0 <= px < W and 0 <= py < H:
            i = (py * W + px) * 3
            buf[i:i+3] = bytes(rgb)

    solid_n = 0
    for y in range(ny):
        for x in range(nx):
            v = at(x, y)
            if v < -0.5:
                rgb = (86, 88, 96); solid_n += 1     # building, at this height
            else:
                rgb = ramp(v / vmax if vmax > 0 else 0.0)
            put(pad + x, pad + (ny - 1 - y), rgb)

    # 200 m scale bar
    n = int(round(200.0 / dx))
    for i in range(n):
        put(pad + 8 + i, pad + ny - 8, (240, 240, 245))
        put(pad + 8 + i, pad + ny - 7, (240, 240, 245))

    png(out, W, H, buf)
    print("  %s  z = %.0f m  vmax %.2f  solid %d/%d cells in plane"
          % (out.split('/')[-1], (k + 0.5) * dx, vmax, solid_n, nx * ny))


main()
