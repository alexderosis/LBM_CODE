#!/usr/bin/env python3
"""Three heights of a wind field in one frame, for showing 3D structure.

Volume rendering does not work on this field and it is worth recording why. A
plume is compact, so shells around it read as shells. A wake field is a boundary
layer: 45% of fluid cells carry a positive velocity deficit and the strong ones
fill the lower domain, so front-to-back compositing accumulates along every ray
and returns an opaque slab at any threshold. Raising the levels to the 99th
percentile does not fix it -- there is simply no empty space to see through.

Stacking horizontal slices does work. Each is unambiguous, and three of them
together show what the volume render was supposed to: that a wake is deep as
well as wide, and that its shape changes with height as the flow goes from
canyon-dominated near the ground to roughness-dominated above the roofs.

One colour scale across all heights AND all frames, passed in as --vmax. A
per-slice scale would make the deep slice look as fast as the high one, which
is the opposite of the point.

  usage: wind_stack.py <wind.vtk> <out.png> <nx> <ny> <nz> --ks 2,8,16
                       --vmax V [--dx 5] [--scale 3] [--trim 25] [--row]
"""
import array, struct, sys, zlib


def read_vtk(path, n):
    d = open(path, 'rb').read()
    key = b'LOOKUP_TABLE default\n'
    a = array.array('f')
    a.frombytes(d[d.index(key) + len(key):][:n * 4])
    a.byteswap()
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


# 5x7 digits, enough to label a height in metres without a font dependency.
GLYPH = {
 '0':("111","101","101","101","111"), '1':("010","110","010","010","111"),
 '2':("111","001","111","100","111"), '3':("111","001","111","001","111"),
 '4':("101","101","111","001","001"), '5':("111","100","111","001","111"),
 '6':("111","100","111","101","111"), '7':("111","001","001","001","001"),
 '8':("111","101","111","101","111"), '9':("111","101","111","001","111"),
 'm':("000","000","110","111","101"), ' ':("000","000","000","000","000"),
}


def main():
    a = sys.argv[1:]
    def opt(n, d=None): return a[a.index(n) + 1] if n in a else d
    src, out = a[0], a[1]
    nx, ny, nz = int(a[2]), int(a[3]), int(a[4])
    ks    = [int(v) for v in opt('--ks', '2,8,16').split(',')]
    vmax  = float(opt('--vmax', '10'))
    dx    = float(opt('--dx', '5'))
    sc    = max(1, int(opt('--scale', '3')))
    trim  = int(opt('--trim', '0'))

    F = read_vtk(src, nx * ny * nz)
    w = nx - 2 * trim
    row = '--row' in a          # panels side by side rather than stacked
    pad, gap = 10, 8
    if row:
        W = pad * 2 + len(ks) * w * sc + (len(ks) - 1) * gap
        H = ny * sc + 2 * pad
    else:
        W = w * sc + 2 * pad
        H = pad * 2 + len(ks) * ny * sc + (len(ks) - 1) * gap
    buf = bytearray([12, 13, 20] * (W * H))

    def put(px, py, rgb):
        if 0 <= px < W and 0 <= py < H:
            i = (py * W + px) * 3
            buf[i:i+3] = bytes(rgb)

    for s, k in enumerate(ks):
        y0 = pad if row else pad + s * (ny * sc + gap)
        x0 = pad + s * (w * sc + gap) if row else pad
        for y in range(ny):
            for x in range(w):
                v = F[(k * ny + y) * nx + (x + trim)]
                rgb = (86, 88, 96) if v < -0.5 else ramp(v / vmax)
                for p in range(sc):
                    for q in range(sc):
                        put(x0 + x * sc + p, y0 + (ny - 1 - y) * sc + q, rgb)
        # height label, top-left of each panel
        text = "%d m" % int((k + 0.5) * dx)
        cx = x0 + 6
        for ch in text:
            g = GLYPH.get(ch, GLYPH[' '])
            for r in range(5):
                for c in range(3):
                    if g[r][c] == '1':
                        for p in range(2):
                            for q in range(2):
                                put(cx + c*2 + p, y0 + 6 + r*2 + q, (245, 245, 250))
            cx += 8
    png(out, W, H, buf)
    print("  %s  heights %s m  vmax %.2f" %
          (out.split('/')[-1], [int((k + .5) * dx) for k in ks], vmax))


main()
