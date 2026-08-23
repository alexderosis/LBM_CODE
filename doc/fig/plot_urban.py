#!/usr/bin/env python3
"""Plan view and vertical section of an urban plume, from demonstrator/urban's
legacy-VTK output.

Pure standard library on purpose: the target machine has neither numpy nor
matplotlib, so this reads the binary with `array` (4 bytes an element, not the
~300 MB a tuple of Python floats would cost for 9.6 million cells) and encodes
the PNG with zlib directly.

  usage: plot_urban.py <conc.vtk> <nx> <ny> <nz> <out.png>
"""
import array, struct, sys, zlib, math

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
    """Dark ground through crimson to white. Keeps the low end dark so the
    building mask stays readable underneath the plume."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (12, 14, 22)), (0.25, (90, 18, 48)), (0.5, (190, 40, 50)),
             (0.75, (240, 140, 44)), (1.0, (255, 246, 214))]
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0])
            return tuple(int(a[1][k] + f * (b[1][k] - a[1][k])) for k in range(3))
    return stops[-1][1]

def main():
    src, nx, ny, nz, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
                           int(sys.argv[4]), sys.argv[5]
    C = read_vtk(src, nx * ny * nz)
    at = lambda x, y, z: C[(z * ny + y) * nx + x]

    # Column-integrated concentration, and the building mask (solid is -1).
    col = [0.0] * (nx * ny)
    hgt = [0] * (nx * ny)
    for z in range(nz):
        base = z * ny
        for y in range(ny):
            row = (base + y) * nx
            for x in range(nx):
                v = C[row + x]
                # Solid is written as exactly -1. Testing v < 0 instead
                # catches the scheme's small negative undershoots -- 3.2M
                # cells here, down to -8e-5 -- and paints most of the city
                # as building.
                if v < -0.5: hgt[y * nx + x] = z + 1
                else:        col[y * nx + x] += max(0.0, v)

    # A vertical section through the plume: the row carrying the most material.
    ybest = max(range(ny), key=lambda y: sum(col[y * nx:(y + 1) * nx]))

    pad, gap, vs = 16, 14, 3           # vs: vertical stretch of the section
    W = pad * 2 + nx
    H = pad * 2 + ny + gap + nz * vs
    buf = bytearray([8, 9, 14] * (W * H))

    def put(px, py, rgb):
        if 0 <= px < W and 0 <= py < H:
            i = (py * W + px) * 3
            buf[i:i+3] = bytes(rgb)

    # --- plan view: log-scaled column load over a grey building mask ---
    lo, hi = 1e-9, max(col) or 1.0
    for y in range(ny):
        for x in range(nx):
            v, hh = col[y * nx + x], hgt[y * nx + x]
            t = 0.0 if v <= lo else (math.log10(v / lo) / math.log10(hi / lo))
            r, g, b = ramp(t)
            if hh:                       # buildings: lift toward grey by height
                f = 0.30 + 0.45 * min(1.0, hh / 12.0)
                r = int(r * (1 - f) + 150 * f)
                g = int(g * (1 - f) + 152 * f)
                b = int(b * (1 - f) + 158 * f)
            put(pad + x, pad + (ny - 1 - y), (r, g, b))

    # --- vertical section at ybest ---
    sl = [max(0.0, at(x, ybest, z)) if at(x, ybest, z) > -0.5 else 0.0
          for z in range(nz) for x in range(nx)]
    smax = max(sl) or 1.0
    y0 = pad + ny + gap
    for z in range(nz):
        for x in range(nx):
            v = sl[z * nx + x]
            solid = at(x, ybest, z) < -0.5
            t = 0.0 if v <= 0 else max(0.0, 1.0 + math.log10(v / smax) / 4.0)
            rgb = (118, 120, 128) if solid else ramp(t)
            for k in range(vs):
                put(pad + x, y0 + (nz - 1 - z) * vs + k, rgb)

    # scale bar: 500 m at 5 m a cell = 100 cells
    for k in range(100):
        put(pad + 8 + k, pad + ny - 10, (235, 235, 240))
        put(pad + 8 + k, pad + ny - 9,  (235, 235, 240))

    png(out, W, H, buf)
    print("  wrote %s  (%dx%d)   peak column load %.4g, section peak %.4g, section at y=%d"
          % (out, W, H, hi, smax, ybest))

main()
