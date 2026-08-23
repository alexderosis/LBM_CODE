#!/usr/bin/env python3
"""Side-by-side vertical sections of a prescribed and a solved wind field, from
demonstrator/urban's -wind-out dumps. Standard library only; see plot_urban.py.

  usage: plot_wind.py <prescribed.vtk> <solved.vtk> <nx> <ny> <nz> <y> <out.png>
"""
import array, struct, sys, zlib

def read(path, n):
    d = open(path, 'rb').read()
    k = b'LOOKUP_TABLE default\n'
    a = array.array('f'); a.frombytes(d[d.index(k) + len(k):][:n * 4]); a.byteswap()
    return a

def png(path, w, h, buf):
    def chunk(t, d):
        return (struct.pack('>I', len(d)) + t + d +
                struct.pack('>I', zlib.crc32(t + d) & 0xffffffff))
    raw = b''.join(b'\x00' + bytes(buf[y*w*3:(y+1)*w*3]) for y in range(h))
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' +
        chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
        chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))

def ramp(t):
    """Speed, not concentration -- a different quantity gets a different ramp, so
    the two figures cannot be confused at a glance."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (16, 22, 46)), (0.3, (32, 78, 140)), (0.55, (36, 158, 168)),
             (0.78, (168, 208, 92)), (1.0, (252, 250, 214))]
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0])
            return tuple(int(a[1][k] + f * (b[1][k] - a[1][k])) for k in range(3))
    return stops[-1][1]

def main():
    pa, pb, nx, ny, nz, ysec, out = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
        int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
    A, B = read(pa, nx*ny*nz), read(pb, nx*ny*nz)
    at = lambda v, x, z: v[(z * ny + ysec) * nx + x]

    # ONE shared scale, so a stagnant wake cannot look like a free stream --
    # scaling each panel to its own maximum would hide the entire result.
    #
    # And a PERCENTILE, not the maximum. The solved field carries a thin jet
    # along the domain edge where the pressure outlet meets the velocity-
    # prescribed faces (see demonstrator/urban.cpp): it reaches 29 m/s against a
    # 7.7 m/s free stream, and scaling to it renders every physical feature in
    # the bottom quarter of the ramp. The same trap as the aorta volume render,
    # where p100 was 2.7 times p99.9.
    vals = sorted(v for V in (A, B) for x in range(nx) for z in range(nz)
                  for v in (at(V, x, z),) if v > -0.5)
    hi = vals[int(0.98 * (len(vals) - 1))]

    sc, pad, gap = 8, 18, 20
    W = pad*2 + nx*sc
    H = pad*2 + 2*nz*sc + gap
    buf = bytearray([10, 11, 16] * (W * H))
    def blk(px, py, rgb):
        for dy in range(sc):
            for dx in range(sc):
                X, Y = px+dx, py+dy
                if 0 <= X < W and 0 <= Y < H:
                    i = (Y*W + X)*3; buf[i:i+3] = bytes(rgb)

    for panel, V in ((0, A), (1, B)):
        y0 = pad + panel*(nz*sc + gap)
        for z in range(nz):
            for x in range(nx):
                v = at(V, x, z)
                rgb = (104, 108, 120) if v < -0.5 else ramp(v / hi)
                blk(pad + x*sc, y0 + (nz-1-z)*sc, rgb)

    png(out, W, H, buf)
    print("  wrote %s (%dx%d), section y=%d, shared scale 0..%.2f m/s (p98; max %.2f)"
          % (out, W, H, ysec, hi, vals[-1]))

main()
