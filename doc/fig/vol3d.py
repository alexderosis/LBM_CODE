"""Volume renderer for the 3D Orszag-Tang current field.

A mid-plane slice shows the current sheets only where they happen to cut that
plane and says nothing about how they are oriented in the box, which for this
flow is the interesting part. This ray-casts the whole volume instead.

Orthographic camera, front-to-back emission-absorption compositing, with:

  * GRADIENT SHADING. Pure emission gives a fog with no sense of form. Using the
    local gradient as a surface normal and adding a diffuse term makes the
    sheets read as surfaces, which is most of the difference between a render
    that looks three-dimensional and one that does not.
  * SUPERSAMPLING. Rendered at 2x and box-filtered down, because the volume is
    coarse and aliased edges on a 64^3 or 128^3 grid are the first thing the eye
    catches.
  * A SHARED SCALE across panels (`vref`). Self-normalising each panel of a time
    sequence hides the decay it exists to show.

No dependencies: pure Python, PNG written through zlib, as in mkpng.py.

  usage: vol3d.py in.bin out.png [az] [el] [size] [gamma] [vref] [ss]
"""
import math, struct, sys, zlib


def read_volume(fn):
    d = open(fn, 'rb').read()
    nx, ny, nz = struct.unpack('<iii', d[:12])
    n = nx * ny * nz
    return nx, ny, nz, list(struct.unpack('<%df' % n, d[12:12 + 4 * n]))


def write_png(fn, nx, ny, rgb):
    raw = b''.join(b'\x00' + bytes(rgb[y * nx * 3:(y + 1) * nx * 3]) for y in range(ny))

    def chunk(typ, data):
        c = struct.pack('>I', len(data)) + typ + data
        return c + struct.pack('>I', zlib.crc32(typ + data) & 0xffffffff)

    open(fn, 'wb').write(
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', nx, ny, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw, 9))
        + chunk(b'IEND', b''))


# Emission ramp: deep indigo where the current is weak, through teal and amber
# to near-white in the sheet cores. Rendered against a dark ground, as emissive
# volumes must be -- composited over white the weak field washes the sheets out
# and no transfer-function tuning recovers the contrast.
RAMP = [(0.00, (38, 44, 96)), (0.18, (52, 104, 176)),
        (0.38, (48, 168, 198)), (0.58, (110, 214, 168)),
        (0.74, (232, 206, 112)), (0.88, (250, 160, 88)),
        (1.00, (255, 248, 236))]


def emit(t):
    if t <= 0.0:
        return RAMP[0][1]
    if t >= 1.0:
        return RAMP[-1][1]
    for i in range(len(RAMP) - 1):
        a, b = RAMP[i], RAMP[i + 1]
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            return tuple(a[1][k] + f * (b[1][k] - a[1][k]) for k in range(3))
    return RAMP[-1][1]


def render(src, dst, az=38.0, el=24.0, size=460, gamma=2.0,
           vref=None, ss=2, samples=190):
    nx, ny, nz, vol = read_volume(src)
    peak = max(vol)
    if peak <= 0:
        raise SystemExit('empty volume')
    vmax = peak if vref is None else vref

    a, e = math.radians(az), math.radians(el)
    d = (-math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), -math.sin(e))
    wu = (0.0, 0.0, 1.0)
    r = (d[1] * wu[2] - d[2] * wu[1], d[2] * wu[0] - d[0] * wu[2],
         d[0] * wu[1] - d[1] * wu[0])
    rl = math.sqrt(sum(c * c for c in r)) or 1.0
    r = tuple(c / rl for c in r)
    u = (r[1] * d[2] - r[2] * d[1], r[2] * d[0] - r[0] * d[2],
         r[0] * d[1] - r[1] * d[0])

    # key light over the viewer's shoulder, a little left and above
    lx = -d[0] + 0.45 * r[0] + 0.55 * u[0]
    ly = -d[1] + 0.45 * r[1] + 0.55 * u[1]
    lz = -d[2] + 0.45 * r[2] + 0.55 * u[2]
    ll = math.sqrt(lx * lx + ly * ly + lz * lz) or 1.0
    lx, ly, lz = lx / ll, ly / ll, lz / ll

    cx, cy, cz = (nx - 1) * 0.5, (ny - 1) * 0.5, (nz - 1) * 0.5
    half = math.sqrt(cx * cx + cy * cy + cz * cz)
    span = 1.12 * half
    step = 2.0 * half / samples

    LUT = 256
    op = [0.0] * LUT
    col = [(0, 0, 0)] * LUT
    for i in range(LUT):
        t = i / (LUT - 1.0)
        s = t ** gamma
        op[i] = 1.0 - math.exp(-11.0 * s * step)
        col[i] = emit(t)

    W = size * ss
    acc = [0.0] * (W * W * 3)
    bg = (9, 11, 19)
    nxm, nym, nzm = nx - 2.001, ny - 2.001, nz - 2.001
    inv = 1.0 / vmax
    nxny = nx * ny
    # gradient-magnitude scale below which shading is treated as unreliable
    gref2 = (0.045 * vmax) ** 2

    for py in range(W):
        sv = (1.0 - 2.0 * (py + 0.5) / W) * span
        base = py * W * 3
        for px in range(W):
            su = (2.0 * (px + 0.5) / W - 1.0) * span
            ox = cx + r[0] * su + u[0] * sv - d[0] * half * 1.35
            oy = cy + r[1] * su + u[1] * sv - d[1] * half * 1.35
            oz = cz + r[2] * su + u[2] * sv - d[2] * half * 1.35
            R = G = B = 0.0
            A = 0.0
            for k in range(samples):
                t = k * step + half * 0.35
                x = ox + d[0] * t
                if x < 1.0 or x > nxm:
                    continue
                y = oy + d[1] * t
                if y < 1.0 or y > nym:
                    continue
                z = oz + d[2] * t
                if z < 1.0 or z > nzm:
                    continue
                i0 = int(x); j0 = int(y); k0 = int(z)
                fx = x - i0; fy = y - j0; fz = z - k0
                b0 = (k0 * ny + j0) * nx + i0
                b1 = b0 + nx; b2 = b0 + nxny; b3 = b2 + nx
                c00 = vol[b0] + (vol[b0 + 1] - vol[b0]) * fx
                c10 = vol[b1] + (vol[b1 + 1] - vol[b1]) * fx
                c01 = vol[b2] + (vol[b2 + 1] - vol[b2]) * fx
                c11 = vol[b3] + (vol[b3 + 1] - vol[b3]) * fx
                c0 = c00 + (c10 - c00) * fy
                c1 = c01 + (c11 - c01) * fy
                val = (c0 + (c1 - c0) * fz) * inv
                if val <= 0.03:
                    continue
                idx = int(val * (LUT - 1))
                if idx > LUT - 1:
                    idx = LUT - 1
                aa = op[idx] * (1.0 - A)
                if aa <= 0.002:
                    continue
                # local gradient as a surface normal
                gx = vol[b0 + 1] - vol[b0 - 1]
                gy = vol[b0 + nx] - vol[b0 - nx]
                gz = vol[b0 + nxny] - vol[b0 - nxny]
                g2 = gx * gx + gy * gy + gz * gz
                if g2 > 1e-24:
                    gl = math.sqrt(g2)
                    dot = -(gx * lx + gy * ly + gz * lz) / gl
                    if dot < 0.0:
                        dot = -dot                # light both faces of a sheet
                    # Where the field is nearly flat the gradient direction is
                    # numerical noise, and shading on it speckles the sheet
                    # edges at voxel scale. Fade the diffuse term out there and
                    # fall back to flat emission.
                    w = g2 / (g2 + gref2)
                    shade = (1.0 - w) * 0.92 + w * (0.62 + 0.42 * dot)
                else:
                    shade = 0.92
                cc = col[idx]
                R += aa * cc[0] * shade
                G += aa * cc[1] * shade
                B += aa * cc[2] * shade
                A += aa
                if A > 0.99:
                    break
            o = base + px * 3
            acc[o]     = R + (1.0 - A) * bg[0]
            acc[o + 1] = G + (1.0 - A) * bg[1]
            acc[o + 2] = B + (1.0 - A) * bg[2]

    # box-filter down from the supersampled buffer
    rgb = bytearray(size * size * 3)
    n2 = float(ss * ss)
    for y in range(size):
        for x in range(size):
            sr = sg = sb = 0.0
            for jy in range(ss):
                ro = ((y * ss + jy) * W + x * ss) * 3
                for jx in range(ss):
                    sr += acc[ro]; sg += acc[ro + 1]; sb += acc[ro + 2]
                    ro += 3
            o = (y * size + x) * 3
            v = sr / n2; rgb[o]     = 255 if v > 255 else int(v)
            v = sg / n2; rgb[o + 1] = 255 if v > 255 else int(v)
            v = sb / n2; rgb[o + 2] = 255 if v > 255 else int(v)
    write_png(dst, size, size, rgb)
    print('  %s  %dx%dx%d, peak %.5g, scaled to %.5g, az=%.0f el=%.0f, %dx SS'
          % (dst, nx, ny, nz, peak, vmax, az, el, ss))


if __name__ == '__main__':
    a = sys.argv[1:]
    render(a[0], a[1],
           float(a[2]) if len(a) > 2 else 38.0,
           float(a[3]) if len(a) > 3 else 24.0,
           int(a[4]) if len(a) > 4 else 460,
           float(a[5]) if len(a) > 5 else 2.0,
           vref=float(a[6]) if len(a) > 6 and a[6] != '-' else None,
           ss=int(a[7]) if len(a) > 7 else 2)
