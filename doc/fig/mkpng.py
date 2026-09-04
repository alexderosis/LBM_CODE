import struct, zlib, sys, math

def read_field(fn):
    d = open(fn, 'rb').read()
    nx, ny = struct.unpack('<ii', d[:8])
    v = struct.unpack('<%df' % (nx*ny), d[8:8+4*nx*ny])
    return nx, ny, list(v)

# diverging map, control points of a coolwarm-style ramp
CP = [(0.00, (  9,  38, 106)), (0.25, ( 62, 120, 200)),
      (0.50, (243, 243, 240)),
      (0.75, (214,  96,  77)), (1.00, (120,  10,  28))]

def cmap(t):
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    for i in range(len(CP)-1):
        a, b = CP[i], CP[i+1]
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            return tuple(int(round(a[1][k] + f*(b[1][k]-a[1][k]))) for k in range(3))
    return CP[-1][1]

# sequential ramp for fields with no meaningful zero -- temperature, speed.
SEQ = [(0.00, ( 12,  28,  62)), (0.30, ( 32, 100, 148)),
       (0.55, ( 84, 168, 152)), (0.78, (206, 190, 110)),
       (1.00, (250, 244, 226))]

def cmap_seq(t):
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    for i in range(len(SEQ)-1):
        a, b = SEQ[i], SEQ[i+1]
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            return tuple(int(round(a[1][k] + f*(b[1][k]-a[1][k]))) for k in range(3))
    return SEQ[-1][1]

def write_png(fn, nx, ny, rgb):
    raw = b''.join(b'\x00' + bytes(rgb[y*nx*3:(y+1)*nx*3]) for y in range(ny))
    def chunk(typ, data):
        c = struct.pack('>I', len(data)) + typ + data
        return c + struct.pack('>I', zlib.crc32(typ + data) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', nx, ny, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 9))
           + chunk(b'IEND', b''))
    open(fn, 'wb').write(png)

def render_seq(src, dst, lo=None, hi=None):
    """Sequential render, for fields whose zero carries no meaning. The range is
    the actual min/max unless given explicitly."""
    nx, ny, v = read_field(src)
    a = min(v) if lo is None else lo
    b = max(v) if hi is None else hi
    rng = (b - a) if b > a else 1.0
    rgb = bytearray(nx*ny*3)
    for y in range(ny):
        yy = ny - 1 - y
        for x in range(nx):
            r, g, bb = cmap_seq((v[yy*nx + x] - a) / rng)
            i = (y*nx + x) * 3
            rgb[i], rgb[i+1], rgb[i+2] = r, g, bb
    write_png(dst, nx, ny, rgb)
    return a, b

def render(src, dst, pct=0.99):
    """Colour saturates at the `pct` percentile of |v|. The current sheets are
    very localised -- at t=1 the peak is 46.4 against a 99th percentile of 22.7 --
    so scaling by the maximum leaves the whole field pale and hides the structure
    the figure exists to show. The clipping is stated in the caption."""
    nx, ny, v = read_field(src)
    a = sorted(abs(x) for x in v)
    vmax = a[min(len(a) - 1, int(pct * len(a)))]
    vpeak = a[-1]
    # A UNIFORMLY ZERO FIELD IS A LEGITIMATE INPUT and used to divide by zero
    # here. render_seq already guarded its range; this did not. It arises for
    # real: the departure of a conductive initial condition from its own
    # horizontal mean is identically zero, so the first frame of a perturbation
    # sequence is all zeros and crashed the render mid-loop, leaving a gap in
    # the numbering that ffmpeg then silently started past.
    if vmax <= 0.0:
        vmax = 1.0
    rgb = bytearray(nx*ny*3)
    for y in range(ny):
        yy = ny - 1 - y                       # flip so y increases upward
        for x in range(nx):
            t = 0.5 * (1.0 + v[yy*nx + x] / vmax)
            r, g, b = cmap(t)
            i = (y*nx + x) * 3
            rgb[i], rgb[i+1], rgb[i+2] = r, g, b
    write_png(dst, nx, ny, rgb)
    return vmax, vpeak

# Turbo, as the aorta project's light theme uses. Sampled control points; the
# ramp matters more than exactness, but the ordering -- blue through green and
# yellow to red -- is what makes speed readable at a glance on a light ground.
TURBO = [(0.00, ( 48,  18,  59)), (0.13, ( 70, 107, 227)),
         (0.25, ( 54, 168, 250)), (0.38, ( 26, 228, 182)),
         (0.50, (109, 251,  94)), (0.63, (191, 231,  47)),
         (0.75, (247, 168,  37)), (0.88, (233,  84,  16)),
         (1.00, (122,   4,   3))]


def cmap_turbo(t):
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    for i in range(len(TURBO) - 1):
        a, b = TURBO[i], TURBO[i + 1]
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            return tuple(int(round(a[1][k] + f * (b[1][k] - a[1][k]))) for k in range(3))
    return TURBO[-1][1]


def dilate_tags(ny, nz, tg, radius=1):
    """Thicken the cap markers.

    A cap is one voxel thick, so on a slice it is a single-pixel line and is
    invisible next to a vessel two hundred pixels tall. Widening it is a
    presentation choice about a boundary whose position is already known, not a
    change to anything measured.
    """
    out = bytearray(tg)
    for _ in range(radius):
        src = bytearray(out)
        for k in range(nz):
            for j in range(ny):
                if src[k * ny + j] != 0 and src[k * ny + j] != 1:
                    continue
                best = 0
                for dk, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    qk, qj = k + dk, j + dj
                    if 0 <= qk < nz and 0 <= qj < ny:
                        t = src[qk * ny + qj]
                        if t in (2, 3, 4):
                            best = t
                if best:
                    out[k * ny + j] = best
    return out


def tag_slice(geom_path, xs):
    """Pull one x-slice of tags out of a voxel geometry file.

    The caps do not move, so they can be read once from the geometry rather than
    re-encoded into every frame dump -- which would otherwise mean re-running the
    simulation just to colour two hundred cells.
    """
    import struct
    with open(geom_path, 'rb') as f:
        nx, ny, nz = struct.unpack('<iii', f.read(12))
        f.seek(12 + 7 * 8)
        body = f.read(nx * ny * nz)
    out = bytearray(ny * nz)
    for k in range(nz):
        base = k * ny * nx
        for j in range(ny):
            out[k * ny + j] = body[base + j * nx + xs]
    return ny, nz, out


def render_light(src, dst, vmax, alpha_range=(0.18, 0.85), tags=None):
    """Light theme, matching the aorta project's own (scripts/render_3d.py,
    THEME=light) rather than an invented palette:

        background  #f3f4f7
        wall face   (0.30, 0.45, 0.75, 0.09) over bg  ->  (228, 232, 242)
        wall edge   (0.15, 0.28, 0.55, 0.35) over bg  ->  (171, 183, 210)
        speed       turbo
        alpha       POINT_ALPHA_RANGE, 0.18 at rest to 0.85 at vmax

    The alpha ramp is the part that matters most and is easy to miss. Painting
    turbo at full opacity across the whole range makes near-stagnant fluid --
    turbo's dark blue -- the most prominent thing in a light frame, which is
    backwards. Ramping opacity with speed instead lets slow fluid sink into the
    background and keeps the eye on the fast core, which is what that project
    does and why its frames read the way they do.
    """
    nx, ny, v = read_field(src)
    BG   = (243, 244, 247)
    WALL = (228, 232, 242)
    EDGE = (171, 183, 210)
    a0, a1 = alpha_range
    amin = a0 * 0.5
    inv = (1.0 / vmax) if vmax > 0 else 0.0

    INLET  = (13, 140, 51)      # deep green, as INLET_COLOR
    OUTLET = (204, 38, 26)      # deep red,   as OUTLET_COLOR
    solid = [x < 0.0 for x in v]
    rgb = bytearray(nx * ny * 3)
    for y in range(ny):
        yy = ny - 1 - y
        row = yy * nx
        for x in range(nx):
            val = v[row + x]
            i = (y * nx + x) * 3
            if val < 0.0:
                # a solid cell touching fluid is the vessel outline, and that
                # outline is most of what makes the frame legible
                edge = False
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    qx, qy = x + dx, yy + dy
                    if 0 <= qx < nx and 0 <= qy < ny and not solid[qy * nx + qx]:
                        edge = True
                        break
                rgb[i], rgb[i + 1], rgb[i + 2] = EDGE if edge else WALL
                continue
            # A cap is a boundary node, not fluid: colour it by what it IS
            # rather than by the speed imposed on it, so the driven faces are
            # identifiable at a glance.
            if tags is not None:
                tg = tags[yy * nx + x]
                if tg == 2 or tg == 4:
                    rgb[i], rgb[i + 1], rgb[i + 2] = INLET
                    continue
                if tg == 3:
                    rgb[i], rgb[i + 1], rgb[i + 2] = OUTLET
                    continue
            t = val * inv
            if t > 1.0:
                t = 1.0
            al = a0 + (a1 - a0) * t
            if al < amin: al = amin
            if al > a1:   al = a1
            c = cmap_turbo(t)
            rgb[i]     = int(round(BG[0] + al * (c[0] - BG[0])))
            rgb[i + 1] = int(round(BG[1] + al * (c[1] - BG[1])))
            rgb[i + 2] = int(round(BG[2] + al * (c[2] - BG[2])))
    write_png(dst, nx, ny, rgb)


def render_masked(src, dst, vmax, mask_rgb=(26, 28, 36)):
    """Sequential render with negative values treated as solid.

    A shared vmax is required rather than optional: frames of an animation
    normalised to their own maximum pulse in brightness as the flow develops,
    which reads as physics and is not."""
    nx, ny, v = read_field(src)
    rgb = bytearray(nx * ny * 3)
    inv = (1.0 / vmax) if vmax > 0 else 0.0
    for y in range(ny):
        yy = ny - 1 - y
        for x in range(nx):
            val = v[yy * nx + x]
            i = (y * nx + x) * 3
            if val < 0.0:
                rgb[i], rgb[i + 1], rgb[i + 2] = mask_rgb
            else:
                t = val * inv
                r, g, b = cmap_seq(1.0 if t > 1.0 else t)
                rgb[i], rgb[i + 1], rgb[i + 2] = r, g, b
    write_png(dst, nx, ny, rgb)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        for tag in ("t05", "t1"):
            for q in ("zeta", "j"):
                m, pk = render(f"ot_{q}_{tag}.bin", f"ot_{q}_{tag}.png")
                print(f"  ot_{q}_{tag}.png   saturates at {m:8.3f}   peak {pk:8.3f}")
    else:
        # usage: mkpng.py div  src.bin dst.png [pct]
        #        mkpng.py seq  src.bin dst.png [lo hi]
        #
        # lo/hi PIN the sequential colour scale. Without them the range is the
        # frame's own min/max, which is right for a single figure and wrong for
        # a sequence: every frame then gets its own scale and the colours move
        # when the field does not. Pin it for an animation, and pin it to the
        # PHYSICAL range when there is one -- a field that leaves the pinned
        # range then saturates visibly instead of quietly rescaling the picture.
        mode, src, dst = args[0], args[1], args[2]
        if mode == "div":
            pct = float(args[3]) if len(args) > 3 else 0.99
            m, pk = render(src, dst, pct)
            print(f"  {dst}  diverging, saturates at {m:.5g}, peak {pk:.5g}")
        else:
            lo = float(args[3]) if len(args) > 4 else None
            hi = float(args[4]) if len(args) > 4 else None
            nxr, nyr, vr = read_field(src)
            a, b = render_seq(src, dst, lo, hi)
            print(f"  {dst}  sequential, scale [{a:.5g}, {b:.5g}]"
                  f"  data [{min(vr):.5g}, {max(vr):.5g}]"
                  + ("  CLIPPED" if (min(vr) < a - 1e-12 or max(vr) > b + 1e-12) else ""))
