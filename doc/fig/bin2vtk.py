#!/usr/bin/env python3
"""Convert this tree's .bin field planes to legacy VTK for ParaView.

WHY THIS EXISTS. The drivers dump planes as two int32 (nx, ny) followed by
nx*ny float32, which is compact and trivial to read from Python but is not a
format ParaView knows. rb_high_ra's own -vtk flag writes real VTK, but ASCII:
at 2004 x 1000 that is ~40 MB per frame against 8 MB for the binary dump, and
it has to be decided before the run rather than after it.

So this converts what you already have. Output is legacy VTK
STRUCTURED_POINTS in BINARY, which ParaView reads directly and which is the
same size as the input rather than five times it.

BIG-ENDIAN, AND THAT IS NOT OPTIONAL. The legacy VTK binary format is
specified big-endian regardless of the machine writing it. Writing native
little-endian floats produces a file ParaView opens without complaint and
renders as garbage -- values around 1e-40 and 1e38 -- which looks like a
simulation that diverged rather than a byte-order mistake. Hence '>f'.

TIME SERIES. Name the outputs with a trailing number and ParaView groups them
automatically: open `rb_T_..vtk` (the collapsed name it offers in the file
dialog) and the animation controls become live.

    usage:
      bin2vtk.py FIELD.bin OUT.vtk                  one plane
      bin2vtk.py --glob 'rb_T_*.bin' --out T        a whole series -> T_0000.vtk...
      bin2vtk.py --glob 'rb_T_*.bin' --out T --pair rb_u    T and |u| in one file

    The --pair form puts both scalars in each file, which is what you want for
    ParaView: colour by Temperature, then switch to Speed without reloading.
"""
import glob as globmod
import os
import struct
import sys


def read_plane(fn):
    with open(fn, 'rb') as f:
        d = f.read()
    nx, ny = struct.unpack('<ii', d[:8])
    n = nx * ny
    return nx, ny, struct.unpack('<%df' % n, d[8:8 + 4 * n])


def write_vtk(fn, nx, ny, arrays):
    """arrays: list of (name, values). Legacy STRUCTURED_POINTS, binary, BE."""
    with open(fn, 'wb') as o:
        o.write(b'# vtk DataFile Version 3.0\n')
        o.write(b'M3LB field plane\n')
        o.write(b'BINARY\n')
        o.write(b'DATASET STRUCTURED_POINTS\n')
        o.write(b'DIMENSIONS %d %d 1\n' % (nx, ny))
        o.write(b'ORIGIN 0 0 0\n')
        o.write(b'SPACING 1 1 1\n')
        o.write(b'POINT_DATA %d\n' % (nx * ny))
        for i, (name, v) in enumerate(arrays):
            kw = b'SCALARS' if i == 0 else b'SCALARS'
            o.write(kw + b' %s float 1\n' % name.encode())
            o.write(b'LOOKUP_TABLE default\n')
            o.write(struct.pack('>%df' % len(v), *v))
            o.write(b'\n')


def main(argv):
    if not argv:
        print(__doc__)
        return 1

    if argv[0] == '--glob':
        pattern = argv[1]
        out = 'field'
        pair = None
        i = 2
        while i < len(argv):
            if argv[i] == '--out':
                out = argv[i + 1]; i += 2
            elif argv[i] == '--pair':
                pair = argv[i + 1]; i += 2
            else:
                i += 1
        files = sorted(globmod.glob(pattern))
        if not files:
            print('no files matching %r' % pattern, file=sys.stderr)
            return 1
        for k, fn in enumerate(files):
            nx, ny, v = read_plane(fn)
            arrays = [('Temperature', v)]
            if pair:
                # same frame number, the paired prefix
                tag = fn.rsplit('_', 1)[-1]
                pf = '%s_%s' % (pair, tag)
                if os.path.exists(pf):
                    _, _, u = read_plane(pf)
                    arrays.append(('Speed', u))
                else:
                    print('  (no pair for %s)' % fn, file=sys.stderr)
            dst = '%s_%04d.vtk' % (out, k)
            write_vtk(dst, nx, ny, arrays)
        print('wrote %d files: %s_0000.vtk .. %s_%04d.vtk  (%d x %d)'
              % (len(files), out, out, len(files) - 1, nx, ny))
        return 0

    src, dst = argv[0], argv[1]
    nx, ny, v = read_plane(src)
    write_vtk(dst, nx, ny, [('Temperature', v)])
    print('%s -> %s  (%d x %d, range [%.5g, %.5g])'
          % (src, dst, nx, ny, min(v), max(v)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
