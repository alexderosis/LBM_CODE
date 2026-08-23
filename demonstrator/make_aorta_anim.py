"""Assemble the aorta volume dumps into an animation.

Ray-casts every aorta_v*.bin with vol_aorta and hands the PPMs to ffmpeg.

Two things here are deliberate:

  SHARED SCALE. The colour scale is fixed across the whole sequence, taken from
  the last frames, by which point the flow is established. Self-normalising each
  frame makes a developing flow look steady and a decaying one look constant.

  PERCENTILE, NOT PEAK. The speed peak in this geometry is a lone hot spot on a
  staircase corner -- p99.9 is 0.026 and p100 is 0.070 -- so scaling to the peak
  compresses the entire vessel into the bottom third of the ramp and everything
  renders pale blue. The scale is a high percentile of the fluid speed instead.

  usage: make_aorta_anim.py <frame_dir> <out.mp4> [fps] [--period P --probe N]

Giving --period and --probe puts the run on a cardiac clock: each frame gets a
phase inset, and the shared scale is taken over the LAST WHOLE CYCLE rather than
the last few frames, so systole sets the top of the ramp. Scaling a pulsatile
run off an arbitrary tail can land in diastole and clip the whole of systole.
"""
import array, glob, os, struct, subprocess, sys

# $LBM_AORTA_GEOM, else relative to the working directory. This was an
# absolute path into one developer's home directory.
GEOM = os.environ.get('LBM_AORTA_GEOM', 'data/geometry.bin')
RENDER = os.environ.get('VOL_AORTA', 'build/demonstrator/vol_aorta')
PCT = 99.9


def pct_of(fn, p=PCT):
    f = open(fn, 'rb')
    nx, ny, nz = struct.unpack('<iii', f.read(12))
    a = array.array('f')
    a.fromfile(f, nx * ny * nz)
    s = sorted(x for x in a if x >= 0.0)          # negative marks solid
    return s[min(len(s) - 1, int(len(s) * p / 100.0))]


def main():
    argv = sys.argv[1:]
    period = probe = None
    for k in ('--period', '--probe'):
        if k in argv:
            i = argv.index(k)
            v = float(argv[i + 1])
            if k == '--period':
                period = v
            else:
                probe = v
            del argv[i:i + 2]
    d = argv[0]
    out = argv[1]
    fps = int(argv[2]) if len(argv) > 2 else 10

    frames = sorted(glob.glob(os.path.join(d, 'aorta_v*.bin')))
    if not frames:
        raise SystemExit('no volumes in ' + d)

    ncyc = int(round(period / probe)) if (period and probe) else 0
    tail = frames[-ncyc:] if ncyc > 1 else frames[max(0, len(frames) - 3):]
    vmax = max(pct_of(f) for f in tail)
    print('  %d frames, shared scale p%.1f = %.5g  (over %d tail frames)'
          % (len(frames), PCT, vmax, len(tail)))

    for i, f in enumerate(frames):
        p = os.path.join(d, 'r_%04d.ppm' % i)
        cmd = [RENDER, '-vol', f, '-geom', GEOM, '-out', p,
               '-vmax', '%.6g' % vmax, '-smooth', '3']
        if ncyc > 1:
            cmd += ['-phase', '%.5f' % ((i * probe % period) / period)]
        subprocess.run(cmd, check=True)

    cmd = ['ffmpeg', '-y', '-framerate', str(fps),
           '-i', os.path.join(d, 'r_%04d.ppm'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', out]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('  wrote ' + out)


if __name__ == '__main__':
    main()
