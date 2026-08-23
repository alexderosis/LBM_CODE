"""Streamline animation of the aorta demonstrator.

Follows the presentation of the source project's scripts/render_3d.py --
translucent vessel, speed-coloured flow, streamlines retraced every frame, a
slowly swinging camera, and a labelled colour bar -- but the drawing is done by
demonstrator/vol_aorta rather than matplotlib, for two reasons. There is no
numpy in this environment, and a volume renderer does the job the point cloud in
that script exists to approximate: matplotlib cannot volume-render, so it
scatters dots to suggest a filled vessel. Rendering the volume directly makes
the dots redundant, and adding them on top would only fog the result.

Text is drawn by ffmpeg's drawtext, when the local ffmpeg has it. Many builds
are compiled without libfreetype and then the filter simply does not exist, so
this checks once and falls back to no captions rather than failing 60 times in a
row. The colour bar and the cardiac-phase inset are drawn by the renderer
itself and are always present, so a caption-less frame is still readable.

  usage: make_aorta_stream.py <dump_dir> <out.mp4> [fps]
                              [--period P --probe N --t0 STEP --sl N]
"""
import glob, os, math, subprocess, sys

# $LBM_AORTA_GEOM, else relative to the working directory. This was an
# absolute path into one developer's home directory.
GEOM = os.environ.get('LBM_AORTA_GEOM', 'data/geometry.bin')
RENDER = os.environ.get('VOL_AORTA', 'build/demonstrator/vol_aorta')
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
PCT = 99.9

W, H = 640, 900
AZ0, AZ_SWING, EL = 25.0, 18.0, 10.0

A1, KSL = '0.42', '3.4'
INK, SUB = '0x16181d', '0x5a5f6b'
GREEN, RED = '0x0d8c33', '0xc63428'


def pct_of(fn, p=PCT):
    out = subprocess.run([RENDER, '-vol', fn, '-geom', GEOM, '-out', '/dev/null',
                          '-printpct', str(p)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def have_drawtext():
    out = subprocess.run(['ffmpeg', '-hide_banner', '-filters'],
                         capture_output=True, text=True)
    return 'drawtext' in out.stdout


def esc(t):
    return t.replace('\\', r'\\').replace(':', r'\:').replace("'", r"\'").replace('%', r'\%')


def main():
    a = sys.argv[1:]
    opt = {}
    for k in ('--period', '--probe', '--t0', '--sl'):
        if k in a:
            i = a.index(k); opt[k] = float(a[i + 1]); del a[i:i + 2]
    d, out = a[0], a[1]
    fps = int(a[2]) if len(a) > 2 else 15
    period = opt.get('--period', 2000.0)
    probe = opt.get('--probe', 100.0)
    t0 = opt.get('--t0', 0.0)
    nsl = int(opt.get('--sl', 34))

    frames = sorted(glob.glob(os.path.join(d, 'aorta_u*.bin')))
    if not frames:
        raise SystemExit('no vector dumps (aorta_u*.bin) in ' + d)

    captions = have_drawtext()
    if not captions:
        print('  ffmpeg has no drawtext filter (built without libfreetype); '
              'rendering without captions')

    ncyc = max(1, int(round(period / probe)))
    tail = frames[-ncyc:] if len(frames) >= ncyc else frames
    vmax = max(pct_of(f) for f in tail)
    print('  %d frames, %d streamlines, shared scale p%.1f = %.5g'
          % (len(frames), nsl, PCT, vmax))

    # colour-bar geometry must match vol_aorta.cpp
    bw, bh = 15, int(H * 0.30)
    bx, by = W - 92, int(H * 0.31)

    pngs = []
    for i, f in enumerate(frames):
        step = t0 + i * probe
        ph = (step % period) / period
        cycle = int(step // period) + 1
        az = AZ0 + AZ_SWING * math.sin(2.0 * math.pi * i / max(1, len(frames)))
        ppm = os.path.join(d, 's_%04d.ppm' % i)
        subprocess.run([RENDER, '-vol', f, '-geom', GEOM, '-out', ppm,
                        '-vmax', '%.6g' % vmax, '-smooth', '3', '-sl', str(nsl),
                        '-cbar', '-phase', '%.5f' % ph, '-az', '%.2f' % az,
                        '-el', '%.2f' % EL, '-w', str(W), '-h', str(H),
                        # A dimmer interior than the speed-only render uses. The
                        # filaments live INSIDE the fluid, so an interior opaque
                        # enough to show speed nicely on its own buries them --
                        # measured: at a1 = 1.6 a ray saturates about two voxels
                        # into the lumen and nothing behind that is ever seen.
                        '-a1', A1, '-ksl', KSL,
                        # Long enough to run the whole vessel. The centreline
                        # from root to descending outlet is over 400 voxels, so
                        # the renderer's 260-step default stops the filaments at
                        # the arch and the descending aorta looks unvisited.
                        '-slsteps', '560'],
                       check=True, stdout=subprocess.DEVNULL)

        def dt(text, x, y, size, colour):
            return ("drawtext=fontfile='%s':text='%s':x=%d:y=%d:fontsize=%d:fontcolor=%s"
                    % (FONT, esc(text), x, y, size, colour))

        vf = [
            dt('Pulsatile flow through a patient-specific aortic arch', 26, 26, 19, INK),
            dt('D3Q27 central moments  ·  cycle %d  ·  phase %3.0f%%  ·  Re 50, Womersley 8.04'
               % (cycle, ph * 100), 26, 52, 13, SUB),
            dt('● inlet (ascending aorta)', 26, 76, 12, GREEN),
            dt('● outlets (branches + descending aorta)', 26, 94, 12, RED),
            dt('speed', bx - 6, by - 26, 12, SUB),
        ]
        for k in range(5):
            y = by + int(k * (bh - 1) / 4.0)
            vf.append(dt('%.3f' % (vmax * (1 - k / 4.0)), bx + bw + 9, y - 6, 11, SUB))
        vf.append(dt('lattice units', bx - 6, by + bh + 12, 11, SUB))

        png = os.path.join(d, 's_%04d.png' % i)
        cmd = ['ffmpeg', '-y', '-i', ppm]
        if captions:
            cmd += ['-vf', ','.join(vf)]
        cmd += [png]
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(ppm)
        pngs.append(png)
        if i % 10 == 0:
            print('  rendered %d/%d' % (i + 1, len(frames)))

    subprocess.run(['ffmpeg', '-y', '-framerate', str(fps),
                    '-i', os.path.join(d, 's_%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('  wrote ' + out)


if __name__ == '__main__':
    main()
