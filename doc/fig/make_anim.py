"""Assemble the aorta frame dumps into an animation.

Renders every aorta_f*.bin with a SHARED colour scale and hands the PNGs to
ffmpeg. The shared scale is the point: frames normalised to their own maximum
pulse in brightness as the flow develops, which looks like physics and is not.
The scale is taken from the last frame, by which point the flow is established.

  usage: make_anim.py <frame_dir> <out.mp4> [fps]
"""
import glob, os, struct, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkpng

# $LBM_AORTA_GEOM, else relative to the working directory. This was an
# absolute path into one developer's home directory.
GEOM = os.environ.get('LBM_AORTA_GEOM', 'data/geometry.bin')
SLICE_X = 36


def peak_of(fn):
    _, _, v = mkpng.read_field(fn)
    return max(v)


def main():
    d = sys.argv[1]
    out = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    frames = sorted(glob.glob(os.path.join(d, 'aorta_f*.bin')))
    if not frames:
        raise SystemExit('no frames in ' + d)

    # shared scale from the last few frames, clipped a little below the very
    # peak so a single fast cell does not wash out the whole vessel
    tail = frames[max(0, len(frames) - 5):]
    vmax = max(peak_of(f) for f in tail) * 0.85
    print('  %d frames, shared scale vmax = %.5g' % (len(frames), vmax))

    # Cap positions are static, so they are read once from the geometry rather
    # than re-encoded into every frame.
    ny, nz, tg = mkpng.tag_slice(GEOM, SLICE_X)
    tg = mkpng.dilate_tags(ny, nz, tg, 2)

    pngs = []
    for i, f in enumerate(frames):
        p = os.path.join(d, 'frame_%04d.png' % i)
        mkpng.render_light(f, p, vmax, tags=tg)
        pngs.append(p)
    print('  rendered %d PNGs' % len(pngs))

    # -vf scale: pad to even dimensions, which h264 requires
    cmd = ['ffmpeg', '-y', '-framerate', str(fps),
           '-i', os.path.join(d, 'frame_%04d.png'),
           '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,scale=iw*2:ih*2:flags=neighbor',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out]
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('  wrote ' + out)


if __name__ == '__main__':
    main()
