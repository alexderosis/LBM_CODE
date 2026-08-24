#!/bin/bash
# Build the urban dispersion collage animation from demonstrator/urban output.
#
# Pipeline, and why each stage is what it is:
#   vol_urban       renders three cameras per frame to PPM. There is no ParaView
#                   here, so the renderer is the one in demonstrator/.
#   rsvg-convert    rasterises the text overlay. This ffmpeg has no drawtext and
#                   the environment has no PIL, but rsvg-convert and ImageMagick
#                   are both present and do typography properly.
#   magick          assembles the three panels and the overlay onto one canvas.
#   ffmpeg          holds each simulation frame for several video frames, so the
#                   result plays at a readable speed rather than flickering past.
#
# Camera positions are ABSOLUTE and are the ones the original project recovered
# by replaying its candidate loop; they transfer unchanged because the domain is
# the same 2 km at 5 m. See render_palette.py in the Pollutant project for why
# they cannot be reconstructed from azimuth/elevation.
#
#   usage: make_urban_anim.sh <vtk_dir> <work_dir> <out.mp4> [minutes_total] [steady_min]
set -euo pipefail

VTK=$1; WORK=$2; OUT=$3; TOTAL=${4:-12}; STEADY=${5:--1}
BIN=$(dirname "$0")/../../build/demonstrator/vol_urban
GEN=$(dirname "$0")/urban_overlay.py

FOC=997.5,997.5,107.49
# The original skyline camera framed a plume that had not yet crossed the
# domain. Pulled back 1.55x along its own view ray -- same framing, same
# orientation, 3.3 km of coverage instead of 2.1 km -- so the plume stays inside
# the panel at t = 12 min instead of running off the left edge.
CAM_H=-133.0,3748.9,1438.4       # skyline sweep
CAM_E=-728.54,-319.17,1316.08    # low oblique
CAM_G=-405.56,940.23,3000.10     # plan view

mkdir -p "$WORK/panels" "$WORK/frames"
FILES=($(ls "$VTK"/conc_*.vtk | sort))
N=${#FILES[@]}
echo "  $N simulation frames"

i=0
for f in "${FILES[@]}"; do
  n=$(printf "%04d" $i)
  "$BIN" -in "$f" -out "$WORK/panels/H_$n.ppm" -w 1160 -h 940 -cam $CAM_H -foc $FOC -fov 34
  "$BIN" -in "$f" -out "$WORK/panels/E_$n.ppm" -w  750 -h 420 -cam $CAM_E -foc $FOC -fov 34
  "$BIN" -in "$f" -out "$WORK/panels/G_$n.ppm" -w  750 -h 510 -cam $CAM_G -foc $FOC -fov 30

  t=$(python3 -c "print(f'{$TOTAL*$i/max(1,$N-1):.2f}')")
  python3 "$GEN" "$t" "$WORK/panels/ov_$n.svg" "$STEADY"
  rsvg-convert -w 1920 -h 1080 "$WORK/panels/ov_$n.svg" -o "$WORK/panels/ov_$n.png"

  magick -size 1920x1080 xc:'#f2f1ee' \
    \( "$WORK/panels/H_$n.ppm" \) -geometry +0+140   -composite \
    \( "$WORK/panels/E_$n.ppm" \) -geometry +1170+140 -composite \
    \( "$WORK/panels/G_$n.ppm" \) -geometry +1170+570 -composite \
    \( "$WORK/panels/ov_$n.png" \) -composite \
    "$WORK/frames/f_$n.png"
  printf "\r  rendered %d/%d" $((i+1)) $N
  i=$((i+1))
done
echo

# 6 video frames per simulation frame at 30 fps = 0.2 s each, which is close to
# the reference's pacing and slow enough to read.
ffmpeg -v error -y -framerate 5 -i "$WORK/frames/f_%04d.png" \
  -vf "fps=30,format=yuv420p" -c:v libx264 -crf 18 -preset slow "$OUT"
echo "  wrote $OUT"
