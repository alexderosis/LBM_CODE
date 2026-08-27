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
# THE RUN LOG IS AN INPUT, NOT A CONVENIENCE. Frame times, the city, the grid,
# the wind and the fetch all come out of it, so a caption cannot disagree with
# the run it is captioning. Frame times in particular are read from the solver's
# own table rather than interpolated as total*i/(N-1): those agree only when the
# run completed, and a run stopped early would have every label wrong.
#
# Camera positions are ABSOLUTE and cannot be reconstructed from azimuth and
# elevation; see render_palette.py in the Pollutant project. The built-in
# defaults frame a 2 km x 2 km x 300 m box -- Manchester -- and a domain of a
# different HEIGHT needs different ones, because -fov is the VERTICAL angle and
# a 500 m box under a camera framed for 300 m sits in the bottom third of the
# panel with two thirds empty sky. Pass a camera file for anything else:
#
#     CAMS=doc/fig/cams/manhattan.env make_urban_anim.sh ...
#
#   usage: make_urban_anim.sh <vtk_dir> <run.log> <work_dir> <out.mp4> [levels] [kappa]
set -euo pipefail

VTK=$1; LOG=$2; WORK=$3; OUT=$4
# Transfer-function levels. Right value depends on the run: a plume spread over
# 2.6 km carries about half the concentration of the same release over 1 km.
LEVELS=${5:-3e-4,3e-3,1.5e-2}; KAPPA=${6:-0.30,4.0,15.0}
BIN=$(dirname "$0")/../../build/demonstrator/vol_urban
GEN=$(dirname "$0")/urban_overlay.py

[ -n "${CAMS:-}" ] && source "$CAMS"

FOC=${FOC:-997.5,997.5,107.49}
# The original skyline camera framed a plume that had not yet crossed the
# domain. Pulled back 1.55x along its own view ray -- same framing, same
# orientation, 3.3 km of coverage instead of 2.1 km -- so the plume stays inside
# the panel at t = 12 min instead of running off the left edge.
CAM_H=${CAM_H:--133.0,3748.9,1438.4}       # skyline sweep
CAM_E=${CAM_E:--728.54,-319.17,1316.08}    # low oblique
CAM_G=${CAM_G:--405.56,940.23,3000.10}     # plan view
# Each panel gets its own focus and angle. They shared one for Manchester,
# where all three looked at the whole box; a panel that deliberately crops to
# part of the plume cannot.
FOC_H=${FOC_H:-$FOC}; FOC_E=${FOC_E:-$FOC}; FOC_G=${FOC_G:-$FOC}
FOV_H=${FOV_H:-34};   FOV_E=${FOV_E:-34};   FOV_G=${FOV_G:-30}

mkdir -p "$WORK/panels" "$WORK/frames"
FILES=($(ls "$VTK"/conc_*.vtk | sort))
N=${#FILES[@]}

# One time per frame, from the solver's table. If the two counts disagree the
# labels would silently slide against the pictures, so that is a hard error.
TIMES=($(grep -E '^ +[0-9]+\.[0-9] +[0-9.eE+-]+ +[0-9.eE+-]+ +[0-9.]+% ' "$LOG" \
         | awk '{print $1}'))
if [ ${#TIMES[@]} -ne $N ]; then
  echo "  $N frames in $VTK but ${#TIMES[@]} rows in $LOG -- they are not the same run" >&2
  exit 1
fi
echo "  $N simulation frames, $(printf '%.1f' $(echo "${TIMES[$((N-1))]}/60" | bc -l)) min"

# The plan-view caption is a claim about the plume, so it is measured once from
# the LAST frame -- the steady one -- and handed to every overlay. Measuring it
# per frame would make the caption change under the reader mid-animation.
NOTE=$(python3 "$(dirname "$0")/plume_stats.py" "${FILES[$((N-1))]}" --log "$LOG" --brief)
echo "  plan view: $NOTE"

i=0
for f in "${FILES[@]}"; do
  n=$(printf "%04d" $i)
  "$BIN" -in "$f" -out "$WORK/panels/H_$n.ppm" -w 1160 -h 940 -cam $CAM_H -foc $FOC_H -fov $FOV_H -levels $LEVELS -kappa $KAPPA
  "$BIN" -in "$f" -out "$WORK/panels/E_$n.ppm" -w  750 -h 420 -cam $CAM_E -foc $FOC_E -fov $FOV_E -levels $LEVELS -kappa $KAPPA
  "$BIN" -in "$f" -out "$WORK/panels/G_$n.ppm" -w  750 -h 510 -cam $CAM_G -foc $FOC_G -fov $FOV_G -levels $LEVELS -kappa $KAPPA

  t=$(python3 -c "print(f'{${TIMES[$i]}/60:.2f}')")
  python3 "$GEN" "$t" "$WORK/panels/ov_$n.svg" --log "$LOG" --levels "$LEVELS" --plan-note "$NOTE"
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
