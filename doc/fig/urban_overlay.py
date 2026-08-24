#!/usr/bin/env python3
"""Per-frame SVG overlay for the urban dispersion collage.

Text is generated as SVG and rasterised with rsvg-convert rather than drawn by
hand, because this ffmpeg build has no drawtext filter and the environment has no
PIL. rsvg-convert and ImageMagick are both present; between them they do
typography properly, which a hand-rolled bitmap font at title size would not.

  usage: urban_overlay.py <t_minutes> <out.svg> [steady_minutes]
"""
import sys

W, H = 1920, 1080
FG, MUTE, WARN = "#1b1b1d", "#5c5c62", "#9a3b2f"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def text(x, y, s, size, weight="400", fill=FG, anchor="start", op="1"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" fill-opacity="{op}" '
            f'text-anchor="{anchor}">{esc(s)}</text>')

def main():
    t = float(sys.argv[1]); out = sys.argv[2]
    # When the run reaches steady state is a property of the run, not a constant.
    # Passing it in stops the caption asserting a time the data does not support;
    # omit it and no claim is made.
    steady = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">']

    # --- title block -----------------------------------------------------------
    o.append(text(30, 62, "Pollutant dispersion through Manchester city centre",
                  44, "700"))
    o.append(text(30, 95,
                  "Lattice-Boltzmann D3Q7 advection–diffusion  ·  2 km domain at 5 m "
                  "resolution  ·  9.6M cells  ·  5 m/s wind toward NE", 21, "400", MUTE))
    # The caveat is part of the figure, not a footnote: the wind is prescribed,
    # so the thing a reader will assume they are seeing is not there.
    o.append(text(30, 122,
                  "prescribed logarithmic wind — no street-canyon recirculation and no "
                  "building wakes;  diffusivity 20 m²/s, set by the stability margin",
                  19, "400", WARN))

    o.append(text(W - 30, 62, f"t = {t:.1f} min", 30, "500", MUTE, "end"))

    # The plume reaches steady state at 5 min and the remaining frames barely
    # change. Saying so turns a static tail into the result it actually is --
    # what enters now leaves -- instead of leaving the viewer to wonder whether
    # the animation has stalled.
    if steady > 0 and t >= steady:
        # Right-anchored, because a left-anchored line of unknown width runs off
        # the canvas -- which is exactly what the first version did.
        o.append(text(W - 30, 88, "steady state — what enters now leaves",
                      18, "400", "#2f7d5a", "end"))

    # --- panel labels ----------------------------------------------------------
    o.append(text(40, H - 74, "skyline sweep", 30, "700"))
    o.append(text(40, H - 46, "street-level release at 7.5 m, 2.6 km of fetch along the diagonal",
                  20, "400", MUTE))

    o.append(text(1196, 520, "low oblique", 26, "700"))
    o.append(text(1196, 546, "the plume lifts into faster air as it runs", 18, "400", MUTE))

    o.append(text(1196, H - 46, "plan view", 26, "700"))
    o.append(text(1196, H - 22, "isotropic lateral spread — the streets do not steer it",
                  18, "400", MUTE))

    # --- isosurface key --------------------------------------------------------
    # These MUST match the band levels in demonstrator/vol_urban.cpp. A key that
    # quotes different numbers from the ones the renderer used is worse than no
    # key: it is a caption asserting something the picture does not show.
    keys = [("1.5×10⁻²", "#e03326"), ("3×10⁻³", "#faa88a"), ("3×10⁻⁴", "#e8e2dd")]
    x0 = W - 330
    o.append(text(x0, 128, "shells", 17, "600", MUTE))
    for i, (lab, col) in enumerate(keys):
        o.append(f'<rect x="{x0 + 62 + i*84}" y="116" width="14" height="14" '
                 f'fill="{col}" stroke="#b9b4ae" stroke-width="0.8"/>')
        o.append(text(x0 + 80 + i*84, 128, lab, 15, "400", MUTE))

    o.append("</svg>")
    open(out, "w").write("\n".join(o))

main()
