#!/usr/bin/env python3
"""Per-frame SVG overlay for the urban dispersion collage.

Text is generated as SVG and rasterised with rsvg-convert rather than drawn by
hand, because this ffmpeg build has no drawtext filter and the environment has no
PIL. rsvg-convert and ImageMagick are both present; between them they do
typography properly, which a hand-rolled bitmap font at title size would not.

EVERY NUMBER AND NAME IN THE CAPTION IS READ BACK OUT OF THE RUN'S OWN LOG.
The first version of this file had the city, the cell count, the wind and the
fetch typed in as literals. That is fine for exactly one run and wrong for the
second: pointed at a Manhattan run it captioned the picture "Manchester city
centre, 9.6M cells" with complete confidence. A caption asserting something the
picture does not show is worse than no caption, so the parser below is strict --
a field it cannot find is an error, never a default.

  usage: urban_overlay.py <t_minutes> <out.svg> --log <run.log>
                          --plan-note <measured one-liner> [--levels l0,l1,l2]
"""
import math, re, sys

W, H = 1920, 1080
FG, MUTE, WARN, GOOD = "#1b1b1d", "#5c5c62", "#9a3b2f", "#2f7d5a"
# The canvas make_urban_anim.sh composites onto; the scrims match it.
GROUND = "#f2f1ee"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def text(x, y, s, size, weight="400", fill=FG, anchor="start", op="1"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" fill-opacity="{op}" '
            f'text-anchor="{anchor}">{esc(s)}</text>')

# Panel labels sit ON the render, and over Manhattan the render is buildings all
# the way to the panel edge -- dark text on a busy mid-grey. A scrim in the page
# colour fixes it without moving the labels off the image. Width is estimated
# from the character count rather than measured, which is crude but only has to
# be generous: the scrim is nearly the ground colour, so overshooting it is
# invisible and undershooting it is not.
def panel_label(x, y, title, sub, tsize=30, ssize=20):
    # The subtitle is measured text of unknown length -- it is generated from the
    # run, so it can be a sentence -- and at the right-hand panels there are only
    # 700 px before the canvas edge. Shrink it to fit rather than let it run off,
    # which is what the first Manhattan render did.
    avail = W - x - 20
    while ssize > 13 and len(sub) * ssize * 0.52 > avail:
        ssize -= 1
    w = min(max(len(title) * tsize * 0.58, len(sub) * ssize * 0.52) + 26, avail + 13)
    h = tsize + ssize + 26
    return [f'<rect x="{x - 13}" y="{y - tsize - 8}" width="{w:.0f}" '
            f'height="{h:.0f}" rx="5" fill="{GROUND}" fill-opacity="0.80"/>',
            text(x, y, title, tsize, "700"),
            text(x, y + ssize + 6, sub, ssize, "400", MUTE)]

# ---------------------------------------------------------------------------
# The run log, parsed.
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, path):
        self.raw = open(path).read()

    def grab(self, pattern, what, cast=str, group=1):
        m = re.search(pattern, self.raw)
        if not m:
            sys.exit(f"urban_overlay: {what} not found in the run log -- "
                     f"the caption would have to invent it")
        return cast(m.group(group))

    def maybe(self, pattern, cast=str, group=1):
        m = re.search(pattern, self.raw)
        return cast(m.group(group)) if m else None

    def parse(self):
        self.nx = self.grab(r"geometry: (\d+) x \d+ x \d+", "grid nx", int)
        self.ny = self.grab(r"geometry: \d+ x (\d+) x \d+", "grid ny", int)
        self.nz = self.grab(r"geometry: \d+ x \d+ x (\d+)", "grid nz", int)
        self.cells = self.grab(r"= (\d+) cells", "cell count", int)
        self.dx = self.grab(r"cells at ([\d.]+) m", "resolution", float)
        # The place line is the line after the geometry line; a height field
        # with no "place" in its metadata simply does not print one.
        self.place = self.maybe(r"cells at [\d.]+ m  \([^)]*\)\n    ([^\n]+)\n")
        if self.place is None:
            sys.exit("urban_overlay: the height field has no place name")

        self.diffusion_only = "DIFFUSION ONLY" in self.raw
        if not self.diffusion_only:
            self.u_ref   = self.grab(r"u_ref ([\d.]+) m/s", "reference wind", float)
            self.z_ref   = self.grab(r"u_ref [\d.]+ m/s at ([\d.]+) m", "wind height", float)
            self.bearing = self.grab(r"bearing ([\d.]+) deg", "wind bearing", float)
        self.diff = self.grab(r"D ([\d.]+) m2/s", "diffusivity", float)
        self.src = tuple(int(v) for v in self.grab(
            r"source: cell \((\d+,\d+,\d+)\)", "source cell").split(","))
        self.src_z = self.grab(r"units/s at z = ([\d.]+) m", "source height", float)

        # dM/dt, from the table the solver printed. Steady state is a property
        # of THIS run: when the last column has settled to a few per cent of the
        # injection rate and stays there. Nothing is claimed if it never does.
        # Matched by SHAPE, not by column count. The first version of this
        # pinned every column of the table into one regex; a column was added to
        # the solver's output and the regex then matched nothing at all, which
        # silently dropped the steady-state annotation from every frame rather
        # than failing. A row here is: leading space, a time, then numbers.
        num = re.compile(r"-?[\d.]+(?:[eE][+-]?\d+)?%?$")
        rows = []
        for line in self.raw.splitlines():
            if not line.startswith(" "): continue
            f = line.split()
            if len(f) < 5 or not re.fullmatch(r"[\d.]+", f[0]): continue
            if not all(num.match(v) for v in f[1:]): continue
            rows.append((float(f[0]), float(f[-1])))
        self.t_end = rows[-1][0] / 60.0 if rows else 0.0
        self.steady = -1.0
        if len(rows) > 3:
            rate = self.maybe(r"open cells, ([\d.eE+-]+) units/s", float) or 1.0
            run = None
            for t, dm in rows[1:]:
                if abs(dm) < 0.02 * rate:
                    run = t / 60.0 if run is None else run
                else:
                    run = None
            self.steady = run if run is not None else -1.0
        return self

# ---------------------------------------------------------------------------
def compass(travel_deg):
    names = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
             "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return names[int(round((travel_deg % 360) / 22.5)) % 16]

def fetch_km(run):
    """How far the release can travel before it leaves the domain. Measured
       along the actual wind direction from the actual source cell, not the box
       diagonal -- those differ by 25% for the Manhattan configuration."""
    if run.diffusion_only:
        return None
    r = math.radians(run.bearing)
    ex, ey = -math.sin(r), -math.cos(r)      # the direction the wind blows TO
    x, y, s = run.src[0] + 0.5, run.src[1] + 0.5, 0
    while 0 <= x < run.nx and 0 <= y < run.ny:
        x += ex; y += ey; s += 1
    return s * run.dx / 1000.0

# ---------------------------------------------------------------------------
def main():
    argv = sys.argv[1:]
    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    t = float(argv[0]); out = argv[1]
    log = opt("--log")
    if not log:
        sys.exit("urban_overlay: --log <run.log> is required")
    run = Run(log).parse()
    lv = [float(v) for v in (opt("--levels") or "3e-4,3e-3,1.5e-2").split(",")]

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">']

    # --- title block ---------------------------------------------------------
    o.append(text(30, 62, f"Pollutant dispersion through {run.place}", 44, "700"))

    km = run.nx * run.dx / 1000.0
    spec = (f"Lattice-Boltzmann D3Q7 advection–diffusion  ·  {km:g} km domain at "
            f"{run.dx:g} m resolution  ·  {run.cells/1e6:.0f}M cells")
    if not run.diffusion_only:
        spec += (f"  ·  {run.u_ref:g} m/s at {run.z_ref:g} m, blowing "
                 f"{compass(run.bearing - 180.0)}")
    o.append(text(30, 95, spec, 21, "400", MUTE))

    # The caveat is part of the figure, not a footnote: the wind is prescribed,
    # so the thing a reader will assume they are seeing is not there.
    o.append(text(30, 122,
                  "prescribed logarithmic wind — no street-canyon recirculation and no "
                  f"building wakes;  diffusivity {run.diff:g} m²/s, set by the "
                  "stability margin", 19, "400", WARN))

    o.append(text(W - 30, 62, f"t = {t:.1f} min", 30, "500", MUTE, "end"))

    # Saying when the plume stops changing turns a static tail into the result
    # it actually is -- what enters now leaves -- instead of leaving the viewer
    # to wonder whether the animation has stalled.
    if run.steady > 0 and t >= run.steady:
        # Right-anchored, because a left-anchored line of unknown width runs off
        # the canvas -- which is exactly what the first version did.
        o.append(text(W - 30, 88, "steady state — what enters now leaves",
                      18, "400", GOOD, "end"))

    # --- panel labels --------------------------------------------------------
    f = fetch_km(run)
    sub = f"street-level release at {run.src_z:g} m"
    if f: sub += f", {f:.1f} km of fetch to the {compass(run.bearing - 180.0)}"
    o += panel_label(40, H - 74, "skyline sweep", sub)

    o += panel_label(1196, 512, "low oblique",
                     "the plume lifts into faster air as it runs", 26, 18)

    # The plan-view subtitle is a claim about what the picture shows, so it is
    # MEASURED (doc/fig/plume_stats.py --brief) and passed in rather than typed
    # here. The literal it replaced said "isotropic lateral spread — the streets
    # do not steer it", which held for Manchester and does not hold for
    # Manhattan: there the canyons carry the plume at 0.67x the spread it has
    # above the rooflines.
    note = opt("--plan-note")
    if not note:
        sys.exit("urban_overlay: --plan-note is required -- it is a claim about "
                 "the picture, so it has to be measured, not assumed")
    o += panel_label(1196, H - 52, "plan view", note, 26, 18)

    # --- isosurface key ------------------------------------------------------
    # These MUST match the band levels the renderer was given; make_urban_anim.sh
    # passes the same string to both. A key that quotes different numbers from
    # the ones the renderer used is a caption asserting what the picture does not
    # show.
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    def fmt(v):
        e = int(math.floor(math.log10(v)))
        m = v / 10 ** e
        ms = ("%g" % round(m, 1))
        return ("%s×10%s" % (ms, str(e).translate(sup))) if ms != "1" \
               else ("10%s" % str(e).translate(sup))
    keys = [(fmt(lv[2]), "#e03326"), (fmt(lv[1]), "#faa88a"), (fmt(lv[0]), "#e8e2dd")]
    x0 = W - 330
    o.append(text(x0, 128, "shells", 17, "600", MUTE))
    for i, (lab, col) in enumerate(keys):
        o.append(f'<rect x="{x0 + 62 + i*84}" y="116" width="14" height="14" '
                 f'fill="{col}" stroke="#b9b4ae" stroke-width="0.8"/>')
        o.append(text(x0 + 80 + i*84, 128, lab, 15, "400", MUTE))

    o.append("</svg>")
    open(out, "w").write("\n".join(o))

if __name__ == "__main__":
    main()
