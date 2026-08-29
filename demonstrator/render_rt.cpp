//==============================================================================
//  Frame renderer for the Rayleigh-Taylor dumps.
//
//  Same split, and the same reasoning, as vol_aorta.cpp and vol_urban.cpp: plain
//  host C++, no Kokkos, no solver headers, reading the raw fields the simulation
//  wrote and turning them into PPMs that ffmpeg makes into a film. A renderer
//  welded to the simulation means every change of colour map costs a re-run --
//  eight and a half minutes here to alter a hue. This way it is a second.
//
//  It is C++ rather than Python for the reason the other two record: there is no
//  numpy in the target environment, and this is 192 x 768 x 3 fields over a
//  couple of hundred frames.
//
//  INPUT is FieldDump.hpp's format, three files per frame:
//      rt_%04d_phi.bin, rt_%04d_ux.bin, rt_%04d_uy.bin
//  each  int32 nx, int32 ny, then nx*ny float32, row major.
//
//  WHAT IT DRAWS. Two panels: the phase field, and the out-of-plane vorticity
//  d(uy)/dx - d(ux)/dy by central differences with periodic wrap in x.
//
//  Vorticity rather than speed, because the sign is the whole story in a shear
//  roll-up -- the two sides of every filament turn opposite ways, and a
//  magnitude map throws exactly that away. It is mapped DIVERGING about zero,
//  with a power law on the magnitude because vorticity here spans orders between
//  the braids and the cores; a linear map shows only the cores.
//
//  AUTO SCALE, CALIBRATED ON THE LAST FRAME. With -wscale 0 the vorticity full
//  scale is taken from a percentile of the field -- not its maximum, so one hot
//  cell cannot flatten everything else -- and the same value is then used for
//  every frame, so the animation stays comparable.
//
//  It calibrates on the LAST frame, not the first. The first frame of a
//  Rayleigh-Taylor run is a fluid at rest: its 99.9th percentile vorticity is
//  round-off, and using it puts the whole film at full saturation. Measured
//  1.4e-07 against the 6.2e-03 the developed flow actually reaches.
//
//    usage: render_rt -in <dir> -out <dir> -n <frames> [-pal aurora|paper|neon]
//                     [-up 2] [-wscale W] [-pct 0.999] [-crop y0,h] [-gamma g]
//==============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Field { int nx = 0, ny = 0; std::vector<float> v; };

bool read_field(const std::string& path, Field& f) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return false;
  std::int32_t a = 0, b = 0;
  i.read(reinterpret_cast<char*>(&a), sizeof a);
  i.read(reinterpret_cast<char*>(&b), sizeof b);
  if (a <= 0 || b <= 0) return false;
  f.nx = a; f.ny = b;
  f.v.resize(std::size_t(a) * std::size_t(b));
  i.read(reinterpret_cast<char*>(f.v.data()),
         std::streamsize(f.v.size() * sizeof(float)));
  return bool(i);
}

struct Rgb { unsigned char r, g, b; };

Rgb mix(Rgb a, Rgb b, double t) {
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  return Rgb{(unsigned char)std::lround(a.r + (b.r - a.r) * t),
             (unsigned char)std::lround(a.g + (b.g - a.g) * t),
             (unsigned char)std::lround(a.b + (b.b - a.b) * t)};
}
Rgb ramp(const Rgb* c, int n, double t) {
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  const double x = t * (n - 1);
  int i = int(x); if (i > n - 2) i = n - 2;
  return mix(c[i], c[i + 1], x - i);
}

//------------------------------------------------------------------------------
// Palettes. Each is a phase ramp plus a pair of diverging vorticity ramps that
// share their first stop, which is the zero-vorticity ground and therefore also
// the panel's background: making the two sides start from the same colour is
// what stops the sign boundary showing as a seam.
//------------------------------------------------------------------------------
struct Palette {
  const char* name;
  int np; Rgb phase[5];
  int nv; Rgb vneg[4]; Rgb vpos[4];
};

const Palette PALETTES[] = {
  // Bright, saturated, dark-ground. Phase runs ivory -> gold -> coral -> plum,
  // so the interface carries a warm band and the bulk phases stay far apart.
  {"aurora",
   5, {{255,252,242},{255,214,124},{240,124,120},{124, 66,148},{ 26, 20, 62}},
   4, {{ 22, 26, 46},{ 30,150,190},{ 96,226,232},{236,255,255}},
      {{ 22, 26, 46},{214,110, 52},{255,186, 78},{255,248,214}}},

  // Light ground, for print and for slides on white. The classic cool-warm
  // diverging map through white, which is the brightest option here.
  {"paper",
   4, {{255,255,255},{206,222,236},{ 92,132,180},{ 20, 42, 84}},
   4, {{250,250,252},{126,178,222},{ 44,102,178},{ 14, 44,104}},
      {{250,250,252},{244,166,140},{214, 82, 58},{124, 22, 22}}},

  // Maximum punch: near-black ground, electric complements.
  {"neon",
   4, {{ 12, 10, 26},{ 78, 40,132},{ 60,196,220},{236,255,250}},
   4, {{  8,  8, 18},{ 24,190,150},{130,255,190},{242,255,236}},
      {{  8,  8, 18},{198, 42,158},{255,124,214},{255,232,250}}},
};

const Palette& pick(const char* name) {
  for (const Palette& p : PALETTES) if (!std::strcmp(p.name, name)) return p;
  return PALETTES[0];
}

}  // namespace

int main(int argc, char** argv) {
  std::string in = ".", out = ".", pal = "aurora";
  int n = 1, up = 2, y0 = -1, hh = -1;
  double wscale = 0.0, pct = 0.999, gamma = 0.45;

  for (int i = 1; i < argc; ++i) {
    auto nx = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-in"))     { if (i+1<argc) in  = argv[++i]; }
    else if (!std::strcmp(argv[i], "-out"))    { if (i+1<argc) out = argv[++i]; }
    else if (!std::strcmp(argv[i], "-pal"))    { if (i+1<argc) pal = argv[++i]; }
    else if (!std::strcmp(argv[i], "-n"))      { if (i+1<argc) n   = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-up"))     { if (i+1<argc) up  = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-wscale")) nx(wscale);
    else if (!std::strcmp(argv[i], "-pct"))    nx(pct);
    else if (!std::strcmp(argv[i], "-gamma"))  nx(gamma);
    else if (!std::strcmp(argv[i], "-crop")) {
      if (i + 1 < argc) { std::sscanf(argv[++i], "%d,%d", &y0, &hh); }
    }
  }
  const Palette& P = pick(pal.c_str());
  std::printf("render_rt: palette %s, %d frame(s), up %d\n", P.name, n, up);

  // ---- calibration pass: find the last frame that exists, scale from it ----
  double held = wscale;
  if (held <= 0.0) {
    int last = -1;
    for (int fr = n - 1; fr >= 0; --fr) {
      char nm[512];
      std::snprintf(nm, sizeof nm, "%s/rt_%04d_phi.bin", in.c_str(), fr);
      std::ifstream probe(nm, std::ios::binary);
      if (probe) { last = fr; break; }
    }
    Field ux, uy;
    char nm[512];
    std::snprintf(nm, sizeof nm, "%s/rt_%04d_ux.bin", in.c_str(), last);
    const bool a = last >= 0 && read_field(nm, ux);
    std::snprintf(nm, sizeof nm, "%s/rt_%04d_uy.bin", in.c_str(), last);
    const bool b = last >= 0 && read_field(nm, uy);
    if (a && b) {
      const int nx = ux.nx, ny = ux.ny;
      std::vector<float> mag;
      mag.reserve(std::size_t(nx) * ny);
      for (int y = 1; y < ny - 1; ++y)
        for (int x = 0; x < nx; ++x) {
          const int xp = (x + 1) % nx, xm = (x + nx - 1) % nx;
          const double w =
              0.5 * (double(uy.v[std::size_t(y) * nx + xp]) -
                     double(uy.v[std::size_t(y) * nx + xm])) -
              0.5 * (double(ux.v[std::size_t(y + 1) * nx + x]) -
                     double(ux.v[std::size_t(y - 1) * nx + x]));
          mag.push_back(float(std::fabs(w)));
        }
      std::size_t k = std::size_t(pct * double(mag.size() - 1));
      std::nth_element(mag.begin(), mag.begin() + k, mag.end());
      held = std::max(double(mag[k]), 1e-12);
      std::printf("  vorticity full scale %.4e (%.3f percentile of frame %d)\n",
                  held, pct, last);
    }
  }

  for (int fr = 0; fr < n; ++fr) {
    char nm[512];
    Field phi, ux, uy;
    std::snprintf(nm, sizeof nm, "%s/rt_%04d_phi.bin", in.c_str(), fr);
    if (!read_field(nm, phi)) { std::printf("  missing %s, stopping\n", nm); break; }
    std::snprintf(nm, sizeof nm, "%s/rt_%04d_ux.bin", in.c_str(), fr);
    if (!read_field(nm, ux)) break;
    std::snprintf(nm, sizeof nm, "%s/rt_%04d_uy.bin", in.c_str(), fr);
    if (!read_field(nm, uy)) break;

    const int nx = phi.nx, ny = phi.ny;
    auto at = [&](const std::vector<float>& v, int x, int y) {
      return double(v[std::size_t(y) * nx + x]);
    };

    // Vorticity, central differences, periodic in x, one-sided-free in y.
    std::vector<float> wz(std::size_t(nx) * ny, 0.f);
    for (int y = 1; y < ny - 1; ++y)
      for (int x = 0; x < nx; ++x) {
        const int xp = (x + 1) % nx, xm = (x + nx - 1) % nx;
        wz[std::size_t(y) * nx + x] =
            float(0.5 * (at(uy.v, xp, y) - at(uy.v, xm, y)) -
                  0.5 * (at(ux.v, x, y + 1) - at(ux.v, x, y - 1)));
      }

    // Auto scale from a percentile, not the max: a single hot cell would
    // otherwise set the scale and flatten everything else to nothing. Once set,
    // it is held for the rest of the run so frames stay comparable.
    const int gap = 6;
    const int W = (2 * nx + gap) * up, H = ny * up;
    const int cy = (y0 >= 0) ? y0 : 0, ch = (hh > 0) ? hh : H;
    std::vector<unsigned char> img(std::size_t(W) * ch * 3);

    for (int py = 0; py < ch; ++py) {
      const int gy = py + cy;
      const int y = ny - 1 - gy / up;                 // y up the page
      for (int px = 0; px < W; ++px) {
        Rgb c = P.vneg[0];                            // the panel ground
        const int col = px / up;
        if (col < nx) {
          c = ramp(P.phase, P.np, at(phi.v, col, y));
        } else if (col >= nx + gap) {
          const int x = col - nx - gap;
          double t = double(wz[std::size_t(y) * nx + x]) / held;
          t = std::max(-1.0, std::min(1.0, t));
          const double m = std::pow(std::fabs(t), gamma);
          c = (t < 0) ? ramp(P.vneg, P.nv, m) : ramp(P.vpos, P.nv, m);
        } else {
          c = Rgb{10, 10, 14};                        // divider
        }
        const std::size_t o = (std::size_t(py) * W + px) * 3;
        img[o] = c.r; img[o + 1] = c.g; img[o + 2] = c.b;
      }
    }
    std::snprintf(nm, sizeof nm, "%s/rt_%04d.ppm", out.c_str(), fr);
    std::FILE* f = std::fopen(nm, "wb");
    if (!f) { std::printf("  cannot write %s\n", nm); return 1; }
    std::fprintf(f, "P6\n%d %d\n255\n", W, ch);
    std::fwrite(img.data(), 1, img.size(), f);
    std::fclose(f);
  }
  std::printf("done\n");
  return 0;
}
