//==============================================================================
//  Volume renderer for the urban plume.
//
//  Same reasoning as vol_aorta.cpp, and the same dependency list -- a C++
//  compiler. There is no ParaView on this machine and no numpy in its Python, so
//  the renderer is native and the output is a PPM that ffmpeg and ImageMagick
//  turn into frames. It reads ONE concentration .vtk written by demonstrator/
//  urban and needs nothing else: the solid mask travels in the same file as the
//  -1 sentinel, so geometry and field can never disagree about which cell is a
//  building.
//
//  WHAT IT DRAWS, and why each choice.
//
//  BUILDINGS are sampled NEAREST-NEIGHBOUR from the sentinel, never interpolated.
//  Trilinear interpolation across a -1 mixes wall and air into values that are
//  neither, and the artefacts land exactly at the roofline where the eye is.
//  Nearest-neighbour also gives the blocky extruded-footprint look that is
//  honest about what the geometry actually is: voxels, not architecture.
//
//  THE PLUME is a banded transfer function -- three Gaussians in log-concentration
//  centred a decade apart, pale to salmon to red. That reads as three nested
//  translucent isosurfaces without the cost or the topology of extracting any:
//  each band is optically thin except near its own level, so the shells appear
//  where the field crosses them.
//
//  COMPOSITING is front-to-back over-compositing onto a LIGHT ground, not
//  emission-absorption. An emissive volume over a light background washes out;
//  this is the same lesson vol_aorta.cpp records.
//
//  Empty space is skipped on a coarse block grid, which is what makes 400x400x60
//  at a million rays a few seconds rather than a few minutes.
//
//    usage: vol_urban -in conc.vtk -out frame.ppm [-w 1160] [-h 920]
//                     [-cam x,y,z] [-foc x,y,z] [-fov 32] [-gain 1.0]
//                     [-levels l0,l1,l2] [-kappa k0,k1,k2]
//==============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct Vec { double x = 0, y = 0, z = 0; };
static Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec operator*(Vec a, double s) { return {a.x * s, a.y * s, a.z * s}; }
static double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec cross(Vec a, Vec b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static Vec norm(Vec a) { const double l = std::sqrt(dot(a, a)); return l > 0 ? a * (1.0 / l) : a; }

static Vec parse_vec(const std::string& s) {
  Vec v; char c;
  std::istringstream is(s);
  is >> v.x >> c >> v.y >> c >> v.z;
  return v;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string in, out = "frame.ppm";
  int W = 1160, H = 920;
  Vec cam{268.15, 2772.56, 966.15}, foc{997.5, 997.5, 107.49};
  double fov = 32.0, gain = 1.0;
  std::string levels, kappas;

  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    if      (s == "-in"   && a + 1 < argc) in  = argv[++a];
    else if (s == "-out"  && a + 1 < argc) out = argv[++a];
    else if (s == "-w"    && a + 1 < argc) W = std::atoi(argv[++a]);
    else if (s == "-h"    && a + 1 < argc) H = std::atoi(argv[++a]);
    else if (s == "-cam"  && a + 1 < argc) cam = parse_vec(argv[++a]);
    else if (s == "-foc"  && a + 1 < argc) foc = parse_vec(argv[++a]);
    else if (s == "-fov"  && a + 1 < argc) fov = std::atof(argv[++a]);
    else if (s == "-gain" && a + 1 < argc) gain = std::atof(argv[++a]);
    else if (s == "-levels" && a + 1 < argc) levels = argv[++a];
    else if (s == "-kappa"  && a + 1 < argc) kappas = argv[++a];
  }
  if (in.empty()) { std::fprintf(stderr, "vol_urban: -in is required\n"); return 2; }

  // ---- read the frame -------------------------------------------------------
  std::ifstream f(in, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", in.c_str()); return 1; }
  std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const std::size_t dpos = all.find("DIMENSIONS");
  const std::size_t spos = all.find("SPACING");
  const std::size_t bpos = all.find("LOOKUP_TABLE default\n");
  if (dpos == std::string::npos || spos == std::string::npos || bpos == std::string::npos) {
    std::fprintf(stderr, "%s: not a legacy VTK STRUCTURED_POINTS file\n", in.c_str());
    return 1;
  }
  int nx, ny, nz; double dx, dyy, dz;
  std::sscanf(all.c_str() + dpos, "DIMENSIONS %d %d %d", &nx, &ny, &nz);
  std::sscanf(all.c_str() + spos, "SPACING %lf %lf %lf", &dx, &dyy, &dz);
  const std::size_t n = std::size_t(nx) * ny * nz;
  const char* body = all.data() + bpos + std::strlen("LOOKUP_TABLE default\n");

  std::vector<float> C(n);
  std::vector<std::uint8_t> solid(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t w;
    std::memcpy(&w, body + i * 4, 4);
    w = __builtin_bswap32(w);                      // legacy VTK binary is big-endian
    float v; std::memcpy(&v, &w, 4);
    if (v < -0.5f) { solid[i] = 1; C[i] = 0.f; }   // sentinel, not a small negative
    else            C[i] = std::max(0.f, v);       // clamp the scheme's undershoot
  }
  auto idx = [&](int x, int y, int z) { return (std::size_t(z) * ny + y) * nx + x; };

  // ---- coarse occupancy, for empty-space skipping ---------------------------
  const int BS = 8;
  const int bx = (nx + BS - 1) / BS, by = (ny + BS - 1) / BS, bz = (nz + BS - 1) / BS;
  std::vector<float> block(std::size_t(bx) * by * bz, 0.f);
  std::vector<std::uint8_t> bsolid(std::size_t(bx) * by * bz, 0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const std::size_t b = (std::size_t(z / BS) * by + y / BS) * bx + x / BS;
        block[b] = std::max(block[b], C[idx(x, y, z)]);
        bsolid[b] |= solid[idx(x, y, z)];
      }

  // ---- transfer function ----------------------------------------------------
  // Three bands a decade apart. Each is optically thin except near its own
  // level, so the field's crossings of 3e-4, 3e-3 and 3e-2 read as nested
  // shells -- isosurfaces without extracting any.
  struct Band { double level, width, r, g, b, kappa; };
  //
  // The opacities are NOT ordered by how important each level is -- they are
  // inverse to how much volume it occupies. On a real frame the 3e-4 shell holds
  // a hundred times the cells of the 3e-2 core, so equal absorption per sample
  // renders a uniform white blob with the core invisible inside it. The faint
  // band therefore gets a small kappa and the core a large one, which is what
  // makes the shells legible as shells.
  //
  // Levels are ABSOLUTE, not percentiles of the current frame. Percentiles would
  // re-normalise every frame and the plume would appear to stop growing.
  Band bands[3] = {
    {3.0e-4, 0.42, 0.99, 0.96, 0.94, 0.30},   // outer haze, near-white
    {3.0e-3, 0.42, 0.98, 0.66, 0.55, 4.0},    // salmon
    {1.5e-2, 0.48, 0.88, 0.20, 0.15, 15.0},   // core, red
  };
  // The right levels depend on the run, not on the renderer: a plume spread over
  // 2.6 km carries roughly half the concentration of the same release over 1 km,
  // and the outer shell accumulates far more along each ray. Exposing them
  // avoids a rebuild per tuning attempt -- and the key in the overlay must be
  // told the same numbers, or it captions something the picture does not show.
  if (!levels.empty()) {
    double l0, l1, l2; char c;
    std::istringstream is(levels);
    if (is >> l0 >> c >> l1 >> c >> l2) {
      bands[0].level = l0; bands[1].level = l1; bands[2].level = l2;
    }
  }
  if (!kappas.empty()) {
    double k0, k1, k2; char c;
    std::istringstream is(kappas);
    if (is >> k0 >> c >> k1 >> c >> k2) {
      bands[0].kappa = k0; bands[1].kappa = k1; bands[2].kappa = k2;
    }
  }

  const double Lx = nx * dx, Ly = ny * dyy, Lz = nz * dz;
  const Vec fwd = norm(foc - cam);
  const Vec right = norm(cross(fwd, Vec{0, 0, 1}));
  const Vec up = cross(right, fwd);
  const double tanf = std::tan(fov * M_PI / 360.0);
  const double aspect = double(W) / double(H);

  std::vector<unsigned char> img(std::size_t(W) * H * 3);
  const double step = dx * 0.5;

  auto sampleC = [&](Vec p) -> double {
    const double gx = p.x / dx - 0.5, gy = p.y / dyy - 0.5, gz = p.z / dz - 0.5;
    const int x0 = int(std::floor(gx)), y0 = int(std::floor(gy)), z0 = int(std::floor(gz));
    if (x0 < 0 || y0 < 0 || z0 < 0 || x0 + 1 >= nx || y0 + 1 >= ny || z0 + 1 >= nz) return 0.0;
    const double fx = gx - x0, fy = gy - y0, fz = gz - z0;
    double v = 0;
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
          const double w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy) * (k ? fz : 1 - fz);
          v += w * double(C[idx(x0 + i, y0 + j, z0 + k)]);
        }
    return v;
  };

  auto render_rows = [&](int r0, int r1) {
    for (int py = r0; py < r1; ++py)
      for (int px = 0; px < W; ++px) {
        const double sx = (2.0 * (px + 0.5) / W - 1.0) * tanf * aspect;
        const double sy = (1.0 - 2.0 * (py + 0.5) / H) * tanf;
        const Vec dir = norm(fwd + right * sx + up * sy);

        // slab test against the domain box
        double t0 = 0, t1 = 1e30;
        const double lo[3] = {0, 0, 0}, hi[3] = {Lx, Ly, Lz};
        const double o[3] = {cam.x, cam.y, cam.z}, d[3] = {dir.x, dir.y, dir.z};
        bool miss = false;
        for (int k = 0; k < 3; ++k) {
          if (std::abs(d[k]) < 1e-12) { if (o[k] < lo[k] || o[k] > hi[k]) miss = true; }
          else {
            double a = (lo[k] - o[k]) / d[k], b = (hi[k] - o[k]) / d[k];
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a); t1 = std::min(t1, b);
          }
        }

        // Light ground, light sky -- a faint vertical gradient so the horizon
        // reads without a separate sky pass.
        const double sky = std::min(1.0, std::max(0.0, 0.5 + 0.5 * dir.z));
        double R = 0.960 - 0.02 * sky, G = 0.960 - 0.015 * sky, B = 0.955;
        double acc_r = 0, acc_g = 0, acc_b = 0, alpha = 0;

        if (!miss && t1 > t0) {
          double t = t0;
          while (t < t1 && alpha < 0.985) {
            const Vec p = cam + dir * t;
            const int cx = int(p.x / dx), cy = int(p.y / dyy), cz = int(p.z / dz);
            if (cx < 0 || cy < 0 || cz < 0 || cx >= nx || cy >= ny || cz >= nz) { t += step; continue; }

            // block skip: nothing solid and nothing above the faintest band
            const std::size_t b = (std::size_t(cz / BS) * by + cy / BS) * bx + cx / BS;
            if (!bsolid[b] && block[b] < bands[0].level * 0.25) { t += BS * dx * 0.5; continue; }

            if (solid[idx(cx, cy, cz)]) {
              // Flat-shaded block. The normal is the face last crossed, taken
              // from which neighbour is air -- exact for voxels, and it keeps
              // roof and wall visibly different without a gradient estimate.
              Vec nrm{0, 0, 1};
              if (cz + 1 < nz && !solid[idx(cx, cy, cz + 1)]) nrm = {0, 0, 1};
              else if (cx > 0 && !solid[idx(cx - 1, cy, cz)]) nrm = {-1, 0, 0};
              else if (cx + 1 < nx && !solid[idx(cx + 1, cy, cz)]) nrm = {1, 0, 0};
              else if (cy > 0 && !solid[idx(cx, cy - 1, cz)]) nrm = {0, -1, 0};
              else if (cy + 1 < ny && !solid[idx(cx, cy + 1, cz)]) nrm = {0, 1, 0};
              const Vec L = norm(Vec{-0.35, -0.5, 0.79});
              const double lam = 0.45 + 0.55 * std::max(0.0, dot(nrm, L));
              const double sh = 0.40 + 0.46 * lam;          // mid grey, never black
              acc_r += (1 - alpha) * sh; acc_g += (1 - alpha) * sh; acc_b += (1 - alpha) * sh * 1.01;
              alpha = 1.0;
              break;
            }

            const double c = sampleC(p);
            if (c > bands[0].level * 0.2) {
              const double lc = std::log10(std::max(c, 1e-12));
              double sr = 0, sg = 0, sb = 0, sigma = 0;
              for (const Band& bd : bands) {
                const double u = (lc - std::log10(bd.level)) / bd.width;
                const double wgt = std::exp(-u * u);
                const double k = bd.kappa * wgt * gain;
                sigma += k; sr += k * bd.r; sg += k * bd.g; sb += k * bd.b;
              }
              if (sigma > 1e-9) {
                const double a = 1.0 - std::exp(-sigma * step / dx);
                sr /= sigma; sg /= sigma; sb /= sigma;
                acc_r += (1 - alpha) * a * sr;
                acc_g += (1 - alpha) * a * sg;
                acc_b += (1 - alpha) * a * sb;
                alpha += (1 - alpha) * a;
              }
            }
            t += step;
          }
        }

        // ground plane, where the ray leaves the box downward without a hit
        if (alpha < 0.985 && dir.z < -1e-9) {
          const double tg = (0.0 - cam.z) / dir.z;
          if (tg > 0) {
            const Vec p = cam + dir * tg;
            const bool inside = p.x > -400 && p.x < Lx + 400 && p.y > -400 && p.y < Ly + 400;
            const double g0 = inside ? 0.895 : 0.93;
            acc_r += (1 - alpha) * g0; acc_g += (1 - alpha) * g0; acc_b += (1 - alpha) * (g0 + 0.01);
            alpha += (1 - alpha) * 1.0;
          }
        }

        R = acc_r + (1 - alpha) * R; G = acc_g + (1 - alpha) * G; B = acc_b + (1 - alpha) * B;
        const std::size_t o2 = (std::size_t(py) * W + px) * 3;
        img[o2 + 0] = (unsigned char)(255.0 * std::min(1.0, std::max(0.0, R)));
        img[o2 + 1] = (unsigned char)(255.0 * std::min(1.0, std::max(0.0, G)));
        img[o2 + 2] = (unsigned char)(255.0 * std::min(1.0, std::max(0.0, B)));
      }
  };

  const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < nt; ++i) {
    const int a = int(std::size_t(H) * i / nt), b = int(std::size_t(H) * (i + 1) / nt);
    pool.emplace_back(render_rows, a, b);
  }
  for (auto& th : pool) th.join();

  std::FILE* fo = std::fopen(out.c_str(), "wb");
  if (!fo) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
  std::fprintf(fo, "P6\n%d %d\n255\n", W, H);
  std::fwrite(img.data(), 1, img.size(), fo);
  std::fclose(fo);
  return 0;
}
