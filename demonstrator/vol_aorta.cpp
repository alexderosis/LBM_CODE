//==============================================================================
//  Volume renderer for the voxelised aorta.
//
//  WHY C++ AND NOT vol3d.py. The pure-Python caster is fine for the two
//  Orszag-Tang panels at 128^3. This volume is 109 x 184 x 361 and there are
//  tens of frames, roughly fifty times the work, which puts Python in the hours.
//  There is no numpy in this environment, so the renderer moved rather than the
//  pipeline: the output is a PPM and ffmpeg does the rest, which keeps the
//  dependency list at "a C++ compiler", same as the solver.
//
//  WHAT IT DRAWS. A vessel is not a cloud, and rendering speed alone as emission
//  shows only the fast core -- the anatomy vanishes, which is the one thing a
//  patient geometry is for. So the transfer function has two parts:
//
//    SURFACE. Where the lumen mask has a gradient -- the vessel wall -- a pale
//      translucent shell is deposited, diffuse-shaded off the mask gradient as a
//      normal. This is what makes the arch read as a tube rather than a fog.
//    INTERIOR. Everywhere inside the lumen, opacity and colour ramp with local
//      speed, so the flow glows through the shell.
//
//  The mask is carried as its own field rather than being inferred from the
//  solid sentinel at sample time: trilinear interpolation across a -1 sentinel
//  mixes wall and fluid into values that are neither, and the artefacts land
//  exactly at the wall, where the eye is.
//
//  Inlet and outlet caps are coloured from the geometry tags, so the driven face
//  and the free faces are identifiable without a separate overlay pass.
//
//  Compositing is front-to-back over-compositing, not emission-absorption: the
//  ground here is light, and an emissive volume over a light ground washes out.
//
//    usage: vol_aorta -vol f.bin -geom g.bin -out f.ppm [-vmax v] [-az a]
//                     [-el e] [-w W] [-h H] [-step s]
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

struct Vec { double x, y, z; };
static Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec operator*(Vec a, double s) { return {a.x * s, a.y * s, a.z * s}; }
static double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec cross(Vec a, Vec b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static Vec norm(Vec a) {
  const double l = std::sqrt(dot(a, a));
  return l > 0 ? a * (1.0 / l) : a;
}

//------------------------------------------------------------------------------
// Light-ground ramp. Starts at a pale blue-grey rather than at a saturated dark
// so that stagnant fluid tints the vessel instead of blackening it; turbo's low
// end is near-navy and reads as a hole on this background.
struct Stop { double t; double r, g, b; };
const Stop RAMP[] = {
    {0.00, 198, 208, 228}, {0.12, 128, 164, 216}, {0.30,  72, 166, 196},
    {0.50,  88, 188, 126}, {0.70, 226, 182,  78}, {0.85, 232, 124,  56},
    {1.00, 206,  56,  44}};
const int NRAMP = int(sizeof RAMP / sizeof RAMP[0]);

static void ramp(double t, double& r, double& g, double& b) {
  if (t <= 0) { r = RAMP[0].r; g = RAMP[0].g; b = RAMP[0].b; return; }
  if (t >= 1) { r = RAMP[NRAMP-1].r; g = RAMP[NRAMP-1].g; b = RAMP[NRAMP-1].b; return; }
  for (int i = 0; i < NRAMP - 1; ++i)
    if (t <= RAMP[i + 1].t) {
      const double f = (t - RAMP[i].t) / (RAMP[i + 1].t - RAMP[i].t);
      r = RAMP[i].r + f * (RAMP[i + 1].r - RAMP[i].r);
      g = RAMP[i].g + f * (RAMP[i + 1].g - RAMP[i].g);
      b = RAMP[i].b + f * (RAMP[i + 1].b - RAMP[i].b);
      return;
    }
}

}  // namespace

int main(int argc, char** argv) {
  std::string volf, geomf, outf;
  double vmax = -1, az = 25.0, el = 10.0, step = 0.5;
  double ksurf = 0.09, a0 = 0.003, a1 = 1.60, gref = 0.30, agam = 2.0;
  double phase = -1.0;   // >= 0 draws the cardiac-phase inset
  int nsl = 0, slsteps = 260; double slstep = 0.9, ksl = 2.6;
  bool cbar = false;
  double printpct = -1.0;  // report a speed percentile and exit, for the driver
  int W = 640, H = 900, smooth = 2;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-vol"  && i + 1 < argc) volf  = argv[++i];
    else if (a == "-geom" && i + 1 < argc) geomf = argv[++i];
    else if (a == "-out"  && i + 1 < argc) outf  = argv[++i];
    else if (a == "-vmax" && i + 1 < argc) vmax = std::atof(argv[++i]);
    else if (a == "-az"   && i + 1 < argc) az   = std::atof(argv[++i]);
    else if (a == "-el"   && i + 1 < argc) el   = std::atof(argv[++i]);
    else if (a == "-w"    && i + 1 < argc) W    = std::atoi(argv[++i]);
    else if (a == "-h"    && i + 1 < argc) H    = std::atoi(argv[++i]);
    else if (a == "-step" && i + 1 < argc) step = std::atof(argv[++i]);
    else if (a == "-smooth" && i + 1 < argc) smooth = std::atoi(argv[++i]);
    else if (a == "-ksurf"  && i + 1 < argc) ksurf = std::atof(argv[++i]);
    else if (a == "-a0"     && i + 1 < argc) a0    = std::atof(argv[++i]);
    else if (a == "-a1"     && i + 1 < argc) a1    = std::atof(argv[++i]);
    else if (a == "-gref"   && i + 1 < argc) gref  = std::atof(argv[++i]);
    else if (a == "-agam"   && i + 1 < argc) agam  = std::atof(argv[++i]);
    else if (a == "-phase"  && i + 1 < argc) phase = std::atof(argv[++i]);
    else if (a == "-sl"      && i + 1 < argc) nsl     = std::atoi(argv[++i]);
    else if (a == "-slsteps" && i + 1 < argc) slsteps = std::atoi(argv[++i]);
    else if (a == "-slstep"  && i + 1 < argc) slstep  = std::atof(argv[++i]);
    else if (a == "-ksl"     && i + 1 < argc) ksl     = std::atof(argv[++i]);
    else if (a == "-cbar") cbar = true;
    else if (a == "-printpct" && i + 1 < argc) printpct = std::atof(argv[++i]);
  }
  if (volf.empty() || outf.empty()) { std::fprintf(stderr, "need -vol and -out\n"); return 2; }

  // ---- volume -------------------------------------------------------------
  std::ifstream f(volf, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", volf.c_str()); return 2; }
  std::int32_t nx, ny, nz;
  f.read(reinterpret_cast<char*>(&nx), 4);
  f.read(reinterpret_cast<char*>(&ny), 4);
  f.read(reinterpret_cast<char*>(&nz), 4);
  const std::size_t N = std::size_t(nx) * ny * nz;
  f.seekg(0, std::ios::end);
  const std::streamoff vbytes = f.tellg();
  f.seekg(12, std::ios::beg);

  // Two dump layouts. A SCALAR file is speed with solid marked by a negative
  // sentinel. A VECTOR file is the three components, component-major, and has
  // no sentinel available -- a velocity component is legitimately negative --
  // so its mask comes from the geometry instead. Distinguished by size, not by
  // a flag: a mislabelled flag would silently reinterpret one as the other.
  const bool isvec = (vbytes == std::streamoff(12 + 3 * N * 4));
  if (!isvec && vbytes != std::streamoff(12 + N * 4)) {
    std::fprintf(stderr, "%s is %lld bytes; expected %lld (scalar) or %lld (vector)\n",
                 volf.c_str(), (long long)vbytes,
                 (long long)(12 + N * 4), (long long)(12 + 3 * N * 4));
    return 2;
  }
  std::vector<float> v(isvec ? 3 * N : N);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4));
  if (!f) { std::fprintf(stderr, "%s truncated\n", volf.c_str()); return 2; }

  const std::size_t nxny_h = std::size_t(nx) * ny;
  std::vector<float> mask(N), spd(N), inl(N), outl(N);
  std::vector<float> vx, vy, vz;
  double peak = 0;
  if (isvec) {
    vx.assign(v.begin(), v.begin() + N);
    vy.assign(v.begin() + N, v.begin() + 2 * N);
    vz.assign(v.begin() + 2 * N, v.begin() + 3 * N);
    for (std::size_t i = 0; i < N; ++i) {
      const double sp = std::sqrt(double(vx[i]) * vx[i] + double(vy[i]) * vy[i]
                                + double(vz[i]) * vz[i]);
      spd[i] = float(sp);
      if (sp > peak) peak = sp;
    }
    // mask filled from the geometry below
  } else {
    for (std::size_t i = 0; i < N; ++i) {
      const bool fl = v[i] >= 0.0f;
      mask[i] = fl ? 1.0f : 0.0f;
      spd[i]  = fl ? v[i] : 0.0f;
      if (fl && v[i] > peak) peak = v[i];
    }
  }
  if (vmax <= 0) vmax = peak > 0 ? peak : 1.0;

  // ---- cap tags from the geometry ----------------------------------------
  bool gotgeom = false;
  if (!geomf.empty()) {
    std::ifstream gf(geomf, std::ios::binary);
    if (gf) {
      gf.seekg(0, std::ios::end);
      const std::streamoff bytes = gf.tellg();
      gf.seekg(0, std::ios::beg);
      std::int32_t gx, gy, gz;
      gf.read(reinterpret_cast<char*>(&gx), 4);
      gf.read(reinterpret_cast<char*>(&gy), 4);
      gf.read(reinterpret_cast<char*>(&gz), 4);
      if (gx == nx && gy == ny && gz == nz) {
        const std::streamoff base = 12 + 7 * 8;
        const std::streamoff hdr = (bytes == base + std::streamoff(N) + 24) ? base + 24 : base;
        gf.seekg(hdr, std::ios::beg);
        std::vector<std::uint8_t> tg(N);
        gf.read(reinterpret_cast<char*>(tg.data()), std::streamsize(N));
        for (std::size_t i = 0; i < N; ++i) {
          if (tg[i] == 2 || tg[i] == 4) inl[i]  = 1.0f;
          else if (tg[i] == 3)          outl[i] = 1.0f;
          if (isvec) mask[i] = (tg[i] != 0) ? 1.0f : 0.0f;
          if (isvec && tg[i] == 0) spd[i] = 0.0f;
        }
        gotgeom = true;
      } else {
        std::fprintf(stderr, "  geometry dims differ from volume; caps not drawn\n");
      }
    }
  }

  // ---- smoothing ----------------------------------------------------------
  // The mask is binary, so its trilinear gradient is a staircase and the shell
  // comes out ringed with contours at voxel scale -- the render reads as a
  // topographic map of the voxeliser rather than as a vessel. A couple of
  // separable 1-2-1 passes move the transition over ~2 voxels and the banding
  // goes. The 0.5 level set barely moves: the kernel is symmetric, so this
  // smooths the surface normal without inflating the lumen.
  //
  // The speed field is smoothed with it. It has to be: filtering the mask but
  // not the speed puts the two out of register at the wall, which shows up as
  // colour bleeding through the shell.
  // The driver needs a shared colour scale across frames, and computing a
  // percentile of 1.2M speeds in pure Python costs about 20 s a frame. Doing it
  // here costs 0.15 s, so the driver shells out to this instead.
  if (printpct >= 0.0) {
    std::vector<float> q;
    q.reserve(N / 4);
    for (std::size_t i = 0; i < N; ++i) if (mask[i] > 0.5f) q.push_back(spd[i]);
    if (q.empty()) { std::printf("0\n"); return 0; }
    std::sort(q.begin(), q.end());
    std::size_t k = std::size_t(q.size() * printpct / 100.0);
    if (k >= q.size()) k = q.size() - 1;
    std::printf("%.8g\n", double(q[k]));
    return 0;
  }

  if (isvec && !gotgeom) {
    std::fprintf(stderr, "vector volume needs -geom: the mask cannot be inferred "
                         "from components, only from the tags\n");
    return 2;
  }

  // ---- streamlines --------------------------------------------------------
  // Retraced from the current field every frame, as in the source project's
  // render_3d.py, so the lines animate with the flow instead of sitting still
  // over a moving volume.
  //
  // They are SPLATTED INTO A VOLUME rather than drawn as polylines over the
  // finished image. Drawn in 2D every line would sit in front of the vessel:
  // the wall is translucent, so there is no single depth to test a line
  // against. Rasterising them into the volume hands them to the same ray march,
  // and a filament behind the arch is then correctly dimmed by the wall in
  // front of it -- which is most of what makes the result read as 3D.
  std::vector<float> sl_int, sl_spd;
  if (nsl > 0 && isvec) {
    sl_int.assign(N, 0.0f);
    sl_spd.assign(N, 0.0f);
    // Deterministic seeding -- inlet voxels at a fixed stride, not a random
    // sample. A fresh random set each frame makes the lines flicker on and off
    // between frames, which reads as noise rather than as flow.
    std::vector<std::size_t> cand;
    for (std::size_t i = 0; i < N; ++i) if (inl[i] > 0.5f) cand.push_back(i);
    auto samp = [&](double x, double y, double z, double& ax, double& ay, double& az2) {
      const int i0 = int(x), j0 = int(y), k0 = int(z);
      const double fx = x - i0, fy = y - j0, fz = z - k0;
      const std::size_t b0 = (std::size_t(k0) * ny + j0) * nx + i0;
      const std::size_t b1 = b0 + nx, b2 = b0 + nxny_h, b3 = b2 + nx;
      const double w000=(1-fx)*(1-fy)*(1-fz), w100=fx*(1-fy)*(1-fz);
      const double w010=(1-fx)*fy*(1-fz),     w110=fx*fy*(1-fz);
      const double w001=(1-fx)*(1-fy)*fz,     w101=fx*(1-fy)*fz;
      const double w011=(1-fx)*fy*fz,         w111=fx*fy*fz;
      auto T=[&](const std::vector<float>& q){
        return q[b0]*w000+q[b0+1]*w100+q[b1]*w010+q[b1+1]*w110
             + q[b2]*w001+q[b2+1]*w101+q[b3]*w011+q[b3+1]*w111; };
      ax = T(vx); ay = T(vy); az2 = T(vz);
    };
    auto inside = [&](double x, double y, double z) {
      return x > 1.0 && x < nx - 2.0 && y > 1.0 && y < ny - 2.0 && z > 1.0 && z < nz - 2.0;
    };
    const int nseed = std::min<int>(nsl, int(cand.size()));
    for (int sidx = 0; sidx < nseed; ++sidx) {
      const std::size_t c = cand[std::size_t(double(sidx) * cand.size() / nseed)];
      double pz0 = double(c / nxny_h);
      double py0 = double((c % nxny_h) / nx);
      double px0 = double(c % nx);
      double px = px0 + 0.5, py = py0 + 0.5, pz = pz0 + 0.5;
      for (int st = 0; st < slsteps; ++st) {
        if (!inside(px, py, pz)) break;
        double ax, ay, az2;
        samp(px, py, pz, ax, ay, az2);
        const double s0 = std::sqrt(ax*ax + ay*ay + az2*az2);
        if (s0 < 1e-9) break;
        // midpoint (RK2), stepping a fixed ARC LENGTH so the sample spacing
        // along the filament does not collapse where the flow is slow
        const double mx = px + 0.5*slstep*ax/s0, my = py + 0.5*slstep*ay/s0,
                     mz = pz + 0.5*slstep*az2/s0;
        if (!inside(mx, my, mz)) break;
        double bx, by2, bz;
        samp(mx, my, mz, bx, by2, bz);
        const double s1 = std::sqrt(bx*bx + by2*by2 + bz*bz);
        if (s1 < 1e-9) break;
        px += slstep*bx/s1; py += slstep*by2/s1; pz += slstep*bz/s1;
        if (!inside(px, py, pz)) break;
        const std::size_t here = (std::size_t(int(pz)) * ny + int(py)) * nx + int(px);
        if (mask[here] < 0.5f) break;              // left the lumen
        for (int dk = -1; dk <= 1; ++dk)
          for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
              const int X = int(px)+di, Y = int(py)+dj, Z = int(pz)+dk;
              if (X < 1 || X >= nx-1 || Y < 1 || Y >= ny-1 || Z < 1 || Z >= nz-1) continue;
              const double ddx = X + 0.5 - px, ddy = Y + 0.5 - py, ddz = Z + 0.5 - pz;
              const double w = std::exp(-(ddx*ddx + ddy*ddy + ddz*ddz) / 0.55);
              if (w < 0.02) continue;
              const std::size_t id = (std::size_t(Z) * ny + Y) * nx + X;
              // MAX, not sum: where filaments cross, summing blows out to white
              if (w > sl_int[id]) { sl_int[id] = float(w); sl_spd[id] = float(s1); }
            }
      }
    }
    std::size_t marked = 0; double slpeak = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (sl_int[i] > 0.0f) { ++marked; if (sl_int[i] > slpeak) slpeak = sl_int[i]; }
    std::printf("  %d streamlines from %zu inlet voxels, %zu voxels marked, peak %.3f\n",
                nseed, cand.size(), marked, slpeak);
  }

  // The cap tags are a single oblique voxel sheet. Interpolated raw they come
  // out as speckle across the cap face, so they are dilated once and then
  // carried through the same blur as the mask -- a cap should read as a disc.
  if (smooth > 0) {
    std::vector<float> dl(N);
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<float>& q = pass == 0 ? inl : outl;
      dl = q;
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            const std::size_t i = (std::size_t(z) * ny + y) * nx + x;
            if (q[i] > 0.5f) continue;
            float m2 = 0;
            if (x > 0)      m2 = std::max(m2, q[i - 1]);
            if (x < nx - 1) m2 = std::max(m2, q[i + 1]);
            if (y > 0)      m2 = std::max(m2, q[i - nx]);
            if (y < ny - 1) m2 = std::max(m2, q[i + nx]);
            if (z > 0)      m2 = std::max(m2, q[i - nxny_h]);
            if (z < nz - 1) m2 = std::max(m2, q[i + nxny_h]);
            dl[i] = m2;
          }
      q.swap(dl);
    }
  }
  if (smooth > 0) {
    std::vector<float> tmp(N);
    auto pass = [&](std::vector<float>& q, int axis) {
      const std::ptrdiff_t st = (axis == 0) ? 1 : (axis == 1) ? nx : std::ptrdiff_t(nxny_h);
      const int lim = (axis == 0) ? nx : (axis == 1) ? ny : nz;
      tmp = q;
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            const int c2 = (axis == 0) ? x : (axis == 1) ? y : z;
            const std::size_t i = (std::size_t(z) * ny + y) * nx + x;
            const float lo = (c2 > 0)       ? tmp[i - st] : tmp[i];
            const float hi = (c2 < lim - 1) ? tmp[i + st] : tmp[i];
            q[i] = 0.25f * lo + 0.5f * tmp[i] + 0.25f * hi;
          }
    };
    for (int it = 0; it < smooth; ++it)
      for (int ax = 0; ax < 3; ++ax) { pass(mask, ax); pass(spd, ax); pass(inl, ax); pass(outl, ax); }
  }

  // ---- coarse occupancy, for empty-space skipping -------------------------
  // Only ~16% of this box is fluid, so most of every ray is spent in wall. The
  // block grid lets a ray leave an empty block in one jump instead of stepping
  // through it half a voxel at a time.
  const int B = 8;
  const int bx = (nx + B - 1) / B, by = (ny + B - 1) / B, bz = (nz + B - 1) / B;
  std::vector<unsigned char> occ(std::size_t(bx) * by * bz, 0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        if (mask[(std::size_t(z) * ny + y) * nx + x] > 0.5f)
          occ[(std::size_t(z / B) * by + y / B) * bx + x / B] = 1;
  // dilate by one block so interpolation near a block face never reads a block
  // that was skipped
  {
    std::vector<unsigned char> d2 = occ;
    for (int k = 0; k < bz; ++k)
      for (int j = 0; j < by; ++j)
        for (int i = 0; i < bx; ++i)
          if (occ[(std::size_t(k) * by + j) * bx + i])
            for (int dk = -1; dk <= 1; ++dk)
              for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                  const int I = i + di, J = j + dj, K = k + dk;
                  if (I < 0 || I >= bx || J < 0 || J >= by || K < 0 || K >= bz) continue;
                  d2[(std::size_t(K) * by + J) * bx + I] = 1;
                }
    occ.swap(d2);
  }

  // ---- camera -------------------------------------------------------------
  const double ar = az * M_PI / 180.0, er = el * M_PI / 180.0;
  const Vec d = {-std::cos(er) * std::cos(ar), -std::cos(er) * std::sin(ar), -std::sin(er)};
  const Vec up = {0, 0, 1};
  const Vec rr = norm(cross(d, up));
  const Vec uu = cross(rr, d);
  const Vec c = {(nx - 1) * 0.5, (ny - 1) * 0.5, (nz - 1) * 0.5};

  // frame to the projected bounding box, so the vessel fills the image without
  // the framing changing between frames
  double eu = 0, ev = 0;
  for (int i = 0; i < 8; ++i) {
    const Vec p = {(i & 1) ? nx - 1.0 : 0.0, (i & 2) ? ny - 1.0 : 0.0, (i & 4) ? nz - 1.0 : 0.0};
    const Vec q = p - c;
    eu = std::max(eu, std::fabs(dot(q, rr)));
    ev = std::max(ev, std::fabs(dot(q, uu)));
  }
  const double ps = std::max(eu / (W * 0.5), ev / (H * 0.5)) * 1.04;
  const double spanU = ps * W * 0.5, spanV = ps * H * 0.5;
  const double half = std::sqrt(dot(c, c)) + 2.0;

  // key light over the viewer's shoulder
  const Vec lt = norm(Vec{-d.x, -d.y, -d.z} + rr * 0.42 + uu * 0.5);

  const double bg[3] = {243, 244, 247};
  const double K_SURF = ksurf;   // shell opacity per unit mask gradient
  const double A0 = a0;          // interior floor, so the lumen tints at all
  const double A1 = a1;          // interior opacity at full speed
  const double GREF = gref;      // mask gradient counted as "wall"
  const double AMB = 0.55;
  const double KSL = ksl;        // streamline emission strength

  std::vector<unsigned char> img(std::size_t(W) * H * 3);
  const std::size_t nxny = std::size_t(nx) * ny;

  for (int py = 0; py < H; ++py) {
    const double sv = (1.0 - 2.0 * (py + 0.5) / H) * spanV;
    for (int px = 0; px < W; ++px) {
      const double su = (2.0 * (px + 0.5) / W - 1.0) * spanU;
      const Vec o = c + rr * su + uu * sv - d * half * 1.6;

      // slab intersection with [0, n-1]^3
      double t0 = -1e30, t1 = 1e30;
      bool hit = true;
      const double oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
      const double hi[3] = {nx - 1.001, ny - 1.001, nz - 1.001};
      for (int k = 0; k < 3; ++k) {
        if (std::fabs(dd[k]) < 1e-12) {
          if (oo[k] < 1.0 || oo[k] > hi[k]) { hit = false; break; }
        } else {
          double a = (1.0 - oo[k]) / dd[k], b = (hi[k] - oo[k]) / dd[k];
          if (a > b) std::swap(a, b);
          t0 = std::max(t0, a); t1 = std::min(t1, b);
        }
      }
      double R = 0, G = 0, Bc = 0, A = 0;
      if (hit && t1 > t0) {
        for (double t = std::max(t0, 0.0); t < t1 && A < 0.995; ) {
          const double x = o.x + d.x * t, y = o.y + d.y * t, z = o.z + d.z * t;
          const int i0 = int(x), j0 = int(y), k0 = int(z);
          // empty-block skip
          if (!occ[(std::size_t(k0 / B) * by + j0 / B) * bx + i0 / B]) {
            double adv = 1e30;
            for (int k = 0; k < 3; ++k) {
              const double dk = dd[k];
              if (std::fabs(dk) < 1e-12) continue;
              const int blk = (k == 0 ? i0 : k == 1 ? j0 : k0) / B;
              const double bnd = (dk > 0 ? (blk + 1) * double(B) : blk * double(B));
              const double pk = (k == 0 ? x : k == 1 ? y : z);
              const double tt = (bnd - pk) / dk;
              if (tt > 0) adv = std::min(adv, tt);
            }
            t += (adv < 1e29 ? adv : step) + 1e-3;
            continue;
          }
          const double fx = x - i0, fy = y - j0, fz = z - k0;
          const std::size_t b0 = (std::size_t(k0) * ny + j0) * nx + i0;
          const std::size_t b1 = b0 + nx, b2 = b0 + nxny, b3 = b2 + nx;
          const double w000 = (1-fx)*(1-fy)*(1-fz), w100 = fx*(1-fy)*(1-fz);
          const double w010 = (1-fx)*fy*(1-fz),     w110 = fx*fy*(1-fz);
          const double w001 = (1-fx)*(1-fy)*fz,     w101 = fx*(1-fy)*fz;
          const double w011 = (1-fx)*fy*fz,         w111 = fx*fy*fz;
          auto tri = [&](const std::vector<float>& q) {
            return q[b0]*w000 + q[b0+1]*w100 + q[b1]*w010 + q[b1+1]*w110
                 + q[b2]*w001 + q[b2+1]*w101 + q[b3]*w011 + q[b3+1]*w111;
          };
          const double m = tri(mask);
          if (m < 0.02) { t += step; continue; }

          // analytic trilinear gradient of the mask -- the wall normal
          const double m000 = mask[b0],   m100 = mask[b0+1];
          const double m010 = mask[b1],   m110 = mask[b1+1];
          const double m001 = mask[b2],   m101 = mask[b2+1];
          const double m011 = mask[b3],   m111 = mask[b3+1];
          const double gx = (m100-m000)*(1-fy)*(1-fz) + (m110-m010)*fy*(1-fz)
                          + (m101-m001)*(1-fy)*fz     + (m111-m011)*fy*fz;
          const double gy = (m010-m000)*(1-fx)*(1-fz) + (m110-m100)*fx*(1-fz)
                          + (m011-m001)*(1-fx)*fz     + (m111-m101)*fx*fz;
          const double gz = (m001-m000)*(1-fx)*(1-fy) + (m101-m100)*fx*(1-fy)
                          + (m011-m010)*(1-fx)*fy     + (m111-m110)*fx*fy;
          const double gl = std::sqrt(gx*gx + gy*gy + gz*gz);

          double shade = 0.92;
          if (gl > 1e-9) {
            double dp = -(gx*lt.x + gy*lt.y + gz*lt.z) / gl;
            if (dp < 0) dp = -dp;                 // light both faces of the wall
            shade = AMB + (1.0 - AMB) * dp;
          }
          const double s = std::min(1.0, tri(spd) / vmax);

          // surface term: pale shell, or a cap colour where the face is driven
          double w = gl / GREF; if (w > 1.0) w = 1.0;
          const double ain = tri(inl), aout = tri(outl);
          double sr = 196, sg = 205, sb = 228;
          if (ain > 0.15)       { sr =  22; sg = 132; sb =  58; }
          else if (aout > 0.15) { sr = 198; sg =  52; sb =  40; }
          const double a_surf = K_SURF * w * m;

          // interior term: speed
          double ir, ig, ib; ramp(s, ir, ig, ib);
          // Opacity ramps STEEPLY with speed. A ray crosses far more slow
          // near-wall fluid than fast core, so a sub-linear ramp lets the
          // boundary layer outvote the jet and the whole vessel goes pale.
          const double a_int = (A0 + A1 * std::pow(s, agam)) * m;

          // streamline filaments, emissive
          double a_sl = 0.0, lr = 0, lg = 0, lb = 0;
          if (!sl_int.empty()) {
            const double si = tri(sl_int);
            if (si > 0.03) {
              double ss = tri(sl_spd) / vmax; if (ss > 1.0) ss = 1.0;
              ramp(ss, lr, lg, lb);
              // lift the core toward white so a filament reads as a line rather
              // than as a smear of the same colour as the fluid around it
              const double lift = 0.38 * si;
              lr += (255 - lr) * lift; lg += (255 - lg) * lift; lb += (255 - lb) * lift;
              a_sl = KSL * si;
            }
          }

          const double a_raw = a_surf + a_int + a_sl;
          if (a_raw <= 1e-9) { t += step; continue; }
          const double cr = (a_surf * sr * shade + a_int * ir + a_sl * lr) / a_raw;
          const double cg = (a_surf * sg * shade + a_int * ig + a_sl * lg) / a_raw;
          const double cb = (a_surf * sb * shade + a_int * ib + a_sl * lb) / a_raw;

          const double al = 1.0 - std::exp(-a_raw * step);
          const double contrib = (1.0 - A) * al;
          R += contrib * cr; G += contrib * cg; Bc += contrib * cb;
          A += contrib;
          t += step;
        }
      }
      R += (1.0 - A) * bg[0]; G += (1.0 - A) * bg[1]; Bc += (1.0 - A) * bg[2];
      const std::size_t oi = (std::size_t(py) * W + px) * 3;
      img[oi]     = (unsigned char)(R  < 0 ? 0 : R  > 255 ? 255 : R);
      img[oi + 1] = (unsigned char)(G  < 0 ? 0 : G  > 255 ? 255 : G);
      img[oi + 2] = (unsigned char)(Bc < 0 ? 0 : Bc > 255 ? 255 : Bc);
    }
  }

  // ---- colour scale -------------------------------------------------------
  if (cbar) {
    // Left of the right edge by enough for the driver to write tick labels
    // beside it; the bar is drawn here but its numbers come from ffmpeg,
    // which has a font.
    const int BW = 15, BH = int(H * 0.30), BX = W - 92, BY = int(H * 0.31);
    for (int i = 0; i < BH; ++i) {
      double r, g, b;
      ramp(1.0 - double(i) / (BH - 1), r, g, b);
      for (int j = 0; j < BW; ++j) {
        const std::size_t o = (std::size_t(BY + i) * W + BX + j) * 3;
        img[o] = (unsigned char)r; img[o+1] = (unsigned char)g; img[o+2] = (unsigned char)b;
      }
    }
    auto line = [&](int x, int y) {
      if (x < 0 || x >= W || y < 0 || y >= H) return;
      const std::size_t o = (std::size_t(y) * W + x) * 3;
      img[o] = 120; img[o+1] = 128; img[o+2] = 145;
    };
    for (int i = -1; i <= BH; ++i) { line(BX - 1, BY + i); line(BX + BW, BY + i); }
    for (int j = -1; j <= BW; ++j) { line(BX + j, BY - 1); line(BX + j, BY + BH); }
    for (int k = 0; k <= 4; ++k) {
      const int y = BY + int(double(k) * (BH - 1) / 4.0);
      for (int j = 0; j < 5; ++j) line(BX + BW + 1 + j, y);
    }
  }

  // ---- cardiac-phase inset ------------------------------------------------
  // A pulsatile sequence is hard to read without knowing where in the beat each
  // frame sits -- systole and the matching point of diastole look similar in a
  // single still, and the eye cannot integrate a cycle from the vessel alone.
  // This draws the driving waveform with a marker at the current phase, so a
  // frame is self-describing when pulled out of the animation as a figure.
  if (phase >= 0.0) {
    const int IW = 170, IH = 62, IX = 26, IY = H - 96;
    auto put = [&](int x, int y, double r, double g, double b, double al) {
      if (x < 0 || x >= W || y < 0 || y >= H) return;
      const std::size_t o = (std::size_t(y) * W + x) * 3;
      img[o]     = (unsigned char)(img[o]     * (1 - al) + r * al);
      img[o + 1] = (unsigned char)(img[o + 1] * (1 - al) + g * al);
      img[o + 2] = (unsigned char)(img[o + 2] * (1 - al) + b * al);
    };
    auto prof = [](double ph) {
      const double w = 0.5 + 0.5 * std::sin(2.0 * M_PI * ph - M_PI / 2.0);
      return 0.15 + 0.85 * std::pow(w, 1.5);
    };
    for (int i = 0; i < IW; ++i) put(IX + i, IY + IH, 150, 158, 175, 0.7);
    int prev = -1;
    for (int i = 0; i < IW; ++i) {
      const int y = IY + IH - int(prof(double(i) / (IW - 1)) * IH);
      const int lo = prev < 0 ? y : std::min(prev, y);
      const int hi = prev < 0 ? y : std::max(prev, y);
      for (int yy = lo; yy <= hi; ++yy) put(IX + i, yy, 70, 96, 140, 0.85);
      prev = y;
    }
    const int mx = IX + int(phase * (IW - 1));
    const int my = IY + IH - int(prof(phase) * IH);
    for (int dy = -4; dy <= 4; ++dy)
      for (int dx = -4; dx <= 4; ++dx) {
        const double d2 = dx * dx + dy * dy;
        if (d2 <= 16.0) put(mx + dx, my + dy, 206, 56, 44, d2 <= 9.0 ? 1.0 : 0.55);
      }
  }

  std::ofstream out(outf, std::ios::binary);
  out << "P6\n" << W << " " << H << "\n255\n";
  out.write(reinterpret_cast<const char*>(img.data()), std::streamsize(img.size()));
  std::printf("  %s  %dx%dx%d  peak %.5g  scale %.5g  az=%.0f el=%.0f  %dx%d\n",
              outf.c_str(), nx, ny, nz, peak, vmax, az, el, W, H);
  return 0;
}
