#pragma once
//==============================================================================
//  Urban geometry from a height field.
//
//  The second geometry source, alongside VoxelGeometry. Where that reads a
//  voxelised surface mesh (the aorta), this reads the extruded-footprint form an
//  urban model naturally produces: a 2D map of building height in metres, plus a
//  small JSON of grid metadata. A cell is solid when its centre lies below the
//  local building height, which turns O(nx*ny) of storage into an O(nx*ny*nz)
//  mask without ever materialising the latter on disk.
//
//  FORMAT, as written by the Pollutant project's build_voxel_city.py:
//
//      <name>_heights.npy   numpy v1/v2, '<f4', C order, shape (nx, ny), metres
//      <name>_meta.json     nx, ny, nz, dx  (and dz, place, ... which are read
//                           when present and ignored when not)
//
//  INDEX ORDER, which is the one thing here that can silently ruin a run. The
//  .npy is C order with shape (nx, ny), so height index = i*ny + j. That is
//  independent of how the 3D field is laid out, and Domain is x-fastest, so the
//  mapping to a node is simply x <- i, y <- j, z <- k with no transpose. Getting
//  this wrong does not fail: it rotates the city and everything downstream still
//  runs, which is why it is stated here rather than left to be inferred.
//
//  MISSING KEYS ARE AN ERROR, not a default. The obvious alternative -- fall
//  back to some built-in grid size -- turns a typo in meta.json into a silently
//  wrong domain, and a wrong domain is not distinguishable from a wrong
//  simulation once it has run.
//
//  Nothing here is device code; geometry is read once on the host.
//==============================================================================
#include "core/Types.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbm {

//------------------------------------------------------------------------------
// Minimal numpy reader/writer: flat arrays, C order, no pickled objects. Enough
// for what the urban pipeline exchanges, and small enough not to be a
// dependency. Anything it cannot represent is rejected rather than guessed at.
//------------------------------------------------------------------------------
namespace npy {

struct Header { std::string descr; std::size_t elements; };

inline Header read_header(std::ifstream& f, const std::string& path) {
  char magic[6];
  f.read(magic, 6);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("not a .npy file: " + path);
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  std::size_t hlen = 0;
  if (ver[0] == 1) {
    std::uint16_t n = 0; f.read(reinterpret_cast<char*>(&n), 2); hlen = n;
  } else {
    std::uint32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4); hlen = n;
  }
  std::string hdr(hlen, ' ');
  f.read(&hdr[0], std::streamsize(hlen));
  if (!f) throw std::runtime_error("truncated .npy header: " + path);
  if (hdr.find("'fortran_order': False") == std::string::npos)
    throw std::runtime_error(path + ": Fortran order is not supported");

  Header h;
  const std::size_t dp = hdr.find("'descr'");
  const std::size_t q1 = hdr.find('\'', hdr.find(':', dp) + 1);
  const std::size_t q2 = hdr.find('\'', q1 + 1);
  h.descr = hdr.substr(q1 + 1, q2 - q1 - 1);

  // Element count from the shape tuple, so a (nx, ny) and a (nx*ny,) file are
  // both acceptable -- the caller knows the grid, and only the total matters.
  const std::size_t sp = hdr.find("'shape'");
  const std::size_t o = hdr.find('(', sp), c = hdr.find(')', o);
  h.elements = 1;
  std::string dims = hdr.substr(o + 1, c - o - 1);
  for (char& ch : dims) if (ch == ',') ch = ' ';
  std::istringstream is(dims);
  std::size_t d; bool any = false;
  while (is >> d) { h.elements *= d; any = true; }
  if (!any) h.elements = 0;
  return h;
}

inline std::vector<float> read_f32(const std::string& path, std::size_t expect) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  const Header h = read_header(f, path);
  if (h.descr != "<f4")
    throw std::runtime_error(path + ": expected float32 ('<f4'), found '" + h.descr + "'");
  if (expect && h.elements != expect)
    throw std::runtime_error(path + ": holds " + std::to_string(h.elements) +
                             " elements, expected " + std::to_string(expect));
  std::vector<float> out(h.elements);
  f.read(reinterpret_cast<char*>(out.data()), std::streamsize(out.size() * 4));
  if (!f) throw std::runtime_error(path + ": truncated body");
  return out;
}

// Written back in the same dialect the project's Python renderers already read,
// so a field dumped here drops into the existing visualisation without a
// converter.
inline void write_f32(const std::string& path, const std::vector<float>& v,
                      Index d0, Index d1, Index d2) {
  char buf[256];
  const int n = std::snprintf(buf, sizeof buf,
      "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d), }",
      int(d0), int(d1), int(d2));
  std::string hdr(buf, std::size_t(n));
  const int total = 10 + n + 1;
  hdr.append(std::size_t((64 - total % 64) % 64), ' ');
  hdr.push_back('\n');
  const std::uint16_t hlen = std::uint16_t(hdr.size());

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write("\x93NUMPY", 6);
  const unsigned char ver[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(ver), 2);
  f.write(reinterpret_cast<const char*>(&hlen), 2);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
}

}  // namespace npy

//------------------------------------------------------------------------------
struct HeightField {
  Index nx = 0, ny = 0, nz = 0;
  double dx = 0;                  // metres per cell, isotropic (dz == dx)
  std::string place;
  std::vector<float> height;      // nx*ny, metres, index i*ny + j

  std::size_t columns() const { return std::size_t(nx) * std::size_t(ny); }
  std::size_t cells() const { return columns() * std::size_t(nz); }

  double at(Index i, Index j) const {
    return double(height[std::size_t(i) * std::size_t(ny) + std::size_t(j)]);
  }
  // Cell centres sit at (k + 1/2) dx, so a column of height h fills exactly the
  // cells whose centre is below h -- the same rule the source data was built
  // with, and the reason a zero-height column is empty rather than one cell deep.
  bool solid(Index i, Index j, Index k) const {
    return (double(k) + 0.5) * dx < at(i, j);
  }
  // Counted with solid() itself rather than a closed form. The obvious
  // expression, floor(h/dx - 1/2) + 1, is off by one whenever h/dx - 1/2 lands
  // exactly on an integer -- which is not a rare input here, since building
  // heights come from a generator that emits round numbers. solid() is
  // monotonic in k, so the early break keeps this O(columns + solid cells).
  std::size_t solid_count() const {
    std::size_t s = 0;
    for (Index i = 0; i < nx; ++i)
      for (Index j = 0; j < ny; ++j)
        for (Index k = 0; k < nz && solid(i, j, k); ++k) ++s;
    return s;
  }
  double max_height() const {
    double m = 0;
    for (float v : height) m = std::max(m, double(v));
    return m;
  }
};

//------------------------------------------------------------------------------
inline HeightField load_height_field(const std::string& heights_path,
                                     const std::string& meta_path) {
  std::ifstream mf(meta_path);
  if (!mf) throw std::runtime_error("cannot open " + meta_path);
  const std::string all((std::istreambuf_iterator<char>(mf)),
                        std::istreambuf_iterator<char>());

  // meta.json is small, flat and machine-written; pulling the few values out by
  // hand keeps a JSON library out of the solver's dependency list.
  auto number = [&](const char* key) -> double {
    const std::string k = std::string("\"") + key + "\"";
    const std::size_t p = all.find(k);
    if (p == std::string::npos)
      throw std::runtime_error(meta_path + ": missing required key \"" + key + "\"");
    const std::size_t c = all.find(':', p + k.size());
    if (c == std::string::npos)
      throw std::runtime_error(meta_path + ": malformed entry for \"" + key + "\"");
    return std::atof(all.c_str() + c + 1);
  };
  auto text = [&](const char* key) -> std::string {
    const std::string k = std::string("\"") + key + "\"";
    const std::size_t p = all.find(k);
    if (p == std::string::npos) return {};
    const std::size_t q1 = all.find('"', all.find(':', p + k.size()) + 1);
    if (q1 == std::string::npos) return {};
    const std::size_t q2 = all.find('"', q1 + 1);
    return all.substr(q1 + 1, q2 - q1 - 1);
  };

  HeightField g;
  g.nx = Index(number("nx"));
  g.ny = Index(number("ny"));
  g.nz = Index(number("nz"));
  g.dx = number("dx");
  g.place = text("place");
  if (g.nx <= 0 || g.ny <= 0 || g.nz <= 0)
    throw std::runtime_error(meta_path + ": non-positive grid dimensions");
  if (!(g.dx > 0))
    throw std::runtime_error(meta_path + ": dx must be positive");

  // dz is carried by the urban metadata but this code is isotropic; a file that
  // disagrees would silently stretch the city vertically.
  const std::size_t dzp = all.find("\"dz\"");
  if (dzp != std::string::npos) {
    const double dz = number("dz");
    if (std::abs(dz - g.dx) > 1e-9 * g.dx)
      throw std::runtime_error(meta_path + ": dz (" + std::to_string(dz) +
                               ") differs from dx (" + std::to_string(g.dx) +
                               "); anisotropic grids are not supported");
  }

  g.height = npy::read_f32(heights_path, g.columns());
  return g;
}

//------------------------------------------------------------------------------
inline void report(const HeightField& g, const char* name = "height field") {
  const std::size_t sol = g.solid_count(), tot = g.cells();
  std::printf("  %s: %d x %d x %d = %zu cells at %.2f m  (%.0f x %.0f x %.0f m)\n",
              name, int(g.nx), int(g.ny), int(g.nz), tot, g.dx,
              g.nx * g.dx, g.ny * g.dx, g.nz * g.dx);
  if (!g.place.empty()) std::printf("    %s\n", g.place.c_str());
  std::printf("    solid %zu (%.2f%%)   tallest column %.1f m of %.1f m modelled\n",
              sol, 100.0 * double(sol) / double(tot), g.max_height(), g.nz * g.dx);
  if (g.max_height() > g.nz * g.dx)
    std::printf("    WARNING buildings are taller than the domain and are truncated\n");
}

}  // namespace lbm
