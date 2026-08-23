#pragma once
//==============================================================================
//  Voxel geometry from file.
//
//  Reads the flat binary a surface mesh is voxelised into, so a simulation can
//  be driven by an arbitrary geometry rather than an analytic one. The format is
//  the one already produced by the aorta project's scripts/voxelize.py:
//
//      header  int32   nx, ny, nz
//              float64 pitch, ox, oy, oz          voxel size and grid origin
//              float64 in_x, in_y, in_z           inward unit normal at the inlet
//            [ float64 in2_x, in2_y, in2_z ]      OPTIONAL second inlet normal
//      body    uint8   tag[nx*ny*nz]
//
//  with the body flattened x-fastest, index = i + nx*(j + ny*k). That is the
//  same ordering Domain uses, so a tag maps to a node without a transpose.
//
//  The optional second normal is detected by size: a file carrying it is exactly
//  24 bytes longer than header plus body. Anything else is a truncated or
//  corrupt file and is rejected rather than guessed at.
//
//  TAGS. 0 solid, 1 fluid, 2 inlet, 3 outlet, 4 second inlet. They are kept as
//  raw tags rather than being mapped to CellType on load, because the caller
//  decides what an inlet IS -- a regularised velocity wall, a pressure
//  condition, or an interior node driven some other way -- and that choice is
//  not the reader's to make.
//
//  Nothing here is device code: geometry is read once on the host and consumed
//  by set_geometry and set_regularized_walls, which already run on the host.
//==============================================================================
#include "boundary/Flags.hpp"
#include "core/Types.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbm {

struct VoxelGeometry {
  // Tag values as written by the voxeliser.
  enum Tag : std::uint8_t {
    TagSolid   = 0,
    TagFluid   = 1,
    TagInlet   = 2,
    TagOutlet  = 3,
    TagInlet2  = 4,
  };

  Index nx = 0, ny = 0, nz = 0;
  double pitch = 0;                       // physical voxel size, file units
  std::array<double, 3> origin{0, 0, 0};
  std::array<double, 3> inlet_normal{0, 0, 0};    // points INTO the domain
  std::array<double, 3> inlet2_normal{0, 0, 0};
  bool has_inlet2 = false;
  std::vector<std::uint8_t> tag;          // nx*ny*nz, x fastest

  std::size_t count() const { return std::size_t(nx) * std::size_t(ny) * std::size_t(nz); }

  // Tag at a grid point, in the file's own indexing.
  std::uint8_t at(Index x, Index y, Index z) const {
    return tag[(std::size_t(z) * std::size_t(ny) + std::size_t(y)) * std::size_t(nx)
               + std::size_t(x)];
  }

  std::size_t count_of(std::uint8_t t) const {
    std::size_t n = 0;
    for (std::uint8_t v : tag) n += (v == t);
    return n;
  }

  // The obvious mapping, offered as a default rather than imposed on load:
  // solid stays solid, everything else is fluid until the caller says otherwise.
  CellType cell_at(Index x, Index y, Index z) const {
    return at(x, y, z) == TagSolid ? Solid : Fluid;
  }
};

//------------------------------------------------------------------------------
inline VoxelGeometry load_voxel_geometry(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open geometry file: " + path);

  f.seekg(0, std::ios::end);
  const std::streamoff bytes = f.tellg();
  f.seekg(0, std::ios::beg);

  VoxelGeometry g;
  std::int32_t n[3];
  f.read(reinterpret_cast<char*>(n), sizeof n);
  if (!f) throw std::runtime_error("geometry file too short for a header: " + path);
  if (n[0] <= 0 || n[1] <= 0 || n[2] <= 0)
    throw std::runtime_error("geometry file has non-positive dimensions: " + path);
  g.nx = Index(n[0]); g.ny = Index(n[1]); g.nz = Index(n[2]);

  double h[7];
  f.read(reinterpret_cast<char*>(h), sizeof h);
  if (!f) throw std::runtime_error("geometry file truncated in the header: " + path);
  g.pitch = h[0];
  g.origin = {h[1], h[2], h[3]};
  g.inlet_normal = {h[4], h[5], h[6]};

  // Size decides whether a second inlet normal is present. Guessing from the
  // stream state instead would silently read 24 bytes of the tag array as
  // doubles on a file that does not have it.
  const std::streamoff base = 12 + 7 * 8;
  const std::streamoff body = std::streamoff(g.count());
  if (bytes == base + body + 24) {
    double h2[3];
    f.read(reinterpret_cast<char*>(h2), sizeof h2);
    g.inlet2_normal = {h2[0], h2[1], h2[2]};
    g.has_inlet2 = true;
  } else if (bytes != base + body) {
    throw std::runtime_error(
        "geometry file " + path + " is " + std::to_string(bytes) +
        " bytes; expected " + std::to_string(base + body) + " (no second inlet) or " +
        std::to_string(base + body + 24) + " (with one)");
  }

  g.tag.resize(g.count());
  f.read(reinterpret_cast<char*>(g.tag.data()), body);
  if (!f) throw std::runtime_error("geometry file truncated in the body: " + path);

  for (std::uint8_t v : g.tag)
    if (v > VoxelGeometry::TagInlet2)
      throw std::runtime_error("geometry file " + path + " has tag " +
                               std::to_string(int(v)) + ", outside 0..4");
  return g;
}

//------------------------------------------------------------------------------
inline void report(const VoxelGeometry& g, const char* name = "geometry") {
  const std::size_t tot = g.count();
  std::printf("  %s: %d x %d x %d = %zu voxels, pitch %.6g\n",
              name, int(g.nx), int(g.ny), int(g.nz), tot, g.pitch);
  std::printf("    origin (%.4g, %.4g, %.4g)   inlet normal (%.4f, %.4f, %.4f)\n",
              g.origin[0], g.origin[1], g.origin[2],
              g.inlet_normal[0], g.inlet_normal[1], g.inlet_normal[2]);
  if (g.has_inlet2)
    std::printf("    second inlet normal (%.4f, %.4f, %.4f)\n",
                g.inlet2_normal[0], g.inlet2_normal[1], g.inlet2_normal[2]);
  const char* nm[5] = {"solid", "fluid", "inlet", "outlet", "inlet2"};
  for (int t = 0; t < 5; ++t) {
    const std::size_t c = g.count_of(std::uint8_t(t));
    if (c) std::printf("    tag %d %-7s %9zu  (%.2f%%)\n", t, nm[t], c, 100.0 * double(c) / double(tot));
  }
}

}  // namespace lbm
