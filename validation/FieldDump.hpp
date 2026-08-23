#pragma once
//==============================================================================
//  Field snapshots for the figures in doc/.
//
//  Writes a raw float32 slice as  int32 nx, int32 ny, then nx*ny floats in row
//  major order -- the format doc/fig/mkpng.py already reads for the
//  Orszag-Tang figures. Kept deliberately dumb: no compression, no metadata,
//  because the renderer supplies the colour map and the caption supplies the
//  units.
//
//  Everything here is diagnostic. Nothing in it feeds back into a solver, and
//  the derived quantities (vorticity, current, speed) use plain second-order
//  central differences with periodic wrap, which is what the validation tests
//  already use when they report peak values.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace lbm::figdump {

inline void write_raw(const std::string& path, int nx, int ny,
                      const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
  std::printf("  wrote %s (%d x %d)\n", path.c_str(), nx, ny);
}

// Host mirror of a device field, for convenience at the call sites.
template <class View>
inline auto host(const View& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace{}, v);
}

//------------------------------------------------------------------------------
// A z = const slice of a scalar field. `get(x, y)` returns the value.
//------------------------------------------------------------------------------
template <class Fn>
inline void scalar_slice(const std::string& path, Index nx, Index ny, Fn&& get) {
  std::vector<float> f(std::size_t(nx) * std::size_t(ny));
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x)
      f[std::size_t(y) * std::size_t(nx) + std::size_t(x)] = float(get(x, y));
  write_raw(path, int(nx), int(ny), f);
}

//------------------------------------------------------------------------------
// Out-of-plane curl of a 2D vector field, (d vy/dx - d vx/dy), central
// differences with periodic wrap. Used for vorticity from u and for the current
// from b -- the same operator, which is why they share one function.
//------------------------------------------------------------------------------
template <class GetX, class GetY>
inline void curl_z_slice(const std::string& path, Index nx, Index ny,
                         GetX&& vx, GetY&& vy) {
  auto wx = [&](Index i) { return ((i % nx) + nx) % nx; };
  auto wy = [&](Index j) { return ((j % ny) + ny) % ny; };
  std::vector<float> f(std::size_t(nx) * std::size_t(ny));
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const double c = 0.5 * (double(vy(wx(x + 1), y)) - double(vy(wx(x - 1), y)))
                     - 0.5 * (double(vx(x, wy(y + 1))) - double(vx(x, wy(y - 1))));
      f[std::size_t(y) * std::size_t(nx) + std::size_t(x)] = float(c);
    }
  write_raw(path, int(nx), int(ny), f);
}

//------------------------------------------------------------------------------
// A full 3D scalar volume: int32 nx, ny, nz then nx*ny*nz floats, x fastest.
// For the volume renderer in doc/fig/vol3d.py.
//------------------------------------------------------------------------------
template <class Fn>
inline void scalar_volume(const std::string& path, Index nx, Index ny, Index nz,
                          Fn&& get) {
  std::vector<float> f(std::size_t(nx) * std::size_t(ny) * std::size_t(nz));
  std::size_t o = 0;
  for (Index z = 0; z < nz; ++z)
    for (Index y = 0; y < ny; ++y)
      for (Index x = 0; x < nx; ++x) f[o++] = float(get(x, y, z));
  std::ofstream out(path, std::ios::binary);
  const std::int32_t a = nx, b = ny, c = nz;
  out.write(reinterpret_cast<const char*>(&a), sizeof a);
  out.write(reinterpret_cast<const char*>(&b), sizeof b);
  out.write(reinterpret_cast<const char*>(&c), sizeof c);
  out.write(reinterpret_cast<const char*>(f.data()),
            std::streamsize(f.size() * sizeof(float)));
  std::printf("  wrote %s (%d x %d x %d)\n", path.c_str(), int(nx), int(ny), int(nz));
}

}  // namespace lbm::figdump
