#pragma once
//==============================================================================
//  Minimal VTK ImageData (.vti) writer -- ASCII.
//
//  Adequate for validation-sized grids. Milestone 6 replaces this with
//  XDMF + HDF5 so that output is parallel and does not scale with rank count.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"

#include <fstream>
#include <string>

namespace lbm {

template <class Solver>
void write_vti(const std::string& path, Solver& s) {
  s.compute_macroscopic();
  const Domain& d = s.domain();

  auto h_rho = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
  auto h_ux  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto h_uy  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto h_uz  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
  auto h_fl  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());

  std::ofstream f(path);
  f.precision(9);
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <ImageData WholeExtent=\"0 " << d.nx - 1 << " 0 " << d.ny - 1 << " 0 "
    << d.nz - 1 << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
    << "    <Piece Extent=\"0 " << d.nx - 1 << " 0 " << d.ny - 1 << " 0 "
    << d.nz - 1 << "\">\n      <PointData Scalars=\"rho\" Vectors=\"u\">\n";

  f << "        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n";
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) f << h_rho(d.id(x, y, z)) << ' ';
  f << "\n        </DataArray>\n";

  f << "        <DataArray type=\"Float64\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) {
        const Index n = d.id(x, y, z);
        f << h_ux(n) << ' ' << h_uy(n) << ' ' << h_uz(n) << ' ';
      }
  f << "\n        </DataArray>\n";

  f << "        <DataArray type=\"UInt8\" Name=\"flag\" format=\"ascii\">\n";
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) f << int(h_fl(d.id(x, y, z))) << ' ';
  f << "\n        </DataArray>\n"
    << "      </PointData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n";
}

}  // namespace lbm
