//==============================================================================
//  Height-field geometry reader: round trip, solid rule, and rejection.
//
//  A reader is not obviously worth a test until you notice how it fails. None of
//  the ways this one can go wrong produce an error: a transposed index order
//  rotates the city, a mis-parsed dx rescales it, an off-by-one in the solid
//  rule shaves a layer off every roof. Each of those runs to completion and
//  produces a plausible plume. So the checks below are all of the form "compare
//  against something independently known", never "did it crash".
//
//  Everything here is self-contained: the test writes its own .npy and
//  meta.json, so it needs no data files and runs anywhere. If the real urban
//  data happens to be present it is loaded and reported as well, but its
//  absence is not a failure.
//==============================================================================
#include "core/Types.hpp"
#include "io/HeightField.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("   %-58s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

template <class F>
static bool throws(F f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}

//------------------------------------------------------------------------------
// A deliberately ASYMMETRIC ramp: h = 10*i + j metres. Any transpose, flip or
// axis swap changes the values read back, which a symmetric test pattern would
// hide completely.
//------------------------------------------------------------------------------
static void write_case(const std::string& dir, Index nx, Index ny, Index nz,
                       double dx, const char* extra = "") {
  std::vector<float> h(std::size_t(nx) * std::size_t(ny));
  for (Index i = 0; i < nx; ++i)
    for (Index j = 0; j < ny; ++j)
      h[std::size_t(i) * std::size_t(ny) + std::size_t(j)] = float(10 * i + j);
  npy::write_f32(dir + "/t_heights.npy", h, nx, ny, 1);

  std::ofstream m(dir + "/t_meta.json");
  m << "{\n  \"place\": \"synthetic ramp\",\n"
    << "  \"nx\": " << int(nx) << ",\n  \"ny\": " << int(ny) << ",\n"
    << "  \"nz\": " << int(nz) << ",\n  \"dx\": " << dx << extra << "\n}\n";
}

int main(int argc, char** argv) {
  // The first argument that is NOT a flag. Kokkos parses its own --kokkos-*
  // options but leaves them in argv, so taking argv[1] blindly makes a scratch
  // directory out of "--kokkos-num-threads=4" and the run dies trying to write
  // into it. This is the only registered test that takes a positional argument,
  // and it broke the moment the suite started passing a thread count.
  std::string dir = ".";
  for (int i = 1; i < argc; ++i)
    if (argv[i][0] != '-') { dir = argv[i]; break; }
  std::printf("\nHeight-field geometry reader\n%s\n\n", std::string(70, '=').c_str());

  const Index nx = 7, ny = 5, nz = 9;
  const double dx = 4.0;
  write_case(dir, nx, ny, nz, dx);

  std::printf("1. ROUND TRIP  (asymmetric ramp h = 10i + j, so a transpose shows)\n\n");
  const HeightField g = load_height_field(dir + "/t_heights.npy", dir + "/t_meta.json");
  check(g.nx == nx && g.ny == ny && g.nz == nz, "dimensions survive meta.json");
  check(std::abs(g.dx - dx) < 1e-12, "dx survives meta.json");
  check(g.place == "synthetic ramp", "place string is parsed");

  bool values = true, ordering = true;
  for (Index i = 0; i < nx; ++i)
    for (Index j = 0; j < ny; ++j)
      if (std::abs(g.at(i, j) - double(10 * i + j)) > 1e-6) values = false;
  // The distinguishing pair: at(1,0) and at(0,1) differ by 9 m, and a
  // transposed reader swaps them.
  if (std::abs(g.at(1, 0) - 10.0) > 1e-6 || std::abs(g.at(0, 1) - 1.0) > 1e-6)
    ordering = false;
  check(values,   "every height is the value that was written");
  check(ordering, "index order is i*ny + j, not transposed");

  std::printf("\n2. SOLID RULE  (cell centre at (k + 1/2) dx lies below the column)\n\n");
  // Column (0,0) has h = 0: no solid cell at all, because the first centre is
  // at dx/2 > 0. Column (2,0) has h = 20 = 5 dx: centres at 2, 6, 10, 14, 18
  // are below it and the one at 22 is not, so exactly 5 cells.
  check(!g.solid(0, 0, 0), "a zero-height column has no solid cell");
  int n20 = 0;
  for (Index k = 0; k < nz; ++k) if (g.solid(2, 0, k)) ++n20;
  check(n20 == 5, "h = 20 m at dx = 4 m fills exactly 5 cells");
  // The boundary case the closed form gets wrong: h/dx - 1/2 exactly integral.
  write_case(dir, 3, 3, 9, 2.0);
  const HeightField gb = load_height_field(dir + "/t_heights.npy", dir + "/t_meta.json");
  int nb = 0;
  for (Index k = 0; k < 9; ++k) if (gb.solid(0, 1, k)) ++nb;   // h = 1, dx = 2
  check(nb == 0, "h = 1 m at dx = 2 m fills 0 cells (centre 1.0 not below 1.0)");
  std::size_t direct = 0;
  for (Index i = 0; i < 3; ++i)
    for (Index j = 0; j < 3; ++j)
      for (Index k = 0; k < 9; ++k) direct += gb.solid(i, j, k) ? 1 : 0;
  check(direct == gb.solid_count(), "solid_count agrees with solid() cell by cell");

  std::printf("\n3. REJECTION  (each of these is silently wrong if accepted)\n\n");
  {
    std::ofstream m(dir + "/t_bad.json");
    m << "{ \"nx\": 7, \"ny\": 5, \"dx\": 4.0 }\n";       // no nz
  }
  check(throws([&]{ load_height_field(dir + "/t_heights.npy", dir + "/t_bad.json"); }),
        "a missing key is an error, not a default");
  {
    std::ofstream m(dir + "/t_bad.json");
    m << "{ \"nx\": 3, \"ny\": 3, \"nz\": 9, \"dx\": 2.0, \"dz\": 5.0 }\n";
  }
  check(throws([&]{ load_height_field(dir + "/t_heights.npy", dir + "/t_bad.json"); }),
        "dz differing from dx is rejected, not silently stretched");
  {
    std::ofstream m(dir + "/t_bad.json");
    m << "{ \"nx\": 40, \"ny\": 40, \"nz\": 9, \"dx\": 2.0 }\n";
  }
  check(throws([&]{ load_height_field(dir + "/t_heights.npy", dir + "/t_bad.json"); }),
        "a shape that disagrees with meta.json is rejected");
  check(throws([&]{ npy::read_f32(dir + "/t_meta.json", 0); }),
        "a file that is not a .npy is rejected");

  //----------------------------------------------------------------------------
  if (const char* p = std::getenv("LBM_CITY")) {
    std::printf("\n4. REAL DATA  ($LBM_CITY = %s)\n\n", p);
    try {
      const HeightField c = load_height_field(std::string(p) + "_heights.npy",
                                              std::string(p) + "_meta.json");
      report(c, "city");
    } catch (const std::exception& e) {
      std::printf("   could not load: %s\n", e.what());
    }
  }

  std::remove((dir + "/t_heights.npy").c_str());
  std::remove((dir + "/t_meta.json").c_str());
  std::remove((dir + "/t_bad.json").c_str());

  std::printf("\n%s\n\n", failures ? "FAILURES ABOVE" : "all checks passed");
  return failures ? 1 : 0;
}
