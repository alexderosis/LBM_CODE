//==============================================================================
//  Hartmann flow: the MHD analogue of Poiseuille flow.
//
//  Reference: P. J. Dellar, "Moment-Based Boundary Conditions for Lattice
//  Boltzmann Magnetohydrodynamics", Sec. 3-4, Eqs. (13)-(15). The Kokkos twin is
//  ../../validation/hartmann.cpp and this follows it line for line.
//
//  A uniform body force F drives flow along a channel that a uniform field B0
//  spans crosswise. The flow stretches that field into a streamwise component b,
//  whose Lorentz force resists the motion. Axes: x ACROSS the channel with walls
//  at x = +-L, y ALONG it and periodic, z periodic.
//
//  ================ WHY THIS CASE COULD NOT BE RUN HERE BEFORE ================
//  TWO DIFFERENT BOUNDARY MECHANISMS MEET, and that is the point of the test:
//
//    u  -- regularised velocity walls (regularized.cuh), u = 0 at the wall NODE
//    B  -- moment-based walls (magnetic.cuh), B = (B0, 0, 0) at the wall NODE
//
//  Maxwell's equations make both components of B continuous across the wall, so
//  B simply takes its external applied value. BOTH conditions place the boundary
//  ON the grid point, so they agree about where the wall is.
//
//  Until regularised walls existed this tree had halfway bounce-back only, which
//  puts the no-slip plane half a cell OUTSIDE the node the magnetic condition
//  pins B on. A Hartmann layer is a handful of cells thick, so that half cell is
//  not a rounding error -- it is a channel of the wrong width with a boundary
//  layer resolved across it. Mixing the two was the reason GPU/'s own README
//  said a wall-bounded MHD benchmark still belonged to the parent.
//  ===========================================================================
//
//  EXACT SOLUTION, Eq. (14), with xi = x/L and H = B0 L / sqrt(nu eta):
//
//     b(xi) = (F L / B0) [ sinh(H xi)/sinh(H) - xi ]
//     u(xi) = (F L / B0) sqrt(eta/nu) coth(H) [ 1 - cosh(H xi)/cosh(H) ]
//
//  Both vanish at xi = +-1, and both should do so HERE to round-off rather than
//  to O(h^2) -- that is exactly what "boundary at the node" buys, and it is the
//  first thing to look at if this case ever regresses.
//
//  THE PLANE BEYOND EACH WALL IS Excluded. This code's indexing is periodic on
//  every axis, so "outside the fluid" is a geometry flag and nothing else; both
//  the regularised wall's unknown-direction mask and the magnetic wall's are
//  built from it. The fluid and the magnetic field are given DIFFERENT geometry
//  arrays for that reason: the fluid's wall nodes are RegWall, but to the
//  magnetic solver they are ordinary transport nodes, and marking them non-Fluid
//  there would tell its mask that the along-wall directions streamed from
//  outside -- which they did not.
//
//  Run:  ./hartmann [-n 65] [-ha 10] [-nu 0.02] [-u 0.02] [-op bgk|trt|cm]
//==============================================================================
#include "lbm/backend.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace lbm;

struct RestInit {
  LBM_HD Macro operator()(int, int, int) const {
    return Macro{Real(1), Real(0), Real(0), Real(0), Real(1)};
  }
};
struct UniformB {
  Real b0;
  LBM_HD void operator()(int, int, int, Real B[3]) const {
    B[0] = b0;  B[1] = Real(0);  B[2] = Real(0);
  }
};
struct ZeroU {
  LBM_HD void operator()(int, int, int, Real u[3]) const {
    u[0] = u[1] = u[2] = Real(0);
  }
};

int main(int argc, char** argv) {
  int n = 65, nyc = 4, nz = 4;
  double Ha = 10.0, nu = 0.02, utarget = 0.02;
  std::size_t steps = 200000;
  Op op = Op::BGK;
  const char* opname = "BGK";
  bool shifted = false;

  for (int i = 1; i < argc; ++i) {
    auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-n"))  { if (i+1<argc) n = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-ha")) num(Ha);
    else if (!std::strcmp(argv[i], "-nu")) num(nu);
    else if (!std::strcmp(argv[i], "-u"))  num(utarget);
    else if (!std::strcmp(argv[i], "-steps")) { if (i+1<argc) steps = std::size_t(std::atol(argv[++i])); }
    else if (!std::strcmp(argv[i], "-shifted")) shifted = true;
    else if (!std::strcmp(argv[i], "-op") && i + 1 < argc) {
      ++i;
      if      (!std::strcmp(argv[i], "trt")) { op = Op::TRT;            opname = "TRT"; }
      else if (!std::strcmp(argv[i], "cm"))  { op = Op::CentralMoments; opname = "CM";  }
    }
  }

  // Wall NODES are 1 and n-2; the planes outside them are Excluded. The channel
  // half-width is therefore (n-3)/2 and the centre is at (n-1)/2.
  const int w0 = 1, w1 = n - 2;
  const double L = 0.5 * double(w1 - w0);
  const double xc = 0.5 * double(w0 + w1);
  const double eta = nu;                                  // Pr_m = 1
  const double B0 = Ha * nu / L;                          // from H = B0 L / sqrt(nu eta)
  const double H = Ha;
  const double shape = (1.0 / std::tanh(H)) * (1.0 - 1.0 / std::cosh(H));
  const double F = utarget * B0 / (L * shape);
  const double A = F * L / B0;

  const backend::DeviceInfo dev = backend::device_info();
  std::printf("Hartmann flow   %s   %s storage   %s, %s\n", opname,
              shifted ? "shifted" : "raw", dev.name.c_str(),
              sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  n = %d   wall nodes %d..%d   L = %.1f   Ha = %.4g\n", n, w0, w1, L, Ha);
  std::printf("  nu = eta = %.5g   B0 = %.6e   F = %.6e   tau = %.4f\n\n",
              nu, B0, F, 1.0 / double(omega_from_viscosity(Real(nu))));

  //---- geometry ---------------------------------------------------------------
  const long N = long(n) * nyc * nz;
  // NN is named rather than written as std::size_t(N) at each use: two
  // function-style casts as the only arguments make `std::vector<T> v(a, b);` a
  // FUNCTION DECLARATION, and the error it produces names neither cause.
  const std::size_t NN = static_cast<std::size_t>(N);
  const std::uint8_t kFluid = Fluid, kExcluded = Excluded, kBulk = MagBulk;
  std::vector<std::uint8_t> geo_f(NN, kFluid);
  std::vector<std::uint8_t> geo_m(NN, kFluid);
  std::vector<RegWallSpec>  spec(NN);
  std::vector<std::uint8_t> mkind(NN, kBulk);
  std::vector<Real> wbx(NN, Real(0)), wby(NN, Real(0)), wbz(NN, Real(0));
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < nyc; ++y) {
      for (int x = 0; x < n; ++x) {
        const long id = node_id(x, y, z, n, nyc);
        if (x < w0 || x > w1) { geo_f[std::size_t(id)] = kExcluded;
                                geo_m[std::size_t(id)] = kExcluded; }
      }
      const long a = node_id(w0, y, z, n, nyc), b = node_id(w1, y, z, n, nyc);
      spec[std::size_t(a)] = RegWallSpec{NrmXm, 0, 0, 0};
      spec[std::size_t(b)] = RegWallSpec{NrmXp, 0, 0, 0};
      mkind[std::size_t(a)] = MagDirichlet;  wbx[std::size_t(a)] = Real(B0);
      mkind[std::size_t(b)] = MagDirichlet;  wbx[std::size_t(b)] = Real(B0);
    }

  //---- solvers ----------------------------------------------------------------
  backend::Magnetic mag(n, nyc, nz, Real(eta));
  mag.set_geometry(geo_m);
  mag.set_walls(mkind, wbx, wby, wbz);

  backend::Fluid fl(n, nyc, nz, op, Real(nu));
  // The driving force scales as 1/L^2, so a wide channel puts the dynamics into
  // ever smaller differences between populations of size w_i -- exactly the
  // cancellation shifted storage removes. See core.cuh.
  fl.set_shifted(shifted);
  fl.set_geometry(geo_f);
  fl.set_regularized_walls(spec);
  fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  BodyForce bf;  bf.fy = Real(F);              // the channel runs along y
  fl.set_force(bf, ForceUniform);

  mag.initialise_with(UniformB{Real(B0)}, ZeroU{});
  fl.initialise_with(RestInit{});
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  //---- march to steady state --------------------------------------------------
  // The coupling must be SIMULTANEOUS: refresh B, step the fluid against it,
  // then step B against the velocity the fluid just wrote. Lagging it is a
  // first-order splitting error that does not vanish under refinement -- see the
  // banner in solver.cuh.
  // TWO GUARDS, AND THE SECOND ONE WAS LEARNED THE HARD WAY.
  //
  // A relative threshold of 1e-11 is BELOW THE FP32 FLOOR: with u ~ 2e-2 the
  // smallest representable change between samples is ~2.4e-9, so consecutive
  // probes agree to 1e-11 the moment the creep drops under precision -- long
  // before steady state. Quoting that gave an l2 error at L = 63 that was
  // WORSE than at L = 31 and broke the convergence, with nothing in the output
  // to say the run had stopped early.
  //
  // So the threshold follows the precision, and a MINIMUM of three diffusive
  // times L^2/nu must have elapsed regardless. At L = 63, nu = 0.02 that is
  // 595 000 steps: this case is slow because it is diffusive, and no residual
  // test substitutes for knowing that.
  const std::size_t probe = 2000;
  const double tol = (sizeof(Real) == 4) ? 1e-7 : 1e-11;
  const std::size_t t_min = std::size_t(3.0 * (2.0 * L) * (2.0 * L) / nu);
  if (steps < t_min + probe) steps = t_min + probe;
  double prev = 0;
  std::size_t taken = 0;
  for (std::size_t t = 0; t < steps; t += probe) {
    for (std::size_t k = 0; k < probe; ++k) {
      mag.compute_field();
      fl.step();
      mag.step();
    }
    taken += probe;
    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    const double cur = double(uy[std::size_t(node_id(n / 2, nyc / 2, nz / 2, n, nyc))]);
    if (!std::isfinite(cur)) { std::printf("  DIVERGED at step %zu\n", taken); return 1; }
    if (taken >= t_min && t > 0 &&
        std::fabs(cur - prev) < tol * (std::fabs(cur) + 1e-30)) break;
    prev = cur;
  }

  //---- compare ----------------------------------------------------------------
  std::vector<Real> rho, ux, uy, uz, bx, by, bz;
  fl.macroscopic_to_host(rho, ux, uy, uz);
  mag.field_to_host(bx, by, bz);

  std::printf("   x/L        u (LB)        u exact       b (LB)        b exact\n");
  double su = 0, sb = 0, umax = 0, uref = 0, bref = 0;
  int cnt = 0;
  for (int x = w0; x <= w1; ++x) {
    const double xi = (double(x) - xc) / L;
    const double au = A * std::sqrt(eta / nu) / std::tanh(H) * (1.0 - std::cosh(H * xi) / std::cosh(H));
    const double ab = A * (std::sinh(H * xi) / std::sinh(H) - xi);
    const long id = node_id(x, nyc / 2, nz / 2, n, nyc);
    const double nu_ = double(uy[std::size_t(id)]);
    const double nb_ = double(by[std::size_t(id)]);
    su += (nu_ - au) * (nu_ - au);  sb += (nb_ - ab) * (nb_ - ab);
    uref += au * au;  bref += ab * ab;  ++cnt;
    umax = std::fmax(umax, std::fabs(nu_));
    if ((x - w0) % ((w1 - w0) / 8) == 0 || x == w1)
      std::printf("  %+6.3f   %12.5e  %12.5e  %12.5e  %12.5e\n", xi, nu_, au, nb_, ab);
  }
  const double eu = std::sqrt(su / uref), eb = std::sqrt(sb / bref);
  std::printf("\n  steps %zu (min %zu = 3 diffusive times)   u_max %.6e (target %.4g)\n",
              taken, t_min, umax, utarget);
  std::printf("  l2 error   u %.4e   b %.4e        Eq. (15)\n", eu, eb);
  std::printf("  AT THE WALL: u %.3e / %.3e   b %.3e / %.3e   (both should be ~0)\n",
              std::fabs(double(uy[std::size_t(node_id(w0, nyc/2, nz/2, n, nyc))])),
              std::fabs(double(uy[std::size_t(node_id(w1, nyc/2, nz/2, n, nyc))])),
              std::fabs(double(by[std::size_t(node_id(w0, nyc/2, nz/2, n, nyc))])),
              std::fabs(double(by[std::size_t(node_id(w1, nyc/2, nz/2, n, nyc))])));
  const bool ok = (eu < 0.02 && eb < 0.05);
  std::printf("\n  %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
