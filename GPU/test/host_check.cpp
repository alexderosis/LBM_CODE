//==============================================================================
//  Numerical checks for the CUDA core, compiled and run WITHOUT a GPU.
//
//  Everything in core.cuh and the indexing half of solver.cuh is marked LBM_HD,
//  which expands to nothing under a plain C++ compiler. So the arithmetic that
//  the kernels will execute can be exercised here, on any machine, before it is
//  ever launched on a device.
//
//  This is not a substitute for running on a GPU -- it cannot catch a launch
//  configuration error, a race, or a register-pressure problem. It catches the
//  things that are far more likely and far more expensive to debug remotely:
//  a wrong velocity table, a broken moment transform, a streaming step that
//  moves populations in the wrong direction.
//
//  Build:  c++ -std=c++17 -O2 -I../include host_check.cpp -o host_check
//==============================================================================
#include "lbm/core.cuh"
#include "lbm/solver.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace lbm;

static int failures = 0;

static void check(bool ok, const char* what, double detail = 0.0) {
  if (ok) {
    std::printf("  PASS  %s\n", what);
  } else {
    std::printf("  FAIL  %s   (%.3e)\n", what, detail);
    ++failures;
  }
}

static double worst(double a, double b) { return std::fabs(a) > std::fabs(b) ? a : b; }

int main() {
  std::printf("D3Q27 core checks, host build, Real = %s\n\n",
              sizeof(Real) == 4 ? "float" : "double");

  const double eps = (sizeof(Real) == 4) ? 2e-5 : 1e-12;

  // ---- 1. the velocity set --------------------------------------------------
  {
    double sw = 0, sc[3] = {0, 0, 0}, scc[3][3] = {{0}};
    for (int i = 0; i < 27; ++i) {
      const double w = D3Q27::w(i);
      const int c[3] = {D3Q27::cx(i), D3Q27::cy(i), D3Q27::cz(i)};
      sw += w;
      for (int a = 0; a < 3; ++a) {
        sc[a] += w * c[a];
        for (int b = 0; b < 3; ++b) scc[a][b] += w * c[a] * c[b];
      }
    }
    check(std::fabs(sw - 1.0) < eps, "weights sum to 1", sw - 1.0);
    double m1 = 0; for (int a = 0; a < 3; ++a) m1 = worst(m1, sc[a]);
    check(std::fabs(m1) < eps, "first moment of the weights vanishes", m1);
    double m2 = 0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        m2 = worst(m2, scc[a][b] - (a == b ? 1.0 / 3.0 : 0.0));
    check(std::fabs(m2) < eps, "second moment of the weights is cs^2 delta", m2);
  }

  // ---- 2. the pairing contract Esoteric Pull depends on ---------------------
  {
    bool ok = true;
    for (int i = 1; i < 27; i += 2)
      if (D3Q27::cx(i) != -D3Q27::cx(i + 1) || D3Q27::cy(i) != -D3Q27::cy(i + 1) ||
          D3Q27::cz(i) != -D3Q27::cz(i + 1)) ok = false;
    check(ok, "opp(i) == i+1 for every odd i");
    check(D3Q27::cx(0) == 0 && D3Q27::cy(0) == 0 && D3Q27::cz(0) == 0,
          "direction 0 is the rest population");
  }

  // ---- 3. the precomputed direction table -----------------------------------
  {
    bool ok = true;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c) {
          const int i = pi(a, b, c);
          if (i < 0 || D3Q27::cx(i) != a - 1 || D3Q27::cy(i) != b - 1 ||
              D3Q27::cz(i) != c - 1) ok = false;
        }
    check(ok, "pi(a,b,c) maps to the right velocity, all 27");
  }

  // ---- 4. equilibrium recovers its own moments ------------------------------
  {
    const Real rho = Real(1.037), ux = Real(0.031), uy = Real(-0.019), uz = Real(0.024);
    Real f[27];
    for (int i = 0; i < 27; ++i) f[i] = feq(i, rho, ux, uy, uz);
    const Macro m = macroscopic(f);
    check(std::fabs(m.rho - rho) < eps * 10, "sum feq = rho", m.rho - rho);
    double e = worst(worst(m.ux - ux, m.uy - uy), m.uz - uz);
    check(std::fabs(e) < eps * 10, "sum c feq = rho u", e);
  }

  // ---- 5. the central-moment transform round-trips --------------------------
  {
    Real f[27], f2[27], k[27];
    for (int i = 0; i < 27; ++i) f[i] = Real(0.001 * (i + 1) + 0.01 * ((i * 7) % 5));
    const Real ub[3] = {Real(0.021), Real(-0.013), Real(0.007)};
    to_moments(f, ub, k);
    to_populations(k, ub, f2);
    double e = 0;
    for (int i = 0; i < 27; ++i) e = worst(e, double(f2[i]) - double(f[i]));
    check(std::fabs(e) < eps, "to_populations(to_moments(f)) == f", e);
  }

  // ---- 6. equilibrium in CENTRAL moment space -------------------------------
  //
  // Shifting by the local velocity should annihilate the first moments and put
  // cs^2 rho on the second diagonal. This is the property the whole operator is
  // built on, so it is worth asserting rather than assuming.
  {
    const Real rho = Real(1.0), u[3] = {Real(0.037), Real(-0.021), Real(0.014)};
    Real f[27], k[27];
    for (int i = 0; i < 27; ++i) f[i] = feq(i, rho, u[0], u[1], u[2]);
    to_moments(f, u, k);
    check(std::fabs(double(k[mi(0, 0, 0)]) - double(rho)) < eps * 10,
          "central k_000 = rho", double(k[mi(0, 0, 0)]) - double(rho));
    double m1 = 0;
    m1 = worst(m1, k[mi(1, 0, 0)]);
    m1 = worst(m1, k[mi(0, 1, 0)]);
    m1 = worst(m1, k[mi(0, 0, 1)]);
    check(std::fabs(m1) < eps * 10, "central first moments vanish", m1);
    // The basis is DEVIATORIC in the second moment: fwd1d subtracts cs^2, so
    // the equilibrium value of the diagonal slots is zero, not cs^2 rho.
    double m2 = 0;
    m2 = worst(m2, k[mi(2, 0, 0)]);
    m2 = worst(m2, k[mi(0, 2, 0)]);
    m2 = worst(m2, k[mi(0, 0, 2)]);
    check(std::fabs(m2) < eps * 20, "central second diagonal vanishes (deviatoric basis)", m2);
  }

  // ---- 7. both operators conserve mass and momentum -------------------------
  {
    for (int which = 0; which < 2; ++which) {
      Real f[27];
      for (int i = 0; i < 27; ++i)
        f[i] = feq(i, Real(1.02), Real(0.03), Real(-0.02), Real(0.01)) *
               Real(1.0 + 0.02 * std::sin(double(i)));   // push it off equilibrium
      const Macro before = macroscopic(f);
      if (which == 0) collide_bgk(f, before, Real(1.2));
      else            collide_cm(f, before, Real(1.2), Real(1.0));
      const Macro after = macroscopic(f);
      const char* nm = which == 0 ? "BGK" : "central moments";
      char buf[128];
      std::snprintf(buf, sizeof buf, "%s conserves mass", nm);
      check(std::fabs(double(after.rho) - double(before.rho)) < eps * 10, buf,
            double(after.rho) - double(before.rho));
      double e = worst(worst(double(after.ux) - double(before.ux),
                             double(after.uy) - double(before.uy)),
                       double(after.uz) - double(before.uz));
      std::snprintf(buf, sizeof buf, "%s conserves momentum", nm);
      check(std::fabs(e) < eps * 10, buf, e);
    }
  }

  // ---- 8. equilibrium is a fixed point of both operators --------------------
  {
    for (int which = 0; which < 2; ++which) {
      const Real rho = Real(1.0), u[3] = {Real(0.02), Real(0.01), Real(-0.015)};
      Real f[27];
      for (int i = 0; i < 27; ++i) f[i] = feq(i, rho, u[0], u[1], u[2]);
      Real g[27]; for (int i = 0; i < 27; ++i) g[i] = f[i];
      const Macro m = macroscopic(f);
      if (which == 0) collide_bgk(g, m, Real(1.7));
      else            collide_cm(g, m, Real(1.7), Real(1.0));
      double e = 0;
      for (int i = 0; i < 27; ++i) e = worst(e, double(g[i]) - double(f[i]));
      check(std::fabs(e) < eps * 20,
            which == 0 ? "BGK leaves equilibrium unchanged"
                       : "central moments leaves equilibrium unchanged", e);
    }
  }

  // ---- 9. gather is the exact inverse of init_scatter -----------------------
  {
    const int nx = 6, ny = 5, nz = 4;
    const long N = long(nx) * ny * nz;
    std::vector<Real> f(27 * N, Real(0));
    std::vector<Real> ref(27 * N);
    for (long n = 0; n < 27 * N; ++n) ref[n] = Real(0.5 + 0.001 * double(n % 97));

    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          Real in[27];
          const long n = node_id(x, y, z, nx, ny);
          for (int i = 0; i < 27; ++i) in[i] = ref[long(i) * N + n];
          init_scatter<0>(f.data(), N, x, y, z, nx, ny, nz, in);
        }
    double e = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          Real out[27];
          gather<0>(f.data(), N, x, y, z, nx, ny, nz, out);
          const long n = node_id(x, y, z, nx, ny);
          for (int i = 0; i < 27; ++i)
            e = worst(e, double(out[i]) - double(ref[long(i) * N + n]));
        }
    check(std::fabs(e) < 1e-12, "gather(init_scatter(x)) == x", e);
  }

  // ---- 10. one step of pure streaming moves populations by exactly c_i ------
  //
  // The test the whole scheme lives or dies on. A marked population at node n0
  // in direction i must arrive at n0 + c_i, and nowhere else, after a single
  // gather/scatter cycle followed by a parity flip.
  {
    const int nx = 7, ny = 6, nz = 5;
    const long N = long(nx) * ny * nz;
    bool all_ok = true;
    double worst_err = 0;

    for (int dir = 1; dir < 27; ++dir) {
      std::vector<Real> f(27 * N, Real(0));
      const int x0 = 3, y0 = 2, z0 = 2;
      // lay down a single marked population
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            Real in[27] = {Real(0)};
            for (int i = 0; i < 27; ++i) in[i] = Real(0);
            if (x == x0 && y == y0 && z == z0) in[dir] = Real(1);
            init_scatter<0>(f.data(), N, x, y, z, nx, ny, nz, in);
          }
      // one step, identity collision, in place (each slot has one reader and
      // one writer and they are the same node, so order does not matter)
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            Real t[27];
            gather<0>(f.data(), N, x, y, z, nx, ny, nz, t);
            scatter<0>(f.data(), N, x, y, z, nx, ny, nz, t);
          }
      // the marker should now sit at n0 + c_dir, read at the next parity
      const int xt = wrap(x0 + D3Q27::cx(dir), nx);
      const int yt = wrap(y0 + D3Q27::cy(dir), ny);
      const int zt = wrap(z0 + D3Q27::cz(dir), nz);
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            Real out[27];
            gather<1>(f.data(), N, x, y, z, nx, ny, nz, out);
            for (int i = 0; i < 27; ++i) {
              const double expect =
                  (x == xt && y == yt && z == zt && i == dir) ? 1.0 : 0.0;
              const double err = double(out[i]) - expect;
              if (std::fabs(err) > 1e-12) { all_ok = false; worst_err = worst(worst_err, err); }
            }
          }
    }
    check(all_ok, "streaming transports each direction by exactly c_i", worst_err);
  }

  std::printf("\n%s  (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
