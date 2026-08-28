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

  // ---- 8. each operator leaves ITS OWN equilibrium unchanged ----------------
  //
  // The two equilibria are not the same populations and it matters which one is
  // used. BGK relaxes toward the second-order truncation `feq`. The
  // central-moment operator sends every moment above second order to the
  // MAXWELLIAN value, which on D3Q27 is the product form -- so its fixed point
  // is the inverse transform of (rho, 0, 0, ...), not `feq`.
  //
  // They differ by the Galilean defects of the truncation, which are O(u^3). An
  // earlier version of this check fed `feq` to both operators and reported a
  // discrepancy of 2.5e-06 at u ~ 0.02 -- exactly u^3 -- which the FP32
  // tolerance was loose enough to swallow and FP64 was not. The operator was
  // never wrong; the test was asserting the wrong thing, and only one of the two
  // precisions could tell.
  {
    for (int which = 0; which < 2; ++which) {
      const Real rho = Real(1.0), u[3] = {Real(0.02), Real(0.01), Real(-0.015)};
      Real f[27];
      if (which == 0) {
        for (int i = 0; i < 27; ++i) f[i] = feq(i, rho, u[0], u[1], u[2]);
      } else {
        const Real du[3] = {Real(0), Real(0), Real(0)};
        Real Qf[3][3];
        eq_factors(du, Qf);
        Real k[27];
        for (int n = 0; n < 27; ++n) k[n] = eq_moment(rho, Qf, n);
        to_populations(k, u, f);
      }
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

  // ---- 11. the D3Q7 velocity set, and cs^2 = 1/4 ---------------------------
  //
  // The scalar and the magnetic field run on this lattice. Its sound speed is
  // NOT 1/3: seven velocities with w_0 = 2/8 and w_i = 1/8 force cs^2 = 1/4, and
  // using 1/3 by habit gives a diffusivity 33% wrong with nothing to show for it.
  {
    double sw = 0, sc[3] = {0, 0, 0}, scc[3][3] = {{0}};
    for (int i = 0; i < 7; ++i) {
      const double w = D3Q7::w(i);
      const int c[3] = {D3Q7::cx(i), D3Q7::cy(i), D3Q7::cz(i)};
      sw += w;
      for (int a = 0; a < 3; ++a) {
        sc[a] += w * c[a];
        for (int b = 0; b < 3; ++b) scc[a][b] += w * c[a] * c[b];
      }
    }
    check(std::fabs(sw - 1.0) < eps, "D3Q7 weights sum to 1", sw - 1.0);
    double m1 = 0; for (int a = 0; a < 3; ++a) m1 = worst(m1, sc[a]);
    check(std::fabs(m1) < eps, "D3Q7 first moment of the weights vanishes", m1);
    double m2 = 0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        m2 = worst(m2, scc[a][b] - (a == b ? 0.25 : 0.0));
    check(std::fabs(m2) < eps, "D3Q7 second moment is cs^2 delta with cs^2 = 1/4", m2);

    bool ok = true;
    for (int i = 1; i < 7; i += 2)
      if (D3Q7::cx(i) != -D3Q7::cx(i + 1) || D3Q7::cy(i) != -D3Q7::cy(i + 1) ||
          D3Q7::cz(i) != -D3Q7::cz(i + 1)) ok = false;
    check(ok && D3Q7::cx(0) == 0 && D3Q7::cy(0) == 0 && D3Q7::cz(0) == 0,
          "D3Q7 obeys the same pairing contract as D3Q27");
  }

  // ---- 12. opp() agrees with the velocity tables on BOTH lattices ----------
  {
    bool ok = true;
    for (int i = 0; i < 27; ++i) {
      const int o = opp(i);
      if (o < 0 || o > 26) { ok = false; continue; }
      if (D3Q27::cx(o) != -D3Q27::cx(i) || D3Q27::cy(o) != -D3Q27::cy(i) ||
          D3Q27::cz(o) != -D3Q27::cz(i)) ok = false;
    }
    for (int i = 0; i < 7; ++i) {
      const int o = opp(i);
      if (D3Q7::cx(o) != -D3Q7::cx(i) || D3Q7::cy(o) != -D3Q7::cy(i) ||
          D3Q7::cz(o) != -D3Q7::cz(i)) ok = false;
    }
    check(ok, "opp(i) reverses the velocity on D3Q27 and D3Q7");
  }

  // ---- 13. the scalar equilibrium carries the right two moments ------------
  //
  // sum_i h^eq = dT and sum_i c_a h^eq = T u_a. The second is the advective flux;
  // if it were wrong the scalar would still diffuse correctly and drift at the
  // wrong speed, which is the kind of error a plume plot does not reveal.
  {
    const Real T_ref = Real(0.4), dT = Real(0.07);
    const Real u[3] = {Real(0.031), Real(-0.019), Real(0.024)};
    double s = 0, f[3] = {0, 0, 0};
    for (int i = 0; i < 7; ++i) {
      const double h = scalar_eq<D3Q7>(i, dT, T_ref, u[0], u[1], u[2]);
      s += h;
      f[0] += h * D3Q7::cx(i); f[1] += h * D3Q7::cy(i); f[2] += h * D3Q7::cz(i);
    }
    check(std::fabs(s - double(dT)) < eps * 10, "sum h^eq = dT", s - double(dT));
    double e = 0;
    for (int a = 0; a < 3; ++a) e = worst(e, f[a] - double(T_ref + dT) * double(u[a]));
    check(std::fabs(e) < eps * 10, "sum c h^eq = T u", e);
  }

  // ---- 14. the induction equilibrium and its ANTISYMMETRIC flux ------------
  //
  // sum_i g^eq_a = B_a and sum_i c_b g^eq_a = u_b B_a - B_b u_a. The
  // antisymmetry of that second moment is what keeps div B from being generated:
  // it is a property of the equilibrium, not something imposed afterwards. A
  // symmetric error there produces a plausible field with a growing monopole.
  {
    const Real B[3] = {Real(0.031), Real(-0.017), Real(0.042)};
    const Real u[3] = {Real(0.021), Real(0.013), Real(-0.008)};
    double worst_m0 = 0, worst_m1 = 0, worst_anti = 0;
    double F[3][3];
    for (int a = 0; a < 3; ++a) {
      double s = 0;
      double fl[3] = {0, 0, 0};
      for (int i = 0; i < 7; ++i) {
        const double g = magnetic_eq<D3Q7>(i, a, B, u);
        s += g;
        fl[0] += g * D3Q7::cx(i); fl[1] += g * D3Q7::cy(i); fl[2] += g * D3Q7::cz(i);
      }
      worst_m0 = worst(worst_m0, s - double(B[a]));
      for (int b = 0; b < 3; ++b) {
        F[a][b] = fl[b];
        worst_m1 = worst(worst_m1,
                         fl[b] - (double(u[b]) * double(B[a]) - double(B[b]) * double(u[a])));
      }
    }
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) worst_anti = worst(worst_anti, F[a][b] + F[b][a]);
    check(std::fabs(worst_m0) < eps * 10, "sum g^eq = B", worst_m0);
    check(std::fabs(worst_m1) < eps * 10, "sum c_b g^eq_a = u_b B_a - B_b u_a", worst_m1);
    check(std::fabs(worst_anti) < eps * 10, "that flux is exactly antisymmetric", worst_anti);
  }

  // ---- 15. the Maxwell stress perturbs the stress and NOTHING else ---------
  //
  // sum df = 0, sum c df = 0, sum c_a c_b df = M_ab with M = |b|^2 I/2 - b b.
  // The first two are what make it safe to add to the equilibrium: it cannot
  // change mass or momentum, so no conservation property is touched.
  {
    const Real B[3] = {Real(0.05), Real(-0.03), Real(0.02)};
    const double b2 = 0.05 * 0.05 + 0.03 * 0.03 + 0.02 * 0.02;
    double s = 0, m1[3] = {0, 0, 0}, m2[3][3] = {{0}};
    for (int i = 0; i < 27; ++i) {
      const double d = maxwell(i, B);
      const int c[3] = {D3Q27::cx(i), D3Q27::cy(i), D3Q27::cz(i)};
      s += d;
      for (int a = 0; a < 3; ++a) {
        m1[a] += d * c[a];
        for (int b = 0; b < 3; ++b) m2[a][b] += d * c[a] * c[b];
      }
    }
    check(std::fabs(s) < eps * 10, "Maxwell term carries no mass", s);
    double e1 = 0; for (int a = 0; a < 3; ++a) e1 = worst(e1, m1[a]);
    check(std::fabs(e1) < eps * 10, "Maxwell term carries no momentum", e1);
    double e2 = 0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const double M = (a == b ? 0.5 * b2 : 0.0) - double(B[a]) * double(B[b]);
        e2 = worst(e2, m2[a][b] - M);
      }
    check(std::fabs(e2) < eps * 100, "sum c_a c_b df = |b|^2 delta/2 - b_a b_b", e2);
  }

  // ---- 16. the Maxwell term in CENTRAL moment space ------------------------
  //
  // The MHD central-moment operator does not hard-code closed forms; it builds
  // the term as populations and transforms it, which is exact at every order.
  // This checks that transform against the two closed forms the parent
  // implementation verified symbolically:
  //     k_ab  = M_ab                    (deviatoric basis: the rho cs^2 is gone)
  //     k_abc = -(u_a M_bc + u_b M_ac + u_c M_ab)
  {
    const Real B[3] = {Real(0.05), Real(-0.03), Real(0.02)};
    const Real ub[3] = {Real(0.02), Real(0.015), Real(-0.01)};
    const double b2 = 0.05 * 0.05 + 0.03 * 0.03 + 0.02 * 0.02;
    auto M = [&](int a, int b) {
      return (a == b ? 0.5 * b2 : 0.0) - double(B[a]) * double(B[b]);
    };
    const MaxwellMoments<true> kM(B, ub);

    const int I2D[3] = {mi(2, 0, 0), mi(0, 2, 0), mi(0, 0, 2)};
    const int I2S[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    const int I2Si[3] = {mi(1, 1, 0), mi(1, 0, 1), mi(0, 1, 1)};
    double e2 = 0;
    for (int a = 0; a < 3; ++a) e2 = worst(e2, double(kM[I2D[a]]) - M(a, a));
    for (int s2 = 0; s2 < 3; ++s2)
      e2 = worst(e2, double(kM[I2Si[s2]]) - M(I2S[s2][0], I2S[s2][1]));
    check(std::fabs(e2) < eps * 100, "Maxwell central moments, order 2: k_ab = M_ab", e2);

    // k_111 = -(u_x M_yz + u_y M_xz + u_z M_xy), the only pure third-order slot
    // whose basis functions are all phi_1 and so is monomial without correction.
    const double k111 = -(double(ub[0]) * M(1, 2) + double(ub[1]) * M(0, 2) +
                          double(ub[2]) * M(0, 1));
    const double e3 = double(kM[mi(1, 1, 1)]) - k111;
    check(std::fabs(e3) < eps * 100,
          "Maxwell central moments, order 3: k_abc = -(u_a M_bc + ...)", e3);
  }

  // ---- 17. the Guo source has the moments the force needs -----------------
  //
  // sum S = 0, sum c_a S = F_a, sum c_a c_b S = u_a F_b + u_b F_a. The last is
  // the one that makes the viscous stress right in a forced flow; getting it
  // wrong gives a parabola of the wrong amplitude, not an obviously broken one.
  {
    const Real F[3] = {Real(1e-5), Real(-4e-6), Real(2e-6)};
    const Real u[3] = {Real(0.02), Real(-0.01), Real(0.015)};
    double s = 0, m1[3] = {0, 0, 0}, m2[3][3] = {{0}};
    for (int i = 0; i < 27; ++i) {
      const double S = guo_source_raw<D3Q27>(i, F, u[0], u[1], u[2]);
      const int c[3] = {D3Q27::cx(i), D3Q27::cy(i), D3Q27::cz(i)};
      s += S;
      for (int a = 0; a < 3; ++a) {
        m1[a] += S * c[a];
        for (int b = 0; b < 3; ++b) m2[a][b] += S * c[a] * c[b];
      }
    }
    check(std::fabs(s) < eps * 1e-3, "Guo source carries no mass", s);
    double e1 = 0; for (int a = 0; a < 3; ++a) e1 = worst(e1, m1[a] - double(F[a]));
    check(std::fabs(e1) < eps * 1e-2, "sum c_a S = F_a", e1);
    double e2 = 0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        e2 = worst(e2, m2[a][b] - (double(u[a]) * double(F[b]) + double(u[b]) * double(F[a])));
    check(std::fabs(e2) < eps * 1e-2, "sum c_a c_b S = u_a F_b + u_b F_a", e2);
  }

  // ---- 18. a forced collision adds EXACTLY F to the momentum ---------------
  //
  // This is the sharpest statement of "the forcing is right", and it holds for
  // both operators for different reasons. Under BGK the half-shift in the
  // velocity contributes omega F / 2 and the source (1 - omega/2) F, summing to
  // F. Under central moments the force enters only the first-order slot, and the
  // inverse transform carries it to the raw momentum unchanged. Neither is
  // obvious; both are checkable.
  {
    const Real F[3] = {Real(2e-5), Real(-1e-5), Real(3e-6)};
    for (int which = 0; which < 2; ++which) {
      Real f[27];
      for (int i = 0; i < 27; ++i)
        f[i] = feq(i, Real(1.02), Real(0.03), Real(-0.02), Real(0.01)) *
               Real(1.0 + 0.02 * std::sin(double(i)));
      double p0[3] = {0, 0, 0};
      for (int i = 0; i < 27; ++i) {
        p0[0] += double(f[i]) * D3Q27::cx(i);
        p0[1] += double(f[i]) * D3Q27::cy(i);
        p0[2] += double(f[i]) * D3Q27::cz(i);
      }
      Macro m = macroscopic(f);
      Coupling cp;
      cp.F[0] = F[0]; cp.F[1] = F[1]; cp.F[2] = F[2];
      shift_velocity(m, cp.F);
      const Real om = Real(1.3);
      if (which == 0) collide_bgk_gen<true, false>(f, m, om, cp);
      else            collide_cm_gen<true, false>(f, m, om, Real(1.0), cp);
      double p1[3] = {0, 0, 0};
      for (int i = 0; i < 27; ++i) {
        p1[0] += double(f[i]) * D3Q27::cx(i);
        p1[1] += double(f[i]) * D3Q27::cy(i);
        p1[2] += double(f[i]) * D3Q27::cz(i);
      }
      double e = 0;
      for (int a = 0; a < 3; ++a) e = worst(e, p1[a] - p0[a] - double(F[a]));
      char buf[128];
      std::snprintf(buf, sizeof buf, "%s forced collision adds exactly F to the momentum",
                    which == 0 ? "BGK" : "central moments");
      check(std::fabs(e) < eps * 1e-2, buf, e);
    }
  }

  // ---- 19. the MHD equilibrium is a fixed point of both operators ----------
  //
  // For BGK the state is feq + df. For central moments it is whatever inverse
  // transform of (Maxwellian moments + Maxwell moments) is -- the equilibrium of
  // that operator, which is not the same populations. Both must be stationary,
  // and both must still report the rho and u they were built from.
  {
    const Real rho = Real(1.0), u[3] = {Real(0.02), Real(0.01), Real(-0.015)};
    const Real B[3] = {Real(0.04), Real(-0.02), Real(0.03)};
    Coupling cp;
    for (int a = 0; a < 3; ++a) { cp.B[a] = B[a]; cp.F[a] = Real(0); }

    {   // BGK
      Real f[27], g[27];
      for (int i = 0; i < 27; ++i) f[i] = feq(i, rho, u[0], u[1], u[2]) + maxwell(i, B);
      for (int i = 0; i < 27; ++i) g[i] = f[i];
      const Macro m = macroscopic(f);
      collide_bgk_gen<false, true>(g, m, Real(1.7), cp);
      double e = 0;
      for (int i = 0; i < 27; ++i) e = worst(e, double(g[i]) - double(f[i]));
      check(std::fabs(e) < eps * 20, "MHD BGK leaves its equilibrium unchanged", e);
    }
    {   // central moments
      const Real ub[3] = {u[0], u[1], u[2]};
      const Real du[3] = {Real(0), Real(0), Real(0)};
      Real Qf[3][3];
      eq_factors(du, Qf);
      const MaxwellMoments<true> kM(B, ub);
      Real k[27];
      for (int n = 0; n < 27; ++n) k[n] = eq_moment(rho, Qf, n) + kM[n];
      Real f[27], g[27];
      to_populations(k, ub, f);
      for (int i = 0; i < 27; ++i) g[i] = f[i];
      const Macro m = macroscopic(f);
      double eu = worst(worst(double(m.ux) - double(u[0]), double(m.uy) - double(u[1])),
                        double(m.uz) - double(u[2]));
      check(std::fabs(double(m.rho) - double(rho)) < eps * 20 && std::fabs(eu) < eps * 20,
            "MHD central equilibrium reports the rho and u it was built from", eu);
      collide_cm_gen<false, true>(g, m, Real(1.7), Real(1.0), cp);
      double e = 0;
      for (int i = 0; i < 27; ++i) e = worst(e, double(g[i]) - double(f[i]));
      check(std::fabs(e) < eps * 20, "MHD central moments leaves its equilibrium unchanged", e);
    }
  }

  // ---- 20. the scalar and magnetic collisions conserve what they must ------
  {
    Real h[7];
    for (int i = 0; i < 7; ++i) h[i] = Real(0.01 * (i + 1));
    const Real dT0 = scalar_deviation<D3Q7>(h);
    collide_scalar<D3Q7>(h, dT0, Real(0.3), Real(0.02), Real(-0.01), Real(0.015), Real(1.4));
    const double e = double(scalar_deviation<D3Q7>(h)) - double(dT0);
    check(std::fabs(e) < eps * 10, "scalar collision conserves the scalar", e);

    // B must be the moment of the populations being collided, exactly as the
    // node update computes it: the operator relaxes toward sum_i g^eq = B_a, so
    // feeding it a B that is not the current moment makes it CHANGE the field
    // rather than conserve it -- which is the correct behaviour and the reason
    // magnetic_node_update gathers all three components before colliding any.
    const Real u[3] = {Real(0.02), Real(0.01), Real(-0.01)};
    const Real B0[3] = {Real(0.03), Real(-0.02), Real(0.01)};
    Real g[3][7], B[3];
    for (int a = 0; a < 3; ++a) {
      for (int i = 0; i < 7; ++i)
        g[a][i] = magnetic_eq<D3Q7>(i, a, B0, u) * Real(1.0 + 0.05 * i);
      B[a] = Real(0);
      for (int i = 0; i < 7; ++i) B[a] += g[a][i];
    }
    double worst_b = 0;
    for (int a = 0; a < 3; ++a) {
      collide_magnetic<D3Q7>(g[a], a, B, u, Real(1.2));
      double b1 = 0; for (int i = 0; i < 7; ++i) b1 += g[a][i];
      worst_b = worst(worst_b, b1 - double(B[a]));
    }
    check(std::fabs(worst_b) < eps * 10, "magnetic collision conserves B_a", worst_b);
  }

  // ---- 21. Esoteric Pull on D3Q7 -----------------------------------------
  //
  // The same two tests as 9 and 10, on the seven-velocity lattice, because the
  // scalar and the magnetic field stream through the same code and a lattice
  // that broke the ordering contract would move populations in the wrong
  // directions while still producing a plausible field.
  {
    const int nx = 6, ny = 5, nz = 4;
    const long N = long(nx) * ny * nz;
    std::vector<Real> f(7 * N, Real(0)), ref(7 * N);
    for (long n = 0; n < 7 * N; ++n) ref[n] = Real(0.5 + 0.001 * double(n % 89));
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          Real in[7];
          const long n = node_id(x, y, z, nx, ny);
          for (int i = 0; i < 7; ++i) in[i] = ref[long(i) * N + n];
          init_scatter<0, D3Q7>(f.data(), N, x, y, z, nx, ny, nz, in);
        }
    double e = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          Real out[7];
          gather<0, D3Q7>(f.data(), N, x, y, z, nx, ny, nz, out);
          const long n = node_id(x, y, z, nx, ny);
          for (int i = 0; i < 7; ++i) e = worst(e, double(out[i]) - double(ref[long(i) * N + n]));
        }
    check(std::fabs(e) < 1e-12, "D3Q7: gather(init_scatter(x)) == x", e);
  }
  {
    const int nx = 7, ny = 6, nz = 5;
    const long N = long(nx) * ny * nz;
    bool all_ok = true;
    double worst_err = 0;
    for (int dir = 1; dir < 7; ++dir) {
      std::vector<Real> f(7 * N, Real(0));
      const int x0 = 3, y0 = 2, z0 = 2;
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            Real in[7] = {Real(0), Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
            if (x == x0 && y == y0 && z == z0) in[dir] = Real(1);
            init_scatter<0, D3Q7>(f.data(), N, x, y, z, nx, ny, nz, in);
          }
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            Real t[7];
            gather<0, D3Q7>(f.data(), N, x, y, z, nx, ny, nz, t);
            scatter<0, D3Q7>(f.data(), N, x, y, z, nx, ny, nz, t);
          }
      const int xt = wrap(x0 + D3Q7::cx(dir), nx);
      const int yt = wrap(y0 + D3Q7::cy(dir), ny);
      const int zt = wrap(z0 + D3Q7::cz(dir), nz);
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            Real out[7];
            gather<1, D3Q7>(f.data(), N, x, y, z, nx, ny, nz, out);
            for (int i = 0; i < 7; ++i) {
              const double expect = (x == xt && y == yt && z == zt && i == dir) ? 1.0 : 0.0;
              const double err = double(out[i]) - expect;
              if (std::fabs(err) > 1e-12) { all_ok = false; worst_err = worst(worst_err, err); }
            }
          }
    }
    check(all_ok, "D3Q7: streaming transports each direction by exactly c_i", worst_err);
  }

  std::printf("\n%s  (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
