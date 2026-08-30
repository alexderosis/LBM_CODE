//==============================================================================
//  Colour-gradient checks for the CUDA port, compiled and run WITHOUT a GPU.
//
//  Two halves, and the second is the one a port needs.
//
//  THE PHYSICS half re-derives the model's invariants from the code as written:
//  that phi_i is a weight set whose second moment is the phase's sound speed,
//  that the equilibrium carries rho and rho u with Phi_i included, that the
//  perturbation is a pure second-moment source delivering sigma = 4 A tau / 9,
//  and that recolouring is an exact partition. These mirror the parent's
//  tests/test_colour_gradient.cpp, because the failure they guard against is a
//  port silently reverting one of the three readings that had to be worked out
//  rather than copied from the paper.
//
//  THE INTEGRATION half runs an actual simulation through the three passes on a
//  small periodic lattice, driving the same LBM_HD node functions the kernels
//  will call. That is where a port fails: not in the algebra, which transcribes,
//  but in the Esoteric Pull parity, the neighbour wrap and the gather/scatter
//  pairing. A droplet that conserves both colours to machine precision over
//  fifty steps has exercised all three.
//
//  Build:  c++ -std=c++17 -O2 -I../include host_colour.cpp -o host_colour
//==============================================================================
#include "lbm/colour.cuh"

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
  std::printf("Colour-gradient checks, host build, Real = %s\n",
              sizeof(Real) == 4 ? "float" : "double");

  const double eps = (sizeof(Real) == 4) ? 2e-5 : 1e-11;
  // The identities in section 3 are EXACT in real arithmetic -- they are sums of
  // lattice constants that cancel algebraically -- so the only thing that can
  // separate them from zero is the representation of those constants. That is a
  // property of the precision, not of the model, and asserting a fixed 1e-12
  // against float constants tests the compiler rather than the code.
  const double ceps = (sizeof(Real) == 4) ? 1e-6 : 1e-12;

  //===========================================================================
  std::printf("\n1. phi_i is a weight set, and its second moment is the phase's cs^2\n\n");
  //===========================================================================
  for (double a : {8.0 / 27.0, 0.2, 0.6}) {
    double s = 0, m2[3][3] = {{0}}, m1[3] = {0, 0, 0};
    for (int i = 0; i < 27; ++i) {
      const double p = ColourModel::phi_i(i, Real(a));
      const int c[3] = {D3Q27::cx(i), D3Q27::cy(i), D3Q27::cz(i)};
      s += p;
      for (int x = 0; x < 3; ++x) {
        m1[x] += p * c[x];
        for (int y = 0; y < 3; ++y) m2[x][y] += p * c[x] * c[y];
      }
    }
    char buf[96];
    const double cs2 = 9.0 * (1.0 - a) / 19.0;
    std::snprintf(buf, sizeof buf, "sum phi_i = 1              (alpha = %.4f)", a);
    check(std::fabs(s - 1.0) < eps, buf, s - 1.0);
    std::snprintf(buf, sizeof buf, "second moment = 9(1-a)/19  (alpha = %.4f)", a);
    check(std::fabs(m2[0][0] - cs2) < eps, buf, m2[0][0] - cs2);
    check(std::fabs(m2[1][1] - cs2) < eps, "  ... isotropic in y", m2[1][1] - cs2);
    check(std::fabs(m2[2][2] - cs2) < eps, "  ... isotropic in z", m2[2][2] - cs2);
    double off = 0;
    for (int x = 0; x < 3; ++x)
      for (int y = 0; y < 3; ++y) if (x != y) off = worst(off, m2[x][y]);
    check(std::fabs(off) < eps, "  ... no off-diagonal part", off);
    double od = 0;
    for (int x = 0; x < 3; ++x) od = worst(od, m1[x]);
    check(std::fabs(od) < eps, "  ... odd moments vanish", od);
  }
  {
    const double cs2 = ColourModel::cs2_of_alpha(Real(8.0 / 27.0));
    check(std::fabs(cs2 - 1.0 / 3.0) < eps,
          "alpha = 8/27 gives cs^2 = 1/3 exactly", cs2 - 1.0 / 3.0);
    for (double g : {10.0, 100.0, 1000.0}) {
      const double ab = 8.0 / 27.0;
      const double ar = ColourModel::alpha_r_from_ratio(Real(g), Real(ab));
      const double back = (1.0 - ab) / (1.0 - ar);
      char buf[80];
      std::snprintf(buf, sizeof buf, "gamma = (1-ab)/(1-ar) inverts    (gamma = %.0f)", g);
      check(std::fabs(back - g) < 1e-4 * g, buf, back - g);
    }
  }

  //===========================================================================
  std::printf("\n2. the equilibrium carries rho and rho u, Phi_i included\n\n");
  //===========================================================================
  {
    ColourModel m;
    m.alpha_r = Real(ColourModel::alpha_r_from_ratio(Real(10), Real(8.0 / 27.0)));
    m.alpha_b = Real(8.0 / 27.0);
    m.rho_r0 = Real(10); m.rho_b0 = Real(1);
    const Real rr = Real(6), rb = Real(0.4);
    const Real u[3] = {Real(0.05), Real(-0.03), Real(0.02)};
    const Real dr[3] = {Real(0.3), Real(-0.1), Real(0.05)};
    Real G[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) G[i][j] = (u[i] * dr[j] + u[j] * dr[i]) / Real(48);
    const Real udrho = u[0] * dr[0] + u[1] * dr[1] + u[2] * dr[2];
    const Real nubar = Real(1.0 / 6.0);

    Real fe[27];
    m.equilibrium(fe, rr, rb, u, G, nubar, udrho);
    const double rho = double(rr) + double(rb);
    double s = 0, mom[3] = {0, 0, 0};
    for (int i = 0; i < 27; ++i) {
      s += fe[i];
      mom[0] += fe[i] * D3Q27::cx(i);
      mom[1] += fe[i] * D3Q27::cy(i);
      mom[2] += fe[i] * D3Q27::cz(i);
    }
    check(std::fabs(s - rho) < 1e-5 * rho, "sum f^eq = rho", s - rho);
    check(std::fabs(mom[0] - rho * u[0]) < 1e-5, "sum c_x f^eq = rho u_x", mom[0] - rho * u[0]);
    check(std::fabs(mom[1] - rho * u[1]) < 1e-5, "sum c_y f^eq = rho u_y", mom[1] - rho * u[1]);
    check(std::fabs(mom[2] - rho * u[2]) < 1e-5, "sum c_z f^eq = rho u_z", mom[2] - rho * u[2]);

    // Phi_i alone: the same equilibrium with G and udrho zeroed.
    Real Z[3][3] = {{0}};
    Real f0[27];
    m.equilibrium(f0, rr, rb, u, Z, nubar, Real(0));
    double dm = 0, dp[3] = {0, 0, 0}, sxy = 0;
    for (int i = 0; i < 27; ++i) {
      const double d = double(fe[i]) - double(f0[i]);
      dm += d;
      dp[0] += d * D3Q27::cx(i);
      dp[1] += d * D3Q27::cy(i);
      dp[2] += d * D3Q27::cz(i);
      sxy += d * D3Q27::cx(i) * D3Q27::cy(i);
    }
    check(std::fabs(dm) < 1e-5, "Phi_i changes no density", dm);
    double dpw = 0; for (int a = 0; a < 3; ++a) dpw = worst(dpw, dp[a]);
    check(std::fabs(dpw) < 1e-5, "Phi_i changes no momentum", dpw);
    check(std::fabs(sxy) > 1e-9,
          "Phi_i DOES change the stress (it would be dead code otherwise)", sxy);
    std::printf("        (its xy stress contribution is %.4e)\n", sxy);

    // At rest, p = sum_k rho_k cs_k^2 -- the per-colour rest term, which is the
    // reading that keeps pressure continuous through the interface.
    double tr = 0;
    for (int i = 0; i < 27; ++i) {
      const double e = m.eq_at_rest(i, rr, rb);
      tr += e * (D3Q27::cx(i) * D3Q27::cx(i));
    }
    const double want = double(rr) * ColourModel::cs2_of_alpha(m.alpha_r)
                      + double(rb) * ColourModel::cs2_of_alpha(m.alpha_b);
    check(std::fabs(tr - want) < 1e-5 * want, "p = sum_k rho_k cs_k^2 at rest", tr - want);
  }

  //===========================================================================
  std::printf("\n3. the perturbation is a pure second-moment source\n\n");
  //===========================================================================
  {
    double sB = 0;
    for (int i = 0; i < 27; ++i) sB += ColourModel::B_i(i);
    check(std::fabs(sB - 1.0 / 3.0) < eps, "sum B_i = 1/3", sB - 1.0 / 3.0);

    // Every direction of n, including ones off the lattice axes.
    const double dirs[5][3] = {{1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0.3, -0.7, 0.5}};
    double wm = 0, wp = 0, wnn = 0, tanmin = 1e30;
    for (auto& d : dirs) {
      const double L = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const double n[3] = {d[0] / L, d[1] / L, d[2] / L};
      double sm = 0, sp[3] = {0, 0, 0}, Snn = 0, Stt = 0;
      // A tangent to n, for the tangential stress.
      double t[3] = {-n[1], n[0], 0};
      double tl = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
      if (tl < 1e-8) { t[0] = 0; t[1] = -n[2]; t[2] = n[1];
                       tl = std::sqrt(t[1] * t[1] + t[2] * t[2]); }
      for (int a = 0; a < 3; ++a) t[a] /= tl;
      for (int i = 0; i < 27; ++i) {
        const double c[3] = {double(D3Q27::cx(i)), double(D3Q27::cy(i)), double(D3Q27::cz(i))};
        const double cn = c[0] * n[0] + c[1] * n[1] + c[2] * n[2];
        const double ct = c[0] * t[0] + c[1] * t[1] + c[2] * t[2];
        const double s = D3Q27::w(i) * cn * cn - ColourModel::B_i(i);
        sm += s;
        for (int a = 0; a < 3; ++a) sp[a] += s * c[a];
        Snn += s * cn * cn;
        Stt += s * ct * ct;
      }
      wm = worst(wm, sm);
      for (int a = 0; a < 3; ++a) wp = worst(wp, sp[a]);
      wnn = worst(wnn, Snn);
      if (std::fabs(Stt) < tanmin) tanmin = std::fabs(Stt);
    }
    check(std::fabs(wm) < ceps, "conserves mass, every direction of n", wm);
    check(std::fabs(wp) < ceps, "conserves momentum, every direction", wp);
    check(std::fabs(wnn) < ceps, "normal capillary stress vanishes, every direction of n", wnn);
    check(tanmin > 1e-3, "but the tangential one does not", tanmin);

    // sigma = 4 A tau / 9, Eq. (32). The capillary stress of a flat interface is
    // -tau sum_i Omega^(2) c_i c_i; integrating (S_tt - S_nn) across the
    // interface turns |grad phi| dn into dphi, which runs -1 to +1, so the
    // integral contributes a factor of 2.
    const double A = 0.01, tau = 0.8;
    double Txx = 0, Tyy = 0;
    for (int i = 0; i < 27; ++i) {
      const double cx = D3Q27::cx(i), cy = D3Q27::cy(i);
      const double s = D3Q27::w(i) * cx * cx - ColourModel::B_i(i);  // n = x
      Txx += s * cx * cx;
      Tyy += s * cy * cy;
    }
    const double sigma = -tau * A * (Tyy - Txx) * 2.0;
    const double want = ColourModel::sigma_from_A(Real(A), Real(tau));
    check(std::fabs(sigma - want) < ceps * std::fabs(want),
          "capillary stress gives sigma = 4 A tau / 9, Eq. (32)", (sigma - want) / want);
    std::printf("        (sigma = %.8e, 4 A tau / 9 = %.8e)\n", sigma, want);
    // And the factor that a port is most likely to revert.
    check(ColourModel::perturbation_coefficient(Real(A)) == Real(A),
          "the perturbation coefficient is A, not A/2");
  }

  //===========================================================================
  std::printf("\n4. recolouring is a partition\n\n");
  //===========================================================================
  {
    ColourModel m;
    m.beta = Real(0.7);
    const Real rr = Real(0.7), rb = Real(0.3);
    const Real g[3] = {Real(0.2), Real(-0.1), Real(0.05)};
    Real f[27];
    for (int i = 0; i < 27; ++i) f[i] = Real(D3Q27::w(i) * (1.0 + 0.01 * i));
    Real fr[27], fb[27];
    colour_recolour(m, f, rr, rb, g, fr, fb);
    double wsum = 0, mass = 0, momw = 0, moved = 0;
    double mr = 0, mb = 0, p0[3] = {0, 0, 0}, p1[3] = {0, 0, 0};
    for (int i = 0; i < 27; ++i) {
      wsum = worst(wsum, double(fr[i]) + double(fb[i]) - double(f[i]));
      mass += double(f[i]);
      mr += double(fr[i]); mb += double(fb[i]);
      moved = worst(moved, double(fr[i]) - double(rr) / (double(rr) + double(rb)) * double(f[i]));
      p0[0] += double(f[i]) * D3Q27::cx(i);
      p1[0] += (double(fr[i]) + double(fb[i])) * D3Q27::cx(i);
      p0[1] += double(f[i]) * D3Q27::cy(i);
      p1[1] += (double(fr[i]) + double(fb[i])) * D3Q27::cy(i);
      p0[2] += double(f[i]) * D3Q27::cz(i);
      p1[2] += (double(fr[i]) + double(fb[i])) * D3Q27::cz(i);
    }
    for (int a = 0; a < 3; ++a) momw = worst(momw, p1[a] - p0[a]);
    check(std::fabs(wsum) < 1e-6, "f^r + f^b = f, population by population", wsum);
    check(std::fabs(mr + mb - mass) < 1e-6, "so mass survives it", mr + mb - mass);
    check(std::fabs(momw) < 1e-6, "and momentum survives it", momw);
    check(std::fabs(moved) > 1e-6, "but it is not the identity: colour moves up grad phi", moved);
  }

  //===========================================================================
  std::printf("\n5. the collision leaves its own equilibrium alone\n\n");
  //===========================================================================
  {
    for (double ph : {-1.0, 0.0, 1.0}) {
      ColourModel m;
      m.A = Real(0);                       // no perturbation: test suboperator (1)
      const Real rr = Real(0.5 * (1.0 + ph)), rb = Real(0.5 * (1.0 - ph));
      const Real u[3] = {Real(0.04), Real(-0.02), Real(0.01)};
      const Real zero[3] = {Real(0), Real(0), Real(0)};
      Real G[3][3] = {{0}};
      Real fe[27];
      m.equilibrium(fe, rr, rb, u, G, m.nu_at(Real(ph)), Real(0));
      Real f[27];
      for (int i = 0; i < 27; ++i) f[i] = fe[i];
      colour_collide(m, f, rr, rb, u, Real(ph), zero, zero);
      double w = 0;
      for (int i = 0; i < 27; ++i) w = worst(w, double(f[i]) - double(fe[i]));
      char buf[80];
      std::snprintf(buf, sizeof buf, "collision leaves f^eq alone      (phi = %+.1f)", ph);
      check(std::fabs(w) < 1e-5, buf, w);
    }
    // ... and off equilibrium it still conserves what it must.
    {
      ColourModel m;
      m.A = Real(0.005);
      const Real rr = Real(0.6), rb = Real(0.4);
      const Real u[3] = {Real(0.03), Real(0.01), Real(-0.02)};
      const Real g[3] = {Real(0.1), Real(-0.05), Real(0.02)};
      const Real dr[3] = {Real(0.02), Real(0.01), Real(0)};
      Real f[27];
      for (int i = 0; i < 27; ++i) f[i] = Real(D3Q27::w(i) * (1.0 + 0.05 * std::sin(double(i))));
      double m0 = 0, p0[3] = {0, 0, 0};
      for (int i = 0; i < 27; ++i) {
        m0 += f[i];
        p0[0] += f[i] * D3Q27::cx(i); p0[1] += f[i] * D3Q27::cy(i); p0[2] += f[i] * D3Q27::cz(i);
      }
      colour_collide(m, f, rr, rb, u, Real(0.2), g, dr);
      double m1 = 0, p1[3] = {0, 0, 0};
      for (int i = 0; i < 27; ++i) {
        m1 += f[i];
        p1[0] += f[i] * D3Q27::cx(i); p1[1] += f[i] * D3Q27::cy(i); p1[2] += f[i] * D3Q27::cz(i);
      }
      check(std::fabs(m1 - m0) < 1e-5, "collision conserves mass off equilibrium", m1 - m0);
      const char* nm[3] = {"  ... and x momentum", "  ... and y momentum", "  ... and z momentum"};
      for (int a = 0; a < 3; ++a)
        check(std::fabs(p1[a] - p0[a]) < 1e-5, nm[a], p1[a] - p0[a]);
    }
  }

  //===========================================================================
  std::printf("\n6. THE PORT ITSELF: three passes, Esoteric Pull, a real droplet\n\n");
  //===========================================================================
  {
    const int nx = 12, ny = 12, nz = 12;
    const long N = long(nx) * ny * nz;
    std::vector<Real> fr(27 * N), fb(27 * N);
    std::vector<Real> fld(12 * N, Real(0));
    std::vector<std::uint8_t> flags(N, Fluid);

    ColourModel m;
    m.A = Real(0.01);
    m.beta = Real(0.7);
    m.rho_r0 = Real(1); m.rho_b0 = Real(1);

    ColourParams p;
    p.fr = fr.data(); p.fb = fb.data();
    p.rho_r = &fld[0];     p.rho_b = &fld[N];     p.phi = &fld[2 * N];
    p.ux    = &fld[3 * N]; p.uy    = &fld[4 * N]; p.uz  = &fld[5 * N];
    p.gx    = &fld[6 * N]; p.gy    = &fld[7 * N]; p.gz  = &fld[8 * N];
    p.rx    = &fld[9 * N]; p.ry    = &fld[10 * N]; p.rz = &fld[11 * N];
    p.flags = flags.data();
    p.nx = nx; p.ny = ny; p.nz = nz;
    p.m = m;

    // A droplet of red in blue, seeded at rest with a tanh profile.
    const double R = 3.5, W = 1.0;
    for (long n = 0; n < N; ++n) {
      int x, y, z; coords(n, nx, ny, x, y, z);
      const double dx = x - 0.5 * nx, dy = y - 0.5 * ny, dz = z - 0.5 * nz;
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double t = 0.5 * (1.0 - std::tanh((r - R) / W));
      const Real rr0 = Real(t), rb0 = Real(1.0 - t);
      Real gr[27], gb[27];
      for (int i = 0; i < 27; ++i) {
        gr[i] = rr0 * ColourModel::phi_i(i, m.alpha_r);
        gb[i] = rb0 * ColourModel::phi_i(i, m.alpha_b);
      }
      init_scatter<0, D3Q27>(fr.data(), N, x, y, z, nx, ny, nz, gr);
      init_scatter<0, D3Q27>(fb.data(), N, x, y, z, nx, ny, nz, gb);
    }

    auto totals = [&](double& tr, double& tb, double& px) {
      tr = tb = px = 0;
      for (long k = 0; k < 27 * N; ++k) { tr += fr[k]; tb += fb[k]; }
      for (long n = 0; n < N; ++n) px += p.ux[n] * (p.rho_r[n] + p.rho_b[n]);
    };

    // The host equivalent of refresh() + step(), parity alternating.
    auto sweep = [&](int t) {
      if (t % 2 == 0) for (long n = 0; n < N; ++n) colour_fields_node<0, false>(p, N, n);
      else            for (long n = 0; n < N; ++n) colour_fields_node<1, false>(p, N, n);
      for (long n = 0; n < N; ++n) colour_gradient_node<false>(p, n);
      if (t % 2 == 0) for (long n = 0; n < N; ++n) colour_node_update<0, false>(p, N, n);
      else            for (long n = 0; n < N; ++n) colour_node_update<1, false>(p, N, n);
    };

    for (long n = 0; n < N; ++n) colour_fields_node<0, false>(p, N, n);
    double r0, b0, px0; totals(r0, b0, px0);

    bool finite = true;
    for (int t = 0; t < 50; ++t) {
      sweep(t);
      for (long n = 0; n < N && finite; ++n)
        if (!std::isfinite(double(p.rho_r[n])) || !std::isfinite(double(p.ux[n])))
          finite = false;
      if (!finite) { std::printf("        (went non-finite at step %d)\n", t); break; }
    }
    double r1, b1, px1; totals(r1, b1, px1);

    check(finite, "fifty steps stay finite");
    const double tol = (sizeof(Real) == 4) ? 2e-4 : 1e-11;
    check(std::fabs(r1 - r0) < tol * std::fabs(r0),
          "red mass conserved through streaming and recolouring", (r1 - r0) / r0);
    check(std::fabs(b1 - b0) < tol * std::fabs(b0),
          "blue mass conserved likewise", (b1 - b0) / b0);
    std::printf("        (red %.10f -> %.10f, blue %.10f -> %.10f)\n", r0, r1, b0, b1);

    // The droplet must still be a droplet: red concentrated at the centre.
    int cx = nx / 2, cy = ny / 2, cz = nz / 2;
    const double phi_c = double(p.phi[node_id(cx, cy, cz, nx, ny)]);
    const double phi_e = double(p.phi[node_id(0, 0, 0, nx, ny)]);
    check(phi_c > 0.5, "still red at the centre", phi_c);
    check(phi_e < -0.5, "still blue in the corner", phi_e);
    std::printf("        (phi centre %+.4f, corner %+.4f)\n", phi_c, phi_e);

    // A droplet at rest should stay very nearly at rest: what velocity there is
    // is the spurious current, and it must be small compared with lattice speed.
    double umax = 0;
    for (long n = 0; n < N; ++n) {
      const double u = std::sqrt(double(p.ux[n]) * double(p.ux[n])
                              + double(p.uy[n]) * double(p.uy[n])
                              + double(p.uz[n]) * double(p.uz[n]));
      if (u > umax) umax = u;
    }
    check(umax < 0.02, "spurious current stays small", umax);
    std::printf("        (max |u| = %.4e)\n", umax);
  }

  std::printf("\n[colour] %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
