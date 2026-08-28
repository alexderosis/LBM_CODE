//==============================================================================
//  PHYSICS checks for the CUDA core, run WITHOUT a GPU.
//
//  host_check.cpp verifies pieces: a velocity set, a transform, one collision.
//  This file runs whole simulations and compares them against solutions that are
//  known in closed form. It can do that on a laptop because every per-node update
//  is an LBM_HD function and the CUDA kernels are three lines of index arithmetic
//  around them -- so these are the kernels, driven by a for-loop instead of by a
//  grid of threads. Under Esoteric Pull each slot has exactly one writer per
//  step, so the loop is not an approximation to the launch; it is the same
//  computation.
//
//  What it covers, and what each case would catch on its own:
//
//    1  Poiseuille flow between bounce-back walls   geometry, forcing, and where
//                                                   the no-slip plane really sits
//    2  a closed box                                that geometry writes every
//                                                   slot exactly once
//    3  an insulating box                           the scalar is conserved
//    4  conduction between Dirichlet walls          anti-bounce-back, and where
//                                                   ITS plane sits
//    5  a decaying sinusoid                         the D3Q7 diffusivity, cs^2=1/4
//    6  an advected sinusoid                        the advective flux
//    7  uniform buoyancy                            the whole Boussinesq path
//    8  resistive decay                             induction alone, no flow
//    9  a shear Alfven wave                         the Lorentz coupling, the
//                                                   induction equation, and the
//                                                   ORDER in which they are
//                                                   evaluated
//
//  Case 9 is the one that matters most. It is an exact solution of the full
//  NONLINEAR incompressible MHD equations, and it fails in a distinctive way if
//  the two-way coupling is lagged by a step: the damping error then GROWS with
//  resolution while the phase speed converges cleanly.
//
//  Build:  c++ -std=c++17 -O2 -Iinclude test/host_physics.cpp -o host_physics
//==============================================================================
#include "lbm/hostsim.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures = 0;
static const bool fp64 = (sizeof(Real) == 8);

static void check(bool ok, const char* what, double got, double want, const char* unit = "") {
  const double rel = (want != 0.0) ? std::fabs(got - want) / std::fabs(want)
                                   : std::fabs(got - want);
  std::printf("  %s  %-52s %12.6g vs %-12.6g %s(%.2e)\n", ok ? "PASS" : "FAIL",
              what, got, want, unit, rel);
  if (!ok) ++failures;
}

static void note(const char* s) { std::printf("        %s\n", s); }

//------------------------------------------------------------------------------
// Amplitude and phase of the fundamental x-mode, averaged over y and z.
// f(x) ~ amp * sin(k x + phase), k = 2 pi / nx.
//------------------------------------------------------------------------------
struct Mode { double amp, phase; };

static Mode fit_x(const Real* v, int nx, int ny, int nz) {
  const double k = 2.0 * M_PI / nx;
  const double inv = 1.0 / (double(ny) * nz);
  double S = 0, C = 0;
  for (int x = 0; x < nx; ++x) {
    double p = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y) p += double(v[node_id(x, y, z, nx, ny)]);
    p *= inv;
    S += p * std::sin(k * x);
    C += p * std::cos(k * x);
  }
  return {2.0 * std::sqrt(S * S + C * C) / nx, std::atan2(C, S)};
}

// Continue a phase sequence across the +-pi branch cut.
static double unwrap(double prev, double now) {
  while (now - prev >  M_PI) now -= 2.0 * M_PI;
  while (now - prev < -M_PI) now += 2.0 * M_PI;
  return now;
}

// Least-squares slope of y against x through n samples.
static double slope(const std::vector<double>& x, const std::vector<double>& y) {
  const double n = double(x.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

//==============================================================================
//  1. Poiseuille flow between halfway bounce-back walls.
//
//  H fluid nodes sit at y = 1..H; the walls are the SOLID layers at y = 0 and
//  y = H+1, so the no-slip planes are at y = 0.5 and y = H + 0.5 and
//
//      u(y) = (G / 2 rho nu) (y - 0.5)(H + 0.5 - y),   u_max = G H^2 / 8 rho nu.
//
//  Getting the plane wrong by half a lattice unit gives a parabola that still
//  looks like a parabola and is wrong by O(1/H). This case is therefore as much
//  a test of where the wall IS as of whether it holds.
//==============================================================================
static void poiseuille(Op op, const char* name) {
  const int H = 16, nx = 4, nz = 4, ny = H + 2;
  const double nu = 1.0 / 6.0;
  const double umax = 0.05;
  const double G = 8.0 * 1.0 * nu * umax / (double(H) * double(H));
  const std::size_t T = 20000;

  host::Fluid fl(nx, ny, nz, op, Real(nu));

  std::vector<std::uint8_t> flags(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      flags[std::size_t(node_id(x, 0,      z, nx, ny))] = Solid;
      flags[std::size_t(node_id(x, ny - 1, z, nx, ny))] = Solid;
    }
  fl.set_geometry(flags);

  BodyForce b;
  b.fx = Real(G);
  fl.set_force(b, ForceUniform);
  fl.initialise_with([](int, int, int) {
    return Macro{Real(1), Real(0), Real(0), Real(0)};
  });

  for (std::size_t t = 0; t < T; ++t) fl.step();

  std::vector<Real> rho, ux, uy, uz;
  fl.macroscopic_to_host(rho, ux, uy, uz);

  // The AMPLITUDE is fitted by least squares against the shape function, not
  // read off as the largest node value. For even H the parabola peaks at
  // y = (H+1)/2, which is half-way BETWEEN two nodes, so the largest sample is
  // below the continuous maximum by O(1/H^2) for reasons that have nothing to do
  // with the solver. Comparing the two is a test of arithmetic, not of physics.
  double worst_rel = 0, num = 0, den = 0;
  for (int y = 1; y <= H; ++y) {
    double u = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) u += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
    u /= double(nx) * nz;
    const double shape = (y - 0.5) * (H + 0.5 - y);
    const double exact = (G / (2.0 * nu)) * shape;
    worst_rel = std::fmax(worst_rel, std::fabs(u - exact) / umax);
    num += u * shape;
    den += shape * shape;
  }
  const double amp = num / den;
  const double amp_exact = G / (2.0 * nu);
  char buf[128];
  std::snprintf(buf, sizeof buf, "Poiseuille %s: fitted parabola amplitude", name);
  check(std::fabs(amp - amp_exact) / amp_exact < (fp64 ? 2e-3 : 3e-3), buf, amp, amp_exact);
  std::snprintf(buf, sizeof buf, "Poiseuille %s: worst profile error / u_max", name);
  check(worst_rel < (fp64 ? 2e-3 : 3e-3), buf, worst_rel, 0.0);
}

//==============================================================================
//  2. A closed box.
//
//  Six solid walls, a swirl inside, no force. Two things must hold, and they
//  test different halves of the geometry implementation:
//
//    * the sum over EVERY slot of the lattice is conserved to round-off. Under
//      Esoteric Pull each slot has one writer per step; if a solid cell were
//      visited when it should not be, or a fluid cell skipped, this is where it
//      shows;
//    * the flow decays to rest, because there is nothing driving it.
//
//  The case also measures something worth knowing rather than asserting: how
//  much of the total the FLUID cells see. A population in flight toward a wall
//  spends a step in a slot the wall owns, so a fluid-only sum always undercounts.
//  Nothing is lost; the plotted field is what is missing it.
//==============================================================================
static void closed_box() {
  const int n = 24;
  host::Fluid fl(n, n, n, Op::CentralMoments, Real(1.0 / 6.0));

  std::vector<std::uint8_t> flags(std::size_t(n) * n * n, std::uint8_t(Fluid));
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        if (x == 0 || y == 0 || z == 0 || x == n - 1 || y == n - 1 || z == n - 1)
          flags[std::size_t(node_id(x, y, z, n, n))] = Solid;
  fl.set_geometry(flags);

  const double k = 2.0 * M_PI / n;
  fl.initialise_with([k, n](int x, int y, int z) {
    Macro m;
    m.rho = Real(1);
    m.ux  = Real( 0.04 * std::sin(k * x) * std::cos(k * y));
    m.uy  = Real(-0.04 * std::cos(k * x) * std::sin(k * y));
    m.uz  = Real(0);
    (void)z; (void)n;
    return m;
  });

  auto energy = [&]() {
    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    double e = 0;
    for (std::size_t i = 0; i < ux.size(); ++i)
      e += 0.5 * (double(ux[i]) * double(ux[i]) + double(uy[i]) * double(uy[i]) +
                  double(uz[i]) * double(uz[i]));
    return e;
  };

  const double m0 = fl.total_mass();
  const double e0 = energy();
  for (std::size_t t = 0; t < 200; ++t) fl.step();
  const double m1 = fl.total_mass();
  const double e1 = energy();

  check(std::fabs(m1 - m0) / m0 < (fp64 ? 1e-12 : 3e-6),
        "closed box: total mass over every slot conserved", m1, m0);
  check(e1 < 0.5 * e0 && e1 > 0.0,
        "closed box: the flow decays with nothing driving it", e1 / e0, 0.0);
  char buf[160];
  std::snprintf(buf, sizeof buf,
                "energy fell to %.2e of its initial value in 200 steps, while the mass "
                "held to %.1e -- decay is physics, the mass is bookkeeping", e1 / e0,
                std::fabs(m1 - m0) / m0);
  note(buf);
}

//==============================================================================
//  3. An insulating box conserves the scalar, exactly.
//
//  Every wall adiabatic, a blob in the middle, no flow. The total population --
//  summed over the WHOLE lattice, not over fluid cells -- must not move.
//
//  The fluid-cell sum is reported alongside, because the gap between the two is
//  the thing most likely to be mistaken for a leak. In the parent implementation
//  it reached 13% on an urban geometry and none of it was lost.
//==============================================================================
static void insulating_box() {
  const int n = 20;
  host::Scalar sc(n, n, n, Real(0.05));

  std::vector<std::uint8_t> flags(std::size_t(n) * n * n, std::uint8_t(ScalarBulk));
  std::vector<Real> wall(std::size_t(n) * n * n, Real(0));
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        if (x == 0 || y == 0 || z == 0 || x == n - 1 || y == n - 1 || z == n - 1)
          flags[std::size_t(node_id(x, y, z, n, n))] = ScalarAdiabatic;
  sc.set_geometry(flags, wall);

  const double c = 0.5 * (n - 1);
  sc.initialise_with([c](int x, int y, int z) {
    const double r2 = (x - c) * (x - c) + (y - c) * (y - c) + (z - c) * (z - c);
    return Real(std::exp(-r2 / 8.0));
  });

  const double m0 = sc.total_population();
  for (std::size_t t = 0; t < 3000; ++t) sc.step();
  const double m1 = sc.total_population();

  check(std::fabs(m1 - m0) / m0 < (fp64 ? 1e-12 : 3e-4),
        "insulating box: total population conserved", m1, m0);

  const std::vector<Real>& T = sc.field();
  double in_fluid = 0;
  for (int z = 1; z < n - 1; ++z)
    for (int y = 1; y < n - 1; ++y)
      for (int x = 1; x < n - 1; ++x) in_fluid += double(T[std::size_t(node_id(x, y, z, n, n))]);
  char buf[160];
  std::snprintf(buf, sizeof buf,
                "the field over bulk cells sees %.2f%% of it -- the rest is in wall slots, "
                "in flight, not lost", 100.0 * in_fluid / m1);
  note(buf);
}

//==============================================================================
//  4. Conduction between Dirichlet walls.
//
//  Anti-bounce-back puts T_wall half-way between the Dirichlet node and its
//  fluid neighbour, exactly as halfway bounce-back does for no-slip. So with
//  Dirichlet layers at y = 0 and y = H+1 the two PLANES are at y = 0.5 and
//  y = H + 0.5, the gap between them is H, and the steady profile is
//
//      T(y) = (y - 0.5) / H.
//
//  Measuring between the NODES instead gives a gradient wrong by (H+1)/H, an
//  O(1/H) error that shrinks under refinement and so is easily mistaken for
//  ordinary discretisation error.
//==============================================================================
static void conduction() {
  const int H = 16, nx = 4, nz = 4, ny = H + 2;
  host::Scalar sc(nx, ny, nz, Real(0.125), Real(0.5));

  std::vector<std::uint8_t> flags(std::size_t(nx) * ny * nz, std::uint8_t(ScalarBulk));
  std::vector<Real> wall(std::size_t(nx) * ny * nz, Real(0));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      flags[std::size_t(node_id(x, 0,      z, nx, ny))] = ScalarDirichlet;
      flags[std::size_t(node_id(x, ny - 1, z, nx, ny))] = ScalarDirichlet;
      wall [std::size_t(node_id(x, 0,      z, nx, ny))] = Real(0);
      wall [std::size_t(node_id(x, ny - 1, z, nx, ny))] = Real(1);
    }
  sc.set_geometry(flags, wall);
  sc.initialise_with([](int, int, int) { return Real(0.5); });

  for (std::size_t t = 0; t < 20000; ++t) sc.step();

  const std::vector<Real>& T = sc.field();
  double worst_err = 0;
  for (int y = 1; y <= H; ++y) {
    double t = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) t += double(T[std::size_t(node_id(x, y, z, nx, ny))]);
    t /= double(nx) * nz;
    worst_err = std::fmax(worst_err, std::fabs(t - (y - 0.5) / double(H)));
  }
  check(worst_err < (fp64 ? 1e-6 : 2e-4),
        "conduction: worst deviation from the exact linear profile", worst_err, 0.0);

  // The same measurement made between the NODES, to show what the wrong
  // convention costs on this grid.
  double t1 = 0, tH = 0;
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      t1 += double(T[std::size_t(node_id(x, 1, z, nx, ny))]);
      tH += double(T[std::size_t(node_id(x, H, z, nx, ny))]);
    }
  t1 /= double(nx) * nz; tH /= double(nx) * nz;
  char buf[160];
  std::snprintf(buf, sizeof buf,
                "measured gradient %.6f per node; planes H = %d apart give %.6f, "
                "nodes H-1 apart would give %.6f",
                (tH - t1) / double(H - 1), H, 1.0 / H, 1.0 / (H - 1));
  note(buf);
}

//==============================================================================
//  5. A decaying sinusoid: the D3Q7 diffusivity.
//
//  T = T0 + A sin(kx) decays as exp(-D k^2 t) with D = cs^2 (1/omega - 1/2) and
//  cs^2 = 1/4. Run at two resolutions: the measured rate must approach the
//  analytic one, and the error must fall like the square of the wavenumber. A
//  single resolution cannot tell a wrong cs^2 from ordinary discretisation error;
//  two can, because a wrong cs^2 does not converge away.
//==============================================================================
static double diffusion_error(int L) {
  const int ny = 4, nz = 4;
  const double D = 0.125;                      // omega = 1
  host::Scalar sc(L, ny, nz, Real(D), Real(1.0));

  const double k = 2.0 * M_PI / L;
  sc.initialise_with([k](int x, int, int) { return Real(1.0 + 0.1 * std::sin(k * x)); });

  // Run for a FIXED NUMBER OF E-FOLDS, not a fixed number of steps. The decay
  // time is 1/(D k^2), which is four times shorter at L = 16 than at L = 32; a
  // step count that suits one resolution lets the other decay into round-off,
  // and the fit then measures noise. (It did, the first time: -66% and +73%.)
  const std::size_t T = std::size_t(2.0 / (D * k * k));
  const std::size_t probe = T / 20 ? T / 20 : 1;
  std::vector<double> ts, la;
  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      const Mode m = fit_x(sc.field().data(), L, ny, nz);
      ts.push_back(double(t));
      la.push_back(std::log(m.amp));
    }
    if (t < T) sc.step();
  }
  const double rate = -slope(ts, la);
  const double exact = D * k * k;
  return (rate - exact) / exact;
}

static void diffusion() {
  const double e16 = diffusion_error(16);
  const double e32 = diffusion_error(32);
  check(std::fabs(e32) < 0.02, "diffusion: rate error at L = 32", e32, 0.0);
  const double ratio = std::fabs(e16 / e32);
  check(ratio > 2.5 && ratio < 5.5,
        "diffusion: error falls as k^2 when L doubles (ratio)", ratio, 4.0);
  char buf[160];
  std::snprintf(buf, sizeof buf,
                "rate error %.3f%% at L=16, %.3f%% at L=32 -- a wrong cs^2 would not converge",
                100.0 * e16, 100.0 * e32);
  note(buf);
}

//==============================================================================
//  6. An advected sinusoid: the advective flux.
//
//  A uniform velocity is imposed directly, so this isolates the scalar from the
//  fluid entirely. The mode's phase must move at exactly u:  dphi/dt = -k u.
//
//  The first-order D3Q7 equilibrium carries an O(u^2) defect by construction --
//  the lattice cannot represent the cross terms of the uu tensor -- so the
//  tolerance here is a statement about that, not about round-off.
//==============================================================================
static void advection() {
  const int L = 32, ny = 4, nz = 4;
  const double u = 0.02, D = 0.05;
  host::Scalar sc(L, ny, nz, Real(D), Real(1.0));

  std::vector<Real> vx(std::size_t(L) * ny * nz, Real(u));
  std::vector<Real> vy(std::size_t(L) * ny * nz, Real(0));
  std::vector<Real> vz(std::size_t(L) * ny * nz, Real(0));
  sc.advect_with(vx.data(), vy.data(), vz.data());

  const double k = 2.0 * M_PI / L;
  sc.initialise_with([k](int x, int, int) { return Real(1.0 + 0.1 * std::sin(k * x)); });

  const std::size_t T = 3000, probe = 100;
  std::vector<double> ts, ph;
  double prev = 0;
  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      const Mode m = fit_x(sc.field().data(), L, ny, nz);
      const double p = ts.empty() ? m.phase : unwrap(prev, m.phase);
      prev = p;
      ts.push_back(double(t));
      ph.push_back(p);
    }
    if (t < T) sc.step();
  }
  const double measured = -slope(ts, ph) / k;
  check(std::fabs(measured - u) / u < 0.02, "advection: phase speed", measured, u);
}

//==============================================================================
//  7. Uniform buoyancy: the whole Boussinesq path in one number.
//
//  A periodic box at rest, held at a uniform T above the reference. The force is
//  then F = rho0 g beta (T - T0) everywhere and there is no pressure gradient to
//  balance it, so the fluid accelerates uniformly:  du/dt = F / rho, exactly.
//
//  Measuring the INCREMENT between two times rather than the value avoids the
//  half-step offset Guo's velocity carries, and makes the test independent of
//  when it is sampled.
//
//  This exercises the whole chain -- scalar populations, compute_field, the
//  BodyForce read of that field, the Guo source -- against an exact answer, in a
//  case with no closed-form convection solution to hide behind.
//==============================================================================
static void buoyancy() {
  const int n = 8;
  const double T0 = 1.0, dT = 0.05, beta = 1.0;

  host::Fluid fl(n, n, n, Op::BGK, Real(1.0 / 6.0));
  host::Scalar sc(n, n, n, Real(0.05), Real(T0));

  fl.enable_velocity_output();
  sc.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  BodyForce b;
  b.T = sc.field_device();
  b.gx = Real(0); b.gy = Real(-1); b.gz = Real(0);
  b.rho0 = Real(1); b.beta = Real(beta); b.T0 = Real(T0);
  fl.set_force(b, ForceBoussinesq);

  fl.initialise_with([](int, int, int) { return Macro{Real(1), Real(0), Real(0), Real(0)}; });
  sc.initialise_with([T0, dT](int, int, int) { return Real(T0 + dT); });

  auto mean_uy = [&]() {
    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    double s = 0;
    for (Real v : uy) s += double(v);
    return s / double(uy.size());
  };

  for (int t = 0; t < 100; ++t) host::coupled_step(fl, &sc, nullptr);
  const double u1 = mean_uy();
  for (int t = 0; t < 100; ++t) host::coupled_step(fl, &sc, nullptr);
  const double u2 = mean_uy();

  const double measured = (u2 - u1) / 100.0;
  const double exact = -beta * dT;             // rho0 g beta dT / rho, with rho = 1
  check(std::fabs(measured - exact) / std::fabs(exact) < (fp64 ? 1e-6 : 2e-3),
        "buoyancy: acceleration per step", measured, exact);
}

//==============================================================================
//  8. Resistive decay: the induction equation on its own.
//
//  B = (0, B0 sin kx, 0) with u = 0 everywhere. The flow is not solved at all,
//  so this isolates the magnetic lattice and its resistivity, eta = cs^2
//  (1/omega - 1/2) with cs^2 = 1/4 again. Exact: B decays as exp(-eta k^2 t).
//
//  Note this case says NOTHING about div B: B_y depends only on x, so div B is
//  structurally zero and would report round-off whatever the scheme did.
//==============================================================================
static void resistive_decay() {
  const int L = 32, ny = 4, nz = 4;
  const double eta = 0.02;
  host::Magnetic mag(L, ny, nz, Real(eta));

  const double k = 2.0 * M_PI / L;
  const double B0 = 0.02;
  mag.initialise_with(
      [k, B0](int x, int, int, Real B[3]) {
        B[0] = Real(0); B[1] = Real(B0 * std::sin(k * x)); B[2] = Real(0);
      },
      [](int, int, int, Real u[3]) { u[0] = u[1] = u[2] = Real(0); });

  const std::size_t T = 4000, probe = 200;
  std::vector<double> ts, la;
  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      const Mode m = fit_x(mag.by().data(), L, ny, nz);
      ts.push_back(double(t));
      la.push_back(std::log(m.amp));
    }
    if (t < T) mag.step();
  }
  const double rate = -slope(ts, la);
  const double exact = eta * k * k;
  check(std::fabs(rate - exact) / exact < 0.02, "resistive decay: rate", rate, exact);
}

//==============================================================================
//  9. The shear Alfven wave.
//
//  An EXACT solution of the full nonlinear incompressible MHD equations, not
//  merely the linearised ones: u is perpendicular to B0 and everything depends
//  only on x, so (u.grad)u vanishes identically while (B.grad)B does not. Both
//  the Lorentz coupling and the induction equation are therefore driven, and both
//  must be right.
//
//      B = (B0, b sin k(x - v_A t), 0),   u = (0, -b sin k(x - v_A t) / sqrt(rho), 0)
//
//  propagating at v_A = B0 / sqrt(rho) and damping at (nu + eta) k^2 / 2.
//
//  PHASE IS MEASURED AS WELL AS AMPLITUDE, and that is the point. An error in the
//  Lorentz coupling shows up as the wrong wave SPEED. An error in the coupling
//  ORDER shows up only in the damping -- and it does not refine away, which is
//  how the parent implementation eventually found it.
//==============================================================================
static void alfven(Op op, const char* name) {
  const int L = 64, ny = 4, nz = 4;
  const double nu = 0.01, eta = 0.01;
  const double B0 = 0.02, b = 0.002, rho = 1.0;
  const double k = 2.0 * M_PI / L;
  const double vA = B0 / std::sqrt(rho);

  host::Magnetic mag(L, ny, nz, Real(eta));
  host::Fluid    fl (L, ny, nz, op, Real(nu));

  fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  fl.initialise_with([k, b, rho](int x, int, int) {
    Macro m;
    m.rho = Real(rho);
    m.ux  = Real(0);
    m.uy  = Real(-b * std::sin(k * x) / std::sqrt(rho));
    m.uz  = Real(0);
    return m;
  });
  mag.initialise_with(
      [k, b, B0](int x, int, int, Real B[3]) {
        B[0] = Real(B0); B[1] = Real(b * std::sin(k * x)); B[2] = Real(0);
      },
      [k, b, rho](int x, int, int, Real u[3]) {
        u[0] = Real(0); u[1] = Real(-b * std::sin(k * x) / std::sqrt(rho)); u[2] = Real(0);
      });

  const std::size_t T = 4000, probe = 200, settle = 400;
  std::vector<double> ts, ph, la;
  double prev = 0;
  bool first = true;
  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      const Mode m = fit_x(mag.by().data(), L, ny, nz);
      const double p = first ? m.phase : unwrap(prev, m.phase);
      prev = p; first = false;
      if (t >= settle) {                       // let the seeding transient leave
        ts.push_back(double(t));
        ph.push_back(p);
        la.push_back(std::log(m.amp));
      }
    }
    if (t < T) { mag.compute_field(); fl.step(); mag.step(); }
  }

  const double speed = -slope(ts, ph) / k;
  const double damp  = -slope(ts, la);
  const double damp_exact = 0.5 * (nu + eta) * k * k;

  char buf[128];
  std::snprintf(buf, sizeof buf, "Alfven %s: wave speed v_A = B0/sqrt(rho)", name);
  check(std::fabs(speed - vA) / vA < 0.03, buf, speed, vA);
  std::snprintf(buf, sizeof buf, "Alfven %s: damping rate (nu+eta) k^2 / 2", name);
  check(std::fabs(damp - damp_exact) / damp_exact < 0.15, buf, damp, damp_exact);
}

//==============================================================================
int main() {
  std::printf("Physics checks, host build, Real = %s\n\n", fp64 ? "double" : "float");

  std::printf("  -- geometry and forcing --\n");
  poiseuille(Op::BGK, "BGK");
  poiseuille(Op::CentralMoments, "CM");
  closed_box();

  std::printf("\n  -- the passive scalar --\n");
  insulating_box();
  conduction();
  diffusion();
  advection();

  std::printf("\n  -- coupled: buoyancy --\n");
  buoyancy();

  std::printf("\n  -- magnetohydrodynamics --\n");
  resistive_decay();
  alfven(Op::BGK, "BGK");
  alfven(Op::CentralMoments, "CM");

  std::printf("\n%s  (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
