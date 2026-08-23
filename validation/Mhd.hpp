#pragma once
//==============================================================================
//  MHD validation cases.
//
//  1. RESISTIVE DECAY      B = (0, B0 sin kx),  u = 0
//     Exact: B decays as exp(-eta k^2 t). Isolates the induction equation and
//     the resistivity, with no coupling to the flow at all.
//
//  2. SHEAR ALFVEN WAVE    B = (B0, b sin k(x - v_A t)),  u = (0, -b/sqrt(rho) ...)
//     An EXACT solution of the full nonlinear incompressible MHD equations, not
//     merely the linearised ones: with u perpendicular to B0 and everything
//     depending only on x, (u.grad)u vanishes identically while (B.grad)B does
//     not, so the Lorentz coupling and the induction equation are both driven
//     and both must be right. Propagates at v_A = B0/sqrt(rho) and damps at
//     (nu + eta) k^2 / 2. Phase is measured as well as amplitude, because an
//     error in the coupling shows up as the wrong wave speed.
//
//  3. ORSZAG-TANG VORTEX
//     u = u0 (-sin 2pi y/L,  sin 2pi x/L)
//     B = B0 (-sin 2pi y/L,  sin 4pi x/L)
//     No closed-form solution -- it steepens into current sheets -- so it is not
//     validated against a formula. What it does test is what a formula cannot:
//     div B preservation over a long nonlinear run, energy budgets, and
//     self-convergence under refinement.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MagneticBGK.hpp"
#include "collision/MhdBGK.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/MagneticSolver.hpp"

#include <cmath>
#include <vector>

namespace lbm {
namespace mhd {

constexpr double PI = 3.14159265358979323846;

using FL = D2Q9;                 // fluid lattice
using ML = D2Q5;                 // magnetic lattice -- deliberately different
using Fluid = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations>;
using Mag   = MagneticBGK<ML>;
using FluidS = FluidSolver<FL, EsotericPull<FL>, Fluid>;
using MagS   = MagneticSolver<ML, EsotericPull<ML>, Mag>;

//------------------------------------------------------------------------------
// max |div B| , normalised by k |B| -- i.e. measured against the field's own
// gradient scale, which is the only normalisation that stays meaningful as the
// grid refines.
//
// A caveat worth stating plainly: in BOTH wave cases above div B is
// structurally zero (B_x depends only on y, B_y only on x), so they report
// round-off no matter what the scheme does and are NOT evidence of divergence
// preservation. Orszag-Tang is the only case here that tests it, because the
// nonlinear dynamics makes every component depend on every coordinate.
//------------------------------------------------------------------------------
inline double max_div_b(MagS& m, double k = 1.0) {
  const Domain& d = m.domain();
  m.compute_field();
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, m.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, m.By());
  auto wrap = [](Index v, Index n) { return ((v % n) + n) % n; };
  double worst = 0, scale = 0;
  for (Index y = 0; y < d.ny; ++y)
    for (Index x = 0; x < d.nx; ++x) {
      const double dv =
          0.5 * (double(bx(d.id(wrap(x + 1, d.nx), y))) - double(bx(d.id(wrap(x - 1, d.nx), y)))) +
          0.5 * (double(by(d.id(x, wrap(y + 1, d.ny)))) - double(by(d.id(x, wrap(y - 1, d.ny)))));
      worst = std::max(worst, std::abs(dv));
      scale = std::max(scale, std::hypot(double(bx(d.id(x, y))), double(by(d.id(x, y)))));
    }
  return (scale > 0) ? worst / (scale * k) : worst;
}

//------------------------------------------------------------------------------
// Case 1 and 2. `alfven == false` runs pure resistive decay.
//------------------------------------------------------------------------------
struct Wave { double rate_eff, rate_err, speed_eff, speed_err, l2, div_b; std::size_t steps;
              std::vector<double> amp_hist, ph_hist; };

inline Wave wave(Index N, Real nu, Real eta, Real amp, Real B0, bool alfven) {
  const double k = 2.0 * PI / double(N);
  const double rho0 = 1.0;
  const double vA = alfven ? double(B0) / std::sqrt(rho0) : 0.0;
  const double decay = alfven ? 0.5 * (double(nu) + double(eta)) * k * k
                              : double(eta) * k * k;
  const auto steps = std::size_t(0.5 / decay);

  Domain d(N, 8, 1, true, true, true);
  Fluid fc; fc.omega = Fluid::omega_from_viscosity(nu);
  Mag   mc; mc.omega = Mag::omega_from_resistivity(eta);
  MagS  mag(d, mc);
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidS fl(d, fc);

  const Real kk = Real(k), A = amp, B0k = B0;
  const Real uAmp = alfven ? Real(-double(amp) / std::sqrt(rho0)) : Real(0);
  mag.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    Kokkos::Array<Real, 3> b;
    b[0] = alfven ? B0k : Real(0);
    b[1] = A * Kokkos::sin(kk * Real(x));
    b[2] = Real(0);
    return b;
  });
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    FlowState s;
    s.rho = Real(1);
    s.ux = Real(0);
    s.uy = uAmp * Kokkos::sin(kk * Real(x));
    s.uz = Real(0);
    return s;
  });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // amplitude/phase history, to tell a wrong damping RATE from a beat
  std::vector<double> amp_hist, ph_hist;
  auto project = [&](double& a, double& ph) {
    mag.compute_field();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
    double S = 0, C = 0;
    for (Index x = 0; x < N; ++x) {
      double row = 0;
      for (Index y = 0; y < d.ny; ++y) row += double(h(d.id(x, y)));
      row /= double(d.ny);
      S += row * std::sin(k * x);  C += row * std::cos(k * x);
    }
    S *= 2.0 / double(N);  C *= 2.0 / double(N);
    a = std::hypot(S, C);  ph = std::atan2(C, S);
  };
  {
    double a, ph; project(a, ph);
    amp_hist.push_back(a); ph_hist.push_back(ph);
  }
  const std::size_t chunk = steps / 20 + 1;
  for (std::size_t t = 0; t < steps; ++t) {
    mag.compute_field(); fl.step(true); mag.step(true);
    if ((t + 1) % chunk == 0) { double a, ph; project(a, ph); amp_hist.push_back(a); ph_hist.push_back(ph); }
  }

  mag.compute_field();
  auto hB = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  double sn = 0, cs = 0;
  for (Index x = 0; x < N; ++x) {
    double row = 0;
    for (Index y = 0; y < d.ny; ++y) row += double(hB(d.id(x, y)));
    row /= double(d.ny);
    sn += row * std::sin(k * x);
    cs += row * std::cos(k * x);
  }
  sn *= 2.0 / double(N);  cs *= 2.0 / double(N);
  const double got_amp = std::hypot(sn, cs);
  double got_ph = std::atan2(cs, sn);

  const double rate_eff = -std::log(got_amp / double(amp)) / (k * k * double(steps));
  const double rate_ref = alfven ? 0.5 * (double(nu) + double(eta)) : double(eta);

  // phase: the wave has travelled v_A * steps, so the recovered speed is
  // -phase / (k * steps).  Unwrap against the expected value.
  double want_ph = -k * vA * double(steps);
  double n2pi = std::round((got_ph - want_ph) / (2 * PI));
  got_ph -= 2 * PI * n2pi;
  const double speed_eff = alfven ? -got_ph / (k * double(steps)) : 0.0;

  const double E = std::exp(-decay * double(steps));
  double num = 0, den = 0;
  for (Index x = 0; x < N; ++x) {
    const double ana = double(amp) * std::sin(k * (double(x) - vA * double(steps))) * E;
    double row = 0;
    for (Index y = 0; y < d.ny; ++y) row += double(hB(d.id(x, y)));
    row /= double(d.ny);
    num += (row - ana) * (row - ana);
    den += ana * ana;
  }
  return {rate_eff, std::abs(rate_eff / rate_ref - 1.0), speed_eff,
          alfven ? std::abs(speed_eff / vA - 1.0) : 0.0,
          std::sqrt(num / den), max_div_b(mag, k), steps, amp_hist, ph_hist};
}

//------------------------------------------------------------------------------
// Case 3: Orszag-Tang.
//------------------------------------------------------------------------------
struct OT {
  double div_b, div_b_max, e_kin, e_mag, e_tot0, e_tot1;
  std::vector<double> div_hist;
  std::vector<double> By_line;      // B_y along y = N/2, for self-convergence
  std::size_t steps;
};

inline OT orszag_tang(Index N, Real u0, Real B0, double Re, double t_star,
                      bool record = false) {
  const double L = double(N);
  const Real nu  = Real(double(u0) * L / Re);
  const Real eta = nu;                              // magnetic Prandtl number 1
  const auto steps = std::size_t(t_star * L / double(u0));
  const double k = 2.0 * PI / L;

  Domain d(N, N, 1, true, true, true);
  Fluid fc; fc.omega = Fluid::omega_from_viscosity(nu);
  Mag   mc; mc.omega = Mag::omega_from_resistivity(eta);
  MagS  mag(d, mc);
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidS fl(d, fc);

  const Real kk = Real(k), U = u0, B = B0;
  mag.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    Kokkos::Array<Real, 3> b;
    b[0] = -B * Kokkos::sin(kk * Real(y));
    b[1] =  B * Kokkos::sin(Real(2) * kk * Real(x));
    b[2] = Real(0);
    return b;
  });
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z);
    FlowState s;
    s.rho = Real(1);
    s.ux = -U * Kokkos::sin(kk * Real(y));
    s.uy =  U * Kokkos::sin(kk * Real(x));
    s.uz = Real(0);
    return s;
  });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  auto energies = [&](double& ek, double& em) {
    fl.compute_macroscopic();
    mag.compute_field();
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
    auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
    auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
    ek = em = 0;
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y);
        ek += 0.5 * double(hr(n)) * (double(hx(n)) * double(hx(n)) + double(hy(n)) * double(hy(n)));
        em += 0.5 * (double(bx(n)) * double(bx(n)) + double(by(n)) * double(by(n)));
      }
    ek /= double(N * N); em /= double(N * N);
  };

  double ek0, em0;
  energies(ek0, em0);
  std::vector<double> div_hist;
  div_hist.push_back(max_div_b(mag, k));
  const std::size_t chunk = steps / 10 + 1;
  for (std::size_t t = 0; t < steps; ++t) {
    mag.compute_field(); fl.step(true); mag.step(true);
    if ((t + 1) % chunk == 0) div_hist.push_back(max_div_b(mag, k));
  }
  double ek1, em1;
  energies(ek1, em1);

  OT r;
  r.div_b = max_div_b(mag, k);
  r.div_hist = div_hist;
  r.div_b_max = 0;
  for (double v : div_hist) r.div_b_max = std::max(r.div_b_max, v);
  r.e_kin = ek1; r.e_mag = em1;
  r.e_tot0 = ek0 + em0; r.e_tot1 = ek1 + em1;
  r.steps = steps;
  if (record) {
    auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
    for (Index x = 0; x < N; ++x) r.By_line.push_back(double(by(d.id(x, N / 2))));
  }
  return r;
}

}  // namespace mhd
}  // namespace lbm
