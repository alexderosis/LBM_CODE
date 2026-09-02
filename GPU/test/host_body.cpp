//==============================================================================
//  A rigid body by volume penalisation, checked WITHOUT a GPU.
//
//  body.cuh's per-node functions and its whole rigid-body solve are LBM_HD or
//  plain host code, so everything the kernels compute is exercised here. What
//  this cannot check is the seven-accumulator block reduction, which is the one
//  genuinely device-side piece -- but it CAN check that the serial answer is the
//  one the reduction has to reproduce, which is the useful half.
//
//  Four things, in increasing order of how much they involve:
//
//    1  the indicator          areas, and the apex rounding that biases a wedge
//    2  Newton, with no fluid  the two cases the classical arrangement cannot do
//    3  the reaction           R = -sum F, which is a derived quantity here
//    4  a body in a fluid      prescribed motion drags the fluid to it
//
//  Build:  c++ -std=c++17 -O2 -Iinclude test/host_body.cpp -o host_body
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

static void check(bool ok, const char* what, double got, double want) {
  const double rel = (want != 0.0) ? std::fabs(got - want) / std::fabs(want)
                                   : std::fabs(got - want);
  std::printf("  %s  %-56s %12.6g vs %-12.6g (%.2e)\n", ok ? "PASS" : "FAIL",
              what, got, want, rel);
  if (!ok) ++failures;
}
static void note(const char* s) { std::printf("        %s\n", s); }

int main() {
  std::printf("Penalised body, host build, Real = %s\n",
              sizeof(Real) == 4 ? "float" : "double");

  //===========================================================================
  std::printf("\n1. THE INDICATOR\n\n");
  //===========================================================================
  //
  // chi is a PRODUCT of tanh indicators, which is exact away from the corners
  // and rounds them. For a rectangle that is a wash -- the rounding removes as
  // much as the tail adds -- but for a wedge it is not, because a wedge apex is
  // a sharper corner than a right angle and there is no matching concave one to
  // cancel it. The penalised wedge is therefore slightly LARGER than the
  // nominal one, with the excess concentrated exactly at the apex.
  //
  // THAT IS THE WORST PLACE FOR IT in the case this shape exists for: water
  // entry starts at the apex and Wagner's wetted length is measured from it. So
  // a shallow-deadrise run wants a BIG body in cells rather than a small one
  // with the smoothing turned down -- `smooth` below about one cell makes the
  // indicator a step again and reintroduces the lattice-quantised pressure
  // pulses penalisation exists to avoid.
  {
    // Rectangle: the two effects cancel, and do so better as the body grows.
    // 0.21% at h = 4 and under a part in 10^5 from h = 8 -- so a rectangle
    // needs no correction and the wedge's excess below is a real asymmetry
    // rather than an artefact of how the integral is taken.
    double worst_small = 0, worst_large = 0;
    for (int k = 0; k < 4; ++k) {
      const double h = 4.0 * (1 << k);
      const int n = int(6 * h) + 20;
      host::Body<Rect> body(n, n, 1);
      body.shape.hx = Real(h);  body.shape.hy = Real(h);
      body.shape.smooth = Real(1.5);
      body.shape.cx = Real(n / 2);  body.shape.cy = Real(n / 2);
      body.shape.set_angle(Real(0));
      const double e = body.indicator_moments().area / (4 * h * h) - 1.0;
      if (k == 0) worst_small = e; else worst_large = std::fmax(worst_large, std::fabs(e));
    }
    check(std::fabs(worst_small) < 5e-3, "rect: area = 4 hx hy to 0.3% at h = 4",
          worst_small, 0.0);
    check(worst_large < 1e-4, "rect: and to a part in 10^4 from h = 8",
          worst_large, 0.0);

    // Wedge: the apex table. THE INTEGRAL MUST BE TAKEN ON A DOMAIN THAT HOLDS
    // THE WHOLE TANH TAIL, and that is not a detail -- measured at b = 30,
    // deadrise 10 deg, the excess reads -10.9% on a 40-wide domain, +4.1% on
    // 64, +5.75% on 80 and converges to +5.999% from 100 upward. A clipped
    // domain removes tail, so it UNDERSTATES the excess.
    //
    // The parent's banner records this table as +0.50 / +0.23 / +0.06 / +0.03
    // at b = 100 and +5.6 / +2.6 / +0.64 / +0.28 at b = 30. This independent
    // implementation gets +0.54 / +0.24 / +0.06 / +0.03 and +6.00 / +2.66 /
    // +0.66 / +0.29 on a converged domain. The b = 100 column agrees to 0.04
    // percentage points; the b = 30 column is consistently a little higher, and
    // the convergence table above says why -- a domain around 75-80 wide
    // reproduces the parent's numbers, so its measurement was slightly
    // tail-clipped and the converged figures are these. Sub-cell placement of
    // the apex is NOT the cause: sweeping it over a whole cell moves the b = 30
    // numbers by 0.05 percentage points.
    const double deg[4] = {10, 15, 30, 45};
    const double want100[4] = {0.54, 0.24, 0.06, 0.03};
    const double want30[4]  = {6.00, 2.66, 0.66, 0.29};
    for (int k = 0; k < 4; ++k) {
      const double phi = deg[k] * M_PI / 180.0;
      const double bs[2] = {100, 30};
      for (int j = 0; j < 2; ++j) {
        const double b = bs[j];
        const int n = int(4 * b) + 40;          // comfortably past convergence
        host::Body<Wedge> body(n, n, 1);
        body.shape.half_beam = Real(b);
        body.shape.smooth = Real(1);
        body.shape.set_deadrise(Real(phi));
        body.shape.cx = Real(n / 2);  body.shape.cy = Real(n / 2);
        body.shape.set_angle(Real(0));
        const double pc = 100.0 * (body.indicator_moments().area /
                                   (b * b * std::tan(phi)) - 1.0);
        const double want = (j == 0) ? want100[k] : want30[k];
        char buf[128];
        std::snprintf(buf, sizeof buf, "wedge %2.0f deg, b = %3.0f: apex excess",
                      deg[k], b);
        check(std::fabs(pc - want) < 0.05, buf, pc, want);
      }
    }
    note("the excess grows as the wedge sharpens or the body shrinks against `smooth`");
  }

  //===========================================================================
  std::printf("\n2. NEWTON, WITH NO FLUID MOTION\n\n");
  //===========================================================================
  //
  // The two cases that decide whether the arrangement is the right one.
  //
  // Uhlmann's classical step divides by (m_b - m_f): it VANISHES for a
  // neutrally buoyant body and CHANGES SIGN for a light one, so floating -- the
  // only interesting thing a density ratio is FOR -- is precisely what it
  // cannot express. The arrangement in body.cuh divides by (m_b + m_f) instead,
  // which is positive for every mass, and the neutrally buoyant case becomes
  // the best-conditioned point on the line rather than a division by zero.
  //
  // With the fluid at rest the momentum deficit dP is zero, so the whole
  // response is the buoyancy term and has a closed form:
  //
  //     dU = (m_b - m_f) g / (m_b + m_f).
  {
    const int n = 96;
    const double h = 12.0, g = 1e-4;

    auto fall = [&](double rho_b) {
      host::Body<Rect> body(n, n, 1);
      body.shape.hx = Real(h);  body.shape.hy = Real(h);
      body.shape.smooth = Real(1.5);
      body.shape.cx = Real(n / 2);  body.shape.cy = Real(n / 2);
      body.shape.set_angle(Real(0));
      body.set_uniform_density(Real(rho_b));
      body.props.by = Real(-g);
      body.props.free_rotation = false;         // no torque in this case anyway
      std::vector<Real> ux(std::size_t(n) * n, Real(0)), uy(ux.size(), Real(0));
      body.couple_velocity(ux.data(), uy.data());
      const BodyReaction r = body.refresh(UniformDensity{Real(1)});
      return std::pair<double, double>(double(body.vy), r.fluid_mass);
    };

    // Neutrally buoyant: m_b = m_f exactly, since set_uniform_density integrates
    // the SAME chi the probe does. This is the division by zero, and it hovers.
    // The bound is relative to ONE STEP'S gravitational increment, not absolute:
    // m_b and m_f are equal in exact arithmetic but m_b is formed in Real and
    // m_f accumulated in double, so in FP32 they differ by about a part in 10^7
    // of themselves and the body drifts at 6e-7 of g per step. In FP64 it is
    // 12 orders smaller. That floor is the precision, not the scheme.
    const std::pair<double, double> neutral = fall(1.0);
    check(std::fabs(neutral.first) < 1e-5 * g,
          "neutrally buoyant: the body hovers (Uhlmann divides by zero here)",
          neutral.first / g, 0.0);

    // Heavy: dU = (10-1)/(10+1) g = 0.8181818 g.
    const std::pair<double, double> heavy = fall(10.0);
    const double want = -(10.0 - 1.0) / (10.0 + 1.0) * g;
    check(std::fabs(heavy.first - want) / std::fabs(want) < 1e-5,
          "heavy body: dU = (m_b - m_f) g / (m_b + m_f)", heavy.first, want);

    // Light: it rises, which is the sign change the classical denominator makes
    // a singularity rather than a physical answer.
    const std::pair<double, double> light = fall(0.2);
    const double wantl = -(0.2 - 1.0) / (0.2 + 1.0) * g;
    check(light.first > 0 && std::fabs(light.first - wantl) / std::fabs(wantl) < 1e-5,
          "light body: it RISES, at the same closed form", light.first, wantl);
    note("the denominator is m_b + m_f, so no mass ratio is a special case");
  }

  //===========================================================================
  std::printf("\n3. THE REACTION IS -SUM F\n\n");
  //===========================================================================
  //
  // The force and torque reported by refresh() are computed in CLOSED FORM from
  // the solve, not by a second reduction -- so they are a derived diagnostic,
  // exact against -sum F but not an independent measurement of it. This checks
  // the derivation rather than the physics, which is exactly what it is for: if
  // the closed form and the array ever disagree, one of them is wrong and there
  // would otherwise be nothing to say which.
  {
    const int n = 96;
    const double h = 10.0;
    host::Body<Rect> body(n, n, 1);
    body.shape.hx = Real(h);  body.shape.hy = Real(0.6 * h);
    body.shape.smooth = Real(1.5);
    body.shape.cx = Real(n / 2 + 0.3);  body.shape.cy = Real(n / 2 - 0.7);
    body.shape.set_angle(Real(0.4));            // tilted, so the torque is real
    body.set_uniform_density(Real(3));
    body.props.by = Real(-1e-4);

    // A sheared velocity field, so dP and dL are both non-zero.
    std::vector<Real> ux(std::size_t(n) * n, Real(0)), uy(ux.size(), Real(0));
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        ux[std::size_t(node_id(x, y, 0, n, n))] = Real(1e-3 * (y - n / 2) / double(n));
        uy[std::size_t(node_id(x, y, 0, n, n))] = Real(5e-4 * (x - n / 2) / double(n));
      }
    body.couple_velocity(ux.data(), uy.data());

    const BodyReaction r = body.refresh(UniformDensity{Real(1)});

    double sx = 0, sy = 0, tz = 0;
    for (long m = 0; m < long(n) * n; ++m) {
      int x, y, z;
      coords(m, n, n, x, y, z);
      const double Fx = double(body.fx()[m]), Fy = double(body.fy()[m]);
      sx += Fx;  sy += Fy;
      const double rx = double(x) - double(body.shape.cx);
      const double ry = double(y) - double(body.shape.cy);
      tz += rx * Fy - ry * Fx;
    }
    const double scale = std::fabs(r.fx) + std::fabs(r.fy) + 1e-30;
    check(std::fabs(r.fx + sx) / scale < 1e-5, "reaction f_x = -sum F_x", r.fx, -sx);
    check(std::fabs(r.fy + sy) / scale < 1e-5, "reaction f_y = -sum F_y", r.fy, -sy);
    const double tscale = std::fabs(r.torque) + 1e-30;
    check(std::fabs(r.torque + tz) / tscale < 1e-4, "reaction torque = -sum r x F",
          r.torque, -tz);
  }

  //===========================================================================
  std::printf("\n4. A BODY IN A FLUID: prescribed motion drags it\n\n");
  //===========================================================================
  //
  // The end-to-end test. A square is held at a prescribed velocity in still
  // fluid, its force written into the field the fluid's ForceField kind reads,
  // and the fluid inside it must reach that velocity -- which is the whole
  // claim of direct forcing.
  //
  // THE ORDER IS THE POINT AND IS THE DRIVER'S RESPONSIBILITY HERE, unlike the
  // phase field where step() owns it: refresh_velocity, then the body's
  // refresh(), then the fluid step. Running the body AFTER the fluid gives it a
  // velocity field that already contains its own next force, which is the
  // double-counting body.cuh's banner records as diverging within a few hundred
  // steps. That path is live: refresh_velocity() is exactly the diagnostic pass
  // whose missing half-force shift (macro_node, solver.cuh) turned this case
  // into a field of NaN when it was first written.
  //
  // THE CHANNEL HAS WALLS, AND IT HAS TO. In a fully periodic box a body driven
  // at U is a momentum source with nowhere to put momentum, so the whole fluid
  // accelerates toward U and there is no far field to check -- measured, 92% of
  // U at the far corner, which is correct physics and a useless test. Walls give
  // the momentum somewhere to go, so "the body drags its neighbourhood and not
  // the whole domain" becomes a statement with content.
  {
    const int n = 64;
    const double h = 8.0, U = 0.01, nu = 1.0 / 6.0;

    host::Fluid fl(n, n, 1, Op::BGK, Real(nu));
    fl.enable_velocity_output();
    std::vector<std::uint8_t> flags(std::size_t(n) * n, std::uint8_t(Fluid));
    for (int x = 0; x < n; ++x) {
      flags[std::size_t(node_id(x, 0, 0, n, n))] = Solid;
      flags[std::size_t(node_id(x, n - 1, 0, n, n))] = Solid;
    }
    fl.set_geometry(flags);
    fl.initialise_with([](int, int, int) {
      return Macro{Real(1), Real(0), Real(0), Real(0)};
    });

    host::Body<Rect> body(n, n, 1);
    body.shape.hx = Real(h);  body.shape.hy = Real(h);
    body.shape.smooth = Real(1.5);
    body.shape.cx = Real(n / 2);  body.shape.cy = Real(n / 2);
    body.shape.set_angle(Real(0));
    body.props.free_translation = false;        // prescribed
    body.props.free_rotation = false;
    body.vx = Real(U);
    body.couple_velocity(fl.ux_device(), fl.uy_device());

    BodyForce bf;
    bf.Fx = body.fx();  bf.Fy = body.fy();  bf.Fz = body.fz();
    fl.set_force(bf, ForceField);

    for (int t = 0; t < 4000; ++t) {
      fl.refresh_velocity();
      body.refresh(UniformDensity{Real(1)});
      fl.step();
    }

    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    // Well inside the body, where chi = 1 to machine precision.
    double worst = 0;
    for (int y = n / 2 - 4; y <= n / 2 + 4; ++y)
      for (int x = n / 2 - 4; x <= n / 2 + 4; ++x)
        worst = std::fmax(worst,
                          std::fabs(double(ux[std::size_t(node_id(x, y, 0, n, n))]) - U));
    check(worst / U < 0.02, "the fluid inside the body reaches the body velocity",
          worst / U, 0.0);

    // And at the wall it is still near rest -- the body drags its neighbourhood,
    // not the domain, and a runaway would show here first.
    const double far = std::fabs(double(ux[std::size_t(node_id(n / 2, 1, 0, n, n))]));
    check(far < 0.35 * U, "and the fluid at the wall has not run away", far / U, 0.0);
    std::printf("        (interior error %.2f%% of U; far field %.2f%% of U)\n",
                100.0 * worst / U, 100.0 * far / U);
  }

  std::printf("\n[body] %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
