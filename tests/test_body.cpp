//==============================================================================
//  The rigid-body side of volume penalisation, checked against its own algebra.
//
//  Nothing here runs a flow. The three things this file pins down are the three
//  that a simulation would hide rather than expose:
//
//   1. THE INDICATOR TURNS. chi is evaluated in the body frame, so the shape has
//      to rotate with theta while its area and second moment do not. A rotation
//      applied to the wrong side of the transform gives a shape that turns the
//      wrong way, which in a symmetric test case looks like nothing at all.
//   2. THE 3x3 IS SOLVED. The coupled sway-heave-roll system is solved in closed
//      form through its Schur complement, and the only honest check of a closed
//      form is to put the answer back into the equations. Done here for a body
//      heavier than the fluid it displaces, one exactly as heavy -- which is the
//      case the classical Uhlmann arrangement divides by zero on -- and one ten
//      times lighter, which is the case it gets the sign wrong on.
//   3. THE REPORTED REACTION IS THE APPLIED FORCE. R and the torque are returned
//      in closed form rather than reduced from the force array, which is worth a
//      full sweep a step and is exact -- but only if the closed form is right.
//      This sums the array that was actually written and compares. It is the one
//      check here that touches a field.
//
//  A note on what is NOT checked: that the equations themselves are the right
//  ones. That is hydrostatics, it has closed forms of its own, and it lives in
//  validation/floating_body.cpp.
//==============================================================================
#include "Check.hpp"
#include "core/Types.hpp"
#include "solver/PenalisedBody.hpp"

#include <cmath>
#include <vector>

using namespace lbm;
using Body = PenalisedBody<D2Q9>;

static Real TOL() { return sizeof(Real) == 4 ? Real(1e-5) : Real(1e-12); }
// Everything algebraic below is checked RELATIVE to the size of the terms in
// it: the 3x3 carries masses of order 1e5 against accelerations of order 1e-4,
// so an absolute tolerance would be meaningless in FP64 and unsatisfiable in
// FP32. The Schur complement is nowhere near cancelling for these numbers --
// |S|^2/A is a few per cent of B -- so single precision loses only its own
// epsilon and this margin is generous rather than tuned.
static double RTOL() { return sizeof(Real) == 4 ? 2e-5 : 1e-11; }

//------------------------------------------------------------------------------
// Put the solution back into the 3x3 and report the largest row residual,
// scaled by the size of the terms in it so the number means something.
//------------------------------------------------------------------------------
static double residual(const Body& b, const BodySums& q,
                       Real dux, Real duy, Real dw) {
  const double A = double(b.mass) + double(q.m);
  const double B = double(b.inertia) + double(q.Iz);
  const double Sx = double(q.Sx), Sy = double(q.Sy);
  const double rx = 2.0 * double(q.Px) + (double(b.mass) - double(q.m)) * double(b.bx);
  const double ry = 2.0 * double(q.Py) + (double(b.mass) - double(q.m)) * double(b.by);
  const double rw = 2.0 * double(q.Lz) - (Sx * double(b.by) - Sy * double(b.bx));
  const double e1 = A * dux - Sy * dw - rx;
  const double e2 = A * duy + Sx * dw - ry;
  const double e3 = -Sy * dux + Sx * duy + B * dw - rw;
  const double s1 = std::fabs(A * dux) + std::fabs(Sy * dw) + std::fabs(rx);
  const double s2 = std::fabs(A * duy) + std::fabs(Sx * dw) + std::fabs(ry);
  const double s3 = std::fabs(Sy * dux) + std::fabs(Sx * duy)
                  + std::fabs(B * dw) + std::fabs(rw);
  return std::max(std::max(std::fabs(e1) / s1, std::fabs(e2) / s2),
                  std::fabs(e3) / s3);
}

// A representative measurement: a body straddling an interface, so the first
// moments are non-zero and the roll and sway rows are genuinely coupled.
static BodySums sample() {
  BodySums q;
  q.m  = Real(2.5e4);
  q.Sx = Real(-3.1e3);   q.Sy = Real(-1.7e5);
  q.Iz = Real(4.2e6);
  q.Px = Real(1.3e1);    q.Py = Real(-7.4e0);
  q.Lz = Real(2.6e2);
  return q;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const Real tol = TOL();

    //--------------------------------------------------------------------------
    std::printf("\n1. the indicator turns, its moments do not\n");
    {
      Domain d(96, 96, 1, true, true, true);
      Body b(d);
      // Deliberately not square: a square would rotate onto itself and every
      // check below would pass without the transform doing anything.
      b.shape = Rect{Real(48), Real(48), Real(20), Real(6), Real(1.5)};

      // A point 12 cells out along the body's long axis is inside; the same
      // world point after a quarter turn is outside, and its perpendicular
      // partner has swapped places with it.
      const Real in0  = b.shape.chi(Real(60), Real(48), Real(0));
      const Real out0 = b.shape.chi(Real(48), Real(60), Real(0));
      b.shape.set_angle(Real(M_PI / 2));
      const Real in1  = b.shape.chi(Real(48), Real(60), Real(0));
      const Real out1 = b.shape.chi(Real(60), Real(48), Real(0));
      check::near(in0,  Real(1), Real(1e-3), "long axis is inside at theta = 0");
      check::near(out0, Real(0), Real(1e-3), "short axis is outside at theta = 0");
      check::near(in1,  in0,  tol, "a quarter turn carries the inside point round");
      check::near(out1, out0, tol, "and carries the outside point with it");

      // theta is tracked unwrapped, so a body that keeps turning keeps counting.
      b.shape.set_angle(Real(5 * M_PI / 2));
      check::near(b.shape.theta, Real(5 * M_PI / 2), tol,
                  "theta is unwrapped, not folded by atan2");
      check::near(b.shape.chi(Real(48), Real(60), Real(0)), in1, tol,
                  "and chi is periodic in it anyway");

      // Area and second moment are properties of the shape, not the pose.
      double a[3], s[3];
      const double th[3] = {0.0, M_PI / 6, M_PI / 2};
      for (int k = 0; k < 3; ++k) {
        b.shape.set_angle(Real(th[k]));
        const Body::Moments m = b.indicator_moments();
        a[k] = double(m.area);  s[k] = double(m.second);
      }
      check::near(a[1] / a[0], 1.0, 5e-3, "area is rotation invariant  (30 deg)");
      check::near(a[2] / a[0], 1.0, 5e-3, "area is rotation invariant  (90 deg)");
      check::near(s[1] / s[0], 1.0, 5e-3, "second moment likewise      (30 deg)");
      check::near(s[2] / s[0], 1.0, 5e-3, "second moment likewise      (90 deg)");
      // ... and both match the rectangle they describe, which is what says the
      // smoothing has not quietly grown the body.
      check::near(a[0] / (40.0 * 12.0), 1.0, 1e-2, "area matches 2hx . 2hy");
      // The second moment does NOT match the sharp rectangle's, and must not:
      // the tanh smoothing moves equal amounts of indicator across each face,
      // and the r^2 weight values what went out more highly than what came in.
      // A few per cent larger is the correct answer, and it is the whole reason
      // mass and inertia are integrals of chi rather than textbook formulae --
      // a body given the nominal inertia would roll a few per cent too fast.
      const double excess = s[0] / (a[0] * (400.0 + 36.0) / 3.0) - 1.0;
      check::ok(excess > 0.0 && excess < 0.06,
                "second moment exceeds the sharp rectangle's, by a few per cent");
      std::printf("        (excess %.2f%%, from a smoothing width of %.1f cells)\n",
                  100.0 * excess, 1.5);
    }

    //--------------------------------------------------------------------------
    std::printf("\n2. the 3x3 is solved, at every density ratio\n");
    {
      Domain d(8, 8, 1, true, true, true);
      Body b(d);
      b.shape = Rect{Real(4), Real(4), Real(2), Real(2), Real(1.5)};
      b.inertia = Real(3.5e6);
      b.by = Real(-2e-4);
      const BodySums q = sample();

      const double ratios[3] = {5.0, 1.0, 0.1};   // m_b / m_f
      const char* names[3] = {"heavy body  (m_b = 5 m_f)",
                              "neutral     (m_b =   m_f)",
                              "light body  (m_b = 0.1 m_f)"};
      for (int k = 0; k < 3; ++k) {
        b.mass = Real(ratios[k] * double(q.m));
        Real dux = 0, duy = 0, dw = 0;
        b.solve(q, dux, duy, dw);
        check::near(residual(b, q, dux, duy, dw), 0.0, RTOL(),
                    std::string("solved to round-off: ") + names[k]);
        check::ok(std::isfinite(double(dux)) && std::isfinite(double(duy))
                      && std::isfinite(double(dw)),
                  std::string("finite: ") + names[k]);
      }

      // Uhlmann's arrangement is singular at m_b = m_f. Ours is best conditioned
      // there, and the check that it is the SAME equation is that the closed-form
      // reaction still satisfies his relation wherever his is defined.
      b.mass = Real(5.0 * double(q.m));
      b.free_rotation = false;
      BodySums q2 = q;  q2.Sx = 0; q2.Sy = 0; q2.Lz = 0;
      Real dux = 0, duy = 0, dw = 0;
      b.solve(q2, dux, duy, dw);
      const double meff = double(b.mass) - double(q2.m);
      const double Rx = 2.0 * double(q2.Px) - 2.0 * double(q2.m) * double(dux);
      const double us = std::fabs(meff * double(dux)) + std::fabs(Rx)
                      + std::fabs(meff * double(b.bx));
      check::near((meff * double(dux) - (Rx + meff * double(b.bx))) / us, 0.0,
                  RTOL(), "agrees with (m_b - m_f) dU = R + (m_b - m_f) g");
      check::near(dw, Real(0), tol, "locked rotation gives no roll");
      b.free_rotation = true;

      // A body alone with gravity and no fluid falls at g, exactly.
      BodySums vac;  vac.m = 0; vac.Iz = 0;
      b.mass = Real(1000);  b.inertia = Real(1e4);
      b.solve(vac, dux, duy, dw);
      check::near(double(duy) / double(b.by), 1.0, RTOL(),
                  "free fall in vacuum accelerates at g");
      check::near(dw,  Real(0), tol, "and does not start spinning");

      // Locking translation must leave the roll row uncoupled, not solve the
      // 3x3 and throw two rows away.
      b.mass = Real(2.0 * double(q.m));
      b.free_translation = false;
      b.solve(q, dux, duy, dw);
      const double rw = 2.0 * double(q.Lz)
                      - (double(q.Sx) * double(b.by) - double(q.Sy) * double(b.bx));
      const double Bt = (double(b.inertia) + double(q.Iz)) * double(dw);
      check::near(dux, Real(0), tol, "locked translation gives no sway");
      check::near((Bt - rw) / (std::fabs(Bt) + std::fabs(rw)), 0.0, RTOL(),
                  "and the roll row stands alone");
    }

    //--------------------------------------------------------------------------
    std::printf("\n3. the reported reaction is the force that was written\n");
    {
      // Wide enough that the body can be moved clear of where it started with
      // no chance of the two footprints meeting round the periodic seam.
      const Index N = 96;
      Domain d(N, N, 1, true, true, true);
      Body b(d);
      b.shape = Rect{Real(30), Real(34), Real(11), Real(7), Real(1.5)};
      b.shape.set_angle(Real(0.35));
      b.vx = Real(0.01);  b.vy = Real(-0.004);  b.omega = Real(1.5e-3);
      b.by = Real(-3e-4);
      b.set_uniform_density(Real(20));

      // A velocity field with shear and a stratified density, so that every one
      // of the seven integrals is non-zero and no term can cancel by symmetry.
      View1D<Real> ux("ux", d.n_padded), uy("uy", d.n_padded);
      View1D<Real> rho("rho", d.n_padded);
      const Domain dd = d;
      const Index hx = d.hx, hy = d.hy;
      Kokkos::parallel_for(Range(0, d.n_padded), KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; dd.coords(n, px, py, pz);
        const Real X = Real(px - hx), Y = Real(py - hy);
        ux(n) = Real(0.02) * Kokkos::sin(Real(0.1) * Y) + Real(0.003) * X / Real(N);
        uy(n) = Real(-0.015) * Kokkos::cos(Real(0.07) * X);
        rho(n) = Real(1) + Real(19) * Real(0.5)
               * (Real(1) + Kokkos::tanh((Real(34) - Y) / Real(3)));
      });
      Kokkos::fence();
      b.set_velocity(ux, uy);

      // Two steps, so the second one runs with a force array already populated
      // and the u* correction is actually exercised.
      const auto dens = KOKKOS_LAMBDA(Index n) { return rho(n); };
      b.refresh(dens);
      const Body::Reaction R = b.refresh(dens);

      auto hfx = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.x());
      auto hfy = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.y());
      auto hfz = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.z());
      double sx = 0, sy = 0, st = 0, sz = 0;
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) {
          const Index n = d.id(x, y);
          const double gx = double(hfx(n)), gy = double(hfy(n));
          const double rx = double(x) - double(b.shape.cx);
          const double ry = double(y) - double(b.shape.cy);
          sx += gx;  sy += gy;  st += rx * gy - ry * gx;
          sz += std::fabs(double(hfz(n)));
        }
      const double scale = std::fabs(sx) + std::fabs(sy);
      check::near(double(R.fx) + sx, 0.0, RTOL() * scale, "R_x  = -sum F_x");
      check::near(double(R.fy) + sy, 0.0, RTOL() * scale, "R_y  = -sum F_y");
      check::near(double(R.torque) + st, 0.0, RTOL() * scale * 40.0,
                  "T    = -sum r x F");
      check::near(sz, 0.0, 1e-30, "the out-of-plane force is identically zero");
      std::printf("        (|R| = %.4e, |T| = %.4e, m_f = %.4e)\n",
                  std::hypot(double(R.fx), double(R.fy)), double(R.torque),
                  double(R.fluid_mass));

      // The body left where it was would keep pushing, so advancing must not
      // leave a stale force behind. Move it a long way and check the vacated
      // ground is clean.
      const Real cx0 = b.shape.cx;
      const double keep = double(b.shape.reach());
      b.shape.cx = Real(70);
      b.refresh(dens);
      auto gfx = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.x());
      double stale = 0;
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x)
          if (std::fabs(double(x) - double(cx0)) <= keep)
            stale = std::max(stale, std::fabs(double(gfx(d.id(x, y)))));
      check::near(stale, 0.0, 1e-30, "no force is left where the body used to be");
    }

    //-------------------------------------------------------------- wedge
    // The Wedge indicator, checked against the geometry it claims rather than
    // against itself. half_beam = 100 with smooth = 1 so the body is resolved:
    // at half_beam = 30 and 15 degrees the wedge is only 8 cells tall and the
    // face and cap smoothing regions overlap, so chi never reaches 1 anywhere
    // inside -- which is a true statement about a wedge that small, not about
    // the shape, and is why the sizes here are what they are.
    std::printf("\n-- the wedge indicator --\n");
    {
      const double PI = 3.14159265358979323846;
      // Measured apex-rounding excess at b = 100, smooth = 1; see the banner.
      const double excess[3] = {1.00232, 1.00058, 1.00025};
      int k = 0;
      for (double deg : {15.0, 30.0, 45.0}) {
        Wedge w;
        w.cx = 0; w.cy = 0; w.half_beam = Real(100); w.smooth = Real(1.0);
        w.set_deadrise(Real(deg * PI / 180.0));
        w.set_angle(Real(0));
        const double H = double(w.height());
        const std::string at = " at " + std::to_string(int(deg));
        check::near(double(w.chi(Real(0), Real(0.5 * H), Real(0))), 1.0, 1e-9,
                    "chi = 1 inside the wedge" + at);
        // Placed at a fixed FACE-NORMAL distance, not a fixed axial one: chi
        // decays as exp(-2d) in the normal distance, so an axial probe would
        // sit 15 cells out at 0 degrees and 10.6 at 45, and the tolerance
        // would be measuring the deadrise angle rather than the indicator.
        const double below = 15.0 / std::cos(deg * PI / 180.0);
        check::near(double(w.chi(Real(0), Real(-below), Real(0))), 0.0, 1e-12,
                    "chi = 0 below the apex" + at);
        check::near(double(w.chi(Real(0), Real(H + 12), Real(0))), 0.0, 1e-9,
                    "chi = 0 above the knuckle" + at);
        // On a face, away from apex and knuckle, the cap factor is 1 to
        // round-off and the indicator is exactly one half.
        const double xf = 0.5 * double(w.half_beam);
        const double yf = xf * std::tan(deg * PI / 180.0);
        check::near(double(w.chi(Real(xf), Real(yf), Real(0))), 0.5, 1e-9,
                    "chi = 1/2 on the face" + at);
        check::near(double(w.chi(Real(xf), Real(yf + 3), Real(0))),
                    double(w.chi(Real(-xf), Real(yf + 3), Real(0))), 1e-14,
                    "chi is symmetric about the axis" + at);
        // The integral of chi is the nominal area PLUS the apex rounding, and
        // the rounding is pinned rather than tolerated: it is the quantity that
        // decides how much bigger than nominal the penalised body actually is.
        double area = 0;
        const double h = 0.25;
        const int xi = int((double(w.half_beam) + 10.0) / h);
        const int ylo = int(-10.0 / h), yhi = int((H + 10.0) / h);
        for (int i = -xi; i <= xi; ++i)
          for (int j = ylo; j <= yhi; ++j)
            area += double(w.chi(Real(i * h), Real(j * h), Real(0))) * h * h;
        const double ratio = area / (double(w.half_beam) * H);
        check::near(ratio, excess[k], 5e-4,
                    "integral of chi matches the recorded apex excess" + at);
        std::printf("        (%2d deg: chi integral is %.5f x nominal)\n",
                    int(deg), ratio);
        ++k;
      }
    }
  }
  const int rc = check::report("body");
  Kokkos::finalize();
  return rc;
}
