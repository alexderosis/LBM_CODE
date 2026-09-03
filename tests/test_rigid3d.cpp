//==============================================================================
//  The six-degree-of-freedom rigid body, before any fluid touches it.
//
//  Everything here is algebra on the 6x6 system and the quaternion, with no
//  lattice and no flow, because that is the part that can be checked exactly.
//  Whether the equations are the RIGHT equations for a body in a fluid is
//  hydrostatics and lives in validation/floating_body.cpp; whether they are
//  SOLVED correctly is here.
//
//  THE CHECK THAT MATTERS IS BLOCK 3. The 2-D solve in PenalisedBody.hpp is
//  validated -- floating_body reproduces Archimedes' draft to 0.14 % and gets
//  the sign of the metacentric height right for a raft and a pillar. So the new
//  6x6 is not asked to be plausible; it is asked to REPRODUCE that 3x3 exactly
//  on a planar problem. If it does, the generalisation carries the older code's
//  validation with it. If it does not, one of them is wrong and the 3x3 is the
//  one with evidence behind it.
//==============================================================================
#include "Check.hpp"
#include "core/Types.hpp"
#include "solver/PenalisedBody.hpp"
#include "solver/RigidBody3D.hpp"

#include <cstdio>

using namespace lbm;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Six-DOF rigid body: quaternion, inertia tensor, 6x6 solve\n");
    std::printf("precision %s\n\n", precision_name());

    const Real tol = sizeof(Real) == 4 ? Real(2e-4) : Real(1e-11);

    //-------------------------------------------------------------------------
    std::printf("1. THE ROTATION IS A ROTATION\n\n");
    {
      Quat q = Quat::from_axis_angle(Real(0), Real(0), Real(1),
                                     Real(1.5707963267948966));
      const Mat3 R = q.matrix();
      Real ox, oy, oz;
      R.mul(Real(1), Real(0), Real(0), ox, oy, oz);
      check::near(ox, Real(0), tol, "quarter turn about z: x -> y, x part");
      check::near(oy, Real(1), tol, "                              y part");
      check::near(oz, Real(0), tol, "                              z part");

      // R^T R = I to round-off. A rotation that is not orthonormal is a
      // rotation plus a stretch, and the stretch would read as the body
      // changing size rather than as an error.
      Real worst = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          Real s = 0;
          for (int k = 0; k < 3; ++k) s += R(k, i) * R(k, j);
          const Real want = (i == j) ? Real(1) : Real(0);
          const Real e = Kokkos::fabs(s - want);
          if (e > worst) worst = e;
        }
      check::near(worst, Real(0), tol, "R^T R = I, worst entry");

      // A tilt about the (1,1,0) diagonal is the pose a cube needs to strike
      // corner first, so it is worth one check that it is not degenerate.
      Quat d = Quat::from_axis_angle(Real(1), Real(1), Real(0), Real(0.6));
      const Mat3 Rd = d.matrix();
      Real dx, dy, dz;
      Rd.mul(Real(0), Real(0), Real(1), dx, dy, dz);
      const Real len = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
      check::near(len, Real(1), tol, "diagonal-axis tilt preserves length");
      check::ok(Kokkos::fabs(dz - Real(1)) > Real(1e-3),
                "and actually moves z (the tilt is not a no-op)");
    }

    //-------------------------------------------------------------------------
    std::printf("\n2. THE INERTIA TENSOR OF A BOX, FROM THE PRODUCTS J\n\n");
    {
      // For a uniform box of half-extents h and mass m, integral rho x^2 is
      // m hx^2 / 3, so J is diagonal with those entries and I_f = tr(J) I - J
      // must give the textbook m (hy^2 + hz^2) / 3 on the diagonal.
      const Real hx = Real(3), hy = Real(5), hz = Real(7);
      const Real m = Real(8) * hx * hy * hz;          // rho = 1
      BodySums6 q;
      q.m = m;
      q.Jxx = m * hx * hx / Real(3);
      q.Jyy = m * hy * hy / Real(3);
      q.Jzz = m * hz * hz / Real(3);
      const Mat3 I = q.fluid_inertia();
      check::near(I(0, 0), m * (hy * hy + hz * hz) / Real(3), tol * m,
                  "I_xx = m (hy^2 + hz^2) / 3");
      check::near(I(1, 1), m * (hx * hx + hz * hz) / Real(3), tol * m,
                  "I_yy = m (hx^2 + hz^2) / 3");
      check::near(I(2, 2), m * (hx * hx + hy * hy) / Real(3), tol * m,
                  "I_zz = m (hx^2 + hy^2) / 3");
      check::near(I(0, 1), Real(0), tol * m, "off-diagonal zero for a box on axis");
      std::printf("        and I_zz is the OLD scalar Iz: Jxx + Jyy = %.6e\n",
                  double(q.Jxx + q.Jyy));
      check::near(I(2, 2), q.Jxx + q.Jyy, tol * m,
                  "I_zz equals the 2-D code's integral chi rho (rx^2 + ry^2)");
    }

    //-------------------------------------------------------------------------
    std::printf("\n3. THE 6x6 REPRODUCES THE VALIDATED 3x3 ON A PLANAR PROBLEM\n\n");
    {
      // A planar body: rz = 0 everywhere, so Sz = Jzz = Jxz = Jyz = 0, the
      // angular deficit is along z alone, and gravity is in the plane. Under
      // those conditions the 6x6 must return the 3x3's answer and zero for the
      // two rotations the 2-D model does not have.
      Domain dom(8, 8);
      PenalisedBody<D2Q9, Rect> two(dom);
      two.mass = Real(37);
      two.inertia = Real(910);
      two.bx = Real(0.001);
      two.by = Real(-0.004);

      BodySums s2;
      s2.m = Real(11);
      s2.Sx = Real(2.5);   s2.Sy = Real(-1.75);
      s2.Iz = Real(64);                       // = Jxx + Jyy, set to match below
      s2.Px = Real(0.31);  s2.Py = Real(-0.22);
      s2.Lz = Real(0.47);

      Real dux2 = 0, duy2 = 0, dw2 = 0;
      two.solve(s2, dux2, duy2, dw2);

      BodySums6 s6;
      s6.m = s2.m;
      s6.Sx = s2.Sx;  s6.Sy = s2.Sy;  s6.Sz = 0;
      // Any split of Jxx + Jyy = Iz will do for the zz entry, and the in-plane
      // entries only enter the two rotations the planar case must return as
      // zero -- so this also tests that they DO come back zero.
      s6.Jxx = Real(25);  s6.Jyy = Real(39);  s6.Jzz = 0;
      s6.Jxy = Real(4);   s6.Jxz = 0;  s6.Jyz = 0;
      s6.Px = s2.Px;  s6.Py = s2.Py;  s6.Pz = 0;
      s6.Lx = 0;  s6.Ly = 0;  s6.Lz = s2.Lz;

      Body6Properties p6;
      p6.mass = two.mass;
      p6.bx = two.bx;  p6.by = two.by;  p6.bz = 0;
      // The body's own tensor: only its zz entry exists in the 2-D model. The
      // in-plane entries are given values so the matrix is non-singular; they
      // must not affect the answer, which is part of what this checks.
      p6.inertia_world(0, 0) = Real(500);
      p6.inertia_world(1, 1) = Real(600);
      p6.inertia_world(2, 2) = two.inertia;

      Real dU[3], dW[3];
      body6_solve(p6, s6, dU, dW);

      std::printf("        3x3: dux %.12e  duy %.12e  dw %.12e\n",
                  double(dux2), double(duy2), double(dw2));
      std::printf("        6x6: dux %.12e  duy %.12e  dw %.12e\n",
                  double(dU[0]), double(dU[1]), double(dW[2]));
      const Real rel = Kokkos::fabs(dux2) + Kokkos::fabs(duy2) + Kokkos::fabs(dw2);
      check::near(dU[0], dux2, tol * (rel + Real(1)), "dux matches the 3x3");
      check::near(dU[1], duy2, tol * (rel + Real(1)), "duy matches the 3x3");
      check::near(dW[2], dw2,  tol * (rel + Real(1)), "domega_z matches the 3x3");
      check::near(dU[2], Real(0), tol, "no out-of-plane translation");
      check::near(dW[0], Real(0), tol, "no rotation about x");
      check::near(dW[1], Real(0), tol, "no rotation about y");
    }

    //-------------------------------------------------------------------------
    std::printf("\n4. NEWTON, WITH NO FLUID MOTION AND NO FIRST MOMENT\n\n");
    {
      // The same closed form the 2-D test pins: with the body concentric on the
      // penalised region there is no coupling, and the acceleration is
      // (m_b - m_f) g / (m_b + m_f) -- a denominator that makes no mass ratio a
      // special case, including a neutrally buoyant body.
      const Real g = Real(-1e-4);
      for (Real chib : {Real(2), Real(1), Real(0.5)}) {
        BodySums6 q;
        q.m = Real(1000);
        q.Jxx = q.Jyy = q.Jzz = Real(4000);
        Body6Properties p;
        p.mass = chib * q.m;
        p.by = g;
        p.inertia_world(0, 0) = p.inertia_world(1, 1) = p.inertia_world(2, 2)
            = chib * Real(4000);
        Real dU[3], dW[3];
        body6_solve(p, q, dU, dW);
        const Real want = (p.mass - q.m) * g / (p.mass + q.m);
        char buf[96];
        std::snprintf(buf, sizeof buf,
                      "rho_b/rho_f = %.1f: dU_y = (m_b - m_f) g / (m_b + m_f)",
                      double(chib));
        check::near(dU[1], want, tol * (Kokkos::fabs(want) + Real(1)), buf);
        check::near(dW[0] + dW[1] + dW[2], Real(0), tol,
                    "  and no rotation from a centred body");
      }
    }

    //-------------------------------------------------------------------------
    std::printf("\n5. A PURE TORQUE TURNS THE BODY AND DOES NOT MOVE IT\n\n");
    {
      // S = 0 decouples the blocks, so an angular deficit alone must give
      // domega = 2 dL / (I_b + I_f) with no translation at all. This is the one
      // place the OFF-DIAGONAL blocks are checked to be doing nothing when they
      // should: if [S]x were assembled with a wrong sign it would still vanish
      // here, but block 3 would have caught that.
      BodySums6 q;
      q.m = Real(500);
      q.Jxx = q.Jyy = q.Jzz = Real(1200);
      q.Lx = Real(0.09);  q.Ly = Real(-0.04);  q.Lz = Real(0.02);
      Body6Properties p;
      p.mass = Real(500);
      p.inertia_world(0, 0) = p.inertia_world(1, 1) = p.inertia_world(2, 2)
          = Real(2400);
      Real dU[3], dW[3];
      body6_solve(p, q, dU, dW);
      const Mat3 If = q.fluid_inertia();
      const Real B = Real(2400) + If(0, 0);
      check::near(dW[0], Real(2) * q.Lx / B, tol * Real(10), "domega_x = 2 dLx / B");
      check::near(dW[1], Real(2) * q.Ly / B, tol * Real(10), "domega_y = 2 dLy / B");
      check::near(dW[2], Real(2) * q.Lz / B, tol * Real(10), "domega_z = 2 dLz / B");
      check::near(Kokkos::fabs(dU[0]) + Kokkos::fabs(dU[1]) + Kokkos::fabs(dU[2]),
                  Real(0), tol, "and no translation at all");
    }

    //-------------------------------------------------------------------------
    std::printf("\n6. THE QUATERNION INTEGRATES TO THE RIGHT ANGLE\n\n");
    {
      // A constant angular velocity for N unit steps must arrive at angle
      // omega N, and the explicit Euler step plus renormalisation must not
      // drift off the unit sphere while doing it.
      const Real w = Real(0.01);
      const int N = 100;
      Quat q;
      for (int i = 0; i < N; ++i) q.integrate(Real(0), Real(0), w, Real(1));
      const Real norm = Kokkos::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
      check::near(norm, Real(1), tol, "still a unit quaternion after 100 steps");
      const Mat3 R = q.matrix();
      Real ox, oy, oz;
      R.mul(Real(1), Real(0), Real(0), ox, oy, oz);
      const Real ang = Kokkos::atan2(oy, ox);
      // Explicit Euler at dt = 1 is second-order accurate in omega dt, so a
      // total angle of 1 rad accumulated in steps of 0.01 lands within ~1e-4 of
      // it. The tolerance is that, not round-off, and it is stated rather than
      // fitted.
      check::near(ang, w * Real(N), Real(2e-4),
                  "angle after 100 steps of omega = 0.01 (Euler, O(w dt)^2)");
    }

    //-------------------------------------------------------------------------
    std::printf("\n7. THE BOX, AND THE DIRECTION OF THE INERTIA ROTATION\n\n");
    // The one line in the 6-DOF path that is wrong silently. set_uniform_density6
    // measures J in the WORLD frame at whatever pose the body is in, then stores
    // I in the BODY frame; refresh6 rotates it back out each step. Get either
    // inverse the wrong way round and the inertia is still symmetric, still
    // positive definite, still gives a solvable system -- it is simply the
    // inertia of a body facing some other direction. Nothing crashes and the
    // tumble is merely wrong.
    //
    // A CUBE CANNOT CATCH THIS: a uniform cube's inertia tensor is isotropic,
    // so R I R^T = I for every R and both directions agree. The test box is
    // deliberately 6 x 6 x 20, whose I_zz is far from its I_xx.
    {
      Domain d(48, 48, 48, true, true, true);
      PenalisedBody<D3Q27, Box> b(d);
      b.shape.cx = 24;  b.shape.cy = 24;  b.shape.cz = 24;
      b.shape.hx = 3;   b.shape.hy = 3;   b.shape.hz = 10;
      b.shape.smooth = Real(1.0);
      b.shape.set_orientation(Quat{});
      b.set_uniform_density6(Real(1));

      // Mass first, against the nominal volume. chi is smoothed so the
      // penalised box is a little larger than 6 x 6 x 20 = 720; the tolerance
      // is that, and it is one-sided in the direction the smoothing goes.
      const Real vol = Real(6) * Real(6) * Real(20);
      check::near(b.mass / vol, Real(1), Real(0.06), "measured mass / nominal");

      const Mat3 I0 = b.inertia_body;
      // A slab is not isotropic: I_zz is about (hx^2+hy^2)/(hx^2+hz^2) of I_xx,
      // and if it were not the test would prove nothing.
      check::ok(I0(2, 2) < Real(0.4) * I0(0, 0),
                   "the test box really is anisotropic (Izz << Ixx)");
      check::near(I0(0, 1) / I0(0, 0), Real(0), Real(2e-3),
                  "axis-aligned box has no product of inertia");

      // Now re-measure at a tilted pose. The BODY-frame tensor is a property of
      // the body, so it must come back the same -- to lattice discretisation,
      // not to round-off, because rotating the box changes which cells chi
      // covers. With the inverse the wrong way round the two differ by tens of
      // per cent.
      const Real ang = Real(0.6154797086703873);   // atan(1/sqrt(2)), corner-down
      b.shape.set_orientation(Quat::from_axis_angle(Real(1), Real(0), Real(0), ang));
      b.set_uniform_density6(Real(1));
      const Mat3 I1 = b.inertia_body;
      Real worst = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          worst = Kokkos::fmax(worst, Kokkos::fabs(I1(i, j) - I0(i, j)) / I0(0, 0));
      std::printf("   I_body at 0 deg vs %.1f deg: worst entry differs by %.3f %%\n",
                  ang * Real(57.29577951308232), Real(100) * worst);
      check::ok(worst < Real(0.03),
                   "body-frame inertia is pose independent (R^T J R, not R J R^T)");

      // And the world-frame tensor at that pose is NOT the body one -- if it
      // were, rotate_tensor would be doing nothing and the check above would
      // pass for the wrong reason.
      const Mat3 Iw = rotate_tensor(b.shape.Rm, b.inertia_body);
      check::ok(Kokkos::fabs(Iw(1, 2)) > Real(0.05) * I0(0, 0),
                   "the tilted world-frame tensor has a real off-diagonal");
    }

    //-------------------------------------------------------------------------
    std::printf("\n8. THE DISC, AGAINST THE CLOSED-FORM CYLINDER\n\n");
    // A uniform cylinder of radius R and half thickness h has, per unit mass,
    //     I about the symmetry axis      = R^2 / 2
    //     I about any diameter           = R^2 / 4 + h^2 / 3
    // so the Disc's whole measurement path has an exact target, and unlike a
    // cube it is genuinely ANISOTROPIC -- transversely isotropic, I_yy against
    // I_xx = I_zz -- which is what makes the pose-independence check below
    // meaningful rather than vacuous.
    //
    // THE TOLERANCE IS THE SMOOTHING, AND IT IS COMPUTED, NOT FITTED. Numerical
    // integration of chi = 1/4 (1+tanh((R-r)/s))(1+tanh((h-|Y|)/s)) at R = 24,
    // h = 4.8, s = 1 gives the penalised disc +0.14 % in volume, +0.71 % in
    // I_yy and +1.22 % in I_xx over the sharp values. 3 % therefore leaves room
    // for the lattice on top of a known continuum excess, rather than being a
    // number chosen until the test passed.
    {
      Domain d(64, 64, 64, true, true, true);
      PenalisedBody<D3Q27, Disc> b(d);
      b.shape.cx = 32;  b.shape.cy = 32;  b.shape.cz = 32;
      b.shape.R = 24;   b.shape.hy = Real(4.8);
      b.shape.smooth = Real(1);
      b.shape.set_orientation(Quat{});
      b.set_uniform_density6(Real(1));

      const Real R = b.shape.R, h = b.shape.hy;
      const Real vol = Real(3.14159265358979323846) * R * R * Real(2) * h;
      check::near(b.penalised_volume() / vol, Real(1), Real(0.01),
                  "measured volume / pi R^2 (2h)");

      const Mat3 I0 = b.inertia_body;
      const Real m = b.mass;
      check::near(I0(1, 1) / (m * R * R / Real(2)), Real(1), Real(0.03),
                  "I about the symmetry axis / (m R^2 / 2)");
      const Real Id = m * (R * R / Real(4) + h * h / Real(3));
      check::near(I0(0, 0) / Id, Real(1), Real(0.03),
                  "I about a diameter / m (R^2/4 + h^2/3)");
      check::near(I0(0, 0) / I0(2, 2), Real(1), Real(2e-3),
                  "I_xx = I_zz -- the disc is transversely isotropic");
      // And the two are FAR apart, so a wrong rotation cannot hide.
      check::ok(I0(1, 1) > Real(1.7) * I0(0, 0),
                "I_yy is nearly twice I_xx, so the tensor is not isotropic");
      check::near(I0(0, 1) / I0(0, 0), Real(0), Real(2e-3),
                  "flat disc has no product of inertia");

      // The same pose-independence check block 7 makes for a slab, now on the
      // shape the skipping driver actually uses, and tilted by the attack angle
      // that driver defaults to.
      b.shape.set_orientation(Quat::from_axis_angle(
          Real(0), Real(0), Real(1), Real(0.3490658503988659)));   // 20 deg
      b.set_uniform_density6(Real(1));
      const Mat3 I1 = b.inertia_body;
      Real worst = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          worst = Kokkos::fmax(worst, Kokkos::fabs(I1(i, j) - I0(i, j)) / I0(1, 1));
      std::printf("   I_body at 0 deg vs 20 deg: worst entry differs by %.3f %%\n",
                  Real(100) * worst);
      check::ok(worst < Real(0.03),
                "body-frame inertia is pose independent for a Disc too");

      // The symmetry axis, which the driver reads to report an attack angle.
      // At 20 deg about z it must have tipped from +y toward -x by exactly that
      // much -- and the SIGN is the part worth pinning, because it decides
      // whether the driver raises the leading edge or buries it.
      Real ax, ay, az;
      b.shape.axis(ax, ay, az);
      check::near(ay, Real(0.9396926207859084), Real(1e-6),
                  "axis . yhat = cos 20 deg");
      check::near(ax, Real(-0.3420201433256687), Real(1e-6),
                  "axis . xhat = -sin 20 deg (leading edge at +x is RAISED)");
      check::near(az, Real(0), Real(1e-6), "axis stays out of z");
    }

    //-------------------------------------------------------------------------
    std::printf("\n9. THE GYROSCOPIC TERM: A TORQUE-FREE SYMMETRIC TOP\n\n");
    // The check that the omega x (I omega) term exists and has the right sign.
    //
    // A rigid body with NO torque on it conserves angular momentum exactly, and
    // a symmetric top spun about an axis TILTED off its symmetry axis therefore
    // precesses: the axis sweeps a cone about the fixed L, while |L| and the
    // component of omega along the symmetry axis both stay put. Those are the
    // two textbook invariants, and they are what this integrates for.
    //
    // WITHOUT the gyroscopic term the solve is I domega = T, so T = 0 gives
    // domega = 0 -- omega frozen in the WORLD frame while the body turns under
    // it. Then I = R I_b R^T changes and |I omega| drifts immediately. So this
    // test fails loudly on the old code and is not merely a tighter version of
    // an existing one.
    {
      Domain d(8, 8, 8, true, true, true);        // no fluid is touched
      PenalisedBody<D3Q27, Disc> b(d);
      // A disc's tensor by hand rather than measured, so the invariants have
      // exact targets: axial A about body y, diametral B about x and z.
      const Real A = Real(2), B = Real(1.25);
      b.mass = Real(1);
      b.inertia_body = Mat3{};
      b.inertia_body(0, 0) = B;  b.inertia_body(1, 1) = A;  b.inertia_body(2, 2) = B;
      b.shape.R = 1;  b.shape.hy = Real(0.1);
      b.shape.set_orientation(Quat{});
      b.free_translation = false;
      b.free_rotation = true;
      b.bx = b.by = b.bz = 0;                     // no gravity, no torque

      // TWO RATES, because the residual has to be IDENTIFIED and not merely
      // tolerated. With the gyroscopic term exactly right, |L| still drifts,
      // because the pose is integrated by explicit Euler -- and the RATE of
      // that drift is what says which it is. Euler is first order GLOBALLY, so
      // halving omega dt at fixed total rotation should halve the drift.
      //
      // Measured: 4.90e-3 at omega dt = 1e-2 and 2.38e-3 at 5e-3, a ratio of
      // 2.06. That is first order, and it identifies the drift as the
      // integrator. (I first wrote this expecting ~4, reasoning from the per
      // step error rather than the accumulated one; the ratio of 2 is the
      // correct signature and the prediction was the thing that was wrong.)
      // A missing term would not halve with the step at all.
      double drift[2] = {0, 0};
      double cone[2] = {0, 0};
      double axial[2] = {0, 0};
      for (int trial = 0; trial < 2; ++trial) {
        const Real W = (trial == 0) ? Real(0.01) : Real(0.005);
        const Real tilt = Real(0.4363323129985824);            // 25 degrees
        b.shape.set_orientation(Quat{});
        b.wx = W * Kokkos::sin(tilt);  b.wy = W * Kokkos::cos(tilt);  b.wz = 0;

        auto Lmag = [&]() {
          const Mat3 Iw = rotate_tensor(b.shape.Rm, b.inertia_body);
          const Real lx = Iw(0,0)*b.wx + Iw(0,1)*b.wy + Iw(0,2)*b.wz;
          const Real ly = Iw(1,0)*b.wx + Iw(1,1)*b.wy + Iw(1,2)*b.wz;
          const Real lz = Iw(2,0)*b.wx + Iw(2,1)*b.wy + Iw(2,2)*b.wz;
          return Kokkos::sqrt(lx*lx + ly*ly + lz*lz);
        };
        auto spin_axial = [&]() {
          Real ax, ay, az;  b.shape.axis(ax, ay, az);
          return b.wx*ax + b.wy*ay + b.wz*az;
        };

        const Real L0 = Lmag(), s0 = spin_axial();
        // A zero fluid: an empty BodySums6 is a body in vacuum, so the entire
        // right-hand side is the gyroscopic term and nothing else.
        const BodySums6 vac;
        Body6Properties p;
        Real cmax = 0;
        // The same TOTAL rotation in both trials -- twice the steps at half the
        // rate -- so the comparison is of integration error at fixed physics.
        const int nstep = (trial == 0) ? 4000 : 8000;
        for (int k = 0; k < nstep; ++k) {
          p.mass = b.mass;
          p.inertia_world = rotate_tensor(b.shape.Rm, b.inertia_body);
          p.wx = b.wx;  p.wy = b.wy;  p.wz = b.wz;
          p.bx = 0;  p.by = 0;  p.bz = 0;
          p.free_translation = false;  p.free_rotation = true;
          Real dU[3], dW[3];
          body6_solve(p, vac, dU, dW);
          b.wx += dW[0];  b.wy += dW[1];  b.wz += dW[2];
          b.advance6();
          Real ax, ay, az;  b.shape.axis(ax, ay, az);
          cmax = Kokkos::fmax(cmax, Kokkos::fabs(ay - Real(1)));
        }
        drift[trial] = Kokkos::fabs(double(Lmag() / L0) - 1.0);
        axial[trial] = Kokkos::fabs(double(spin_axial() / s0) - 1.0);
        cone[trial] = Kokkos::acos(1.0 - double(cmax)) * 57.29577951308232;
      }
      std::printf("   omega dt = 1.0e-2: |L| drift %.3e, cone %.2f deg\n",
                  drift[0], cone[0]);
      std::printf("   omega dt = 5.0e-3: |L| drift %.3e, cone %.2f deg\n",
                  drift[1], cone[1]);
      std::printf("   drift ratio %.2f (first-order Euler predicts 2)\n",
                  drift[1] > 0 ? drift[0] / drift[1] : 0.0);
      // THE AXIAL SPIN IS EXACT, not merely small: omega . axis is a strict
      // invariant of a symmetric top and nothing in the integration threatens
      // it, so this one is held to round-off rather than to a tolerance.
      check::near(axial[0], 0.0, 1e-12, "axial spin exactly conserved (omega dt 1e-2)");
      check::near(axial[1], 0.0, 1e-12, "axial spin exactly conserved (omega dt 5e-3)");
      check::near(drift[0] / drift[1], 2.0, 0.4,
                  "|L| residual halves with omega dt -- first-order Euler, "
                  "not a missing term");
      check::ok(drift[1] < 5e-3, "and it is small in absolute terms");
      // And the axis DID move -- otherwise the invariants are conserved
      // trivially and the whole block proves nothing.
      check::ok(cone[0] > 5.0,
                "the symmetry axis actually precessed through a real cone");
    }
  }
  const int rc = check::report("rigid3d");
  Kokkos::finalize();
  return rc;
}
