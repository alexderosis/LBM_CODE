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
  }
  const int rc = check::report("rigid3d");
  Kokkos::finalize();
  return rc;
}
