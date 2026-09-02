//==============================================================================
//  THE SIX-DEGREE-OF-FREEDOM RIGID BODY: quaternion, inertia tensor, 6x6 solve.
//
//  PenalisedBody.hpp carries one angle about z. That is exact for a 2-D body and
//  for a sphere on its axis, and wrong for everything else -- a cube released
//  corner-down strikes on a vertex, and the reaction on that vertex is not
//  through the centre in any single plane, so the response is a rotation about
//  an axis that is not z and that MOVES as the body turns. This header is what
//  that needs, and it is deliberately separate: the 2-D path must not pay for it
//  and must not change because of it.
//
//  THE SYSTEM, and it is the 2-D one generalised rather than a new derivation.
//  The rigid velocity field is v(r) = U + omega x r. Writing the momentum and
//  angular-momentum balances of the penalised region against it gives
//
//      [ A I3     -[S]x ] [dU  ]   [ 2 dP + (m_b - m_f) g ]
//      [ [S]x      B    ] [domega] = [ 2 dL - S x g         ]
//
//  with  A = m_b + m_f  (a scalar, times the identity),
//        S = integral chi rho r                    -- the first moment,
//        B = I_b + I_f,  I_f = integral chi rho (|r|^2 I3 - r r^T),
//        [S]x the skew matrix for which [S]x a = S x a.
//
//  IT IS SYMMETRIC, because [S]x^T = -[S]x makes the off-diagonal blocks each
//  other's transpose. And it REDUCES to the existing 3x3: put omega = (0,0,w)
//  and r in a plane, and the linear rows become A dux - S_y dw = R_x and
//  A duy + S_x dw = R_y, the angular row becomes -S_y dux + S_x duy + B_zz dw
//  = T_z, and B_zz = integral chi rho (rx^2 + ry^2), which is exactly the old
//  scalar Iz. That correspondence is checked in tests/test_rigid3d.cpp rather
//  than asserted here.
//
//  WHY A QUATERNION AND A MATRIX BOTH. The quaternion is the state, because
//  integrating a rotation matrix directly loses orthonormality and there is no
//  cheap way to put it back; the matrix is a cached derivative of it, because
//  chi is evaluated at EVERY node of the domain EVERY step and rebuilding a
//  rotation from four numbers per node would be the most expensive thing in the
//  sweep. That is the same argument Rect makes for caching cos and sin, one
//  dimension up.
//
//  WHAT THIS DOES NOT DO. No collision model, so a body meeting a wall still
//  interpenetrates -- and in 3-D that now produces a spurious TORQUE as well as
//  a spurious force, which is worse rather than merely different. The 2-D
//  squares resting on the tank floor already show the 1-DOF version of it.
//==============================================================================
#pragma once

#include "core/Types.hpp"

#include <Kokkos_Core.hpp>

namespace lbm {

//------------------------------------------------------------------------------
// A 3x3, row major. Plain aggregate so it captures into a device lambda.
//------------------------------------------------------------------------------
struct Mat3 {
  Real a[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  KOKKOS_INLINE_FUNCTION Real operator()(int i, int j) const { return a[3 * i + j]; }
  KOKKOS_INLINE_FUNCTION Real& operator()(int i, int j) { return a[3 * i + j]; }

  // R^T v, which is the direction the indicator needs: taking a world point
  // into the body frame is the INVERSE rotation, and for an orthonormal R the
  // inverse is the transpose.
  KOKKOS_INLINE_FUNCTION void tmul(Real vx, Real vy, Real vz,
                                   Real& ox, Real& oy, Real& oz) const {
    ox = a[0] * vx + a[3] * vy + a[6] * vz;
    oy = a[1] * vx + a[4] * vy + a[7] * vz;
    oz = a[2] * vx + a[5] * vy + a[8] * vz;
  }

  KOKKOS_INLINE_FUNCTION void mul(Real vx, Real vy, Real vz,
                                  Real& ox, Real& oy, Real& oz) const {
    ox = a[0] * vx + a[1] * vy + a[2] * vz;
    oy = a[3] * vx + a[4] * vy + a[5] * vz;
    oz = a[6] * vx + a[7] * vy + a[8] * vz;
  }
};

// R M R^T -- how a body-frame inertia tensor is expressed in the world frame.
inline Mat3 rotate_tensor(const Mat3& R, const Mat3& M) {
  Mat3 RM, out;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Real s = 0;
      for (int k = 0; k < 3; ++k) s += R(i, k) * M(k, j);
      RM(i, j) = s;
    }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Real s = 0;
      for (int k = 0; k < 3; ++k) s += RM(i, k) * R(j, k);   // times R^T
      out(i, j) = s;
    }
  return out;
}

//------------------------------------------------------------------------------
// A unit quaternion, w + xi + yj + zk, as the orientation STATE.
//------------------------------------------------------------------------------
struct Quat {
  Real w = 1, x = 0, y = 0, z = 0;

  void normalise() {
    const Real n = Kokkos::sqrt(w * w + x * x + y * y + z * z);
    if (n > Real(0)) { w /= n; x /= n; y /= n; z /= n; }
  }

  static Quat from_axis_angle(Real ax, Real ay, Real az, Real ang) {
    const Real n = Kokkos::sqrt(ax * ax + ay * ay + az * az);
    if (!(n > Real(0))) return Quat{};
    const Real s = Kokkos::sin(Real(0.5) * ang) / n;
    return Quat{Kokkos::cos(Real(0.5) * ang), ax * s, ay * s, az * s};
  }

  // ONE explicit Euler step of qdot = 1/2 omega_quat * q, then renormalise.
  // Explicit Euler is not chosen for accuracy: dt = 1 is the fluid step, there
  // is nothing faster in the body to resolve, and the pose error it leaves is
  // second order in omega dt while omega dt here is a small fraction of a
  // radian. The renormalisation is not optional -- without it the drift off the
  // unit sphere becomes a scaling of the body, which reads as the shape
  // breathing rather than as an integration error.
  void integrate(Real ox, Real oy, Real oz, Real dt) {
    const Real dw = Real(-0.5) * (ox * x + oy * y + oz * z);
    const Real dx = Real( 0.5) * (ox * w + oy * z - oz * y);
    const Real dy = Real( 0.5) * (oy * w + oz * x - ox * z);
    const Real dz = Real( 0.5) * (oz * w + ox * y - oy * x);
    w += dt * dw;  x += dt * dx;  y += dt * dy;  z += dt * dz;
    normalise();
  }

  Mat3 matrix() const {
    Mat3 R;
    const Real xx = x * x, yy = y * y, zz = z * z;
    const Real xy = x * y, xz = x * z, yz = y * z;
    const Real wx = w * x, wy = w * y, wz = w * z;
    R(0, 0) = Real(1) - Real(2) * (yy + zz);
    R(0, 1) = Real(2) * (xy - wz);
    R(0, 2) = Real(2) * (xz + wy);
    R(1, 0) = Real(2) * (xy + wz);
    R(1, 1) = Real(1) - Real(2) * (xx + zz);
    R(1, 2) = Real(2) * (yz - wx);
    R(2, 0) = Real(2) * (xz - wy);
    R(2, 1) = Real(2) * (yz + wx);
    R(2, 2) = Real(1) - Real(2) * (xx + yy);
    return R;
  }
};

//------------------------------------------------------------------------------
// The SIXTEEN integrals of one sweep. The 2-D struct's nine, with the first
// moment and the momentum deficit completed to vectors, the single inertia
// scalar replaced by the six independent products J_ij = integral chi rho ri rj,
// and the scalar angular deficit completed to a vector.
//
// J rather than I_f directly: I_f = tr(J) I3 - J, and carrying the products
// means the reduction adds six numbers instead of six numbers plus a trace it
// would have to keep consistent with them.
//------------------------------------------------------------------------------
struct BodySums6 {
  Real m;
  Real Sx, Sy, Sz;
  Real Jxx, Jyy, Jzz, Jxy, Jxz, Jyz;
  Real Px, Py, Pz;
  Real Lx, Ly, Lz;

  KOKKOS_INLINE_FUNCTION BodySums6()
      : m(0), Sx(0), Sy(0), Sz(0),
        Jxx(0), Jyy(0), Jzz(0), Jxy(0), Jxz(0), Jyz(0),
        Px(0), Py(0), Pz(0), Lx(0), Ly(0), Lz(0) {}

  KOKKOS_INLINE_FUNCTION void operator+=(const BodySums6& o) {
    m += o.m;
    Sx += o.Sx;  Sy += o.Sy;  Sz += o.Sz;
    Jxx += o.Jxx;  Jyy += o.Jyy;  Jzz += o.Jzz;
    Jxy += o.Jxy;  Jxz += o.Jxz;  Jyz += o.Jyz;
    Px += o.Px;  Py += o.Py;  Pz += o.Pz;
    Lx += o.Lx;  Ly += o.Ly;  Lz += o.Lz;
  }

  // I_f = tr(J) I3 - J, the fictitious fluid's inertia tensor about the shape
  // origin. Its zz entry is Jxx + Jyy, which is the old scalar Iz exactly.
  Mat3 fluid_inertia() const {
    const Real tr = Jxx + Jyy + Jzz;
    Mat3 I;
    I(0, 0) = tr - Jxx;  I(0, 1) = -Jxy;      I(0, 2) = -Jxz;
    I(1, 0) = -Jxy;      I(1, 1) = tr - Jyy;  I(1, 2) = -Jyz;
    I(2, 0) = -Jxz;      I(2, 1) = -Jyz;      I(2, 2) = tr - Jzz;
    return I;
  }
};

}  // namespace lbm

namespace Kokkos {
template <>
struct reduction_identity<lbm::BodySums6> {
  KOKKOS_FORCEINLINE_FUNCTION static lbm::BodySums6 sum() {
    return lbm::BodySums6();
  }
};
}  // namespace Kokkos

namespace lbm {

//------------------------------------------------------------------------------
// The 6x6, by Gaussian elimination with partial pivoting.
//
// NOT a closed form, and not Cholesky. The 3x3 version has a closed form
// because a 2x2 Schur complement is one division; the 6x6's would be pages of
// algebra for a solve that happens ONCE PER STEP on the host, against a sweep
// over millions of nodes. Partial pivoting rather than Cholesky because the
// positive-definiteness argument the 3x3 banner makes -- Cauchy-Schwarz on the
// measure chi rho -- carries over in principle but the matrix is assembled from
// measured integrals in finite precision, and a solver that cannot fail is
// worth more here than the factor of two.
//
// Locking a set of degrees of freedom is done by ELIMINATING those rows and
// columns, not by discarding them after the fact: with dU pinned to zero the
// rotation equations are B domega = T on their own, which is a different and
// correct answer, not the 6x6 with three rows thrown away.
//------------------------------------------------------------------------------
inline bool solve6(const Real A6[6][6], const Real r6[6], Real out[6]) {
  Real M[6][7];
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) M[i][j] = A6[i][j];
    M[i][6] = r6[i];
  }
  for (int c = 0; c < 6; ++c) {
    int piv = c;
    Real best = Kokkos::fabs(M[c][c]);
    for (int i = c + 1; i < 6; ++i) {
      const Real v = Kokkos::fabs(M[i][c]);
      if (v > best) { best = v;  piv = i; }
    }
    if (!(best > Real(0))) { for (int i = 0; i < 6; ++i) out[i] = 0; return false; }
    if (piv != c) for (int j = c; j < 7; ++j) { const Real t = M[c][j]; M[c][j] = M[piv][j]; M[piv][j] = t; }
    const Real inv = Real(1) / M[c][c];
    for (int i = c + 1; i < 6; ++i) {
      const Real f = M[i][c] * inv;
      if (f == Real(0)) continue;
      for (int j = c; j < 7; ++j) M[i][j] -= f * M[c][j];
    }
  }
  for (int i = 5; i >= 0; --i) {
    Real s = M[i][6];
    for (int j = i + 1; j < 6; ++j) s -= M[i][j] * out[j];
    out[i] = s / M[i][i];
  }
  return true;
}

//------------------------------------------------------------------------------
// Assemble and solve. props carries the body's own mass and its inertia tensor
// ALREADY IN THE WORLD FRAME -- the caller rotates it, because only the caller
// knows the pose.
//------------------------------------------------------------------------------
struct Body6Properties {
  Real mass = 0;
  Mat3 inertia_world;          // I_b, world frame; caller does R I_body R^T
  Real bx = 0, by = 0, bz = 0; // body force per unit mass, the SAME vector the
                               // collision operator is given
  bool free_translation = true;
  bool free_rotation = true;
};

inline void body6_solve(const Body6Properties& p, const BodySums6& q,
                        Real dU[3], Real dW[3]) {
  for (int i = 0; i < 3; ++i) { dU[i] = 0; dW[i] = 0; }
  const Real A = p.mass + q.m;
  if (!(A > Real(0))) return;

  const Mat3 If = q.fluid_inertia();
  const Real S[3] = {q.Sx, q.Sy, q.Sz};
  const Real g[3] = {p.bx, p.by, p.bz};

  // R = 2 dP + (m_b - m_f) g,  T = 2 dL - S x g.
  const Real R[3] = {Real(2) * q.Px + (p.mass - q.m) * g[0],
                     Real(2) * q.Py + (p.mass - q.m) * g[1],
                     Real(2) * q.Pz + (p.mass - q.m) * g[2]};
  const Real SxG[3] = {S[1] * g[2] - S[2] * g[1],
                       S[2] * g[0] - S[0] * g[2],
                       S[0] * g[1] - S[1] * g[0]};
  const Real T[3] = {Real(2) * q.Lx - SxG[0],
                     Real(2) * q.Ly - SxG[1],
                     Real(2) * q.Lz - SxG[2]};

  Real A6[6][6] = {};
  Real r6[6];
  for (int i = 0; i < 3; ++i) {
    A6[i][i] = A;
    r6[i] = R[i];
    r6[3 + i] = T[i];
    for (int j = 0; j < 3; ++j) A6[3 + i][3 + j] = p.inertia_world(i, j) + If(i, j);
  }
  // The off-diagonal blocks: -[S]x above, +[S]x below. [S]x a = S x a, so
  // ([S]x)_{ij} = -eps_{ijk} S_k, i.e. row 0 = (0, -Sz, Sy) and so on.
  const Real Sk[3][3] = {{     0, -S[2],  S[1]},
                         { S[2],      0, -S[0]},
                         {-S[1],  S[0],      0}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      A6[i][3 + j] = -Sk[i][j];
      A6[3 + i][j] =  Sk[i][j];
    }

  // Held degrees of freedom, by elimination rather than by discarding rows.
  if (!p.free_translation && !p.free_rotation) return;
  if (!p.free_translation) {
    Real B[3][3], t[3];
    for (int i = 0; i < 3; ++i) {
      t[i] = T[i];
      for (int j = 0; j < 3; ++j) B[i][j] = A6[3 + i][3 + j];
    }
    Real A3[6][6] = {}, r3[6] = {}, o[6] = {};
    for (int i = 0; i < 3; ++i) {
      r3[i] = t[i];
      A3[i][i] = Real(1);                 // identity padding on the unused half
      for (int j = 0; j < 3; ++j) A3[i][j] = B[i][j];
      A3[3 + i][3 + i] = Real(1);
    }
    if (solve6(A3, r3, o)) for (int i = 0; i < 3; ++i) dW[i] = o[i];
    return;
  }
  if (!p.free_rotation) {
    for (int i = 0; i < 3; ++i) dU[i] = R[i] / A;
    return;
  }

  Real o[6];
  if (!solve6(A6, r6, o)) return;
  for (int i = 0; i < 3; ++i) { dU[i] = o[i];  dW[i] = o[3 + i]; }
}

}  // namespace lbm
