#pragma once
//==============================================================================
//  The six-degree-of-freedom rigid body: quaternion pose, inertia tensor, 6x6.
//
//  A PORT of ../src/solver/RigidBody3D.hpp, and deliberately an independent
//  re-implementation rather than a shared header -- this tree keeps two codes so
//  that agreement between them is evidence. The ALGEBRA is identical on purpose,
//  down to the pivoting order in solve6, so a planar case can be run through
//  both and compared digit for digit. Where they differ is where they must:
//  the accumulators here are double even in an FP32 build, for the reason
//  BodySums gives, and the math calls are std:: rather than Kokkos::.
//
//  WHY THIS EXISTS WHEN body.cuh ALREADY HAS A BODY. Because that one rotates
//  about z and nothing else. It carries `theta` and `omega`, both scalars, and
//  a Sphere with rotation switched off -- which is exact for a sphere, whose
//  chi is rotation invariant, and useless for anything with an orientation.
//  A skipping stone is the case that needs the rest: what holds its angle of
//  attack through the impact is gyroscopic stiffness from spin about its own
//  axis, and that cannot be expressed with one angular coordinate.
//
//  WHAT THE SYSTEM IS. Writing dU and domega for the change in the body's
//  velocity and angular velocity over one step, A = m_b + m_f for the combined
//  mass, B = I_b + I_f for the combined inertia tensor, and S for the first
//  moment of the fictitious fluid:
//
//      [ A I3     -[S]x ] [ dU     ]   [ 2 dP + (m_b - m_f) g ]
//      [ [S]x       B   ] [ domega ] = [ 2 dL - S x g         ]
//
//  with I_f = tr(J) I3 - J and J_ij = integral chi rho r_i r_j. It reduces to
//  the validated 3x3 on a planar problem -- ../tests/test_rigid3d.cpp block 3
//  pins that at 1.7e-18, and test/host_body.cpp repeats it here, because
//  agreeing with the parent is the only check this port has that is not just
//  internal consistency.
//==============================================================================
#include "core.cuh"

#include <cmath>
#include <cstddef>

namespace lbm {

//------------------------------------------------------------------------------
// A 3x3, row major. Plain aggregate so it captures into a kernel by value.
//
// tmul is LBM_HD and the rest is not, and that split is the whole design: the
// only thing a KERNEL does with a rotation is take a world point into the body
// frame, which is R^T v, and that runs at every node every step. Assembling and
// rotating an inertia tensor happens once per step on the host.
//------------------------------------------------------------------------------
struct Mat3 {
  Real a[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  LBM_HD LBM_INLINE Real operator()(int i, int j) const { return a[3 * i + j]; }
  LBM_HD LBM_INLINE Real& operator()(int i, int j) { return a[3 * i + j]; }

  // R^T v -- world into body.
  LBM_HD LBM_INLINE void tmul(Real vx, Real vy, Real vz,
                              Real& ox, Real& oy, Real& oz) const {
    ox = a[0] * vx + a[3] * vy + a[6] * vz;
    oy = a[1] * vx + a[4] * vy + a[7] * vz;
    oz = a[2] * vx + a[5] * vy + a[8] * vz;
  }
  // R v -- body into world.
  LBM_HD LBM_INLINE void mul(Real vx, Real vy, Real vz,
                             Real& ox, Real& oy, Real& oz) const {
    ox = a[0] * vx + a[1] * vy + a[2] * vz;
    oy = a[3] * vx + a[4] * vy + a[5] * vz;
    oz = a[6] * vx + a[7] * vy + a[8] * vz;
  }
  LBM_HD LBM_INLINE Mat3 transposed() const {
    Mat3 t;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) t.a[3 * i + j] = a[3 * j + i];
    return t;
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
      for (int k = 0; k < 3; ++k) s += RM(i, k) * R(j, k);
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
    const Real n = Real(std::sqrt(double(w) * w + double(x) * x
                                + double(y) * y + double(z) * z));
    if (n > Real(0)) { w /= n; x /= n; y /= n; z /= n; }
  }

  static Quat from_axis_angle(Real ax, Real ay, Real az, Real ang) {
    const double n = std::sqrt(double(ax) * ax + double(ay) * ay + double(az) * az);
    if (!(n > 0.0)) return Quat{};
    const double s = std::sin(0.5 * double(ang)) / n;
    return Quat{Real(std::cos(0.5 * double(ang))),
                Real(ax * s), Real(ay * s), Real(az * s)};
  }

  // q2 * q1 -- q1 applied first, then q2. Present because composing a
  // reference pose with a perturbation is what every driver does, and writing
  // the eight products out at the call site is where a sign gets lost.
  static Quat mul(const Quat& q2, const Quat& q1) {
    return Quat{q2.w * q1.w - q2.x * q1.x - q2.y * q1.y - q2.z * q1.z,
                q2.w * q1.x + q2.x * q1.w + q2.y * q1.z - q2.z * q1.y,
                q2.w * q1.y - q2.x * q1.z + q2.y * q1.w + q2.z * q1.x,
                q2.w * q1.z + q2.x * q1.y - q2.y * q1.x + q2.z * q1.w};
  }

  // ONE explicit Euler step of qdot = 1/2 omega_quat * q, then renormalise.
  // Explicit Euler is not chosen for accuracy: dt = 1 is the fluid step, there
  // is nothing faster in the body to resolve, and the pose error is second
  // order in omega dt while omega dt here is a small fraction of a radian. The
  // renormalisation is not optional -- the drift off the unit sphere becomes a
  // SCALING of the body, which reads as the shape breathing rather than as an
  // integration error.
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
// The SIXTEEN integrals of one sweep. BodySums' nine, with the first moment and
// the momentum deficit completed to vectors, the single inertia scalar replaced
// by the six independent products J_ij, and the scalar angular deficit
// completed to a vector.
//
// J rather than I_f directly: I_f = tr(J) I3 - J, and carrying the products
// means the reduction adds six numbers instead of six numbers plus a trace it
// would have to keep consistent with them.
//
// DOUBLE, in an FP32 build too, for BodySums' reason: these are sums over
// millions of nodes and the whole body force is the small difference between
// two large ones.
//------------------------------------------------------------------------------
struct BodySums6 {
  double m = 0;
  double Sx = 0, Sy = 0, Sz = 0;
  double Jxx = 0, Jyy = 0, Jzz = 0, Jxy = 0, Jxz = 0, Jyz = 0;
  double Px = 0, Py = 0, Pz = 0;
  double Lx = 0, Ly = 0, Lz = 0;

  // I_f = tr(J) I3 - J, the fictitious fluid's inertia tensor about the shape
  // origin. Its zz entry is Jxx + Jyy, which is the old scalar Iz exactly.
  Mat3 fluid_inertia() const {
    const double tr = Jxx + Jyy + Jzz;
    Mat3 I;
    I(0, 0) = Real(tr - Jxx);  I(0, 1) = Real(-Jxy);      I(0, 2) = Real(-Jxz);
    I(1, 0) = Real(-Jxy);      I(1, 1) = Real(tr - Jyy);  I(1, 2) = Real(-Jyz);
    I(2, 0) = Real(-Jxz);      I(2, 1) = Real(-Jyz);      I(2, 2) = Real(tr - Jzz);
    return I;
  }
};

// The number of doubles the reduction moves. Named rather than spelled 16 in
// four places, because a mismatch between the shared-memory size, the partial
// buffer and the unpack loop is a silent wrong answer.
static constexpr int BODY6_SUMS = 16;

//------------------------------------------------------------------------------
// The 6x6, by Gaussian elimination with partial pivoting.
//
// NOT a closed form and not Cholesky. The 3x3 has a closed form because a 2x2
// Schur complement is one division; the 6x6's would be pages of algebra for a
// solve that happens ONCE PER STEP on the host, against a sweep over millions
// of nodes. Partial pivoting rather than Cholesky because the
// positive-definiteness argument -- Cauchy-Schwarz on the measure chi rho --
// carries over in principle but the matrix is assembled from measured integrals
// in finite precision, and a solver that cannot fail is worth more than the
// factor of two.
//
// In DOUBLE regardless of Real, matching the accumulators it consumes: this is
// one 6x6 per step, so the precision is free, and an FP32 elimination on
// entries spanning m_b ~ 1e6 against S ~ 1e5 is exactly where a pivot goes bad.
//------------------------------------------------------------------------------
inline bool solve6(const double A6[6][6], const double r6[6], double out[6]) {
  double M[6][7];
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) M[i][j] = A6[i][j];
    M[i][6] = r6[i];
  }
  for (int c = 0; c < 6; ++c) {
    int piv = c;
    double best = std::fabs(M[c][c]);
    for (int i = c + 1; i < 6; ++i) {
      const double v = std::fabs(M[i][c]);
      if (v > best) { best = v;  piv = i; }
    }
    if (!(best > 0.0)) { for (int i = 0; i < 6; ++i) out[i] = 0; return false; }
    if (piv != c)
      for (int j = c; j < 7; ++j) { const double t = M[c][j]; M[c][j] = M[piv][j]; M[piv][j] = t; }
    const double inv = 1.0 / M[c][c];
    for (int i = c + 1; i < 6; ++i) {
      const double f = M[i][c] * inv;
      if (f == 0.0) continue;
      for (int j = c; j < 7; ++j) M[i][j] -= f * M[c][j];
    }
  }
  for (int i = 5; i >= 0; --i) {
    double s = M[i][6];
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
  // THE CURRENT ANGULAR VELOCITY. Not bookkeeping -- it is what makes this
  // Euler's equation rather than a linear one. See the gyroscopic term below.
  Real wx = 0, wy = 0, wz = 0;
  Real bx = 0, by = 0, bz = 0; // body force per unit mass -- the SAME vector the
                               // collision operator is given
  bool free_translation = true;
  bool free_rotation = true;
};

inline void body6_solve(const Body6Properties& p, const BodySums6& q,
                        double dU[3], double dW[3]) {
  for (int i = 0; i < 3; ++i) { dU[i] = 0; dW[i] = 0; }
  const double A = double(p.mass) + q.m;
  if (!(A > 0.0)) return;

  const Mat3 If = q.fluid_inertia();
  const double S[3] = {q.Sx, q.Sy, q.Sz};
  const double g[3] = {double(p.bx), double(p.by), double(p.bz)};

  // R = 2 dP + (m_b - m_f) g,  T = 2 dL - S x g.
  const double R[3] = {2.0 * q.Px + (double(p.mass) - q.m) * g[0],
                       2.0 * q.Py + (double(p.mass) - q.m) * g[1],
                       2.0 * q.Pz + (double(p.mass) - q.m) * g[2]};
  const double SxG[3] = {S[1] * g[2] - S[2] * g[1],
                         S[2] * g[0] - S[0] * g[2],
                         S[0] * g[1] - S[1] * g[0]};
  // THE GYROSCOPIC TERM, -omega x (I_b omega), and leaving it out is not a
  // small error for a spinning body -- it is the whole of why a spinning body
  // behaves differently from a still one.
  //
  // What has to be integrated is d(I omega)/dt = T with I changing as the body
  // turns, and expanding that gives I domega/dt = T - omega x (I omega). Drop
  // the second piece and the solve is I domega = T: a body whose angular
  // momentum may change direction for free, so a torque TIPS the axis instead
  // of precessing it, and a top falls over.
  //
  // WHY THIS SURVIVED THE CUBE. For an ISOTROPIC inertia, I = lambda I3, the
  // term is omega x (lambda omega) = 0 identically -- so demonstrator/cube_entry
  // and ../tests/test_rigid3d block 7 are bit-for-bit unaffected by adding it, and
  // neither could ever have found it missing. It took a DISC, whose axial and
  // diametral moments differ by a factor of two, to expose it: released
  // spinning at a 20 degree attack angle the disc lost 4 degrees in a quarter
  // of a diameter, which is a stone that digs in rather than skips.
  //
  // I_b ALONE, not B = I_b + I_f. The fluid's contribution to the system matrix
  // is a penalisation artefact standing in for added mass; it is not a rigid
  // body with angular momentum of its own, so it has no gyroscopic term.
  const double w[3] = {double(p.wx), double(p.wy), double(p.wz)};
  double Iw[3];
  for (int i = 0; i < 3; ++i)
    Iw[i] = double(p.inertia_world(i, 0)) * w[0]
          + double(p.inertia_world(i, 1)) * w[1]
          + double(p.inertia_world(i, 2)) * w[2];
  const double gyro[3] = {w[1] * Iw[2] - w[2] * Iw[1],
                          w[2] * Iw[0] - w[0] * Iw[2],
                          w[0] * Iw[1] - w[1] * Iw[0]};

  const double T[3] = {2.0 * q.Lx - SxG[0] - gyro[0],
                       2.0 * q.Ly - SxG[1] - gyro[1],
                       2.0 * q.Lz - SxG[2] - gyro[2]};

  double A6[6][6] = {};
  double r6[6];
  for (int i = 0; i < 3; ++i) {
    A6[i][i] = A;
    r6[i] = R[i];
    r6[3 + i] = T[i];
    for (int j = 0; j < 3; ++j)
      A6[3 + i][3 + j] = double(p.inertia_world(i, j)) + double(If(i, j));
  }
  // The off-diagonal blocks: -[S]x above, +[S]x below. [S]x a = S x a, so
  // ([S]x)_{ij} = -eps_{ijk} S_k, i.e. row 0 = (0, -Sz, Sy) and so on.
  const double Sk[3][3] = {{    0.0, -S[2],  S[1]},
                           {  S[2],    0.0, -S[0]},
                           { -S[1],  S[0],    0.0}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      A6[i][3 + j] = -Sk[i][j];
      A6[3 + i][j] =  Sk[i][j];
    }

  // Held degrees of freedom, by elimination rather than by discarding rows.
  if (!p.free_translation && !p.free_rotation) return;
  if (!p.free_translation) {
    double A3[6][6] = {}, r3[6] = {}, o[6] = {};
    for (int i = 0; i < 3; ++i) {
      r3[i] = T[i];
      for (int j = 0; j < 3; ++j) A3[i][j] = A6[3 + i][3 + j];
      A3[3 + i][3 + i] = 1.0;                 // identity padding, unused half
    }
    if (solve6(A3, r3, o)) for (int i = 0; i < 3; ++i) dW[i] = o[i];
    return;
  }
  if (!p.free_rotation) {
    for (int i = 0; i < 3; ++i) dU[i] = R[i] / A;
    return;
  }

  double o[6];
  if (!solve6(A6, r6, o)) return;
  for (int i = 0; i < 3; ++i) { dU[i] = o[i];  dW[i] = o[3 + i]; }
}

}  // namespace lbm
