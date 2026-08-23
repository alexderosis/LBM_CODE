#pragma once
//==============================================================================
//  Native CUDA lattice Boltzmann -- core.
//
//  A second implementation, deliberately independent of the Kokkos code in the
//  parent directory. Nothing here includes anything from ../src.
//
//  WHY EVERY FUNCTION IS `LBM_HD` AND NOT `__device__`.
//  ---------------------------------------------------
//  `LBM_HD` expands to `__host__ __device__` under nvcc and to nothing under a
//  plain C++ compiler. That is not portability for its own sake: it means the
//  whole numerical core -- equilibrium, the central-moment transform, the
//  collision -- can be compiled and unit-tested on a machine with no GPU
//  (test/host_check.cpp does exactly that). The CUDA kernels are then thin
//  wrappers around code that has already been checked.
//
//  The alternative -- writing several hundred lines of `__device__` code that
//  can only first be exercised on a remote GPU -- is how this kind of port goes
//  wrong. Verify the arithmetic where it is cheap to verify.
//
//  LAYOUT. Populations are stored SoA: f[i * n_nodes + node]. Consecutive
//  threads handle consecutive nodes, so each warp reads 32 contiguous floats
//  per direction. This is the single most important decision in a GPU LBM code
//  and it is why the array is indexed this way round rather than AoS.
//==============================================================================
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
  #define LBM_HD __host__ __device__
  #define LBM_INLINE __forceinline__
#else
  #define LBM_HD
  #define LBM_INLINE inline
#endif

namespace lbm {

#if defined(LBM_DOUBLE)
using Real = double;
#else
using Real = float;
#endif

//==============================================================================
//  D3Q27.
//
//  Direction ordering obeys opp(i) == i + 1 for odd i. Esoteric Pull depends on
//  that contract: the pair (i, i+1) shares one memory slot, so if the ordering
//  is broken the streaming step silently transports populations in the wrong
//  direction and the flow looks plausible while being wrong.
//==============================================================================
struct D3Q27 {
  static constexpr int D = 3;
  static constexpr int Q = 27;

  //                                0   1   2   3   4   5   6   7   8   9  10
  //                               11  12  13  14  15  16  17  18  19  20  21
  //                               22  23  24  25  26
  static LBM_HD LBM_INLINE int cx(int i) {
    constexpr int v[27] = { 0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,
                            1, -1,  1, -1,  0,  0,  0,  0,  1, -1,  1,
                           -1,  1, -1, -1,  1 };
    return v[i];
  }
  static LBM_HD LBM_INLINE int cy(int i) {
    constexpr int v[27] = { 0,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,
                            0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1,
                           -1, -1,  1,  1, -1 };
    return v[i];
  }
  static LBM_HD LBM_INLINE int cz(int i) {
    constexpr int v[27] = { 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,
                            1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,
                            1,  1, -1,  1, -1 };
    return v[i];
  }
  // Weights as exact rationals, so the sum is 1 to the last bit.
  static LBM_HD LBM_INLINE Real w(int i) {
    constexpr int num[27] = {64, 16, 16, 16, 16, 16, 16, 4, 4, 4, 4,
                              4,  4,  4,  4,  4,  4,  4,  4, 1, 1, 1,
                              1,  1,  1,  1,  1};
    return Real(num[i]) / Real(216);
  }
  static LBM_HD LBM_INLINE Real cs2() { return Real(1) / Real(3); }
  static LBM_HD LBM_INLINE Real inv_cs2() { return Real(3); }
};

//------------------------------------------------------------------------------
// Population index for velocity (a-1, b-1, c-1), a,b,c in {0,1,2}.
//
// Precomputed, not searched. A linear search over the velocity set folds away
// only when its arguments are compile-time constants; here they are loop
// variables, and searching cost 2*Q*Q comparisons per node in the parent
// implementation and made the operator 7.7x slower than BGK. Same trap applies
// here, so the table is built once at compile time.
//------------------------------------------------------------------------------
struct DirTable { int v[27]; };

constexpr int dir_lookup(int ex, int ey, int ez) {
  constexpr int X[27] = { 0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,
                          1, -1,  1, -1,  0,  0,  0,  0,  1, -1,  1,
                         -1,  1, -1, -1,  1 };
  constexpr int Y[27] = { 0,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,
                          0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1,
                         -1, -1,  1,  1, -1 };
  constexpr int Z[27] = { 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,
                          1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,
                          1,  1, -1,  1, -1 };
  for (int i = 0; i < 27; ++i)
    if (X[i] == ex && Y[i] == ey && Z[i] == ez) return i;
  return -1;
}

constexpr DirTable make_dir_table() {
  DirTable t{};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c)
        t.v[(a * 3 + b) * 3 + c] = dir_lookup(a - 1, b - 1, c - 1);
  return t;
}

LBM_HD LBM_INLINE int pi(int a, int b, int c) {
  constexpr DirTable t = make_dir_table();
  return t.v[(a * 3 + b) * 3 + c];
}
// Moment slot for exponents (p, q, r), each in {0,1,2}.
LBM_HD LBM_INLINE constexpr int mi(int p, int q, int r) { return (p * 3 + q) * 3 + r; }
LBM_HD LBM_INLINE constexpr int p_of(int n) { return n / 9; }
LBM_HD LBM_INLINE constexpr int q_of(int n) { return (n / 3) % 3; }
LBM_HD LBM_INLINE constexpr int r_of(int n) { return n % 3; }
LBM_HD LBM_INLINE constexpr int order_of(int n) { return p_of(n) + q_of(n) + r_of(n); }

//==============================================================================
//  Separable central-moment transform.
//
//  The 27-moment transform is done as three 1D passes of three points each,
//  never as a 27x27 matrix. Each pass turns the triple at c = -1, 0, +1 into
//  (m0, m1, m2) about the shift velocity u. Cost is O(3 * 9) per node instead
//  of O(27^2), and the whole thing stays in registers.
//==============================================================================
LBM_HD LBM_INLINE void fwd1d(Real& a, Real& b, Real& c, Real u) {
  const Real cs2v = D3Q27::cs2();
  const Real s0 = a + b + c;   // sum
  const Real s1 = c - a;       // first raw moment
  const Real s2 = c + a;       // second raw moment
  a = s0;
  b = s1 - u * s0;
  c = s2 - Real(2) * u * s1 + (u * u - cs2v) * s0;
}

LBM_HD LBM_INLINE void inv1d(Real& a, Real& b, Real& c, Real u) {
  const Real cs2v = D3Q27::cs2();
  const Real m0 = a, m1 = b, m2 = c;
  const Real s1 = m1 + u * m0;
  const Real s2 = m2 + Real(2) * u * s1 - (u * u - cs2v) * m0;
  a = Real(0.5) * (s2 - s1);
  b = m0 - s2;
  c = Real(0.5) * (s2 + s1);
}

LBM_HD LBM_INLINE void to_moments(const Real f[27], const Real ub[3], Real k[27]) {
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) k[mi(a, b, c)] = f[pi(a, b, c)];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      fwd1d(k[mi(a, b, 0)], k[mi(a, b, 1)], k[mi(a, b, 2)], ub[2]);
  for (int a = 0; a < 3; ++a)
    for (int r = 0; r < 3; ++r)
      fwd1d(k[mi(a, 0, r)], k[mi(a, 1, r)], k[mi(a, 2, r)], ub[1]);
  for (int q = 0; q < 3; ++q)
    for (int r = 0; r < 3; ++r)
      fwd1d(k[mi(0, q, r)], k[mi(1, q, r)], k[mi(2, q, r)], ub[0]);
}

LBM_HD LBM_INLINE void to_populations(Real k[27], const Real ub[3], Real f[27]) {
  for (int q = 0; q < 3; ++q)
    for (int r = 0; r < 3; ++r)
      inv1d(k[mi(0, q, r)], k[mi(1, q, r)], k[mi(2, q, r)], ub[0]);
  for (int a = 0; a < 3; ++a)
    for (int r = 0; r < 3; ++r)
      inv1d(k[mi(a, 0, r)], k[mi(a, 1, r)], k[mi(a, 2, r)], ub[1]);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      inv1d(k[mi(a, b, 0)], k[mi(a, b, 1)], k[mi(a, b, 2)], ub[2]);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) f[pi(a, b, c)] = k[mi(a, b, c)];
}

//==============================================================================
//  Macroscopic moments and equilibrium.
//==============================================================================
struct Macro { Real rho, ux, uy, uz; };

LBM_HD LBM_INLINE Macro macroscopic(const Real f[27]) {
  Macro m{Real(0), Real(0), Real(0), Real(0)};
  for (int i = 0; i < 27; ++i) {
    m.rho += f[i];
    m.ux  += f[i] * Real(D3Q27::cx(i));
    m.uy  += f[i] * Real(D3Q27::cy(i));
    m.uz  += f[i] * Real(D3Q27::cz(i));
  }
  const Real inv = Real(1) / m.rho;
  m.ux *= inv; m.uy *= inv; m.uz *= inv;
  return m;
}

// Second-order equilibrium, used by BGK.
LBM_HD LBM_INLINE Real feq(int i, Real rho, Real ux, Real uy, Real uz) {
  const Real cu = Real(D3Q27::cx(i)) * ux + Real(D3Q27::cy(i)) * uy
                + Real(D3Q27::cz(i)) * uz;
  const Real u2 = ux * ux + uy * uy + uz * uz;
  return D3Q27::w(i) * rho *
         (Real(1) + Real(3) * cu + Real(4.5) * cu * cu - Real(1.5) * u2);
}

//------------------------------------------------------------------------------
// Equilibrium of one moment slot, as a product of three 1D factors.
//
// In CENTRAL moments the shift is the local velocity, so du = u - ub = 0 and
// every factor above order 0 collapses: the equilibrium central moments are
// simply rho times a product of {1, 0, cs^2}. That is the whole reason the
// operator is cheap to write, and it is worth not obscuring.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void eq_factors(const Real du[3], Real Qf[3][3]) {
  for (int a = 0; a < 3; ++a) {
    Qf[a][0] = Real(1);
    Qf[a][1] = du[a];
    // NOT du*du + cs^2. fwd1d already subtracts cs^2 from the second moment
    // (the `(u*u - cs2v) * s0` term), so the basis carries the DEVIATORIC
    // second moment and the equilibrium factor is du^2 alone. Adding cs^2 here
    // double-counts it: equilibrium stops being a fixed point of the operator,
    // by exactly cs^2 rho per diagonal component. host_check.cpp caught this.
    Qf[a][2] = du[a] * du[a];
  }
}

LBM_HD LBM_INLINE Real eq_moment(Real rho, const Real Qf[3][3], int n) {
  return rho * Qf[0][p_of(n)] * Qf[1][q_of(n)] * Qf[2][r_of(n)];
}

//==============================================================================
//  Collision operators.
//==============================================================================
enum class Op { BGK, CentralMoments };

LBM_HD LBM_INLINE void collide_bgk(Real f[27], const Macro& m, Real omega) {
  for (int i = 0; i < 27; ++i)
    f[i] += omega * (feq(i, m.rho, m.ux, m.uy, m.uz) - f[i]);
}

//------------------------------------------------------------------------------
// Central moments.
//
//   order 0     conserved
//   order 1     conserved (no forcing here yet)
//   order 2     trace at omega_bulk, deviatoric and shear at omega
//   order >= 3  straight to equilibrium
//
// This mirrors the relaxation schedule of the parent implementation exactly, so
// the two codes are comparable term by term.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void collide_cm(Real f[27], const Macro& m, Real omega, Real omega_bulk) {
  const Real ub[3] = {m.ux, m.uy, m.uz};
  const Real du[3] = {Real(0), Real(0), Real(0)};   // central: shift == velocity

  Real Qf[3][3];
  eq_factors(du, Qf);

  Real k[27];
  to_moments(f, ub, k);

  // ---- order 2 ----
  constexpr int I2D[3] = {mi(2, 0, 0), mi(0, 2, 0), mi(0, 0, 2)};
  constexpr int I2S[3] = {mi(1, 1, 0), mi(1, 0, 1), mi(0, 1, 1)};

  Real d[3], e[3];
  Real tr = Real(0), tre = Real(0);
  for (int a = 0; a < 3; ++a) {
    d[a] = k[I2D[a]];
    e[a] = eq_moment(m.rho, Qf, I2D[a]);
    tr += d[a]; tre += e[a];
  }
  const Real invD = Real(1) / Real(3);
  const Real tr_post = (Real(1) - omega_bulk) * tr + omega_bulk * tre;
  for (int a = 0; a < 3; ++a)
    k[I2D[a]] = (Real(1) - omega) * (d[a] - tr * invD)
              + omega * (e[a] - tre * invD) + tr_post * invD;
  for (int a = 0; a < 3; ++a)
    k[I2S[a]] = (Real(1) - omega) * k[I2S[a]] + omega * eq_moment(m.rho, Qf, I2S[a]);

  // ---- order >= 3 ----
  for (int n = 0; n < 27; ++n)
    if (order_of(n) >= 3) k[n] = eq_moment(m.rho, Qf, n);

  to_populations(k, ub, f);
}

LBM_HD LBM_INLINE Real omega_from_viscosity(Real nu) {
  return Real(1) / (nu * D3Q27::inv_cs2() + Real(0.5));
}

}  // namespace lbm
