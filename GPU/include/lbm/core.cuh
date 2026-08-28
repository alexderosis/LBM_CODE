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


//==============================================================================
//  D3Q7 -- the lattice the scalar and the magnetic field run on.
//
//  cs^2 = 1/4, not 1/3. That is not a typo and not a choice: it is what
//  sum_i w_i c_ia c_ib = cs^2 delta_ab forces once the weights are fixed by
//  isotropy on a seven-velocity set (w_0 = 2/8, w_i = 1/8).
//
//  WHY A SECOND LATTICE AT ALL. The scalar carries only its zeroth moment and
//  the magnetic field only its first, so neither needs the fourth-order
//  isotropy that Navier-Stokes does. Seven populations instead of 27 is a 3.9x
//  saving in the memory traffic that dominates an LBM kernel, and running the
//  coupled field on a smaller lattice is the point rather than an economy.
//
//  The pairing contract holds here too: (1,2) = -+x, (3,4) = -+y, (5,6) = -+z,
//  so opp(i) == i + 1 for odd i and Esoteric Pull works unchanged.
//==============================================================================
struct D3Q7 {
  static constexpr int D = 3;
  static constexpr int Q = 7;

  static LBM_HD LBM_INLINE int cx(int i) {
    constexpr int v[7] = {0, 1, -1, 0, 0, 0, 0};
    return v[i];
  }
  static LBM_HD LBM_INLINE int cy(int i) {
    constexpr int v[7] = {0, 0, 0, 1, -1, 0, 0};
    return v[i];
  }
  static LBM_HD LBM_INLINE int cz(int i) {
    constexpr int v[7] = {0, 0, 0, 0, 0, 1, -1};
    return v[i];
  }
  static LBM_HD LBM_INLINE Real w(int i) {
    constexpr int num[7] = {2, 1, 1, 1, 1, 1, 1};
    return Real(num[i]) / Real(8);
  }
  static LBM_HD LBM_INLINE Real cs2() { return Real(1) / Real(4); }
  static LBM_HD LBM_INLINE Real inv_cs2() { return Real(4); }
};

//------------------------------------------------------------------------------
// Opposite direction. Valid on BOTH lattices above, and only because both obey
// the same ordering contract -- direction 0 is the rest population and every
// odd i is paired with i+1. Anything that breaks that breaks Esoteric Pull too.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE constexpr int opp(int i) {
  return i == 0 ? 0 : ((i & 1) ? i + 1 : i - 1);
}

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
//  Passive scalar (temperature, concentration) -- advection-diffusion.
//
//  The scalar differs from the fluid in one structural way: its velocity is an
//  INPUT, not something recovered from its own populations. It carries only its
//  zeroth moment, T = sum_i h_i.
//
//  EQUILIBRIUM IS FIRST ORDER IN u, ON PURPOSE. D3Q7 has an isotropic second
//  moment but not an isotropic fourth-order one, and every one of its non-rest
//  velocities has a single nonzero component, so (c.u)^2 collapses to c_a^2
//  u_a^2 and the cross terms of the uu tensor cannot be represented at all.
//  A second-order term would therefore add anisotropy, not accuracy. The price
//  is an O(u^2) defect in the advection term, which is why this lattice suits
//  low-Mach transport -- exactly the regime Boussinesq convection lives in.
//
//  STORAGE REFERENCE. The arrays hold h_i = g_i - w_i T_ref. The collision works
//  on differences far smaller than the populations themselves and in FP32 that
//  cancellation costs most of the mantissa. Unlike density there is no universal
//  reference -- rho is always near 1, but a temperature scale is whatever the
//  problem says it is -- so T_ref is a runtime parameter and leaving it at 0
//  reproduces the unshifted scheme exactly. Set it to the mean temperature.
//==============================================================================
template <class L>
LBM_HD LBM_INLINE Real scalar_deviation(const Real h[L::Q]) {
  Real t = Real(0);
  for (int i = 0; i < L::Q; ++i) t += h[i];
  return t;
}

// h_i^eq = w_i [ dT + T (c_i.u)/cs^2 ],  T = T_ref + dT.
// Every term is small when dT and u are, which is the whole point of storing
// the deviation rather than writing eq_raw - w_i T_ref.
template <class L>
LBM_HD LBM_INLINE Real scalar_eq(int i, Real dT, Real T_ref, Real ux, Real uy, Real uz) {
  const Real cu = Real(L::cx(i)) * ux + Real(L::cy(i)) * uy + Real(L::cz(i)) * uz;
  return L::w(i) * (dT + (T_ref + dT) * L::inv_cs2() * cu);
}

template <class L>
LBM_HD LBM_INLINE void collide_scalar(Real h[L::Q], Real dT, Real T_ref,
                                      Real ux, Real uy, Real uz, Real omega) {
  for (int i = 0; i < L::Q; ++i)
    h[i] += omega * (scalar_eq<L>(i, dT, T_ref, ux, uy, uz) - h[i]);
}

template <class L>
LBM_HD LBM_INLINE Real omega_from_diffusivity(Real d) {
  return Real(1) / (d * L::inv_cs2() + Real(0.5));
}
template <class L>
LBM_HD LBM_INLINE Real diffusivity_from_omega(Real w) {
  return (Real(1) / w - Real(0.5)) * L::cs2();
}

//==============================================================================
//  Magnetic induction -- Dellar's vector-valued distribution.
//
//  The magnetic field is carried by a distribution that is a VECTOR at every
//  lattice link, g_i^alpha, with B_alpha = sum_i g_i^alpha and
//
//      g_i^{alpha,eq} = w_i [ B_a + (1/cs^2) c_ib (u_b B_a - B_b u_a) ].
//
//  Its first moment is u_b B_a - B_b u_a, exactly the antisymmetric flux of
//
//      d_t B_a + d_b (u_b B_a - B_b u_a) = eta lap B_a,
//
//  so only that moment is required of the lattice and D3Q7 suffices. The
//  antisymmetry is what keeps div B from being generated: it is a property of
//  the equilibrium, not something enforced afterwards, and host_check asserts it.
//
//  Storage is unshifted. B oscillates about zero in every case here, so there is
//  no nonzero reference for a shift to subtract -- the mirror image of the
//  argument for the scalar above.
//==============================================================================
template <class L>
LBM_HD LBM_INLINE Real magnetic_eq(int i, int a, const Real B[3], const Real u[3]) {
  const Real c[3] = {Real(L::cx(i)), Real(L::cy(i)), Real(L::cz(i))};
  Real flux = Real(0);                     // c_b (u_b B_a - B_b u_a)
  for (int b = 0; b < 3; ++b) flux += c[b] * (u[b] * B[a] - B[b] * u[a]);
  return L::w(i) * (B[a] + L::inv_cs2() * flux);
}

template <class L>
LBM_HD LBM_INLINE void collide_magnetic(Real g[L::Q], int a, const Real B[3],
                                        const Real u[3], Real omega) {
  for (int i = 0; i < L::Q; ++i)
    g[i] += omega * (magnetic_eq<L>(i, a, B, u) - g[i]);
}

template <class L>
LBM_HD LBM_INLINE Real omega_from_resistivity(Real eta) {
  return Real(1) / (eta * L::inv_cs2() + Real(0.5));
}

//------------------------------------------------------------------------------
// The Maxwell stress, as an addition to the FLUID equilibrium.
//
// The momentum equation gains -div(BB - |B|^2 I / 2). The clean way to get it is
// not to form that divergence and apply it as a body force -- that needs
// derivatives of B and loses an order -- but to give the fluid equilibrium the
// right second moment directly (Dellar 2002):
//
//     sum_i c_a c_b f^eq = rho u_a u_b + (p + |B|^2/2) delta_ab - B_a B_b,
//
// achieved by adding df_i = (w_i / 2 cs^4) M_ab (c_ia c_ib - cs^2 delta_ab),
// with M_ab = (|B|^2/2) delta_ab - B_a B_b.
//
// df contributes nothing to mass or momentum -- sum df = 0 and sum c df = 0 --
// so it perturbs the stress alone. host_check asserts both.
//
// Written out rather than looped: M_ab c_a c_b = |b|^2 |c|^2 / 2 - (c.b)^2, and
// cs^2 M_aa = cs^2 (D/2 - 1) |b|^2, which is |b|^2/6 in three dimensions.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE Real maxwell(int i, const Real B[3]) {
  constexpr Real cs2v = Real(1) / Real(3);
  const Real b2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
  const Real c[3] = {Real(D3Q27::cx(i)), Real(D3Q27::cy(i)), Real(D3Q27::cz(i))};
  Real c2 = Real(0), cb = Real(0);
  for (int a = 0; a < 3; ++a) { c2 += c[a] * c[a]; cb += c[a] * B[a]; }
  const Real trace = cs2v * (Real(3) * Real(0.5) - Real(1)) * b2;
  return D3Q27::w(i) * (Real(0.5) * b2 * c2 - cb * cb - trace)
       / (Real(2) * cs2v * cs2v);
}

//==============================================================================
//  Body force -- Guo et al. (2002).
//
//      u   = ( sum_i c_i f_i + F/2 ) / rho
//      S_i = (1 - omega/2) w_i [ (c_i - u)/cs^2 + (c_i.u) c_i / cs^4 ] . F
//
//  The half-force shift in the velocity is not optional. Omitting it biases the
//  measured velocity by F/(2 rho), which on a forced channel is a systematic
//  offset in the profile rather than a small error.
//==============================================================================
template <class L>
LBM_HD LBM_INLINE Real guo_source_raw(int i, const Real F[3], Real ux, Real uy, Real uz) {
  const Real ics2 = L::inv_cs2();
  const Real cx = Real(L::cx(i)), cy = Real(L::cy(i)), cz = Real(L::cz(i));
  const Real cu = cx * ux + cy * uy + cz * uz;
  const Real bx = (cx - ux) * ics2 + cu * ics2 * ics2 * cx;
  const Real by = (cy - uy) * ics2 + cu * ics2 * ics2 * cy;
  const Real bz = (cz - uz) * ics2 + cu * ics2 * ics2 * cz;
  return L::w(i) * (bx * F[0] + by * F[1] + bz * F[2]);
}

//------------------------------------------------------------------------------
// How the force at a node is obtained. A POD passed to the kernel by value; the
// KIND is a template parameter so the branch folds away and an unforced run
// costs nothing.
//
//   ForceNone        no force
//   ForceUniform     a constant (fx, fy, fz) -- a pressure gradient, gravity
//   ForceBoussinesq  F = rho0 g beta (T(n) - T0), plus the uniform part
//
// Boussinesq holds the density constant everywhere except in the buoyancy term,
// which is why a thermal field couples back into the flow through a force
// rather than through the equation of state.
//------------------------------------------------------------------------------
enum ForceKind { ForceNone = 0, ForceUniform = 1, ForceBoussinesq = 2 };

struct BodyForce {
  Real fx = Real(0), fy = Real(0), fz = Real(0);      // uniform part
  const Real* T = nullptr;                            // temperature field
  Real gx = Real(0), gy = Real(-1), gz = Real(0);     // gravity direction
  Real rho0 = Real(1), beta = Real(1), T0 = Real(0);
};

template <int Kind>
LBM_HD LBM_INLINE void force_at(const BodyForce& b, long n, Real F[3]) {
  if (Kind == ForceNone) { F[0] = F[1] = F[2] = Real(0); return; }
  if (Kind == ForceUniform) { F[0] = b.fx; F[1] = b.fy; F[2] = b.fz; return; }
  const Real s = b.rho0 * b.beta * (b.T[n] - b.T0);
  F[0] = b.fx + b.gx * s;
  F[1] = b.fy + b.gy * s;
  F[2] = b.fz + b.gz * s;
}

//==============================================================================
//  Collision operators.
//==============================================================================
enum class Op { BGK, CentralMoments };

//------------------------------------------------------------------------------
// Guo's half-force shift. The velocity a forced scheme must collide with, and
// must report, is the one that includes F/(2 rho); leaving it out biases the
// profile of a forced channel by a constant rather than by a small amount.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void shift_velocity(Macro& m, const Real F[3]) {
  const Real h = Real(0.5) / m.rho;
  m.ux += h * F[0];  m.uy += h * F[1];  m.uz += h * F[2];
}

//------------------------------------------------------------------------------
// Everything the fluid collision needs at a node beyond its own populations.
// Filled by the kernel; both members are ignored unless the matching template
// flag is set, so an uncoupled run never touches them.
//------------------------------------------------------------------------------
struct Coupling {
  Real F[3] = {Real(0), Real(0), Real(0)};   // body force at this node
  Real B[3] = {Real(0), Real(0), Real(0)};   // magnetic field at this node
};

//------------------------------------------------------------------------------
// The central moments of the Maxwell-stress term, computed exactly.
//
// Rather than derive closed forms for all 27 slots -- orders 4 to 6 do not
// factorise compactly -- the term is built as populations and transformed. The
// transform is linear, so this is exact at every order, and it costs one extra
// forward pass on the MHD kernel alone. The parent implementation reached the
// same conclusion for the same reason.
//
// The false specialisation holds no state and returns zero, so a non-MHD
// instantiation carries not one extra register.
//------------------------------------------------------------------------------
template <bool Mhd>
struct MaxwellMoments {
  Real k[27];
  LBM_HD LBM_INLINE MaxwellMoments(const Real B[3], const Real ub[3]) {
    Real df[27];
    for (int i = 0; i < 27; ++i) df[i] = maxwell(i, B);
    to_moments(df, ub, k);
  }
  LBM_HD LBM_INLINE Real operator[](int n) const { return k[n]; }
};
template <>
struct MaxwellMoments<false> {
  LBM_HD LBM_INLINE MaxwellMoments(const Real*, const Real*) {}
  LBM_HD LBM_INLINE Real operator[](int) const { return Real(0); }
};

//------------------------------------------------------------------------------
// BGK. The Maxwell stress enters the equilibrium; the body force enters as
// Guo's source with the (1 - omega/2) prefactor BGK is entitled to fold in
// because it relaxes every mode at the same rate.
//------------------------------------------------------------------------------
template <bool Forced, bool Mhd>
LBM_HD LBM_INLINE void collide_bgk_gen(Real f[27], const Macro& m, Real omega,
                                       const Coupling& cp) {
  const Real w2 = Real(1) - Real(0.5) * omega;
  for (int i = 0; i < 27; ++i) {
    Real e = feq(i, m.rho, m.ux, m.uy, m.uz);
    if (Mhd) e += maxwell(i, cp.B);
    f[i] += omega * (e - f[i]);
    if (Forced) f[i] += w2 * guo_source_raw<D3Q27>(i, cp.F, m.ux, m.uy, m.uz);
  }
}

//------------------------------------------------------------------------------
// Central moments.
//
//   order 0     conserved
//   order 1     conserved, and the ONLY place the force enters
//   order 2     trace at omega_bulk, deviatoric and shear at omega
//   order >= 3  straight to equilibrium
//
// This mirrors the relaxation schedule of the parent implementation exactly, so
// the two codes are comparable term by term.
//
// WHY THE FORCE ENTERS ONLY AT ORDER 1. The Guo source has raw moments
// sum c_a S_i = F_a and sum c_a c_b S_i = u_a F_b + u_b F_a. Transformed to the
// basis shifted by u_b those become F_a and (F_b du_a + F_a du_b), and for
// CENTRAL moments du = 0, so the second-order force contribution vanishes
// identically. Nor is a third-order term needed: this basis is Hermite,
// phi_2(C) = C^2 - cs^2, and k_21(monomial) = k_21(this basis) + cs^2 k_01, so
// adding F at order 1 already delivers cs^2 F in the monomial third moments for
// free. Adding them explicitly double counts -- measured in the parent, it gives
// 1.5 cs^2 F where cs^2 F is right.
//------------------------------------------------------------------------------
template <bool Forced, bool Mhd>
LBM_HD LBM_INLINE void collide_cm_gen(Real f[27], const Macro& m, Real omega,
                                      Real omega_bulk, const Coupling& cp) {
  const Real ub[3] = {m.ux, m.uy, m.uz};
  const Real du[3] = {Real(0), Real(0), Real(0)};   // central: shift == velocity

  Real Qf[3][3];
  eq_factors(du, Qf);

  Real k[27];
  to_moments(f, ub, k);

  const MaxwellMoments<Mhd> kM(cp.B, ub);

  // ---- order 1: the force, and nothing else ----
  if (Forced) {
    k[mi(1, 0, 0)] += cp.F[0];
    k[mi(0, 1, 0)] += cp.F[1];
    k[mi(0, 0, 1)] += cp.F[2];
  }

  // ---- order 2 ----
  constexpr int I2D[3] = {mi(2, 0, 0), mi(0, 2, 0), mi(0, 0, 2)};
  constexpr int I2S[3] = {mi(1, 1, 0), mi(1, 0, 1), mi(0, 1, 1)};

  Real d[3], e[3];
  Real tr = Real(0), tre = Real(0);
  for (int a = 0; a < 3; ++a) {
    d[a] = k[I2D[a]];
    e[a] = eq_moment(m.rho, Qf, I2D[a]) + kM[I2D[a]];
    tr += d[a]; tre += e[a];
  }
  const Real invD = Real(1) / Real(3);
  const Real tr_post = (Real(1) - omega_bulk) * tr + omega_bulk * tre;
  for (int a = 0; a < 3; ++a)
    k[I2D[a]] = (Real(1) - omega) * (d[a] - tr * invD)
              + omega * (e[a] - tre * invD) + tr_post * invD;
  for (int a = 0; a < 3; ++a)
    k[I2S[a]] = (Real(1) - omega) * k[I2S[a]]
              + omega * (eq_moment(m.rho, Qf, I2S[a]) + kM[I2S[a]]);

  // ---- order >= 3 ----
  for (int n = 0; n < 27; ++n)
    if (order_of(n) >= 3) k[n] = eq_moment(m.rho, Qf, n) + kM[n];

  to_populations(k, ub, f);
}

//------------------------------------------------------------------------------
// The uncoupled forms, unchanged in meaning and in signature.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void collide_bgk(Real f[27], const Macro& m, Real omega) {
  const Coupling cp{};
  collide_bgk_gen<false, false>(f, m, omega, cp);
}

LBM_HD LBM_INLINE void collide_cm(Real f[27], const Macro& m, Real omega, Real omega_bulk) {
  const Coupling cp{};
  collide_cm_gen<false, false>(f, m, omega, omega_bulk, cp);
}

LBM_HD LBM_INLINE Real omega_from_viscosity(Real nu) {
  return Real(1) / (nu * D3Q27::inv_cs2() + Real(0.5));
}

}  // namespace lbm
