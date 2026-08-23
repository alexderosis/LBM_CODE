//==============================================================================
//  Moment transform and moment-space collision.
//
//  The round-trip test is the analogue of `M^-1 M == I` for the factorised
//  transform: it is the first thing to check for any moment-based operator, and
//  it catches an index or sign slip instantly.
//==============================================================================
#include "Check.hpp"
#include "collision/BGK.hpp"
#include "collision/MhdCentralMoments.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/MonomialBasis.hpp"
#include "collision/ProductBasis.hpp"
#include "collision/TRT.hpp"
#include "core/Types.hpp"

#include <cmath>

using namespace lbm;

static Real TOL() { return sizeof(Real) == 4 ? Real(2e-6) : Real(5e-13); }

// Deterministic, non-equilibrium, strictly positive populations.
template <class L, class Store>
static void fill(Real f[L::Q]) {
  for (int i = 0; i < L::Q; ++i) {
    f[i] = weight<L, Real>(i) * (Real(1) + Real(0.13) * Real((i * 7) % 5 - 2) +
                                 Real(0.05) * Real(i % 3));
    if constexpr (Store::shifted) f[i] -= weight<L, Real>(i);
  }
}

//------------------------------------------------------------------------------
template <class L>
void round_trip(Real ubx, Real uby, Real ubz) {
  using B = typename SelectBasis<L>::type;
  const std::string n = std::string(L::name) + " ub=(" +
                        std::to_string(double(ubx)).substr(0, 5) + ",..)";
  Real f0[L::Q], f[L::Q], k[B::NM];
  fill<L, RawPopulations>(f0);
  for (int i = 0; i < L::Q; ++i) f[i] = f0[i];
  const Real ub[3] = {ubx, uby, ubz};
  B::template to_moments<true>(f, ub, k);
  B::template to_populations<true>(k, ub, f);
  double worst = 0;
  for (int i = 0; i < L::Q; ++i) worst = std::max(worst, std::abs(double(f[i] - f0[i])));
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()), n + ": transform round-trip is the identity" + buf);
}

// The factorised transform must equal the direct contraction with the basis.
template <class L>
void against_direct(Real ubx, Real uby, Real ubz) {
  using B = typename SelectBasis<L>::type;
  Real f[L::Q], k[B::NM];
  fill<L, RawPopulations>(f);
  const Real ub[3] = {ubx, uby, ubz};
  B::template to_moments<true>(f, ub, k);

  double worst = 0;
  for (int n = 0; n < B::NM; ++n) {
    Real direct = 0;
    for (int i = 0; i < L::Q; ++i) {
      Real t = f[i] * B::phi(B::p_of(n), cvel<L>(i, 0), ub[0])
                    * B::phi(B::q_of(n), cvel<L>(i, 1), ub[1]);
      if constexpr (L::D == 3) t *= B::phi(B::r_of(n), cvel<L>(i, 2), ub[2]);
      direct += t;
    }
    worst = std::max(worst, std::abs(double(direct - k[n])));
  }
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string(L::name) + ": factorised transform == direct contraction" + buf);
}

// The equilibrium's CENTRAL MONOMIAL moments must be the Maxwellian ones:
//     k_pqr = rho M(p) M(q) M(r),   M = {1, 0, cs2}
// contracted directly here, with no reference to the basis machinery, so this is
// a check on the physics rather than on the code reproducing itself. It holds on
// every monomial the lattice can represent -- which for D3Q19 is the 19 in its
// basis, not the full 27.
template <class L, class Store>
void equilibrium_is_maxwellian(Real rho, Real ux, Real uy, Real uz) {
  using C = MomentCollision<L, NoForcing, Store, true>;
  using B = typename SelectBasis<L>::type;
  constexpr Real cs2v = cs2<L, Real>();
  const Real Mx[3] = {Real(1), Real(0), cs2v};

  Real f[L::Q];
  for (int i = 0; i < L::Q; ++i) {
    f[i] = C::seed_value(i, rho, ux, uy, uz);
    if constexpr (Store::shifted) f[i] += weight<L, Real>(i);   // back to raw f
  }
  const Real u[3] = {ux, uy, uz};
  double worst = 0;
  for (int n = 0; n < B::NM; ++n) {
    const int pq[3] = {B::p_of(n), B::q_of(n), B::r_of(n)};
    Real got = 0;
    for (int i = 0; i < L::Q; ++i) {
      Real t = f[i];
      for (int a = 0; a < L::D; ++a)
        for (int e = 0; e < pq[a]; ++e) t *= (Real(cvel<L>(i, a)) - u[a]);
      got += t;
    }
    Real want = rho * Mx[pq[0]] * Mx[pq[1]];
    if constexpr (L::D == 3) want *= Mx[pq[2]];
    worst = std::max(worst, std::abs(double(got - want)));
  }
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string(L::name) + "/" + Store::name +
            ": equilibrium central moments are Maxwellian" + buf);
}

//------------------------------------------------------------------------------
template <class Coll, class L, class Store>
void conservation(const char* tag, Coll coll) {
  const std::string n = std::string(L::name) + "/" + Store::name + " " + tag;
  Real f[L::Q]; fill<L, Store>(f);
  Real m0 = 0, p0[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m0 += f[i];
    for (int a = 0; a < 3; ++a) p0[a] += f[i] * Real(cvel<L>(i, a));
  }
  const Macro m = coll.macroscopic(f);
  coll.collide(f, m);
  Real m1 = 0, p1[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m1 += f[i];
    for (int a = 0; a < 3; ++a) p1[a] += f[i] * Real(cvel<L>(i, a));
  }
  check::near(m1, m0, TOL(), n + ": conserves mass");
  for (int a = 0; a < L::D; ++a)
    check::near(p1[a], p0[a], TOL(), n + ": conserves momentum");
}

template <class Coll, class L, class Store>
void forced_momentum(const char* tag, Coll coll, const Real F[3]) {
  const std::string n = std::string(L::name) + "/" + Store::name + " " + tag;
  Real f[L::Q]; fill<L, Store>(f);
  Real m0 = 0, p0[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m0 += f[i];
    for (int a = 0; a < 3; ++a) p0[a] += f[i] * Real(cvel<L>(i, a));
  }
  const Macro m = coll.macroscopic(f);
  coll.collide(f, m);
  Real m1 = 0, p1[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m1 += f[i];
    for (int a = 0; a < 3; ++a) p1[a] += f[i] * Real(cvel<L>(i, a));
  }
  check::near(m1, m0, TOL(), n + ": forcing conserves mass");
  for (int a = 0; a < L::D; ++a)
    check::near(p1[a], p0[a] + F[a], TOL(), n + ": forcing adds exactly F to momentum");
}

// With omega = 1 and every higher moment sent to equilibrium, the whole
// post-collision state must BE the equilibrium.
template <class L, class Store, bool Central>
void full_relaxation() {
  using C = MomentCollision<L, NoForcing, Store, Central>;
  const std::string n = std::string(L::name) + "/" + Store::name + " " + C::name;
  C coll; coll.omega = Real(1); coll.omega_bulk = Real(1);
  Real f[L::Q]; fill<L, Store>(f);
  const Macro m = coll.macroscopic(f);
  const Real rho = C::density(m);
  coll.collide(f, m);
  double worst = 0;
  for (int i = 0; i < L::Q; ++i)
    worst = std::max(worst, std::abs(double(f[i] - C::seed_value(i, rho, m.ux, m.uy, m.uz))));
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()), n + ": omega=1 lands exactly on the equilibrium" + buf);
}

// At rest the two operators are the same thing: the basis velocity is 0 either way.
template <class L, class Store>
void cm_equals_mrt_at_rest() {
  MomentCollision<L, NoForcing, Store, true>  cm;  cm.omega = Real(1.3);
  MomentCollision<L, NoForcing, Store, false> mr;  mr.omega = Real(1.3);
  Real a[L::Q], b[L::Q];
  // a state with exactly zero momentum
  for (int i = 0; i < L::Q; ++i) {
    const Real v = weight<L, Real>(i) * (Real(1) + Real(0.07) * Real((i % 3) - 1));
    a[i] = b[i] = Store::shifted ? v - weight<L, Real>(i) : v;
  }
  for (int i = 1; i < L::Q; i += 2) {            // symmetrise to kill the momentum
    const Real s = Real(0.5) * (a[i] + a[i + 1]);
    a[i] = a[i + 1] = b[i] = b[i + 1] = s;
  }
  cm.collide(a, cm.macroscopic(a));
  mr.collide(b, mr.macroscopic(b));
  double worst = 0;
  for (int i = 0; i < L::Q; ++i) worst = std::max(worst, std::abs(double(a[i] - b[i])));
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string(L::name) + "/" + Store::name +
            ": central moments == raw MRT when u = 0" + buf);
}

//------------------------------------------------------------------------------
// The hybrid MHD central-moment operator relaxes toward Eq. (11) of De Rosis,
// Leveque & Chahine (2018). Those are the central moments of that paper's
// Eq. (1) equilibrium -- second-order truncated hydrodynamics PLUS the Maxwell
// term -- so the check is to build that equilibrium, contract it directly, and
// compare. This catches a transcription slip and, just as importantly, a sign
// convention mismatch on b.
//------------------------------------------------------------------------------
void mhd_cm_equilibrium(Real rho, Real ux, Real uy, Real bx, Real by) {
  constexpr Real cs2v = cs2<D2Q9, Real>();
  constexpr Real cs4v = cs2v * cs2v;
  Real feq[9];
  for (int i = 0; i < 9; ++i) {
    const Real cx = Real(D2Q9::cx(i)), cy = Real(D2Q9::cy(i));
    const Real cu = cx * ux + cy * uy, uu = ux * ux + uy * uy;
    const Real hydro = weight<D2Q9, Real>(i) * rho *
        (Real(1) + cu / cs2v + cu * cu / (Real(2) * cs4v) - uu / (Real(2) * cs2v));
    const Real c2 = cx * cx + cy * cy, cb = cx * bx + cy * by;
    const Real b2 = bx * bx + by * by;
    const Real mag = weight<D2Q9, Real>(i) / (Real(2) * cs4v) *
        (Real(0.5) * c2 * b2 - cb * cb);
    feq[i] = hydro + mag;
  }
  auto cm = [&](int p, int q) {
    Real s = 0;
    for (int i = 0; i < 9; ++i) {
      Real t = feq[i];
      for (int e = 0; e < p; ++e) t *= (Real(D2Q9::cx(i)) - ux);
      for (int e = 0; e < q; ++e) t *= (Real(D2Q9::cy(i)) - uy);
      s += t;
    }
    return s;
  };
  const Real bx2 = bx * bx, by2 = by * by, bxy = bx * by;
  const Real want[9] = {
    rho, Real(0), Real(0), Real(2) / Real(3) * rho, by2 - bx2, -bxy,
    -rho * ux * ux * uy + Real(0.5) * uy * (bx2 - by2) + Real(2) * ux * bxy,
    -rho * ux * uy * uy + Real(0.5) * ux * (by2 - bx2) + Real(2) * uy * bxy,
    rho / Real(9) * (Real(27) * ux * ux * uy * uy + Real(1))
      + Real(0.5) * (ux * ux - uy * uy) * (bx2 - by2) - Real(4) * ux * uy * bxy};
  const Real got[9] = {cm(0, 0), cm(1, 0), cm(0, 1),
                       cm(2, 0) + cm(0, 2), cm(2, 0) - cm(0, 2), cm(1, 1),
                       cm(2, 1), cm(1, 2), cm(2, 2)};
  double worst = 0;
  for (int i = 0; i < 9; ++i) worst = std::max(worst, std::abs(double(got[i] - want[i])));
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string("MhdCM: Eq.(11) equals the central moments of Eq.(1)") + buf);
}

// Mass and momentum are collision invariants of the hybrid operator.
void mhd_cm_conservation(Real omega) {
  View1D<Real> bx("bx", 1), by("by", 1), bz("bz", 1);
  auto hbx = Kokkos::create_mirror_view(bx); hbx(0) = Real(0.021);
  auto hby = Kokkos::create_mirror_view(by); hby(0) = Real(-0.013);
  Kokkos::deep_copy(bx, hbx); Kokkos::deep_copy(by, hby);
  MhdCentralMoments<D2Q9, false> c;
  c.omega = omega; c.omega_bulk = Real(1);
  c.Bx = bx; c.By = by; c.Bz = bz;
  Real f[9]; fill<D2Q9, RawPopulations>(f);
  Real m0 = 0, p0[2] = {0, 0};
  for (int i = 0; i < 9; ++i) {
    m0 += f[i]; p0[0] += f[i] * Real(D2Q9::cx(i)); p0[1] += f[i] * Real(D2Q9::cy(i));
  }
  const Macro m = c.macroscopic(f, 0);
  c.collide(f, m, 0);
  Real m1 = 0, p1[2] = {0, 0};
  for (int i = 0; i < 9; ++i) {
    m1 += f[i]; p1[0] += f[i] * Real(D2Q9::cx(i)); p1[1] += f[i] * Real(D2Q9::cy(i));
  }
  const std::string n = "MhdCM omega=" + std::to_string(double(omega));
  check::near(m1, m0, TOL(), n + ": conserves mass");
  check::near(p1[0], p0[0], TOL(), n + ": conserves x-momentum");
  check::near(p1[1], p0[1], TOL(), n + ": conserves y-momentum");
}


//------------------------------------------------------------------------------
// D3Q27 hybrid MHD operator. Two independent things are checked here.
//
//  (a) The monomial central-moment transform used by the 3D operator agrees
//      with a brute-force contraction sum_i f_i (c-u)^p ... . This is NOT
//      covered by round_trip(): ProductBasis::fwd1d subtracts cs^2 (Hermite
//      basis) whereas the MHD operator uses plain monomials, matching Eq. (11).
//
//  (b) The equilibrium central moments obey the closed forms derived from the
//      3D equilibrium (second order plus Maxwell stress), with
//      M_ab = |b|^2 delta_ab / 2 - b_a b_b:
//          k_ab   = rho cs2 delta_ab + M_ab
//          k_abc  = -rho u_a u_b u_c - (u_a M_bc + u_b M_ac + u_c M_ab)
//      plus the two order-four families written out below. Orders 5 and 6 are
//      left to (a) -- they have no compact form worth transcribing twice.
//------------------------------------------------------------------------------
template <bool HighOrder>
void mhd_cm3d_moments(Real rho, const Real u[3], const Real b[3]) {
  using Op = MhdCentralMoments<D3Q27, HighOrder>;
  using B  = ProductBasis<D3Q27>;
  Op op;
  Real fe[27];  op.equilibrium(fe, rho, u, b);
  Real ke[27];  Op::to_moments(fe, u, ke);

  auto direct = [&](int p, int q, int r) {
    Real s = 0;
    for (int i = 0; i < 27; ++i) {
      Real t = fe[i];
      for (int e = 0; e < p; ++e) t *= (Real(D3Q27::cx(i)) - u[0]);
      for (int e = 0; e < q; ++e) t *= (Real(D3Q27::cy(i)) - u[1]);
      for (int e = 0; e < r; ++e) t *= (Real(D3Q27::cz(i)) - u[2]);
      s += t;
    }
    return s;
  };
  double worst = 0;
  for (int p = 0; p < 3; ++p)
    for (int q = 0; q < 3; ++q)
      for (int r = 0; r < 3; ++r)
        worst = std::max(worst, std::abs(double(ke[B::mi(p, q, r)] - direct(p, q, r))));
  { char s[48]; std::snprintf(s, sizeof s, " (worst %.2e)", worst);
    check::ok(worst <= double(TOL()),
              std::string("MhdCM/D3Q27") + (HighOrder ? "/ProductForm" : "/SecondOrder") +
              ": monomial transform == direct contraction" + s); }

  // closed forms
  const Real b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
  auto M = [&](int a, int c) { return Real(0.5) * b2 * Real(a == c) - b[a] * b[c]; };
  constexpr Real cs2v = cs2<D3Q27, Real>();
  auto e = [](int a) { int v[3] = {0, 0, 0}; v[a] = 1; return B::mi(v[0], v[1], v[2]); };
  auto idx = [](int p, int q, int r) { return B::mi(p, q, r); };

  worst = std::abs(double(ke[B::mi(0, 0, 0)] - rho));
  for (int a = 0; a < 3; ++a) worst = std::max(worst, std::abs(double(ke[e(a)])));
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c) {
      int v[3] = {0, 0, 0}; ++v[a]; ++v[c];
      const Real want = rho * cs2v * Real(a == c) + M(a, c);
      worst = std::max(worst, std::abs(double(ke[idx(v[0], v[1], v[2])] - want)));
    }
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c)
      for (int d = 0; d < 3; ++d) {
        int v[3] = {0, 0, 0}; ++v[a]; ++v[c]; ++v[d];
        if (v[0] > 2 || v[1] > 2 || v[2] > 2) continue;   // no c^3 on a product lattice
        const Real want = (HighOrder ? Real(0) : -rho * u[a] * u[c] * u[d])
                        - (u[a] * M(c, d) + u[c] * M(a, d) + u[d] * M(a, c));
        worst = std::max(worst, std::abs(double(ke[idx(v[0], v[1], v[2])] - want)));
      }
  // order four: the two distinct families, over every axis assignment
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c) {
      if (a == c) continue;
      const int d = 3 - a - c;                       // the remaining axis
      int v[3] = {0, 0, 0}; v[a] = 2; v[c] = 2;      // k_aacc
      Real want = rho * (Real(1) / Real(9) +
                         (HighOrder ? Real(0) : Real(3) * u[a] * u[a] * u[c] * u[c]))
                + Real(1) / Real(3) * b[d] * b[d]
                + Real(0.5) * u[c] * u[c] * (b2 - Real(2) * b[a] * b[a])
                + Real(0.5) * u[a] * u[a] * (b2 - Real(2) * b[c] * b[c])
                - Real(4) * u[a] * u[c] * b[a] * b[c];
      worst = std::max(worst, std::abs(double(ke[idx(v[0], v[1], v[2])] - want)));
      int w[3] = {0, 0, 0}; w[a] = 2; w[c] = 1; w[d] = 1;   // k_aacd
      want = (HighOrder ? Real(0) : Real(3) * rho * u[a] * u[a] * u[c] * u[d])
           - b[c] * b[d] / Real(3)
           + Real(0.5) * u[c] * u[d] * (b2 - Real(2) * b[a] * b[a])
           - Real(2) * u[a] * u[d] * b[a] * b[c] - Real(2) * u[a] * u[c] * b[a] * b[d]
           - u[a] * u[a] * b[c] * b[d];
      worst = std::max(worst, std::abs(double(ke[idx(w[0], w[1], w[2])] - want)));
    }
  { char s[48]; std::snprintf(s, sizeof s, " (worst %.2e)", worst);
    check::ok(worst <= double(TOL()),
              std::string("MhdCM/D3Q27") + (HighOrder ? "/ProductForm" : "/SecondOrder") +
              ": equilibrium CMs match the closed forms" + s); }
}

// With w = b_z = 0 the D3Q27 equilibrium must reproduce every moment of the
// published 2D set, Eq. (11) -- the 3D scheme is a genuine extension, not a
// different model that happens to agree in the bulk.
template <bool HighOrder>
void mhd_cm3d_reduces_to_2d(Real rho, Real ux, Real uy, Real bx, Real by) {
  using Op = MhdCentralMoments<D3Q27, HighOrder>;
  using B  = ProductBasis<D3Q27>;
  const Real u[3] = {ux, uy, Real(0)}, b[3] = {bx, by, Real(0)};
  Real fe[27];  Op op;  op.equilibrium(fe, rho, u, b);
  Real k[27];   Op::to_moments(fe, u, k);
  const Real bx2 = bx * bx, by2 = by * by, bxy = bx * by;
  const Real want[6] = {
    Real(2) / Real(3) * rho, by2 - bx2, -bxy,
    (HighOrder ? Real(0) : -rho * ux * ux * uy)
      + Real(0.5) * uy * (bx2 - by2) + Real(2) * ux * bxy,
    (HighOrder ? Real(0) : -rho * ux * uy * uy)
      + Real(0.5) * ux * (by2 - bx2) + Real(2) * uy * bxy,
    (HighOrder ? rho / Real(9)
               : rho / Real(9) * (Real(27) * ux * ux * uy * uy + Real(1)))
      + Real(0.5) * (ux * ux - uy * uy) * (bx2 - by2) - Real(4) * ux * uy * bxy};
  const Real got[6] = {k[B::mi(2, 0, 0)] + k[B::mi(0, 2, 0)],
                       k[B::mi(2, 0, 0)] - k[B::mi(0, 2, 0)],
                       k[B::mi(1, 1, 0)], k[B::mi(2, 1, 0)],
                       k[B::mi(1, 2, 0)], k[B::mi(2, 2, 0)]};
  double worst = 0;
  for (int i = 0; i < 6; ++i) worst = std::max(worst, std::abs(double(got[i] - want[i])));
  char s[48]; std::snprintf(s, sizeof s, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string("MhdCM/D3Q27") + (HighOrder ? "/ProductForm" : "/SecondOrder") +
            ": reduces to the 2D scheme when w = b_z = 0" + s);
}

// Mass and all three momentum components are collision invariants in 3D too.
template <bool HighOrder>
void mhd_cm3d_conservation(Real omega) {
  View1D<Real> bx("bx", 1), by("by", 1), bz("bz", 1);
  auto hbx = Kokkos::create_mirror_view(bx); hbx(0) = Real(0.021);
  auto hby = Kokkos::create_mirror_view(by); hby(0) = Real(-0.013);
  auto hbz = Kokkos::create_mirror_view(bz); hbz(0) = Real(0.008);
  Kokkos::deep_copy(bx, hbx); Kokkos::deep_copy(by, hby); Kokkos::deep_copy(bz, hbz);
  MhdCentralMoments<D3Q27, HighOrder> c;
  c.omega = omega; c.omega_bulk = Real(1);
  c.Bx = bx; c.By = by; c.Bz = bz;
  Real f[27]; fill<D3Q27, RawPopulations>(f);
  Real m0 = 0, p0[3] = {0, 0, 0};
  for (int i = 0; i < 27; ++i) {
    m0 += f[i]; p0[0] += f[i] * Real(D3Q27::cx(i));
    p0[1] += f[i] * Real(D3Q27::cy(i)); p0[2] += f[i] * Real(D3Q27::cz(i));
  }
  const Macro m = c.macroscopic(f, 0);
  c.collide(f, m, 0);
  Real m1 = 0, p1[3] = {0, 0, 0};
  for (int i = 0; i < 27; ++i) {
    m1 += f[i]; p1[0] += f[i] * Real(D3Q27::cx(i));
    p1[1] += f[i] * Real(D3Q27::cy(i)); p1[2] += f[i] * Real(D3Q27::cz(i));
  }
  const std::string n = "MhdCM/D3Q27 omega=" + std::to_string(double(omega));
  check::near(m1, m0, TOL(), n + ": conserves mass");
  const char* ax = "xyz";
  for (int a = 0; a < 3; ++a)
    check::near(p1[a], p0[a], TOL(), n + ": conserves " + ax[a] + "-momentum");
}


//------------------------------------------------------------------------------
// The higher-order equilibria taken from the MATLAB generators -- product form
// (6th order) on D3Q27, 4th order on D3Q19, product form (4th) on D2Q9 -- must
// be Maxwellian in every REPRESENTABLE central moment.
//------------------------------------------------------------------------------
template <class L>
void high_order_eq_is_maxwellian(Real rho, Real ux, Real uy, Real uz) {
  using Eq = HighOrderEquilibrium<L>;
  using B  = typename SelectBasis<L>::type;
  constexpr Real cs2v = cs2<L, Real>();
  const Real Mx[3] = {Real(1), Real(0), cs2v};
  const Real u[3] = {ux, uy, uz};
  Real f[L::Q];
  for (int i = 0; i < L::Q; ++i) f[i] = Eq::eq(i, rho, ux, uy, uz);
  double worst = 0;
  for (int n = 0; n < B::NM; ++n) {
    const int pq[3] = {B::p_of(n), B::q_of(n), B::r_of(n)};
    Real got = 0;
    for (int i = 0; i < L::Q; ++i) {
      Real t = f[i];
      for (int a = 0; a < L::D; ++a)
        for (int e = 0; e < pq[a]; ++e) t *= (Real(cvel<L>(i, a)) - u[a]);
      got += t;
    }
    Real want = rho * Mx[pq[0]] * Mx[pq[1]];
    if constexpr (L::D == 3) want *= Mx[pq[2]];
    worst = std::max(worst, std::abs(double(got - want)));
  }
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string(L::name) + "/" + Eq::name +
            ": central moments are Maxwellian in all " + std::to_string(B::NM) + buf);
}

// Stronger statement: the representable central moments determine the populations
// uniquely, so an equilibrium that is Maxwellian in all of them must BE the one
// the moment collision builds in moment space. This checks the transcription of
// the MATLAB expressions population by population, not just moment by moment.
template <class L, class Store>
void high_order_eq_matches_moment_space(Real rho, Real ux, Real uy, Real uz) {
  using Eq = HighOrderEquilibrium<L>;
  using C  = MomentCollision<L, NoForcing, Store, true>;
  double worst = 0;
  for (int i = 0; i < L::Q; ++i) {
    Real got = C::seed_value(i, rho, ux, uy, uz);
    if constexpr (Store::shifted) got += weight<L, Real>(i);
    worst = std::max(worst, std::abs(double(got - Eq::eq(i, rho, ux, uy, uz))));
  }
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string(L::name) + "/" + Store::name + ": " + Eq::name +
            " == the moment-space equilibrium" + buf);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const Real r = Real(1.03), ux = Real(0.031), uy = Real(-0.017), uz = Real(0.022);
    const Real F[3] = {Real(1.7e-5), Real(-9e-6), Real(4e-6)};

    round_trip<D2Q9>(0, 0, 0);
    round_trip<D2Q9>(ux, uy, 0);
    round_trip<D3Q19>(0, 0, 0);
    round_trip<D3Q19>(ux, uy, uz);
    round_trip<D3Q27>(0, 0, 0);
    round_trip<D3Q27>(ux, uy, uz);
    against_direct<D2Q9>(ux, uy, 0);
    against_direct<D3Q19>(ux, uy, uz);
    against_direct<D3Q27>(ux, uy, uz);

    equilibrium_is_maxwellian<D2Q9,  RawPopulations>(r, ux, uy, 0);
    equilibrium_is_maxwellian<D2Q9,  ShiftedPopulations>(r, ux, uy, 0);
    equilibrium_is_maxwellian<D3Q19, RawPopulations>(r, ux, uy, uz);
    equilibrium_is_maxwellian<D3Q19, ShiftedPopulations>(r, ux, uy, uz);
    equilibrium_is_maxwellian<D3Q27, RawPopulations>(r, ux, uy, uz);
    equilibrium_is_maxwellian<D3Q27, ShiftedPopulations>(r, ux, uy, uz);

    for (Real w : {Real(0.4), Real(1.0), Real(1.8)}) {
      { MomentCollision<D2Q9,  NoForcing, RawPopulations, true>  c; c.omega = w;
        conservation<decltype(c), D2Q9,  RawPopulations>("CM", c); }
      { MomentCollision<D3Q27, NoForcing, RawPopulations, true>  c; c.omega = w;
        conservation<decltype(c), D3Q27, RawPopulations>("CM", c); }
      { MomentCollision<D3Q27, NoForcing, ShiftedPopulations, true> c; c.omega = w;
        conservation<decltype(c), D3Q27, ShiftedPopulations>("CM", c); }
      { MomentCollision<D3Q27, NoForcing, RawPopulations, false> c; c.omega = w;
        conservation<decltype(c), D3Q27, RawPopulations>("MRT", c); }
      { MomentCollision<D3Q19, NoForcing, ShiftedPopulations, true>  c; c.omega = w;
        conservation<decltype(c), D3Q19, ShiftedPopulations>("CM", c); }
      { MomentCollision<D3Q19, NoForcing, RawPopulations, false> c; c.omega = w;
        conservation<decltype(c), D3Q19, RawPopulations>("MRT", c); }
      { TRT<D3Q19, SecondOrderEquilibrium<D3Q19>, NoForcing, RawPopulations> c;
        c.omega_p = w; c.omega_m = TRT<D3Q19>::omega_minus_for(w, TRT<D3Q19>::magic_3_16);
        conservation<decltype(c), D3Q19, RawPopulations>("TRT", c); }

      { MomentCollision<D2Q9, Guo, RawPopulations, true> c; c.omega = w;
        c.forcing = Guo{F[0], F[1], Real(0)};
        const Real F2[3] = {F[0], F[1], Real(0)};
        forced_momentum<decltype(c), D2Q9, RawPopulations>("CM", c, F2); }
      { MomentCollision<D3Q27, Guo, ShiftedPopulations, true> c; c.omega = w;
        c.forcing = Guo{F[0], F[1], F[2]};
        forced_momentum<decltype(c), D3Q27, ShiftedPopulations>("CM", c, F); }
      { MomentCollision<D3Q27, Guo, RawPopulations, false> c; c.omega = w;
        c.forcing = Guo{F[0], F[1], F[2]};
        forced_momentum<decltype(c), D3Q27, RawPopulations>("MRT", c, F); }
      { MomentCollision<D3Q19, Guo, ShiftedPopulations, true> c; c.omega = w;
        c.forcing = Guo{F[0], F[1], F[2]};
        forced_momentum<decltype(c), D3Q19, ShiftedPopulations>("CM", c, F); }
      { MomentCollision<D3Q19, Guo, RawPopulations, false> c; c.omega = w;
        c.forcing = Guo{F[0], F[1], F[2]};
        forced_momentum<decltype(c), D3Q19, RawPopulations>("MRT", c, F); }
      { TRT<D3Q19, SecondOrderEquilibrium<D3Q19>, Guo> c;
        c.omega_p = w; c.omega_m = TRT<D3Q19>::omega_minus_for(w, TRT<D3Q19>::magic_3_16);
        c.forcing = Guo{F[0], F[1], F[2]};
        forced_momentum<decltype(c), D3Q19, RawPopulations>("TRT", c, F); }
    }

    full_relaxation<D2Q9,  RawPopulations,     true>();
    full_relaxation<D2Q9,  ShiftedPopulations, true>();
    full_relaxation<D3Q27, RawPopulations,     true>();
    full_relaxation<D3Q27, ShiftedPopulations, true>();
    full_relaxation<D3Q27, RawPopulations,     false>();
    full_relaxation<D3Q19, RawPopulations,     true>();
    full_relaxation<D3Q19, ShiftedPopulations, true>();
    full_relaxation<D3Q19, RawPopulations,     false>();

    cm_equals_mrt_at_rest<D2Q9,  RawPopulations>();
    cm_equals_mrt_at_rest<D3Q27, ShiftedPopulations>();
    cm_equals_mrt_at_rest<D3Q19, RawPopulations>();
    cm_equals_mrt_at_rest<D3Q19, ShiftedPopulations>();

    mhd_cm_equilibrium(Real(1.07), Real(0.031), Real(-0.017), Real(0.022), Real(-0.033));
    mhd_cm_equilibrium(Real(1.0),  Real(0.0),   Real(0.0),    Real(0.05), Real(0.0));
    for (Real w : {Real(0.6), Real(1.2), Real(1.9)}) mhd_cm_conservation(w);

    {
      const Real u1[3] = {Real(0.031), Real(-0.017), Real(0.022)};
      const Real b1[3] = {Real(0.022), Real(-0.033), Real(0.011)};
      const Real z[3]  = {Real(0), Real(0), Real(0)};
      mhd_cm3d_moments<true>(Real(1.07), u1, b1);
      mhd_cm3d_moments<true>(Real(1.0),  z,  b1);   // b alone, no flow
      mhd_cm3d_moments<true>(Real(0.98), u1, z);    // flow alone, no field
      mhd_cm3d_moments<false>(Real(1.07), u1, b1);  // the second-order form too
      mhd_cm3d_moments<false>(Real(0.98), u1, z);
    }
    mhd_cm3d_reduces_to_2d<true>(Real(1.07), Real(0.031), Real(-0.017), Real(0.022), Real(-0.033));
    mhd_cm3d_reduces_to_2d<false>(Real(1.07), Real(0.031), Real(-0.017), Real(0.022), Real(-0.033));
    for (Real w : {Real(0.6), Real(1.2), Real(1.9)}) {
      mhd_cm3d_conservation<true>(w); mhd_cm3d_conservation<false>(w);
    }

    // higher-order equilibria: D3Q27 to 6th order, D3Q19 to 4th, D2Q9 to 4th
    high_order_eq_is_maxwellian<D2Q9>(r, ux, uy, 0);
    high_order_eq_is_maxwellian<D3Q19>(r, ux, uy, uz);
    high_order_eq_is_maxwellian<D3Q27>(r, ux, uy, uz);
    high_order_eq_is_maxwellian<D3Q27>(r, Real(0.2), Real(-0.15), Real(0.1));
    high_order_eq_matches_moment_space<D2Q9,  RawPopulations>(r, ux, uy, 0);
    high_order_eq_matches_moment_space<D3Q19, RawPopulations>(r, ux, uy, uz);
    high_order_eq_matches_moment_space<D3Q27, RawPopulations>(r, ux, uy, uz);
    high_order_eq_matches_moment_space<D3Q27, ShiftedPopulations>(r, ux, uy, uz);

    // TRT magic-parameter round trip
    for (Real w : {Real(0.6), Real(1.2), Real(1.9)}) {
      const Real wm = TRT<D2Q9>::omega_minus_for(w, TRT<D2Q9>::magic_3_16);
      check::near(TRT<D2Q9>::magic_parameter(w, wm), TRT<D2Q9>::magic_3_16, TOL(),
                  "TRT: omega_minus realises Lambda = 3/16");
    }
  }
  const int r = check::report("moments");
  Kokkos::finalize();
  return r;
}
