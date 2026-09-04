//==============================================================================
//  Collision invariants, for both storage conventions.
//    - BGK conserves mass and momentum exactly, at any omega, for any f.
//    - With Guo forcing, momentum increases by exactly F per collision:
//         sum_i c_i f_i^post  ==  sum_i c_i f_i^pre + F
//      and mass is still conserved. That pins the forcing scheme down: get the
//      half-velocity shift or the (1 - omega/2) prefactor wrong and it fails.
//    - Raw and shifted storage produce the same physics.
//    - The regularised scalar operator conserves the scalar, is IDENTICAL to
//      ScalarBGK at omega = 1, leaves the same scalar and flux moments as BGK
//      at any omega, and differs from it only in the ghosts. Those four
//      together say it is the same physics with a different ghost treatment,
//      which is exactly the claim ScalarRegularised.hpp makes.
//==============================================================================
#include "Check.hpp"
#include "collision/BGK.hpp"
#include "collision/ScalarRegularised.hpp"
#include "core/Types.hpp"

#include <cmath>
#include <string>

using namespace lbm;

static Real TOL() { return sizeof(Real) == 4 ? Real(5e-6) : Real(5e-14); }

// Deterministic, non-equilibrium, strictly positive populations (raw form).
template <class L>
static void fill_raw(Real f[L::Q]) {
  for (int i = 0; i < L::Q; ++i)
    f[i] = weight<L, Real>(i) * (Real(1) + Real(0.13) * Real((i * 7) % 5 - 2) +
                                 Real(0.05) * Real(i % 3));
}
// Same state, expressed in whichever variable Store uses.
template <class L, class Store>
static void fill(Real f[L::Q]) {
  fill_raw<L>(f);
  if constexpr (Store::shifted)
    for (int i = 0; i < L::Q; ++i) f[i] -= weight<L, Real>(i);
}

template <class L, class Store>
void conservation(Real omega) {
  const std::string n = std::string(L::name) + "/" + Store::name +
                        " omega=" + std::to_string(double(omega));
  Real f[L::Q]; fill<L, Store>(f);
  Real m0 = 0, p0[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m0 += f[i];
    for (int a = 0; a < 3; ++a) p0[a] += f[i] * Real(cvel<L>(i, a));
  }
  BGK<L, SecondOrderEquilibrium<L>, NoForcing, Store> bgk;
  bgk.omega = omega;
  const Macro m = bgk.macroscopic(f);
  bgk.collide(f, m);

  Real m1 = 0, p1[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m1 += f[i];
    for (int a = 0; a < 3; ++a) p1[a] += f[i] * Real(cvel<L>(i, a));
  }
  check::near(m1, m0, TOL(), n + ": BGK conserves mass");
  for (int a = 0; a < L::D; ++a)
    check::near(p1[a], p0[a], TOL(), n + ": BGK conserves momentum");
}

template <class L, class Store>
void forced_momentum(Real omega) {
  const std::string n = std::string(L::name) + "/" + Store::name +
                        " omega=" + std::to_string(double(omega));
  const Real F[3] = {Real(1.7e-5), Real(-9e-6), L::D == 3 ? Real(4e-6) : Real(0)};

  Real f[L::Q]; fill<L, Store>(f);
  Real m0 = 0, p0[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m0 += f[i];
    for (int a = 0; a < 3; ++a) p0[a] += f[i] * Real(cvel<L>(i, a));
  }

  BGK<L, SecondOrderEquilibrium<L>, Guo, Store> bgk;
  bgk.omega   = omega;
  bgk.forcing = Guo{F[0], F[1], F[2]};
  const Macro m = bgk.macroscopic(f);
  bgk.collide(f, m);

  Real m1 = 0, p1[3] = {0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    m1 += f[i];
    for (int a = 0; a < 3; ++a) p1[a] += f[i] * Real(cvel<L>(i, a));
  }
  check::near(m1, m0, TOL(), n + ": Guo forcing conserves mass");
  for (int a = 0; a < L::D; ++a)
    check::near(p1[a], p0[a] + F[a], TOL(),
                n + ": Guo forcing adds exactly F to momentum");
}

// The two storage conventions must describe the same physical state and the
// same post-collision state.
template <class L>
void storage_equivalence(Real omega) {
  const std::string n = std::string(L::name);
  Real fr[L::Q], fs[L::Q];
  fill<L, RawPopulations>(fr);
  fill<L, ShiftedPopulations>(fs);

  BGK<L, SecondOrderEquilibrium<L>, Guo, RawPopulations> br;
  BGK<L, SecondOrderEquilibrium<L>, Guo, ShiftedPopulations> bs;
  br.omega = bs.omega = omega;
  br.forcing = bs.forcing = Guo{Real(1.7e-5), Real(-9e-6), Real(0)};

  const Macro mr = br.macroscopic(fr);
  const Macro ms = bs.macroscopic(fs);
  check::near(br.density(mr), bs.density(ms), TOL(), n + ": same rho from both storages");
  check::near(mr.ux, ms.ux, TOL(), n + ": same ux from both storages");
  check::near(mr.uy, ms.uy, TOL(), n + ": same uy from both storages");

  br.collide(fr, mr);
  bs.collide(fs, ms);
  double worst = 0;
  for (int i = 0; i < L::Q; ++i)
    worst = std::max(worst, std::abs(double(fr[i] - weight<L, Real>(i)) - double(fs[i])));
  check::ok(worst <= double(TOL()), n + ": same post-collision state from both storages");
}

template <class L>
void viscosity_roundtrip() {
  for (Real nu : {Real(1e-3), Real(0.05), Real(0.3)}) {
    const Real w = BGK<L>::omega_from_viscosity(nu);
    check::near(BGK<L>::viscosity_from_omega(w), nu, TOL(),
                std::string(L::name) + ": omega <-> nu round-trip");
  }
}

//------------------------------------------------------------------------------
//  THE REGULARISED SCALAR, against ScalarBGK on the same state.
//
//  The ghost moments on these lattices are the AXIAL SECOND MOMENTS
//  M_a = sum_i h_i c_ia^2. Their equilibrium is cs2 * dT, so "annihilated" and
//  "at equilibrium" are the same statement, and the check below is that BGK
//  does NOT put them there while this operator does.
//------------------------------------------------------------------------------
template <class L>
static void scalar_regularised(Real w) {
  const std::string n = std::string(L::name) + " w=" + std::to_string(double(w));
  const Real dT = Real(0.37), ux = Real(0.031), uy = Real(-0.017), uz = Real(0.009);

  ScalarBGK<L>         bgk;   bgk.omega = w;   bgk.T_ref = Real(0.5);
  ScalarRegularised<L> reg;   reg.omega = w;   reg.T_ref = Real(0.5);

  Real hb[L::Q], hr[L::Q];
  for (int i = 0; i < L::Q; ++i)
    hb[i] = hr[i] = weight<L, Real>(i) * dT +
                    Real(0.021) * Real((i * 5) % 4 - 1);   // off equilibrium
  const Real dT0 = ScalarBGK<L>::deviation(hb);

  bgk.collide(hb, dT0, ux, uy, uz, w);
  reg.collide(hr, dT0, ux, uy, uz, w);

  // 1. the scalar is conserved, exactly
  check::near(ScalarBGK<L>::deviation(hr), dT0, TOL(),
              n + ": regularised conserves the scalar");

  // 2. and the flux moments agree with BGK's -- the physical content is the same
  for (int a = 0; a < L::D; ++a) {
    Real jb = Real(0), jr = Real(0);
    for (int i = 0; i < L::Q; ++i) {
      jb += hb[i] * Real(cvel<L>(i, a));
      jr += hr[i] * Real(cvel<L>(i, a));
    }
    check::near(jr, jb, TOL(), n + ": same flux moment " + std::to_string(a));
  }

  // 3. the ghosts are at equilibrium, which BGK's are not (except at w = 1)
  const Real d_eq = cs2<L, Real>() * dT0;
  for (int a = 0; a < L::D; ++a) {
    Real Mb = Real(0), Mr = Real(0);
    for (int i = 0; i < L::Q; ++i) {
      const Real c2 = Real(cvel<L>(i, a)) * Real(cvel<L>(i, a));
      Mb += hb[i] * c2;  Mr += hr[i] * c2;
    }
    check::near(Mr, d_eq, TOL(), n + ": ghost " + std::to_string(a) + " annihilated");
    if (w != Real(1))
      check::ok(std::abs(double(Mb - d_eq)) > 1e-4,
                n + ": BGK leaves ghost " + std::to_string(a) + " OFF equilibrium");
  }

  // 4. at omega = 1 the two operators are the same function, population by
  //    population. If this ever fails the two are not the scheme they claim.
  if (w == Real(1)) {
    double worst = 0.0;
    for (int i = 0; i < L::Q; ++i)
      worst = std::max(worst, std::abs(double(hb[i]) - double(hr[i])));
    check::ok(worst <= double(TOL()),
              n + ": identical to BGK at omega = 1 (worst " +
                  std::to_string(worst) + ")");
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    for (Real w : {Real(0.4), Real(1.0), Real(1.8)}) {
      conservation<D2Q9,  RawPopulations>(w);
      conservation<D3Q19, RawPopulations>(w);
      conservation<D3Q27, RawPopulations>(w);
      conservation<D2Q9,  ShiftedPopulations>(w);
      conservation<D3Q19, ShiftedPopulations>(w);
      forced_momentum<D2Q9,  RawPopulations>(w);
      forced_momentum<D3Q19, RawPopulations>(w);
      forced_momentum<D2Q9,  ShiftedPopulations>(w);
      forced_momentum<D3Q19, ShiftedPopulations>(w);
      storage_equivalence<D2Q9>(w);
      storage_equivalence<D3Q19>(w);
    }
    for (Real w : {Real(0.4), Real(1.0), Real(1.8), Real(1.997)}) {
      scalar_regularised<D3Q7>(w);
      scalar_regularised<D2Q5>(w);
    }
    viscosity_roundtrip<D2Q9>();
    viscosity_roundtrip<D3Q19>();
  }
  const int r = check::report("collision");
  Kokkos::finalize();
  return r;
}
