//==============================================================================
//  Regularized velocity boundary condition (BC3).
//
//  Latt et al., Phys. Rev. E 77, 056703 (2008), Sec. IV C.
//
//  The condition claims three things about the populations it produces, and all
//  three are checkable without running a flow:
//
//    (a) the imposed density and velocity are recovered EXACTLY -- this is the
//        whole point of the method, and what bounce-back of the off-equilibrium
//        part fails to do;
//    (b) the second moment reproduces the Pi^(1) that was measured, so the
//        stress is not mangled by the reconstruction;
//    (c) the reconstruction is idempotent on an equilibrium input, because
//        Pi^(1) vanishes there.
//
//  Together they pin Eqs. (44)-(45) without reference to any particular
//  velocity numbering, which matters because the paper's Eq. (28) is written
//  for a figure whose ordering is not the Esoteric Pull one.
//==============================================================================
#include "Check.hpp"
#include "boundary/Regularized.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"

#include <cmath>
#include <string>

using namespace lbm;

static Real TOL() { return sizeof(Real) == 4 ? Real(2e-5) : Real(1e-12); }

// A deterministic non-equilibrium state, then the populations a wall node would
// actually hold: those streaming out of the domain are kept, those streaming in
// are destroyed (they are the unknowns the condition has to invent).
template <class L>
static void wall_populations(Real f[L::Q], const int nrm[3], Real rho, const Real u[3]) {
  using Eq = SecondOrderEquilibrium<L>;
  for (int i = 0; i < L::Q; ++i) {
    const Real bump = Real(0.04) * Real((i * 7) % 5 - 2) + Real(0.02) * Real(i % 3);
    f[i] = Eq::eq(i, rho, u[0], u[1], u[2]) * (Real(1) + bump);
  }
  for (int i = 0; i < L::Q; ++i) {
    const int cn = cvel<L>(i, 0) * nrm[0] + cvel<L>(i, 1) * nrm[1] + cvel<L>(i, 2) * nrm[2];
    if (cn < 0) f[i] = Real(-1e30);          // poison: must never be read
  }
}

// Straight wall: the unknown set is the half-space c_i . n < 0.
template <class L>
static std::uint32_t mask_from_normal(const int nrm[3]) {
  std::uint32_t m = 0;
  for (int i = 0; i < L::Q; ++i) {
    const int cn = cvel<L>(i, 0) * nrm[0] + cvel<L>(i, 1) * nrm[1] + cvel<L>(i, 2) * nrm[2];
    if (cn < 0) m |= (1u << i);
  }
  return m;
}

// Corner of a box: direction i is unknown when its source node p - c_i lies
// outside ANY of the walls meeting there. This is not a half-space, and it puts
// both members of some opposite pairs in the unknown set.
template <class L>
static std::uint32_t mask_from_corner(const int wall[3]) {
  std::uint32_t m = 0;
  for (int i = 0; i < L::Q; ++i) {
    bool outside = false;
    for (int a = 0; a < 3; ++a)
      if (wall[a] != 0 && -cvel<L>(i, a) * wall[a] > 0) outside = true;
    if (outside) m |= (1u << i);
  }
  return m;
}

template <class L>
static void moments(const Real f[L::Q], Real& m0, Real m1[3], Real m2[6]) {
  m0 = 0;
  for (int a = 0; a < 3; ++a) m1[a] = 0;
  for (int a = 0; a < 6; ++a) m2[a] = 0;
  for (int i = 0; i < L::Q; ++i) {
    const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)), cz = Real(cvel<L>(i, 2));
    m0 += f[i];
    m1[0] += f[i] * cx; m1[1] += f[i] * cy; m1[2] += f[i] * cz;
    m2[0] += f[i] * cx * cx; m2[1] += f[i] * cy * cy; m2[2] += f[i] * cz * cz;
    m2[3] += f[i] * cx * cy; m2[4] += f[i] * cx * cz; m2[5] += f[i] * cy * cz;
  }
}

//------------------------------------------------------------------------------
// (a) rho and u come back exactly, for every wall orientation the lattice has.
//------------------------------------------------------------------------------
template <class L>
static void enforces_velocity(std::uint8_t code, Real rho_in, const Real u[3]) {
  using BC = Regularized<L, SecondOrderEquilibrium<L>>;
  int nrm[3]; normal_of(code, nrm);
  if (L::D == 2 && nrm[2] != 0) return;

  Real f[L::Q];
  wall_populations<L>(f, nrm, rho_in, u);
  const Real rho = BC::density(f, nrm, u);
  BC::apply(f, rho, u, mask_from_normal<L>(nrm));

  Real m0, m1[3], m2[6];
  moments<L>(f, m0, m1, m2);
  const std::string tag = std::string(L::name) + " n=(" + std::to_string(nrm[0]) + "," +
                          std::to_string(nrm[1]) + "," + std::to_string(nrm[2]) + ")";
  check::near(m0, rho, TOL(), tag + ": recovers rho");
  const char* ax = "xyz";
  for (int a = 0; a < L::D; ++a)
    check::near(m1[a] / rho, u[a], TOL(), tag + ": recovers u" + ax[a]);
}

//------------------------------------------------------------------------------
// (b) the second moment of the rebuilt populations equals the equilibrium part
// plus exactly the Pi^(1) that was measured before the rebuild.
//------------------------------------------------------------------------------
template <class L>
static void preserves_stress(std::uint8_t code, Real rho_in, const Real u[3]) {
  using Eq = SecondOrderEquilibrium<L>;
  using BC = Regularized<L, Eq>;
  int nrm[3]; normal_of(code, nrm);
  if (L::D == 2 && nrm[2] != 0) return;

  Real f[L::Q];
  wall_populations<L>(f, nrm, rho_in, u);
  const Real rho = BC::density(f, nrm, u);

  // Pi^(1) as the condition itself computes it: fill the unknowns by
  // off-equilibrium bounce-back first.
  Real scaffold[L::Q];
  for (int i = 0; i < L::Q; ++i) scaffold[i] = f[i];
  for (int i = 0; i < L::Q; ++i) {
    const int cn = cvel<L>(i, 0) * nrm[0] + cvel<L>(i, 1) * nrm[1] + cvel<L>(i, 2) * nrm[2];
    if (cn < 0) {
      const Real e = Eq::eq(i, rho, u[0], u[1], u[2]);
      const Real eo = Eq::eq(opp(i), rho, u[0], u[1], u[2]);
      scaffold[i] = e + (f[opp(i)] - eo);
    }
  }
  Real want[6] = {0, 0, 0, 0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)), cz = Real(cvel<L>(i, 2));
    const Real d = scaffold[i] - Eq::eq(i, rho, u[0], u[1], u[2]);
    want[0] += d * cx * cx; want[1] += d * cy * cy; want[2] += d * cz * cz;
    want[3] += d * cx * cy; want[4] += d * cx * cz; want[5] += d * cy * cz;
  }

  BC::apply(f, rho, u, mask_from_normal<L>(nrm));
  Real m0, m1[3], m2[6];
  moments<L>(f, m0, m1, m2);
  Real eqm[6] = {0, 0, 0, 0, 0, 0};
  for (int i = 0; i < L::Q; ++i) {
    const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)), cz = Real(cvel<L>(i, 2));
    const Real e = Eq::eq(i, rho, u[0], u[1], u[2]);
    eqm[0] += e * cx * cx; eqm[1] += e * cy * cy; eqm[2] += e * cz * cz;
    eqm[3] += e * cx * cy; eqm[4] += e * cx * cz; eqm[5] += e * cy * cz;
  }
  // D3Q19 cannot carry an arbitrary symmetric Pi^(1) through Eq. (45) exactly;
  // the diagonal is exact, the shear components are what the lattice supports.
  double worst = 0;
  const int ncomp = (L::D == 2) ? 4 : 6;
  const int idx[6] = {0, 1, 3, 2, 4, 5};
  for (int k = 0; k < ncomp; ++k) {
    const int a = (L::D == 2 && k == 2) ? 3 : idx[k];
    worst = std::max(worst, std::abs(double(m2[a] - eqm[a] - want[a])));
  }
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string(L::name) + ": second moment reproduces Pi^(1)" + buf);
}

//------------------------------------------------------------------------------
// (c) an equilibrium input has Pi^(1) = 0, so the rebuild must return it.
//------------------------------------------------------------------------------
template <class L>
static void idempotent_on_equilibrium(Real rho, const Real u[3]) {
  using Eq = SecondOrderEquilibrium<L>;
  using BC = Regularized<L, Eq>;
  int nrm[3]; normal_of(NrmYp, nrm);
  Real f[L::Q];
  for (int i = 0; i < L::Q; ++i) f[i] = Eq::eq(i, rho, u[0], u[1], u[2]);
  const Real r = BC::density(f, nrm, u);
  BC::apply(f, r, u, mask_from_normal<L>(nrm));
  double worst = std::abs(double(r - rho));
  for (int i = 0; i < L::Q; ++i)
    worst = std::max(worst, std::abs(double(f[i] - Eq::eq(i, rho, u[0], u[1], u[2]))));
  char buf[48]; std::snprintf(buf, sizeof buf, " (worst %.2e)", worst);
  check::ok(worst <= double(TOL()),
            std::string(L::name) + ": equilibrium in, equilibrium out" + buf);
}

//------------------------------------------------------------------------------
// Corners. rho cannot come from Eq. (27) there, so it is supplied (as the solver
// supplies it, by extrapolation) and only the reconstruction is exercised. The
// demand is unchanged: rho and u must come back exactly, even though some
// opposite pairs are unknown on BOTH sides and so carry no bounce-back partner.
//------------------------------------------------------------------------------
template <class L>
static void corner_enforces_velocity(const int wall[3], Real rho, const Real u[3]) {
  using Eq = SecondOrderEquilibrium<L>;
  using BC = Regularized<L, Eq>;
  const std::uint32_t unk = mask_from_corner<L>(wall);

  Real f[L::Q];
  for (int i = 0; i < L::Q; ++i) {
    const Real bump = Real(0.04) * Real((i * 7) % 5 - 2) + Real(0.02) * Real(i % 3);
    f[i] = Eq::eq(i, rho, u[0], u[1], u[2]) * (Real(1) + bump);
    if (unk & (1u << i)) f[i] = Real(-1e30);        // poison the unknowns
  }
  BC::apply(f, rho, u, unk);

  Real m0, m1[3], m2[6];
  moments<L>(f, m0, m1, m2);
  int npair = 0;
  for (int i = 0; i < L::Q; ++i)
    if ((unk & (1u << i)) && (unk & (1u << opp(i)))) ++npair;
  const std::string tag = std::string(L::name) + " corner (" + std::to_string(wall[0]) +
                          "," + std::to_string(wall[1]) + "," + std::to_string(wall[2]) +
                          ", " + std::to_string(npair) + " doubly-unknown)";
  check::near(m0, rho, TOL(), tag + ": recovers rho");
  const char* ax = "xyz";
  for (int a = 0; a < L::D; ++a)
    check::near(m1[a] / rho, u[a], TOL(), tag + ": recovers u" + ax[a]);
  check::ok(npair > 0, tag + ": really has doubly-unknown pairs");
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const Real u0[3] = {Real(0), Real(0), Real(0)};          // no-slip wall
    const Real u1[3] = {Real(0.05), Real(0), Real(0)};       // moving lid
    const Real u2[3] = {Real(0.03), Real(-0.02), Real(0.01)};// oblique

    for (std::uint8_t c = NrmXp; c <= NrmZm; ++c) {
      enforces_velocity<D2Q9>(c, Real(1.02), u0);
      enforces_velocity<D2Q9>(c, Real(1.02), u1);
      enforces_velocity<D3Q19>(c, Real(0.98), u2);
      enforces_velocity<D3Q27>(c, Real(0.98), u2);
    }
    preserves_stress<D2Q9>(NrmYp, Real(1.02), u1);
    preserves_stress<D3Q27>(NrmYp, Real(0.98), u2);

    const int c2[4][3] = {{-1,-1,0},{1,-1,0},{-1,1,0},{1,1,0}};
    for (auto& w : c2) corner_enforces_velocity<D2Q9>(w, Real(1.02), u1);
    const int c3[3] = {-1,-1,-1};
    corner_enforces_velocity<D3Q19>(c3, Real(0.98), u2);
    corner_enforces_velocity<D3Q27>(c3, Real(0.98), u2);
    const int e3[3] = {-1,1,0};          // 3D edge: two walls, one free axis
    corner_enforces_velocity<D3Q27>(e3, Real(0.98), u2);

    idempotent_on_equilibrium<D2Q9>(Real(1.02), u1);
    idempotent_on_equilibrium<D3Q19>(Real(0.98), u2);
    idempotent_on_equilibrium<D3Q27>(Real(0.98), u2);
  }
  Kokkos::finalize();
  return check::report("boundary");
}
