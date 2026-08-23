#pragma once
//==============================================================================
//  Regularized velocity boundary condition (BC3).
//
//  Latt, Chopard, Malaspinas, Deville & Michler, "Straight velocity boundaries
//  in the lattice Boltzmann method", Phys. Rev. E 77, 056703 (2008), Sec. IV C,
//  Eqs. (44)-(45); originally Latt & Chopard.
//
//  The node's populations are ALL replaced. The algorithm is
//
//    1. rho from the known populations, Eq. (27);
//    2. the unknown populations are given, TEMPORARILY, the value produced by a
//       bounce-back of their off-equilibrium part;
//    3. Pi^(1) = sum_i c_i c_i (f_i - f_i^eq) is evaluated from that;
//    4. every population is rebuilt as
//
//          g_i = f_i^eq(rho, u) + (w_i / 2 cs^4) Q_i : Pi^(1),
//          Q_i = c_i c_i - cs^2 I.                             Eq. (45)
//
//  Step 2 is a scaffold only: the paper is explicit that bounce-back of the
//  off-equilibrium part may NOT be used as the boundary condition itself,
//  because it cannot enforce the velocity exactly. It exists solely to give
//  Pi^(1) a value.
//
//  Note what Eq. (45) does NOT contain: omega. The reconstruction is therefore
//  independent of the collision operator, and the same code serves BGK, TRT,
//  MRT and the central-moment operators.
//
//  Wall placement. Halfway bounce-back puts the wall midway between the last
//  fluid node and the first solid node. This condition puts the wall ON the
//  boundary node, because that is the node whose velocity is imposed. A channel
//  of H fluid nodes is therefore H-1 lattice units wide here, not H.
//
//  Density closure. The paper's Eq. (28) is written for one particular figure
//  and velocity numbering. It is implemented here geometrically instead --
//  known populations are those with c_i . n > 0, n being the OUTWARD normal --
//  which is numbering-independent and so survives the Esoteric Pull ordering.
//==============================================================================
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "lattice/Lattices.hpp"

#include <type_traits>

namespace lbm {

//------------------------------------------------------------------------------
// Which equilibrium a collision operator reconstructs against. The moment-space
// operators do not expose one -- they build their equilibrium as moments -- but
// that equilibrium is provably identical to HighOrderEquilibrium<L> (see the
// moment tests), so it is the correct fallback rather than a guess.
//------------------------------------------------------------------------------
template <class C, class L, class = void>
struct EquilibriumOf { using type = HighOrderEquilibrium<L>; };
template <class C, class L>
struct EquilibriumOf<C, L, std::void_t<typename C::Equilibrium>> {
  using type = typename C::Equilibrium;
};

//------------------------------------------------------------------------------
// The SHEAR relaxation rate of a collision operator -- the one that sets the
// viscosity, nu = cs^2 (1/omega - 1/2).
//
// Only the finite-difference corner route needs this. That route infers the
// stress from a measured strain rate through Eq. (21),
//     Pi^(1) = -(2 cs^2 / omega) rho S,
// whose coefficient IS the viscosity; the reconstruction Eq. (45) itself has no
// omega in it. Naming the wrong rate makes the boundary impose a different
// viscosity than the bulk, which surfaces as spurious slip -- hence TRT's even
// rate omega_p rather than omega_m, and the deviatoric rate for the moment
// operators.
//------------------------------------------------------------------------------
template <class C>
KOKKOS_INLINE_FUNCTION Real shear_omega(const C& c) {
  if constexpr (requires { c.omega; })        return c.omega;
  else if constexpr (requires { c.omega_p; }) return c.omega_p;
  else                                        return Real(1);
}
template <class C>
inline constexpr bool has_shear_omega =
    requires(const C& c) { c.omega; } || requires(const C& c) { c.omega_p; };

// Whether a collision operator carries an active body force. The regularised
// reconstruction has to know: under Guo forcing the physical velocity carries a
// half-force term that the raw first moment does not.
template <class C>
inline constexpr bool has_forcing =
    requires { typename C::ForcingPolicy; } &&
    []{ if constexpr (requires { typename C::ForcingPolicy; })
          return C::ForcingPolicy::active; else return false; }();

//------------------------------------------------------------------------------
// Outward normal codes. `Corner` marks a node where two or more walls meet: the
// closure Eq. (27) cannot be evaluated there (it needs a single normal), so rho
// is extrapolated from neighbours instead and only the reconstruction is run.
//------------------------------------------------------------------------------
enum NormalCode : std::uint8_t {
  NrmNone = 0,
  NrmXp   = 1, NrmXm = 2,
  NrmYp   = 3, NrmYm = 4,
  NrmZp   = 5, NrmZm = 6,
  NrmCorner = 7,
  // Constant back-pressure outlet on the +x face. rho is IMPOSED (WallSpec::rho)
  // and the normal velocity is read off the same closure that gives rho on a
  // velocity wall, inverted; the tangential components are zero-gradient from
  // the upstream neighbour. The node is then reconstructed exactly like a
  // velocity wall.
  //
  // rho MUST be imposed here, not extrapolated. With a velocity inlet and a
  // zero-gradient outlet nothing anchors the pressure: the boundary nodes are
  // not mass conserving -- they overwrite populations -- so the mass the inlet
  // injects has no way out and rho drifts without bound. Measured: a
  // zero-gradient outlet on this channel settles at rho ~ 181 with a density
  // ramp along x and a centreline velocity 2x too small.
  NrmOutXp  = 8,
  // Outflow on an ARBITRARY face. Same closure as NrmOutXp -- rho imposed, the
  // velocity taken from an upstream neighbour -- but the neighbour is found per
  // node from the geometry instead of being fixed to -x, which is what a
  // voxelised oblique cap needs. There is no single axis normal for such a cap,
  // so normal_of() cannot describe it and the unknown set is built
  // geometrically (it already is, for every code).
  NrmOutFree = 9,
};

// Direction pointing INTO the domain from an outflow face, i.e. toward the
// upstream neighbour whose state is copied.
KOKKOS_INLINE_FUNCTION
constexpr void upstream_of(std::uint8_t code, int e[3]) {
  e[0] = e[1] = e[2] = 0;
  if (code == NrmOutXp) e[0] = -1;
}

KOKKOS_INLINE_FUNCTION
constexpr void normal_of(std::uint8_t code, int n[3]) {
  n[0] = n[1] = n[2] = 0;
  switch (code) {
    case NrmXp: n[0] =  1; break;
    case NrmXm: n[0] = -1; break;
    case NrmYp: n[1] =  1; break;
    case NrmYm: n[1] = -1; break;
    case NrmZp: n[2] =  1; break;
    case NrmZm: n[2] = -1; break;
    case NrmOutXp: n[0] = 1; break;      // outward normal, for the unknown set
    default: break;
  }
}

//------------------------------------------------------------------------------
template <class L, class Eq>
struct Regularized {
  using EqType = Eq;
  static constexpr int Q = L::Q;
  static constexpr int D = L::D;

  //----------------------------------------------------------------------------
  // Eq. (27). Known populations are those travelling OUT of the domain: they
  // were streamed from the interior and so carry information. u_perp is the
  // imposed velocity projected on the outward normal.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static Real density(const Real f[Q], const int nrm[3], const Real u[3]) {
    Real out = Real(0), tang = Real(0);
    for (int i = 0; i < Q; ++i) {
      const int cn = cvel<L>(i, 0) * nrm[0] + cvel<L>(i, 1) * nrm[1] +
                     cvel<L>(i, 2) * nrm[2];
      if (cn > 0)      out  += f[i];
      else if (cn == 0) tang += f[i];
    }
    const Real uperp = u[0] * nrm[0] + u[1] * nrm[1] + u[2] * nrm[2];
    return (Real(2) * out + tang) / (Real(1) + uperp);
  }

  //----------------------------------------------------------------------------
  // Eq. (27) inverted: rho known, outward normal velocity unknown. This is the
  // pressure-boundary counterpart of density() and uses the identical partition
  // of the populations, so the two are consistent by construction.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static Real normal_velocity(const Real f[Q], const int nrm[3], Real rho) {
    Real out = Real(0), tang = Real(0);
    for (int i = 0; i < Q; ++i) {
      const int cn = cvel<L>(i, 0) * nrm[0] + cvel<L>(i, 1) * nrm[1] +
                     cvel<L>(i, 2) * nrm[2];
      if (cn > 0)       out  += f[i];
      else if (cn == 0) tang += f[i];
    }
    return (Real(2) * out + tang) / rho - Real(1);
  }

  //----------------------------------------------------------------------------
  // Eq. (45) with Pi^(1) supplied from outside instead of measured locally.
  //
  // This is the corner/edge route of Latt et al., Sec. V: there the stress is
  // obtained from finite-difference velocity gradients (their Eq. 46, the BC4
  // form) rather than from the node's own populations, which at a corner carry
  // too little information -- only three of nine directions have streamed from
  // the fluid on a 2D corner.
  //
  // Pi is packed xx, yy, zz, xy, xz, yz.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static void apply_with_stress(Real f[Q], Real rho, const Real u[3], const Real Pi[6]) {
    constexpr Real cs2v = cs2<L, Real>();
    constexpr Real cs4v = cs2v * cs2v;
    for (int i = 0; i < Q; ++i) {
      const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)),
                 cz = Real(cvel<L>(i, 2));
      const Real QP = (cx * cx - cs2v) * Pi[0] + (cy * cy - cs2v) * Pi[1]
                    + ((D == 3) ? (cz * cz - cs2v) * Pi[2] : Real(0))
                    + Real(2) * (cx * cy * Pi[3]
                                 + ((D == 3) ? (cx * cz * Pi[4] + cy * cz * Pi[5])
                                             : Real(0)));
      f[i] = Eq::eq(i, rho, u[0], u[1], u[2])
           + weight<L, Real>(i) / (Real(2) * cs4v) * QP;
    }
  }

  // Pi^(1) = -(2 cs^2 / omega) rho S,  S = (grad u + grad u^T) / 2   Eqs. (21)-(22).
  // `g[a][b]` is d u_b / d x_a.
  KOKKOS_INLINE_FUNCTION
  static void stress_from_gradient(Real rho, Real omega, const Real g[3][3], Real Pi[6]) {
    constexpr Real cs2v = cs2<L, Real>();
    const Real k = -Real(2) * cs2v * rho / omega;
    Pi[0] = k * g[0][0];
    Pi[1] = k * g[1][1];
    Pi[2] = k * g[2][2];
    Pi[3] = k * Real(0.5) * (g[0][1] + g[1][0]);
    Pi[4] = k * Real(0.5) * (g[0][2] + g[2][0]);
    Pi[5] = k * Real(0.5) * (g[1][2] + g[2][1]);
  }

  //----------------------------------------------------------------------------
  // Eqs. (44)-(45). `f` is overwritten with the reconstructed populations.
  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  // `unknown` is a bitmask over directions: bit i set means the population that
  // should have streamed into direction i came from outside the fluid, so it
  // holds nothing meaningful and must be invented.
  //
  // A mask is used rather than the sign of c_i . n because at a corner there is
  // no single normal, and the unknown set is not a half-space. At the corner
  // (0,0) of a box, BOTH (1,-1) and (-1,1) are unknown -- each has its source
  // node outside a different wall -- which no dot-product test can express.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static void apply(Real f[Q], Real rho, const Real u[3], std::uint32_t unknown,
                    const Real Fv[3] = nullptr) {
    constexpr Real cs2v = cs2<L, Real>();
    constexpr Real cs4v = cs2v * cs2v;

    Real feq[Q];
    for (int i = 0; i < Q; ++i) feq[i] = Eq::eq(i, rho, u[0], u[1], u[2]);

    // Step 2: unknown populations get a bounce-back of their off-equilibrium
    // part. Scaffolding for Pi^(1) only -- overwritten in step 4.
    //
    // When BOTH members of a pair are unknown -- which happens on corners and
    // edges, never on a straight wall -- there is nothing to bounce back from.
    // Those are set to equilibrium, contributing zero to Pi^(1), which is the
    // only choice that adds no invented stress.
    //
    // FORCED FLOW. The Chapman-Enskog expansion with a Guo source gives
    // sum_i c_i f^(1) = -F/2, so f^(1) has an ODD part -(w_i / 2 cs^2) c_i . F.
    // Bounce-back reverses that sign, so each unknown is wrong by
    // (w_i / cs^2) c_i . F. Summing c_i c_i over that error would vanish over
    // the full velocity set, but the unknowns are a HALF-SPACE and it does not:
    // on D2Q9 a tangential force biases Pi_xy by F/6, a normal force biases
    // Pi_xx by F/6 and Pi_yy by F/2. Different tensors entirely, which is why
    // no single velocity shift can repair both cases.
    const Real hodd = Real(0.5) / cs2v;
    for (int i = 0; i < Q; ++i) {
      if (!(unknown & (1u << i))) continue;
      const Real cF = Fv ? (Real(cvel<L>(i, 0)) * Fv[0] + Real(cvel<L>(i, 1)) * Fv[1] +
                            Real(cvel<L>(i, 2)) * Fv[2]) : Real(0);
      const Real odd = weight<L, Real>(i) * hodd * cF;      // (w_i / 2 cs^2) c_i . F
      f[i] = (unknown & (1u << opp(i)))
           ? feq[i] - odd                                    // no partner: use the true odd part
           : feq[i] + (f[opp(i)] - feq[opp(i)]) - Real(2) * odd;  // undo the reversed sign
    }

    // Step 3: Pi^(1) from the off-equilibrium parts.
    Real Pi[6] = {0, 0, 0, 0, 0, 0};        // xx, yy, zz, xy, xz, yz
    for (int i = 0; i < Q; ++i) {
      const Real d  = f[i] - feq[i];
      const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)),
                 cz = Real(cvel<L>(i, 2));
      Pi[0] += d * cx * cx;  Pi[1] += d * cy * cy;  Pi[2] += d * cz * cz;
      Pi[3] += d * cx * cy;  Pi[4] += d * cx * cz;  Pi[5] += d * cy * cz;
    }

    // Step 4: rebuild every population. Q_i : Pi^(1) with the off-diagonal
    // terms counted twice, both tensors being symmetric.
    for (int i = 0; i < Q; ++i) {
      const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)),
                 cz = Real(cvel<L>(i, 2));
      const Real QP = (cx * cx - cs2v) * Pi[0] + (cy * cy - cs2v) * Pi[1]
                    + ((D == 3) ? (cz * cz - cs2v) * Pi[2] : Real(0))
                    + Real(2) * (cx * cy * Pi[3]
                                 + ((D == 3) ? (cx * cz * Pi[4] + cy * cz * Pi[5])
                                             : Real(0)));
      // The second-order Hermite reconstruction of f^(1) given
      // sum_i c_i f^(1) = -F/2 and sum_i c_i c_i f^(1) = Pi^(1): the first
      // moment needs its own term, which Q_i : Pi cannot supply since
      // sum_i w_i c_i Q_i = 0.
      const Real cF = Fv ? (cx * Fv[0] + cy * Fv[1] + cz * Fv[2]) : Real(0);
      f[i] = feq[i] - weight<L, Real>(i) * hodd * cF
           + weight<L, Real>(i) / (Real(2) * cs4v) * QP;
    }
  }
};

}  // namespace lbm
