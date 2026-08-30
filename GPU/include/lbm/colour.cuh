#pragma once
//==============================================================================
//  Colour-gradient two-component flow with nonorthogonal central moments,
//  on the GPU.
//
//  Saito, De Rosis, Festuccia, Kaneko, Abe & Koyama, Phys. Rev. E 98, 013305
//  (2018). Equation numbers below are that paper's. D3Q27 only, which is what
//  the paper derives and, conveniently, the only fluid lattice this code has.
//
//  A PORT OF ../src/collision/ColourGradient.hpp AND ../src/solver/
//  ColourGradientSolver.hpp, and the first multiphase module here. The physics
//  is the parent's, including three readings that had to be worked out rather
//  than copied -- they are restated at the point of use below, because a port
//  that silently reverts one of them would reproduce the paper's printed
//  equations and the wrong answer.
//
//  WHY THIS ONE PORTED AND THE OTHER TWO DID NOT. Of the parent's three
//  multiphase engines this is the only one that needs nothing new from the
//  storage layer. It is D3Q27-only, which is this code's lattice; it carries two
//  distributions on that same lattice, which is what scalar.cuh and magnetic.cuh
//  already do; and every one of its three passes reads populations at a single
//  time level, so Esoteric Pull is undisturbed. The free-surface engine, by
//  contrast, static_asserts AGAINST Esoteric Pull -- its population
//  reconstruction reads a slot the in-place scheme has already overwritten,
//  which is the same class of race as the scalar outflow this code lists as
//  absent -- and would need a two-lattice storage path that does not exist here.
//
//  THREE PASSES, AND THE FENCES BETWEEN THEM ARE THE POINT.
//
//    1. FIELDS. rho_r, rho_b, phi and u at every node, from the two population
//       sets. Read-only, so it may run at the parity the next step will collide
//       at -- the same argument scalar_field_kernel carries.
//    2. GRADIENTS. grad phi and grad rho, Eq. (40), by a lattice-weighted
//       stencil over the NEIGHBOUR field values pass 1 wrote.
//    3. STREAM-COLLIDE. Gather both colours, collide the colour-blind sum,
//       perturb, recolour, scatter.
//
//  Pass 2 reads neighbours, so it cannot be fused into pass 1: it needs every
//  node's field finished. Pass 3 reads only its OWN node's gradient, so it could
//  in principle fuse with pass 2 -- it is kept separate because the gradient
//  stencil substitutes this node's value at a wall neighbour, and that
//  substitution needs pass 1 complete for the neighbour too. Three kernel
//  launches on one stream give three fences for free; there is no
//  cudaDeviceSynchronize on the step path.
//
//  WHY THE GRADIENT CANNOT COME FROM THE POPULATIONS. Under Esoteric Pull the
//  neighbour populations visible inside a fused kernel are a mixture of two time
//  levels, so a gradient taken from them is not a gradient of anything. This is
//  the same reason the parent gives at length, and it is why the fields are
//  materialised into arrays at all rather than recomputed per neighbour.
//
//  MEMORY. Two population sets at 27 Real each, plus twelve node fields. At
//  FP32 that is 108 + 48 = 156 bytes per node against the single-phase core's
//  108 + 16. A 256^3 run therefore needs about 2.6 GB, which fits a T4; 384^3
//  does not.
//
//  IT WAS 47x SLOWER THAN THE SINGLE-PHASE CORE, AND IS NOW 3.8x. The first
//  version built the equilibrium and the perturbation as populations and
//  transformed both: 20.2 MLUPS on a T4 at 64^3 in FP32 against the core's 950,
//  with ptxas reporting a 216-byte stack frame -- two 27-arrays in local memory,
//  touched at every node. The closed forms below removed it:
//
//      before   119 registers, 216 bytes stack frame,  20.2 MLUPS
//      after    124 registers,   0 bytes stack frame, 252.5 MLUPS
//
//  FP64 results are unchanged to every digit printed, so this is the same
//  operator rather than a cheaper approximation of it; test/host_colour.cpp
//  keeps the old path and asserts the two agree to 5.9e-16 (FP64) over 60
//  states.
//==============================================================================
#include "streaming.cuh"

namespace lbm {

using ColourLattice = D3Q27;

// Explicit rather than std::sqrt so nvcc picks the right overload in both
// precisions without an ambiguity that only shows up in one of them.
LBM_HD LBM_INLINE Real cg_sqrt(Real x) {
#if defined(LBM_DOUBLE)
  return ::sqrt(x);
#else
  return ::sqrtf(x);
#endif
}

// |c_i|^2, which is 0, 1, 2 or 3 on this lattice and is what phi_i, B_i and
// Phi_coeff are all keyed off.
LBM_HD LBM_INLINE int cg_csq(int i) {
  const int x = ColourLattice::cx(i), y = ColourLattice::cy(i),
            z = ColourLattice::cz(i);
  return x * x + y * y + z * z;
}

//------------------------------------------------------------------------------
// Moment slots, in the same form the parent's ProductBasis writes them, so the
// two can be compared line by line. mi() is core.cuh's exponent-triple index.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE constexpr int cg_i1(int a) {
  return mi(a == 0, a == 1, a == 2);
}
LBM_HD LBM_INLINE constexpr int cg_i2d(int a) {
  return mi(2 * (a == 0), 2 * (a == 1), 2 * (a == 2));
}
LBM_HD LBM_INLINE constexpr int cg_i2s(int a, int b) {
  return mi(a == 0 || b == 0, a == 1 || b == 1, a == 2 || b == 2);
}

//==============================================================================
//  The model constants and the equilibrium.
//
//  A plain POD with LBM_HD members, so host_check.cpp and hostsim.hpp exercise
//  exactly the arithmetic the kernels run. Passed BY VALUE into every kernel.
//==============================================================================
struct ColourModel {
  Real alpha_r = Real(8) / Real(27);   // rest weights; gamma = (1-ab)/(1-ar)
  Real alpha_b = Real(8) / Real(27);
  Real nu_r = Real(1) / Real(6);       // kinematic viscosities, harmonic-mixed
  Real nu_b = Real(1) / Real(6);
  Real A = Real(0);                    // interfacial tension strength, Eq. (32)
  Real beta = Real(0.7);               // recolouring sharpness
  Real omega_bulk = Real(1);           // s2b
  Real bx = Real(0), by = Real(0), bz = Real(0);   // body force per unit mass
  // F = (rho - rho_ref) b. In a FULLY PERIODIC box rho b injects net momentum
  // every step -- no wall for the hydrostatic gradient to push against -- so the
  // whole fluid falls. Setting rho_ref to the domain mean removes the mean force
  // and leaves the buoyancy difference. Not a Boussinesq approximation: the full
  // density difference is retained, only its mean is subtracted.
  Real rho_ref = Real(0);
  // The initial density of each PURE phase, Eq. (23). Carried explicitly rather
  // than derived from alpha: the derived form inverts easily and SILENTLY, and
  // upside down it leaves both pure phases at phi = +-1 -- every bulk check
  // still passes -- while corrupting only the interface.
  Real rho_r0 = Real(1);
  Real rho_b0 = Real(1);

  //---- sigma = (4/9) A tau, Eq. (32) ----------------------------------------
  static Real A_from_sigma(Real sigma, Real tau) {
    return Real(9) * sigma / (Real(4) * tau);
  }
  static Real sigma_from_A(Real a, Real tau) {
    return Real(4) * a * tau / Real(9);
  }
  // The coefficient the colour-blind perturbation actually carries -- A, not
  // A/2. Exposed so a test asserts the relation rather than the constant.
  static constexpr Real perturbation_coefficient(Real a) { return a; }

  // cs^2 = 9(1-alpha)/19, the second moment of phi_i. NOT the lattice constant.
  static LBM_HD LBM_INLINE Real cs2_of_alpha(Real a) {
    return Real(9) * (Real(1) - a) / Real(19);
  }
  // gamma = (1 - alpha_b)/(1 - alpha_r), Eq. (25), inverted for alpha_r.
  static Real alpha_r_from_ratio(Real gamma, Real ab) {
    return Real(1) - (Real(1) - ab) / gamma;
  }

  //--------------------------------------------------------------------------
  // phi_i, Eq. (20). Sums to one for any alpha; second moment 9(1-alpha)/19.
  //--------------------------------------------------------------------------
  static LBM_HD LBM_INLINE Real phi_i(int i, Real a) {
    const int q = cg_csq(i);
    const Real c = Real(1) - a;
    if (q == 0) return a;
    if (q == 1) return Real(2) * c / Real(19);
    if (q == 2) return c / Real(38);
    return c / Real(152);
  }

  // B_i, Eq. (31). sum_i B_i = 1/3, which is what makes the perturbation
  // conserve mass against sum_i w_i (c_i.n)^2 = 1/3.
  static LBM_HD LBM_INLINE Real B_i(int i) {
    const int q = cg_csq(i);
    if (q == 0) return Real(-10) / Real(27);
    if (q == 1) return Real(2) / Real(27);
    if (q == 2) return Real(1) / Real(54);
    return Real(1) / Real(216);
  }

  // The coefficient of (G : c_i x c_i) in Phi_i, Eq. (21).
  static LBM_HD LBM_INLINE Real Phi_coeff(int i) {
    const int q = cg_csq(i);
    if (q == 0) return Real(0);
    if (q == 1) return Real(16);
    if (q == 2) return Real(4);
    return Real(1);
  }

  //--------------------------------------------------------------------------
  // The order parameter, Eq. (23). +1 in pure red, -1 in pure blue, 0 where the
  // two are present in equal PROPORTION of their own bulk densities -- which is
  // not equal mass, and is the whole point of dividing by rho^0.
  //--------------------------------------------------------------------------
  LBM_HD LBM_INLINE Real order_parameter(Real rr, Real rb) const {
    const Real a = rr / rho_r0, b = rb / rho_b0;
    const Real s = a + b;
    return (s > Real(0)) ? (a - b) / s : Real(0);
  }

  //--------------------------------------------------------------------------
  // Interpolants across the interface. alpha linearly, Eq. (27); the viscosity
  // HARMONICALLY, Eq. (22). The harmonic mean is not stylistic -- it keeps the
  // shear stress continuous across a viscosity ratio, and a linear mean there
  // makes a stress jump that shows up as a spurious current.
  //--------------------------------------------------------------------------
  LBM_HD LBM_INLINE Real alpha_at(Real p) const {
    return Real(0.5) * ((Real(1) + p) * alpha_r + (Real(1) - p) * alpha_b);
  }
  LBM_HD LBM_INLINE Real nu_at(Real p) const {
    const Real inv = Real(0.5) * ((Real(1) + p) / nu_r + (Real(1) - p) / nu_b);
    return Real(1) / inv;
  }

  //--------------------------------------------------------------------------
  // The shear rate, Eq. (16): nu = (c^2/3)(1/s2v - 1/2). THE 1/3 IS THE
  // LATTICE'S, NOT THE PHASE'S, and this is the single most consequential
  // detail in the operator.
  //
  // cs^2 appears in two unrelated roles and they are different numbers. The
  // EQUATION OF STATE is p = rho cs^2(alpha) with cs^2(alpha) = 9(1-alpha)/19,
  // because phi_i carries the density ratio and the equilibrium's trace follows
  // it. The VISCOUS STRESS does not: it comes from the non-equilibrium part,
  // whose second moment is governed by the standard weights w_i in the
  // velocity-dependent terms of Eq. (18), and those carry cs^2 = 1/3 in every
  // phase.
  //
  // Using the phase's cs^2 here was measured in the parent, and it fails in the
  // way hardest to attribute: at a ratio of 1000 it drives omega to 1.992 in the
  // middle of the interface -- inside the stability limit by a hair, wrong by a
  // factor of 500 in the viscosity -- and returns NaN a few hundred steps later
  // with nothing in between to point at it.
  //--------------------------------------------------------------------------
  LBM_HD LBM_INLINE Real omega_at(Real p) const {
    return Real(1) / (nu_at(p) * ColourLattice::inv_cs2() + Real(0.5));
  }
  static Real viscosity_from_tau(Real tau) {
    return ColourLattice::cs2() * (tau - Real(0.5));
  }

  //--------------------------------------------------------------------------
  // THE REST TERM, AND THE ONE READING IN THIS PAPER THAT HAD TO BE WORKED OUT
  // RATHER THAN COPIED.
  //
  // Eq. (18) writes the rest contribution as rho phi_i, and Eq. (27)
  // interpolates alpha linearly in phi, which together read as
  // rho phi_i(alpha-bar). That cannot be what is meant. The trace of THAT
  // equilibrium is rho cs^2(alpha-bar), and with the tanh profiles of
  // Eqs. (41)-(42) it is not continuous through the interface: at a ratio of 100
  // the midpoint carries p = 8.5 against a bulk 1/3, a twenty-five-fold pressure
  // spike two cells wide. Measured in the parent, that is exactly what happens
  // -- gamma = 1 and 10 survive it, 100 and 1000 return NaN.
  //
  // The rest term is PER COLOUR, sum_k rho_k phi_i(alpha_k), whose trace is
  // sum_k rho_k cs_k^2 and which is continuous by construction. It is also what
  // Eq. (26) says, with the colour subscript it carries, and the only reading
  // consistent with Eq. (25).
  //
  // At a matched ratio the two readings coincide exactly, which is why the error
  // hides until a density ratio is asked for -- and why a port is exactly where
  // it would come back.
  //--------------------------------------------------------------------------
  LBM_HD LBM_INLINE Real rest_term(int i, Real rho_r, Real rho_b) const {
    return rho_r * phi_i(i, alpha_r) + rho_b * phi_i(i, alpha_b);
  }

  //--------------------------------------------------------------------------
  // The pressure, Eq. (26): p = sum_k rho_k cs_k^2, each phase with its own
  // alpha. This is the trace of the rest term and the quantity that is
  // continuous through the interface -- the whole reason that term is per
  // colour. It is also what the closed-form central moments are written in.
  //--------------------------------------------------------------------------
  LBM_HD LBM_INLINE Real pressure(Real rho_r, Real rho_b) const {
    return rho_r * cs2_of_alpha(alpha_r) + rho_b * cs2_of_alpha(alpha_b);
  }

  // f_i^eq(rho_r, rho_b, u = 0): every other term of Eq. (18) carries a u or a
  // G, and G vanishes with u, so the rest term is the whole equilibrium at rest.
  // This is what the recolouring is written against.
  LBM_HD LBM_INLINE Real eq_at_rest(int i, Real rho_r, Real rho_b) const {
    return rest_term(i, rho_r, rho_b);
  }

  //--------------------------------------------------------------------------
  // The equilibrium, Eq. (18) with the Phi_i of Eq. (21). Third order in u, and
  // that is deliberate in the source: the O(u^3) terms are what make the model
  // Galilean invariant at a density ratio, which is the regime it exists for.
  //--------------------------------------------------------------------------
  LBM_HD LBM_INLINE
  void equilibrium(Real fe[27], Real rho_r, Real rho_b, const Real u[3],
                   const Real G[3][3], Real nubar, Real udrho) const {
    const Real rho = rho_r + rho_b;
    const Real u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    for (int i = 0; i < 27; ++i) {
      const Real cx = Real(ColourLattice::cx(i)),
                 cy = Real(ColourLattice::cy(i)),
                 cz = Real(ColourLattice::cz(i));
      const Real cu = cx * u[0] + cy * u[1] + cz * u[2];
      const Real w  = ColourLattice::w(i);
      Real e = rest_term(i, rho_r, rho_b)
             + rho * w * (Real(3) * cu + Real(4.5) * cu * cu - Real(1.5) * u2
                  + Real(4.5) * cu * cu * cu - Real(4.5) * cu * u2);
      // Phi_i: the rest slot balances the rest, so that sum_i Phi_i = 0.
      const Real k = Phi_coeff(i);
      if (k == Real(0)) {
        e += Real(-3) * nubar * udrho;
      } else {
        const Real c[3] = {cx, cy, cz};
        Real gcc = Real(0);
        for (int p = 0; p < 3; ++p)
          for (int q = 0; q < 3; ++q) gcc += G[p][q] * c[p] * c[q];
        e += k * nubar * gcc;
      }
      fe[i] = e;
    }
  }
};

//==============================================================================
//  CLOSED-FORM CENTRAL MOMENTS, AND WHY THEY ARE WORTH THE DERIVATION.
//
//  The first version of this operator built the equilibrium and the
//  perturbation as POPULATIONS and transformed both, which is exact by
//  construction and was the right way to get it correct. It cost four
//  transforms per node and six live 27-arrays, and a T4 measured what that
//  buys: 20.2 MLUPS against the single-phase core's 950, with ptxas reporting
//  119 registers and a 216-byte STACK FRAME -- two arrays that did not fit in
//  registers and lived in local memory, read and written at every node.
//
//  So the equilibrium's and the perturbation's central moments are derived
//  here instead. NOTHING BELOW IS ASSUMED FROM THE PRODUCT-FORM EQUILIBRIUM:
//  Eq. (18) is not that equilibrium, so eq_moment() of core.cuh -- where every
//  central moment above order 0 collapses to zero -- does not apply and would
//  be silently wrong if reused. Each piece was derived symbolically over exact
//  rationals and checked against the transform it replaces, and
//  test/host_colour.cpp asserts that agreement on random states.
//
//  THE THREE PIECES OF Eq. (18), each with its own structure.
//
//  (a) THE REST TERM, sum_k rho_k phi_i(alpha_k). Its raw moments are
//      isotropic and, remarkably, GEOMETRIC in the number of squared indices:
//      sum_i phi_i c_x^2 = cs_k^2, and then c_x^2 c_y^2 gives cs_k^2/3 and
//      c_x^2 c_y^2 c_z^2 gives cs_k^2/9. That makes the central moments a
//      difference of two separable products,
//
//          k = 3P * prod_a W(p_a) - 3S * prod_a D(p_a),
//          W = (1, -u_a, u_a^2),   D = (1, -u_a, u_a^2 - cs^2),
//
//      with P = sum_k rho_k cs_k^2 the PRESSURE and S = P - rho cs^2. Note W is
//      just (-u_a)^p, so the first product is (-1)^n prod u^p.
//
//  (b) THE VELOCITY POLYNOMIAL, rho w_i (3cu + 4.5cu^2 - 1.5u^2 + 4.5cu^3
//      - 4.5 cu u^2). Its central moments collapse to
//
//          k = rho * C[p+q+r] * ux^p uy^q uz^r,   C = 0, 1, -1, 1, -2, 5, -11,
//
//      i.e. the coefficient depends only on the TOTAL order. That is a property
//      of the third-order truncation on this lattice, not a general one.
//
//  (c) Phi_i of Eq. (21) AND the perturbation of Eq. (30) share a structure:
//      every ODD raw moment vanishes, leaving a second-order tensor T, a
//      fourth-order pair (Q, R) and a sixth-order scalar H. cg_source_high()
//      below is the shift of exactly such a source, derived once and used for
//      both. Its 17 cases are machine-generated from the symbolic derivation
//      rather than transcribed, because 27 formulas copied by eye is precisely
//      how a silent error gets in.
//
//  TWO CONSEQUENCES WORTH STATING, both now visible rather than computed:
//
//    * ORDER 1 VANISHES FOR BOTH SOURCES. The perturbation contributes nothing
//      to momentum and neither does the equilibrium -- so the first-order line
//      of the collision carries only the body force. That is the "conserves
//      momentum" property the test measures, here as an identity.
//    * ORDER 2 OF THE EQUILIBRIUM IS S delta_ab + Psi_ab, with no u-dependence
//      at all: the c_n and 3S products cancel exactly at second order. At a
//      matched density S = 0 and the classical result -- equilibrium central
//      moments vanish above order 0 -- is recovered as a special case.
//==============================================================================

// The raw moments of a source whose odd moments all vanish. Both Phi_i and the
// perturbation are of this kind, so one struct and one shift serve both.
struct CgSource {
  Real Txx, Tyy, Tzz, Txy, Txz, Tyz;   // order 2
  Real Qxy, Qxz, Qyz;                  // order 4, the (2,2,0) family
  Real Rx, Ry, Rz;                     // order 4, the (2,1,1) family
  Real H;                              // order 6
};

//------------------------------------------------------------------------------
// The central moment of such a source at one slot, for order >= 3 only -- order
// 0 and 1 vanish identically and order 2 is just T, so the caller reads those
// off directly. MACHINE-GENERATED; see the banner above.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE Real cg_source_high(int slot, const CgSource& s,
                                      Real ux, Real uy, Real uz) {
  const Real ux2 = ux * ux, uy2 = uy * uy, uz2 = uz * uz;
  switch (slot) {
      case 5:   // (0,1,2)
        return -Real(2)*s.Tyz*uz - s.Tzz*uy ;
      case 7:   // (0,2,1)
        return -Real(2)*s.Tyz*uy - s.Tyy*uz ;
      case 8:   // (0,2,2)
        return s.Qyz - (Real(1)/Real(3))*s.Tzz - (Real(1)/Real(3))*s.Tyy +
               Real(4)*s.Tyz*uy*uz + s.Tzz*uy2 + s.Tyy*uz2 ;
      case 11:   // (1,0,2)
        return -Real(2)*s.Txz*uz - s.Tzz*ux ;
      case 13:   // (1,1,1)
        return -s.Tyz*ux - s.Txz*uy - s.Txy*uz ;
      case 14:   // (1,1,2)
        return s.Rz - (Real(1)/Real(3))*s.Txy + Real(2)*s.Tyz*ux*uz +
               Real(2)*s.Txz*uy*uz + s.Txy*uz2 + s.Tzz*ux*uy ;
      case 15:   // (1,2,0)
        return -Real(2)*s.Txy*uy - s.Tyy*ux ;
      case 16:   // (1,2,1)
        return s.Ry - (Real(1)/Real(3))*s.Txz + Real(2)*s.Tyz*ux*uy + s.Txz*uy2 +
               Real(2)*s.Txy*uy*uz + s.Tyy*ux*uz ;
      case 17:   // (1,2,2)
        return -Real(2)*s.Rz*uy - Real(2)*s.Ry*uz - s.Qyz*ux +
               (Real(2)/Real(3))*s.Txz*uz + (Real(2)/Real(3))*s.Txy*uy +
               (Real(1)/Real(3))*s.Tzz*ux + (Real(1)/Real(3))*s.Tyy*ux -
               Real(4)*s.Tyz*ux*uy*uz - Real(2)*s.Txz*uy2*uz - Real(2)*s.Txy*uy*uz2 -
               s.Tzz*ux*uy2 - s.Tyy*ux*uz2 ;
      case 19:   // (2,0,1)
        return -Real(2)*s.Txz*ux - s.Txx*uz ;
      case 20:   // (2,0,2)
        return s.Qxz - (Real(1)/Real(3))*s.Tzz - (Real(1)/Real(3))*s.Txx +
               Real(4)*s.Txz*ux*uz + s.Tzz*ux2 + s.Txx*uz2 ;
      case 21:   // (2,1,0)
        return -Real(2)*s.Txy*ux - s.Txx*uy ;
      case 22:   // (2,1,1)
        return s.Rx - (Real(1)/Real(3))*s.Tyz + s.Tyz*ux2 + Real(2)*s.Txz*ux*uy +
               Real(2)*s.Txy*ux*uz + s.Txx*uy*uz ;
      case 23:   // (2,1,2)
        return -Real(2)*s.Rz*ux - Real(2)*s.Rx*uz - s.Qxz*uy +
               (Real(2)/Real(3))*s.Tyz*uz + (Real(2)/Real(3))*s.Txy*ux +
               (Real(1)/Real(3))*s.Tzz*uy + (Real(1)/Real(3))*s.Txx*uy -
               Real(2)*s.Tyz*ux2*uz - Real(4)*s.Txz*ux*uy*uz - Real(2)*s.Txy*ux*uz2 -
               s.Tzz*ux2*uy - s.Txx*uy*uz2 ;
      case 24:   // (2,2,0)
        return s.Qxy - (Real(1)/Real(3))*s.Tyy - (Real(1)/Real(3))*s.Txx +
               Real(4)*s.Txy*ux*uy + s.Tyy*ux2 + s.Txx*uy2 ;
      case 25:   // (2,2,1)
        return -Real(2)*s.Ry*ux - Real(2)*s.Rx*uy - s.Qxy*uz +
               (Real(2)/Real(3))*s.Tyz*uy + (Real(2)/Real(3))*s.Txz*ux +
               (Real(1)/Real(3))*s.Tyy*uz + (Real(1)/Real(3))*s.Txx*uz -
               Real(2)*s.Tyz*ux2*uy - Real(2)*s.Txz*ux*uy2 - Real(4)*s.Txy*ux*uy*uz -
               s.Tyy*ux2*uz - s.Txx*uy2*uz ;
      case 26:   // (2,2,2)
        return s.H - (Real(1)/Real(3))*s.Qyz - (Real(1)/Real(3))*s.Qxz -
               (Real(1)/Real(3))*s.Qxy + (Real(1)/Real(9))*s.Tzz +
               (Real(1)/Real(9))*s.Tyy + (Real(1)/Real(9))*s.Txx + Real(4)*s.Rz*ux*uy
               + Real(4)*s.Ry*ux*uz + Real(4)*s.Rx*uy*uz + s.Qyz*ux2 + s.Qxz*uy2 +
               s.Qxy*uz2 - (Real(4)/Real(3))*s.Tyz*uy*uz -
               (Real(4)/Real(3))*s.Txz*ux*uz - (Real(4)/Real(3))*s.Txy*ux*uy -
               (Real(1)/Real(3))*s.Tzz*uy2 - (Real(1)/Real(3))*s.Tzz*ux2 -
               (Real(1)/Real(3))*s.Tyy*uz2 - (Real(1)/Real(3))*s.Tyy*ux2 -
               (Real(1)/Real(3))*s.Txx*uz2 - (Real(1)/Real(3))*s.Txx*uy2 +
               Real(4)*s.Tyz*ux2*uy*uz + Real(4)*s.Txz*ux*uy2*uz +
               Real(4)*s.Txy*ux*uy*uz2 + s.Tzz*ux2*uy2 + s.Tyy*ux2*uz2 +
               s.Txx*uy2*uz2 ;
      default: return Real(0);
  }
}

//------------------------------------------------------------------------------
// The per-axis factors of the rest term's two separable products, and the
// total-order coefficient of the velocity polynomial. Written as ternaries
// rather than indexed tables ON PURPOSE: a local array indexed by a runtime
// value is placed in local memory, which is the very thing this rewrite exists
// to remove.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE Real cg_upow(int e, Real u) {
  return (e == 0) ? Real(1) : ((e == 1) ? u : u * u);
}
LBM_HD LBM_INLINE Real cg_dfac(int e, Real u) {
  return (e == 0) ? Real(1)
                  : ((e == 1) ? -u : u * u - ColourLattice::cs2());
}
// c_n = 3P(-1)^n + rho C_n, the rest term and the velocity polynomial combined.
LBM_HD LBM_INLINE Real cg_cN(int ord, Real P3, Real rho) {
  Real C;
  switch (ord) {
    case 0:  C = Real(0);   break;
    case 1:  C = Real(1);   break;
    case 2:  C = -Real(1);  break;
    case 3:  C = Real(1);   break;
    case 4:  C = -Real(2);  break;
    case 5:  C = Real(5);   break;
    default: C = -Real(11); break;
  }
  return ((ord & 1) ? -P3 : P3) + rho * C;
}

//==============================================================================
//  Suboperators (1) and (2), on the colour-blind populations.
//
//  f is overwritten with the post-collision, post-perturbation state; the caller
//  recolours it. The gradients arrive BY VALUE rather than as arrays, so this
//  function reads no memory at all and a test can drive it directly.
//
//  ONE transform in, one out. The equilibrium and the perturbation never become
//  populations and never become moment arrays; they are evaluated per slot from
//  the closed forms above.
//==============================================================================
LBM_HD LBM_INLINE
void colour_collide(const ColourModel& m, Real f[27], Real rho_r, Real rho_b,
                    const Real u[3], Real p, const Real g[3], const Real dr[3]) {
  const Real rho   = rho_r + rho_b;
  const Real nubar = m.nu_at(p);
  const Real omega = m.omega_at(p);
  const Real ux = u[0], uy = u[1], uz = u[2];

  //---- Phi_i, Eq. (21), through its raw moments.
  // 48 G_ab = u_a dr_b + u_b dr_a, so G is never formed.
  const Real udr = ux * dr[0] + uy * dr[1] + uz * dr[2];
  const Real Exx = nubar * (Real(2) * ux * dr[0] + udr);
  const Real Eyy = nubar * (Real(2) * uy * dr[1] + udr);
  const Real Ezz = nubar * (Real(2) * uz * dr[2] + udr);
  const Real Exy = nubar * (ux * dr[1] + uy * dr[0]);
  const Real Exz = nubar * (ux * dr[2] + uz * dr[0]);
  const Real Eyz = nubar * (uy * dr[2] + uz * dr[1]);
  const Real t3  = nubar * udr / Real(3);

  //---- (2) THE PERTURBATION, Eq. (30), through its raw moments.
  //
  // THE COEFFICIENT IS A, NOT A/2, and this is the reading most likely to be
  // reverted by someone checking against the paper. Eq. (30) is written PER
  // COLOUR and Eq. (39) sums the capillary stress over k as well as over i;
  // applied to the colour-blind population the two halves add. Derived from
  // Eq. (39), a flat interface gives sigma = (2 A tau / 9) delta(phi) with phi
  // running -1 to +1, so delta = 2 and sigma = 4 A tau / 9, which is Eq. (32)
  // exactly. With A/2 a static droplet reports a tension 50% low with every
  // other property of the model intact.
  //
  // The second moment is (2/9) A |grad phi| (n_a n_b - delta_ab) with n the unit
  // colour gradient -- so its NORMAL component vanishes identically for every
  // direction of n, and the tangential one does not. Writing it as
  // g_a g_b / |g| keeps the unit vector implicit and costs one reciprocal.
  Real Pxx = Real(0), Pyy = Real(0), Pzz = Real(0);
  Real Pxy = Real(0), Pxz = Real(0), Pyz = Real(0);
  Real Qxy = Real(0), Qxz = Real(0), Qyz = Real(0);
  Real Rx  = Real(0), Ry  = Real(0), Rz  = Real(0);
  const Real gm2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
  if (gm2 > Real(1e-24) && m.A != Real(0)) {
    const Real gm  = cg_sqrt(gm2);
    const Real inv = Real(1) / gm;
    const Real ca  = Real(2) * m.A / Real(9);
    const Real cq  = Real(2) * m.A / Real(27) * inv;
    Pxx = ca * (g[0] * g[0] * inv - gm);
    Pyy = ca * (g[1] * g[1] * inv - gm);
    Pzz = ca * (g[2] * g[2] * inv - gm);
    Pxy = ca * g[0] * g[1] * inv;
    Pxz = ca * g[0] * g[2] * inv;
    Pyz = ca * g[1] * g[2] * inv;
    Qxy = -cq * g[2] * g[2];
    Qxz = -cq * g[1] * g[1];
    Qyz = -cq * g[0] * g[0];
    Rx  =  cq * g[1] * g[2];
    Ry  =  cq * g[0] * g[2];
    Rz  =  cq * g[0] * g[1];
  }

  // Above second order the two sources are only ever needed added together.
  CgSource s;
  s.Txx = Exx + Pxx;  s.Tyy = Eyy + Pyy;  s.Tzz = Ezz + Pzz;
  s.Txy = Exy + Pxy;  s.Txz = Exz + Pxz;  s.Tyz = Eyz + Pyz;
  s.Qxy = nubar * (Real(2) * (ux * dr[0] + uy * dr[1]) / Real(3)) + t3 + Qxy;
  s.Qxz = nubar * (Real(2) * (ux * dr[0] + uz * dr[2]) / Real(3)) + t3 + Qxz;
  s.Qyz = nubar * (Real(2) * (uy * dr[1] + uz * dr[2]) / Real(3)) + t3 + Qyz;
  s.Rx  = nubar * (uy * dr[2] + uz * dr[1]) / Real(3) + Rx;
  s.Ry  = nubar * (ux * dr[2] + uz * dr[0]) / Real(3) + Ry;
  s.Rz  = nubar * (ux * dr[1] + uy * dr[0]) / Real(3) + Rz;
  s.H   = t3;                                  // the perturbation has none

  //---- (1) the central-moment collision. The only transform in the operator.
  const Real ub[3] = {ux, uy, uz};
  Real k[27];
  to_moments(f, ub, k);

  const Real P  = m.pressure(rho_r, rho_b);
  const Real S  = P - rho * ColourLattice::cs2();
  const Real P3 = Real(3) * P;
  const Real S3 = P3 - rho;                        // 3S, exactly

  // order 1: conserved. s0 = s1 = 0 in Eq. (15) means the collision leaves them
  // alone, and BOTH sources have a vanishing first central moment, so the body
  // force is the only thing that moves them.
  const Real fw = rho - m.rho_ref;
  k[cg_i1(0)] += fw * m.bx;
  k[cg_i1(1)] += fw * m.by;
  k[cg_i1(2)] += fw * m.bz;

  // order 2: trace at s2b, deviatoric and shear at s2v. THE PERTURBATION IS NOT
  // RELAXED -- it is a source, and the (1 - s/2) factor a Guo force carries does
  // not apply to it: Eq. (39) defines the capillary stress as
  // -tau sum_i Omega^(2) c_i c_i, the UNRELAXED second moment integrated over
  // one relaxation time. Applying (1 - s/2) here puts sigma out by that factor,
  // and the static droplet reports it.
  {
    const Real e[3] = {S + Exx, S + Eyy, S + Ezz};   // equilibrium, closed form
    const Real q[3] = {Pxx, Pyy, Pzz};               // perturbation
    Real d[3];
    Real tr = Real(0), tre = Real(0), trq = Real(0);
    for (int a = 0; a < 3; ++a) {
      d[a] = k[cg_i2d(a)];
      tr += d[a];  tre += e[a];  trq += q[a];
    }
    const Real invD = Real(1) / Real(3);
    const Real tr_post = (Real(1) - m.omega_bulk) * tr + m.omega_bulk * tre + trq;
    for (int a = 0; a < 3; ++a)
      k[cg_i2d(a)] = (Real(1) - omega) * (d[a] - tr * invD)
                   + omega * (e[a] - tre * invD)
                   + (q[a] - trq * invD) + tr_post * invD;
    k[cg_i2s(0, 1)] = (Real(1) - omega) * k[cg_i2s(0, 1)] + omega * Exy + Pxy;
    k[cg_i2s(0, 2)] = (Real(1) - omega) * k[cg_i2s(0, 2)] + omega * Exz + Pxz;
    k[cg_i2s(1, 2)] = (Real(1) - omega) * k[cg_i2s(1, 2)] + omega * Eyz + Pyz;
  }

  // order >= 3: s3 = s4 = s5 = s6 = 1, so straight to equilibrium plus source.
#pragma unroll
  for (int n = 0; n < 27; ++n) {
    const int pe = p_of(n), qe = q_of(n), re = r_of(n);
    const int ord = pe + qe + re;
    if (ord < 3) continue;
    k[n] = cg_cN(ord, P3, rho) * cg_upow(pe, ux) * cg_upow(qe, uy) * cg_upow(re, uz)
         - S3 * cg_dfac(pe, ux) * cg_dfac(qe, uy) * cg_dfac(re, uz)
         + cg_source_high(n, s, ux, uy, uz);
  }

  to_populations(k, ub, f);
}

//==============================================================================
//  Suboperator (3), Eqs. (33)-(34). The two outputs sum to f_i identically, so
//  this cannot lose mass or momentum however wrong beta is.
//==============================================================================
LBM_HD LBM_INLINE
void colour_recolour(const ColourModel& m, const Real f[27], Real rho_r,
                     Real rho_b, const Real g[3], Real fr[27], Real fb[27]) {
  const Real rho = rho_r + rho_b;
  const Real inv = (rho > Real(0)) ? Real(1) / rho : Real(0);
  const Real mix = m.beta * rho_r * rho_b * inv * inv;

  const Real gm2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
  const Real gm  = (gm2 > Real(1e-24)) ? cg_sqrt(gm2) : Real(0);

  for (int i = 0; i < 27; ++i) {
    Real cosine = Real(0);
    if (gm > Real(0)) {
      const Real cx = Real(ColourLattice::cx(i)),
                 cy = Real(ColourLattice::cy(i)),
                 cz = Real(ColourLattice::cz(i));
      const Real cm2 = cx * cx + cy * cy + cz * cz;
      if (cm2 > Real(0))
        cosine = (cx * g[0] + cy * g[1] + cz * g[2]) / (cg_sqrt(cm2) * gm);
    }
    const Real split = mix * cosine * m.eq_at_rest(i, rho_r, rho_b);
    fr[i] = rho_r * inv * f[i] + split;
    fb[i] = rho_b * inv * f[i] - split;
  }
}

//==============================================================================
//  Kernel parameters. One struct for all three passes.
//==============================================================================
struct ColourParams {
  Real* fr = nullptr;                        // red populations
  Real* fb = nullptr;                        // blue populations
  Real* rho_r = nullptr;                     // pass 1 writes these
  Real* rho_b = nullptr;
  Real* phi = nullptr;
  Real* ux = nullptr;
  Real* uy = nullptr;
  Real* uz = nullptr;
  Real* gx = nullptr;                        // pass 2 writes these: grad phi
  Real* gy = nullptr;
  Real* gz = nullptr;
  Real* rx = nullptr;                        // and grad rho, for Phi_i
  Real* ry = nullptr;
  Real* rz = nullptr;
  const std::uint8_t* flags = nullptr;
  int nx = 0, ny = 0, nz = 0;
  ColourModel m;
};

//------------------------------------------------------------------------------
// PASS 1. The macroscopic fields.
//
// A wall cell's populations are in transit, not a state: summing them gives a
// number that is not a density. The gradient stencil substitutes for wall
// neighbours rather than reading these, so they are left alone.
//------------------------------------------------------------------------------
template <int Parity, bool HasGeometry>
LBM_HD LBM_INLINE void colour_fields_node(const ColourParams& p, long N, long n) {
  const std::uint8_t fl = HasGeometry ? p.flags[n] : std::uint8_t(Fluid);
  if (fl != Fluid) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real fr[27], fb[27];
  gather<Parity, ColourLattice>(p.fr, N, x, y, z, p.nx, p.ny, p.nz, fr);
  gather<Parity, ColourLattice>(p.fb, N, x, y, z, p.nx, p.ny, p.nz, fb);

  Real sr = Real(0), sb = Real(0), mom[3] = {Real(0), Real(0), Real(0)};
  for (int i = 0; i < 27; ++i) {
    sr += fr[i];  sb += fb[i];
    const Real t = fr[i] + fb[i];
    mom[0] += t * Real(ColourLattice::cx(i));
    mom[1] += t * Real(ColourLattice::cy(i));
    mom[2] += t * Real(ColourLattice::cz(i));
  }
  const Real rho = sr + sb;
  const Real inv = (rho > Real(0)) ? Real(1) / rho : Real(0);
  // Eq. (13): the half body force is part of the velocity's definition.
  const Real fw = rho - p.m.rho_ref;
  p.ux[n] = (mom[0] + Real(0.5) * fw * p.m.bx) * inv;
  p.uy[n] = (mom[1] + Real(0.5) * fw * p.m.by) * inv;
  p.uz[n] = (mom[2] + Real(0.5) * fw * p.m.bz) * inv;
  p.rho_r[n] = sr;
  p.rho_b[n] = sb;
  p.phi[n] = p.m.order_parameter(sr, sb);
}

//------------------------------------------------------------------------------
// PASS 2. grad phi and grad rho, Eq. (40). One gather serves both.
//
// A non-fluid neighbour contributes THIS node's value, which is a zero normal
// derivative at the wall -- neutral wetting.
//------------------------------------------------------------------------------
template <bool HasGeometry>
LBM_HD LBM_INLINE void colour_gradient_node(const ColourParams& p, long n) {
  if (HasGeometry && p.flags[n] != Fluid) {
    p.gx[n] = Real(0); p.gy[n] = Real(0); p.gz[n] = Real(0);
    p.rx[n] = Real(0); p.ry[n] = Real(0); p.rz[n] = Real(0);
    return;
  }
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  const Real p0 = p.phi[n], r0 = p.rho_r[n] + p.rho_b[n];
  Real gp[3] = {Real(0), Real(0), Real(0)};
  Real gr[3] = {Real(0), Real(0), Real(0)};
  for (int i = 1; i < 27; ++i) {
    const Real w = ColourLattice::w(i);
    const long j = neighbour<ColourLattice>(x, y, z, i, p.nx, p.ny, p.nz);
    const bool wet = HasGeometry ? (p.flags[j] == Fluid) : true;
    const Real pj = wet ? p.phi[j] : p0;
    const Real rj = wet ? (p.rho_r[j] + p.rho_b[j]) : r0;
    gp[0] += w * pj * Real(ColourLattice::cx(i));
    gp[1] += w * pj * Real(ColourLattice::cy(i));
    gp[2] += w * pj * Real(ColourLattice::cz(i));
    gr[0] += w * rj * Real(ColourLattice::cx(i));
    gr[1] += w * rj * Real(ColourLattice::cy(i));
    gr[2] += w * rj * Real(ColourLattice::cz(i));
  }
  const Real ics = ColourLattice::inv_cs2();
  p.gx[n] = ics * gp[0];  p.gy[n] = ics * gp[1];  p.gz[n] = ics * gp[2];
  p.rx[n] = ics * gr[0];  p.ry[n] = ics * gr[1];  p.rz[n] = ics * gr[2];
}

//------------------------------------------------------------------------------
// PASS 3. Stream, collide, perturb, recolour, stream.
//
// Esoteric Pull's bounce-back is the identity on the storage, so a solid cell is
// skipped outright. Doing anything at all here -- zeroing it, writing
// equilibrium into it -- would destroy the populations it is holding in transit
// for its fluid neighbours.
//------------------------------------------------------------------------------
template <int Parity, bool HasGeometry>
LBM_HD LBM_INLINE void colour_node_update(const ColourParams& p, long N, long n) {
  const std::uint8_t fl = HasGeometry ? p.flags[n] : std::uint8_t(Fluid);
  if (fl != Fluid) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real fr[27], fb[27];
  gather<Parity, ColourLattice>(p.fr, N, x, y, z, p.nx, p.ny, p.nz, fr);
  gather<Parity, ColourLattice>(p.fb, N, x, y, z, p.nx, p.ny, p.nz, fb);

  Real f[27];
  for (int i = 0; i < 27; ++i) f[i] = fr[i] + fb[i];

  const Real srr = p.rho_r[n], srb = p.rho_b[n];
  const Real u[3] = {p.ux[n], p.uy[n], p.uz[n]};
  const Real g[3] = {p.gx[n], p.gy[n], p.gz[n]};
  const Real dr[3] = {p.rx[n], p.ry[n], p.rz[n]};
  const Real ph = p.phi[n];

  colour_collide(p.m, f, srr, srb, u, ph, g, dr);
  colour_recolour(p.m, f, srr, srb, g, fr, fb);

  scatter<Parity, ColourLattice>(p.fr, N, x, y, z, p.nx, p.ny, p.nz, fr);
  scatter<Parity, ColourLattice>(p.fb, N, x, y, z, p.nx, p.ny, p.nz, fb);
}

#if defined(__CUDACC__)

template <int Parity, bool HasGeometry>
__global__ void colour_fields_kernel(ColourParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  colour_fields_node<Parity, HasGeometry>(p, N, n);
}

template <bool HasGeometry>
__global__ void colour_gradient_kernel(ColourParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  colour_gradient_node<HasGeometry>(p, n);
}

template <int Parity, bool HasGeometry>
__global__ void colour_kernel(ColourParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  colour_node_update<Parity, HasGeometry>(p, N, n);
}

//------------------------------------------------------------------------------
// Seed both colours at rest, from a caller-supplied (rho_r, rho_b) per node.
//
// init_scatter, not scatter: the latter streams, so using it to lay down an
// initial condition shifts every population by one cell before the first step.
//------------------------------------------------------------------------------
template <class Init>
__global__ void colour_initialise(Real* __restrict__ fr, Real* __restrict__ fb,
                                  int nx, int ny, int nz, ColourModel m,
                                  Init init) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int x, y, z;
  coords(n, nx, ny, x, y, z);
  Real rr = Real(1), rb = Real(0);
  init(x, y, z, rr, rb);
  Real gr[27], gb[27];
  for (int i = 0; i < 27; ++i) {
    // At rest the whole equilibrium is the rest term, and it is per colour:
    // red gets rho_r phi_i(alpha_r), blue gets rho_b phi_i(alpha_b).
    gr[i] = rr * ColourModel::phi_i(i, m.alpha_r);
    gb[i] = rb * ColourModel::phi_i(i, m.alpha_b);
  }
  init_scatter<0, ColourLattice>(fr, N, x, y, z, nx, ny, nz, gr);
  init_scatter<0, ColourLattice>(fb, N, x, y, z, nx, ny, nz, gb);
}

//==============================================================================
//  Host-side driver.
//==============================================================================
class ColourSolver {
 public:
  ColourSolver(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    N_ = long(nx) * ny * nz;
    const std::size_t pop = sizeof(Real) * 27 * std::size_t(N_);
    const std::size_t fld = sizeof(Real) * std::size_t(N_);
    LBM_CUDA_CHECK(cudaMalloc(&fr_, pop));
    LBM_CUDA_CHECK(cudaMalloc(&fb_, pop));
    for (int k = 0; k < 12; ++k) LBM_CUDA_CHECK(cudaMalloc(&field_[k], fld));
    LBM_CUDA_CHECK(cudaMalloc(&flags_, sizeof(std::uint8_t) * std::size_t(N_)));
    LBM_CUDA_CHECK(cudaMemset(flags_, Fluid, sizeof(std::uint8_t) * std::size_t(N_)));
  }
  ~ColourSolver() {
    cudaFree(fr_); cudaFree(fb_);
    for (int k = 0; k < 12; ++k) cudaFree(field_[k]);
    cudaFree(flags_);
  }
  ColourSolver(const ColourSolver&) = delete;
  ColourSolver& operator=(const ColourSolver&) = delete;

  ColourModel model;

  void set_geometry(const std::vector<std::uint8_t>& flags) {
    LBM_CUDA_CHECK(cudaMemcpy(flags_, flags.data(),
                              sizeof(std::uint8_t) * std::size_t(N_),
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;
  }

  // init(x, y, z, rho_r&, rho_b&)
  template <class Init>
  void initialise_with(Init init) {
    const int B = 128;
    colour_initialise<<<int((N_ + B - 1) / B), B>>>(fr_, fb_, nx_, ny_, nz_,
                                                    model, init);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    t_ = 0;
    refresh();
  }

  // Passes 1 and 2. Must precede step(): the collision reads fields the step
  // kernel cannot compute for itself, because the value it would compute is
  // consumed and overwritten in the same launch.
  void refresh() {
    if (t_ % 2 == 0) launch_fields<0>(); else launch_fields<1>();
    launch_gradients();
  }

  void step() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) {
      if (has_geometry_) colour_kernel<0, true><<<G, B>>>(params(), N_);
      else               colour_kernel<0, false><<<G, B>>>(params(), N_);
    } else {
      if (has_geometry_) colour_kernel<1, true><<<G, B>>>(params(), N_);
      else               colour_kernel<1, false><<<G, B>>>(params(), N_);
    }
    LBM_CUDA_CHECK(cudaGetLastError());
    ++t_;
  }

  void field_to_host(const Real* src, std::vector<Real>& out) {
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    out.resize(std::size_t(N_));
    LBM_CUDA_CHECK(cudaMemcpy(out.data(), src, sizeof(Real) * std::size_t(N_),
                              cudaMemcpyDeviceToHost));
  }

  const Real* rho_red_device()  const { return field_[0]; }
  const Real* rho_blue_device() const { return field_[1]; }
  const Real* phi_device()      const { return field_[2]; }
  const Real* ux_device()       const { return field_[3]; }
  const Real* uy_device()       const { return field_[4]; }
  const Real* uz_device()       const { return field_[5]; }
  std::size_t timestep() const { return t_; }
  long nodes() const { return N_; }

 private:
  template <int P> void launch_fields() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (has_geometry_) colour_fields_kernel<P, true><<<G, B>>>(params(), N_);
    else               colour_fields_kernel<P, false><<<G, B>>>(params(), N_);
    LBM_CUDA_CHECK(cudaGetLastError());
  }
  void launch_gradients() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (has_geometry_) colour_gradient_kernel<true><<<G, B>>>(params(), N_);
    else               colour_gradient_kernel<false><<<G, B>>>(params(), N_);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  ColourParams params() const {
    ColourParams p;
    p.fr = fr_; p.fb = fb_;
    p.rho_r = field_[0]; p.rho_b = field_[1]; p.phi = field_[2];
    p.ux = field_[3]; p.uy = field_[4]; p.uz = field_[5];
    p.gx = field_[6]; p.gy = field_[7]; p.gz = field_[8];
    p.rx = field_[9]; p.ry = field_[10]; p.rz = field_[11];
    p.flags = flags_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.m = model;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real* fr_ = nullptr;
  Real* fb_ = nullptr;
  Real* field_[12] = {};
  std::uint8_t* flags_ = nullptr;
  bool has_geometry_ = false;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
