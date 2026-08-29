#pragma once
//==============================================================================
//  Colour-gradient two-component flow, with nonorthogonal central moments.
//
//  Saito, De Rosis, Festuccia, Kaneko, Abe & Koyama, "Color-gradient lattice
//  Boltzmann model with nonorthogonal central moments: Hydrodynamic melt-jet
//  breakup simulations", Phys. Rev. E 98, 013305 (2018). Equation numbers below
//  are that paper's. D3Q27 only, which is what the paper derives.
//
//  A SECOND ROUTE TO TWO-PHASE FLOW, and a genuinely different one from the
//  phase field of PhaseFieldSolver.hpp. There the interface is a field with its
//  own advection equation and its own distribution set; here there is no
//  interface equation at all. Two distributions f_i^r and f_i^b are carried, the
//  interface is wherever both are non-zero, and it is held together by a
//  RECOLOURING step that redistributes the colour-blind post-collision state
//  back into red and blue along the colour gradient. Nothing diffuses and
//  nothing is reinitialised: segregation is algebraic.
//
//  Which one to use is not settled by this file. The phase field has a
//  conservative advection equation and a prescribed interface width; the colour
//  gradient has neither, and its interface width is an outcome. What the colour
//  gradient has instead is the density ratio: it reaches 1000 in the paper's own
//  static tests, where the pressure form of MultiphasePotentialBGK.hpp is
//  fighting the conditioning problem of its own banner by ratio 100.
//
//  THE THREE SUBOPERATORS, Eq. (10):
//
//      Omega_i^k = (Omega_i^k)^(3) [ (Omega_i)^(1) + (Omega_i)^(2) ],
//
//  read right to left: collide the colour-blind f_i = f_i^r + f_i^b, perturb it
//  to make surface tension, then recolour. Only the third carries a colour
//  index, which is the whole economy of the scheme -- the expensive part, the
//  central-moment collision, is done ONCE for both fluids.
//
//  (1) THE SINGLE-PHASE COLLISION is a general MRT in central moments,
//  Eq. (11), with the relaxation matrix of Eq. (15),
//
//      K = diag[s0, s1 x3, s2v x5, s2b, s3 x7, s4 x6, s5 x3, s6],
//
//  and s0 = s1 = 0, s2b = s3 = s4 = s5 = s6 = 1. That grouping is ALREADY what
//  MomentCollision.hpp does: five second-order moments at the shear rate, the
//  trace on its own at the bulk rate, everything above second order straight to
//  equilibrium. The paper reaches it through De Rosis's nonorthogonal basis and
//  this code reaches it through a Hermite product basis, and above second order
//  the two cannot disagree -- when every moment of a subspace relaxes at the
//  same rate, any basis of that subspace gives the same operator. At second
//  order the split into trace and traceless deviatoric is basis-independent for
//  the same reason. So ProductBasis is reused unchanged and only the
//  EQUILIBRIUM is new.
//
//  IT IS NEW IN A WAY THAT MATTERS. Eq. (18) is not the product-form
//  equilibrium, so its moments have no factorised closed form and the
//  eq_moment() of MomentCollision.hpp does not apply. This operator builds the
//  equilibrium POPULATIONS and transforms them, which costs one extra forward
//  transform per node and is exactly right by construction. A closed form could
//  be derived; until it is measured to matter, it would be a second thing to
//  keep correct.
//
//  THE DENSITY RATIO LIVES IN THE REST WEIGHT. phi_i of Eq. (20) replaces w_i in
//  the rest term of the equilibrium, and carries a free parameter alpha:
//
//      phi_0 = alpha,   phi_{|c|^2=1} = 2(1-alpha)/19,
//      phi_{|c|^2=2} = (1-alpha)/38,   phi_{|c|^2=3} = (1-alpha)/152,
//
//  which sums to one for any alpha, and whose second moment is
//  cs^2 = 9(1-alpha)/19. The sound speed is therefore a PROPERTY OF THE PHASE.
//  Pressure continuity across the interface, p = rho cs^2, then fixes the
//  density ratio without a single density appearing in the collision, Eq. (25):
//
//      gamma = rho_r^0 / rho_b^0 = (1 - alpha_b) / (1 - alpha_r).
//
//  alpha_b = 8/27 recovers cs^2 = 1/3 exactly for the blue fluid, which is the
//  paper's choice and the one that makes the blue phase an ordinary lattice.
//  Note what this does NOT do: it does not make cs^2 = 1/3 anywhere else, so the
//  standard lattice sound speed is not available as a global constant here and
//  nothing in this file assumes it. ProductBasis's cs^2 is a basis constant, not
//  a physical one, and the two are deliberately not the same number.
//
//  Phi_i of Eq. (21) is the last term of the equilibrium and the one that is
//  easiest to drop by accident. It restores Galilean invariance when the density
//  varies, through the second moment, using
//
//      G = (1/48) [ u (x) grad rho + (u (x) grad rho)^T ].
//
//  It is the same physics that ViscousInterfaceForce.hpp adds to the phase-field
//  module as an explicit body force nu (grad u + grad u^T) . grad rho -- the same
//  missing stress, entered by a different door. Here it costs no extra field
//  beyond grad rho, because it goes into the equilibrium rather than into F.
//
//  (2) THE PERTURBATION makes the surface tension. Eq. (30) is written PER
//  COLOUR,
//
//      (Omega_i^k)^(2) = (A_k/2) |grad phi| [ w_i (c_i . n)^2 - B_i ],
//
//  and Eq. (39) sums the capillary stress over k as well as over i, with the
//  paper taking A = A_r = A_b. Applied to the colour-blind population -- which is
//  what this code does, and what the paper's own text says is done -- the two
//  halves add and the coefficient is A, NOT A/2:
//
//      (Omega_i)^(2) = A |grad phi| [ w_i (c_i . n)^2 - B_i ].
//
//  THAT FACTOR IS NOT COSMETIC and it is the one thing here most likely to be
//  got wrong, because A/2 is what Eq. (30) literally reads and it is off by
//  exactly two. It was caught by deriving sigma from Eq. (39) rather than
//  copying Eq. (32): the capillary stress of a flat interface gives
//
//      sigma = integral (S_tt - S_nn) dn = (2 A tau / 9) * delta(phi),
//
//  and phi runs from -1 to +1, so delta(phi) = 2 and sigma = 4 A tau / 9, which
//  is Eq. (32) exactly. With A/2 it would have come out half that, and a static
//  droplet would have reported a surface tension 50% low with every other
//  property of the model intact.
//
//  n is the unit colour gradient and B_i the lattice constants of Eq. (31). The
//  operator conserves mass and momentum exactly, and it acts only ACROSS the
//  interface: sum_i w_i (c_i.n)^4 = 3 cs^4 = 1/3 cancels sum_i B_i (c_i.n)^2 =
//  1/3 identically, so the NORMAL component of the capillary stress vanishes for
//  every direction of n, while the tangential component does not. Both are
//  direction-independent because D3Q27's fourth moment is isotropic.
//
//  (3) THE RECOLOURING, Eqs. (33)-(34), splits the post-collision f_i back:
//
//      f_i^r = (rho_r/rho) f_i + beta (rho_r rho_b / rho^2) cos(theta_i) f_i^eq(rho,0),
//      f_i^b = (rho_b/rho) f_i - beta (rho_r rho_b / rho^2) cos(theta_i) f_i^eq(rho,0),
//
//  with cos(theta_i) the cosine of the angle between c_i and grad phi. The two
//  sum to f_i identically, so mass and momentum survive it whatever beta is; all
//  beta does is push colour up the gradient. beta = 0.7 is the paper's value and
//  the largest that keeps the interface smooth. f_i^eq(rho,0) is simply rho phi_i
//  -- at zero velocity every other term of Eq. (18) vanishes, Phi_i included.
//
//  WHAT THIS OPERATOR DOES NOT DO. It does not stream, it does not own the two
//  distributions, and it does not compute a gradient: with Esoteric Pull the
//  neighbour populations during a fused kernel are a mixture of two time levels
//  and a gradient taken from them is meaningless. grad phi and grad rho are node
//  FIELDS, computed by ColourGradientSolver in a separate pass, for the same
//  reason PhaseFieldSolver.hpp gives at length.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/ProductBasis.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

template <class L>
struct ColourGradient {
  using Lattice = L;
  using Storage = RawPopulations;      // two colours; a w_i shift fits neither
  using Basis   = ProductBasis<L>;
  static constexpr const char* name = "ColourGradientCM";
  static constexpr int D  = L::D;
  static constexpr int NM = Basis::NM;

  static_assert(L::D == 3 && L::Q == 27,
                "the colour-gradient model of Saito et al. (2018) is derived for "
                "D3Q27; phi_i, B_i and sigma = 4 A tau / 9 are all lattice-specific.");
  static_assert(Basis::enabled, "no product basis for this lattice.");

  //---- parameters ------------------------------------------------------------
  Real alpha_r = Real(8) / Real(27);   // rest weights; gamma = (1-ab)/(1-ar)
  Real alpha_b = Real(8) / Real(27);
  Real nu_r = Real(1) / Real(6);       // kinematic viscosities, harmonic-mixed
  Real nu_b = Real(1) / Real(6);
  Real A = Real(0);                    // interfacial tension strength, Eq. (32)
  Real beta = Real(0.7);               // recolouring sharpness
  Real omega_bulk = Real(1);           // s2b
  Real bx = 0, by = 0, bz = 0;         // body force per unit mass
  // The density the body force is measured against: F = (rho - rho_ref) b.
  // Zero gives the plain rho b. In a FULLY PERIODIC box rho b injects net
  // momentum every step -- there is no wall for the hydrostatic pressure
  // gradient to push against and no boundary to absorb it, so the whole fluid
  // simply falls. Setting rho_ref to the domain mean removes the mean force and
  // leaves the buoyancy difference, which is the part a Rayleigh-Taylor problem
  // is actually about. It is not a Boussinesq approximation: the full density
  // difference is retained, only its mean is subtracted.
  Real rho_ref = 0;

  // The initial density of each PURE phase, Eq. (23). Carried explicitly rather
  // than derived from alpha, even though equal pressure ties them together as
  // rho^0 = const / (1 - alpha): the derived form inverts easily and does so
  // SILENTLY. Getting it upside down leaves both pure phases at phi = +-1 --
  // every bulk check still passes -- and corrupts only the interface, where the
  // weighting between the two colours decides alpha and therefore the local
  // pressure. Measured at a ratio of 10 it dropped the droplet's density from
  // 10 to 7.95 and put the Laplace tension out by 91%; at 100 it went to NaN.
  Real rho_r0 = Real(1);
  Real rho_b0 = Real(1);

  //---- node fields, written by the solver ------------------------------------
  View1D<const Real> phi;              // order parameter, Eq. (23)
  View1D<const Real> Gx, Gy, Gz;       // grad phi
  View1D<const Real> Rx, Ry, Rz;       // grad rho, for Phi_i of Eq. (21)

  //----------------------------------------------------------------------------
  // sigma = (4/9) A tau, Eq. (32), and its inverse. tau = 1/s2v is the SHEAR
  // relaxation time of the mixture, so a viscosity ratio makes sigma depend on
  // where the interface sits; the paper's tests use a matched tau and so does
  // the validation case.
  //----------------------------------------------------------------------------
  static Real A_from_sigma(Real sigma, Real tau) {
    return Real(9) * sigma / (Real(4) * tau);
  }
  static Real sigma_from_A(Real a, Real tau) {
    return Real(4) * a * tau / Real(9);
  }
  // The coefficient the colour-blind perturbation actually carries -- A, not
  // A/2; see the banner. Exposed so the test can assert the relation rather
  // than reproduce the constant.
  static constexpr Real perturbation_coefficient(Real a) { return a; }

  //----------------------------------------------------------------------------
  // The order parameter, Eq. (23). +1 in pure red, -1 in pure blue, 0 where the
  // two are present in equal PROPORTION of their own bulk densities -- which is
  // not the same as equal mass, and is the whole point of dividing by rho^0.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real order_parameter(Real rr, Real rb) const {
    const Real a = rr / rho_r0, b = rb / rho_b0;
    const Real s = a + b;
    return (s > Real(0)) ? (a - b) / s : Real(0);
  }

  // cs^2 = 9(1-alpha)/19, the second moment of phi_i. NOT the lattice constant.
  KOKKOS_INLINE_FUNCTION static Real cs2_of_alpha(Real a) {
    return Real(9) * (Real(1) - a) / Real(19);
  }
  // gamma = (1 - alpha_b)/(1 - alpha_r), Eq. (25), inverted for alpha_r.
  static Real alpha_r_from_ratio(Real gamma, Real ab) {
    return Real(1) - (Real(1) - ab) / gamma;
  }

  //----------------------------------------------------------------------------
  // phi_i, Eq. (20). Keyed off |c_i|^2, which on this lattice is 0, 1, 2 or 3.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION static Real phi_i(int i, Real a) {
    const int q = cvel<L>(i, 0) * cvel<L>(i, 0) + cvel<L>(i, 1) * cvel<L>(i, 1)
                + cvel<L>(i, 2) * cvel<L>(i, 2);
    const Real c = Real(1) - a;
    if (q == 0) return a;
    if (q == 1) return Real(2) * c / Real(19);
    if (q == 2) return c / Real(38);
    return c / Real(152);
  }

  // B_i, Eq. (31). sum_i B_i = 1/3, which is what makes the perturbation
  // conserve mass against sum_i w_i (c_i.n)^2 = 1/3.
  KOKKOS_INLINE_FUNCTION static Real B_i(int i) {
    const int q = cvel<L>(i, 0) * cvel<L>(i, 0) + cvel<L>(i, 1) * cvel<L>(i, 1)
                + cvel<L>(i, 2) * cvel<L>(i, 2);
    if (q == 0) return Real(-10) / Real(27);
    if (q == 1) return Real(2) / Real(27);
    if (q == 2) return Real(1) / Real(54);
    return Real(1) / Real(216);
  }

  // The coefficient of (G : c_i x c_i) in Phi_i, Eq. (21).
  KOKKOS_INLINE_FUNCTION static Real Phi_coeff(int i) {
    const int q = cvel<L>(i, 0) * cvel<L>(i, 0) + cvel<L>(i, 1) * cvel<L>(i, 1)
                + cvel<L>(i, 2) * cvel<L>(i, 2);
    if (q == 0) return Real(0);
    if (q == 1) return Real(16);
    if (q == 2) return Real(4);
    return Real(1);
  }

  //----------------------------------------------------------------------------
  // Interpolants across the interface. alpha and the viscosity are mixed
  // differently on purpose: alpha linearly, Eq. (27), and the viscosity
  // HARMONICALLY, Eq. (22). The harmonic mean is not a stylistic choice -- it is
  // what keeps the shear stress continuous across an interface with a viscosity
  // ratio, and a linear mean there produces a jump in stress that shows up as a
  // spurious current.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real alpha_at(Real p) const {
    return Real(0.5) * ((Real(1) + p) * alpha_r + (Real(1) - p) * alpha_b);
  }
  KOKKOS_INLINE_FUNCTION Real nu_at(Real p) const {
    const Real inv = Real(0.5) * ((Real(1) + p) / nu_r + (Real(1) - p) / nu_b);
    return Real(1) / inv;
  }
  //----------------------------------------------------------------------------
  // The shear rate, Eq. (16): nu = (c^2/3)(1/s2v - 1/2). THE 1/3 IS THE
  // LATTICE'S, not the phase's, and the distinction is the single most
  // consequential detail in this operator.
  //
  // cs^2 appears in two unrelated roles here and they are different numbers.
  // The EQUATION OF STATE is p = rho cs^2(alpha) with cs^2(alpha) = 9(1-alpha)/19,
  // because the rest weight phi_i carries the density ratio and the
  // equilibrium's trace follows it. The VISCOUS STRESS does not: it comes from
  // the non-equilibrium part, whose second moment is governed by the standard
  // weights w_i in the velocity-dependent terms of Eq. (18), and those carry
  // cs^2 = 1/3 in every phase.
  //
  // Using the phase's cs^2 here instead was measured, and it fails in the way
  // that is hardest to attribute: at a density ratio of 1000 it drives omega to
  // 1.992 in the middle of the interface -- inside the stability limit by a
  // hair, wrong by a factor of 500 in the viscosity, and the droplet returns
  // NaN a few hundred steps later with nothing in between to point at it.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real omega_at(Real p) const {
    return Real(1) / (nu_at(p) * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real viscosity_from_tau(Real tau) {
    return cs2<L, Real>() * (tau - Real(0.5));
  }

  //----------------------------------------------------------------------------
  // The equilibrium, Eq. (18) with the Phi_i of Eq. (21).
  //
  // Third order in u, and that is deliberate in the source: the O(u^3) terms are
  // what make the model Galilean invariant at a density ratio, which is the
  // regime it exists for.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void equilibrium(Real fe[L::Q], Real rho_r, Real rho_b, const Real u[3],
                   const Real G[3][3], Real nubar, Real udrho) const {
    const Real rho = rho_r + rho_b;
    const Real u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    for (int i = 0; i < L::Q; ++i) {
      const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)),
                 cz = Real(cvel<L>(i, 2));
      const Real cu = cx * u[0] + cy * u[1] + cz * u[2];
      const Real w  = weight<L, Real>(i);
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

  //----------------------------------------------------------------------------
  // THE REST TERM, and the one reading in this paper that had to be worked out
  // rather than copied.
  //
  // Eq. (18) writes the equilibrium's rest contribution as rho phi_i, and Eq.
  // (27) interpolates alpha linearly in phi, which together read as
  // rho phi_i(alpha-bar). That cannot be what is meant. The trace of that
  // equilibrium is rho cs^2(alpha-bar), and with the tanh density profiles of
  // Eqs. (41)-(42) it is NOT continuous through the interface: at a density
  // ratio of 100 the midpoint carries p = 8.5 against a bulk 1/3, a
  // twenty-five-fold pressure spike two cells wide. Measured, that is exactly
  // what happens -- gamma = 1 and 10 survive it and 100 and 1000 return NaN.
  //
  // The rest term is PER COLOUR:
  //
  //      sum_k rho_k phi_i(alpha_k),
  //
  // whose trace is sum_k rho_k cs_k^2 and which is continuous by construction --
  // it is identically 1/3 everywhere through the interface for the seeded
  // profiles. It is also what Eq. (26) says, p = rho_k * 9(1-alpha)/19, with the
  // colour subscript it carries; and it is the ONLY reading consistent with Eq.
  // (25), since gamma = (1-alpha_b)/(1-alpha_r) is derived from balancing
  // rho_r^0 cs_r^2 against rho_b^0 cs_b^2, each phase with its own alpha.
  //
  // At a matched ratio the two readings coincide exactly (alpha_r = alpha_b), so
  // nothing that worked before is disturbed -- which is also why the error hid
  // until a density ratio was asked for.
  //
  // alpha-bar and Eq. (27) are still used, for the viscosity blend and for the
  // recolouring weight, where a single mixture alpha is what is wanted.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real rest_term(int i, Real rho_r, Real rho_b) const {
    return rho_r * phi_i(i, alpha_r) + rho_b * phi_i(i, alpha_b);
  }

  // f_i^eq(rho_r, rho_b, u = 0): every other term of Eq. (18) carries a u or a
  // G, and G vanishes with u, so the rest term is the whole equilibrium at rest.
  // This is what the recolouring is written against.
  KOKKOS_INLINE_FUNCTION Real eq_at_rest(int i, Real rho_r, Real rho_b) const {
    return rest_term(i, rho_r, rho_b);
  }

  //----------------------------------------------------------------------------
  // Suboperators (1) and (2), on the colour-blind populations.
  //
  // `rho` and `u` are the mixture's, `p` the order parameter, `n` the node.
  // f is overwritten with the post-collision, post-perturbation state; the
  // caller recolours it.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], Real rho_r, Real rho_b, const Real u[3], Real p,
               Index n) const {
    const Real rho   = rho_r + rho_b;
    const Real nubar = nu_at(p);
    const Real omega = omega_at(p);

    // ---- G and u.grad(rho) for Phi_i, Eq. (24) ----
    const Real dr[3] = {Rx(n), Ry(n), Rz(n)};
    Real G[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        G[i][j] = (u[i] * dr[j] + u[j] * dr[i]) / Real(48);
    const Real udrho = u[0] * dr[0] + u[1] * dr[1] + u[2] * dr[2];

    Real fe[L::Q];
    equilibrium(fe, rho_r, rho_b, u, G, nubar, udrho);

    // ---- (2) the perturbation, Eq. (30), added to the populations before the
    // transform: it is a second-moment source and the transform carries it into
    // the right slots without it having to be written there by hand.
    const Real g[3] = {Gx(n), Gy(n), Gz(n)};
    const Real gm2  = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    Real pert[L::Q];
    if (gm2 > Real(1e-24) && A != Real(0)) {
      const Real gm   = Kokkos::sqrt(gm2);
      const Real invm = Real(1) / gm;
      const Real nh[3] = {g[0] * invm, g[1] * invm, g[2] * invm};
      for (int i = 0; i < L::Q; ++i) {
        const Real cn = Real(cvel<L>(i, 0)) * nh[0] + Real(cvel<L>(i, 1)) * nh[1]
                      + Real(cvel<L>(i, 2)) * nh[2];
        pert[i] = A * gm * (weight<L, Real>(i) * cn * cn - B_i(i));
      }
    } else {
      for (int i = 0; i < L::Q; ++i) pert[i] = Real(0);
    }

    // ---- (1) the central-moment collision ----
    const Real ub[3] = {u[0], u[1], u[2]};
    Real k[NM], ke[NM], kp[NM];
    Basis::template to_moments<true>(f,    ub, k);
    Basis::template to_moments<true>(fe,   ub, ke);
    Basis::template to_moments<true>(pert, ub, kp);

    const Real fw = rho - rho_ref;
    Real F[3] = {fw * bx, fw * by, fw * bz};

    // order 1: conserved, plus the body force. s0 = s1 = 0 in Eq. (15) means the
    // collision leaves them alone; the force is the only thing that moves them.
    for (int a2 = 0; a2 < D; ++a2) k[i1(a2)] += F[a2] + kp[i1(a2)];

    // order 2: trace at s2b, deviatoric and shear at s2v. The perturbation is
    // NOT relaxed -- it is a source, and the (1 - s/2) factor that a Guo force
    // carries does not apply to it: Eq. (39) defines the capillary stress as
    // -tau sum_i Omega^(2) c_i c_i, which is the UNRELAXED second moment
    // integrated over one relaxation time. Applying (1 - s/2) here would put
    // sigma out by that factor, and the static droplet would report it.
    {
      Real d[3], e[3], q[3];
      Real tr = 0, tre = 0, trq = 0;
      for (int a2 = 0; a2 < D; ++a2) {
        d[a2] = k[i2d(a2)];  e[a2] = ke[i2d(a2)];  q[a2] = kp[i2d(a2)];
        tr += d[a2];  tre += e[a2];  trq += q[a2];
      }
      const Real invD = Real(1) / Real(D);
      const Real tr_post = (Real(1) - omega_bulk) * tr + omega_bulk * tre + trq;
      for (int a2 = 0; a2 < D; ++a2)
        k[i2d(a2)] = (Real(1) - omega) * (d[a2] - tr * invD)
                   + omega * (e[a2] - tre * invD)
                   + (q[a2] - trq * invD) + tr_post * invD;
      for (int a2 = 0; a2 < D; ++a2)
        for (int b2 = a2 + 1; b2 < D; ++b2) {
          const int id = i2s(a2, b2);
          k[id] = (Real(1) - omega) * k[id] + omega * ke[id] + kp[id];
        }
    }

    // order >= 3: s3 = s4 = s5 = s6 = 1, so straight to equilibrium.
    for (int m = 0; m < NM; ++m)
      if (Basis::order(m) >= 3) k[m] = ke[m] + kp[m];

    Basis::template to_populations<true>(k, ub, f);
  }

  //----------------------------------------------------------------------------
  // Suboperator (3), Eqs. (33)-(34). The two outputs sum to f_i identically, so
  // this cannot lose mass or momentum however wrong beta is.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void recolour(const Real f[L::Q], Real rho_r, Real rho_b, Real p, Index n,
                Real fr[L::Q], Real fb[L::Q]) const {
    const Real rho = rho_r + rho_b;
    const Real inv = (rho > Real(0)) ? Real(1) / rho : Real(0);
    const Real mix = beta * rho_r * rho_b * inv * inv;
    (void)p;

    const Real g[3] = {Gx(n), Gy(n), Gz(n)};
    const Real gm2  = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    const Real gm   = (gm2 > Real(1e-24)) ? Kokkos::sqrt(gm2) : Real(0);

    for (int i = 0; i < L::Q; ++i) {
      Real cosine = Real(0);
      if (gm > Real(0)) {
        const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)),
                   cz = Real(cvel<L>(i, 2));
        const Real cm2 = cx * cx + cy * cy + cz * cz;
        if (cm2 > Real(0))
          cosine = (cx * g[0] + cy * g[1] + cz * g[2])
                 / (Kokkos::sqrt(cm2) * gm);
      }
      const Real split = mix * cosine * eq_at_rest(i, rho_r, rho_b);
      fr[i] = rho_r * inv * f[i] + split;
      fb[i] = rho_b * inv * f[i] - split;
    }
  }

  //----------------------------------------------------------------------------
  // Moment slots, located through the basis rather than hardcoded.
  //----------------------------------------------------------------------------
  static constexpr int i1(int a) {
    return Basis::index_of(a == 0, a == 1, a == 2);
  }
  static constexpr int i2d(int a) {
    return Basis::index_of(2 * (a == 0), 2 * (a == 1), 2 * (a == 2));
  }
  static constexpr int i2s(int a, int b) {
    return Basis::index_of((a == 0 || b == 0), (a == 1 || b == 1),
                           (a == 2 || b == 2));
  }
};

}  // namespace lbm
