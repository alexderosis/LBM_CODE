#pragma once
//==============================================================================
//  Phase-field two-phase flow at a density ratio, on the GPU.
//
//  De Rosis & Enan, Phys. Fluids 33, 043315 (2021). Equation numbers are that
//  paper's. A port of ../src/collision/PhaseFieldBGK.hpp,
//  ../src/collision/MultiphasePotentialBGK.hpp, ../src/solver/
//  PhaseFieldSolver.hpp and ../src/solver/ViscousInterfaceForce.hpp.
//
//  THE SECOND MULTIPHASE ENGINE HERE, and a genuinely different one from
//  colour.cuh. There the interface is wherever both colours are non-zero and
//  segregation is algebraic; here the interface is a FIELD with its own
//  advection equation, its own distribution set and a prescribed width W. The
//  colour gradient reaches a density ratio through the rest weight; this one
//  reaches it by moving the density out of the distribution entirely.
//
//  THREE DISTRIBUTIONS' WORTH OF WORK, ON TWO LATTICES:
//
//    f_i on D3Q27   the fluid, carrying a normalised PRESSURE
//    h_i on D3Q7    the phase field, carrying phi in [0, 1]
//    grad phi and lap phi on D3Q27, by stencil from the phi FIELD
//
//  WHY THE ZEROTH MOMENT CARRIES PRESSURE. At a density ratio rho is no longer
//  a small perturbation about 1 and a density-carrying distribution stops being
//  well conditioned. The incompressible model moves rho out of the distribution:
//  rho is a function of phi (Eq. 25, linear interpolation) and the populations
//  carry
//
//      p~ = sum_i f_i,      u = sum_i c_i f_i + F/(2 rho),             (24)
//
//  with the equilibrium of Eq. (10), f_i^eq = w_i [p~ + Phi_i(u)], whose zeroth
//  moment is p~ exactly and whose FIRST MOMENT IS u, NOT rho u. There is no
//  division by rho in the momentum sum; only the half-force correction carries
//  one. Getting that wrong is a viscosity error that reads like a bad boundary.
//
//  THE COUPLING ORDER IS INSIDE THIS CLASS ON PURPOSE. The parent documents it
//  as the thing a driver gets wrong, so here step() owns it rather than a
//  caller:
//
//      1  phi from h                     (read-only, so any parity)
//      2  grad phi and lap phi from phi  (needs 1 complete at the NEIGHBOURS)
//      3  the viscous interface force    (needs u(t-1) and grad phi(t))
//      4  grad p~                        (constant-reference form only)
//      5  the fluid step                 (collides against grad phi(t), writes u(t))
//      6  the phase step                 (advects h with u(t))
//
//  Refreshing phi AFTER the fluid would be a first-order splitting error that
//  does NOT refine away, and it is worse in kind than the usual one: the
//  interface MOVES, so the error is a systematically misplaced interface rather
//  than a damping offset. Six kernel launches on one stream give six fences and
//  there is no cudaDeviceSynchronize on the step path.
//
//  ONE GATHER SERVES BOTH DERIVATIVES. The Laplacian needs exactly the
//  neighbours the gradient has already loaded, so it is one more accumulator in
//  the same kernel. Eqs. (20)-(21):
//
//      grad phi = (1/cs2) sum_i w_i c_i phi(x + c_i)
//      lap  phi = (2/cs2) sum_i w_i     [phi(x + c_i) - phi(x)]
//
//  taken on D3Q27 rather than the phase field's own D3Q7. That is deliberate:
//  spurious currents around a static droplet are dominated by the isotropy of
//  this stencil and the axis-only one is markedly worse.
//
//  THE ANTI-DIFFUSION SOURCE, AND WHY ITS PREFACTOR IS NOT GUO'S. With
//  sum_i S_i = 0 and sum_i c_i S_i = A_S, Chapman-Enskog gives
//  d_t phi + div(phi u) = M lap phi - (1/omega) div A_S, so A_S = omega M theta n
//  and, using omega M = cs2(1 - omega/2), the source is simply
//
//      S_i = (1 - omega/2) theta w_i (c_i . n),      theta = (4/W) phi (1 - phi)
//
//  carrying no mobility and no cs2 at all. The coefficient is 1/omega, NOT the
//  1/(1 - omega/2) a Guo force compensates: a body force enters the FIRST-moment
//  equation, where the half step is absorbed into the velocity, while this
//  enters the ZEROTH-moment equation through the divergence of the first, so it
//  contributes twice. Getting it wrong is silent in its cause -- the interface
//  simply spreads as though the term were absent.
//
//  TWO PRESSURE NORMALISATIONS, AND NEITHER DOMINATES.
//
//    rho_0 <= 0 (default)  p = rho(phi) cs2 p~,  F_p = -p~ cs2 grad rho
//    rho_0 >  0            p = rho_0    cs2 p~,  F_p =  cs2 (rho - rho_0) grad p~
//
//  The first is the paper's. Because p is continuous while rho is not, p~ jumps
//  by the whole density ratio across an interface, so F_p scales with the
//  pressure LEVEL and is amplified by the ratio. The parent measures the cost on
//  a layered Poiseuille flow: the residual does not converge (order -0.21) and a
//  pure gauge shift in p~, which changes no physics, moves the error
//  seventeenfold. The second normalises by a constant, so F_p scales with the
//  pressure GRADIENT: order 0.95 and a gauge sensitivity of 1.9x on the same
//  case. It is not a free win -- the recovered pressure term becomes
//  -(rho_0/rho) cs2 grad p~, so one phase always pays, and rho_0 MUST be the
//  light phase or the heavy one runs past the lattice speed. Local-rho stays the
//  default because that is what every validated result was obtained with.
//
//  WHAT IS NOT HERE, and absent rather than untested:
//
//   * CENTRAL MOMENTS for the fluid. The parent has
//     MultiphaseCentralMoments.hpp; this is the BGK potential form only.
//   * WETTING. A phase wall is zero-flux on the populations, which is right for
//     the transport, but sets no contact angle: that needs a condition on
//     grad phi at the wall. Do not put an interface against a wall.
//   * AN EXTERNAL FORCE FIELD. The parent carries Ex/Ey/Ez for a penalised
//     solid to write into. Nothing on this side writes one.
//   * OPEN BOUNDARIES for phi.
//
//  MEMORY. f at 27 Real, h at 7, plus phi, grad phi (3), lap, u (3) and, when
//  used, the viscous force (3) and grad p~ (3). At FP32 that is 108 + 28 + 32 =
//  168 bytes per node in the common case, 192 with both optional fields.
//
//  MEASURED ON A T4, FP32, 64^3: 354.6 MLUPS, against the colour gradient's
//  252.5 and the single-phase core's 950 -- six passes and two distributions for
//  37% of a single-phase step. -DLBM_PTXAS_VERBOSE=ON reports 0 bytes of stack
//  frame and no spills on every kernel here (43, 64 and 72 registers for the
//  phase, fluid and derivative passes). That is colour.cuh's lesson applied by
//  construction: nothing above is materialised as an array beyond the one f[27]
//  the collision needs.
//
//  ONE PARAMETER TRAP, MEASURED. A static droplet at a density ratio of 100
//  DIVERGES with the dynamic viscosity matched across the phases, and that is
//  arithmetic rather than the model: matched mu leaves the heavy phase with a
//  hundred-fold smaller kinematic viscosity, so
//  omega = 1/(mu/(rho cs2) + 1/2) = 1.994 against a limit of 2. Matching the
//  KINEMATIC viscosity instead runs the same case to -3.75% of the asked surface
//  tension with a spurious current twenty times smaller. When one of these
//  diverges at a ratio, omega is the first thing to print.
//==============================================================================
#include "streaming.cuh"

namespace lbm {

using PhaseLattice = D3Q7;     // transport
using FluidLattice = D3Q27;    // flow, and the gradient stencil

LBM_HD LBM_INLINE Real pf_sqrt(Real x) {
#if defined(LBM_DOUBLE)
  return ::sqrt(x);
#else
  return ::sqrtf(x);
#endif
}

//------------------------------------------------------------------------------
// Cell roles. The geometry is shared with the fluid but the CONDITIONS are not,
// which is why this has its own enum rather than reusing CellType.
//------------------------------------------------------------------------------
enum PhaseCell : std::uint8_t {
  PhaseBulk     = 0,   // transport
  PhaseWall     = 1,   // zero flux -- bounce-back, and NO wetting condition
  PhaseExcluded = 2,   // not part of the simulation
};

//==============================================================================
//  The conservative Allen-Cahn operator for the phase field.
//==============================================================================
struct PhaseModel {
  Real omega = Real(1);      // sets the mobility
  Real width = Real(4);      // interface width W, in lattice units

  // Eq. (13): tau_phi = M / cs2, omega = 1/(tau_phi + 1/2). NOTE cs2 here is
  // D3Q7's 1/4, not the fluid lattice's 1/3.
  static Real omega_from_mobility(Real m) {
    return Real(1) / (m * PhaseLattice::inv_cs2() + Real(0.5));
  }
  static Real mobility_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * PhaseLattice::cs2();
  }
  LBM_HD LBM_INLINE Real mobility() const {
    return (Real(1) / omega - Real(0.5)) * PhaseLattice::cs2();
  }

  // g_i^eq = w_i phi (1 + c.u/cs2). FIRST order, because D3Q7 has no isotropic
  // fourth-order moment and every velocity has a single non-zero component, so
  // (c.u)^2 collapses to c_a^2 u_a^2 and the cross terms of uu cannot be
  // represented at all. A second-order term there adds anisotropy, not accuracy.
  static LBM_HD LBM_INLINE Real eq(int i, Real phi, Real ux, Real uy, Real uz) {
    const Real cu = Real(PhaseLattice::cx(i)) * ux
                  + Real(PhaseLattice::cy(i)) * uy
                  + Real(PhaseLattice::cz(i)) * uz;
    return PhaseLattice::w(i) * phi * (Real(1) + PhaseLattice::inv_cs2() * cu);
  }

  // A = theta n = (4/W) phi (1 - phi) G/|G|. Purely geometric: no mobility and
  // no cs2, both of which cancelled in the prefactor. For the equilibrium tanh
  // profile A is grad phi itself, which is the cheapest check on this function.
  // Away from the interface phi(1-phi) is zero anyway, so the |G| guard is a
  // second line of defence against 0/0 in the bulk rather than the only one.
  LBM_HD LBM_INLINE void anti_diffusion(Real phi, const Real G[3], Real A[3]) const {
    const Real g2 = G[0]*G[0] + G[1]*G[1] + G[2]*G[2];
    const Real gn = pf_sqrt(g2);
    if (!(gn > Real(1e-12))) { A[0] = A[1] = A[2] = Real(0); return; }
    const Real s = (Real(4) / width) * phi * (Real(1) - phi) / gn;
    A[0] = s*G[0];  A[1] = s*G[1];  A[2] = s*G[2];
  }
};

//==============================================================================
//  The fluid operator: pressure-based, surface tension as a body force.
//==============================================================================
struct MultiphaseModel {
  Real phi_L = Real(0), phi_H = Real(1);
  Real rho_L = Real(1), rho_H = Real(1);
  // Reference density for the pressure normalisation. ZERO (the default) keeps
  // the local-rho form; a positive value selects the constant form and MUST be
  // the light phase or the acoustic CFL is violated. See the banner.
  Real rho_0 = Real(0);
  Real mu_L  = Real(0.1), mu_H = Real(0.1);        // DYNAMIC viscosities
  Real beta = Real(0), kappa = Real(0);
  Real bx = Real(0), by = Real(0), bz = Real(0);   // body acceleration

  // The pair consistent with the tanh profile of width W that the conservative
  // Allen-Cahn equation maintains.
  static Real kappa_from_sigma(Real sigma, Real W) { return Real(1.5) * sigma * W; }
  static Real beta_from_sigma (Real sigma, Real W) { return Real(12) * sigma / W; }

  LBM_HD LBM_INLINE bool constant_reference() const { return rho_0 > Real(0); }
  LBM_HD LBM_INLINE Real drho_dphi() const {
    return (rho_H - rho_L) / (phi_H - phi_L);       // Eq. (23)
  }

  struct Local { Real p, rho, mu, omega; };

  //--------------------------------------------------------------------------
  // Everything the interpolations need, from one read of phi.
  //
  // CLAMPED. The conservative Allen-Cahn form keeps phi close to [0,1] but does
  // not guarantee it: shear against a solid, or any under-resolved feature,
  // overshoots locally. Interpolating rho off an unclamped phi then puts the
  // density outside the two phases it lies between -- and a large enough
  // undershoot makes rho NEGATIVE, at which point 1/rho is unbounded. Clamping
  // the INTERPOLANT rather than phi itself leaves the transported field alone;
  // only the equation of state is bounded.
  //--------------------------------------------------------------------------
  LBM_HD LBM_INLINE Local local(Real p) const {
    Real s = (p - phi_L) / (phi_H - phi_L);
    s = s < Real(0) ? Real(0) : (s > Real(1) ? Real(1) : s);
    const Real r   = rho_L + s * (rho_H - rho_L);            // Eq. (25)
    const Real m   = mu_L  + s * (mu_H  - mu_L);
    const Real tau = m / (r * FluidLattice::cs2());          // Eq. (12)
    return Local{p, r, m, Real(1) / (tau + Real(0.5))};
  }
  LBM_HD LBM_INLINE Real viscosity_at(Real p) const {
    const Local l = local(p);
    return l.mu / l.rho;
  }
};

//------------------------------------------------------------------------------
// Phi_i(u) of Eq. (10) -- the equilibrium's velocity part, second order.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE Real pf_phi_eq(int i, Real ux, Real uy, Real uz) {
  const Real ics2 = FluidLattice::inv_cs2();
  const Real cu = Real(FluidLattice::cx(i)) * ux + Real(FluidLattice::cy(i)) * uy
                + Real(FluidLattice::cz(i)) * uz;
  const Real uu = ux*ux + uy*uy + uz*uz;
  return ics2 * cu + Real(0.5) * ics2 * ics2 * cu * cu - Real(0.5) * ics2 * uu;
}

//==============================================================================
//  Kernel parameters. One struct for all six passes.
//==============================================================================
struct PfParams {
  Real* f = nullptr;                 // fluid populations, D3Q27
  Real* h = nullptr;                 // phase populations, D3Q7
  Real* phi = nullptr;
  Real* gx = nullptr;  Real* gy = nullptr;  Real* gz = nullptr;
  Real* lap = nullptr;
  Real* ux = nullptr;  Real* uy = nullptr;  Real* uz = nullptr;
  Real* pt = nullptr;                // p~, the normalised pressure
  Real* vx = nullptr;  Real* vy = nullptr;  Real* vz = nullptr;   // F_nu, optional
  Real* px = nullptr;  Real* py = nullptr;  Real* pz = nullptr;   // grad p~, optional
  const std::uint8_t* pflags = nullptr;    // phase cell roles
  const std::uint8_t* fflags = nullptr;    // fluid cell roles
  int nx = 0, ny = 0, nz = 0;
  PhaseModel pm;
  MultiphaseModel fm;
};

//------------------------------------------------------------------------------
// F = F_s + F_p + F_nu + F_b.
//
// F_s AND F_p SHARE A DIRECTION in the local-rho form -- one through mu_phi, the
// other through grad rho = (drho/dphi) grad phi -- so they are assembled as a
// single coefficient on one vector rather than two vectors added. In the
// constant-rho_0 form F_p follows grad p~ instead and is carried separately.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE
void pf_force(const PfParams& p, long n, const MultiphaseModel::Local& l,
              Real p_tilde, Real F[3]) {
  const Real cs2v = FluidLattice::cs2();
  const MultiphaseModel& m = p.fm;
  const Real phi0 = Real(0.5) * (m.phi_L + m.phi_H);
  const Real mu_phi = Real(4) * m.beta * (l.p - m.phi_L) * (l.p - m.phi_H)
                                       * (l.p - phi0)
                    - m.kappa * p.lap[n];                        // Eq. (19)
  const bool cref = m.constant_reference();
  const Real coef = cref ? mu_phi : (mu_phi - p_tilde * cs2v * m.drho_dphi());
  const Real dr   = cref ? cs2v * (l.rho - m.rho_0) : Real(0);
  const bool have_p = cref && p.px != nullptr;
  const bool have_v = p.vx != nullptr;
  F[0] = coef * p.gx[n] + (have_p ? dr * p.px[n] : Real(0))
       + (have_v ? p.vx[n] : Real(0)) + l.rho * m.bx;
  F[1] = coef * p.gy[n] + (have_p ? dr * p.py[n] : Real(0))
       + (have_v ? p.vy[n] : Real(0)) + l.rho * m.by;
  F[2] = coef * p.gz[n] + (have_p ? dr * p.pz[n] : Real(0))
       + (have_v ? p.vz[n] : Real(0)) + l.rho * m.bz;
}

//------------------------------------------------------------------------------
// PASS 1. phi from h. Read-only, so it may run at the parity the next step will
// collide at.
//
// A zero-flux wall never collides, so its slots hold whatever a neighbour
// emitted toward it one step ago. That sum is not an order parameter, and the
// gradient stencil reading it is exactly the gap the banner lists under
// WETTING -- so it is left at its initialised value rather than given a
// meaningless one.
//------------------------------------------------------------------------------
template <int Parity, bool HasGeometry>
LBM_HD LBM_INLINE void pf_field_node(const PfParams& p, long N, long n) {
  const std::uint8_t fl = HasGeometry ? p.pflags[n] : std::uint8_t(PhaseBulk);
  if (fl == PhaseExcluded) { p.phi[n] = Real(0); return; }
  if (fl == PhaseWall) return;
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real h[PhaseLattice::Q];
  gather<Parity, PhaseLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);
  Real s = Real(0);
  for (int i = 0; i < PhaseLattice::Q; ++i) s += h[i];
  p.phi[n] = s;
}

//------------------------------------------------------------------------------
// PASS 2. grad phi and lap phi, Eqs. (20)-(21), on D3Q27 from ONE gather.
//
// The i = 0 term contributes nothing to either -- c_0 is zero and the
// Laplacian's summand is phi(x) - phi(x) -- so the loop starts at 1 for both.
//------------------------------------------------------------------------------
template <bool HasGeometry>
LBM_HD LBM_INLINE void pf_derivatives_node(const PfParams& p, long n) {
  if (HasGeometry && p.pflags[n] == PhaseExcluded) {
    p.gx[n] = Real(0); p.gy[n] = Real(0); p.gz[n] = Real(0); p.lap[n] = Real(0);
    return;
  }
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  const Real p0 = p.phi[n];
  Real g[3] = {Real(0), Real(0), Real(0)}, l = Real(0);
  for (int i = 1; i < FluidLattice::Q; ++i) {
    const Real w = FluidLattice::w(i);
    const long j = neighbour<FluidLattice>(x, y, z, i, p.nx, p.ny, p.nz);
    const Real pn = p.phi[j];
    const Real wp = w * pn;
    g[0] += wp * Real(FluidLattice::cx(i));
    g[1] += wp * Real(FluidLattice::cy(i));
    g[2] += wp * Real(FluidLattice::cz(i));
    l += w * (pn - p0);
  }
  const Real ics = FluidLattice::inv_cs2();
  p.gx[n] = ics * g[0];  p.gy[n] = ics * g[1];  p.gz[n] = ics * g[2];
  p.lap[n] = Real(2) * ics * l;
}

//------------------------------------------------------------------------------
// PASS 3. F_nu = nu (grad u + grad u^T) . grad rho, Eq. (22).
//
// The second mismatch between what the LBE recovers and the equation wanted:
// mu lap(u) against the full divergence of the stress at variable mu. It
// vanishes identically at a matched density, because grad rho does, and it is
// not a correction one may drop at a ratio.
//------------------------------------------------------------------------------
template <bool HasGeometry>
LBM_HD LBM_INLINE void pf_viscous_node(const PfParams& p, long n) {
  if (HasGeometry && p.pflags[n] == PhaseExcluded) {
    p.vx[n] = Real(0); p.vy[n] = Real(0); p.vz[n] = Real(0); return;
  }
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  // du[a][b] = d u_b / d x_a, by the same isotropic stencil as grad phi.
  Real du[3][3] = {{Real(0),Real(0),Real(0)},
                   {Real(0),Real(0),Real(0)},
                   {Real(0),Real(0),Real(0)}};
  for (int i = 1; i < FluidLattice::Q; ++i) {
    const long j = neighbour<FluidLattice>(x, y, z, i, p.nx, p.ny, p.nz);
    const Real w = FluidLattice::w(i);
    const Real U[3] = {p.ux[j], p.uy[j], p.uz[j]};
    const Real wc[3] = {w * Real(FluidLattice::cx(i)),
                        w * Real(FluidLattice::cy(i)),
                        w * Real(FluidLattice::cz(i))};
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) du[a][b] += wc[a] * U[b];
  }
  // One inv_cs2 for the whole contraction: du above is the RAW sum
  // sum_i w_i c_ia u_b, and both it and its transpose need the same 1/cs2, so
  // the normalisation factors out of the bracket.
  const Real k  = p.fm.drho_dphi() * FluidLattice::inv_cs2();
  const Real gr[3] = {p.gx[n], p.gy[n], p.gz[n]};
  const Real nu = p.fm.viscosity_at(p.phi[n]);
  Real out[3];
  for (int a = 0; a < 3; ++a) {
    Real acc = Real(0);
    for (int b = 0; b < 3; ++b) acc += (du[a][b] + du[b][a]) * gr[b];
    out[a] = nu * k * acc;
  }
  p.vx[n] = out[0];  p.vy[n] = out[1];  p.vz[n] = out[2];
}

//------------------------------------------------------------------------------
// PASS 4. grad p~, for the constant-reference normalisation only.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void pf_pgrad_node(const PfParams& p, long n) {
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real g[3] = {Real(0), Real(0), Real(0)};
  for (int i = 1; i < FluidLattice::Q; ++i) {
    const long j = neighbour<FluidLattice>(x, y, z, i, p.nx, p.ny, p.nz);
    const Real wf = FluidLattice::w(i) * p.pt[j];
    g[0] += wf * Real(FluidLattice::cx(i));
    g[1] += wf * Real(FluidLattice::cy(i));
    g[2] += wf * Real(FluidLattice::cz(i));
  }
  const Real ics = FluidLattice::inv_cs2();
  p.px[n] = ics * g[0];  p.py[n] = ics * g[1];  p.pz[n] = ics * g[2];
}

//------------------------------------------------------------------------------
// PASS 5. The fluid: stream, collide against Eq. (8) with the source of
// Eq. (14), stream. Writes u(t) and p~(t) for the phase step and the diagnostics.
//------------------------------------------------------------------------------
template <int Parity, bool HasGeometry>
LBM_HD LBM_INLINE void pf_fluid_node(const PfParams& p, long N, long n) {
  const std::uint8_t fl = HasGeometry ? p.fflags[n] : std::uint8_t(Fluid);
  if (fl != Fluid) return;
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real f[27];
  gather<Parity, FluidLattice>(p.f, N, x, y, z, p.nx, p.ny, p.nz, f);

  // Eq. (24). The first moment is u, NOT rho u -- no division by rho here.
  Real s = Real(0), mx = Real(0), my = Real(0), mz = Real(0);
  for (int i = 0; i < 27; ++i) {
    s  += f[i];
    mx += f[i] * Real(FluidLattice::cx(i));
    my += f[i] * Real(FluidLattice::cy(i));
    mz += f[i] * Real(FluidLattice::cz(i));
  }
  const MultiphaseModel::Local l = p.fm.local(p.phi[n]);
  Real F[3];
  pf_force(p, n, l, s, F);
  const Real hh = Real(0.5) / l.rho;
  const Real ux = mx + hh * F[0];
  const Real uy = my + hh * F[1];
  const Real uz = mz + hh * F[2];

  const Real w    = l.omega;
  const Real pref = (Real(1) - Real(0.5) * w) * FluidLattice::inv_cs2() / l.rho;
  for (int i = 0; i < 27; ++i) {
    const Real feq = FluidLattice::w(i) * (s + pf_phi_eq(i, ux, uy, uz));
    const Real cF = Real(FluidLattice::cx(i)) * F[0]
                  + Real(FluidLattice::cy(i)) * F[1]
                  + Real(FluidLattice::cz(i)) * F[2];
    f[i] += w * (feq - f[i]) + pref * FluidLattice::w(i) * cF;
  }
  scatter<Parity, FluidLattice>(p.f, N, x, y, z, p.nx, p.ny, p.nz, f);
  p.ux[n] = ux;  p.uy[n] = uy;  p.uz[n] = uz;  p.pt[n] = s;
}

//------------------------------------------------------------------------------
// PASS 6. The phase field: stream, relax, add the anti-diffusion source, stream.
//
// Bounce-back is the identity on Esoteric Pull's storage, so a zero-flux wall
// needs no work at all.
//------------------------------------------------------------------------------
template <int Parity, bool HasGeometry>
LBM_HD LBM_INLINE void pf_phase_node(const PfParams& p, long N, long n) {
  const std::uint8_t fl = HasGeometry ? p.pflags[n] : std::uint8_t(PhaseBulk);
  if (fl != PhaseBulk) return;
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  constexpr int Q = PhaseLattice::Q;
  Real h[Q];
  gather<Parity, PhaseLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);

  Real ph = Real(0);
  for (int i = 0; i < Q; ++i) ph += h[i];
  const Real G[3] = {p.gx[n], p.gy[n], p.gz[n]};
  Real A[3];
  p.pm.anti_diffusion(ph, G, A);
  const Real ux = p.ux[n], uy = p.uy[n], uz = p.uz[n];

  const Real om   = p.pm.omega;
  const Real pref = Real(1) - Real(0.5) * om;
  for (int i = 0; i < Q; ++i) {
    const Real cA = Real(PhaseLattice::cx(i)) * A[0]
                  + Real(PhaseLattice::cy(i)) * A[1]
                  + Real(PhaseLattice::cz(i)) * A[2];
    h[i] += om * (PhaseModel::eq(i, ph, ux, uy, uz) - h[i])
          + pref * PhaseLattice::w(i) * cA;
  }
  scatter<Parity, PhaseLattice>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);
  p.phi[n] = ph;
}

#if defined(__CUDACC__)

template <int Parity, bool HasGeometry>
__global__ void pf_field_kernel(PfParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  pf_field_node<Parity, HasGeometry>(p, N, n);
}
template <bool HasGeometry>
__global__ void pf_derivatives_kernel(PfParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  pf_derivatives_node<HasGeometry>(p, n);
}
template <bool HasGeometry>
__global__ void pf_viscous_kernel(PfParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  pf_viscous_node<HasGeometry>(p, n);
}
__global__ void pf_pgrad_kernel(PfParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  pf_pgrad_node(p, n);
}
template <int Parity, bool HasGeometry>
__global__ void pf_fluid_kernel(PfParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  pf_fluid_node<Parity, HasGeometry>(p, N, n);
}
template <int Parity, bool HasGeometry>
__global__ void pf_phase_kernel(PfParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  pf_phase_node<Parity, HasGeometry>(p, N, n);
}

//------------------------------------------------------------------------------
// Seed. init(x,y,z, phi&, p_tilde&) supplies both fields; the flow starts at
// rest and the phase field at its own equilibrium, which for u = 0 is w_i phi.
//
// THE PRESSURE GAUGE IS THE CALLER'S AND IT MATTERS. p~ = 0 is the right seed
// for a problem whose pressure is near zero at the interface; seeding
// rho_L cs2 instead is a gauge shift that changes no physics in the continuum
// and is NOT a pure gauge on the lattice, because f^eq = w_i[p~ + Phi] carries
// (p~ - 1) into its higher moments.
//------------------------------------------------------------------------------
template <class Init>
__global__ void pf_initialise(Real* __restrict__ f, Real* __restrict__ h,
                              Real* __restrict__ phi, Real* __restrict__ pt,
                              int nx, int ny, int nz, Init init) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int x, y, z;
  coords(n, nx, ny, x, y, z);
  Real ph = Real(0), p_tilde = Real(0);
  init(x, y, z, ph, p_tilde);
  Real g[27];
  for (int i = 0; i < 27; ++i) g[i] = FluidLattice::w(i) * p_tilde;
  init_scatter<0, FluidLattice>(f, N, x, y, z, nx, ny, nz, g);
  Real e[PhaseLattice::Q];
  for (int i = 0; i < PhaseLattice::Q; ++i) e[i] = PhaseLattice::w(i) * ph;
  init_scatter<0, PhaseLattice>(h, N, x, y, z, nx, ny, nz, e);
  phi[n] = ph;  pt[n] = p_tilde;
}

//==============================================================================
//  Host-side driver. Owns both distributions and the coupling ORDER.
//==============================================================================
class PhaseFieldSolver {
 public:
  PhaseFieldSolver(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    N_ = long(nx) * ny * nz;
    const std::size_t fld = sizeof(Real) * std::size_t(N_);
    LBM_CUDA_CHECK(cudaMalloc(&f_, sizeof(Real) * 27 * std::size_t(N_)));
    LBM_CUDA_CHECK(cudaMalloc(&h_, sizeof(Real) * PhaseLattice::Q * std::size_t(N_)));
    for (int k = 0; k < NFIELD; ++k) LBM_CUDA_CHECK(cudaMalloc(&field_[k], fld));
    LBM_CUDA_CHECK(cudaMalloc(&pflags_, sizeof(std::uint8_t) * std::size_t(N_)));
    LBM_CUDA_CHECK(cudaMalloc(&fflags_, sizeof(std::uint8_t) * std::size_t(N_)));
    LBM_CUDA_CHECK(cudaMemset(pflags_, PhaseBulk, sizeof(std::uint8_t) * std::size_t(N_)));
    LBM_CUDA_CHECK(cudaMemset(fflags_, Fluid,     sizeof(std::uint8_t) * std::size_t(N_)));
  }
  ~PhaseFieldSolver() {
    cudaFree(f_); cudaFree(h_);
    for (int k = 0; k < NFIELD; ++k) cudaFree(field_[k]);
    cudaFree(pflags_); cudaFree(fflags_);
  }
  PhaseFieldSolver(const PhaseFieldSolver&) = delete;
  PhaseFieldSolver& operator=(const PhaseFieldSolver&) = delete;

  PhaseModel      phase;
  MultiphaseModel fluid;

  // F_nu is off by default: it is identically zero at a matched density, and
  // costs a 27-neighbour gather when it is on.
  void enable_viscous_force(bool on) { viscous_ = on; }

  void set_geometry(const std::vector<std::uint8_t>& pflags,
                    const std::vector<std::uint8_t>& fflags) {
    LBM_CUDA_CHECK(cudaMemcpy(pflags_, pflags.data(),
                              sizeof(std::uint8_t) * std::size_t(N_),
                              cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(fflags_, fflags.data(),
                              sizeof(std::uint8_t) * std::size_t(N_),
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;
  }

  // init(x, y, z, phi&, p_tilde&)
  template <class Init>
  void initialise_with(Init init) {
    const int B = 128;
    pf_initialise<<<int((N_ + B - 1) / B), B>>>(f_, h_, field_[0], field_[8],
                                                nx_, ny_, nz_, init);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    t_ = 0;
    // Zero the velocity so pass 3 has something defined to read on step 0.
    for (int k = 5; k <= 7; ++k)
      LBM_CUDA_CHECK(cudaMemset(field_[k], 0, sizeof(Real) * std::size_t(N_)));
    derivatives();
  }

  //--------------------------------------------------------------------------
  // ONE STEP, IN THE ORDER THE COUPLING REQUIRES -- see the banner. The order
  // lives here rather than in a driver because getting it wrong is a
  // systematically misplaced interface, not a damping offset.
  //--------------------------------------------------------------------------
  void step() {
    if (t_ % 2 == 0) run<0>(); else run<1>();
    ++t_;
  }

  void field_to_host(const Real* src, std::vector<Real>& out) {
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    out.resize(std::size_t(N_));
    LBM_CUDA_CHECK(cudaMemcpy(out.data(), src, sizeof(Real) * std::size_t(N_),
                              cudaMemcpyDeviceToHost));
  }

  const Real* phi_device() const { return field_[0]; }
  const Real* lap_device() const { return field_[4]; }
  const Real* ux_device()  const { return field_[5]; }
  const Real* uy_device()  const { return field_[6]; }
  const Real* uz_device()  const { return field_[7]; }
  const Real* pt_device()  const { return field_[8]; }
  std::size_t timestep() const { return t_; }
  long nodes() const { return N_; }

 private:
  static constexpr int NFIELD = 15;
  // 0 phi | 1..3 grad phi | 4 lap | 5..7 u | 8 p~ | 9..11 F_nu | 12..14 grad p~

  void derivatives() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (has_geometry_) pf_derivatives_kernel<true><<<G, B>>>(params(), N_);
    else               pf_derivatives_kernel<false><<<G, B>>>(params(), N_);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  template <int P> void run() {
    const int B = 128, G = int((N_ + B - 1) / B);
    const PfParams p = params();
    if (has_geometry_) pf_field_kernel<P, true><<<G, B>>>(p, N_);
    else               pf_field_kernel<P, false><<<G, B>>>(p, N_);
    derivatives();
    if (viscous_) {
      if (has_geometry_) pf_viscous_kernel<true><<<G, B>>>(p, N_);
      else               pf_viscous_kernel<false><<<G, B>>>(p, N_);
    }
    if (fluid.constant_reference()) pf_pgrad_kernel<<<G, B>>>(p, N_);
    if (has_geometry_) pf_fluid_kernel<P, true><<<G, B>>>(p, N_);
    else               pf_fluid_kernel<P, false><<<G, B>>>(p, N_);
    if (has_geometry_) pf_phase_kernel<P, true><<<G, B>>>(p, N_);
    else               pf_phase_kernel<P, false><<<G, B>>>(p, N_);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  PfParams params() const {
    PfParams p;
    p.f = f_;  p.h = h_;
    p.phi = field_[0];
    p.gx = field_[1];  p.gy = field_[2];  p.gz = field_[3];
    p.lap = field_[4];
    p.ux = field_[5];  p.uy = field_[6];  p.uz = field_[7];
    p.pt = field_[8];
    if (viscous_) { p.vx = field_[9]; p.vy = field_[10]; p.vz = field_[11]; }
    if (fluid.constant_reference()) {
      p.px = field_[12]; p.py = field_[13]; p.pz = field_[14];
    }
    p.pflags = pflags_;  p.fflags = fflags_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.pm = phase;  p.fm = fluid;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real* f_ = nullptr;
  Real* h_ = nullptr;
  Real* field_[NFIELD] = {};
  std::uint8_t* pflags_ = nullptr;
  std::uint8_t* fflags_ = nullptr;
  bool has_geometry_ = false;
  bool viscous_ = false;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
