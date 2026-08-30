#pragma once
//==============================================================================
//  A rigid body in the flow, by volume penalisation (direct forcing).
//
//  WHY NOT AN IMMERSED BOUNDARY, and why not moving bounce-back.
//
//  De Rosis & Enan drive their wedge with a Peskin-style immersed boundary:
//  Lagrangian markers on the surface, interpolation and spreading kernels, a
//  body force fed into F_b. That is the standard tool and it is more accurate at
//  the surface than this. It is also a great deal more machinery, and for a body
//  whose INTERIOR must move rigidly -- a solid square, not a thin shell -- a
//  volumetric penalisation enforces the same condition with a fraction of it.
//
//  Moving BOUNCE-BACK geometry would be the other route and is worse here. It
//  needs the solid mask rebuilt on the host every time the body advances, which
//  is a full pass over the domain plus rebuild_lists(); it needs a refill scheme
//  for nodes the body uncovers, whose populations are meaningless; and it
//  quantises the body's position to whole cells, so a smoothly falling square
//  arrives in lattice-sized jumps and each one radiates a pressure pulse.
//  Penalisation has none of those problems: nothing about the geometry is ever
//  rebuilt, the body's position and ANGLE are continuous, and there are no fresh
//  nodes. Rotation in particular is nearly free here and would be a rewrite in
//  either of the other two.
//
//  THE METHOD. A smooth solid indicator chi in [0, 1] and a force that drives
//  the fluid there toward the body's rigid velocity field,
//
//      F(x) = chi(x) * 2 rho(x) [ U + omega x r - u*(x) ],   r = x - x_c,
//
//  which is direct forcing: applied through the same half-force machinery the
//  rest of the code uses, one step of it takes u to the rigid field exactly
//  where chi = 1. It must run AFTER compute_macroscopic() and BEFORE the fluid
//  steps.
//
//  TWO DENSITIES, AND THEY ARE NOT THE SAME ONE. refresh() takes an optional
//  second functor, and a free surface is why.
//
//    THE LIQUID DENSITY scales the force and everything Newton needs: the
//      fictitious mass, its moments, the momentum deficit. A cell a tenth full
//      of liquid holds a tenth of the fluid and must be pushed a tenth as hard,
//      or a body would feel a full cell's resistance from a nearly empty one --
//      and would displace a full cell's worth of buoyancy from it.
//    THE LBM DENSITY is what the fluid solver divides the force BY. The stored
//      velocity is u = sum_i c_i f_i / rho + F/(2 rho) with rho the zeroth
//      moment, whatever the cell's fill level, so undoing this body's previous
//      contribution to u must divide by that same rho and nothing else.
//
//  With a phase field the two are identical and the second functor is omitted.
//  With the free surface of FreeSurfaceSolver.hpp they differ across the whole
//  interface shell, and conflating them is a 1/epsilon amplification on exactly
//  the cells a body enters the water through. Measured: a square in free fall
//  reached the surface at the right speed and then diverged within a step of
//  touching it, taking the rotation solve with it -- 285000 degrees of tilt.
//
//  A CELL WITH NO FLUID IN IT IS NOT FORCED, and the reciprocal is guarded for
//  it. With a phase field the local density never approaches zero --
//  the light phase is still a fluid -- and the guard never fires. With the free
//  surface of FreeSurfaceSolver.hpp it fires constantly: above the waterline
//  there are no populations at all, so the force written there is never
//  integrated by anything, and subtracting it from u* on the next step is
//  subtracting something that did not happen. Dividing by a near-zero density
//  turns that into a large number: measured with a floor of 1e-3 under the
//  density, the correction was amplified five hundredfold and settled into a
//  fixed point where the reaction exactly cancelled gravity -- a body in free
//  fall that hovered, with nothing in the output to say why. With the guard, an
//  empty cell contributes no force, no fictitious mass and no momentum deficit,
//  which is what "there is no fluid here" should mean.
//
//  u* IS NOT THE STORED VELOCITY, and this is the one thing that has to be right.
//  The solver defines u = sum_i c_i f_i + F/(2 rho) with F the TOTAL force, so
//  the velocity field already contains this body's force from the previous step.
//  Feeding that back in applies the correction twice: the body over-shoots, the
//  fluid over-shoots back, and the pair diverges within a few hundred steps --
//  measured, the square left the domain at six million times the impact speed.
//
//  So the previous contribution is removed first,
//
//      u* = u_stored - F_prev / (2 rho),
//
//  which leaves the velocity the fluid would have had from every OTHER force,
//  and that is what direct forcing is defined against. The arrays hold F_prev
//  already, so this costs one subtraction and no extra state.
//
//  chi IS SMOOTHED over about a cell and a half rather than being a sharp mask.
//  A sharp one is a staircase again -- the body's surface would sit on lattice
//  planes and its effective shape would change as it fell, or as it TURNED. The
//  smoothing is what lets the square occupy a continuously varying pose.
//
//  THE REACTION IS FREE, and that is the point of doing it this way: the force
//  the body exerts on the fluid is known at every node, so the force and torque
//  the fluid exerts on the body are minus its sum and minus the sum of r x F.
//  That closes Newton's equations and lets the body FALL, FLOAT and ROLL under
//  gravity and buoyancy rather than being pushed at a prescribed speed.
//==============================================================================
//
//  NEWTON, AND WHY THE OBVIOUS ARRANGEMENT OF IT CANNOT FLOAT.
//
//  The penalised region is not empty: it is full of fluid that the forcing drags
//  along, and that fluid's inertia and weight are both already inside the
//  reaction. Using the body's own mass alone would count them twice. Writing m_f
//  for that fictitious fluid mass, the exact bookkeeping is
//
//      m_b dU/dt = R + dP_in/dt + (m_b - m_f) g,
//
//  and Uhlmann's classical step approximates dP_in/dt as m_f dU/dt, moves it
//  across, and divides:
//
//      dU = [ R + (m_b - m_f) g ] / (m_b - m_f).
//
//  That denominator is the whole problem. It vanishes for a neutrally buoyant
//  body and changes sign for a light one, so a floating body -- the only
//  interesting kind, since floating is what a density ratio is FOR -- is
//  precisely the case the arrangement cannot express. The previous version of
//  this file said so and refused to try.
//
//  The fix is not a different model. It is the observation that the reaction is
//  LINEAR in the body velocity that produces it,
//
//      R = -sum chi 2 rho (U + omega x r - u*),
//
//  so if the force written into the arrays targets the velocity the body is
//  ABOUT to have rather than the one it already has, R moves to the left-hand
//  side and the same equation comes out with a different denominator:
//
//      (m_b + m_f) dU = 2 dP + (m_b - m_f) g,
//
//  where dP = sum chi rho (u* - u_rigid) is the inner fluid's momentum deficit
//  relative to rigid motion -- the hydrodynamic signal, and the only place the
//  fluid enters. Algebraically this is Uhlmann's equation multiplied through; the
//  difference is entirely that the body and the fluid are now solved at the SAME
//  instant instead of the fluid lagging the body by a step. The denominator
//  m_b + m_f is positive for every mass, the homogeneous factor is
//  (m_b - m_f)/(m_b + m_f) whose modulus is below one for every mass, and the
//  neutrally buoyant case -- previously a division by zero -- is now the
//  best-conditioned point on the line. Nothing is tuned and no limit is imposed.
//
//  WITH ROTATION the same substitution gives a 3x3 system rather than a scalar,
//  and it does not decouple: the first moments S = sum chi rho r of the
//  fictitious mass about the body centre are non-zero the moment the body
//  straddles an interface, because half of it is standing in water and half in
//  air. Those moments couple sway to roll,
//
//      [ A    0   -S_y ] [dU_x]   [ 2 dP_x + (m_b - m_f) g_x ]
//      [ 0    A    S_x ] [dU_y] = [ 2 dP_y + (m_b - m_f) g_y ],
//      [-S_y  S_x   B  ] [dw  ]   [ 2 dL   - (S_x g_y - S_y g_x) ]
//
//  with A = m_b + m_f and B = I_b + I_f. The matrix is symmetric, and it is
//  positive definite for every body: Cauchy-Schwarz on the measure chi rho gives
//  |S|^2 <= m_f I_f <= A B, so the Schur complement B - |S|^2/A is positive
//  whenever the body has mass and inertia of its own. It is solved in closed
//  form, on the host, once a step -- three lines.
//
//  THE LAST TERM IN THE THIRD ROW IS THE RIGHTING MOMENT, and it is the reason a
//  raft comes back upright. The body's own weight acts at its centre and exerts
//  no torque about it; the displaced fluid's weight does not, because the
//  displaced fluid is heavier on the submerged side, and -(S x g) is exactly the
//  hydrostatic couple that metacentric theory calls rho g V GM sin(theta). It is
//  not put in by hand for that purpose -- it falls out of the same subtraction
//  that produced (m_b - m_f) g in the rows above. See validation/floating_body.
//
//  ONE REDUCTION, NOT TWO. All seven integrals come from a single sweep, the
//  3x3 is solved from them, and the force is then written in a plain
//  parallel_for -- there is no second reduction, because R and the torque follow
//  in closed form from the solve:
//
//      R = 2 dP - 2 [ m_f dU + dw (z x S) ],   T = 2 dL - 2 [ I_f dw + S x dU ].
//
//  So the reported reaction is a derived diagnostic, exact to round-off against
//  -sum F but not an independent measurement of it. The physics enters through
//  dP and dL, which are measured.
//
//  WHAT THIS DOES NOT MODEL.
//
//   * NO CONTACT LINE. The phase field is advected by the fluid velocity and
//     knows nothing about the body, so nothing sets the angle at which the free
//     surface meets the solid. Inside the body u is driven to the rigid field, so
//     the interface is carried with it rather than through it, and the
//     conservative Allen-Cahn form keeps the profile from simply diffusing in --
//     but the wetting physics is absent, and water entry is a contact-line
//     problem. Read the splash, not the meniscus. For a FLOATING body the same
//     gap means there is no surface-tension contribution to the draft: a body
//     small enough for that to matter (Bond number of order one) is out of scope.
//   * NO SUB-CELL SURFACE. Penalisation resolves the body to the smoothing
//     width, so a pressure read at the face of the square is a smoothed pressure
//     over a cell or two. Integrated force is far more trustworthy than local
//     pressure, which is the usual situation with this family of methods. The
//     same applies to the draft, which is resolved to the interface width.
//   * NO COLLISION MODEL. Two bodies, or a body meeting a wall, will simply
//     interpenetrate: nothing here computes a contact force. The single body's
//     floor stop in the water-entry demonstrator is a clamp in the driver, not
//     physics.
//   * TWO DIMENSIONS. The rigid-body solve carries one angle and one angular
//     velocity about z. In 3D it would be a 6x6 with a rotating inertia tensor
//     and a quaternion, which is a different piece of work; the fluid side would
//     not change at all.
//   * UNIFORM DENSITY ONLY. The body's centre of mass is taken to be the
//     centre of the rectangle, which is what lets its own weight drop out of
//     the roll equation. Ballast low in the hull is the standard way to make a
//     tall body float upright, and it is exactly what this cannot express: it
//     would need a second centre, a BG that no longer follows from the draft,
//     and a weight torque restored to the third row.
//   * NO SHAPE BUT A RECTANGLE. Rect::chi is the only indicator provided.
//     Anything with a signed distance function drops straight in.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// A rectangle, described by its centre, half-extents and tilt. Kept a plain
// struct so it captures into a device lambda by value.
//
// The tilt is carried as its cosine and sine as well as the angle itself: the
// indicator is evaluated at every node of the domain every step, and computing
// two transcendentals per node to rotate into the body frame -- when the pair is
// the same for all of them -- would be the single most expensive thing here.
// theta is kept alongside so a capsizing body's angle is reported unwrapped
// rather than folded back into (-pi, pi] by atan2.
//------------------------------------------------------------------------------
struct Rect {
  Real cx = 0, cy = 0;         // centre
  Real hx = 0, hy = 0;         // half width, half height, in the BODY frame
  Real smooth = Real(1.5);     // indicator smoothing width, in cells
  Real theta = 0;              // tilt, unwrapped
  Real ct = Real(1);           // cos(theta)
  Real st = Real(0);           // sin(theta)

  KOKKOS_INLINE_FUNCTION void set_angle(Real th) {
    theta = th;  ct = Kokkos::cos(th);  st = Kokkos::sin(th);
  }

  // A circumscribing radius, generous by four smoothing widths so that chi is
  // certainly negligible outside it. Used only to reject the vast majority of
  // the domain before the two tanh calls.
  KOKKOS_INLINE_FUNCTION Real reach() const {
    return Kokkos::sqrt(hx * hx + hy * hy) + Real(4) * smooth;
  }

  // chi in [0,1]: 1 well inside, 0 well outside, tanh across the faces. The
  // product of the two axis indicators, which for a rectangle is exact away
  // from the corners and rounds them slightly -- harmless, and it keeps the
  // function separable and cheap. The point is taken into the body frame first,
  // so the shape turns with theta and the rounding turns with it.
  KOKKOS_INLINE_FUNCTION Real chi(Real x, Real y) const {
    const Real dx = x - cx, dy = y - cy;
    const Real X =  ct * dx + st * dy;
    const Real Y = -st * dx + ct * dy;
    const Real ax = (hx - Kokkos::fabs(X)) / smooth;
    const Real ay = (hy - Kokkos::fabs(Y)) / smooth;
    const Real sx = Real(0.5) * (Real(1) + Kokkos::tanh(ax));
    const Real sy = Real(0.5) * (Real(1) + Kokkos::tanh(ay));
    return sx * sy;
  }
};

//------------------------------------------------------------------------------
// The seven integrals of one sweep over the penalised region. Everything the
// rigid-body solve needs, and nothing else.
//
// The constructor and operator+= are written out rather than defaulted because
// a Kokkos reducer's identity must be callable on the device, and an implicit
// aggregate constructor is not reliably host-device under nvcc.
//------------------------------------------------------------------------------
struct BodySums {
  Real m;             // integral of chi rho              -- fictitious mass
  Real Sx, Sy;        // integral of chi rho r            -- its first moments
  Real Iz;            // integral of chi rho |r|^2        -- its inertia
  Real Px, Py;        // integral of chi rho (u* - rigid) -- momentum deficit
  Real Lz;            // integral of chi rho r x (u* - rigid)

  KOKKOS_INLINE_FUNCTION BodySums()
      : m(0), Sx(0), Sy(0), Iz(0), Px(0), Py(0), Lz(0) {}

  KOKKOS_INLINE_FUNCTION void operator+=(const BodySums& o) {
    m += o.m;  Sx += o.Sx;  Sy += o.Sy;  Iz += o.Iz;
    Px += o.Px;  Py += o.Py;  Lz += o.Lz;
  }
};

}  // namespace lbm

namespace Kokkos {
template <>
struct reduction_identity<lbm::BodySums> {
  KOKKOS_FORCEINLINE_FUNCTION static lbm::BodySums sum() { return lbm::BodySums(); }
};
}  // namespace Kokkos

namespace lbm {

template <class L>
class PenalisedBody {
 public:
  // Two dimensions, and said in the compiler's voice rather than only in the
  // limitation list: instantiated on a 3D lattice this would silently model an
  // infinite prism in z, with a rigid-body solve that carries one angle when it
  // needs three. A 3D body is a 6x6 and a quaternion, and does not exist here.
  static_assert(L::D == 2, "PenalisedBody is two-dimensional");

  explicit PenalisedBody(const Domain& dom)
      : dom_(dom),
        fx_("body_fx", dom.n_padded),
        fy_("body_fy", dom.n_padded),
        fz_("body_fz", dom.n_padded) {}

  void set_velocity(View1D<Real> ux, View1D<Real> uy) { ux_ = ux; uy_ = uy; }

  View1D<Real> x() const { return fx_; }
  View1D<Real> y() const { return fy_; }
  View1D<Real> z() const { return fz_; }

  //---- state, public so a driver can prescribe, clamp or read any of it -------
  Rect shape;
  Real vx = 0, vy = 0;             // centre-of-mass velocity
  Real omega = 0;                  // angular velocity about z

  //---- properties -------------------------------------------------------------
  Real mass = 0;                   // m_b
  Real inertia = 0;                // I_b about the centre
  Real bx = 0, by = 0;             // body force per unit mass; the SAME vector
                                   // the collision operator is given, or the
                                   // buoyancy term is inconsistent with it

  // Held degrees of freedom. Locking translation leaves the roll equation
  // uncoupled (B dw = rhs) rather than solving the 3x3 and discarding two rows,
  // which would be a different and wrong answer.
  bool free_translation = true;
  bool free_rotation = true;

  //----------------------------------------------------------------------------
  // Integrals of the indicator alone: its area, and its second moment about the
  // centre. Taken from chi rather than from 4 hx hy and the textbook (w^2+h^2)/12
  // so that mass and inertia describe the body the force actually acts on -- the
  // smoothing makes the penalised body slightly larger than the nominal
  // rectangle, and using nominal values would bias both.
  //
  // Both are invariant under the body's own rotation, so this is called once at
  // setup and not per step.
  //----------------------------------------------------------------------------
  struct Moments { Real area = 0, second = 0; };

  Moments indicator_moments() const {
    const Domain d = dom_;
    const Rect b = shape;
    const Index hx = dom_.hx, hy = dom_.hy;
    Real a = 0, s = 0;
    Kokkos::parallel_reduce("penalised_moments", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc_a, Real& acc_s) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real rx = Real(px - hx) - b.cx, ry = Real(py - hy) - b.cy;
        const Real c = b.chi(Real(px - hx), Real(py - hy));
        acc_a += c;
        acc_s += c * (rx * rx + ry * ry);
      }, a, s);
    return Moments{a, s};
  }

  // Mass and inertia of a body of uniform density, from those moments.
  void set_uniform_density(Real rho_b) {
    const Moments m = indicator_moments();
    mass = rho_b * m.area;
    inertia = rho_b * m.second;
  }

  // Kept for callers that only want the area.
  Real penalised_area() const { return indicator_moments().area; }

  //----------------------------------------------------------------------------
  // One coupling step: measure, solve Newton for the new body state, write the
  // force that drives the fluid to it.
  //
  // `density_of` maps a node to its local fluid density, so the force scales
  // with the fluid actually there: the same body meets a thousandfold heavier
  // medium when it reaches the water, and a penalisation that ignored that would
  // decelerate it as though it were still in air. The same locality is what puts
  // a non-zero S in the roll equation and makes the body right itself.
  //----------------------------------------------------------------------------
  struct Reaction {
    Real fx = 0, fy = 0;    // force the fluid exerts on the body
    Real torque = 0;        // and its moment about the body centre
    Real fluid_mass = 0;    // mass of fluid standing in the penalised region
    // The hydrostatic couple -(S x g), reported separately because it is the
    // only part of the roll balance with a closed form to check: metacentric
    // theory says it is rho g V GM sin(theta), and nothing in the way it is
    // computed here knows that. See validation/floating_body.
    Real righting = 0;
  };

  // One density for both, which is what a two-fluid model wants.
  template <class DensityOf>
  Reaction refresh(DensityOf density_of) {
    return refresh(density_of, density_of);
  }

  template <class LiquidOf, class LbmOf>
  Reaction refresh(LiquidOf density_of, LbmOf lbm_of) {
    const BodySums q = probe(density_of, lbm_of);
    Real dux = 0, duy = 0, dw = 0;
    solve(q, dux, duy, dw);
    if (free_translation) { vx += dux; vy += duy; }
    if (free_rotation)    { omega += dw; }
    apply(density_of, lbm_of);

    // R and T in closed form; see the header. Exact against -sum F, and free.
    const Real Zx = -q.Sy, Zy = q.Sx;              // z x S
    return Reaction{
      Real(2) * q.Px - Real(2) * (q.m * dux + dw * Zx),
      Real(2) * q.Py - Real(2) * (q.m * duy + dw * Zy),
      Real(2) * q.Lz - Real(2) * (q.Iz * dw + (q.Sx * duy - q.Sy * dux)),
      q.m,
      -(q.Sx * by - q.Sy * bx)};
  }

  // Advance the pose. Explicit Euler at dt = 1 -- the fluid step is the
  // timescale and there is nothing faster in the body to resolve.
  void advance() {
    shape.cx += vx;
    shape.cy += vy;
    shape.set_angle(shape.theta + omega);
  }

  //----------------------------------------------------------------------------
  // The single sweep. Public because it launches a Kokkos lambda, which nvcc
  // will not accept from a private member.
  //----------------------------------------------------------------------------
  template <class DensityOf>
  BodySums probe(DensityOf density_of) const { return probe(density_of, density_of); }

  template <class LiquidOf, class LbmOf>
  BodySums probe(LiquidOf density_of, LbmOf lbm_of) const {
    const Domain d = dom_;
    const Rect b = shape;
    const Real bvx = vx, bvy = vy, bw = omega;
    auto ux = ux_, uy = uy_;
    auto fx = fx_, fy = fy_;
    const auto dens = density_of;
    const auto ldens = lbm_of;
    const Index hx = dom_.hx, hy = dom_.hy;
    const Real reach2 = shape.reach() * shape.reach();

    BodySums out;
    Kokkos::parallel_reduce("penalised_body_probe", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, BodySums& acc) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real rx = Real(px - hx) - b.cx, ry = Real(py - hy) - b.cy;
        if (rx * rx + ry * ry > reach2) return;
        const Real c = b.chi(Real(px - hx), Real(py - hy));
        if (c < Real(1e-6)) return;
        const Real r = dens(n);              // liquid: force and Newton
        const Real rl = ldens(n);            // LBM: what the fluid divides by
        const Real cr = c * r;
        // Undo this body's own previous contribution to the stored velocity --
        // against the density the fluid actually used, and only where there was
        // a fluid at all. See the note above.
        const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
        const Real usx = ux(n) - fx(n) * inv;
        const Real usy = uy(n) - fy(n) * inv;
        // ... and measure what is left against the rigid field it should match.
        const Real dx = usx - (bvx - bw * ry);
        const Real dy = usy - (bvy + bw * rx);
        acc.m  += cr;
        acc.Sx += cr * rx;   acc.Sy += cr * ry;
        acc.Iz += cr * (rx * rx + ry * ry);
        acc.Px += cr * dx;   acc.Py += cr * dy;
        acc.Lz += cr * (rx * dy - ry * dx);
      }, Kokkos::Sum<BodySums>(out));
    Kokkos::fence();
    return out;
  }

  // Write F = chi 2 rho (U + omega x r - u*) with the NEW body state, and zero
  // it everywhere else -- the body moves, and a force left behind where it used
  // to be would keep pushing.
  template <class DensityOf>
  void apply(DensityOf density_of) { apply(density_of, density_of); }

  template <class LiquidOf, class LbmOf>
  void apply(LiquidOf density_of, LbmOf lbm_of) {
    const Domain d = dom_;
    const Rect b = shape;
    const Real bvx = vx, bvy = vy, bw = omega;
    auto ux = ux_, uy = uy_;
    auto fx = fx_, fy = fy_, fz = fz_;
    const auto dens = density_of;
    const auto ldens = lbm_of;
    const Index hx = dom_.hx, hy = dom_.hy;
    const Real reach2 = shape.reach() * shape.reach();

    Kokkos::parallel_for("penalised_body_apply", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        fz(n) = Real(0);
        const Real rx = Real(px - hx) - b.cx, ry = Real(py - hy) - b.cy;
        if (rx * rx + ry * ry > reach2) { fx(n) = Real(0); fy(n) = Real(0); return; }
        const Real c = b.chi(Real(px - hx), Real(py - hy));
        if (c < Real(1e-6)) { fx(n) = Real(0); fy(n) = Real(0); return; }
        const Real r = dens(n);
        const Real rl = ldens(n);
        const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
        const Real usx = ux(n) - fx(n) * inv;
        const Real usy = uy(n) - fy(n) * inv;
        fx(n) = c * Real(2) * r * ((bvx - bw * ry) - usx);
        fy(n) = c * Real(2) * r * ((bvy + bw * rx) - usy);
      });
    Kokkos::fence();
  }

  //----------------------------------------------------------------------------
  // The 3x3, in closed form. Symmetric and positive definite for any body with
  // mass and inertia -- see the header -- so there is no pivoting and no case
  // where this has to give up.
  //----------------------------------------------------------------------------
  void solve(const BodySums& q, Real& dux, Real& duy, Real& dw) const {
    const Real A = mass + q.m;
    const Real B = inertia + q.Iz;
    if (A <= Real(0)) { dux = duy = dw = 0; return; }

    const Real rx = Real(2) * q.Px + (mass - q.m) * bx;
    const Real ry = Real(2) * q.Py + (mass - q.m) * by;
    const Real rw = Real(2) * q.Lz - (q.Sx * by - q.Sy * bx);

    // Locking translation removes the coupling rather than the rows: with dU
    // pinned to zero the roll equation is B dw = rw on its own.
    const Real invA = free_translation ? Real(1) / A : Real(0);

    dw = 0;
    if (free_rotation && B > Real(0)) {
      const Real schur = B - (q.Sx * q.Sx + q.Sy * q.Sy) * invA;
      if (schur > Real(0))
        dw = (rw - (q.Sx * ry - q.Sy * rx) * invA) / schur;
    }
    dux = (rx + q.Sy * dw) * invA;
    duy = (ry - q.Sx * dw) * invA;
  }

 private:
  Domain dom_;
  View1D<Real> fx_, fy_, fz_;
  View1D<Real> ux_, uy_;
};

}  // namespace lbm
