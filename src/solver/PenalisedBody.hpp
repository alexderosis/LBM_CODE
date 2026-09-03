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
#include "solver/RigidBody3D.hpp"
#include "grid/Domain.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// A symmetric wedge, for De Rosis & Enan's Sec. III.G water entry: an apex
// pointing DOWN and two faces rising from it at the deadrise angle phi, capped
// at the knuckle. Same idiom as Rect below -- a plain struct, tilt carried as
// its cosine and sine, a product of tanh indicators -- so everything the body
// does with a Rect it does with this unchanged.
//
// THE ORIGIN IS THE APEX, NOT THE CENTROID, and that is a deliberate difference
// from Rect. A water-entry case prescribes the depth of the apex and compares
// against a wetted length measured from it, so carrying the apex is what the
// driver actually wants; putting the centroid there would mean converting on
// every read. The consequence is that `free_rotation` about this origin is NOT
// the physical roll axis -- the case this shape exists for drives both degrees
// of freedom, and a freely rotating wedge would need the centroid offset
// (0, 2 height / 3) folded in first.
//
// GEOMETRY. In the body frame with the apex at the origin, the wedge is the set
//
//     Y > |X| tan(phi)      above the two faces
//     Y < H                 below the knuckle,  H = half_beam tan(phi)
//
// and the face condition is written as a true normal distance,
//
//     d_face = Y cos(phi) - |X| sin(phi),
//
// because cos^2 + sin^2 = 1 makes that the perpendicular distance to whichever
// face the point is on. Scaling `smooth` against a non-normalised distance
// would make the smoothing width depend on the deadrise angle, so that a
// shallow wedge got a diffuse surface and a steep one a sharp surface -- the
// kind of thing that quietly turns a deadrise sweep into a smoothing sweep.
//
// THE APEX IS ROUNDED, AND IT MATTERS MOST WHERE THE PHYSICS DOES. The product
// of two tanh indicators rounds a convex corner, exactly as Rect's does, and a
// wedge apex is a sharper corner than a rectangle's right angle. Measured as
// the integral of chi over the nominal area b^2 tan(phi):
//
//     deadrise     b = 100, smooth = 1     b = 30, smooth = 1
//     10 deg           +0.50 %                 +5.6 %
//     15 deg           +0.23 %                 +2.6 %
//     30 deg           +0.06 %                 +0.64 %
//     45 deg           +0.03 %                 +0.28 %
//
// So the penalised wedge is slightly LARGER than the nominal one, the excess
// concentrated at the apex, and it grows as the wedge sharpens or the body
// shrinks against the smoothing width. For water entry that is the worst place
// for it: the apex is where the impact starts and where Wagner's wetted length
// is measured from, so a shallow-deadrise run wants a big body in cells rather
// than a small one with the smoothing turned down -- reducing `smooth` below
// about one cell makes the indicator a step again and reintroduces the
// lattice-quantised pressure pulses penalisation exists to avoid.
//------------------------------------------------------------------------------
struct Wedge {
  Real cx = 0, cy = 0;         // THE APEX
  Real half_beam = 0;          // half-width at the knuckle
  Real smooth = Real(1.5);     // indicator smoothing width, in cells
  Real theta = 0;              // tilt, unwrapped
  Real ct = Real(1);           // cos(theta)
  Real st = Real(0);           // sin(theta)
  Real cphi = Real(1);         // cos(deadrise)
  Real sphi = Real(0);         // sin(deadrise)

  KOKKOS_INLINE_FUNCTION void set_angle(Real th) {
    theta = th;  ct = Kokkos::cos(th);  st = Kokkos::sin(th);
  }
  // Deadrise from the horizontal, in radians. Stored as its pair for the same
  // reason theta is: it is used at every node of the domain every step.
  KOKKOS_INLINE_FUNCTION void set_deadrise(Real p) {
    cphi = Kokkos::cos(p);  sphi = Kokkos::sin(p);
  }
  KOKKOS_INLINE_FUNCTION Real deadrise() const { return Kokkos::atan2(sphi, cphi); }
  KOKKOS_INLINE_FUNCTION Real height() const {
    return half_beam * sphi / cphi;                       // H = b tan(phi)
  }
  // Apex to knuckle corner: sqrt(b^2 + H^2) = b / cos(phi).
  KOKKOS_INLINE_FUNCTION Real reach() const {
    return half_beam / cphi + Real(4) * smooth;
  }
  // A PRISM: chi does not depend on z, so outside() must not reject on z
  // either. One shared formula would cull the body to a single plane and the
  // reduction would silently return 1/nz of its mass.
  static constexpr bool three_d = false;
  static constexpr bool six_dof = false;
  KOKKOS_INLINE_FUNCTION bool outside(Real rx, Real ry, Real, Real reach2) const {
    return rx * rx + ry * ry > reach2;
  }
  KOKKOS_INLINE_FUNCTION Real chi(Real x, Real y, Real) const {
    const Real dx = x - cx, dy = y - cy;
    const Real X =  ct * dx + st * dy;
    const Real Y = -st * dx + ct * dy;
    const Real df = (Y * cphi - Kokkos::fabs(X) * sphi) / smooth;
    const Real dt = (height() - Y) / smooth;
    const Real sf = Real(0.5) * (Real(1) + Kokkos::tanh(df));
    const Real st_ = Real(0.5) * (Real(1) + Kokkos::tanh(dt));
    return sf * st_;
  }
};

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
  static constexpr bool three_d = false;      // a prism; see Wedge::outside
  static constexpr bool six_dof = false;
  KOKKOS_INLINE_FUNCTION bool outside(Real rx, Real ry, Real, Real reach2) const {
    return rx * rx + ry * ry > reach2;
  }
  KOKKOS_INLINE_FUNCTION Real chi(Real x, Real y, Real) const {
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
// A SPHERE, and the only shape here that is not a prism.
//
// This is the port of lbm::Sphere in ../../GPU/include/lbm/body.cuh, and the two
// exist deliberately: this tree keeps two independent implementations of the
// physics so that agreement is evidence and disagreement is a bug in one of
// them. Until now the 3-D body existed only on the CUDA side, so it had nothing
// to be checked against.
//
// chi is one tanh in the radius -- cheaper than the prisms' two, and with no
// body frame to rotate into, because a sphere is its own rotation group.
//
// WHAT THIS DOES NOT MAKE. Adding a sphere does not turn this into a 3-D rigid
// body. Rotation is still one angle about z, and for a sphere it is inert: chi
// is invariant under it, so the roll equation measures nothing and changes
// nothing. Translation becomes three components, which for a sphere entering on
// its axis is the whole of the dynamics -- there is no torque about any axis, so
// 3-DOF translation is EXACT for that case rather than an approximation to a
// 6-DOF solve. It is not exact for anything asymmetric, and a tilted, spinning
// or tumbling body is still absent: that needs the quaternion the class banner
// describes.
//------------------------------------------------------------------------------
struct Sphere {
  Real cx = 0, cy = 0, cz = 0;      // centre
  Real R = 0;                       // radius
  Real smooth = Real(1.5);          // indicator smoothing width, in cells
  // Present so a Sphere satisfies the same interface as the prisms and drops
  // into PenalisedBody unchanged. Both are inert here; see the banner.
  Real theta = 0;
  KOKKOS_INLINE_FUNCTION void set_angle(Real th) { theta = th; }

  static constexpr bool three_d = true;
  // A sphere's chi is invariant under rotation, so its roll equation measures
  // nothing: three translations are the whole of its dynamics and the 6-DOF
  // path would be six unknowns for three answers.
  static constexpr bool six_dof = false;

  KOKKOS_INLINE_FUNCTION Real reach() const { return R + Real(4) * smooth; }

  // The only shape whose cheap rejection is genuinely spherical.
  KOKKOS_INLINE_FUNCTION bool outside(Real rx, Real ry, Real rz, Real reach2) const {
    return rx * rx + ry * ry + rz * rz > reach2;
  }

  KOKKOS_INLINE_FUNCTION Real chi(Real x, Real y, Real z) const {
    const Real dx = x - cx, dy = y - cy, dz = z - cz;
    const Real r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
    return Real(0.5) * (Real(1) + Kokkos::tanh((R - r) / smooth));
  }
};

//------------------------------------------------------------------------------
// A BOX with a full orientation -- the first shape here that needs six degrees
// of freedom rather than three translations and an inert angle.
//
// WHY IT NEEDS THEM AND THE SPHERE DID NOT. A sphere's chi is invariant under
// rotation, so its roll equation measures nothing; a cube's is not. Released
// corner-down it strikes on a vertex, and the reaction on that vertex is not
// through the centre in ANY single plane -- so the response is a rotation about
// an axis that is neither x, y nor z, and that axis MOVES as the body turns.
// One angle cannot express it and neither can three.
//
// chi is the product of three tanh axis indicators in the BODY frame, which is
// Rect's construction with a third factor: exact away from the edges, rounding
// them slightly, and separable so it stays three transcendentals rather than a
// distance-to-a-polyhedron. The rounding turns with the body because the point
// is taken into the body frame first.
//
// The orientation is a quaternion for the state and its matrix cached beside
// it, for the reason Rect caches cos and sin: chi is evaluated at every node of
// the domain every step.
//------------------------------------------------------------------------------
struct Box {
  Real cx = 0, cy = 0, cz = 0;      // centre
  Real hx = 0, hy = 0, hz = 0;      // half extents, in the BODY frame
  Real smooth = Real(1.5);
  Quat q;                           // orientation, the state
  Mat3 Rm;                          // its matrix, cached

  static constexpr bool three_d = true;
  static constexpr bool six_dof = true;

  void set_orientation(const Quat& qq) { q = qq;  q.normalise();  Rm = q.matrix(); }
  // Present so a Box satisfies the same interface as the prisms. theta is
  // meaningless for a 3-D orientation and is deliberately not tracked: a caller
  // that wants the pose should read q.
  Real theta = 0;
  KOKKOS_INLINE_FUNCTION void set_angle(Real) {}

  KOKKOS_INLINE_FUNCTION Real reach() const {
    return Kokkos::sqrt(hx * hx + hy * hy + hz * hz) + Real(4) * smooth;
  }
  KOKKOS_INLINE_FUNCTION bool outside(Real rx, Real ry, Real rz, Real reach2) const {
    return rx * rx + ry * ry + rz * rz > reach2;
  }
  KOKKOS_INLINE_FUNCTION Real chi(Real x, Real y, Real z) const {
    Real X, Y, Z;
    Rm.tmul(x - cx, y - cy, z - cz, X, Y, Z);       // world -> body is R^T
    const Real ax = (hx - Kokkos::fabs(X)) / smooth;
    const Real ay = (hy - Kokkos::fabs(Y)) / smooth;
    const Real az = (hz - Kokkos::fabs(Z)) / smooth;
    return Real(0.125) * (Real(1) + Kokkos::tanh(ax))
                       * (Real(1) + Kokkos::tanh(ay))
                       * (Real(1) + Kokkos::tanh(az));
  }
};

//------------------------------------------------------------------------------
// A DISC -- a flat circular cylinder, for a skipping stone.
//
// THE SYMMETRY AXIS IS BODY y, NOT BODY z, and that is a deliberate choice of
// reference rather than a convention inherited from anywhere. Gravity here is
// -y, so at the identity orientation this disc lies FLAT with its axis
// vertical: the pose a stone is in just before it is thrown. Every angle a
// caller then sets is a departure from that, which is what makes an attack
// angle readable in the driver instead of being a quaternion nobody can check
// by eye. With the axis on body z the identity pose would be a disc standing on
// its edge, and every case would open by rotating 90 degrees for no reason.
//
// chi is a PRODUCT of a radial and an axial profile, so the rim and the two
// faces are each smoothed over `smooth` cells and the edge where they meet is
// rounded. That is the same construction as Box's three tanhs and has the same
// consequence: the penalised disc is slightly larger than the nominal one, so
// the volume and the inertia are MEASURED (set_uniform_density6) rather than
// taken from pi R^2 (2 hy).
//
// WHAT IT IS NOT. A real skipping stone is lenticular -- thicker at the centre,
// tapering to a sharp rim -- and this is a flat cylinder with a rounded edge.
// The lift in a skip comes from the pressure on the WETTED AREA at an angle of
// attack, which a flat underside reproduces; the rim profile matters for how
// the flow leaves, and there is no contact-line model here anyway, so the
// tapered rim would be modelling something the scheme cannot carry.
//------------------------------------------------------------------------------
struct Disc {
  Real cx = 0, cy = 0, cz = 0;      // centre
  Real R = 0;                       // radius, in the plane perpendicular to y
  Real hy = 0;                      // HALF thickness along the symmetry axis
  Real smooth = Real(1);
  Quat q;                           // orientation, the state
  Mat3 Rm;                          // its matrix, cached

  static constexpr bool three_d = true;
  static constexpr bool six_dof = true;

  void set_orientation(const Quat& qq) { q = qq;  q.normalise();  Rm = q.matrix(); }
  // Present so a Disc satisfies the same interface as the prisms; a 3-D pose is
  // the quaternion, and theta is deliberately not tracked.
  Real theta = 0;
  KOKKOS_INLINE_FUNCTION void set_angle(Real) {}

  // The symmetry axis in the WORLD frame: the body y column of R. This is what
  // a driver needs to report an attack angle and to spin the body about its own
  // axis, and it is one matrix-vector product rather than a re-derivation.
  void axis(Real& ax, Real& ay, Real& az) const {
    ax = Rm(0, 1);  ay = Rm(1, 1);  az = Rm(2, 1);
  }

  KOKKOS_INLINE_FUNCTION Real reach() const {
    return Kokkos::sqrt(R * R + hy * hy) + Real(4) * smooth;
  }
  KOKKOS_INLINE_FUNCTION bool outside(Real rx, Real ry, Real rz, Real reach2) const {
    return rx * rx + ry * ry + rz * rz > reach2;
  }
  KOKKOS_INLINE_FUNCTION Real chi(Real x, Real y, Real z) const {
    Real X, Y, Z;
    Rm.tmul(x - cx, y - cy, z - cz, X, Y, Z);       // world -> body is R^T
    const Real rp = Kokkos::sqrt(X * X + Z * Z);    // radius about the y axis
    const Real ar = (R - rp) / smooth;
    const Real ay = (hy - Kokkos::fabs(Y)) / smooth;
    return Real(0.25) * (Real(1) + Kokkos::tanh(ar))
                      * (Real(1) + Kokkos::tanh(ay));
  }
};

//------------------------------------------------------------------------------
// The NINE integrals of one sweep over the penalised region. Everything the
// rigid-body solve needs, and nothing else.
//
// The constructor and operator+= are written out rather than defaulted because
// a Kokkos reducer's identity must be callable on the device, and an implicit
// aggregate constructor is not reliably host-device under nvcc.
//------------------------------------------------------------------------------
struct BodySums {
  Real m;             // integral of chi rho              -- fictitious mass
  Real Sx, Sy;        // integral of chi rho r            -- its first moments
  Real Iz;            // integral of chi rho (rx^2+ry^2)  -- inertia ABOUT z
  Real Px, Py;        // integral of chi rho (u* - rigid) -- momentum deficit
  Real Lz;            // integral of chi rho r x (u* - rigid)
  // The third translation. Sz is carried for symmetry of the bookkeeping and is
  // NOT coupled into the roll equation: rotation here is about z alone, so a
  // first moment along z has no equation to enter. It is reported rather than
  // used, and for a sphere on its axis it is zero to round-off, which makes it
  // a free check that the entry really is axisymmetric.
  Real Sz, Pz;

  KOKKOS_INLINE_FUNCTION BodySums()
      : m(0), Sx(0), Sy(0), Iz(0), Px(0), Py(0), Lz(0), Sz(0), Pz(0) {}

  KOKKOS_INLINE_FUNCTION void operator+=(const BodySums& o) {
    m += o.m;  Sx += o.Sx;  Sy += o.Sy;  Iz += o.Iz;
    Px += o.Px;  Py += o.Py;  Lz += o.Lz;
    Sz += o.Sz;  Pz += o.Pz;
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

template <class L, class Shape = Rect>
class PenalisedBody {
 public:
  // THE GUARD IS NARROWED, NOT REMOVED. It used to say two dimensions flatly.
  // What it was protecting against is a PRISM on a 3-D lattice: Rect and Wedge
  // have a chi independent of z, so instantiated on D3Q27 they silently model
  // an infinite prism with a solve that carries one angle. That is still true
  // and still refused. A genuinely 3-D shape (Sphere) is now allowed, and its
  // banner states what it does and does not model -- three translations, one
  // angle about z, no quaternion. A tilted or tumbling 3-D body remains absent.
  static_assert(L::D == 2 || Shape::three_d,
                "a 3-D lattice needs a 3-D shape: Rect and Wedge are prisms in "
                "z, so on D3Q19/D3Q27 they would model an infinite prism with a "
                "one-angle solve. Use Sphere, or a 2-D lattice.");

  explicit PenalisedBody(const Domain& dom)
      : dom_(dom),
        fx_("body_fx", dom.n_padded),
        fy_("body_fy", dom.n_padded),
        fz_("body_fz", dom.n_padded) {}

  // The two-argument form leaves the z limb OFF -- uz_ stays empty and every
  // prism case is the arithmetic it was before a sphere existed. Pass three to
  // enable it.
  void set_velocity(View1D<Real> ux, View1D<Real> uy) {
    ux_ = ux; uy_ = uy; uz_ = View1D<Real>();
  }
  void set_velocity(View1D<Real> ux, View1D<Real> uy, View1D<Real> uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  View1D<Real> x() const { return fx_; }
  View1D<Real> y() const { return fy_; }
  View1D<Real> z() const { return fz_; }

  //---- state, public so a driver can prescribe, clamp or read any of it -------
  Shape shape;
  Real vx = 0, vy = 0, vz = 0;     // centre-of-mass velocity
  // THE SIX-DOF STATE, used only by a shape whose six_dof is true. The angular
  // velocity is a VECTOR and the body's inertia is a tensor held in the BODY
  // frame -- world-frame inertia changes as the body turns, so storing that
  // would mean recomputing it from something, and the body frame is the
  // something. See RigidBody3D.hpp.
  Real wx = 0, wy = 0, wz = 0;     // angular velocity, world frame
  Mat3 inertia_body;               // I_b in the body frame; rotated per step
  Real omega = 0;                  // angular velocity about z

  //---- properties -------------------------------------------------------------
  Real mass = 0;                   // m_b
  Real inertia = 0;                // I_b about the centre
  Real bx = 0, by = 0, bz = 0;     // body force per unit mass; the SAME vector
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
    const Shape b = shape;
    const Index hx = dom_.hx, hy = dom_.hy, hz = dom_.hz;
    Real a = 0, s = 0;
    Kokkos::parallel_reduce("penalised_moments", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc_a, Real& acc_s) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real rx = Real(px - hx) - b.cx, ry = Real(py - hy) - b.cy;
        const Real c = b.chi(Real(px - hx), Real(py - hy), Real(pz - hz));
        acc_a += c;
        // The second moment is about z, matching BodySums::Iz -- for a sphere
        // that is its polar moment rather than its full inertia, which is
        // correct for the only rotation this class has and inert for a sphere.
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

  //--------------------------------------------------------------------------
  //  THE SIX-DOF PATH. Only instantiated for a shape with six_dof, because
  //  members of a class template are instantiated on use -- so these bodies
  //  may refer to Box's cz and Rm freely without a Rect user paying for it or
  //  failing to compile.
  //--------------------------------------------------------------------------

  // Volume and the six products integral chi ri rj at the CURRENT pose, in the
  // world frame, with rho = 1. Reduced into a BodySums6 so the existing reducer
  // is reused; only m and the J entries are filled.
  BodySums6 moments6() const {
    const Domain d = dom_;
    const Shape b = shape;
    const Index hx = dom_.hx, hy = dom_.hy, hz = dom_.hz;
    const Real reach2 = shape.reach() * shape.reach();
    BodySums6 out;
    Kokkos::parallel_reduce("penalised_moments6", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, BodySums6& acc) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real X = Real(px - hx), Y = Real(py - hy), Z = Real(pz - hz);
        const Real rx = X - b.cx, ry = Y - b.cy, rz = Z - b.cz;
        if (b.outside(rx, ry, rz, reach2)) return;
        const Real c = b.chi(X, Y, Z);
        if (c < Real(1e-6)) return;
        acc.m += c;
        acc.Jxx += c * rx * rx;  acc.Jyy += c * ry * ry;  acc.Jzz += c * rz * rz;
        acc.Jxy += c * rx * ry;  acc.Jxz += c * rx * rz;  acc.Jyz += c * ry * rz;
      }, Kokkos::Sum<BodySums6>(out));
    Kokkos::fence();
    return out;
  }

  // Mass and the BODY-frame inertia tensor of a body of uniform density, from
  // those moments. Measured from chi rather than from the nominal box, for the
  // reason the 2-D version gives: the smoothing makes the penalised body
  // slightly larger than the nominal one and nominal values would bias both.
  //
  // The measurement is in the WORLD frame at the current pose, so it is rotated
  // back: I_body = R^T I_world R. Getting that inverse the wrong way round
  // leaves an inertia that is right only at the release orientation.
  void set_uniform_density6(Real rho_b) {
    BodySums6 q = moments6();
    q.m *= rho_b;
    q.Jxx *= rho_b;  q.Jyy *= rho_b;  q.Jzz *= rho_b;
    q.Jxy *= rho_b;  q.Jxz *= rho_b;  q.Jyz *= rho_b;
    mass = q.m;
    const Mat3 R = shape.Rm;
    inertia_body = rotate_tensor(R.transposed(), q.fluid_inertia());
  }

  Real penalised_volume() const { return moments6().m; }

  template <class LiquidOf, class LbmOf>
  BodySums6 probe6(LiquidOf density_of, LbmOf lbm_of) {
    const Domain d = dom_;
    const Shape b = shape;
    const Real bvx = vx, bvy = vy, bvz = vz;
    const Real bwx = wx, bwy = wy, bwz = wz;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto fx = fx_, fy = fy_, fz = fz_;
    const auto dens = density_of;
    const auto ldens = lbm_of;
    const Index hx = dom_.hx, hy = dom_.hy, hz = dom_.hz;
    const Real reach2 = shape.reach() * shape.reach();

    BodySums6 out;
    Kokkos::parallel_reduce("penalised_body_probe6", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, BodySums6& acc) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real X = Real(px - hx), Y = Real(py - hy), Z = Real(pz - hz);
        const Real rx = X - b.cx, ry = Y - b.cy, rz = Z - b.cz;
        if (b.outside(rx, ry, rz, reach2)) return;
        const Real c = b.chi(X, Y, Z);
        if (c < Real(1e-6)) return;
        const Real r = dens(n);
        const Real rl = ldens(n);
        const Real cr = c * r;
        // Undo this body's own previous contribution to the stored velocity.
        const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
        const Real usx = ux(n) - fx(n) * inv;
        const Real usy = uy(n) - fy(n) * inv;
        const Real usz = uz(n) - fz(n) * inv;
        // The rigid field is now U + omega x r with omega a VECTOR, which is
        // the whole difference from the 2-D probe.
        const Real vrx = bvx + (bwy * rz - bwz * ry);
        const Real vry = bvy + (bwz * rx - bwx * rz);
        const Real vrz = bvz + (bwx * ry - bwy * rx);
        const Real dx = usx - vrx, dy = usy - vry, dz = usz - vrz;
        acc.m += cr;
        acc.Sx += cr * rx;  acc.Sy += cr * ry;  acc.Sz += cr * rz;
        acc.Jxx += cr * rx * rx;  acc.Jyy += cr * ry * ry;  acc.Jzz += cr * rz * rz;
        acc.Jxy += cr * rx * ry;  acc.Jxz += cr * rx * rz;  acc.Jyz += cr * ry * rz;
        acc.Px += cr * dx;  acc.Py += cr * dy;  acc.Pz += cr * dz;
        acc.Lx += cr * (ry * dz - rz * dy);
        acc.Ly += cr * (rz * dx - rx * dz);
        acc.Lz += cr * (rx * dy - ry * dx);
      }, Kokkos::Sum<BodySums6>(out));
    Kokkos::fence();
    return out;
  }

  template <class LiquidOf, class LbmOf>
  void apply6(LiquidOf density_of, LbmOf lbm_of) {
    const Domain d = dom_;
    const Shape b = shape;
    const Real bvx = vx, bvy = vy, bvz = vz;
    const Real bwx = wx, bwy = wy, bwz = wz;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto fx = fx_, fy = fy_, fz = fz_;
    const auto dens = density_of;
    const auto ldens = lbm_of;
    const Index hx = dom_.hx, hy = dom_.hy, hz = dom_.hz;
    const Real reach2 = shape.reach() * shape.reach();

    Kokkos::parallel_for("penalised_body_apply6", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real X = Real(px - hx), Y = Real(py - hy), Z = Real(pz - hz);
        const Real rx = X - b.cx, ry = Y - b.cy, rz = Z - b.cz;
        if (b.outside(rx, ry, rz, reach2)) {
          fx(n) = Real(0); fy(n) = Real(0); fz(n) = Real(0); return;
        }
        const Real c = b.chi(X, Y, Z);
        if (c < Real(1e-6)) {
          fx(n) = Real(0); fy(n) = Real(0); fz(n) = Real(0); return;
        }
        const Real r = dens(n);
        const Real rl = ldens(n);
        const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
        const Real usx = ux(n) - fx(n) * inv;
        const Real usy = uy(n) - fy(n) * inv;
        const Real usz = uz(n) - fz(n) * inv;
        const Real vrx = bvx + (bwy * rz - bwz * ry);
        const Real vry = bvy + (bwz * rx - bwx * rz);
        const Real vrz = bvz + (bwx * ry - bwy * rx);
        fx(n) = c * Real(2) * r * (vrx - usx);
        fy(n) = c * Real(2) * r * (vry - usy);
        fz(n) = c * Real(2) * r * (vrz - usz);
      });
    Kokkos::fence();
  }

  // The 2-D Reaction, with the force and the moment completed to vectors. Same
  // closed forms, generalised: dw * (zhat x S) becomes dW x S and Iz * dw
  // becomes I_f dW.
  struct Reaction6 {
    Real fx = 0, fy = 0, fz = 0;      // force the fluid exerts on the body
    Real tx = 0, ty = 0, tz = 0;      // and its moment about the body centre
    Real fluid_mass = 0;              // fluid standing in the penalised region
    Real rx = 0, ry = 0, rz = 0;      // the hydrostatic couple -(S x g)
  };

  // One 6-DOF coupling step. Same contract as refresh(): AFTER the macroscopic
  // pass and BEFORE the fluid steps.
  template <class DensityOf>
  Reaction6 refresh6(DensityOf dens) { return refresh6(dens, dens); }

  template <class LiquidOf, class LbmOf>
  Reaction6 refresh6(LiquidOf dens, LbmOf ldens) {
    const BodySums6 q = probe6(dens, ldens);
    Body6Properties p;
    p.mass = mass;
    // The body's tensor rotated into the world frame at the CURRENT pose. This
    // is the line that makes it a 3-D body rather than three translations: a
    // tumbling body's resistance to a torque depends on which way it is facing.
    p.inertia_world = rotate_tensor(shape.Rm, inertia_body);
    // The current angular velocity, for the gyroscopic term: without it a
    // spinning body's axis tips under torque instead of precessing.
    p.wx = wx;  p.wy = wy;  p.wz = wz;
    p.bx = bx;  p.by = by;  p.bz = bz;
    p.free_translation = free_translation;
    p.free_rotation = free_rotation;
    Real dU[3], dW[3];
    body6_solve(p, q, dU, dW);
    if (free_translation) { vx += dU[0];  vy += dU[1];  vz += dU[2]; }
    if (free_rotation)    { wx += dW[0];  wy += dW[1];  wz += dW[2]; }
    apply6(dens, ldens);

    // F = 2 dP - 2 (m_f dU + dW x S),  T = 2 dL - 2 (I_f dW + S x dU).
    const Real S[3] = {q.Sx, q.Sy, q.Sz};
    const Real WxS[3] = {dW[1] * S[2] - dW[2] * S[1],
                         dW[2] * S[0] - dW[0] * S[2],
                         dW[0] * S[1] - dW[1] * S[0]};
    const Real SxU[3] = {S[1] * dU[2] - S[2] * dU[1],
                         S[2] * dU[0] - S[0] * dU[2],
                         S[0] * dU[1] - S[1] * dU[0]};
    const Mat3 If = q.fluid_inertia();
    Real IfW[3];
    for (int i = 0; i < 3; ++i)
      IfW[i] = If(i, 0) * dW[0] + If(i, 1) * dW[1] + If(i, 2) * dW[2];
    const Real g[3] = {bx, by, bz};
    Reaction6 out;
    out.fx = Real(2) * q.Px - Real(2) * (q.m * dU[0] + WxS[0]);
    out.fy = Real(2) * q.Py - Real(2) * (q.m * dU[1] + WxS[1]);
    out.fz = Real(2) * q.Pz - Real(2) * (q.m * dU[2] + WxS[2]);
    out.tx = Real(2) * q.Lx - Real(2) * (IfW[0] + SxU[0]);
    out.ty = Real(2) * q.Ly - Real(2) * (IfW[1] + SxU[1]);
    out.tz = Real(2) * q.Lz - Real(2) * (IfW[2] + SxU[2]);
    out.fluid_mass = q.m;
    out.rx = -(S[1] * g[2] - S[2] * g[1]);
    out.ry = -(S[2] * g[0] - S[0] * g[2]);
    out.rz = -(S[0] * g[1] - S[1] * g[0]);
    return out;
  }

  // Advance the pose: the centre by Euler, the orientation by one quaternion
  // step. dt = 1 because the fluid step is the timescale.
  void advance6() {
    shape.cx += vx;
    shape.cy += vy;
    shape.cz += vz;
    Quat q = shape.q;
    q.integrate(wx, wy, wz, Real(1));
    shape.set_orientation(q);
  }

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
    Real dux = 0, duy = 0, dw = 0, duz = 0;
    solve(q, dux, duy, dw, duz);
    if (free_translation) {
      vx += dux; vy += duy;
      if (uz_.size() > 0) vz += duz;
    }
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
    bump_z(shape, vz);
    shape.set_angle(shape.theta + omega);
  }

  // A prism has no cz to read or move, so the z limb of the pose is touched
  // only for a shape that has one. Overloads on three_d rather than a runtime
  // flag: a Rect that grew a cz member by accident would otherwise start
  // behaving like a sphere silently.
  template <class S>
  KOKKOS_INLINE_FUNCTION static Real shape_cz(const S&) { return Real(0); }
  KOKKOS_INLINE_FUNCTION static Real shape_cz(const Sphere& sh) { return sh.cz; }
  template <class S> static void bump_z(S&, Real) {}
  static void bump_z(Sphere& sh, Real dz) { sh.cz += dz; }

  //----------------------------------------------------------------------------
  // The single sweep. Public because it launches a Kokkos lambda, which nvcc
  // will not accept from a private member.
  //----------------------------------------------------------------------------
  template <class DensityOf>
  BodySums probe(DensityOf density_of) const { return probe(density_of, density_of); }

  template <class LiquidOf, class LbmOf>
  BodySums probe(LiquidOf density_of, LbmOf lbm_of) const {
    const Domain d = dom_;
    const Shape b = shape;
    const Real bvx = vx, bvy = vy, bw = omega;
    auto ux = ux_, uy = uy_;
    auto fx = fx_, fy = fy_, fz = fz_;
    const auto dens = density_of;
    const auto ldens = lbm_of;
    const Index hx = dom_.hx, hy = dom_.hy, hz = dom_.hz;
    const Real reach2 = shape.reach() * shape.reach();
    const bool have_z = uz_.size() > 0;
    auto uz = uz_.size() > 0 ? uz_ : ux_;      // never dereferenced when !have_z
    const Real bvz = vz;

    BodySums out;
    Kokkos::parallel_reduce("penalised_body_probe", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, BodySums& acc) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real X = Real(px - hx), Y = Real(py - hy), Z = Real(pz - hz);
        const Real rx = X - b.cx, ry = Y - b.cy, rz = Z - shape_cz(b);
        // THE SHAPE decides the rejection, because a prism must not be culled
        // in z and a sphere must be. One formula here was the bug this
        // interface exists to prevent.
        if (b.outside(rx, ry, rz, reach2)) return;
        const Real c = b.chi(X, Y, Z);
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
        // A 2-D caller passes no z velocity at all: an EMPTY view means "no
        // third component" rather than "a component that happens to be zero",
        // so the whole z limb folds away and the prism cases stay bit for bit
        // what they were.
        const Real usz = have_z ? (uz(n) - fz(n) * inv) : Real(0);
        // ... and measure what is left against the rigid field it should match.
        const Real dx = usx - (bvx - bw * ry);
        const Real dy = usy - (bvy + bw * rx);
        const Real dz = have_z ? (usz - bvz) : Real(0);
        acc.m  += cr;
        acc.Sx += cr * rx;   acc.Sy += cr * ry;
        acc.Iz += cr * (rx * rx + ry * ry);
        acc.Px += cr * dx;   acc.Py += cr * dy;
        acc.Lz += cr * (rx * dy - ry * dx);
        acc.Sz += cr * rz;   acc.Pz += cr * dz;
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
    const Shape b = shape;
    const Real bvx = vx, bvy = vy, bw = omega;
    auto ux = ux_, uy = uy_;
    auto fx = fx_, fy = fy_, fz = fz_;
    const auto dens = density_of;
    const auto ldens = lbm_of;
    const Index hx = dom_.hx, hy = dom_.hy, hz = dom_.hz;
    const Real reach2 = shape.reach() * shape.reach();
    const bool have_z = uz_.size() > 0;
    auto uz = uz_.size() > 0 ? uz_ : ux_;      // never dereferenced when !have_z
    const Real bvz = vz;

    Kokkos::parallel_for("penalised_body_apply", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        const Real X = Real(px - hx), Y = Real(py - hy), Z = Real(pz - hz);
        const Real rx = X - b.cx, ry = Y - b.cy, rz = Z - shape_cz(b);
        if (b.outside(rx, ry, rz, reach2)) {
          fx(n) = Real(0); fy(n) = Real(0); fz(n) = Real(0); return;
        }
        const Real c = b.chi(X, Y, Z);
        if (c < Real(1e-6)) {
          fx(n) = Real(0); fy(n) = Real(0); fz(n) = Real(0); return;
        }
        const Real r = dens(n);
        const Real rl = ldens(n);
        const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
        const Real usx = ux(n) - fx(n) * inv;
        const Real usy = uy(n) - fy(n) * inv;
        const Real usz = have_z ? (uz(n) - fz(n) * inv) : Real(0);
        fx(n) = c * Real(2) * r * ((bvx - bw * ry) - usx);
        fy(n) = c * Real(2) * r * ((bvy + bw * rx) - usy);
        fz(n) = have_z ? (c * Real(2) * r * (bvz - usz)) : Real(0);
      });
    Kokkos::fence();
  }

  //----------------------------------------------------------------------------
  // The 3x3, in closed form. Symmetric and positive definite for any body with
  // mass and inertia -- see the header -- so there is no pivoting and no case
  // where this has to give up.
  //----------------------------------------------------------------------------
  // duz is the third TRANSLATION and is deliberately uncoupled from the roll
  // equation: rotation here is about z, so a z-translation exerts no moment on
  // it and receives none. That is exact, but exact only BECAUSE rotation is
  // 2-D -- a genuine 3-D body would couple all six.
  void solve(const BodySums& q, Real& dux, Real& duy, Real& dw, Real& duz) const {
    solve(q, dux, duy, dw);
    const Real A = mass + q.m;
    duz = (A > Real(0) && free_translation)
              ? (Real(2) * q.Pz + (mass - q.m) * bz) / A : Real(0);
  }

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
  View1D<Real> ux_, uy_, uz_;
};

}  // namespace lbm
