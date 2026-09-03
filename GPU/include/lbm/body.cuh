#pragma once
//==============================================================================
//  A rigid body in the flow, by volume penalisation (direct forcing).
//
//  A port of ../src/solver/PenalisedBody.hpp. The physics, the bookkeeping and
//  the two shapes are that file's; what is new here is the reduction, which on a
//  device is the whole difficulty.
//
//  WHY NOT AN IMMERSED BOUNDARY, AND WHY NOT MOVING BOUNCE-BACK. A Peskin-style
//  immersed boundary -- Lagrangian markers, interpolation and spreading kernels
//  -- is more accurate at the surface than this and a great deal more
//  machinery, and for a body whose INTERIOR must move rigidly (a solid square,
//  not a thin shell) a volumetric penalisation enforces the same condition with
//  a fraction of it.
//
//  Moving bounce-back geometry is worse here, and worse on a GPU than anywhere:
//  it needs the solid mask rebuilt and re-uploaded every time the body advances,
//  it needs a refill scheme for nodes the body uncovers, and it quantises the
//  body's position to whole cells, so a smoothly falling square arrives in
//  lattice-sized jumps and each one radiates a pressure pulse. Penalisation has
//  none of those: nothing about the geometry is ever rebuilt, the position and
//  ANGLE are continuous, and there are no fresh nodes. Rotation in particular is
//  nearly free here and would be a rewrite in either of the other two.
//
//  THE METHOD. A smooth solid indicator chi in [0, 1] and a force driving the
//  fluid there toward the body's rigid velocity field,
//
//      F(x) = chi(x) 2 rho(x) [ U + omega x r - u*(x) ],   r = x - x_c,
//
//  applied through the same half-force machinery as everything else, so one step
//  of it takes u to the rigid field exactly where chi = 1. It must run AFTER the
//  macroscopic pass and BEFORE the fluid steps.
//
//  u* IS NOT THE STORED VELOCITY, and this is the one thing that has to be
//  right. The solver defines u = sum_i c_i f_i + F/(2 rho) with F the TOTAL
//  force, so the velocity field ALREADY CONTAINS this body's force from the
//  previous step. Feeding that back applies the correction twice: the body
//  overshoots, the fluid overshoots back, and the pair diverges within a few
//  hundred steps -- measured in the parent, the square left the domain at six
//  million times the impact speed. So the previous contribution is removed
//  first,
//
//      u* = u_stored - F_prev / (2 rho),
//
//  which is what direct forcing is defined against. The arrays hold F_prev
//  already, so this costs one subtraction and no extra state.
//
//  chi IS SMOOTHED over about a cell and a half rather than being a sharp mask.
//  A sharp one is a staircase again -- the body's surface would sit on lattice
//  planes and its effective shape would change as it fell, or as it TURNED --
//  and the smoothing is what lets the body occupy a continuously varying pose.
//
//  NEWTON, AND WHY THE OBVIOUS ARRANGEMENT CANNOT FLOAT. The penalised region is
//  full of fluid the forcing drags along, whose inertia and weight are already
//  inside the reaction; using the body's own mass alone counts them twice.
//  Uhlmann's classical step approximates the internal momentum rate as
//  m_f dU/dt and divides by (m_b - m_f) -- which VANISHES for a neutrally
//  buoyant body and CHANGES SIGN for a light one, so floating, the only
//  interesting case, is precisely what it cannot express.
//
//  The fix is not a different model. The reaction is LINEAR in the body velocity
//  that produces it, so if the force written into the arrays targets the
//  velocity the body is ABOUT to have rather than the one it already has, R
//  moves to the left-hand side and the same equation comes out as
//
//      (m_b + m_f) dU = 2 dP + (m_b - m_f) g,
//
//  with dP = sum chi rho (u* - u_rigid) the inner fluid's momentum deficit.
//  Algebraically this is Uhlmann's equation multiplied through; the difference
//  is that the body and the fluid are now solved at the SAME instant instead of
//  the fluid lagging by a step. The denominator is positive for every mass and
//  the neutrally buoyant case is the best-conditioned point on the line.
//
//  With rotation it is a 3x3 rather than a scalar, and it does not decouple: the
//  first moments S = sum chi rho r are non-zero the moment the body straddles an
//  interface, because half of it stands in water and half in air. The last term
//  of the third row is the RIGHTING MOMENT -- it is not put in by hand, it falls
//  out of the same subtraction that produced (m_b - m_f) g above, and it is why
//  a raft comes back upright.
//
//  ============================ THE GPU PART =================================
//  ONE REDUCTION, SEVEN ACCUMULATORS, AND NO ATOMICS. The sweep writes seven
//  per-block partial sums into a scratch array and the host adds them up, which
//  is the same pattern reduce_population uses in solver.cuh and for the same
//  reasons: the accumulator is DOUBLE even in an FP32 build, there is no
//  compute-capability floor from a double atomicAdd, and the order of
//  accumulation is deterministic run to run. A body that fell to a slightly
//  different place on each run would be a miserable thing to debug.
//
//  The 3x3 is then solved ON THE HOST, once a step, in closed form -- three
//  lines against a kernel launch, and it needs the reduction anyway.
//
//  TWO DENSITIES, AND THEY ARE NOT THE SAME ONE. refresh() takes two functors.
//  The LIQUID density scales the force and everything Newton needs; the LBM
//  density is what the fluid solver divides the force by. With a phase field
//  they are identical and the second is omitted. With a free surface they differ
//  across the whole interface shell, and conflating them is a 1/epsilon
//  amplification on exactly the cells a body enters the water through.
//
//  WHAT THIS DOES NOT MODEL -- unchanged from the parent, and worth repeating
//  because none of it is checked at run time:
//
//   * NO CONTACT LINE, so no wetting. Read the splash, not the meniscus.
//   * NO SUB-CELL SURFACE: the body is resolved to the smoothing width, so a
//     pressure read at a face is smoothed over a cell or two. Integrated force
//     is far more trustworthy than local pressure.
//   * NO COLLISION MODEL. Two bodies, or a body meeting a wall, interpenetrate.
//   * TWO DIMENSIONS. One angle, one angular velocity, about z. A 3-D body is a
//     6x6 with a rotating inertia tensor and a quaternion, and does not exist
//     here. On a domain with nz > 1 this is an infinite PRISM in z: chi does not
//     depend on z, and mass, inertia and the reaction all scale by nz together,
//     so the dynamics are the 2-D ones exactly.
//   * UNIFORM DENSITY ONLY: the centre of mass is the centre of the shape, which
//     is what lets the body's own weight drop out of the roll equation. Ballast
//     low in a hull is exactly what this cannot express.
//==============================================================================
#include "rigid3d.cuh"
#include "streaming.cuh"

namespace lbm {

LBM_HD LBM_INLINE Real body_tanh(Real x) {
#if defined(LBM_DOUBLE)
  return ::tanh(x);
#else
  return ::tanhf(x);
#endif
}
LBM_HD LBM_INLINE Real body_sqrt(Real x) {
#if defined(LBM_DOUBLE)
  return ::sqrt(x);
#else
  return ::sqrtf(x);
#endif
}

//------------------------------------------------------------------------------
// A rectangle: centre, half-extents, tilt.
//
// The tilt is carried as its cosine and sine as well as the angle, because the
// indicator is evaluated at every node every step and computing two
// transcendentals per node -- when the pair is the same for all of them -- would
// be the single most expensive thing here. theta is kept alongside so a
// capsizing body's angle is reported unwrapped rather than folded into (-pi, pi]
// by atan2.
//
// A plain struct, so it is captured into a kernel by value.
//------------------------------------------------------------------------------
struct Rect {
  Real cx = 0, cy = 0;         // centre
  Real hx = 0, hy = 0;         // half width, half height, in the BODY frame
  Real smooth = Real(1.5);     // indicator smoothing width, in cells
  Real theta = 0;              // tilt, unwrapped
  Real ct = Real(1), st = Real(0);

  void set_angle(Real th) { theta = th; ct = Real(std::cos(double(th))); st = Real(std::sin(double(th))); }

  // Circumscribing radius, generous by four smoothing widths so chi is
  // certainly negligible outside it. Used only to reject the vast majority of
  // the domain before the two tanh calls -- which is most of the saving, since
  // a body occupies a few per cent of a domain and tanh is not cheap.
  LBM_HD LBM_INLINE Real reach() const {
    return body_sqrt(hx * hx + hy * hy) + Real(4) * smooth;
  }

  // chi in [0,1]: 1 well inside, 0 well outside, tanh across the faces. The
  // product of the two axis indicators, exact away from the corners and rounding
  // them slightly -- harmless, and it keeps the function separable and cheap.
  // The point is taken into the body frame first, so the shape turns with theta
  // and the rounding turns with it.
  LBM_HD LBM_INLINE Real chi(Real x, Real y, Real) const {
    const Real dx = x - cx, dy = y - cy;
    const Real X =  ct * dx + st * dy;
    const Real Y = -st * dx + ct * dy;
    const Real ax = (hx - (X < 0 ? -X : X)) / smooth;
    const Real ay = (hy - (Y < 0 ? -Y : Y)) / smooth;
    return Real(0.25) * (Real(1) + body_tanh(ax)) * (Real(1) + body_tanh(ay));
  }
  // A PRISM IS UNBOUNDED IN z, so the cheap rejection must not test rz. Getting
  // this wrong culls the whole body except one plane and the reduction silently
  // returns 1/nz of the mass.
  LBM_HD LBM_INLINE bool outside(Real rx, Real ry, Real, Real reach2) const {
    return rx * rx + ry * ry > reach2;
  }
};

//------------------------------------------------------------------------------
// A symmetric wedge, for De Rosis & Enan's Sec. III.G water entry: an apex
// pointing DOWN and two faces rising from it at the deadrise angle, capped at
// the knuckle.
//
// THE ORIGIN IS THE APEX, NOT THE CENTROID, and that is a deliberate difference
// from Rect. A water-entry case prescribes the depth of the apex and compares
// against a wetted length measured from it, so carrying the apex is what the
// driver wants; putting the centroid there would mean converting on every read.
// The consequence is that free rotation about this origin is NOT the physical
// roll axis -- a freely rotating wedge would need the centroid offset
// (0, 2 height / 3) folded in first.
//
// GEOMETRY. In the body frame with the apex at the origin the wedge is
//
//     Y > |X| tan(phi)      above the two faces
//     Y < H                 below the knuckle,  H = half_beam tan(phi)
//
// and the face condition is written as a TRUE NORMAL DISTANCE,
//
//     d_face = Y cos(phi) - |X| sin(phi),
//
// because cos^2 + sin^2 = 1 makes that the perpendicular distance to whichever
// face the point is on. Scaling `smooth` against a non-normalised distance would
// make the smoothing width depend on the deadrise angle -- a shallow wedge would
// get a diffuse surface and a steep one a sharp surface, which quietly turns a
// deadrise sweep into a smoothing sweep.
//
// THE APEX IS ROUNDED, AND IT MATTERS MOST WHERE THE PHYSICS DOES. The product
// of two tanh indicators rounds a convex corner, and a wedge apex is sharper
// than a rectangle's right angle. Measured in the parent as the integral of chi
// over the nominal area b^2 tan(phi):
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
// for it -- the apex is where impact starts and where Wagner's wetted length is
// measured from -- so a shallow-deadrise run wants a BIG body in cells rather
// than a small one with the smoothing turned down. Reducing `smooth` below about
// one cell makes the indicator a step again and reintroduces exactly the
// lattice-quantised pressure pulses penalisation exists to avoid.
//------------------------------------------------------------------------------
struct Wedge {
  Real cx = 0, cy = 0;         // THE APEX
  Real half_beam = 0;          // half-width at the knuckle
  Real smooth = Real(1.5);
  Real theta = 0;
  Real ct = Real(1), st = Real(0);
  Real cphi = Real(1), sphi = Real(0);       // cos, sin of the deadrise

  void set_angle(Real th) { theta = th; ct = Real(std::cos(double(th))); st = Real(std::sin(double(th))); }
  // Deadrise from the horizontal, in radians. Stored as its pair for the same
  // reason theta is: it is used at every node of the domain every step.
  void set_deadrise(Real p) {
    cphi = Real(std::cos(double(p)));  sphi = Real(std::sin(double(p)));
  }
  LBM_HD LBM_INLINE Real height() const { return half_beam * sphi / cphi; }
  // Apex to knuckle corner: sqrt(b^2 + H^2) = b / cos(phi).
  LBM_HD LBM_INLINE Real reach() const { return half_beam / cphi + Real(4) * smooth; }

  LBM_HD LBM_INLINE Real chi(Real x, Real y, Real) const {
    const Real dx = x - cx, dy = y - cy;
    const Real X =  ct * dx + st * dy;
    const Real Y = -st * dx + ct * dy;
    const Real aX = (X < 0) ? -X : X;
    const Real df = (Y * cphi - aX * sphi) / smooth;
    const Real dt = (height() - Y) / smooth;
    return Real(0.25) * (Real(1) + body_tanh(df)) * (Real(1) + body_tanh(dt));
  }
  LBM_HD LBM_INLINE bool outside(Real rx, Real ry, Real, Real reach2) const {
    return rx * rx + ry * ry > reach2;
  }
};

//------------------------------------------------------------------------------
// A SPHERE, and the ONE shape here that is not a prism.
//
// chi is a single tanh in the radius, which is what makes it cheap: one
// transcendental per node against the prisms' two, and no body frame to rotate
// into, because a sphere is its own rotation group. That last point is not a
// convenience -- it is why this shape can be used at all without a quaternion
// (see the note on BodySums::Sz below).
//
// WHAT THIS DOES NOT MAKE. Adding a sphere does NOT turn the body here into a
// 3-D rigid body. Rotation is still one angle about z, and for a sphere that is
// inert: chi is invariant under it, so the roll equation measures nothing and
// changes nothing. Translation is now three components, which for a sphere is
// the whole of the dynamics -- a sphere entering water on its axis carries no
// torque about any axis, so 3-DOF translation is EXACT for that case and is not
// an approximation to a 6-DOF solve. It is not exact for anything asymmetric,
// and the tilted, spinning or tumbling sphere is still absent.
//------------------------------------------------------------------------------
struct Sphere {
  Real cx = 0, cy = 0, cz = 0;      // centre
  Real R = 0;                       // radius
  Real smooth = Real(1.5);          // indicator smoothing width, in cells
  // Present so a Sphere satisfies the same interface as the prisms and can be
  // dropped into PenalisedBody unchanged. Both are inert: see the banner.
  Real theta = 0;
  void set_angle(Real th) { theta = th; }

  LBM_HD LBM_INLINE Real reach() const { return R + Real(4) * smooth; }

  LBM_HD LBM_INLINE Real chi(Real x, Real y, Real z) const {
    const Real dx = x - cx, dy = y - cy, dz = z - cz;
    const Real r = body_sqrt(dx * dx + dy * dy + dz * dz);
    return Real(0.5) * (Real(1) + body_tanh((R - r) / smooth));
  }
  // The only shape whose rejection is genuinely spherical.
  LBM_HD LBM_INLINE bool outside(Real rx, Real ry, Real rz, Real reach2) const {
    return rx * rx + ry * ry + rz * rz > reach2;
  }
};

//------------------------------------------------------------------------------
// A DISC -- a flat circular cylinder, for a skipping stone. six_dof.
//
// A port of ../src/solver/PenalisedBody.hpp's Disc, including the choice that
// matters: THE SYMMETRY AXIS IS BODY y, NOT BODY z. Gravity is -y, so at the
// identity orientation this disc lies FLAT with its axis vertical -- the pose a
// stone is in before it is thrown. Every angle a driver then sets is a
// departure from that, which is what makes an attack angle readable instead of
// being a quaternion nobody can check by eye. On body z the identity pose would
// be a disc standing on its edge and every case would open by rotating 90
// degrees for nothing.
//
// chi is a PRODUCT of a radial and an axial tanh, so the rim and the two faces
// are each smoothed and the edge between them is rounded. Same construction as
// Rect's two tanhs and the same consequence: the penalised disc is slightly
// larger than the nominal one, so volume and inertia are MEASURED rather than
// taken from pi R^2 (2 hy). The parent measures the excess against an
// independent quadrature -- +0.14 % in volume, +0.71 % in the axial inertia and
// +1.22 % in the diametral one at R = 24, hy = 4.8, smooth = 1.
//
// Rm is cached beside q for the reason Rect caches ct and st: chi runs at every
// node of the domain every step, and a quaternion-to-matrix there would be the
// single most expensive thing in the sweep.
//------------------------------------------------------------------------------
struct Disc {
  Real cx = 0, cy = 0, cz = 0;      // centre
  Real R = 0;                       // radius, perpendicular to the axis
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
  LBM_HD LBM_INLINE void set_angle(Real) {}

  // The symmetry axis in the WORLD frame: the body y column of R. This is what
  // a driver reads to report an attack angle and to spin the body about its own
  // axis, and it is one column rather than a re-derivation.
  void axis(Real& ax, Real& ay, Real& az) const {
    ax = Rm(0, 1);  ay = Rm(1, 1);  az = Rm(2, 1);
  }

  LBM_HD LBM_INLINE Real reach() const {
    return body_sqrt(R * R + hy * hy) + Real(4) * smooth;
  }
  LBM_HD LBM_INLINE bool outside(Real rx, Real ry, Real rz, Real reach2) const {
    return rx * rx + ry * ry + rz * rz > reach2;
  }
  LBM_HD LBM_INLINE Real chi(Real x, Real y, Real z) const {
    Real X, Y, Z;
    Rm.tmul(x - cx, y - cy, z - cz, X, Y, Z);       // world -> body is R^T
    const Real rp = body_sqrt(X * X + Z * Z);       // radius about the y axis
    const Real ar = (R - rp) / smooth;
    const Real ay = (hy - (Y < Real(0) ? -Y : Y)) / smooth;
    return Real(0.25) * (Real(1) + body_tanh(ar))
                      * (Real(1) + body_tanh(ay));
  }
};

//------------------------------------------------------------------------------
// The seven integrals of one sweep. Everything the rigid-body solve needs, and
// nothing else.
//------------------------------------------------------------------------------
struct BodySums {
  double m = 0;               // integral chi rho              -- fictitious mass
  double Sx = 0, Sy = 0;      // integral chi rho r            -- its first moments
  double Iz = 0;              // integral chi rho (rx^2+ry^2)  -- inertia ABOUT z
  double Px = 0, Py = 0;      // integral chi rho (u* - rigid) -- momentum deficit
  double Lz = 0;              // integral chi rho r x (u* - rigid)
  // The third translation. Sz is carried for symmetry of the bookkeeping and is
  // NOT coupled into the roll equation: rotation here is about z only, so a
  // first moment along z has no equation to enter. It is reported rather than
  // used, and for a sphere on its axis it is zero to round-off -- which makes it
  // a free check that the entry really is axisymmetric.
  double Sz = 0, Pz = 0;
};

//------------------------------------------------------------------------------
// Density functors. Both are PODs so they capture into a kernel by value.
//
// A body needs the density of the fluid it is standing in: the same body meets a
// thousandfold heavier medium when it reaches the water, and a penalisation that
// ignored that would decelerate it as though it were still in air. The same
// locality is what puts a non-zero S in the roll equation and makes a floating
// body right itself.
//------------------------------------------------------------------------------
struct UniformDensity {
  Real rho = Real(1);
  LBM_HD LBM_INLINE Real operator()(long) const { return rho; }
};

struct FieldDensity {
  const Real* rho = nullptr;
  LBM_HD LBM_INLINE Real operator()(long n) const { return rho[n]; }
};

//------------------------------------------------------------------------------
// The two per-node operations, as plain LBM_HD functions so the host reference
// driver runs the same arithmetic the kernels do.
//
// A CELL WITH NO FLUID IN IT IS NOT FORCED, and the reciprocal is guarded for
// it. With a phase field the local density never approaches zero -- the light
// phase is still a fluid -- and the guard never fires. With a free surface it
// fires constantly: above the waterline there are no populations at all, so a
// force written there is never integrated by anything, and subtracting it from
// u* next step subtracts something that did not happen. Dividing by a near-zero
// density turns that into a large number: measured in the parent with a floor of
// 1e-3, the correction was amplified five hundredfold and settled into a fixed
// point where the reaction exactly cancelled gravity -- a body in free fall that
// HOVERED, with nothing in the output to say why.
//------------------------------------------------------------------------------
struct BodyState {
  Real cx = 0, cy = 0, cz = 0;    // shape origin, for r = x - x_c
  Real vx = 0, vy = 0, vz = 0, omega = 0;
  int nx = 0, ny = 0;
};

template <class Shape, class LiquidOf, class LbmOf>
LBM_HD LBM_INLINE bool body_probe_node(const Shape& b, const BodyState& st,
                                       const Real* ux, const Real* uy,
                                       const Real* uz,
                                       const Real* fx, const Real* fy,
                                       const Real* fz,
                                       LiquidOf dens, LbmOf ldens,
                                       long n, Real reach2, BodySums& out) {
  int x, y, z;
  coords(n, st.nx, st.ny, x, y, z);
  const Real rx = Real(x) - st.cx, ry = Real(y) - st.cy, rz = Real(z) - st.cz;
  // The SHAPE decides the rejection, because a prism must not be culled in z
  // and a sphere must be. Doing it here with one formula was the bug this
  // interface exists to prevent.
  if (b.outside(rx, ry, rz, reach2)) return false;
  const Real c = b.chi(Real(x), Real(y), Real(z));
  if (c < Real(1e-6)) return false;

  const Real r  = dens(n);             // liquid: the force and Newton
  const Real rl = ldens(n);            // LBM: what the fluid divides the force by
  const Real cr = c * r;
  // Undo this body's own previous contribution to the stored velocity -- against
  // the density the fluid actually used, and only where there was a fluid.
  const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
  const Real usx = ux[n] - fx[n] * inv;
  const Real usy = uy[n] - fy[n] * inv;
  // A 2-D caller passes no z velocity at all; nullptr means "no third
  // component" rather than "a component that happens to be zero", so the whole
  // z limb folds away and the prism cases are bit-for-bit what they were.
  const Real usz = (uz && fz) ? (uz[n] - fz[n] * inv) : Real(0);
  // ... and measure what is left against the rigid field it should match.
  const Real dx = usx - (st.vx - st.omega * ry);
  const Real dy = usy - (st.vy + st.omega * rx);
  const Real dz = (uz && fz) ? (usz - st.vz) : Real(0);

  out.m  += double(cr);
  out.Sx += double(cr * rx);   out.Sy += double(cr * ry);
  out.Iz += double(cr * (rx * rx + ry * ry));
  out.Px += double(cr * dx);   out.Py += double(cr * dy);
  out.Lz += double(cr * (rx * dy - ry * dx));
  out.Sz += double(cr * rz);   out.Pz += double(cr * dz);
  return true;
}

// Write F = chi 2 rho (U + omega x r - u*) with the NEW body state, and zero it
// everywhere else -- the body moves, and a force left behind where it used to be
// would keep pushing.
template <class Shape, class LiquidOf, class LbmOf>
LBM_HD LBM_INLINE void body_apply_node(const Shape& b, const BodyState& st,
                                       const Real* ux, const Real* uy,
                                       const Real* uz,
                                       Real* fx, Real* fy, Real* fz,
                                       LiquidOf dens, LbmOf ldens,
                                       long n, Real reach2) {
  int x, y, z;
  coords(n, st.nx, st.ny, x, y, z);
  const bool three = (uz != nullptr);
  const Real rx = Real(x) - st.cx, ry = Real(y) - st.cy, rz = Real(z) - st.cz;
  if (b.outside(rx, ry, rz, reach2)) {
    fx[n] = Real(0); fy[n] = Real(0); fz[n] = Real(0); return;
  }
  const Real c = b.chi(Real(x), Real(y), Real(z));
  if (c < Real(1e-6)) {
    fx[n] = Real(0); fy[n] = Real(0); fz[n] = Real(0); return;
  }

  const Real r  = dens(n);
  const Real rl = ldens(n);
  const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
  const Real usx = ux[n] - fx[n] * inv;
  const Real usy = uy[n] - fy[n] * inv;
  const Real usz = three ? (uz[n] - fz[n] * inv) : Real(0);
  fx[n] = c * Real(2) * r * ((st.vx - st.omega * ry) - usx);
  fy[n] = c * Real(2) * r * ((st.vy + st.omega * rx) - usy);
  fz[n] = three ? (c * Real(2) * r * (st.vz - usz)) : Real(0);
}

//------------------------------------------------------------------------------
// The 6-DOF state a kernel needs, and the two per-node operations that use it.
//
// The angular velocity is a VECTOR here and a scalar in BodyState, which is the
// whole difference between the two paths: the rigid field is U + omega x r with
// three components of omega instead of one, so a point on the body can be
// moving in a direction no planar rotation could produce.
//
// LBM_HD, like their 3-DOF counterparts, so the host reference runs the same
// arithmetic the device does -- which is what makes test/host_body.cpp's
// comparison against the parent's 3x3 worth anything.
//------------------------------------------------------------------------------
struct BodyState6 {
  Real cx = 0, cy = 0, cz = 0;      // shape origin, for r = x - x_c
  Real vx = 0, vy = 0, vz = 0;      // centre-of-mass velocity
  Real wx = 0, wy = 0, wz = 0;      // angular velocity, WORLD frame
  int nx = 0, ny = 0;
};

template <class Shape, class LiquidOf, class LbmOf>
LBM_HD LBM_INLINE bool body_probe6_node(const Shape& b, const BodyState6& st,
                                        const Real* ux, const Real* uy,
                                        const Real* uz,
                                        const Real* fx, const Real* fy,
                                        const Real* fz,
                                        LiquidOf dens, LbmOf ldens,
                                        long n, Real reach2, BodySums6& out) {
  int x, y, z;
  coords(n, st.nx, st.ny, x, y, z);
  const Real rx = Real(x) - st.cx, ry = Real(y) - st.cy, rz = Real(z) - st.cz;
  if (b.outside(rx, ry, rz, reach2)) return false;
  const Real c = b.chi(Real(x), Real(y), Real(z));
  if (c < Real(1e-6)) return false;

  const Real r  = dens(n);             // liquid: the force and Newton
  const Real rl = ldens(n);            // LBM: what the fluid divides the force by
  const Real cr = c * r;
  const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
  const Real usx = ux[n] - fx[n] * inv;
  const Real usy = uy[n] - fy[n] * inv;
  const Real usz = uz[n] - fz[n] * inv;
  // The rigid field, with omega a vector.
  const Real vrx = st.vx + (st.wy * rz - st.wz * ry);
  const Real vry = st.vy + (st.wz * rx - st.wx * rz);
  const Real vrz = st.vz + (st.wx * ry - st.wy * rx);
  const Real dx = usx - vrx, dy = usy - vry, dz = usz - vrz;

  out.m   += double(cr);
  out.Sx  += double(cr) * double(rx);
  out.Sy  += double(cr) * double(ry);
  out.Sz  += double(cr) * double(rz);
  out.Jxx += double(cr) * double(rx) * double(rx);
  out.Jyy += double(cr) * double(ry) * double(ry);
  out.Jzz += double(cr) * double(rz) * double(rz);
  out.Jxy += double(cr) * double(rx) * double(ry);
  out.Jxz += double(cr) * double(rx) * double(rz);
  out.Jyz += double(cr) * double(ry) * double(rz);
  out.Px  += double(cr) * double(dx);
  out.Py  += double(cr) * double(dy);
  out.Pz  += double(cr) * double(dz);
  out.Lx  += double(cr) * (double(ry) * dz - double(rz) * dy);
  out.Ly  += double(cr) * (double(rz) * dx - double(rx) * dz);
  out.Lz  += double(cr) * (double(rx) * dy - double(ry) * dx);
  return true;
}

template <class Shape, class LiquidOf, class LbmOf>
LBM_HD LBM_INLINE void body_apply6_node(const Shape& b, const BodyState6& st,
                                        const Real* ux, const Real* uy,
                                        const Real* uz,
                                        Real* fx, Real* fy, Real* fz,
                                        LiquidOf dens, LbmOf ldens,
                                        long n, Real reach2) {
  int x, y, z;
  coords(n, st.nx, st.ny, x, y, z);
  const Real rx = Real(x) - st.cx, ry = Real(y) - st.cy, rz = Real(z) - st.cz;
  if (b.outside(rx, ry, rz, reach2)) {
    fx[n] = Real(0); fy[n] = Real(0); fz[n] = Real(0); return;
  }
  const Real c = b.chi(Real(x), Real(y), Real(z));
  if (c < Real(1e-6)) {
    fx[n] = Real(0); fy[n] = Real(0); fz[n] = Real(0); return;
  }
  const Real r  = dens(n);
  const Real rl = ldens(n);
  const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
  const Real usx = ux[n] - fx[n] * inv;
  const Real usy = uy[n] - fy[n] * inv;
  const Real usz = uz[n] - fz[n] * inv;
  const Real vrx = st.vx + (st.wy * rz - st.wz * ry);
  const Real vry = st.vy + (st.wz * rx - st.wx * rz);
  const Real vrz = st.vz + (st.wx * ry - st.wy * rx);
  fx[n] = c * Real(2) * r * (vrx - usx);
  fy[n] = c * Real(2) * r * (vry - usy);
  fz[n] = c * Real(2) * r * (vrz - usz);
}

//------------------------------------------------------------------------------
// The 3x3, in closed form. Symmetric, and positive definite for any body with
// mass and inertia: Cauchy-Schwarz on the measure chi rho gives |S|^2 <= m_f I_f
// <= A B, so the Schur complement B - |S|^2/A is positive. No pivoting, and no
// case where this has to give up.
//
// Locking translation removes the COUPLING rather than the rows: with dU pinned
// to zero the roll equation is B dw = rw on its own, which is a different and
// correct answer, not the 3x3 with two rows discarded.
//------------------------------------------------------------------------------
struct BodyProperties {
  Real mass = 0;              // m_b
  Real inertia = 0;           // I_b about the shape origin
  Real bx = 0, by = 0, bz = 0;  // body force per unit mass -- the SAME vector the
                              // collision operator is given, or the buoyancy
                              // term is inconsistent with it
  bool free_translation = true;
  bool free_rotation = true;
};

// duz is the third TRANSLATION and is deliberately uncoupled from the roll
// equation: rotation here is about z, so a z-translation exerts no moment on it
// and receives none. That is exact, not an omission -- but it is exact only
// because rotation is 2-D. A genuine 3-D body would couple all six.
inline void body_solve(const BodyProperties& p, const BodySums& q,
                       double& dux, double& duy, double& dw, double& duz) {
  const double A = double(p.mass) + q.m;
  const double B = double(p.inertia) + q.Iz;
  dux = duy = dw = duz = 0;
  if (A <= 0) return;

  const double rx = 2.0 * q.Px + (double(p.mass) - q.m) * double(p.bx);
  const double ry = 2.0 * q.Py + (double(p.mass) - q.m) * double(p.by);
  // The last term is the RIGHTING MOMENT, -(S x g). It is not put in for that
  // purpose: it falls out of the same subtraction that produced (m_b - m_f) g
  // above, and it is exactly the hydrostatic couple metacentric theory calls
  // rho g V GM sin(theta).
  const double rw = 2.0 * q.Lz - (q.Sx * double(p.by) - q.Sy * double(p.bx));

  const double invA = p.free_translation ? 1.0 / A : 0.0;
  if (p.free_rotation && B > 0) {
    const double schur = B - (q.Sx * q.Sx + q.Sy * q.Sy) * invA;
    if (schur > 0) dw = (rw - (q.Sx * ry - q.Sy * rx) * invA) / schur;
  }
  dux = (rx + q.Sy * dw) * invA;
  duy = (ry - q.Sx * dw) * invA;
  const double rzz = 2.0 * q.Pz + (double(p.mass) - q.m) * double(p.bz);
  duz = rzz * invA;
}

// The 2-D entry point, kept so every existing caller and test is untouched.
inline void body_solve(const BodyProperties& p, const BodySums& q,
                       double& dux, double& duy, double& dw) {
  double duz = 0;
  body_solve(p, q, dux, duy, dw, duz);
}

// The force and torque the fluid exerts on the body, in closed form from the
// solve. Exact to round-off against -sum F, and free -- but a DERIVED
// diagnostic, not an independent measurement of it. The physics enters through
// dP and dL, which are measured.
struct BodyReaction {
  double fx = 0, fy = 0;
  double torque = 0;
  double fluid_mass = 0;      // mass of fluid standing in the penalised region
  double righting = 0;        // -(S x g), the only part with a closed form to check
};

inline BodyReaction body_reaction(const BodyProperties& p, const BodySums& q,
                                  double dux, double duy, double dw) {
  const double Zx = -q.Sy, Zy = q.Sx;             // z x S
  BodyReaction r;
  r.fx = 2.0 * q.Px - 2.0 * (q.m * dux + dw * Zx);
  r.fy = 2.0 * q.Py - 2.0 * (q.m * duy + dw * Zy);
  r.torque = 2.0 * q.Lz - 2.0 * (q.Iz * dw + (q.Sx * duy - q.Sy * dux));
  r.fluid_mass = q.m;
  r.righting = -(q.Sx * double(p.by) - q.Sy * double(p.bx));
  return r;
}

#if defined(__CUDACC__)

//------------------------------------------------------------------------------
// The sweep. Seven per-block partial sums; the host adds them. See the banner.
//------------------------------------------------------------------------------
template <class Shape, class LiquidOf, class LbmOf>
__global__ void body_probe_kernel(Shape b, BodyState st,
                                  const Real* __restrict__ ux,
                                  const Real* __restrict__ uy,
                                  const Real* __restrict__ uz,
                                  const Real* __restrict__ fx,
                                  const Real* __restrict__ fy,
                                  const Real* __restrict__ fz,
                                  LiquidOf dens, LbmOf ldens,
                                  long N, Real reach2, double* __restrict__ partial) {
  extern __shared__ double sm[];             // 9 * blockDim.x
  const unsigned T = blockDim.x;
  BodySums acc;
  const long stride = long(T) * gridDim.x;
  for (long n = long(blockIdx.x) * T + threadIdx.x; n < N; n += stride)
    body_probe_node(b, st, ux, uy, uz, fx, fy, fz, dens, ldens, n, reach2, acc);

  // Written out rather than through `&acc.m` as seven contiguous doubles: that
  // works and is one line, but it is an aliasing assumption about a struct
  // layout, and the compiler is entitled to disagree.
  sm[0 * T + threadIdx.x] = acc.m;
  sm[1 * T + threadIdx.x] = acc.Sx;
  sm[2 * T + threadIdx.x] = acc.Sy;
  sm[3 * T + threadIdx.x] = acc.Iz;
  sm[4 * T + threadIdx.x] = acc.Px;
  sm[5 * T + threadIdx.x] = acc.Py;
  sm[6 * T + threadIdx.x] = acc.Lz;
  sm[7 * T + threadIdx.x] = acc.Sz;
  sm[8 * T + threadIdx.x] = acc.Pz;
  __syncthreads();
  for (unsigned s = T / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      for (int k = 0; k < 9; ++k) sm[k * T + threadIdx.x] += sm[k * T + threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int k = 0; k < 9; ++k) partial[k * gridDim.x + blockIdx.x] = sm[k * T];
}

template <class Shape, class LiquidOf, class LbmOf>
__global__ void body_apply_kernel(Shape b, BodyState st,
                                  const Real* __restrict__ ux,
                                  const Real* __restrict__ uy,
                                  const Real* __restrict__ uz,
                                  Real* __restrict__ fx, Real* __restrict__ fy,
                                  Real* __restrict__ fz,
                                  LiquidOf dens, LbmOf ldens, long N, Real reach2) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  body_apply_node(b, st, ux, uy, uz, fx, fy, fz, dens, ldens, n, reach2);
}

// Integrals of the indicator alone: its area and its second moment about the
// origin. Taken from chi rather than from 4 hx hy and the textbook
// (w^2 + h^2)/12 so that mass and inertia describe the body the force actually
// acts on -- the smoothing makes the penalised body slightly larger than the
// nominal one, and nominal values would bias both. Both are invariant under the
// body's own rotation, so this runs once at setup.
template <class Shape>
__global__ void body_moments_kernel(Shape b, int nx, int ny, long N,
                                    double* __restrict__ partial) {
  extern __shared__ double sm2[];            // 2 * blockDim.x
  const unsigned T = blockDim.x;
  double a = 0, s = 0;
  const long stride = long(T) * gridDim.x;
  for (long n = long(blockIdx.x) * T + threadIdx.x; n < N; n += stride) {
    int x, y, z;
    coords(n, nx, ny, x, y, z);
    const Real c = b.chi(Real(x), Real(y), Real(z));
    const Real rx = Real(x) - b.cx, ry = Real(y) - b.cy;
    a += double(c);
    // The second moment is about z, matching BodySums::Iz. For a sphere this is
    // the polar moment of a sphere rather than its full inertia -- correct for
    // the only rotation this code has, and inert for a sphere anyway.
    s += double(c) * (double(rx) * rx + double(ry) * ry);
  }
  sm2[threadIdx.x] = a;  sm2[T + threadIdx.x] = s;
  __syncthreads();
  for (unsigned k = T / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) {
      sm2[threadIdx.x] += sm2[threadIdx.x + k];
      sm2[T + threadIdx.x] += sm2[T + threadIdx.x + k];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = sm2[0];
    partial[gridDim.x + blockIdx.x] = sm2[T];
  }
}

//------------------------------------------------------------------------------
// The 6-DOF sweep. SIXTEEN per-block partial sums instead of nine; the host
// adds them. BODY6_SUMS names the count in all four places it appears -- shared
// memory, the launch, the partial buffer and the unpack -- because a mismatch
// between any two of them is a silent wrong answer rather than a crash.
//------------------------------------------------------------------------------
template <class Shape, class LiquidOf, class LbmOf>
__global__ void body_probe6_kernel(Shape b, BodyState6 st,
                                   const Real* __restrict__ ux,
                                   const Real* __restrict__ uy,
                                   const Real* __restrict__ uz,
                                   const Real* __restrict__ fx,
                                   const Real* __restrict__ fy,
                                   const Real* __restrict__ fz,
                                   LiquidOf dens, LbmOf ldens,
                                   long N, Real reach2,
                                   double* __restrict__ partial) {
  extern __shared__ double sm6[];            // BODY6_SUMS * blockDim.x
  const unsigned T = blockDim.x;
  BodySums6 acc;
  const long stride = long(T) * gridDim.x;
  for (long n = long(blockIdx.x) * T + threadIdx.x; n < N; n += stride)
    body_probe6_node(b, st, ux, uy, uz, fx, fy, fz, dens, ldens, n, reach2, acc);

  // Written out rather than reinterpreted as sixteen contiguous doubles: that
  // works and is one line, but it is an aliasing assumption about a struct
  // layout and the compiler is entitled to disagree.
  sm6[ 0 * T + threadIdx.x] = acc.m;
  sm6[ 1 * T + threadIdx.x] = acc.Sx;
  sm6[ 2 * T + threadIdx.x] = acc.Sy;
  sm6[ 3 * T + threadIdx.x] = acc.Sz;
  sm6[ 4 * T + threadIdx.x] = acc.Jxx;
  sm6[ 5 * T + threadIdx.x] = acc.Jyy;
  sm6[ 6 * T + threadIdx.x] = acc.Jzz;
  sm6[ 7 * T + threadIdx.x] = acc.Jxy;
  sm6[ 8 * T + threadIdx.x] = acc.Jxz;
  sm6[ 9 * T + threadIdx.x] = acc.Jyz;
  sm6[10 * T + threadIdx.x] = acc.Px;
  sm6[11 * T + threadIdx.x] = acc.Py;
  sm6[12 * T + threadIdx.x] = acc.Pz;
  sm6[13 * T + threadIdx.x] = acc.Lx;
  sm6[14 * T + threadIdx.x] = acc.Ly;
  sm6[15 * T + threadIdx.x] = acc.Lz;
  __syncthreads();
  for (unsigned s = T / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      for (int k = 0; k < BODY6_SUMS; ++k)
        sm6[k * T + threadIdx.x] += sm6[k * T + threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int k = 0; k < BODY6_SUMS; ++k)
      partial[k * gridDim.x + blockIdx.x] = sm6[k * T];
}

template <class Shape, class LiquidOf, class LbmOf>
__global__ void body_apply6_kernel(Shape b, BodyState6 st,
                                   const Real* __restrict__ ux,
                                   const Real* __restrict__ uy,
                                   const Real* __restrict__ uz,
                                   Real* __restrict__ fx, Real* __restrict__ fy,
                                   Real* __restrict__ fz,
                                   LiquidOf dens, LbmOf ldens, long N,
                                   Real reach2) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  body_apply6_node(b, st, ux, uy, uz, fx, fy, fz, dens, ldens, n, reach2);
}

// Volume and the six products integral chi ri rj at the CURRENT pose, in the
// world frame, with rho = 1. Seven sums, reduced the same way. Unlike
// body_moments_kernel this is NOT pose invariant -- the products are world
// frame -- which is exactly why set_uniform_density6 rotates the result back
// into the body frame before storing it.
template <class Shape>
__global__ void body_moments6_kernel(Shape b, int nx, int ny, long N,
                                     double* __restrict__ partial) {
  extern __shared__ double sm7[];            // 7 * blockDim.x
  const unsigned T = blockDim.x;
  double a[7] = {0, 0, 0, 0, 0, 0, 0};
  const long stride = long(T) * gridDim.x;
  for (long n = long(blockIdx.x) * T + threadIdx.x; n < N; n += stride) {
    int x, y, z;
    coords(n, nx, ny, x, y, z);
    const Real rx = Real(x) - b.cx, ry = Real(y) - b.cy, rz = Real(z) - b.cz;
    const Real c = b.chi(Real(x), Real(y), Real(z));
    if (c < Real(1e-6)) continue;
    const double d = double(c);
    a[0] += d;
    a[1] += d * double(rx) * double(rx);
    a[2] += d * double(ry) * double(ry);
    a[3] += d * double(rz) * double(rz);
    a[4] += d * double(rx) * double(ry);
    a[5] += d * double(rx) * double(rz);
    a[6] += d * double(ry) * double(rz);
  }
  for (int k = 0; k < 7; ++k) sm7[k * T + threadIdx.x] = a[k];
  __syncthreads();
  for (unsigned s = T / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      for (int k = 0; k < 7; ++k)
        sm7[k * T + threadIdx.x] += sm7[k * T + threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int k = 0; k < 7; ++k)
      partial[k * gridDim.x + blockIdx.x] = sm7[k * T];
}

//==============================================================================
//  Host-side driver.
//==============================================================================
template <class Shape = Rect>
class PenalisedBody {
 public:
  PenalisedBody(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    N_ = long(nx) * ny * nz;
    LBM_CUDA_CHECK(cudaMalloc(&fx_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&fy_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&fz_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(fx_, 0, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(fy_, 0, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(fz_, 0, sizeof(Real) * N_));
    // BODY6_SUMS, not 9: the 6-DOF probe reduces sixteen sums through the same
    // buffer, and 16 covers the 3-DOF path's nine and moments6's seven too.
    LBM_CUDA_CHECK(cudaMalloc(&partial_, sizeof(double) * BODY6_SUMS * GRID));
  }
  ~PenalisedBody() {
    cudaFree(fx_); cudaFree(fy_); cudaFree(fz_); cudaFree(partial_);
  }
  PenalisedBody(const PenalisedBody&) = delete;
  PenalisedBody& operator=(const PenalisedBody&) = delete;

  //---- state, public so a driver can prescribe, clamp or read any of it ------
  Shape shape;
  BodyProperties props;
  Real vx = 0, vy = 0, vz = 0;     // centre-of-mass velocity
  Real omega = 0;                  // angular velocity about z -- 3-DOF path
  // THE SIX-DOF STATE, used only by a shape whose six_dof is true. The angular
  // velocity is a VECTOR and the inertia is a tensor held in the BODY frame --
  // the world-frame tensor changes as the body turns, so storing that would
  // mean recomputing it from something, and the body frame is the something.
  Real wx = 0, wy = 0, wz = 0;     // angular velocity, world frame
  Mat3 inertia_body;               // I_b in the body frame; rotated per step

  // Device pointers owned by the fluid solver. The two-argument form leaves the
  // z limb switched OFF -- uz_ stays null, and every prism case is the
  // arithmetic it was before a sphere existed. Pass three to enable it.
  void couple_velocity(const Real* ux, const Real* uy) { ux_ = ux; uy_ = uy; uz_ = nullptr; }
  void couple_velocity(const Real* ux, const Real* uy, const Real* uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  const Real* fx() const { return fx_; }
  const Real* fy() const { return fy_; }
  const Real* fz() const { return fz_; }

  struct Moments { double area = 0, second = 0; };
  Moments indicator_moments() {
    double* d = nullptr;
    LBM_CUDA_CHECK(cudaMalloc(&d, sizeof(double) * 2 * GRID));
    body_moments_kernel<<<GRID, BLOCK, sizeof(double) * 2 * BLOCK>>>(
        shape, nx_, ny_, N_, d);
    LBM_CUDA_CHECK(cudaGetLastError());
    std::vector<double> h(std::size_t(2 * GRID));
    LBM_CUDA_CHECK(cudaMemcpy(h.data(), d, sizeof(double) * 2 * GRID,
                              cudaMemcpyDeviceToHost));
    cudaFree(d);
    Moments m;
    for (int i = 0; i < GRID; ++i) { m.area += h[std::size_t(i)]; m.second += h[std::size_t(GRID + i)]; }
    return m;
  }

  // Mass and inertia of a body of uniform density, from those moments.
  void set_uniform_density(Real rho_b) {
    const Moments m = indicator_moments();
    props.mass    = Real(double(rho_b) * m.area);
    props.inertia = Real(double(rho_b) * m.second);
  }

  //--------------------------------------------------------------------------
  // One coupling step: measure, solve Newton for the new body state, write the
  // force that drives the fluid to it. Must run AFTER the macroscopic pass and
  // BEFORE the fluid steps.
  //--------------------------------------------------------------------------
  template <class DensityOf>
  BodyReaction refresh(DensityOf dens) { return refresh(dens, dens); }

  template <class LiquidOf, class LbmOf>
  BodyReaction refresh(LiquidOf dens, LbmOf ldens) {
    const BodySums q = probe(dens, ldens);
    double dux = 0, duy = 0, dw = 0, duz = 0;
    body_solve(props, q, dux, duy, dw, duz);
    if (props.free_translation) {
      vx += Real(dux); vy += Real(duy);
      if (uz_) vz += Real(duz);
    }
    if (props.free_rotation)    { omega += Real(dw); }
    apply(dens, ldens);
    return body_reaction(props, q, dux, duy, dw);
  }

  // Advance the pose. Explicit Euler at dt = 1 -- the fluid step is the
  // timescale and there is nothing faster in the body to resolve.
  void advance() {
    shape.cx += vx;
    shape.cy += vy;
    advance_z(shape);
    shape.set_angle(shape.theta + omega);
  }

  //--------------------------------------------------------------------------
  //  THE SIX-DOF PATH. Only instantiated for a shape with six_dof, because
  //  members of a class template are instantiated on use -- so these may refer
  //  to Disc's q and Rm freely without a Rect user paying for it or failing to
  //  compile.
  //--------------------------------------------------------------------------

  // The 2-DOF Reaction with the force and the moment completed to vectors.
  // Same closed forms, generalised: dw * (zhat x S) becomes dW x S and Iz * dw
  // becomes I_f dW.
  struct Reaction6 {
    Real fx = 0, fy = 0, fz = 0;      // force the fluid exerts on the body
    Real tx = 0, ty = 0, tz = 0;      // and its moment about the shape origin
    double fluid_mass = 0;            // fluid standing in the penalised region
    Real rx = 0, ry = 0, rz = 0;      // the hydrostatic couple -(S x g)
  };

  BodySums6 moments6() {
    double* d = nullptr;
    LBM_CUDA_CHECK(cudaMalloc(&d, sizeof(double) * 7 * GRID));
    body_moments6_kernel<<<GRID, BLOCK, sizeof(double) * 7 * BLOCK>>>(
        shape, nx_, ny_, N_, d);
    LBM_CUDA_CHECK(cudaGetLastError());
    std::vector<double> h(std::size_t(7 * GRID));
    LBM_CUDA_CHECK(cudaMemcpy(h.data(), d, sizeof(double) * 7 * GRID,
                              cudaMemcpyDeviceToHost));
    cudaFree(d);
    double t[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 7; ++k)
      for (int i = 0; i < GRID; ++i) t[k] += h[std::size_t(k * GRID + i)];
    BodySums6 q;
    q.m = t[0];
    q.Jxx = t[1]; q.Jyy = t[2]; q.Jzz = t[3];
    q.Jxy = t[4]; q.Jxz = t[5]; q.Jyz = t[6];
    return q;
  }

  double penalised_volume() { return moments6().m; }

  // Mass and the BODY-frame inertia tensor of a body of uniform density.
  // Measured from chi rather than from the nominal shape, for the reason
  // set_uniform_density gives: the smoothing makes the penalised body slightly
  // larger and nominal values would bias both.
  //
  // The measurement is in the WORLD frame at the current pose, so it is rotated
  // BACK: I_body = R^T I_world R. Getting that inverse the wrong way round
  // leaves an inertia that is right only at the release orientation, is still
  // symmetric and positive definite, and produces a plausible tumble rather
  // than a failure. ../tests/test_rigid3d.cpp block 7 pins the direction on an
  // anisotropic slab, because a CUBE cannot catch it -- a uniform cube's
  // inertia is isotropic and both directions agree.
  void set_uniform_density6(Real rho_b) {
    BodySums6 q = moments6();
    const double r = double(rho_b);
    q.m *= r;
    q.Jxx *= r; q.Jyy *= r; q.Jzz *= r;
    q.Jxy *= r; q.Jxz *= r; q.Jyz *= r;
    props.mass = Real(q.m);
    inertia_body = rotate_tensor(shape.Rm.transposed(), q.fluid_inertia());
  }

  template <class DensityOf>
  Reaction6 refresh6(DensityOf dens) { return refresh6(dens, dens); }

  template <class LiquidOf, class LbmOf>
  Reaction6 refresh6(LiquidOf dens, LbmOf ldens) {
    const BodySums6 q = probe6(dens, ldens);
    Body6Properties p;
    p.mass = props.mass;
    // The body's tensor rotated into the world frame at the CURRENT pose. This
    // is the line that makes it a 3-D body rather than three translations: a
    // tumbling body's resistance to a torque depends on which way it faces.
    p.inertia_world = rotate_tensor(shape.Rm, inertia_body);
    // The current angular velocity, for the gyroscopic term.
    p.wx = wx;  p.wy = wy;  p.wz = wz;
    p.bx = props.bx;  p.by = props.by;  p.bz = props.bz;
    p.free_translation = props.free_translation;
    p.free_rotation = props.free_rotation;
    double dU[3], dW[3];
    body6_solve(p, q, dU, dW);
    if (props.free_translation) {
      vx += Real(dU[0]);  vy += Real(dU[1]);  vz += Real(dU[2]);
    }
    if (props.free_rotation) {
      wx += Real(dW[0]);  wy += Real(dW[1]);  wz += Real(dW[2]);
    }
    apply6(dens, ldens);

    // F = 2 dP - 2 (m_f dU + dW x S),  T = 2 dL - 2 (I_f dW + S x dU).
    const double S[3] = {q.Sx, q.Sy, q.Sz};
    const double WxS[3] = {dW[1] * S[2] - dW[2] * S[1],
                           dW[2] * S[0] - dW[0] * S[2],
                           dW[0] * S[1] - dW[1] * S[0]};
    const double SxU[3] = {S[1] * dU[2] - S[2] * dU[1],
                           S[2] * dU[0] - S[0] * dU[2],
                           S[0] * dU[1] - S[1] * dU[0]};
    const Mat3 If = q.fluid_inertia();
    double IfW[3];
    for (int i = 0; i < 3; ++i)
      IfW[i] = double(If(i, 0)) * dW[0] + double(If(i, 1)) * dW[1]
             + double(If(i, 2)) * dW[2];
    const double g[3] = {double(props.bx), double(props.by), double(props.bz)};
    Reaction6 out;
    out.fx = Real(2.0 * q.Px - 2.0 * (q.m * dU[0] + WxS[0]));
    out.fy = Real(2.0 * q.Py - 2.0 * (q.m * dU[1] + WxS[1]));
    out.fz = Real(2.0 * q.Pz - 2.0 * (q.m * dU[2] + WxS[2]));
    out.tx = Real(2.0 * q.Lx - 2.0 * (IfW[0] + SxU[0]));
    out.ty = Real(2.0 * q.Ly - 2.0 * (IfW[1] + SxU[1]));
    out.tz = Real(2.0 * q.Lz - 2.0 * (IfW[2] + SxU[2]));
    out.fluid_mass = q.m;
    out.rx = Real(-(S[1] * g[2] - S[2] * g[1]));
    out.ry = Real(-(S[2] * g[0] - S[0] * g[2]));
    out.rz = Real(-(S[0] * g[1] - S[1] * g[0]));
    return out;
  }

  // Advance the pose: the centre by Euler, the orientation by one quaternion
  // step. dt = 1, because the fluid step is the timescale.
  void advance6() {
    shape.cx += vx;
    shape.cy += vy;
    shape.cz += vz;
    Quat q = shape.q;
    q.integrate(wx, wy, wz, Real(1));
    shape.set_orientation(q);
  }

  template <class LiquidOf, class LbmOf>
  BodySums6 probe6(LiquidOf dens, LbmOf ldens) {
    const Real r2 = shape.reach() * shape.reach();
    body_probe6_kernel<<<GRID, BLOCK, sizeof(double) * BODY6_SUMS * BLOCK>>>(
        shape, state6(), ux_, uy_, uz_, fx_, fy_, fz_, dens, ldens, N_, r2,
        partial_);
    LBM_CUDA_CHECK(cudaGetLastError());
    std::vector<double> h(std::size_t(BODY6_SUMS * GRID));
    LBM_CUDA_CHECK(cudaMemcpy(h.data(), partial_,
                              sizeof(double) * BODY6_SUMS * GRID,
                              cudaMemcpyDeviceToHost));
    double t[BODY6_SUMS] = {};
    for (int k = 0; k < BODY6_SUMS; ++k)
      for (int i = 0; i < GRID; ++i) t[k] += h[std::size_t(k * GRID + i)];
    BodySums6 q;
    q.m = t[0];
    q.Sx = t[1];  q.Sy = t[2];  q.Sz = t[3];
    q.Jxx = t[4]; q.Jyy = t[5]; q.Jzz = t[6];
    q.Jxy = t[7]; q.Jxz = t[8]; q.Jyz = t[9];
    q.Px = t[10]; q.Py = t[11]; q.Pz = t[12];
    q.Lx = t[13]; q.Ly = t[14]; q.Lz = t[15];
    return q;
  }

  template <class LiquidOf, class LbmOf>
  void apply6(LiquidOf dens, LbmOf ldens) {
    const Real r2 = shape.reach() * shape.reach();
    const int B = 128;
    body_apply6_kernel<<<int((N_ + B - 1) / B), B>>>(
        shape, state6(), ux_, uy_, uz_, fx_, fy_, fz_, dens, ldens, N_, r2);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  template <class LiquidOf, class LbmOf>
  BodySums probe(LiquidOf dens, LbmOf ldens) {
    const Real r2 = shape.reach() * shape.reach();
    body_probe_kernel<<<GRID, BLOCK, sizeof(double) * 9 * BLOCK>>>(
        shape, state(), ux_, uy_, uz_, fx_, fy_, fz_, dens, ldens, N_, r2, partial_);
    LBM_CUDA_CHECK(cudaGetLastError());
    std::vector<double> h(std::size_t(9 * GRID));
    LBM_CUDA_CHECK(cudaMemcpy(h.data(), partial_, sizeof(double) * 9 * GRID,
                              cudaMemcpyDeviceToHost));
    double t[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 9; ++k)
      for (int i = 0; i < GRID; ++i) t[k] += h[std::size_t(k * GRID + i)];
    BodySums q;
    q.m = t[0]; q.Sx = t[1]; q.Sy = t[2]; q.Iz = t[3];
    q.Px = t[4]; q.Py = t[5]; q.Lz = t[6];
    q.Sz = t[7]; q.Pz = t[8];
    return q;
  }

  template <class LiquidOf, class LbmOf>
  void apply(LiquidOf dens, LbmOf ldens) {
    const Real r2 = shape.reach() * shape.reach();
    const int B = 128;
    body_apply_kernel<<<int((N_ + B - 1) / B), B>>>(
        shape, state(), ux_, uy_, uz_, fx_, fy_, fz_, dens, ldens, N_, r2);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

 private:
  static constexpr int BLOCK = 256, GRID = 256;

  // A prism has no cz to read, so the z limb of the state is filled only for a
  // shape that has one. SFINAE rather than a flag: a Rect that grew a cz member
  // by accident would otherwise start behaving like a sphere silently.
  template <class S> static auto shape_cz(const S& sh, int) -> decltype(sh.cz) { return sh.cz; }
  template <class S> static Real shape_cz(const S&, long) { return Real(0); }

  template <class S> static auto bump_z(S& sh, Real dz, int) -> decltype(sh.cz, void()) { sh.cz += dz; }
  template <class S> static void bump_z(S&, Real, long) {}
  template <class S> void advance_z(S& sh) { bump_z(sh, vz, 0); }

  BodyState6 state6() const {
    BodyState6 st;
    st.cx = shape.cx;  st.cy = shape.cy;  st.cz = shape_cz(shape, 0);
    st.vx = vx;  st.vy = vy;  st.vz = vz;
    st.wx = wx;  st.wy = wy;  st.wz = wz;
    st.nx = nx_; st.ny = ny_;
    return st;
  }

  BodyState state() const {
    BodyState st;
    st.cx = shape.cx;  st.cy = shape.cy;  st.cz = shape_cz(shape, 0);
    st.vx = vx;  st.vy = vy;  st.vz = vz;  st.omega = omega;
    st.nx = nx_; st.ny = ny_;
    return st;
  }

  int nx_, ny_, nz_;
  long N_;
  Real *fx_ = nullptr, *fy_ = nullptr, *fz_ = nullptr;
  double* partial_ = nullptr;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
};

#endif  // __CUDACC__

}  // namespace lbm
