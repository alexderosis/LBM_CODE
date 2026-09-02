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
  LBM_HD LBM_INLINE Real chi(Real x, Real y) const {
    const Real dx = x - cx, dy = y - cy;
    const Real X =  ct * dx + st * dy;
    const Real Y = -st * dx + ct * dy;
    const Real ax = (hx - (X < 0 ? -X : X)) / smooth;
    const Real ay = (hy - (Y < 0 ? -Y : Y)) / smooth;
    return Real(0.25) * (Real(1) + body_tanh(ax)) * (Real(1) + body_tanh(ay));
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

  LBM_HD LBM_INLINE Real chi(Real x, Real y) const {
    const Real dx = x - cx, dy = y - cy;
    const Real X =  ct * dx + st * dy;
    const Real Y = -st * dx + ct * dy;
    const Real aX = (X < 0) ? -X : X;
    const Real df = (Y * cphi - aX * sphi) / smooth;
    const Real dt = (height() - Y) / smooth;
    return Real(0.25) * (Real(1) + body_tanh(df)) * (Real(1) + body_tanh(dt));
  }
};

//------------------------------------------------------------------------------
// The seven integrals of one sweep. Everything the rigid-body solve needs, and
// nothing else.
//------------------------------------------------------------------------------
struct BodySums {
  double m = 0;               // integral chi rho              -- fictitious mass
  double Sx = 0, Sy = 0;      // integral chi rho r            -- its first moments
  double Iz = 0;              // integral chi rho |r|^2        -- its inertia
  double Px = 0, Py = 0;      // integral chi rho (u* - rigid) -- momentum deficit
  double Lz = 0;              // integral chi rho r x (u* - rigid)
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
  Real cx = 0, cy = 0;            // shape origin, for r = x - x_c
  Real vx = 0, vy = 0, omega = 0;
  int nx = 0, ny = 0;
};

template <class Shape, class LiquidOf, class LbmOf>
LBM_HD LBM_INLINE bool body_probe_node(const Shape& b, const BodyState& st,
                                       const Real* ux, const Real* uy,
                                       const Real* fx, const Real* fy,
                                       LiquidOf dens, LbmOf ldens,
                                       long n, Real reach2, BodySums& out) {
  int x, y, z;
  coords(n, st.nx, st.ny, x, y, z);
  const Real rx = Real(x) - st.cx, ry = Real(y) - st.cy;
  if (rx * rx + ry * ry > reach2) return false;
  const Real c = b.chi(Real(x), Real(y));
  if (c < Real(1e-6)) return false;

  const Real r  = dens(n);             // liquid: the force and Newton
  const Real rl = ldens(n);            // LBM: what the fluid divides the force by
  const Real cr = c * r;
  // Undo this body's own previous contribution to the stored velocity -- against
  // the density the fluid actually used, and only where there was a fluid.
  const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
  const Real usx = ux[n] - fx[n] * inv;
  const Real usy = uy[n] - fy[n] * inv;
  // ... and measure what is left against the rigid field it should match.
  const Real dx = usx - (st.vx - st.omega * ry);
  const Real dy = usy - (st.vy + st.omega * rx);

  out.m  += double(cr);
  out.Sx += double(cr * rx);   out.Sy += double(cr * ry);
  out.Iz += double(cr * (rx * rx + ry * ry));
  out.Px += double(cr * dx);   out.Py += double(cr * dy);
  out.Lz += double(cr * (rx * dy - ry * dx));
  return true;
}

// Write F = chi 2 rho (U + omega x r - u*) with the NEW body state, and zero it
// everywhere else -- the body moves, and a force left behind where it used to be
// would keep pushing.
template <class Shape, class LiquidOf, class LbmOf>
LBM_HD LBM_INLINE void body_apply_node(const Shape& b, const BodyState& st,
                                       const Real* ux, const Real* uy,
                                       Real* fx, Real* fy, Real* fz,
                                       LiquidOf dens, LbmOf ldens,
                                       long n, Real reach2) {
  int x, y, z;
  coords(n, st.nx, st.ny, x, y, z);
  fz[n] = Real(0);
  const Real rx = Real(x) - st.cx, ry = Real(y) - st.cy;
  if (rx * rx + ry * ry > reach2) { fx[n] = Real(0); fy[n] = Real(0); return; }
  const Real c = b.chi(Real(x), Real(y));
  if (c < Real(1e-6)) { fx[n] = Real(0); fy[n] = Real(0); return; }

  const Real r  = dens(n);
  const Real rl = ldens(n);
  const Real inv = (rl > Real(1e-12)) ? Real(0.5) / rl : Real(0);
  const Real usx = ux[n] - fx[n] * inv;
  const Real usy = uy[n] - fy[n] * inv;
  fx[n] = c * Real(2) * r * ((st.vx - st.omega * ry) - usx);
  fy[n] = c * Real(2) * r * ((st.vy + st.omega * rx) - usy);
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
  Real bx = 0, by = 0;        // body force per unit mass -- the SAME vector the
                              // collision operator is given, or the buoyancy
                              // term is inconsistent with it
  bool free_translation = true;
  bool free_rotation = true;
};

inline void body_solve(const BodyProperties& p, const BodySums& q,
                       double& dux, double& duy, double& dw) {
  const double A = double(p.mass) + q.m;
  const double B = double(p.inertia) + q.Iz;
  dux = duy = dw = 0;
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
                                  const Real* __restrict__ fx,
                                  const Real* __restrict__ fy,
                                  LiquidOf dens, LbmOf ldens,
                                  long N, Real reach2, double* __restrict__ partial) {
  extern __shared__ double sm[];             // 7 * blockDim.x
  const unsigned T = blockDim.x;
  BodySums acc;
  const long stride = long(T) * gridDim.x;
  for (long n = long(blockIdx.x) * T + threadIdx.x; n < N; n += stride)
    body_probe_node(b, st, ux, uy, fx, fy, dens, ldens, n, reach2, acc);

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
  __syncthreads();
  for (unsigned s = T / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      for (int k = 0; k < 7; ++k) sm[k * T + threadIdx.x] += sm[k * T + threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int k = 0; k < 7; ++k) partial[k * gridDim.x + blockIdx.x] = sm[k * T];
}

template <class Shape, class LiquidOf, class LbmOf>
__global__ void body_apply_kernel(Shape b, BodyState st,
                                  const Real* __restrict__ ux,
                                  const Real* __restrict__ uy,
                                  Real* __restrict__ fx, Real* __restrict__ fy,
                                  Real* __restrict__ fz,
                                  LiquidOf dens, LbmOf ldens, long N, Real reach2) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  body_apply_node(b, st, ux, uy, fx, fy, fz, dens, ldens, n, reach2);
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
    const Real c = b.chi(Real(x), Real(y));
    const Real rx = Real(x) - b.cx, ry = Real(y) - b.cy;
    a += double(c);
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
    LBM_CUDA_CHECK(cudaMalloc(&partial_, sizeof(double) * 7 * GRID));
  }
  ~PenalisedBody() {
    cudaFree(fx_); cudaFree(fy_); cudaFree(fz_); cudaFree(partial_);
  }
  PenalisedBody(const PenalisedBody&) = delete;
  PenalisedBody& operator=(const PenalisedBody&) = delete;

  //---- state, public so a driver can prescribe, clamp or read any of it ------
  Shape shape;
  BodyProperties props;
  Real vx = 0, vy = 0;             // centre-of-mass velocity
  Real omega = 0;                  // angular velocity about z

  // Device pointers owned by the fluid solver.
  void couple_velocity(const Real* ux, const Real* uy) { ux_ = ux; uy_ = uy; }

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
    double dux = 0, duy = 0, dw = 0;
    body_solve(props, q, dux, duy, dw);
    if (props.free_translation) { vx += Real(dux); vy += Real(duy); }
    if (props.free_rotation)    { omega += Real(dw); }
    apply(dens, ldens);
    return body_reaction(props, q, dux, duy, dw);
  }

  // Advance the pose. Explicit Euler at dt = 1 -- the fluid step is the
  // timescale and there is nothing faster in the body to resolve.
  void advance() {
    shape.cx += vx;
    shape.cy += vy;
    shape.set_angle(shape.theta + omega);
  }

  template <class LiquidOf, class LbmOf>
  BodySums probe(LiquidOf dens, LbmOf ldens) {
    const Real r2 = shape.reach() * shape.reach();
    body_probe_kernel<<<GRID, BLOCK, sizeof(double) * 7 * BLOCK>>>(
        shape, state(), ux_, uy_, fx_, fy_, dens, ldens, N_, r2, partial_);
    LBM_CUDA_CHECK(cudaGetLastError());
    std::vector<double> h(std::size_t(7 * GRID));
    LBM_CUDA_CHECK(cudaMemcpy(h.data(), partial_, sizeof(double) * 7 * GRID,
                              cudaMemcpyDeviceToHost));
    double t[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 7; ++k)
      for (int i = 0; i < GRID; ++i) t[k] += h[std::size_t(k * GRID + i)];
    BodySums q;
    q.m = t[0]; q.Sx = t[1]; q.Sy = t[2]; q.Iz = t[3];
    q.Px = t[4]; q.Py = t[5]; q.Lz = t[6];
    return q;
  }

  template <class LiquidOf, class LbmOf>
  void apply(LiquidOf dens, LbmOf ldens) {
    const Real r2 = shape.reach() * shape.reach();
    const int B = 128;
    body_apply_kernel<<<int((N_ + B - 1) / B), B>>>(
        shape, state(), ux_, uy_, fx_, fy_, fz_, dens, ldens, N_, r2);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

 private:
  static constexpr int BLOCK = 256, GRID = 256;

  BodyState state() const {
    BodyState st;
    st.cx = shape.cx;  st.cy = shape.cy;
    st.vx = vx;  st.vy = vy;  st.omega = omega;
    st.nx = nx_; st.ny = ny_;
    return st;
  }

  int nx_, ny_, nz_;
  long N_;
  Real *fx_ = nullptr, *fy_ = nullptr, *fz_ = nullptr;
  double* partial_ = nullptr;
  const Real *ux_ = nullptr, *uy_ = nullptr;
};

#endif  // __CUDACC__

}  // namespace lbm
