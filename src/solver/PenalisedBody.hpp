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
//  rebuilt, the body's position is continuous, and there are no fresh nodes.
//
//  THE METHOD. A smooth solid indicator chi in [0, 1] and a force that drives
//  the fluid there toward the body's velocity,
//
//      F(x) = chi(x) * 2 rho(x) [ U_body - u*(x) ],
//
//  which is direct forcing: applied through the same half-force machinery the
//  rest of the code uses, one step of it takes u to U_body exactly where
//  chi = 1. It must run AFTER compute_macroscopic() and BEFORE the fluid steps.
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
//  planes and its effective shape would change as it fell. The smoothing is what
//  lets the square occupy a continuously varying position.
//
//  THE REACTION IS FREE, and that is the point of doing it this way: the force
//  the body exerts on the fluid is known at every node, so the force the fluid
//  exerts on the body is minus its sum. That closes Newton's equation and lets
//  the square FALL under gravity and buoyancy rather than being pushed at a
//  prescribed speed -- which is what the reference does, and what a "falling"
//  body actually means.
//
//  WHAT THIS DOES NOT MODEL.
//
//   * NO CONTACT LINE. The phase field is advected by the fluid velocity and
//     knows nothing about the body, so nothing sets the angle at which the free
//     surface meets the solid. Inside the body u is driven to U_body, so the
//     interface is carried with it rather than through it, and the conservative
//     Allen-Cahn form keeps the profile from simply diffusing in -- but the
//     wetting physics is absent, and water entry is a contact-line problem.
//     Read the splash, not the meniscus.
//   * NO SUB-CELL SURFACE. Penalisation resolves the body to the smoothing
//     width, so a pressure read at the face of the square is a smoothed pressure
//     over a cell or two. Integrated force is far more trustworthy than local
//     pressure, which is the usual situation with this family of methods.
//   * NO ROTATION. Translation only. A square landing off-axis would rotate and
//     this will not let it.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// An axis-aligned rectangle, described by its centre and half-extents. Kept a
// plain struct so it captures into a device lambda by value.
//------------------------------------------------------------------------------
struct Rect {
  Real cx = 0, cy = 0;         // centre
  Real hx = 0, hy = 0;         // half width, half height
  Real smooth = Real(1.5);     // indicator smoothing width, in cells

  // chi in [0,1]: 1 well inside, 0 well outside, tanh across the faces. The
  // product of the two axis indicators, which for a rectangle is exact away
  // from the corners and rounds them slightly -- harmless, and it keeps the
  // function separable and cheap.
  KOKKOS_INLINE_FUNCTION Real chi(Real x, Real y) const {
    const Real ax = (hx - Kokkos::fabs(x - cx)) / smooth;
    const Real ay = (hy - Kokkos::fabs(y - cy)) / smooth;
    const Real sx = Real(0.5) * (Real(1) + Kokkos::tanh(ax));
    const Real sy = Real(0.5) * (Real(1) + Kokkos::tanh(ay));
    return sx * sy;
  }
};

template <class L>
class PenalisedBody {
 public:
  explicit PenalisedBody(const Domain& dom)
      : dom_(dom),
        fx_("body_fx", dom.n_padded),
        fy_("body_fy", dom.n_padded),
        fz_("body_fz", dom.n_padded) {}

  void set_velocity(View1D<Real> ux, View1D<Real> uy) { ux_ = ux; uy_ = uy; }

  View1D<Real> x() const { return fx_; }
  View1D<Real> y() const { return fy_; }
  View1D<Real> z() const { return fz_; }

  // Body state, owned by the caller so the driver can integrate it however it
  // likes -- free fall, prescribed motion, or held fixed.
  Rect shape;
  Real vx = 0, vy = 0;

  //----------------------------------------------------------------------------
  // Write the penalisation force and return the reaction on the body.
  //
  // `density_of` maps a node to its local fluid density, so the force scales
  // with the fluid actually there: the same body meets a thousandfold heavier
  // medium when it reaches the water, and a penalisation that ignored that would
  // decelerate it as though it were still in air.
  //----------------------------------------------------------------------------
  // What one refresh reports back: the hydrodynamic reaction, and the mass of
  // the fluid the penalised region is standing in.
  struct Reaction { Real fx = 0, fy = 0, fluid_mass = 0; };

  template <class DensityOf>
  Reaction refresh(DensityOf density_of) {
    const Domain d = dom_;
    const Rect b = shape;
    const Real bvx = vx, bvy = vy;
    auto ux = ux_, uy = uy_;
    auto fx = fx_, fy = fy_, fz = fz_;
    const auto dens = density_of;
    const Index hx = dom_.hx, hy = dom_.hy;

    Real sx = 0, sy = 0, sm = 0;
    Kokkos::parallel_reduce("penalised_body", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& ax, Real& ay, Real& am) {
        Index px, py, pz; d.coords(n, px, py, pz);
        fz(n) = Real(0);
        const Real X = Real(px - hx), Y = Real(py - hy);
        const Real c = b.chi(X, Y);
        if (c < Real(1e-6)) { fx(n) = Real(0); fy(n) = Real(0); return; }
        const Real r = dens(n);
        // Undo this body's own previous contribution to the stored velocity.
        const Real inv = Real(0.5) / r;
        const Real usx = ux(n) - fx(n) * inv;
        const Real usy = uy(n) - fy(n) * inv;
        const Real gx = c * Real(2) * r * (bvx - usx);
        const Real gy = c * Real(2) * r * (bvy - usy);
        fx(n) = gx;  fy(n) = gy;
        ax -= gx;  ay -= gy;              // reaction: minus what the body applied
        am += c * r;                      // fluid standing in the penalised region
      }, sx, sy, sm);
    Kokkos::fence();
    return Reaction{sx, sy, sm};
  }

  //----------------------------------------------------------------------------
  // NEWTON, WITH THE FICTITIOUS FLUID TAKEN OUT.
  //
  // The penalised region is not empty: it is full of fluid that the forcing
  // drags along, and that fluid's inertia and weight are both already inside the
  // reaction. Using the body's own mass alone would count them twice. Uhlmann's
  // correction subtracts them, which with a density ratio has to be done with
  // the LOCAL fluid mass rather than a constant, because the same square is
  // standing in air one moment and in water the next:
  //
  //     (m_body - m_fluid) dV/dt = R + (m_body - m_fluid) g.
  //
  // This requires the body to be DENSER than the fluid it displaces. Lighter
  // than water, the bracket changes sign and the integration blows up; a
  // floating body needs a formulation this one is not.
  //----------------------------------------------------------------------------
  // Area, as the integral of chi rather than 4 hx hy, so it matches whatever the
  // force actually acted on -- the smoothing makes the penalised body slightly
  // larger than the nominal rectangle and the nominal area would bias it.
  Real penalised_area() const {
    const Domain d = dom_;
    const Rect b = shape;
    const Index hx = dom_.hx, hy = dom_.hy;
    Real a = 0;
    Kokkos::parallel_reduce("penalised_area", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc) {
        Index px, py, pz; d.coords(n, px, py, pz);
        acc += b.chi(Real(px - hx), Real(py - hy));
      }, a);
    return a;
  }

 private:
  Domain dom_;
  View1D<Real> fx_, fy_, fz_;
  View1D<Real> ux_, uy_;
};

}  // namespace lbm
