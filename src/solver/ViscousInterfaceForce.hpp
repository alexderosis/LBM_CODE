#pragma once
//==============================================================================
//  F_nu = nu ( grad u + grad u^T ) . grad rho     -- De Rosis & Enan Eq. (22).
//
//  The third of the four forces in the potential-form multiphase operator, and
//  the only one that cannot be assembled from node-local data: it needs the
//  velocity GRADIENT, so it needs a pass of its own.
//
//  WHAT IT IS FOR. The LBE recovers mu lap(u), while the momentum equation at
//  variable viscosity wants div( mu (grad u + grad u^T) ). The difference is
//  this term. It is identically zero wherever grad rho is -- that is, everywhere
//  except the interface -- so at a matched density the whole object can be left
//  unbound and MultiphasePotentialBGK will read zero for it.
//
//  ONLY THE CONTRACTION IS STORED. The full gradient tensor is nine numbers per
//  node; contracted against grad rho it is three. Since grad rho is available in
//  the same kernel (it is grad phi times a constant, Eq. 23), the contraction is
//  done where the tensor is still in registers and only the resulting vector is
//  written. Nine fields per node become three.
//
//  COUPLING ORDER. This reads the velocity FIELD, which FluidSolver fills from
//  the populations. Those are the populations at time t, so calling
//
//      pf.refresh();  fl.compute_macroscopic();  vf.refresh(coll);  fl.step();
//
//  gives a force built from u(t) and phi(t) and consumed by the collision at
//  time t, with no lag anywhere. The price is one extra pass over the
//  populations per step, because the fused `step(true)` writes u only as it
//  goes -- too late for a neighbour gather in the same step. That is a real
//  cost, and it is the reason this pass is opt-in rather than automatic.
//
//  NOT VALID AGAINST A WALL. The stencil reads u at every neighbour, including
//  non-fluid ones, where the velocity field is whatever FluidSolver left there
//  (zero). At an interface touching a wall that is a one-sided derivative taken
//  as if it were central. Every case that currently runs this operator is
//  periodic; see the wetting note in PhaseFieldSolver.hpp for the related gap.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"
#include "solver/PhaseFieldSolver.hpp"

namespace lbm {

template <class L, class GL = typename GradientLatticeOf<L>::type>
class ViscousInterfaceForce {
 public:
  explicit ViscousInterfaceForce(const Domain& dom)
      : dom_(dom),
        fx_("fnu_x", dom.n_padded),
        fy_("fnu_y", dom.n_padded),
        fz_("fnu_z", dom.n_padded) {}

  void set_velocity(View1D<Real> ux, View1D<Real> uy, View1D<Real> uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }
  void set_phase_gradient(View1D<Real> gx, View1D<Real> gy, View1D<Real> gz) {
    gx_ = gx; gy_ = gy; gz_ = gz;
  }

  View1D<Real> x() const { return fx_; }
  View1D<Real> y() const { return fy_; }
  View1D<Real> z() const { return fz_; }

  //----------------------------------------------------------------------------
  // `coll` supplies nu(n) and drho/dphi. It is taken by value into the kernel,
  // which is what every other policy in this code base does: it is a handful of
  // View handles and scalars.
  //----------------------------------------------------------------------------
  template <class Collision>
  void refresh(const Collision& coll) {
    const Domain d = dom_;
    const Collision c = coll;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto gx = gx_, gy = gy_, gz = gz_;
    auto fx = fx_, fy = fy_, fz = fz_;
    constexpr Real icsg = inv_cs2<GL, Real>();

    Kokkos::parallel_for("viscous_interface_force", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        Neighbours<GL> nb;
        d.template fill_neighbours<GL, 1, 1>(n, nb);

        // du[a][b] = d u_b / d x_a, by the same isotropic stencil as grad phi.
        Real du[3][3] = {{Real(0), Real(0), Real(0)},
                         {Real(0), Real(0), Real(0)},
                         {Real(0), Real(0), Real(0)}};
        for (int i = 1; i < GL::Q; ++i) {
          const Index j = nb.j[i];
          const Real w = weight<GL, Real>(i);
          const Real U[3] = {ux(j), uy(j), (GL::D == 3) ? uz(j) : Real(0)};
          for (int a = 0; a < GL::D; ++a) {
            const Real wc = w * Real(cvel<GL>(i, a));
            for (int b = 0; b < GL::D; ++b) du[a][b] += wc * U[b];
          }
        }

        // One icsg for the whole contraction: du[a][b] above is the RAW sum
        // sum_i w_i c_ia u_b, and both it and its transpose need the same
        // 1/cs2, so the normalisation factors out of the bracket.
        const Real k = c.drho_dphi() * icsg;
        const Real gr[3] = {gx(n), gy(n), (GL::D == 3) ? gz(n) : Real(0)};
        const Real nu = c.viscosity_at(n);
        Real out[3] = {Real(0), Real(0), Real(0)};
        for (int a = 0; a < GL::D; ++a) {
          Real acc = Real(0);
          for (int b = 0; b < GL::D; ++b) acc += (du[a][b] + du[b][a]) * gr[b];
          out[a] = nu * k * acc;
        }
        fx(n) = out[0];  fy(n) = out[1];  fz(n) = out[2];
      });
    Kokkos::fence();
  }

 private:
  Domain dom_;
  View1D<Real> fx_, fy_, fz_;
  View1D<Real> ux_, uy_, uz_;
  View1D<Real> gx_, gy_, gz_;
};

}  // namespace lbm
