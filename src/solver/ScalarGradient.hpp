#pragma once
//==============================================================================
//  Gradient of an arbitrary node field, on the isotropic lattice stencil.
//
//  The same stencil PhaseFieldSolver uses for grad phi,
//
//      grad f (x) = (1/cs2) sum_i w_i c_i f(x + c_i),
//
//  but applied to a field this class does not own. It exists for the pressure:
//  the constant-reference-density formulation needs grad p~, and p~ is the
//  fluid solver's zeroth moment rather than anything the phase field carries.
//
//  Isotropic rather than axis-only for the reason the phase gradient is: this
//  gradient multiplies a density difference and is differenced against the LBE's
//  own pressure term, and an anisotropic stencil would leave a direction-
//  dependent residual exactly where the two are meant to cancel.
//
//  Reads the field, writes only its own three arrays, so it races with nothing.
//  Call it AFTER the field it differentiates has been filled -- for the pressure
//  that means after FluidSolver::compute_macroscopic().
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"
#include "solver/PhaseFieldSolver.hpp"      // GradientLatticeOf

namespace lbm {

template <class L, class GL = typename GradientLatticeOf<L>::type>
class ScalarGradient {
 public:
  explicit ScalarGradient(const Domain& dom)
      : dom_(dom),
        gx_("grad_x", dom.n_padded),
        gy_("grad_y", dom.n_padded),
        gz_("grad_z", dom.n_padded) {}

  View1D<Real> x() const { return gx_; }
  View1D<Real> y() const { return gy_; }
  View1D<Real> z() const { return gz_; }

  void refresh(View1D<Real> f) {
    const Domain d = dom_;
    auto gx = gx_, gy = gy_, gz = gz_;
    constexpr Real icsg = inv_cs2<GL, Real>();
    Kokkos::parallel_for("scalar_gradient", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        Neighbours<GL> nb;
        d.template fill_neighbours<GL, 1, 1>(n, nb);
        Real g[3] = {Real(0), Real(0), Real(0)};
        for (int i = 1; i < GL::Q; ++i) {
          const Real wf = weight<GL, Real>(i) * f(nb.j[i]);
          for (int a = 0; a < GL::D; ++a) g[a] += wf * Real(cvel<GL>(i, a));
        }
        gx(n) = icsg * g[0];  gy(n) = icsg * g[1];  gz(n) = icsg * g[2];
      });
    Kokkos::fence();
  }

 private:
  Domain dom_;
  View1D<Real> gx_, gy_, gz_;
};

}  // namespace lbm
