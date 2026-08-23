#pragma once
//==============================================================================
//  Two-lattice (A/B) streaming -- the REFERENCE implementation.
//
//  Two population arrays, 2x the memory, obviously race-free, obviously correct.
//  Not the production scheme, but it stays in the tree permanently: when an
//  exotic collision operator produces garbage, being able to swap to a streaming
//  scheme that cannot possibly be wrong is what makes the bug bisectable.
//  test_streaming cross-checks Esoteric Pull against it step by step.
//
//  Semantics are PULL: gather the populations that have arrived at node n,
//  collide, write locally. Streaming and collision are one fused kernel, which
//  is what the in-place schemes require anyway.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

template <class L>
class TwoLattice {
 public:
  using Lattice = L;
  static constexpr int Q = L::Q;
  static constexpr const char* name = "TwoLattice";

  // Solid cells must actively reflect; there is no in-place trick here.
  static constexpr bool implicit_bounce_back = false;
  // Every direction is pulled from a distinct neighbour, so the full list.
  static constexpr int nb_first = 1, nb_stride = 1;

  TwoLattice() = default;
  explicit TwoLattice(const Domain& d)
      : a_("f_a", d.n_padded, Q), b_("f_b", d.n_padded, Q) {}

  //----------------------------------------------------------------------------
  // Captured BY VALUE into kernels: two View handles, nothing else.
  // `Parity` is unused here but keeps one interface across streaming schemes.
  //----------------------------------------------------------------------------
  template <int Parity>
  struct Access {
    View2D<Real> src, dst;

    // Population i arriving at n departed from n - c_i = n + c_opp(i).
    KOKKOS_INLINE_FUNCTION Real load(const Neighbours<L>& nb, int i) const {
      return src(i == 0 ? nb.n : nb.j[opp(i)], i);
    }
    KOKKOS_INLINE_FUNCTION void store(const Neighbours<L>& nb, int i, Real v) const {
      dst(nb.n, i) = v;
    }

    // Pair form used by the hot kernel: i is odd and (i, i+1) are opposites.
    // Same work as two load()/store() calls, but with no runtime parity test.
    KOKKOS_INLINE_FUNCTION
    void load_pair(const Neighbours<L>& nb, int i, Real& a, Real& b) const {
      a = src(nb.j[i + 1], i);
      b = src(nb.j[i], i + 1);
    }
    KOKKOS_INLINE_FUNCTION
    void store_pair(const Neighbours<L>& nb, int i, Real a, Real b) const {
      dst(nb.n, i) = a;
      dst(nb.n, i + 1) = b;
    }
    KOKKOS_INLINE_FUNCTION Real load_rest(const Neighbours<L>& nb) const { return src(nb.n, 0); }
    KOKKOS_INLINE_FUNCTION void store_rest(const Neighbours<L>& nb, Real v) const { dst(nb.n, 0) = v; }
    // Exact inverse of load(): write the value that load() would return.
    // Applied over every node it writes each slot exactly once, so it is a
    // complete initialisation.
    KOKKOS_INLINE_FUNCTION void scatter(const Neighbours<L>& nb, int i, Real v) const {
      src(i == 0 ? nb.n : nb.j[opp(i)], i) = v;
    }
  };

  template <int Parity>
  Access<Parity> access() const { return Access<Parity>{a_, b_}; }

  void end_of_step() { auto t = a_; a_ = b_; b_ = t; }

  static constexpr std::size_t bytes_per_node() { return 2 * Q * sizeof(Real); }

 private:
  View2D<Real> a_, b_;
};

}  // namespace lbm
