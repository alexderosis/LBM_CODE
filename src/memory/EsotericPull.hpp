#pragma once
//==============================================================================
//  Esoteric Pull (Lehmann 2022) -- in-place streaming, ONE population array.
//
//  DERIVATION (the code below is just this table transcribed).
//
//  Work per opposite pair (i, i+1), i odd, and write A[node][slot] for storage.
//  The partner node is ALWAYS n + c_i, at both parities; only which of the two
//  slots lives remotely flips. Writing f^in for what is read and f^out for what
//  is written:
//
//    even step t, node n            odd step t, node n
//    ------------------------       ------------------------
//    f_i   <- A[n      ][i  ]       f_i   <- A[n      ][i+1]
//    f_i+1 <- A[n+c_i  ][i+1]       f_i+1 <- A[n+c_i  ][i  ]
//    A[n+c_i][i+1] <- f_i^out       A[n+c_i][i  ] <- f_i^out
//    A[n    ][i  ] <- f_i+1^out     A[n    ][i+1] <- f_i+1^out
//
//  Each parity reads exactly two slots and writes those same two, crossed. Every
//  slot has exactly one reader and one writer, and they are the same node, so
//  the scheme is race-free without any temporary buffer.
//
//  Consistency check: at an even step node n puts its outgoing f_i into
//  A[n+c_i][i+1]; at the next (odd) step node m = n+c_i reads f_i from A[m][i+1].
//  Which is exactly the odd-step rule. The scheme closes.
//
//  IMPLICIT BOUNCE-BACK. On a solid cell, halfway bounce-back sets
//  f_i^out = f_i+1^in and f_i+1^out = f_i^in. Substituting into the table, every
//  write lands back in the slot it was read from -- the whole update is the
//  identity. Solid cells therefore do NOTHING: no load, no store, no branch in
//  the arithmetic. The population a fluid node emits toward a wall sits
//  untouched for one step and is read back as the opposite direction on the
//  next, which puts the wall exactly midway between the two cells.
//
//  ORDERING. All of this assumes opp(i) = i+1 for odd i, which is the contract
//  static_asserted in Lattices.hpp.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

template <class L>
class EsotericPull {
 public:
  using Lattice = L;
  static constexpr int Q = L::Q;
  static constexpr const char* name = "EsotericPull";

  // Bounce-back is the identity on the storage; solid cells are skipped.
  static constexpr bool implicit_bounce_back = true;
  // Only the odd-indexed directions are ever used as neighbour offsets, which
  // halves the neighbour list and its register footprint.
  static constexpr int nb_first = 1, nb_stride = 2;

  EsotericPull() = default;
  explicit EsotericPull(const Domain& d) : f_("f", d.n_padded, Q) {}

  template <int Parity>
  struct Access {
    static_assert(Parity == 0 || Parity == 1, "parity must be a compile-time 0 or 1");
    View2D<Real> f;

    KOKKOS_INLINE_FUNCTION Real load(const Neighbours<L>& nb, int i) const {
      if (i == 0) return f(nb.n, 0);
      if constexpr (Parity == 0) {
        return (i & 1) ? f(nb.n, i) : f(nb.j[i - 1], i);
      } else {
        return (i & 1) ? f(nb.n, i + 1) : f(nb.j[i - 1], i - 1);
      }
    }

    KOKKOS_INLINE_FUNCTION void store(const Neighbours<L>& nb, int i, Real v) const {
      if (i == 0) { f(nb.n, 0) = v; return; }
      if constexpr (Parity == 0) {
        if (i & 1) f(nb.j[i], i + 1) = v; else f(nb.n, i - 1) = v;
      } else {
        if (i & 1) f(nb.j[i], i) = v;     else f(nb.n, i) = v;
      }
    }

    //--------------------------------------------------------------------------
    // Pair form used by the hot kernel -- this IS the table at the top of the
    // file, one line per row, with no runtime test on the parity or on i&1.
    //--------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    void load_pair(const Neighbours<L>& nb, int i, Real& a, Real& b) const {
      if constexpr (Parity == 0) { a = f(nb.n, i);     b = f(nb.j[i], i + 1); }
      else                       { a = f(nb.n, i + 1); b = f(nb.j[i], i);     }
    }
    KOKKOS_INLINE_FUNCTION
    void store_pair(const Neighbours<L>& nb, int i, Real a, Real b) const {
      if constexpr (Parity == 0) { f(nb.j[i], i + 1) = a; f(nb.n, i)     = b; }
      else                       { f(nb.j[i], i)     = a; f(nb.n, i + 1) = b; }
    }
    KOKKOS_INLINE_FUNCTION Real load_rest(const Neighbours<L>& nb) const { return f(nb.n, 0); }
    KOKKOS_INLINE_FUNCTION void store_rest(const Neighbours<L>& nb, Real v) const { f(nb.n, 0) = v; }

    // Exact inverse of load(): write the value that load() would return, into
    // the very slot load() reads. Every slot has exactly one reader, so applying
    // this over all nodes writes each slot exactly once -- a complete
    // initialisation, and the readout in the streaming test is just load().
    KOKKOS_INLINE_FUNCTION void scatter(const Neighbours<L>& nb, int i, Real v) const {
      if (i == 0) { f(nb.n, 0) = v; return; }
      if constexpr (Parity == 0) {
        if (i & 1) f(nb.n, i) = v;     else f(nb.j[i - 1], i) = v;
      } else {
        if (i & 1) f(nb.n, i + 1) = v; else f(nb.j[i - 1], i - 1) = v;
      }
    }
  };

  template <int Parity>
  Access<Parity> access() const { return Access<Parity>{f_}; }

  void end_of_step() {}   // nothing to swap: the array is updated in place

  static constexpr std::size_t bytes_per_node() { return Q * sizeof(Real); }

 private:
  View2D<Real> f_;
};

}  // namespace lbm
