//==============================================================================
// The lattice identities are already static_asserted in Lattices.hpp; this
// re-checks them at runtime in floating point (so a bad rational -> float
// conversion would show up) and verifies the Esoteric Pull pairing rule.
//==============================================================================
#include "Check.hpp"
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

using namespace lbm;

template <class L>
void run() {
  const std::string n = L::name;
  // opp() is an involution and really maps to -c
  bool inv = true, neg = true;
  for (int i = 0; i < L::Q; ++i) {
    inv = inv && (opp(opp(i)) == i);
    for (int a = 0; a < 3; ++a) neg = neg && (cvel<L>(opp(i), a) == -cvel<L>(i, a));
  }
  check::ok(inv, n + ": opp() is an involution");
  check::ok(neg, n + ": opp(i) is the reversed velocity");
  check::ok(opp(0) == 0, n + ": rest velocity is self-opposite");

  // Esoteric Pull needs the pairs adjacent so the kernel can walk i += 2
  bool paired = true;
  for (int i = 1; i < L::Q; i += 2) paired = paired && (opp(i) == i + 1);
  check::ok(paired, n + ": opposite directions are adjacent pairs (EsoPull contract)");

  const Real tol = sizeof(Real) == 4 ? Real(1e-6) : Real(1e-14);
  Real sw = 0;
  for (int i = 0; i < L::Q; ++i) sw += weight<L, Real>(i);
  check::near(sw, Real(1), tol, n + ": sum w_i == 1");

  for (int a = 0; a < L::D; ++a)
    for (int b = 0; b < L::D; ++b) {
      Real m = 0;
      for (int i = 0; i < L::Q; ++i)
        m += weight<L, Real>(i) * Real(cvel<L>(i, a)) * Real(cvel<L>(i, b));
      check::near(m, (a == b) ? cs2<L, Real>() : Real(0), tol,
                  n + ": <c" + char('x' + a) + " c" + char('x' + b) + "> ");
    }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  { run<D2Q5>(); run<D2Q9>(); run<D3Q7>(); run<D3Q19>(); run<D3Q27>(); }
  const int r = check::report("lattice");
  Kokkos::finalize();
  return r;
}
