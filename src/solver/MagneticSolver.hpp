#pragma once
//==============================================================================
//  Magnetic field solver: a VECTOR-valued distribution set.
//
//  Where the thermal module added one extra distribution, this adds D of them --
//  one per component of B, on their own lattice, streamed independently but
//  collided together, because the induction equilibrium mixes components through
//  u_beta B_alpha - B_beta u_alpha.
//
//  Each component gets its own Streaming instance, so Esoteric Pull, the halo
//  logic and the neighbour machinery are reused verbatim; only the collision is
//  new. The fluid runs on D2Q9/D3Q19/D3Q27 while this runs on D2Q5/D3Q7 at the
//  same time, on the same Domain.
//
//  BOUNDARY CONDITIONS. Two kinds, both moment based:
//    * Dirichlet -- the node takes an imposed external field (Dellar, 13a-13b);
//    * outflow   -- the node takes B from its upstream neighbour, zero gradient.
//  Conducting and insulating magnetic walls are a separate piece of work and are
//  not faked here.
//==============================================================================
#include "collision/MagneticBGK.hpp"
#include "boundary/MomentDirichlet.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbm {

// Per-node magnetic boundary kind.
//
// MagOutXp IS NOT VALIDATED and has a measured defect. On the inlet-driven
// Hartmann flow of validation/hartmann_inlet.cpp -- whose exact solution is
// independent of x, so any streamwise variation is boundary error -- it drives
// B about 6% high over the last ~10 nodes, and the overshoot decays upstream
// only slowly. Moving the outlet from Lx = 121 to Lx = 241 cut the sampled
// error by 3.4x. The condition is kept because with the outlet far enough from
// anything being measured the interior converges cleanly at second order, but
// it should be fixed before being relied on. The cause is not identified; the
// natural suspects are the interaction with div(B) = 0 (see the known
// limitation on divergence preservation) and the fact that a pure Neumann
// condition does not constrain the field level at all.
enum MagWallCode : std::uint8_t { MagNone = 0, MagDirichlet = 1, MagOutXp = 2 };

template <class L, class Streaming, class Collision>
class MagneticSolver {
 public:
  using Lattice = L;
  static constexpr int Q  = L::Q;
  static constexpr int NC = L::D;                 // components of B
  static constexpr int NF = Streaming::nb_first;
  static constexpr int NS = Streaming::nb_stride;

  MagneticSolver(const Domain& dom, const Collision& coll)
      : dom_(dom), coll_(coll) {
    for (int a = 0; a < NC; ++a) pop_[a] = Streaming(dom);
    B_[0] = View1D<Real>("Bx", dom.n_padded);
    B_[1] = View1D<Real>("By", dom.n_padded);
    B_[2] = View1D<Real>("Bz", dom.n_padded);
    wall_ = View1D<std::uint8_t>("mwall", dom.n_padded);
    unk_  = View1D<std::uint32_t>("munk", dom.n_padded);
    // One entry per DISTINCT imposed field, not per node -- but a profiled
    // magnetic inlet makes nearly every node distinct, so this must not be a
    // uint8_t. At uint8_t the index wrapped silently past 256 states and nodes
    // beyond that took another node's field. Same defect, same fix, as the
    // fluid solver's wall tag.
    tag_  = View1D<std::uint16_t>("mtag", dom.n_padded);
    wallB_ = View2D<Real>("mwallB", 1, 3);
  }

  //----------------------------------------------------------------------------
  // Moment-based Dirichlet walls on B (Dellar, Eqs. 13a-13b).
  //
  // fn(x, y, z) -> WallB. `is_wall == false` leaves the node alone. The imposed
  // field is the EXTERNAL applied value: Maxwell's equations make both the
  // normal and the tangential components of B continuous at a wall, so B simply
  // takes its outside value there -- unlike u, which has no such freedom.
  //----------------------------------------------------------------------------
  struct WallB {
    bool is_wall = false;
    Real Bx = 0, By = 0, Bz = 0;
    // Zero-gradient outflow on the +x face. The node takes B from its upstream
    // neighbour each step instead of an imposed value, so Bx/By/Bz are ignored.
    // Set is_wall as well -- outflow is a kind of boundary node, not a separate
    // flag orthogonal to it.
    bool outflow = false;
  };

  template <class Fn>
  void set_moment_walls(Fn fn) {
    auto h_wall = Kokkos::create_mirror_view(wall_);
    auto h_unk  = Kokkos::create_mirror_view(unk_);
    auto h_tag  = Kokkos::create_mirror_view(tag_);
    for (Index n = 0; n < dom_.n_padded; ++n) { h_wall(n) = 0; h_unk(n) = 0; h_tag(n) = 0; }

    std::vector<std::array<Real, 3>> table;
    auto in_fluid = [&](Index x, Index y, Index z) { (void)x; (void)y; (void)z; return true; };
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x) {
          const WallB w = fn(x, y, z);
          if (!w.is_wall) continue;
          const Index n = dom_.id(x, y, z);
          std::array<Real, 3> v{w.Bx, w.By, w.Bz};
          std::size_t k = 0;
          for (; k < table.size(); ++k)
            if (table[k][0] == v[0] && table[k][1] == v[1] && table[k][2] == v[2]) break;
          if (k == table.size()) table.push_back(v);
          h_wall(n) = w.outflow ? std::uint8_t(MagOutXp) : std::uint8_t(MagDirichlet);
          h_tag(n)  = static_cast<std::uint16_t>(k);
          h_unk(n)  = unknown_mask<L>(dom_, x, y, z, in_fluid);
          has_walls_ = true;
        }
    if (table.empty()) table.push_back({Real(0), Real(0), Real(0)});
    if (table.size() > std::size_t(std::numeric_limits<std::uint16_t>::max()))
      throw std::runtime_error("set_moment_walls: " + std::to_string(table.size()) +
                               " distinct wall fields exceeds the tag width");
    n_wall_states_ = table.size();
    wallB_ = View2D<Real>("mwallB", table.size(), 3);
    auto h_B = Kokkos::create_mirror_view(wallB_);
    for (std::size_t k = 0; k < table.size(); ++k)
      for (int a = 0; a < 3; ++a) h_B(k, a) = table[k][a];
    Kokkos::deep_copy(wallB_, h_B);
    Kokkos::deep_copy(wall_, h_wall);
    Kokkos::deep_copy(unk_, h_unk);
    Kokkos::deep_copy(tag_, h_tag);
  }

  void set_velocity(View1D<Real> ux, View1D<Real> uy, View1D<Real> uz) {
    u_[0] = ux; u_[1] = uy; u_[2] = uz;
  }

  // fn(n) -> Kokkos::Array<Real,3>, device-callable: the initial B at node n.
  template <class Fn>
  void initialize_field(Fn fn) {
    t_ = 0;
    const Domain d = dom_;
    for (int a = 0; a < NC; ++a) {
      const auto acc = pop_[a].template access<0>();
      auto Ba = B_[a];
      Kokkos::parallel_for("minit", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        const auto b = fn(n);
        const Real Bv[3] = {b[0], b[1], b[2]};
        const Real z[3]  = {Real(0), Real(0), Real(0)};
        Ba(n) = Bv[a];
        for (int i = 0; i < Q; ++i) acc.scatter(nb, i, Collision::eq(i, a, Bv, z));
      });
    }
    Kokkos::fence();
  }

  // `field_is_current` skips the internal field pass when the caller has already
  // refreshed B for this step -- which a coupled driver must do, so that the
  // fluid collides against B(t) rather than B(t-1). See the note in run_step.
  void step(bool field_is_current = false) {
    if (t_ % 2 == 0) run_step<0>(field_is_current);
    else             run_step<1>(field_is_current);
    for (int a = 0; a < NC; ++a) pop_[a].end_of_step();
    ++t_;
  }

  void compute_field() {
    if (t_ % 2 == 0) field_kernel<0>(); else field_kernel<1>();
  }

  View1D<Real> Bx() const { return B_[0]; }
  View1D<Real> By() const { return B_[1]; }
  View1D<Real> Bz() const { return B_[2]; }
  const Domain& domain() const { return dom_; }
  std::size_t n_wall_states() const { return n_wall_states_; }
  std::size_t timestep() const { return t_; }

  // Magnetic energy sum |B|^2 / 2 over the interior.
  Real magnetic_energy() const {
    const Domain d = dom_;
    auto bx = B_[0], by = B_[1], bz = B_[2];
    Real e = 0;
    Kokkos::parallel_reduce("me", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& s) {
        Index px, py, pz; d.coords(n, px, py, pz);
        if (!d.is_interior(px, py, pz)) return;
        s += Real(0.5) * (bx(n) * bx(n) + by(n) * by(n) + bz(n) * bz(n));
      }, e);
    return e;
  }

 private:
  template <int P>
  void run_step(bool field_is_current) {
    const auto coll = coll_;
    const Domain d = dom_;
    auto bx = B_[0], by = B_[1], bz = B_[2];
    auto ux = u_[0], uy = u_[1], uz = u_[2];
    auto wall = wall_; auto unk = unk_; auto tag = tag_; auto wallB = wallB_;
    const bool have_u = ux.data() != nullptr;

    // All components must be collided from the SAME pre-collision B and u, so the
    // field is refreshed first and only then are the components updated.
    //
    // TIME LAG. A coupled driver must refresh B BEFORE stepping the fluid,
    // otherwise the fluid collides against B(t-1) and the splitting error is
    // first order in time. That error does not vanish under refinement: with
    // diffusive scaling the ratio (omega^2 dt) / (nu k^2) is independent of N,
    // so it shows up as a viscosity/damping offset that survives every
    // refinement. Measured on a shear Alfven wave it cost 3% of the damping
    // rate and did not converge; refreshing first removes it.
    if (!field_is_current) field_kernel<P>();

    typename Streaming::template Access<P> acc[NC];
    for (int a = 0; a < NC; ++a) acc[a] = pop_[a].template access<P>();

    Kokkos::parallel_for("magnetic_stream_collide", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        if (!d.is_interior(px, py, pz)) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        const Real B[3] = {bx(n), by(n), bz(n)};
        const Real u[3] = {have_u ? ux(n) : Real(0),
                           have_u ? uy(n) : Real(0),
                           have_u ? uz(n) : Real(0)};
        const std::uint8_t code = wall(n);
        const int  tg       = int(tag(n));
        for (int a = 0; a < NC; ++a) {
          Real g[Q];
          g[0] = acc[a].load_rest(nb);
          for (int i = 1; i < Q; i += 2) acc[a].load_pair(nb, i, g[i], g[i + 1]);
          // Eqs. (13a)-(13b): choose the inward-pointing populations so the
          // zeroth moment is the target field, then collide as usual. For an
          // outflow node the target is B[a] -- field_kernel has already put the
          // upstream neighbour's value there, so the two paths differ only in
          // where the number comes from.
          if (code == MagDirichlet)   impose_moment<L>(g, wallB(tg, a), unk(n));
          else if (code == MagOutXp)  impose_moment<L>(g, B[a], unk(n));
          coll.collide(g, a, B, u);
          acc[a].store_rest(nb, g[0]);
          for (int i = 1; i < Q; i += 2) acc[a].store_pair(nb, i, g[i], g[i + 1]);
        }
      });
  }

  template <int P>
  void field_kernel() {
    const Domain d = dom_;
    auto wall = wall_; auto unk = unk_; auto tag = tag_; auto wallB = wallB_;
    for (int a = 0; a < NC; ++a) {
      const auto acc = pop_[a].template access<P>();
      auto Ba = B_[a];
      Kokkos::parallel_for("magnetic_field", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        if (!d.is_interior(px, py, pz)) { Ba(n) = Real(0); return; }
        // At a moment wall the streamed populations still hold the unfixed
        // inward direction, so their sum is NOT the field. The field there is
        // the imposed one by construction. Reading the raw sum instead feeds a
        // wrong B into the magnetic equilibrium and the run diverges.
        const std::uint8_t code = wall(n);
        if (code == MagDirichlet) { Ba(n) = wallB(int(tag(n)), a); return; }
        // Zero-gradient outflow: the node's own populations are missing the
        // inflow directions, so its raw sum is not the field. Take the upstream
        // neighbour's instead. Reading a neighbour is safe here -- this kernel
        // only reads populations and writes one Ba per thread -- whereas
        // reading Ba(upstream) would race against whoever writes it.
        Index src = n;
        if (code == MagOutXp) {
          Index qx, qy, qz; d.coords(n, qx, qy, qz);
          src = d.id(qx - d.hx - 1, qy - d.hy, qz - d.hz);
        }
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(src, nb);
        Real g[Q];
        g[0] = acc.load_rest(nb);
        for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, g[i], g[i + 1]);
        Ba(n) = Collision::field(g);
      });
    }
    Kokkos::fence();
  }

  Domain dom_;
  Collision coll_;
  Streaming pop_[3];
  View1D<Real> B_[3];
  View1D<std::uint8_t>  wall_;
  View1D<std::uint16_t> tag_;
  View1D<std::uint32_t> unk_;
  View2D<Real>          wallB_;
  bool                  has_walls_ = false;
  std::size_t           n_wall_states_ = 0;
  View1D<Real> u_[3];
  std::size_t t_ = 0;
};

}  // namespace lbm
