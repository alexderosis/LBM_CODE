#pragma once
//==============================================================================
//  Passive-scalar solver: a SECOND distribution set, on its own lattice, with
//  its own streaming scheme and collision operator.
//
//  This is the first real test of the module composition the whole design was
//  built around. It shares the Domain and the streaming machinery with the fluid
//  but nothing else: D3Q7 has seven populations to the fluid's nineteen, a
//  different speed of sound (1/4 against 1/3), a different collision concept
//  (velocity is an input, not an output) and different boundary conditions.
//  Nothing in FluidSolver, the lattices, the streaming schemes or the moment
//  operators had to change to accommodate it.
//
//  BOUNDARY CONDITIONS, as alternative collisions on marked cells:
//    adiabatic  h_i^out = h_opp(i)^in                          (zero flux)
//    Dirichlet  h_i^out = -h_opp(i)^in + 2 w_i (T_wall - T_ref) (anti-bounce-back)
//
//  The Dirichlet form carries T_ref because the arrays hold h = g - w_i T_ref;
//  since w_i == w_opp(i) the reference simply shifts the target value.
//
//  Note the asymmetry with the fluid: bounce-back is the identity on Esoteric
//  Pull's storage, so adiabatic cells can be skipped outright, but anti-bounce-
//  back flips a sign and adds a source, so Dirichlet cells must be processed at
//  every step. Both still write back into the same two slots they read, so
//  neither disturbs the in-place scheme.
//==============================================================================
#include "collision/ScalarBGK.hpp"
#include "boundary/MomentDirichlet.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

template <class L, class Streaming, class Collision>
class ScalarSolver {
 public:
  using Lattice = L;
  static constexpr int Q = L::Q;
  static constexpr int NF = Streaming::nb_first;
  static constexpr int NS = Streaming::nb_stride;

  ScalarSolver(const Domain& dom, const Collision& coll)
      : dom_(dom), coll_(coll), pop_(dom),
        flags_("sflags", dom.n_padded),
        unk_("sunk", dom.n_padded),
        wall_("Twall", dom.n_padded),
        field_("T", dom.n_padded) {
    h_flags_ = Kokkos::create_mirror_view(flags_);
    h_wall_  = Kokkos::create_mirror_view(wall_);
    for (Index n = 0; n < dom.n_padded; ++n) {
      Index px, py, pz; dom_.coords(n, px, py, pz);
      h_flags_(n) = dom_.is_interior(px, py, pz) ? ScalarBulk : ScalarExcluded;
      h_wall_(n)  = Real(0);
    }
    Kokkos::deep_copy(flags_, h_flags_);
    Kokkos::deep_copy(wall_, h_wall_);
  }

  // fn(x, y, z) -> ScalarCell, over interior coordinates.
  template <class Fn>
  void set_geometry(Fn fn) {
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x)
          h_flags_(dom_.id(x, y, z)) = fn(x, y, z);
    Kokkos::deep_copy(flags_, h_flags_);
  }
  // fn(x, y, z) -> Real, the wall value used by Dirichlet cells.
  template <class Fn>
  void set_wall_values(Fn fn) {
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x)
          h_wall_(dom_.id(x, y, z)) = fn(x, y, z);
    Kokkos::deep_copy(wall_, h_wall_);
  }

  // The velocity field that advects this scalar -- owned by the fluid solver.
  void set_velocity(View1D<Real> ux, View1D<Real> uy, View1D<Real> uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  void initialize(Real T0) {
    initialize_field(KOKKOS_LAMBDA(Index) { return T0; });
  }
  // fn(n) -> Real, device-callable.
  template <class Fn>
  void initialize_field(Fn fn) {
    t_ = 0;
    const auto acc = pop_.template access<0>();
    const auto coll = coll_;
    const Domain d = dom_;
    auto wall = wall_; auto flags = flags_;
    Kokkos::parallel_for("sinit", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      const Real T = (flags(n) == ScalarDirichlet) ? wall(n) : fn(n);
      for (int i = 0; i < Q; ++i)
        acc.scatter(nb, i, coll.eq(i, T - coll.T_ref, Real(0), Real(0), Real(0)));
    });
    Kokkos::fence();
  }

  void step() {
    if (t_ % 2 == 0) run_step<0>(); else run_step<1>();
    pop_.end_of_step();
    ++t_;
  }

  // Fill the temperature field from the current populations.
  void compute_field() {
    if (t_ % 2 == 0) field_kernel<0>(); else field_kernel<1>();
  }

  View1D<Real> temperature() const { return field_; }
  View1D<std::uint8_t> flags() const { return flags_; }

  //--------------------------------------------------------------------------
  // Rebuild the unknown-direction masks used by ScalarMoment nodes. Call once
  // after set_geometry; geometry is static, so this is setup-time work.
  //--------------------------------------------------------------------------
  void finalize_geometry() {
    auto h_unk = Kokkos::create_mirror_view(unk_);
    for (Index n = 0; n < dom_.n_padded; ++n) h_unk(n) = 0;
    auto in_field = [&](Index x, Index y, Index z) {
      const std::uint8_t f = h_flags_(dom_.id(x, y, z));
      return f != ScalarExcluded;
    };
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x)
          if (h_flags_(dom_.id(x, y, z)) == ScalarMoment)
            h_unk(dom_.id(x, y, z)) = unknown_mask<L>(dom_, x, y, z, in_field);
    Kokkos::deep_copy(unk_, h_unk);
  }
  const Domain& domain() const { return dom_; }
  std::size_t timestep() const { return t_; }

 private:
  template <int P>
  void run_step() {
    const auto acc  = pop_.template access<P>();
    const auto coll = coll_;
    const Domain d  = dom_;
    auto flags = flags_; auto wall = wall_; auto field = field_; auto unk = unk_;
    auto ux = ux_, uy = uy_, uz = uz_;
    const bool have_u = ux.data() != nullptr;

    Kokkos::parallel_for("scalar_stream_collide", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        const std::uint8_t flag = flags(n);
        if (flag == ScalarExcluded) return;
        // Bounce-back is the identity on an in-place scheme, so an insulating
        // cell needs no work at all there. Anti-bounce-back is not, so a
        // Dirichlet cell is always processed.
        if constexpr (Streaming::implicit_bounce_back)
          if (flag == ScalarAdiabatic) return;

        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        Real g[Q];
        g[0] = acc.load_rest(nb);
        for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, g[i], g[i + 1]);

        if (flag == ScalarAdiabatic) {
          acc.store_rest(nb, g[0]);
          for (int i = 1; i < Q; i += 2) acc.store_pair(nb, i, g[i + 1], g[i]);
          return;
        }
        if (flag == ScalarMoment) {
          // Dellar's moment condition: pick the inward populations so that
          // sum_i h_i is the imposed deviation, putting T_wall exactly ON the
          // node rather than half-way to the next one, then collide normally.
          impose_moment<L>(g, wall(n) - coll.T_ref, unk(n));
          const Real dTm = Collision::deviation(g);
          const Real vx = have_u ? ux(n) : Real(0);
          const Real vy = have_u ? uy(n) : Real(0);
          const Real vz = have_u ? uz(n) : Real(0);
          coll.collide(g, dTm, vx, vy, vz);
          field(n) = coll.T_ref + dTm;
          acc.store_rest(nb, g[0]);
          for (int i = 1; i < Q; i += 2) acc.store_pair(nb, i, g[i], g[i + 1]);
          return;
        }
        if (flag == ScalarDirichlet) {
          const Real Tw = wall(n) - coll.T_ref;
          acc.store_rest(nb, -g[0] + Real(2) * weight<L, Real>(0) * Tw);
          for (int i = 1; i < Q; i += 2)
            acc.store_pair(nb, i,
                           -g[i + 1] + Real(2) * weight<L, Real>(i) * Tw,
                           -g[i]     + Real(2) * weight<L, Real>(i + 1) * Tw);
          return;
        }

        const Real dT = Collision::deviation(g);
        const Real vx = have_u ? ux(n) : Real(0);
        const Real vy = have_u ? uy(n) : Real(0);
        const Real vz = have_u ? uz(n) : Real(0);
        coll.collide(g, dT, vx, vy, vz);
        field(n) = coll.T_ref + dT;
        acc.store_rest(nb, g[0]);
        for (int i = 1; i < Q; i += 2) acc.store_pair(nb, i, g[i], g[i + 1]);
      });
  }

  template <int P>
  void field_kernel() {
    const auto acc = pop_.template access<P>();
    const Domain d = dom_;
    const auto coll = coll_;
    auto flags = flags_; auto wall = wall_; auto field = field_;
    Kokkos::parallel_for("scalar_field", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t flag = flags(n);
      if (flag == ScalarExcluded) { field(n) = Real(0); return; }
      if (flag == ScalarDirichlet) { field(n) = wall(n); return; }
      // A moment wall attains its value AT the node, so that is the field
      // there. The streamed populations still hold the unfixed inward
      // direction, so their sum is not the temperature.
      if (flag == ScalarMoment)    { field(n) = wall(n); return; }
      if (flag == ScalarAdiabatic) { field(n) = Real(0); return; }
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      Real g[Q];
      g[0] = acc.load_rest(nb);
      for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, g[i], g[i + 1]);
      field(n) = coll.temperature(g);
    });
    Kokkos::fence();
  }

  Domain dom_;
  Collision coll_;
  Streaming pop_;
  View1D<std::uint8_t> flags_;
  View1D<std::uint32_t> unk_;
  HostView1D<std::uint8_t> h_flags_;
  View1D<Real> wall_, field_;
  HostView1D<Real> h_wall_;
  View1D<Real> ux_, uy_, uz_;
  std::size_t t_ = 0;
};

}  // namespace lbm
