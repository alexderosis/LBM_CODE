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
//    outflow    h_i^out = h_i^eq(T_donor, u)                   (open boundary)
//
//  The Dirichlet form carries T_ref because the arrays hold h = g - w_i T_ref;
//  since w_i == w_opp(i) the reference simply shifts the target value.
//
//  Note the asymmetry with the fluid: bounce-back is the identity on Esoteric
//  Pull's storage, so adiabatic cells can be skipped outright, but anti-bounce-
//  back flips a sign and adds a source, so Dirichlet cells must be processed at
//  every step. Both still write back into the same two slots they read, so
//  neither disturbs the in-place scheme.
//
//  OUTFLOW, AND WHY IT NEEDS A SECOND PASS.
//
//  An open boundary has to be zero-gradient on the scalar, which means the node
//  must learn its value from an interior DONOR. The obvious implementation --
//  read the donor's populations inside the main kernel, as FluidSolver's
//  NrmOutFree does -- is a genuine read/write race under Esoteric Pull: the two
//  slots a node reads are exactly the two it writes, so a donor being processed
//  concurrently is rewriting the very slots the outflow node is reading. Whether
//  you get the pre- or post-collision state then depends on thread scheduling.
//
//  So outflow nodes are SKIPPED by the main kernel and handled by a second one
//  after a fence, and that pass reads `field`, not populations: compute_field
//  semantics are already "the concentration at this node", the main kernel has
//  just written it for every bulk node, and donors are guaranteed to be bulk --
//  so nothing the second pass reads is written by the second pass.
//
//  This costs a fence per step and stays race-free by construction rather than
//  by argument. Every storage slot still has exactly one writer: the main pass
//  skips outflow nodes entirely, and the outflow pass writes only the slots
//  those nodes own, so each slot is written exactly once per step either way.
//
//  The condition itself is equilibrium extrapolation -- the donor's value, the
//  node's own velocity, no non-equilibrium part. Discarding the non-equilibrium
//  part slightly damps the diffusive flux at the exit, which is negligible when
//  the exit is advection-dominated and is measured, not assumed, by the flux
//  balance in validation/gaussian_plume.cpp.
//==============================================================================
#include "collision/ScalarBGK.hpp"
#include "boundary/MomentDirichlet.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"

#include <cstdio>

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
        don_("sdon", dom.n_padded),
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
    auto wall = wall_; auto flags = flags_; auto field = field_;
    Kokkos::parallel_for("sinit", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      const Real T = (flags(n) == ScalarDirichlet) ? wall(n) : fn(n);
      field(n) = T;    // so compute_field is meaningful before the first step
      for (int i = 0; i < Q; ++i)
        acc.scatter(nb, i, coll.eq(i, T - coll.T_ref, Real(0), Real(0), Real(0)));
    });
    Kokkos::fence();
  }

  void step() {
    // The outflow pass must see the main pass's `field` writes, hence the fence.
    // It is skipped entirely when no cell is marked ScalarOutflow, so a problem
    // without open boundaries pays nothing for this.
    if (t_ % 2 == 0) {
      run_step<0>();
      if (has_outflow_) { Kokkos::fence(); outflow_kernel<0>(); }
    } else {
      run_step<1>();
      if (has_outflow_) { Kokkos::fence(); outflow_kernel<1>(); }
    }
    pop_.end_of_step();
    ++t_;
  }

  //--------------------------------------------------------------------------
  // Add fn(n) to the concentration, as w_i dC spread over the populations --
  // i.e. injected at rest, carrying no gradient and no momentum. A continuous
  // release is this called once per step with fn = rate * dt.
  //
  // scatter() is the exact inverse of load(), so this is a read-modify-write on
  // the slots load() reads. Every slot has exactly one reader across the whole
  // domain, so the update is race-free for the same reason the streaming is.
  //--------------------------------------------------------------------------
  template <class Fn>
  void add_source(Fn fn) {
    if (t_ % 2 == 0) source_kernel<0>(fn); else source_kernel<1>(fn);
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
    build_donors();
  }

  //--------------------------------------------------------------------------
  // Donor for every ScalarOutflow node: the interior cell whose value the node
  // copies. The outward directions are the axes whose neighbour is outside the
  // field, and the donor is the neighbour one step INWARD along all of them at
  // once -- so a face node takes its axis neighbour, an edge node the diagonal,
  // and a corner node the body diagonal. Without that, a box edge has no purely
  // axial interior neighbour at all and every edge and corner would be inert.
  //
  // A donor must be ScalarBulk, never another outflow node: the second pass
  // reads `field` at the donor, and reading it at a node the same pass writes
  // would put the race back. A node with no bulk neighbour is left pointing at
  // itself, which makes the condition a no-op there -- the node then keeps
  // whatever streamed into it, i.e. it falls back to bounce-back -- and is
  // counted and reported rather than silently reading a neighbour's garbage.
  //--------------------------------------------------------------------------
  void build_donors() {
    auto h_don = Kokkos::create_mirror_view(don_);
    for (Index n = 0; n < dom_.n_padded; ++n) h_don(n) = n;

    auto flag_at = [&](Index x, Index y, Index z) -> std::uint8_t {
      auto w = [](Index v, Index n, bool per) -> Index {
        if (!per) return v;
        if (v < 0) return v + n;
        if (v >= n) return v - n;
        return v;
      };
      x = w(x, dom_.nx, dom_.periodic[0]);
      y = w(y, dom_.ny, dom_.periodic[1]);
      z = w(z, dom_.nz, dom_.periodic[2]);
      if (x < 0 || x >= dom_.nx || y < 0 || y >= dom_.ny || z < 0 || z >= dom_.nz)
        return ScalarExcluded;
      return h_flags_(dom_.id(x, y, z));
    };

    const int dirs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
    Index nout = 0, degenerate = 0;
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x) {
          const Index n = dom_.id(x, y, z);
          if (h_flags_(n) != ScalarOutflow) continue;
          ++nout;
          Index ix = 0, iy = 0, iz = 0;
          for (int k = 0; k < 6; ++k)
            if (flag_at(x + dirs[k][0], y + dirs[k][1], z + dirs[k][2]) == ScalarExcluded) {
              ix -= dirs[k][0]; iy -= dirs[k][1]; iz -= dirs[k][2];
            }
          Index best = n;
          if ((ix || iy || iz) && flag_at(x + ix, y + iy, z + iz) == ScalarBulk)
            best = dom_.id(x + ix, y + iy, z + iz);
          if (best == n)                       // fall back to any bulk neighbour
            for (int k = 0; k < 6; ++k)
              if (flag_at(x + dirs[k][0], y + dirs[k][1], z + dirs[k][2]) == ScalarBulk) {
                best = dom_.id(x + dirs[k][0], y + dirs[k][1], z + dirs[k][2]);
                break;
              }
          h_don(n) = best;
          if (best == n) ++degenerate;
        }
    has_outflow_ = nout > 0;
    Kokkos::deep_copy(don_, h_don);
    if (degenerate)
      std::printf("  [scalar] %d of %d outflow node(s) have no bulk neighbour and are inert\n",
                  int(degenerate), int(nout));
  }
  const Domain& domain() const { return dom_; }
  std::size_t timestep() const { return t_; }

 public:
  //----------------------------------------------------------------------------
  // NVCC CONSTRAINT, NOT A DESIGN CHOICE. Everything from here to the data
  // members is an implementation detail and morally private. It is public
  // because CUDA forbids it otherwise: "the enclosing parent function for an
  // extended __host__ __device__ lambda cannot have private or protected access
  // within its class". Every one of these launches a Kokkos kernel, so every
  // one contains such a lambda, and marking them private makes the whole solver
  // uncompilable with nvcc. Found by building on a T4; the Threads backend
  // never complains, which is exactly why it went unnoticed.
  //
  // Do not call these from outside. They assume parity, fences and flag state
  // that only the public methods maintain.
  //----------------------------------------------------------------------------
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
        if (flag == ScalarOutflow) return;   // second pass; see the header
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
      // Already set by outflow_kernel, and the streamed populations here are
      // not the imposed state: the node is fully prescribed every step, so its
      // incoming directions are whatever the halo happened to hold.
      if (flag == ScalarOutflow) return;
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      Real g[Q];
      g[0] = acc.load_rest(nb);
      for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, g[i], g[i + 1]);
      field(n) = coll.temperature(g);
    });
    Kokkos::fence();
  }

  //--------------------------------------------------------------------------
  // Open boundary, second pass. The node is fully prescribed: every outgoing
  // population is set to equilibrium at the donor's concentration and the
  // node's own velocity, so whatever streamed in is irrelevant and the inward
  // directions carry the interior's value back into the domain.
  //--------------------------------------------------------------------------
  template <int P>
  void outflow_kernel() {
    const auto acc  = pop_.template access<P>();
    const auto coll = coll_;
    const Domain d  = dom_;
    auto flags = flags_; auto field = field_; auto don = don_;
    auto ux = ux_, uy = uy_, uz = uz_;
    const bool have_u = ux.data() != nullptr;

    Kokkos::parallel_for("scalar_outflow", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        if (flags(n) != ScalarOutflow) return;
        const Index src = don(n);
        if (src == n) return;                  // degenerate, reported at setup
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        const Real dT = field(src) - coll.T_ref;
        const Real vx = have_u ? ux(n) : Real(0);
        const Real vy = have_u ? uy(n) : Real(0);
        const Real vz = have_u ? uz(n) : Real(0);
        acc.store_rest(nb, coll.eq(0, dT, vx, vy, vz));
        for (int i = 1; i < Q; i += 2)
          acc.store_pair(nb, i, coll.eq(i, dT, vx, vy, vz),
                                coll.eq(i + 1, dT, vx, vy, vz));
        field(n) = coll.T_ref + dT;
      });
  }

  template <int P, class Fn>
  void source_kernel(Fn fn) {
    const auto acc = pop_.template access<P>();
    const Domain d = dom_;
    auto flags = flags_;
    Kokkos::parallel_for("scalar_source", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        if (flags(n) != ScalarBulk) return;
        const Real dC = fn(n);
        if (dC == Real(0)) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        for (int i = 0; i < Q; ++i)
          acc.scatter(nb, i, acc.load(nb, i) + weight<L, Real>(i) * dC);
      });
  }

 private:
  Domain dom_;
  Collision coll_;
  Streaming pop_;
  View1D<std::uint8_t> flags_;
  View1D<std::uint32_t> unk_;
  View1D<Index> don_;
  bool has_outflow_ = false;
  HostView1D<std::uint8_t> h_flags_;
  View1D<Real> wall_, field_;
  HostView1D<Real> h_wall_;
  View1D<Real> ux_, uy_, uz_;
  std::size_t t_ = 0;
};

}  // namespace lbm
