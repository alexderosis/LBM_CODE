#pragma once
//==============================================================================
//  Fused stream-collide fluid solver.
//
//  One kernel per timestep: pull the arriving populations, apply the collision
//  operator (or bounce-back on solid cells), write them back out. There is
//  deliberately no separate streaming pass -- the in-place schemes cannot have
//  one, so the solver is written the way Esoteric Pull needs it.
//
//  PARITY is a compile-time template parameter, never a runtime branch inside
//  the loop: Esoteric Pull selects different storage slots on even and odd
//  steps, and that decision must fold away at compile time.
//
//  Everything the kernel touches is a compile-time policy. The only dynamic
//  dispatch in the code base lives above this class, in the host-side factory.
//==============================================================================
#include "boundary/Flags.hpp"
#include "boundary/Regularized.hpp"
#include "core/Types.hpp"

#include <array>
#include <vector>
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

// Macroscopic state used to initialise a non-uniform flow field.
struct FlowState {
  Real rho = Real(1), ux = Real(0), uy = Real(0), uz = Real(0);
};

template <class L, class Streaming, class Collision>
class FluidSolver {
 public:
  using Lattice = L;
  static constexpr int Q = L::Q;
  static constexpr int NF = Streaming::nb_first;
  static constexpr int NS = Streaming::nb_stride;

  FluidSolver(const Domain& dom, const Collision& coll)
      : dom_(dom), coll_(coll), pop_(dom),
        flags_("flags", dom.n_padded),
        bc_nrm_("bc_nrm", dom.n_padded),
        bc_ext_("bc_ext", dom.n_padded),
        bc_tag_("bc_tag", dom.n_padded),
        bc_don_("bc_don", dom.n_padded),
        bc_onrm_("bc_onrm", dom.n_padded),
        bc_unk_("bc_unk", dom.n_padded),
        bc_rho_("bc_rho", dom.n_padded),
        wall_u_("wall_u", 1, 4),
        wall_u0_("wall_u0", 1, 4),
        rho_("rho", dom.n_padded),
        ux_("ux", dom.n_padded), uy_("uy", dom.n_padded), uz_("uz", dom.n_padded) {
    h_flags_ = Kokkos::create_mirror_view(flags_);
    // Halo cells exist only in non-periodic directions. They are never
    // processed, but solid cells at the domain edge read from them, so they are
    // marked Excluded rather than removed.
    for (Index n = 0; n < dom.n_padded; ++n) {
      Index px, py, pz; dom_.coords(n, px, py, pz);
      h_flags_(n) = dom_.is_interior(px, py, pz) ? Fluid : Excluded;
    }
    Kokkos::deep_copy(flags_, h_flags_);
    rebuild_lists();
  }

  //----------------------------------------------------------------------------
  // Geometry. fn(x, y, z) -> CellType, called on the host over INTERIOR coords.
  //----------------------------------------------------------------------------
  template <class Fn>
  void set_geometry(Fn fn) {
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x)
          h_flags_(dom_.id(x, y, z)) = fn(x, y, z);
    Kokkos::deep_copy(flags_, h_flags_);
    rebuild_lists();
  }

  //----------------------------------------------------------------------------
  // Regularised velocity walls (Latt et al. 2008, Sec. IV C).
  //
  // fn(x, y, z) -> WallSpec. A node with normal == NrmNone is left alone; any
  // other value marks it RegWall and imposes the given velocity. The normal
  // points OUT of the fluid domain. NrmCorner marks a node where two walls
  // meet, where the density closure cannot be evaluated.
  //
  // Wall velocities are deduplicated into a small table rather than stored per
  // node: a full-size velocity field would cost 24 bytes/node, which is
  // hundreds of megabytes on a 3D grid, and real geometries use a handful of
  // distinct values.
  //----------------------------------------------------------------------------
  struct WallSpec {
    std::uint8_t normal = NrmNone;
    Real ux = 0, uy = 0, uz = 0;
    // Imposed density. Read only by the pressure/outflow codes (NrmOutXp);
    // velocity walls close for rho themselves and ignore it.
    Real rho = Real(1);
  };

  // Corners and edges: how Pi^(1) is obtained there. Default is the
  // finite-difference route, which is what Latt et al. use (Sec. V) and which
  // measured ~35% lower error than the local closure on the Re = 1000 cavity.
  //
  //   true  -- from finite-difference velocity gradients. Infers the stress
  //            through Eq. (21), so it needs the shear relaxation rate; that is
  //            taken from the collision operator, never from the caller.
  //   false -- from the node's own populations, as on a straight wall. Cheap,
  //            local, and operator-agnostic, but a 2D corner offers only three
  //            streamed directions to build a stress from.
  //
  // Operators exposing no shear rate cannot use the FD route at all; for those
  // the local closure is selected at compile time rather than silently running
  // Eq. (21) with omega = 1, which would impose the wrong viscosity.
  //----------------------------------------------------------------------------
  // Rebuild the node lists from the current flags.
  //
  // MUST be called from EVERY path that writes h_flags_, and there are three:
  // the constructor, set_geometry and set_regularized_walls. Missing one is not
  // a small bug. An earlier version of this omitted set_geometry, so a case that
  // marks its walls Solid there and never calls set_regularized_walls -- the
  // thermal cavity does exactly that -- ran the whole simulation against a list
  // built when every interior node was still Fluid. Solid cells were then
  // collided as fluid, the bounce-back wall stopped existing, and the
  // natural-convection Nusselt number came out 15-21% high with nothing in the
  // output to say why.
  //
  // `active` mirrors exactly what the kernel used to skip, so semantics are
  // unchanged: Excluded never did work, and Solid does none either when the
  // streaming scheme bounces back implicitly. With an explicit scheme Solid
  // cells still have to be visited to write their reflected populations, so
  // they stay in the list.
  void rebuild_lists() {
    std::vector<Index> act, wal;
    act.reserve(std::size_t(dom_.n_padded));
    for (Index n = 0; n < dom_.n_padded; ++n) {
      const std::uint8_t fl = h_flags_(n);
      if (fl == Excluded) continue;
      if (fl == Solid && Streaming::implicit_bounce_back) continue;
      act.push_back(n);
      if (fl == RegWall) wal.push_back(n);
    }
    n_active_ = Index(act.size());
    n_walls_  = Index(wal.size());
    active_ = View1D<Index>("active", std::max<std::size_t>(1, act.size()));
    walls_  = View1D<Index>("walls",  std::max<std::size_t>(1, wal.size()));
    auto ha = Kokkos::create_mirror_view(active_);
    auto hw = Kokkos::create_mirror_view(walls_);
    for (std::size_t i = 0; i < act.size(); ++i) ha(i) = act[i];
    for (std::size_t i = 0; i < wal.size(); ++i) hw(i) = wal[i];
    Kokkos::deep_copy(active_, ha);
    Kokkos::deep_copy(walls_, hw);
  }

  Index active_count() const { return n_active_; }
  Index wall_count() const { return n_walls_; }

  void set_fd_corners(bool on) { fd_corners_ = on; }

  //----------------------------------------------------------------------------
  // Scale every stored wall velocity, relative to what set_regularized_walls
  // was given. For a driven boundary whose DIRECTION is fixed and whose
  // magnitude varies in time -- a pulsatile inlet -- this is the whole update.
  // The velocities live in a deduplicated table of a handful of entries, so
  // there is nothing per-node to rebuild and this costs nothing per step.
  //
  // ABSOLUTE, NOT INCREMENTAL. The scale multiplies the ORIGINAL table, held in
  // wall_u0_. Scaling the live table in place instead would compound: a
  // waveform bounded in [0, 1] would become a product of its own history and
  // decay to zero within a few hundred steps, which looks like a plausible
  // physical relaxation and is not one.
  //
  // Only the three velocity components scale. Slot 3 is an imposed density --
  // a pressure boundary, not part of the drive -- and is left alone, so an
  // outflow entry sitting in the same table is unaffected.
  //
  // corner_density() re-reads this table at the top of every step, so a change
  // here takes effect on the next step with no parity lag.
  //----------------------------------------------------------------------------
  void set_wall_velocity_scale(Real sc) {
    auto u = wall_u_; auto u0 = wall_u0_;
    Kokkos::parallel_for("wall_vel_scale", Range(0, Index(u.extent(0))),
      KOKKOS_LAMBDA(Index k) {
        u(k, 0) = u0(k, 0) * sc;
        u(k, 1) = u0(k, 1) * sc;
        u(k, 2) = u0(k, 2) * sc;
      });
    Kokkos::fence();
  }

  template <class Fn>
  void set_regularized_walls(Fn fn) {
    auto h_nrm = Kokkos::create_mirror_view(bc_nrm_);
    auto h_tag = Kokkos::create_mirror_view(bc_tag_);
    std::vector<std::array<Real, 4>> table;
    for (Index n = 0; n < dom_.n_padded; ++n) { h_nrm(n) = NrmNone; h_tag(n) = 0; }

    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x) {
          const WallSpec w = fn(x, y, z);
          if (w.normal == NrmNone) continue;
          const Index n = dom_.id(x, y, z);
          std::array<Real, 4> v{w.ux, w.uy, w.uz, w.rho};
          std::size_t k = 0;
          for (; k < table.size(); ++k)
            if (table[k][0] == v[0] && table[k][1] == v[1] &&
                table[k][2] == v[2] && table[k][3] == v[3]) break;
          if (k == table.size()) table.push_back(v);
          h_nrm(n) = w.normal;
          h_tag(n) = static_cast<std::uint16_t>(k);
          h_flags_(n) = RegWall;
        }
    // Corner nodes: Eq. (27) needs a single normal, so rho is extrapolated
    // along a wall instead. Pick, per corner, an axis direction whose next two
    // nodes are BOTH straight-wall nodes -- those have a well-defined rho, so
    // the stencil never reaches into the bulk and never needs a density field.
    auto h_ext = Kokkos::create_mirror_view(bc_ext_);
    for (Index n = 0; n < dom_.n_padded; ++n) h_ext(n) = NrmNone;
    has_corners_ = false;
    const int dirs[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x) {
          const Index n = dom_.id(x, y, z);
          if (h_nrm(n) != NrmCorner) continue;
          has_corners_ = true;
          for (int k = 0; k < (dom_.nz > 1 ? 6 : 4); ++k) {
            const Index x1 = x + dirs[k][0], y1 = y + dirs[k][1], z1 = z + dirs[k][2];
            const Index x2 = x + 2*dirs[k][0], y2 = y + 2*dirs[k][1], z2 = z + 2*dirs[k][2];
            if (x1 < 0 || y1 < 0 || z1 < 0 || x2 < 0 || y2 < 0 || z2 < 0) continue;
            if (x1 >= dom_.nx || y1 >= dom_.ny || z1 >= dom_.nz) continue;
            if (x2 >= dom_.nx || y2 >= dom_.ny || z2 >= dom_.nz) continue;
            const std::uint8_t a = h_nrm(dom_.id(x1, y1, z1));
            const std::uint8_t b = h_nrm(dom_.id(x2, y2, z2));
            if (a != NrmNone && a != NrmCorner && b != NrmNone && b != NrmCorner) {
              h_ext(n) = static_cast<std::uint8_t>(k + 1);
              break;
            }
          }
        }
    Kokkos::deep_copy(bc_ext_, h_ext);

    // Unknown-direction masks. Population i arrives from node (p - c_i); if that
    // source is outside the domain or not a fluid-like cell, nothing streamed
    // along i and the value sitting there is stale.
    auto h_unk = Kokkos::create_mirror_view(bc_unk_);
    for (Index n = 0; n < dom_.n_padded; ++n) h_unk(n) = 0u;
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x) {
          const Index n = dom_.id(x, y, z);
          if (h_nrm(n) == NrmNone) continue;
          std::uint32_t m = 0;
          for (int i = 0; i < Q; ++i) {
            Index sx = x - cvel<L>(i, 0), sy = y - cvel<L>(i, 1), sz = z - cvel<L>(i, 2);
            bool outside = false;
            if (dom_.periodic[0]) { sx = (sx + dom_.nx) % dom_.nx; }
            else if (sx < 0 || sx >= dom_.nx) outside = true;
            if (dom_.periodic[1]) { sy = (sy + dom_.ny) % dom_.ny; }
            else if (sy < 0 || sy >= dom_.ny) outside = true;
            if (dom_.periodic[2]) { sz = (sz + dom_.nz) % dom_.nz; }
            else if (sz < 0 || sz >= dom_.nz) outside = true;
            if (!outside && h_flags_(dom_.id(sx, sy, sz)) == Excluded) outside = true;
            if (outside) m |= (1u << i);
          }
          h_unk(n) = m;
        }
    Kokkos::deep_copy(bc_unk_, h_unk);

    if (table.empty()) table.push_back({Real(0), Real(0), Real(0)});
    // Donor lookup for arbitrary-face outflow nodes: step one cell along each
    // axis and take the first FLUID neighbour. On a voxelised cap the interior
    // is whichever side is not solid, so this finds it without needing a
    // normal. A cap node with no fluid neighbour at all is degenerate -- it
    // would be a one-cell pocket -- and is left pointing at itself, which makes
    // the condition a no-op there rather than reading a solid node's garbage.
    {
      auto h_don = Kokkos::create_mirror_view(bc_don_);
      auto h_onrm = Kokkos::create_mirror_view(bc_onrm_);
      const int dirs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
      const std::uint8_t codes[6] = {NrmXm, NrmXp, NrmYm, NrmYp, NrmZm, NrmZp};
      for (Index n = 0; n < dom_.n_padded; ++n) { h_don(n) = n; h_onrm(n) = NrmNone; }
      Index degenerate = 0;
      for (Index z = 0; z < dom_.nz; ++z)
        for (Index y = 0; y < dom_.ny; ++y)
          for (Index x = 0; x < dom_.nx; ++x) {
            const Index n = dom_.id(x, y, z);
            if (h_nrm(n) != NrmOutFree) continue;
            // The outward normal is the axis step that LEAVES the fluid, and
            // the donor is the fluid neighbour opposite it. Taking both from
            // the same axis keeps the closure below exactly the axis-aligned
            // one, which is what makes a voxel face tractable.
            int best = -1;
            for (int k = 0; k < 6; ++k) {
              const Index qx = x + dirs[k][0], qy = y + dirs[k][1], qz = z + dirs[k][2];
              const bool leaves =
                  (qx < 0 || qx >= dom_.nx || qy < 0 || qy >= dom_.ny ||
                   qz < 0 || qz >= dom_.nz) ||
                  (h_flags_(dom_.id(qx, qy, qz)) == Solid);
              if (!leaves) continue;
              // require the OPPOSITE neighbour to be fluid, so a donor exists
              const int o = k ^ 1;
              const Index rx = x + dirs[o][0], ry = y + dirs[o][1], rz = z + dirs[o][2];
              if (rx < 0 || rx >= dom_.nx || ry < 0 || ry >= dom_.ny ||
                  rz < 0 || rz >= dom_.nz) continue;
              const Index m = dom_.id(rx, ry, rz);
              if (h_flags_(m) != Fluid) continue;
              best = k; h_don(n) = m; h_onrm(n) = codes[k];
              break;
            }
            if (best < 0) ++degenerate;
          }
      if (degenerate)
        std::printf("  [walls] %d outflow node(s) have no fluid neighbour and are inert\n",
                    int(degenerate));
      Kokkos::deep_copy(bc_don_, h_don);
      Kokkos::deep_copy(bc_onrm_, h_onrm);
    }

    if (table.size() > std::size_t(std::numeric_limits<std::uint16_t>::max()))
      throw std::runtime_error("set_regularized_walls: " + std::to_string(table.size()) +
                               " distinct wall states exceeds the tag width");
    n_wall_states_ = table.size();
    wall_u_ = View2D<Real>("wall_u", table.size(), 4);
    auto h_u = Kokkos::create_mirror_view(wall_u_);
    for (std::size_t k = 0; k < table.size(); ++k)
      for (int a = 0; a < 4; ++a) h_u(k, a) = table[k][a];
    Kokkos::deep_copy(wall_u_, h_u);
    // Keep the table as given, so a time-varying drive can rescale it without
    // compounding. See set_wall_velocity_scale.
    wall_u0_ = View2D<Real>("wall_u0", table.size(), 4);
    Kokkos::deep_copy(wall_u0_, wall_u_);
    Kokkos::deep_copy(bc_nrm_, h_nrm);
    Kokkos::deep_copy(bc_tag_, h_tag);
    Kokkos::deep_copy(flags_, h_flags_);
    rebuild_lists();
  }

  //----------------------------------------------------------------------------
  // Equilibrium initialisation.
  //
  // Solid and halo cells are seeded with the REST equilibrium at rho0, not with
  // zero. A bounce-back cell is a real storage cell holding in-transit
  // populations; starting it empty makes the wall layer soak mass out of the
  // fluid and leaves the bulk density low by O(area/volume) = O(1/H), a
  // first-order error in a quantity the Poiseuille amplitude depends on
  // (mu = rho nu).
  //----------------------------------------------------------------------------
  void initialize(Real rho0, Real ux0 = 0, Real uy0 = 0, Real uz0 = 0) {
    t_ = 0;
    auto flags = flags_;
    seed_populations(KOKKOS_LAMBDA(Index n, int i) {
      return (flags(n) == Fluid || flags(n) == RegWall)
           ? Collision::seed_value(i, rho0, ux0, uy0, uz0)
           : Collision::seed_value(i, rho0, Real(0), Real(0), Real(0));
    });
  }

  //----------------------------------------------------------------------------
  // Non-uniform initialisation. fn must be device-callable: (Index n) -> FlowState,
  // where n is a padded linear index (use domain().coords() to get position).
  // Solid and halo cells are still seeded at rest.
  //----------------------------------------------------------------------------
  template <class Fn>
  void initialize_field(Fn fn) {
    t_ = 0;
    auto flags = flags_;
    seed_populations(KOKKOS_LAMBDA(Index n, int i) {
      const FlowState s = (flags(n) == Fluid || flags(n) == RegWall) ? fn(n) : FlowState{};
      return Collision::seed_value(i, s.rho, s.ux, s.uy, s.uz);
    });
  }

  //----------------------------------------------------------------------------
  void step(bool store_macroscopic = false) {
    if (has_corners_) { if (t_ % 2 == 0) corner_density<0>(); else corner_density<1>(); }
    if (t_ % 2 == 0) {
      if (store_macroscopic) run_step<0, true>(); else run_step<0, false>();
    } else {
      if (store_macroscopic) run_step<1, true>(); else run_step<1, false>();
    }
    pop_.end_of_step();
    ++t_;
  }

  //----------------------------------------------------------------------------
  // Fill rho/u from the current populations without advancing time. The parity
  // to read with is the one the NEXT step would load with.
  //----------------------------------------------------------------------------
  void compute_macroscopic() {
    if (t_ % 2 == 0) macro_kernel<0>(); else macro_kernel<1>();
  }

  Real total_mass() const {
    return (t_ % 2 == 0) ? reduce<0>(-1) : reduce<1>(-1);
  }
  Real total_momentum(int a) const {
    return (t_ % 2 == 0) ? reduce<0>(a) : reduce<1>(a);
  }

  const Domain& domain() const { return dom_; }
  Streaming&    populations()  { return pop_; }
  Collision&    collision()    { return coll_; }
  View1D<Real>  rho() const { return rho_; }
  View1D<Real>  ux()  const { return ux_; }
  View1D<Real>  uy()  const { return uy_; }
  View1D<Real>  uz()  const { return uz_; }
  View1D<std::uint8_t> flags() const { return flags_; }
  std::size_t n_wall_states() const { return n_wall_states_; }
  std::size_t   timestep() const { return t_; }

  //----------------------------------------------------------------------------
  // Logical population access, i.e. "the population travelling in direction i
  // that is about to arrive at node n", independent of how the streaming scheme
  // actually lays that out in memory. Used by the streaming tests and by the
  // TwoLattice/EsotericPull cross-check; not on any hot path.
  //----------------------------------------------------------------------------
  View2D<Real> gather_populations() const {
    View2D<Real> out("gathered", dom_.n_padded, Q);
    if (t_ % 2 == 0) gather<0>(out); else gather<1>(out);
    return out;
  }

  // fn must be device-callable: (Index n, int i) -> Real, giving the population
  // travelling in direction i that arrives at node n. Exact inverse of
  // gather_populations().
  template <class Fn>
  void seed_populations(Fn fn) {
    if (t_ % 2 == 0) scatter<0>(fn); else scatter<1>(fn);
  }

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
  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  // Corner densities, run before the main sweep and only when corners exist.
  //
  // Pass 1 evaluates Eq. (27) on every straight wall node; pass 2 extrapolates
  // those to the corners with second-order accuracy. Both read populations at
  // the SAME parity the main sweep is about to read with, so the corner sees
  // the same state as everything else -- reading a stale parity here would put
  // a one-step lag into the corner density and quietly spoil the order.
  //----------------------------------------------------------------------------
  template <int P>
  void corner_density() {
    const auto acc = pop_.template access<P>();
    const Domain d = dom_;
    auto flags = flags_; auto bc_nrm = bc_nrm_; auto bc_tag = bc_tag_;
    auto bc_ext = bc_ext_; auto bc_rho = bc_rho_; auto wall_u = wall_u_;
    const Real out_rho = outflow_rho_;

    auto walls = walls_;
    Kokkos::parallel_for("bc_wall_rho", Range(0, n_walls_), KOKKOS_LAMBDA(Index wi) {
      const Index n = walls(wi);
      if (flags(n) != RegWall) return;
      const std::uint8_t code = bc_nrm(n);
      if (code == NrmCorner || code == NrmNone) return;
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      Real f[Q];
      f[0] = acc.load_rest(nb);
      for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, f[i], f[i + 1]);
      if constexpr (Collision::Storage::shifted)
        for (int i = 0; i < Q; ++i) f[i] += weight<L, Real>(i);
      int nrm[3]; normal_of(code, nrm);
      const int tg = int(bc_tag(n));
      const Real uw[3] = {wall_u(tg, 0), wall_u(tg, 1), wall_u(tg, 2)};
      bc_rho(n) = (code == NrmOutXp)   ? wall_u(tg, 3)
                : (code == NrmOutFree) ? out_rho
                                       : BC::density(f, nrm, uw);
    });

    Kokkos::parallel_for("bc_corner_rho", Range(0, n_walls_), KOKKOS_LAMBDA(Index wi) {
      const Index n = walls(wi);
      if (flags(n) != RegWall || bc_nrm(n) != NrmCorner) return;
      const std::uint8_t k = bc_ext(n);
      if (k == NrmNone) { bc_rho(n) = Real(1); return; }   // no valid stencil
      const int dirs[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
      const int* e = dirs[k - 1];
      // coords() returns PADDED coordinates; id() takes INTERIOR ones. Mixing
      // them silently reads a different node entirely.
      Index px, py, pz; d.coords(n, px, py, pz);
      const Index x = px - d.hx, y = py - d.hy, z = pz - d.hz;
      const Index n1 = d.id(x + e[0], y + e[1], z + e[2]);
      const Index n2 = d.id(x + 2*e[0], y + 2*e[1], z + 2*e[2]);
      bc_rho(n) = Real(2) * bc_rho(n1) - bc_rho(n2);
    });
  }

  template <int P, bool StoreMacro>
  void run_step() {
    const auto acc  = pop_.template access<P>();
    const auto coll = coll_;                  // by value: no `this` on device
    const Domain d  = dom_;
    auto flags = flags_;
    auto bc_nrm = bc_nrm_; auto bc_tag = bc_tag_; auto wall_u = wall_u_;
    auto bc_rho = bc_rho_; auto bc_unk = bc_unk_; auto bc_don = bc_don_; auto bc_onrm = bc_onrm_;
    const bool fd_corners = fd_corners_ && has_shear_omega<Collision>;
    const bool force_bc   = force_bc_;
    const int  out_order  = out_order_;
    const Real out_rho    = outflow_rho_;
    const Real bc_omega   = shear_omega(coll_);   // host side, once per step
    [[maybe_unused]] auto rho = rho_; [[maybe_unused]] auto ux = ux_;
    [[maybe_unused]] auto uy  = uy_;  [[maybe_unused]] auto uz = uz_;

    auto active = active_;
    Kokkos::parallel_for("stream_collide", Range(0, n_active_), KOKKOS_LAMBDA(Index ai) {
      const Index n = active(ai);
      const std::uint8_t flag = flags(n);
      // The list already excludes both of these. The tests are KEPT anyway, as
      // a guard: they cost one flag read on cells that are going to read flags
      // regardless, and they turn a stale list into a performance loss instead
      // of silent corruption. That distinction is not hypothetical -- see the
      // note on rebuild_lists().
      if (flag == Excluded) return;
      // Esoteric Pull's bounce-back is the identity on the storage, so a solid
      // cell is skipped outright -- no load, no store, no arithmetic.
      if constexpr (Streaming::implicit_bounce_back) {
        if (flag == Solid) return;
      }

      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);

      Real f[Q];
      f[0] = acc.load_rest(nb);
      for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, f[i], f[i + 1]);

      if constexpr (!Streaming::implicit_bounce_back) {
        if (flag == Solid) {
          // Halfway bounce-back as a collision operator: reflect everything
          // received, which puts the wall midway between this cell and its
          // fluid neighbours.
          for (int i = 0; i < Q; ++i) acc.store(nb, i, f[opp(i)]);
          return;
        }
      }

      if (flag == RegWall) {
        // Replace every population from (rho, u, Pi^(1)) before colliding.
        // Eq. (45) carries no omega, so this is collision-operator agnostic.
        //
        // The reconstruction is written for RAW populations. Under shifted
        // storage f holds g_i = f_i - w_i, so undo the shift around it; the
        // condition would otherwise rebuild the wrong distribution entirely.
        if constexpr (Collision::Storage::shifted)
          for (int i = 0; i < Q; ++i) f[i] += weight<L, Real>(i);
        const std::uint8_t code = bc_nrm(n);
        int nrm[3]; normal_of(code, nrm);
        const int tg = int(bc_tag(n));
        const Real uw[3] = {wall_u(tg, 0), wall_u(tg, 1), wall_u(tg, 2)};

        // KNOWN DEFECT, mechanism not identified. With a body force this
        // condition develops a wall slip proportional to F. Measured facts:
        //
        //   * with NO force the wall sits on the node to round-off (4e-15) for
        //     every tau from 0.6 to 2.0 -- the condition itself is exact;
        //   * with a force the slip is proportional to F, scales as 1/W, and
        //     depends strongly on tau, changing SIGN near tau = 0.889.
        //
        // Moment analysis says the reconstruction violates
        //     sum_i c_i g_i = rho u_w - F/2
        // by exactly -F/2, repaired by the additive Hermite term
        //     -(w_i / 2 cs^2) c_i . F.
        // But -F/2 is tau-INDEPENDENT and the measured defect is not, so that
        // is not the dominant mechanism and the correction does not remove it.
        // Four hypotheses were tested against both forced Poiseuille and
        // Rayleigh-Benard and all were falsified: the velocity shift
        // u_w -/+ F/(2 rho) (each helps one flow and harms the other), applying
        // no force at the wall node, and the scaffold-symmetrisation argument.
        // No correction is applied rather than one fitted to a single flow.
        Real Fv[3] = {Real(0), Real(0), Real(0)};
        if constexpr (has_forcing<Collision>) coll.forcing.at(n, Fv);
        Real ur[3] = {uw[0], uw[1], uw[2]};
        Real rw;
        if (code == NrmOutFree) {
          // Arbitrary-face outflow, by POPULATION extrapolation rather than by
          // a moment closure.
          //
          // Two closures were tried first and both failed on a voxelised cap.
          // Imposing rho together with the full donor velocity over-determines
          // the node: the outlet passed 2.4 against an inlet flux of 17.0 and
          // blew up in ~1500 steps at every viscosity. Deriving the normal
          // velocity from the inverted closure instead is worse, diverging in
          // 500, because that closure assumes ONE open axis face -- at a cap
          // node inside a vessel several other directions are solid too, so the
          // populations it treats as "streamed from the interior" are partly
          // bounce-back and the partition is wrong.
          //
          // What works is to take the donor's whole distribution, which is by
          // construction a valid state carrying the interior's velocity AND its
          // non-equilibrium stress, and rescale it to the imposed density. That
          // is zero-gradient on the shape of the distribution and Dirichlet on
          // the pressure, which anchors rho without over-specifying anything.
          const Index src = bc_don(n);
          Neighbours<L> nbu;
          d.template fill_neighbours<L, NF, NS>(src, nbu);
          Real g[Q];
          g[0] = acc.load_rest(nbu);
          for (int i = 1; i < Q; i += 2) acc.load_pair(nbu, i, g[i], g[i + 1]);
          if constexpr (Collision::Storage::shifted)
            for (int i = 0; i < Q; ++i) g[i] += weight<L, Real>(i);
          Real sm = Real(0);
          for (int i = 0; i < Q; ++i) sm += g[i];
          if (sm > Real(0)) {
            const Real sc = out_rho / sm;
            for (int i = 0; i < Q; ++i) f[i] = g[i] * sc;
          }
          if constexpr (Collision::Storage::shifted)
            for (int i = 0; i < Q; ++i) f[i] -= weight<L, Real>(i);
          for (int i = 1; i < Q; i += 2) acc.store_pair(nb, i, f[i], f[i + 1]);
          acc.store_rest(nb, f[0]);
          return;
        } else if (code == NrmOutXp) {
          // Constant back-pressure outlet. rho is imposed -- see the NrmOutXp
          // comment in Regularized.hpp for why extrapolating it instead makes
          // the channel ill-posed. The normal velocity then follows from the
          // inverted closure, so the outlet passes whatever flux the interior
          // sends it, and the tangential velocity is zero-gradient from the
          // upstream neighbour (the node's own populations carry no inflow
          // directions and cannot supply it).
          rw = wall_u(tg, 3);
          int e[3]; upstream_of(code, e);
          Index px, py, pz; d.coords(n, px, py, pz);
          const Index bx = px - d.hx, by = py - d.hy, bz = pz - d.hz;
          // Tangential velocity at the k-th upstream neighbour. The outlet
          // node's own populations carry no inflow directions, so it cannot
          // supply this itself.
          auto up_vel = [&](int k, Real v[3]) {
            const Index mm = d.id(bx + k * e[0], by + k * e[1], bz + k * e[2]);
            Neighbours<L> nbu;
            d.template fill_neighbours<L, NF, NS>(mm, nbu);
            Real g[Q];
            g[0] = acc.load_rest(nbu);
            for (int i = 1; i < Q; i += 2) acc.load_pair(nbu, i, g[i], g[i + 1]);
            if constexpr (Collision::Storage::shifted)
              for (int i = 0; i < Q; ++i) g[i] += weight<L, Real>(i);
            Real sm = Real(0); v[0] = v[1] = v[2] = Real(0);
            for (int i = 0; i < Q; ++i) {
              sm += g[i];
              for (int a = 0; a < 3; ++a) v[a] += g[i] * Real(cvel<L>(i, a));
            }
            const Real ir = Real(1) / sm;
            for (int a = 0; a < 3; ++a) v[a] *= ir;
          };
          Real ut[3]; up_vel(1, ut);
          if (out_order >= 2) {               // linear extrapolation, 2nd order
            Real v2[3]; up_vel(2, v2);
            for (int a = 0; a < 3; ++a) ut[a] = Real(2) * ut[a] - v2[a];
          }
          const Real un = BC::normal_velocity(f, nrm, rw);
          Real utn = Real(0);
          for (int a = 0; a < 3; ++a) utn += ut[a] * Real(nrm[a]);
          for (int a = 0; a < 3; ++a) ur[a] = ut[a] + (un - utn) * Real(nrm[a]);
        } else {
          rw = (code == NrmCorner) ? bc_rho(n) : BC::density(f, nrm, ur);
        }

        if (fd_corners && code == NrmCorner) {
          // Velocity at a neighbour: imposed if it is itself a wall node --
          // its streamed populations still hold unfixed unknowns and would give
          // a wrong answer -- otherwise taken from its populations.
          auto vel_at = [&](Index m, Real out[3]) {
            if (flags(m) == RegWall) {
              const int t2 = int(bc_tag(m));
              out[0] = wall_u(t2, 0); out[1] = wall_u(t2, 1); out[2] = wall_u(t2, 2);
              return;
            }
            Neighbours<L> nbm;
            d.template fill_neighbours<L, NF, NS>(m, nbm);
            Real g2[Q];
            g2[0] = acc.load_rest(nbm);
            for (int i = 1; i < Q; i += 2) acc.load_pair(nbm, i, g2[i], g2[i + 1]);
            Real sm = 0, mx = 0, my = 0, mz = 0;
            for (int i = 0; i < Q; ++i) {
              Real v = g2[i];
              if constexpr (Collision::Storage::shifted) v += weight<L, Real>(i);
              sm += v;
              mx += v * Real(cvel<L>(i, 0));
              my += v * Real(cvel<L>(i, 1));
              mz += v * Real(cvel<L>(i, 2));
            }
            const Real ir = Real(1) / sm;
            out[0] = mx * ir; out[1] = my * ir; out[2] = mz * ir;
          };

          Index px, py, pz; d.coords(n, px, py, pz);
          const Index ic[3] = {px - d.hx, py - d.hy, pz - d.hz};
          const Index nn[3] = {d.nx, d.ny, d.nz};
          Real grad[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
          for (int a = 0; a < L::D; ++a) {
            if (nn[a] < 3) continue;
            int dir = 0;
            if (!d.periodic[a]) {
              if (ic[a] == 0)            dir =  1;   // one-sided, into the fluid
              else if (ic[a] == nn[a]-1) dir = -1;
            }
            Real u1[3], u2[3];
            Index c1[3] = {ic[0], ic[1], ic[2]}, c2[3] = {ic[0], ic[1], ic[2]};
            if (dir != 0) {
              c1[a] += dir; c2[a] += 2*dir;
              vel_at(d.id(c1[0], c1[1], c1[2]), u1);
              vel_at(d.id(c2[0], c2[1], c2[2]), u2);
              for (int b = 0; b < L::D; ++b)          // (-3 f0 + 4 f1 - f2) / 2
                grad[a][b] = Real(dir) * (Real(-1.5)*uw[b] + Real(2)*u1[b] - Real(0.5)*u2[b]);
            } else {
              c1[a] = (ic[a] + 1) % nn[a];
              c2[a] = (ic[a] - 1 + nn[a]) % nn[a];
              vel_at(d.id(c1[0], c1[1], c1[2]), u1);
              vel_at(d.id(c2[0], c2[1], c2[2]), u2);
              for (int b = 0; b < L::D; ++b) grad[a][b] = Real(0.5) * (u1[b] - u2[b]);
            }
          }
          Real Pi[6];
          BC::stress_from_gradient(rw, bc_omega, grad, Pi);
          BC::apply_with_stress(f, rw, ur, Pi);
        } else {
          BC::apply(f, rw, ur, bc_unk(n), force_bc ? Fv : nullptr);
        }
        if constexpr (Collision::Storage::shifted)
          for (int i = 0; i < Q; ++i) f[i] -= weight<L, Real>(i);
      }

      const Macro m = coll.macroscopic(f, n);
      coll.collide(f, m, n);
      acc.store_rest(nb, f[0]);
      for (int i = 1; i < Q; i += 2) acc.store_pair(nb, i, f[i], f[i + 1]);

      // A plain `if`, deliberately not `if constexpr`. NVCC cannot let an
      // extended __host__ __device__ lambda FIRST capture a variable inside an
      // `if constexpr` body, and rho/ux/uy/uz are first named right here.
      // StoreMacro is a compile-time constant so the branch still folds away;
      // the only cost is that the four views are captured unconditionally,
      // which is a few bytes of closure and no runtime work.
      if (StoreMacro) {
        rho(n) = Collision::density(m);
        ux(n) = m.ux; uy(n) = m.uy; uz(n) = m.uz;
      }
    });
  }

  //----------------------------------------------------------------------------
  template <int P>
  void macro_kernel() {
    const auto acc  = pop_.template access<P>();
    const auto coll = coll_;
    const Domain d  = dom_;
    auto flags = flags_;
    auto bc_nrm = bc_nrm_; auto bc_tag = bc_tag_;
    auto bc_rho = bc_rho_; auto wall_u = wall_u_; auto bc_don = bc_don_; auto bc_onrm = bc_onrm_;
    const Real out_rho = outflow_rho_;
    const int out_order = out_order_;
    auto rho = rho_; auto ux = ux_; auto uy = uy_; auto uz = uz_;
    Kokkos::parallel_for("macro", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (flags(n) != Fluid && flags(n) != RegWall) {
        rho(n) = Real(0); ux(n) = uy(n) = uz(n) = Real(0); return;
      }
      // At a regularised wall the streamed populations still hold the unknown
      // directions, so their raw moments are meaningless. The velocity there is
      // the imposed one by construction; report that, and the density the
      // closure gives, rather than a number that means nothing.
      if (flags(n) == RegWall) {
        const std::uint8_t code = bc_nrm(n);
        int nrm[3]; normal_of(code, nrm);
        const int tg = int(bc_tag(n));
        Neighbours<L> nbw;
        d.template fill_neighbours<L, NF, NS>(n, nbw);
        Real fw[Q];
        fw[0] = acc.load_rest(nbw);
        for (int i = 1; i < Q; i += 2) acc.load_pair(nbw, i, fw[i], fw[i + 1]);
        if constexpr (Collision::Storage::shifted)
          for (int i = 0; i < Q; ++i) fw[i] += weight<L, Real>(i);
        Real uw[3] = {wall_u(tg, 0), wall_u(tg, 1), wall_u(tg, 2)};
        Real rw;
        if (code == NrmOutFree) {
          rw = out_rho;
          int onr[3]; normal_of(bc_onrm(n), onr);
          const Index src = bc_don(n);
          Neighbours<L> nbu;
          d.template fill_neighbours<L, NF, NS>(src, nbu);
          Real g[Q];
          g[0] = acc.load_rest(nbu);
          for (int i = 1; i < Q; i += 2) acc.load_pair(nbu, i, g[i], g[i + 1]);
          if constexpr (Collision::Storage::shifted)
            for (int i = 0; i < Q; ++i) g[i] += weight<L, Real>(i);
          Real sm = 0, m4[3] = {Real(0), Real(0), Real(0)};
          for (int i = 0; i < Q; ++i) {
            sm += g[i];
            for (int a = 0; a < 3; ++a) m4[a] += g[i] * Real(cvel<L>(i, a));
          }
          const Real ir = (sm > Real(0)) ? Real(1) / sm : Real(0);
          const Real un = BC::normal_velocity(fw, onr, rw);
          Real utn = Real(0);
          for (int a = 0; a < 3; ++a) utn += m4[a] * ir * Real(onr[a]);
          for (int a = 0; a < 3; ++a) uw[a] = m4[a] * ir + (un - utn) * Real(onr[a]);
        } else if (code == NrmOutXp) {
          // Mirrors the outflow branch of run_step, which is the source of
          // truth: rho imposed, normal velocity from the inverted closure,
          // tangential zero-gradient. Kept in step with it by hand.
          rw = wall_u(tg, 3);
          int e[3]; upstream_of(code, e);
          Index px, py, pz; d.coords(n, px, py, pz);
          const Index bx = px - d.hx, by = py - d.hy, bz = pz - d.hz;
          // Tangential velocity at the k-th upstream neighbour. The outlet
          // node's own populations carry no inflow directions, so it cannot
          // supply this itself.
          auto up_vel = [&](int k, Real v[3]) {
            const Index mm = d.id(bx + k * e[0], by + k * e[1], bz + k * e[2]);
            Neighbours<L> nbu;
            d.template fill_neighbours<L, NF, NS>(mm, nbu);
            Real g[Q];
            g[0] = acc.load_rest(nbu);
            for (int i = 1; i < Q; i += 2) acc.load_pair(nbu, i, g[i], g[i + 1]);
            if constexpr (Collision::Storage::shifted)
              for (int i = 0; i < Q; ++i) g[i] += weight<L, Real>(i);
            Real sm = Real(0); v[0] = v[1] = v[2] = Real(0);
            for (int i = 0; i < Q; ++i) {
              sm += g[i];
              for (int a = 0; a < 3; ++a) v[a] += g[i] * Real(cvel<L>(i, a));
            }
            const Real ir = Real(1) / sm;
            for (int a = 0; a < 3; ++a) v[a] *= ir;
          };
          Real ut[3]; up_vel(1, ut);
          if (out_order >= 2) {               // linear extrapolation, 2nd order
            Real v2[3]; up_vel(2, v2);
            for (int a = 0; a < 3; ++a) ut[a] = Real(2) * ut[a] - v2[a];
          }
          const Real un = BC::normal_velocity(fw, nrm, rw);
          Real utn = Real(0);
          for (int a = 0; a < 3; ++a) utn += ut[a] * Real(nrm[a]);
          for (int a = 0; a < 3; ++a) uw[a] = ut[a] + (un - utn) * Real(nrm[a]);
        } else {
          rw = (code == NrmCorner) ? bc_rho(n) : BC::density(fw, nrm, uw);
        }
        // BC::density() and bc_rho() both run on UN-SHIFTED populations, so
        // they already return the full density -- the same convention as
        // Collision::density() at a fluid node. Do not shift it again: doing so
        // made wall nodes report rho - 1 while their fluid neighbours reported
        // rho, an exact off-by-one across the boundary.
        rho(n) = rw;
        ux(n) = uw[0]; uy(n) = uw[1]; uz(n) = uw[2];
        return;
      }
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      Real f[Q];
      f[0] = acc.load_rest(nb);
      for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, f[i], f[i + 1]);
      const Macro m = coll.macroscopic(f, n);
      rho(n) = Collision::density(m);
      ux(n) = m.ux; uy(n) = m.uy; uz(n) = m.uz;
    });
    Kokkos::fence();
  }

  // axis < 0 -> total mass; otherwise total momentum along that axis.
  template <int P>
  Real reduce(int axis) const {
    const auto acc  = pop_.template access<P>();
    const auto coll = coll_;
    const Domain d  = dom_;
    auto flags = flags_;
    Real s = 0;
    Kokkos::parallel_reduce("reduce", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& sum) {
        if (flags(n) != Fluid && flags(n) != RegWall) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        Real f[Q];
        for (int i = 0; i < Q; ++i) f[i] = acc.load(nb, i);
        const Macro m = coll.macroscopic(f, n);
        const Real r = Collision::density(m);
        sum += (axis < 0) ? r
             : r * (axis == 0 ? m.ux : (axis == 1 ? m.uy : m.uz));
      }, s);
    return s;
  }

  template <int P, class Fn>
  void scatter(Fn fn) {
    const auto acc = pop_.template access<P>();
    const Domain d = dom_;
    Kokkos::parallel_for("scatter", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      for (int i = 0; i < Q; ++i) acc.scatter(nb, i, fn(n, i));
    });
    Kokkos::fence();
  }

  template <int P>
  void gather(View2D<Real> out) const {
    const auto acc = pop_.template access<P>();
    const Domain d = dom_;
    Kokkos::parallel_for("gather", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      for (int i = 0; i < Q; ++i) out(n, i) = acc.load(nb, i);
    });
    Kokkos::fence();
  }

  Domain dom_;
  Collision coll_;
  Streaming pop_;
  using BC = Regularized<L, typename EquilibriumOf<Collision, L>::type>;

  View1D<std::uint8_t> flags_;
  View1D<std::uint8_t> bc_nrm_, bc_ext_;
  // One entry per DISTINCT wall state, not per wall node -- but a profiled
  // inlet makes almost every node distinct, so this must not be a uint8_t.
  // At uint8_t the index wrapped silently at 256 states and nodes past that
  // took another node's velocity; a 321-wide profiled inlet was enough.
  View1D<std::uint16_t> bc_tag_;
  // For NrmOutFree: the node index this outflow node takes its velocity from.
  // A voxelised cap has no axis normal, so the donor is found per node from the
  // geometry rather than implied by a face code.
  View1D<Index>         bc_don_;
  // For NrmOutFree: the axis face code of the node's own outward normal. A
  // voxelised cap is a staircase of axis-aligned faces, so each node HAS an
  // exact axis normal even though the cap as a whole does not.
  View1D<std::uint8_t>  bc_onrm_;
  View1D<std::uint32_t> bc_unk_;
  View1D<Real>         bc_rho_;
  bool                 has_corners_ = false;
  bool                 fd_corners_  = has_shear_omega<Collision>;
  bool                 force_bc_    = true;    // see set_force_correction
  // Outflow tangential velocity: 1 = zero-gradient (first order), 2 = linear
  // extrapolation from two upstream nodes. See set_outflow_order.
  int                  out_order_   = 1;
  // Target density at NrmOutFree nodes. Pinning this to a constant makes the
  // outflow a mass source and sink: the nodes are overwritten every step, so
  // whatever rho is imposed there is injected regardless of what arrived. Left
  // as a variable so a driver can close the loop on TOTAL mass instead, which
  // is what makes the boundary conserving. See set_outflow_density.
  Real                 outflow_rho_ = Real(1);
 public:
  // The Guo-force corrections at regularised walls, both following from the
  // forced Chapman-Enskog expansion: the odd (first-moment) Hermite term the
  // reconstruction omits, and the half-space bias the bounce-back scaffold puts
  // into the measured Pi^(1). On by default.
  //
  // What remains after these is NOT a force effect. It is a curvature-induced
  // slip intrinsic to the regularisation,
  //     du_slip = -(2/3) (omega-1)/omega^2 * d2 u_par / dn^2,
  // independent of F to five digits over an 8x range, zero for a linear profile
  // (Couette is exact to 1e-15 at every tau), and zero at omega = 1. It is a
  // pre-existing property of truncating f^(1) at second Hermite order, not a
  // defect of the force treatment.
  void set_force_correction(bool on) { force_bc_ = on; }

  // Order of the tangential extrapolation at an outflow face. The imposed
  // density and the closure-derived normal velocity are second order either
  // way; this only controls the tangential component.
  //
  // MEASURED: order 2 is NOT better and the default stays 1. On the inlet-driven
  // Poiseuille of validation/poiseuille_inlet.cpp it was uniformly worse on the
  // coarse grids (Ly = 41: 1.388e-3 against 1.160e-3) and indistinguishable on
  // the finest (Ly = 641: 6.184e-6 against 6.192e-6). The order-limiting term
  // there is not the outlet at all -- see that file's header. Kept because it
  // is the natural thing to reach for and the negative result is worth being
  // able to reproduce.
  void set_outflow_order(int k) { out_order_ = k; }

  // Target density at arbitrary-face outflow nodes. A driver that wants a
  // conserving outlet adjusts this from the global mass error rather than
  // leaving it at rho0: raising it pushes mass back in, lowering it lets more
  // out, and the fixed point is the density at which inflow and outflow match.
  void set_outflow_density(Real r) { outflow_rho_ = r; }
  Real outflow_density() const { return outflow_rho_; }
 private:
  View2D<Real>         wall_u_;
  View2D<Real>         wall_u0_;   // as given; wall_u_ is this times the scale
  // Node lists. Sweeping the padded box costs the same whether a cell does work
  // or returns immediately, and on a real geometry most of it returns: the
  // aorta is 84% solid, and corner_density's two passes act only on the few
  // thousand RegWall nodes yet were paying for a full sweep each, twice a step.
  View1D<Index>        active_;      // cells stream_collide must visit
  View1D<Index>        walls_;       // RegWall cells only
  Index                n_active_ = 0, n_walls_ = 0;
  std::size_t          n_wall_states_ = 0;
  HostView1D<std::uint8_t> h_flags_;
  View1D<Real> rho_, ux_, uy_, uz_;
  std::size_t t_ = 0;
};

}  // namespace lbm
