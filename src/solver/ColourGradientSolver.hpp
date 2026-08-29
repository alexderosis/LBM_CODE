#pragma once
//==============================================================================
//  Two-component flow by the colour-gradient method: the driver.
//
//  Owns TWO population sets, f_i^r and f_i^b, and steps them together. See
//  ColourGradient.hpp for the model and for the paper it comes from; this file
//  is about the order the pieces run in, which is the part an implementation
//  gets wrong rather than the algebra.
//
//  THREE PASSES A STEP, and the split is forced by the storage scheme.
//
//    1. compute_fields()     gather both colours, form rho_r, rho_b, rho and u,
//                            write them to node fields. READ ONLY.
//    2. compute_gradients()  grad phi and grad rho from those node fields, by
//                            the isotropic stencil of Eq. (40).
//    3. step()               gather both, collide the sum, perturb, recolour,
//                            scatter both.
//
//  IT CANNOT BE ONE PASS, and this is the same constraint PhaseFieldSolver.hpp
//  documents. Esoteric Pull stores a node's populations across its neighbours'
//  slots, and during a fused collide-and-stream kernel those slots hold a
//  mixture of two time levels: a gradient assembled from neighbour POPULATIONS
//  mid-kernel is not a gradient of anything. Both gradients therefore come from
//  node fields written in a completed, fenced pass.
//
//  THAT MATTERS MORE HERE THAN IT DID FOR THE PHASE FIELD. grad phi is not a
//  diagnostic in this model -- it is the only thing holding the interface
//  together. It sets the direction the perturbation pushes, and it sets the
//  direction the recolouring moves colour. An interface held together by a
//  gradient computed from the wrong time level does not blur slowly; it detaches.
//
//  THE VELOCITY IS THE MIXTURE'S, Eq. (13), rho u = sum_i f_i c_i + F dt / 2,
//  with f_i the colour-blind sum and F the body force. Both colours are then
//  collided against that single velocity, which is what makes this a
//  one-collision scheme rather than two coupled ones.
//
//  WHAT THIS SOLVER DOES NOT DO.
//
//  WALLS ARE HALFWAY BOUNCE-BACK, AND FREE UNDER ESOTERIC PULL. A cell marked
//  Solid does exactly nothing -- no load, no store, no arithmetic -- because on
//  an in-place scheme every bounce-back write lands back in the slot it was read
//  from. Both colours get it at once, from the same branch.
//
//  Two things about that are easy to get wrong and are done deliberately here.
//  A wall cell OWNS LIVE STORAGE: the population a fluid node emits toward it
//  sits in the wall's slot for one step, which is what puts the wall half-way
//  between the two. So wall cells are seeded with the rest state like every
//  other cell, never left empty -- an empty wall layer soaks mass out of the
//  fluid and leaves the bulk low by O(1/H). And this solver sweeps the full
//  padded range rather than a compacted index list, so there is no list to keep
//  in step with the flags; FluidSolver's rebuild_lists() has no counterpart
//  here, and cannot be forgotten because it does not exist.
//
//  THE COLOUR GRADIENT AT A WALL IS NEUTRAL, AND THAT IS A CHOICE. A wall
//  neighbour contributes its own centre value to the stencil rather than a wall
//  value, which makes grad(phi) and grad(rho) tangential at a solid: zero normal
//  derivative, a ninety-degree contact angle, no wetting preference either way.
//  The colour-gradient literature sets a contact angle by prescribing the colour
//  gradient at the solid instead, and NONE of that is implemented. The source
//  paper is no help on this point -- its own walls are free-slip and it says
//  nothing about what the recolouring does at one -- so this is the neutral
//  default, chosen because it is the one that assumes least, not because it was
//  validated against anything.
//   * NO SHIFTED STORAGE. Two colours summing to rho leave nothing sensible to
//     shift by, so RawPopulations it is, and the FP32 accuracy argument that
//     motivates shifted storage elsewhere in this code does not apply.
//   * NO D3Q19. phi_i, B_i and sigma = 4 A tau / 9 are all derived for D3Q27 in
//     the source paper, and the operator static_asserts it.
//   * NO MASS-CONSERVING RECOLOURING GUARANTEE PER COLOUR. The partition
//     conserves the TOTAL exactly (f^r + f^b = f identically), but individual
//     colour masses drift by the amount the recolouring moves across the
//     interface. That is the method, not a defect, and validation/static_droplet
//     reports the drift rather than asserting it away.
//==============================================================================
#include "boundary/Flags.hpp"
#include "collision/ColourGradient.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"

namespace lbm {

template <class L, class Streaming, class Collision = ColourGradient<L>>
class ColourGradientSolver {
 public:
  using Lattice = L;
  static constexpr int Q  = L::Q;
  static constexpr int NF = Streaming::nb_first;
  static constexpr int NS = Streaming::nb_stride;

  ColourGradientSolver(const Domain& dom, const Collision& coll)
      : dom_(dom), coll_(coll), red_(dom), blue_(dom),
        rr_("rho_r", dom.n_padded), rb_("rho_b", dom.n_padded),
        phi_("phi", dom.n_padded),
        ux_("ux", dom.n_padded), uy_("uy", dom.n_padded), uz_("uz", dom.n_padded),
        gx_("gphi_x", dom.n_padded), gy_("gphi_y", dom.n_padded),
        gz_("gphi_z", dom.n_padded),
        drx_("grho_x", dom.n_padded), dry_("grho_y", dom.n_padded),
        drz_("grho_z", dom.n_padded),
        flags_("cg_flags", dom.n_padded) {
    h_flags_ = Kokkos::create_mirror_view(flags_);
    for (Index n = 0; n < dom.n_padded; ++n) {
      Index px, py, pz; dom_.coords(n, px, py, pz);
      h_flags_(n) = dom_.is_interior(px, py, pz) ? Fluid : Excluded;
    }
    Kokkos::deep_copy(flags_, h_flags_);
    coll_.phi = phi_;
    coll_.Gx = gx_;  coll_.Gy = gy_;  coll_.Gz = gz_;
    coll_.Rx = drx_; coll_.Ry = dry_; coll_.Rz = drz_;
  }

  //----------------------------------------------------------------------------
  // fn(n) -> {rho_r, rho_b}. Seeded at rest, which is what every case here
  // starts from and what f^eq(rho, 0) = rho phi_i makes trivial.
  //----------------------------------------------------------------------------
  struct Colours { Real red = 0, blue = 0; };

  //----------------------------------------------------------------------------
  // fn(x, y, z) -> CellType, over interior coordinates. Must be called BEFORE
  // initialize(), which reads the flags.
  //
  // There is no list to rebuild afterwards. This solver's kernels sweep the full
  // padded range and branch on the flag every time, so the flags ARE the
  // geometry -- there is no second, cached representation of it to fall out of
  // step. FluidSolver pays for a compacted list because the aorta is 84% solid;
  // nothing here is, and the trap that cost that solver a 15-21% error in the
  // Nusselt number is structurally absent rather than merely avoided.
  //----------------------------------------------------------------------------
  template <class Fn>
  void set_geometry(Fn fn) {
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x)
          h_flags_(dom_.id(x, y, z)) = fn(x, y, z);
    Kokkos::deep_copy(flags_, h_flags_);
  }

  View1D<std::uint8_t> flags() const { return flags_; }

  template <class Fn>
  void initialize(Fn fn) {
    const auto accr = red_.template access<0>();
    const auto accb = blue_.template access<0>();
    const auto coll = coll_;
    const Domain d  = dom_;
    auto rr = rr_, rb = rb_, phi = phi_;
    auto ux = ux_, uy = uy_, uz = uz_;
    Kokkos::parallel_for("cg_init", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const Colours c = fn(n);
      const Real rho = c.red + c.blue;
      const Real p = coll.order_parameter(c.red, c.blue);
      Real fr[Q], fb[Q];
      // Each colour seeded with its OWN alpha, so that the initial pressure
      // sum_k rho_k cs_k^2 is continuous through the interface.
      for (int i = 0; i < Q; ++i) {
        fr[i] = c.red  * Collision::phi_i(i, coll.alpha_r);
        fb[i] = c.blue * Collision::phi_i(i, coll.alpha_b);
      }
      (void)p;
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      accr.store_rest(nb, fr[0]);
      accb.store_rest(nb, fb[0]);
      for (int i = 1; i < Q; i += 2) {
        accr.store_pair(nb, i, fr[i], fr[i + 1]);
        accb.store_pair(nb, i, fb[i], fb[i + 1]);
      }
      rr(n) = c.red;  rb(n) = c.blue;  phi(n) = p;
      ux(n) = Real(0); uy(n) = Real(0); uz(n) = Real(0);
      (void)rho;
    });
    Kokkos::fence();
    refresh();
  }


  void refresh() {
    if (t_ % 2 == 0) fields_kernel<0>(); else fields_kernel<1>();
    compute_gradients();
  }

  void step() {
    if (t_ % 2 == 0) run_step<0>(); else run_step<1>();
    red_.end_of_step();
    blue_.end_of_step();
    ++t_;
  }

  View1D<Real> rho_red()  const { return rr_; }
  View1D<Real> rho_blue() const { return rb_; }
  View1D<Real> phi()      const { return phi_; }
  View1D<Real> ux() const { return ux_; }
  View1D<Real> uy() const { return uy_; }
  View1D<Real> uz() const { return uz_; }
  View1D<Real> grad_phi_x() const { return gx_; }
  const Domain& domain() const { return dom_; }
  std::size_t timestep() const { return t_; }

  // Total of each colour over every slot, walls included -- the same argument
  // ScalarSolver::total_population carries about in-flight populations.
  Real total_red()  const { return total(red_); }
  Real total_blue() const { return total(blue_); }

 public:
  //----------------------------------------------------------------------------
  // NVCC CONSTRAINT, NOT A DESIGN CHOICE -- same note as FluidSolver and
  // PhaseFieldSolver carry. Public only because CUDA forbids an extended
  // host-device lambda inside a private member function.
  //----------------------------------------------------------------------------
  template <class S>
  Real total(const S& pop) const {
    auto f = pop.storage();
    Real s = Real(0);
    Kokkos::parallel_reduce("cg_total",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
            {Index(f.extent(0)), Index(f.extent(1))}),
        KOKKOS_LAMBDA(Index n, Index i, Real& acc) { acc += f(n, i); }, s);
    return s;
  }

  // Pass 1: the macroscopic fields. Read only -- nothing is stored back, so it
  // is safe to run at the same parity the next step will collide at.
  template <int P>
  void fields_kernel() {
    const auto accr = red_.template access<P>();
    const auto accb = blue_.template access<P>();
    const auto coll = coll_;
    const Domain d  = dom_;
    auto rr = rr_, rb = rb_, phi = phi_;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto flags = flags_;
    Kokkos::parallel_for("cg_fields", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      // A wall cell's populations are in transit, not a state: summing them
      // gives a number that is not a density. The gradient stencil substitutes
      // for wall neighbours rather than reading these, so they are left alone.
      if (flags(n) != Fluid) return;
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      Real fr[Q], fb[Q];
      fr[0] = accr.load_rest(nb);
      fb[0] = accb.load_rest(nb);
      for (int i = 1; i < Q; i += 2) {
        accr.load_pair(nb, i, fr[i], fr[i + 1]);
        accb.load_pair(nb, i, fb[i], fb[i + 1]);
      }
      Real sr = Real(0), sb = Real(0), m[3] = {Real(0), Real(0), Real(0)};
      for (int i = 0; i < Q; ++i) {
        sr += fr[i];  sb += fb[i];
        const Real t = fr[i] + fb[i];
        for (int a = 0; a < 3; ++a) m[a] += t * Real(cvel<L>(i, a));
      }
      const Real rho = sr + sb;
      const Real inv = (rho > Real(0)) ? Real(1) / rho : Real(0);
      // Eq. (13): the half body force is part of the velocity's definition.
      const Real fw = rho - coll.rho_ref;
      ux(n) = (m[0] + Real(0.5) * fw * coll.bx) * inv;
      uy(n) = (m[1] + Real(0.5) * fw * coll.by) * inv;
      uz(n) = (m[2] + Real(0.5) * fw * coll.bz) * inv;
      rr(n) = sr;  rb(n) = sb;
      phi(n) = coll.order_parameter(sr, sb);
    });
    Kokkos::fence();
  }

  // Pass 2: grad phi and grad rho, Eq. (40). One gather serves both.
  void compute_gradients() {
    const Domain d = dom_;
    auto rr = rr_, rb = rb_, phi = phi_;
    auto gx = gx_, gy = gy_, gz = gz_;
    auto rx = drx_, ry = dry_, rz = drz_;
    auto flags = flags_;
    constexpr Real ics = inv_cs2<L, Real>();
    Kokkos::parallel_for("cg_gradients", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (flags(n) != Fluid) {
        gx(n) = Real(0); gy(n) = Real(0); gz(n) = Real(0);
        rx(n) = Real(0); ry(n) = Real(0); rz(n) = Real(0);
        return;
      }
      Neighbours<L> nb;
      d.template fill_neighbours<L, 1, 1>(n, nb);
      const Real p0 = phi(n), r0 = rr(n) + rb(n);
      Real gp[3] = {Real(0), Real(0), Real(0)};
      Real gr[3] = {Real(0), Real(0), Real(0)};
      for (int i = 1; i < Q; ++i) {
        const Real w = weight<L, Real>(i);
        const Index j = nb.j[i];
        // A non-fluid neighbour contributes THIS node's value, which is a zero
        // normal derivative at the wall -- neutral wetting. See the banner.
        const bool wet = (flags(j) == Fluid);
        const Real pj = wet ? phi(j) : p0;
        const Real rj = wet ? (rr(j) + rb(j)) : r0;
        for (int a = 0; a < 3; ++a) {
          const Real c = Real(cvel<L>(i, a));
          gp[a] += w * pj * c;
          gr[a] += w * rj * c;
        }
      }
      gx(n) = ics * gp[0];  gy(n) = ics * gp[1];  gz(n) = ics * gp[2];
      rx(n) = ics * gr[0];  ry(n) = ics * gr[1];  rz(n) = ics * gr[2];
    });
    Kokkos::fence();
  }

  // Pass 3: collide the sum, perturb, recolour, scatter both.
  template <int P>
  void run_step() {
    const auto accr = red_.template access<P>();
    const auto accb = blue_.template access<P>();
    const auto coll = coll_;
    const Domain d  = dom_;
    auto rr = rr_, rb = rb_, phi = phi_;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto flags = flags_;
    Kokkos::parallel_for("cg_stream_collide", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        // Esoteric Pull's bounce-back is the identity on the storage, so a solid
        // cell is skipped outright. Doing anything at all here -- zeroing it,
        // writing equilibrium into it -- would destroy the populations it is
        // holding in transit for its fluid neighbours.
        if (flags(n) != Fluid) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        Real fr[Q], fb[Q];
        fr[0] = accr.load_rest(nb);
        fb[0] = accb.load_rest(nb);
        for (int i = 1; i < Q; i += 2) {
          accr.load_pair(nb, i, fr[i], fr[i + 1]);
          accb.load_pair(nb, i, fb[i], fb[i + 1]);
        }
        Real f[Q];
        for (int i = 0; i < Q; ++i) f[i] = fr[i] + fb[i];

        const Real srr = rr(n), srb = rb(n);
        const Real u[3] = {ux(n), uy(n), uz(n)};
        const Real p = phi(n);

        coll.collide(f, srr, srb, u, p, n);
        coll.recolour(f, srr, srb, p, n, fr, fb);

        accr.store_rest(nb, fr[0]);
        accb.store_rest(nb, fb[0]);
        for (int i = 1; i < Q; i += 2) {
          accr.store_pair(nb, i, fr[i], fr[i + 1]);
          accb.store_pair(nb, i, fb[i], fb[i + 1]);
        }
      });
  }

 private:
  Domain dom_;
  Collision coll_;
  Streaming red_, blue_;
  View1D<Real> rr_, rb_, phi_;
  View1D<Real> ux_, uy_, uz_;
  View1D<Real> gx_, gy_, gz_;
  View1D<Real> drx_, dry_, drz_;
  View1D<std::uint8_t> flags_;
  HostView1D<std::uint8_t> h_flags_;
  std::size_t t_ = 0;
};

}  // namespace lbm
