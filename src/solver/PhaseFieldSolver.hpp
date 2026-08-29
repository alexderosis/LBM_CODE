#pragma once
//==============================================================================
//  Phase-field solver: a THIRD distribution set, carrying the interface.
//
//  Structurally this is ScalarSolver with two additions -- an anti-diffusion
//  source in the collision, and a gradient field that both the source and the
//  fluid's capillary stress read. Everything else is the same shape: its own
//  lattice, its own streaming scheme, velocity as an input, phi = sum_i h_i.
//  Nothing in FluidSolver, the lattices, the streaming schemes or the moment
//  operators had to change to accommodate it, which is the third time that has
//  now held.
//
//  COUPLING ORDER -- read this before writing a driver.
//
//  phi and grad phi must be refreshed BEFORE the fluid steps, so the fluid
//  collides against the capillary stress at time t rather than t-1. This is the
//  same first-order splitting error MagneticSolver documents, and it does not
//  refine away: under diffusive scaling the ratio (omega^2 dt) / (nu k^2) is
//  independent of N. On the Alfven wave it cost 3% of the damping rate. Here it
//  is worse in kind rather than in size -- the interface MOVES, so the error is
//  a systematically misplaced interface, not a damping offset. The order is
//
//      pf.refresh();          // phi(t) and grad phi(t)
//      fluid.step(true);      // collides against grad phi(t), writes u(t)
//      pf.step();             // advects with u(t)
//
//  One refresh serves both consumers, and that is not an accident: the fluid's
//  capillary stress and the phase field's interface normal are the same
//  gradient, so evaluating them from one array at one time is what keeps them
//  consistent. Evaluating them separately would let the stress and the normal
//  disagree about where the interface is.
//
//  THE GRADIENT IS A SEPARATE PASS, AND HAS TO BE.
//
//  grad phi is formed from the phi FIELD by a lattice-weighted stencil, never
//  from neighbouring populations inside the fused kernel. Reading a neighbour's
//  populations there is a genuine race under Esoteric Pull -- the two slots a
//  node reads are exactly the two it writes -- which is the same reason the
//  scalar's outflow condition needs a second pass. compute_field() ends in a
//  fence and compute_gradient() reads only what it wrote.
//
//  The stencils are De Rosis & Enan Eqs. (20) and (21),
//
//      grad phi (x) = (1/cs2)  sum_i w_i c_i [phi(x + c_i)],
//      lap  phi (x) = (2/cs2)  sum_i w_i     [phi(x + c_i) - phi(x)],
//
//  taken on the GRADIENT LATTICE, which defaults to the full Navier-Stokes
//  lattice (D2Q9 / D3Q27) rather than the phase field's own D2Q5 / D3Q7. That is
//  deliberate: spurious currents around a static droplet are dominated by the
//  isotropy of this stencil, and the axis-only one is markedly worse. It costs a
//  27-neighbour gather once per node per step, in a kernel that does nothing
//  else.
//
//  BOTH COME FROM THE SAME GATHER. The Laplacian needs exactly the neighbours
//  the gradient has already loaded, so it is one extra accumulator and one extra
//  store in the same kernel -- no second pass, no second neighbour list. Only
//  the potential-form fluid operator reads it (the capillary stress form needs
//  no second derivative at all), but it is cheap enough not to be worth making
//  conditional.
//
//  MEMORY. Five extra Real fields per node -- phi, three gradient components and
//  the Laplacian -- on top of the Q populations. In FP32 on D3Q7 that is 28
//  bytes of populations against 20 of fields, so the fields are not a rounding
//  error in the bandwidth budget; they are the price of not differentiating
//  populations.
//
//  NOT IMPLEMENTED, so that nobody looks for it:
//
//   * DENSITY AND VISCOSITY CONTRAST. This solver pairs with MultiphaseBGK,
//     which is matched-density: phi enters the flow only through the capillary
//     stress. A density ratio needs the fluid distribution to carry pressure
//     rather than density -- ShiftedPopulations assumes rho_ref is exactly 1
//     (see Storage.hpp), and at a ratio of 10 the stored g_i are no longer
//     small, so the shift stops buying anything. A viscosity contrast is
//     closer: MultiphaseBGK::omega_n takes a per-node rate already.
//   * WETTING. PhaseWall is zero-flux on the populations, which is right for
//     the transport, but it sets no contact angle: that needs a condition on
//     grad phi at the wall, and the gradient stencil currently reads whatever
//     phi the wall node holds. Do not put an interface against a wall yet.
//   * OPEN BOUNDARIES for phi. The scalar's donor machinery would port, with
//     the same second pass and the same fence.
//==============================================================================
#include "collision/PhaseFieldBGK.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Which lattice the gradient stencil runs on. Isotropy of this stencil is what
// sets the spurious-current floor, so the default is the richest velocity set
// of the same dimension, not the phase field's own.
//------------------------------------------------------------------------------
template <class L> struct GradientLatticeOf;
template <> struct GradientLatticeOf<D2Q5>  { using type = D2Q9;  };
template <> struct GradientLatticeOf<D2Q9>  { using type = D2Q9;  };
template <> struct GradientLatticeOf<D3Q7>  { using type = D3Q27; };
template <> struct GradientLatticeOf<D3Q19> { using type = D3Q19; };
template <> struct GradientLatticeOf<D3Q27> { using type = D3Q27; };

template <class L, class Streaming, class Collision,
          class GL = typename GradientLatticeOf<L>::type>
class PhaseFieldSolver {
 public:
  using Lattice        = L;
  using GradientLattice = GL;
  static constexpr int Q  = L::Q;
  static constexpr int NF = Streaming::nb_first;
  static constexpr int NS = Streaming::nb_stride;
  static_assert(int(GL::D) == int(L::D),
                "the gradient lattice must have the same dimension as the phase "
                "field's own.");

  PhaseFieldSolver(const Domain& dom, const Collision& coll)
      : dom_(dom), coll_(coll), pop_(dom),
        flags_("pflags", dom.n_padded),
        phi_("phi", dom.n_padded),
        gx_("gradphi_x", dom.n_padded),
        gy_("gradphi_y", dom.n_padded),
        gz_("gradphi_z", dom.n_padded),
        lap_("lapphi", dom.n_padded) {
    h_flags_ = Kokkos::create_mirror_view(flags_);
    for (Index n = 0; n < dom.n_padded; ++n) {
      Index px, py, pz; dom_.coords(n, px, py, pz);
      h_flags_(n) = dom_.is_interior(px, py, pz) ? PhaseBulk : PhaseExcluded;
    }
    Kokkos::deep_copy(flags_, h_flags_);
  }

  // fn(x, y, z) -> PhaseCell, over interior coordinates.
  template <class Fn>
  void set_geometry(Fn fn) {
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x)
          h_flags_(dom_.id(x, y, z)) = fn(x, y, z);
    Kokkos::deep_copy(flags_, h_flags_);
  }

  // The velocity field that advects the interface -- owned by the fluid solver.
  void set_velocity(View1D<Real> ux, View1D<Real> uy, View1D<Real> uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  void initialize(Real phi0) {
    initialize_field(KOKKOS_LAMBDA(Index) { return phi0; });
  }

  //----------------------------------------------------------------------------
  // fn(n) -> Real, device-callable. Seeds the populations at equilibrium with
  // u = 0, fills phi, and forms the gradient -- so the solver is consistent
  // before the first step and a driver can read grad_x() immediately.
  //----------------------------------------------------------------------------
  template <class Fn>
  void initialize_field(Fn fn) {
    t_ = 0;
    const auto acc = pop_.template access<0>();
    const auto coll = coll_;
    const Domain d = dom_;
    auto flags = flags_; auto phi = phi_;
    Kokkos::parallel_for("phase_init", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      const Real p = (flags(n) == PhaseExcluded) ? Real(0) : fn(n);
      phi(n) = p;
      for (int i = 0; i < Q; ++i)
        acc.scatter(nb, i, coll.eq(i, p, Real(0), Real(0), Real(0)));
    });
    Kokkos::fence();
    compute_derivatives();
  }

  //----------------------------------------------------------------------------
  // phi(t) and grad phi(t). Call this BEFORE stepping the fluid; see the header.
  //----------------------------------------------------------------------------
  void refresh() {
    compute_field();      // ends in a fence
    compute_derivatives();
  }

  void step() {
    if (t_ % 2 == 0) run_step<0>(); else run_step<1>();
    pop_.end_of_step();
    ++t_;
  }

  void compute_field() {
    if (t_ % 2 == 0) field_kernel<0>(); else field_kernel<1>();
  }

  View1D<Real> phi() const { return phi_; }
  View1D<Real> grad_x() const { return gx_; }
  View1D<Real> grad_y() const { return gy_; }
  View1D<Real> grad_z() const { return gz_; }
  View1D<Real> laplacian() const { return lap_; }
  View1D<std::uint8_t> flags() const { return flags_; }
  const Domain& domain() const { return dom_; }
  std::size_t timestep() const { return t_; }

  //----------------------------------------------------------------------------
  // Every population in the lattice, wall slots included -- the conserved
  // quantity, and a sharp check: the Allen-Cahn form conserves phi EXACTLY,
  // since sum_i S_i = 0 and both streaming and collision are conservative.
  //
  // This is not the same number as summing phi() over the bulk, and the
  // difference is not a leak: under an in-place scheme a population in flight
  // toward a wall spends a step in a slot the wall owns. Same argument as
  // ScalarSolver::total_population; see validation/scalar_mass.cpp.
  //----------------------------------------------------------------------------
  Real total_population() const {
    auto f = pop_.storage();
    Real s = Real(0);
    Kokkos::parallel_reduce(
        "phase_total",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
            {Index(f.extent(0)), Index(f.extent(1))}),
        KOKKOS_LAMBDA(Index n, Index i, Real& acc) { acc += f(n, i); }, s);
    return s;
  }

 public:
  //----------------------------------------------------------------------------
  // NVCC CONSTRAINT, NOT A DESIGN CHOICE. Everything from here to the data
  // members is morally private and public only because CUDA forbids an extended
  // __host__ __device__ lambda inside a private member function. Same note as
  // FluidSolver and ScalarSolver carry, same reason.
  //
  // Do not call these from outside: they assume parity and fence state that only
  // the public methods maintain.
  //----------------------------------------------------------------------------
  template <int P>
  void run_step() {
    const auto acc  = pop_.template access<P>();
    const auto coll = coll_;
    const Domain d  = dom_;
    auto flags = flags_; auto phi = phi_;
    auto gx = gx_, gy = gy_, gz = gz_;
    auto ux = ux_, uy = uy_, uz = uz_;
    const bool have_u = ux.data() != nullptr;

    Kokkos::parallel_for("phase_stream_collide", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        const std::uint8_t flag = flags(n);
        if (flag == PhaseExcluded) return;
        // Bounce-back is the identity on an in-place scheme, so a zero-flux
        // wall needs no work at all there.
        if constexpr (Streaming::implicit_bounce_back)
          if (flag == PhaseWall) return;

        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        Real h[Q];
        h[0] = acc.load_rest(nb);
        for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, h[i], h[i + 1]);

        if (flag == PhaseWall) {          // only reached on an explicit scheme
          acc.store_rest(nb, h[0]);
          for (int i = 1; i < Q; i += 2) acc.store_pair(nb, i, h[i + 1], h[i]);
          return;
        }

        const Real p = Collision::order_parameter(h);
        const Real G[3] = {gx(n), gy(n), gz(n)};
        Real A[3];
        coll.anti_diffusion(p, G, A);
        const Real u[3] = {have_u ? ux(n) : Real(0),
                           have_u ? uy(n) : Real(0),
                           have_u ? uz(n) : Real(0)};
        coll.collide(h, p, u, A);
        phi(n) = p;
        acc.store_rest(nb, h[0]);
        for (int i = 1; i < Q; i += 2) acc.store_pair(nb, i, h[i], h[i + 1]);
      });
  }

  template <int P>
  void field_kernel() {
    const auto acc = pop_.template access<P>();
    const Domain d = dom_;
    auto flags = flags_; auto phi = phi_;
    Kokkos::parallel_for("phase_field", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t flag = flags(n);
      if (flag == PhaseExcluded) { phi(n) = Real(0); return; }
      // A zero-flux wall never collides, so its slots hold whatever a
      // neighbour emitted toward it one step ago. That sum is not an order
      // parameter, and the gradient stencil reading it is exactly the gap the
      // header lists under WETTING -- so it is left at its initialised value
      // rather than being given a meaningless one.
      if (flag == PhaseWall) return;
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      Real h[Q];
      h[0] = acc.load_rest(nb);
      for (int i = 1; i < Q; i += 2) acc.load_pair(nb, i, h[i], h[i + 1]);
      phi(n) = Collision::order_parameter(h);
    });
    Kokkos::fence();
  }

  //----------------------------------------------------------------------------
  // grad phi and lap phi, on the gradient lattice, from ONE neighbour gather.
  //
  // Reads phi and writes only the derivative arrays, so it races with nothing.
  // The neighbour list is the full Q here, not the odd half: this is not a
  // streaming pass and every direction contributes.
  //
  // The i = 0 term contributes nothing to either -- c_0 is zero, and the
  // Laplacian's summand is phi(x) - phi(x), so the loop starts at 1 for both.
  //----------------------------------------------------------------------------
  void compute_derivatives() {
    const Domain d = dom_;
    auto flags = flags_; auto phi = phi_;
    auto gx = gx_, gy = gy_, gz = gz_, lap = lap_;
    constexpr Real icsg = inv_cs2<GL, Real>();
    Kokkos::parallel_for("phase_derivatives", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (flags(n) == PhaseExcluded) {
        gx(n) = Real(0); gy(n) = Real(0); gz(n) = Real(0); lap(n) = Real(0);
        return;
      }
      Neighbours<GL> nb;
      d.template fill_neighbours<GL, 1, 1>(n, nb);
      const Real p0 = phi(n);
      Real g[3] = {Real(0), Real(0), Real(0)}, l = Real(0);
      for (int i = 1; i < GL::Q; ++i) {
        const Real w = weight<GL, Real>(i);
        const Real pn = phi(nb.j[i]);
        const Real wp = w * pn;
        for (int a = 0; a < GL::D; ++a) g[a] += wp * Real(cvel<GL>(i, a));
        l += w * (pn - p0);
      }
      gx(n) = icsg * g[0];  gy(n) = icsg * g[1];  gz(n) = icsg * g[2];
      lap(n) = Real(2) * icsg * l;
    });
    Kokkos::fence();
  }

 private:
  Domain dom_;
  Collision coll_;
  Streaming pop_;
  View1D<std::uint8_t> flags_;
  HostView1D<std::uint8_t> h_flags_;
  View1D<Real> phi_, gx_, gy_, gz_, lap_;
  View1D<Real> ux_, uy_, uz_;
  std::size_t t_ = 0;
};

}  // namespace lbm
