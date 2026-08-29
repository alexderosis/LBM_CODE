//==============================================================================
//  Layered Poiseuille flow -- two immiscible fluids in a channel, exact.
//
//  THE CASE THAT WAS MISSING. ViscousInterfaceForce has been in the tree since
//  the pressure form landed, implemented and unexercised: it is identically zero
//  in a static droplet, and the Rayleigh-Taylor and water-entry cases that do
//  excite it have no exact answer to check against. This is the one that does.
//
//  Two layers between parallel plates, driven along the channel by a uniform
//  force G per unit volume, with a flat interface at the centreline. Fluid 1
//  (mu_1, rho_1) below, fluid 2 (mu_2, rho_2) above, no-slip at y = +-h.
//
//  IT IS AN EXACT SOLUTION AT ANY DENSITY RATIO, and that is what makes it
//  valuable here. The flow is fully developed -- u = u(y) x-hat with du/dx = 0 --
//  so the nonlinear term vanishes IDENTICALLY rather than approximately, and the
//  steady profile solves the full Navier-Stokes for any rho_1, rho_2:
//
//      u_k(y) = -G y^2 / (2 mu_k) + a_k y + c,
//      a_1 = (G h / 2)(mu_1 - mu_2) / (mu_1 (mu_1 + mu_2)),   a_2 = a_1 mu_1/mu_2,
//      c   = G h^2 / (mu_1 + mu_2),
//
//  from u(+-h) = 0, continuity of u, and continuity of the SHEAR STRESS
//  mu du/dy at the interface. The interface velocity c is the cleanest number in
//  the problem: one closed form, no fitting, and it depends on both viscosities.
//
//  WHAT EACH INGREDIENT IS ON THE HOOK FOR:
//
//    mu(phi)      the kink in du/dy at the interface. Get the interpolation
//                 wrong and the two parabolas join at the wrong slope ratio.
//    F_nu         the viscous interface force, nu (grad u + grad u^T) . grad rho.
//                 It is proportional to grad rho, so it is active here and ONLY
//                 here among the cases with exact answers -- a viscosity ratio
//                 alone would leave it zero. -novisc switches it off, which is
//                 the measurement that says whether it was needed.
//    rho(phi)     enters through F_nu and through the forcing prefactor, but NOT
//                 through the answer: the exact profile is density-independent.
//                 So a density ratio changes what the code has to do while
//                 leaving what it has to produce fixed, which is exactly the
//                 kind of test that catches a term rather than a constant.
//
//  AN OPEN ISSUE THIS CASE FOUND, stated because it is not fixed. Turning the
//  two contrasts on one at a time gives
//
//      mu ratio  rho ratio   L2 at H = 32, 64, 128       order
//      1         1           2.7e-04  6.7e-05  1.6e-05   2.00, 2.08
//      5         1           3.6e-02  2.4e-02  1.3e-02   0.62, 0.82
//      1         10          2.5e-02  2.9e-02  2.4e-02   NONE
//
//  The first row says the base scheme is exact and second order. The second is
//  the diffuse interface smoothing a kink in du/dy over W cells, which is first
//  order in W/h and converges as it should. The third should not happen at all:
//  the exact profile does not contain rho, so a density ratio alone must leave
//  the answer unchanged, and instead it leaves a few per cent that refinement
//  does not remove.
//
//  It is therefore something on the grad-rho path. F_nu = nu (grad u + grad u^T)
//  . grad rho is the obvious suspect and was checked: the LBE with this
//  equilibrium recovers div(nu S) rather than (1/rho) div(mu S), and the
//  difference is exactly nu S . grad rho, so the implemented form is right.
//  Replacing it with (grad mu) . S -- which is what the difference would be if
//  the LBE recovered the other one -- makes the error LARGER (1.8e-01 at H = 32)
//  while making it converge, which says the current term is cancelling part of
//  whatever the real defect is. Not resolved.
//
//  The driving force is applied through the operator's EXTERNAL force slot, as a
//  uniform G, rather than through the body-force acceleration b -- F_b is rho b,
//  which at a density ratio would drive the two layers unequally and give a
//  different (and much less clean) exact solution.
//==============================================================================
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/MultiphasePotentialBGK.hpp"
#include "collision/PhaseFieldBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/PhaseFieldSolver.hpp"
#include "solver/ViscousInterfaceForce.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace lbm;

using FL = D2Q9;
using PL = D2Q9;
using BgkColl = MultiphasePotentialBGK<FL, SecondOrderPhi<FL>, RawPopulations>;
using CmColl  = MultiphaseCentralMoments<FL>;
using PColl   = PhaseFieldBGK<PL>;
using PhaseSlv = PhaseFieldSolver<PL, EsotericPull<PL>, PColl>;

//------------------------------------------------------------------------------
// The exact profile, with the interface at an ARBITRARY height s rather than at
// the centreline.
//
// It has to be general in s because the diffuse interface does not stay exactly
// where it is placed: with a viscosity contrast the equilibrium phi profile is
// slightly skewed by the shear, so its 0.5 crossing settles a fraction of a cell
// off centre while the conserved integral of phi is unchanged. Measured at
// +0.28 cells, and NOT shrinking with resolution. Comparing against a
// centreline solution therefore measures where the interface ended up rather
// than whether the profile around it is right, and those are separate questions
// that deserve separate answers.
//
// Layer 1 spans [-h, s] and layer 2 [s, h], so with h1 = h + s, h2 = h - s:
//     a1 = G (h2^2 mu1 - h1^2 mu2) / (2 mu1 (mu2 h1 + mu1 h2)),  a2 = a1 mu1/mu2,
//     c  = G h1^2 / (2 mu1) + a1 h1,
// which reduces to the symmetric form at s = 0.
//------------------------------------------------------------------------------
struct Exact {
  double mu1, mu2, G, h, s = 0.0;
  KOKKOS_INLINE_FUNCTION double h1() const { return h + s; }
  KOKKOS_INLINE_FUNCTION double h2() const { return h - s; }
  KOKKOS_INLINE_FUNCTION double a1() const {
    return G * (h2() * h2() * mu1 - h1() * h1() * mu2) /
           (2 * mu1 * (mu2 * h1() + mu1 * h2()));
  }
  KOKKOS_INLINE_FUNCTION double c() const {
    return G * h1() * h1() / (2 * mu1) + a1() * h1();
  }
  KOKKOS_INLINE_FUNCTION
  double u(double y) const {                    // y in [-h, h]
    const double xi = y - s;
    return (xi < 0.0) ? (-G * xi * xi / (2 * mu1) + a1() * xi + c())
                      : (-G * xi * xi / (2 * mu2) + a1() * (mu1 / mu2) * xi + c());
  }
  KOKKOS_INLINE_FUNCTION double interface_u() const { return c(); }
};

struct Result {
  double l2 = 0, u_iface = 0, u_iface_err = 0, umax = 0;
  double phi_drift = 0;
  // Diagnostics for a case that is not converging: where the interface actually
  // ended up, how much cross-channel velocity there is (there should be none in
  // a parallel flow), and how steady the profile really was when we stopped.
  double y_iface = 0, vmax = 0, residual = 0, u_iface_ref = 0;
  std::size_t steps = 0;
  bool ok = false;
};

//------------------------------------------------------------------------------
template <class FColl>
static Result run(Index H, double mu1, double mu2, double rho1, double rho2,
                  double umax_target, double iw, double M, bool use_visc,
                  std::size_t max_steps) {
  using FluidSlv = FluidSolver<FL, EsotericPull<FL>, FColl>;

  const Index nx = 16, ny = H + 2;              // solid rows at 0 and ny-1
  const double h = 0.5 * double(H);
  // Choose G so the interface velocity hits the target, keeping Mach fixed as
  // the grid refines -- otherwise a finer grid is also a faster flow and the
  // fitted order picks up the Mach error as well as the discretisation.
  const double G = umax_target * (mu1 + mu2) / (h * h);
  const Exact ex{mu1, mu2, G, h};

  Domain d(nx, ny, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  PColl pc;
  pc.omega = PColl::omega_from_mobility(Real(M));
  pc.width = Real(iw);
  PhaseSlv pf(d, pc);

  const Index hxp = d.hx, hyp = d.hy;
  const Real iwr = Real(iw), hh = Real(h);
  // phi = 1 BELOW the centreline: fluid 1 is the lower layer.
  pf.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real Y = Real(py - hyp) - Real(0.5) - hh;      // -h at the bottom wall
    return Real(0.5) * (Real(1) + Kokkos::tanh(Real(-2) * Y / iwr));
  });

  ViscousInterfaceForce<FL> vf(d);

  // The uniform driving force, in the external slot.
  View1D<Real> gx("gx", d.n_padded), gy("gy", d.n_padded), gz("gz", d.n_padded);
  Kokkos::deep_copy(gx, Real(G));

  FColl fc;
  fc.phi = pf.phi();
  fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
  fc.Lap = pf.laplacian();
  if (use_visc) { fc.Vx = vf.x(); fc.Vy = vf.y(); fc.Vz = vf.z(); }
  fc.Ex = gx;  fc.Ey = gy;  fc.Ez = gz;
  fc.rho_L = Real(rho2);  fc.rho_H = Real(rho1);        // phi = 1 is fluid 1
  fc.mu_L  = Real(mu2);   fc.mu_H  = Real(mu1);
  fc.kappa = Real(0);  fc.beta = Real(0);               // no surface tension: flat
  FluidSlv fl(d, fc);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });
  // SEEDED WITH THE EXACT PROFILE, not with rest. The relaxation to steady state
  // is diffusive, so it takes H^2/nu steps -- 330,000 at H = 128, which is more
  // than a plausible run and left the first version of this case reporting an
  // unconverged answer as a converged one. Starting from the analytic solution
  // leaves only the discretisation difference to relax, which is quick.
  //
  // This does not beg the question: the scheme relaxes to ITS OWN steady state,
  // and the error being measured is the distance between that and the exact one.
  // The steady-state detector below still has to fire, and the step count is
  // printed so a run that simply never moved is visible.
  {
    const Exact e2 = ex;
    const Real hh2 = Real(h);
    fl.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real Y = Real(py - hyp) - Real(0.5) - hh2;
      return FlowState{Real(0), Real(e2.u(double(Y))), Real(0), Real(0)};
    });
  }

  pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
  vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());

  const double phi0 = double(pf.total_population());

  //--- march to steady state, detected rather than assumed -------------------
  Result r;
  std::vector<double> prev(std::size_t(ny), 0.0);
  const std::size_t check = 500;
  for (std::size_t step = 0; step < max_steps; ++step) {
    pf.refresh();
    fl.compute_macroscopic();
    if (use_visc) vf.refresh(fc);
    fl.step(true);
    pf.step();

    if ((step + 1) % check == 0) {
      fl.compute_macroscopic();
      auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      double dmax = 0, umx = 0;
      for (Index y = 1; y < ny - 1; ++y) {
        const double u = double(hu(d.id(nx / 2, y)));
        if (!std::isfinite(u)) return r;
        dmax = std::max(dmax, std::abs(u - prev[std::size_t(y)]));
        umx = std::max(umx, std::abs(u));
        prev[std::size_t(y)] = u;
      }
      r.steps = step + 1;
      r.residual = (umx > 0) ? dmax / umx : 0.0;
      if (r.residual < 1e-10) break;                     // steady
    }
  }

  //--- measure ---------------------------------------------------------------
  // Where the interface actually is, BEFORE scoring anything against it.
  fl.compute_macroscopic();
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hp0 = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
  for (Index y = 1; y < ny - 2; ++y) {
    const double a = double(hp0(d.id(nx / 2, y))), b = double(hp0(d.id(nx / 2, y + 1)));
    if ((a - 0.5) * (b - 0.5) <= 0 && a != b)
      r.y_iface = (double(y) + (a - 0.5) / (a - b)) - 0.5 - h;
  }
  Exact at_measured = ex;
  at_measured.s = r.y_iface;

  double num = 0, den = 0;
  for (Index y = 1; y < ny - 1; ++y) {
    const double Y = double(y) - 0.5 - h;
    const double u = double(hu(d.id(nx / 2, y))), e = at_measured.u(Y);
    num += (u - e) * (u - e);
    den += e * e;
    r.umax = std::max(r.umax, std::abs(u));
  }
  r.l2 = std::sqrt(num / den);
  r.u_iface_ref = at_measured.interface_u();

  // Interface velocity: the interface sits at a half-integer y, so the two
  // straddling nodes are averaged rather than one of them being called the
  // centre. Their mean is second-order accurate there and unbiased.
  // Interpolated AT the measured interface, not at the nominal centreline.
  {
    const double yreal = r.y_iface + 0.5 + h;            // back to node coordinates
    const Index y0 = Index(std::floor(yreal));
    const double f = yreal - double(y0);
    const double ua = double(hu(d.id(nx / 2, y0))), ub = double(hu(d.id(nx / 2, y0 + 1)));
    r.u_iface = ua + f * (ub - ua);
  }
  r.u_iface_err = std::abs(r.u_iface / r.u_iface_ref - 1.0);
  r.phi_drift = std::abs(double(pf.total_population()) - phi0) / std::abs(phi0);

  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  for (Index y = 1; y < ny - 1; ++y)
    r.vmax = std::max(r.vmax, std::abs(double(hv(d.id(nx / 2, y)))));
  r.ok = std::isfinite(r.l2);
  return r;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    double umax = 0.02, iw = 4.0, M = 0.02;
    bool use_cm = true, fine = false;
    std::size_t max_steps = 120000;
    for (int i = 1; i < argc; ++i) {
      auto nx = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-u"))    nx(umax);
      else if (!std::strcmp(argv[i], "-iw"))   nx(iw);
      else if (!std::strcmp(argv[i], "-m"))    nx(M);
      else if (!std::strcmp(argv[i], "-fine")) fine = true;
      else if (!std::strcmp(argv[i], "-op")) {
        if (i + 1 < argc) use_cm = !std::strcmp(argv[++i], "cm");
      }
    }
    const double mu2 = 0.1, rho2 = 1.0;

    std::printf("Layered Poiseuille flow   D2Q9 (pressure form, %s) + D2Q9 phase field\n",
                use_cm ? "central moments" : "BGK");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    auto go = [&](Index H, double mr, double rr, bool visc) {
      return use_cm
        ? run<CmColl>(H, mu2 * mr, mu2, rho2 * rr, rho2, umax, iw, M, visc, max_steps)
        : run<BgkColl>(H, mu2 * mr, mu2, rho2 * rr, rho2, umax, iw, M, visc, max_steps);
    };

    std::vector<Index> Hs = {32, 64};
    if (fine) Hs.push_back(128);

    //------------------------------------------------------------------------
    // The two contrasts are turned on ONE AT A TIME. That is what makes this
    // table diagnostic rather than merely a score: the exact profile depends on
    // mu and not at all on rho, so a density ratio alone must not change the
    // answer, and if it does the fault has to live in a term proportional to
    // grad rho. It does, and this is how it was found.
    //------------------------------------------------------------------------
    struct Cfg { const char* name; double mr, rr; };
    const Cfg cfgs[4] = {{"single phase   ", 1.0,  1.0},
                         {"viscosity only ", 5.0,  1.0},
                         {"density only   ", 1.0, 10.0},
                         {"both           ", 5.0, 10.0}};

    std::printf("%-16s %-6s %-12s %-8s %-11s %-10s %-9s\n",
                "configuration", "H", "L2(u)", "order", "u_iface err", "y_iface", "steps");
    std::printf("%s\n", std::string(80, '-').c_str());

    double ctrl_ord = 0, dens_ord = 0;
    bool finite = true;
    std::vector<Result> both;
    for (int ci = 0; ci < 4; ++ci) {
      std::vector<Result> rs;
      for (std::size_t i = 0; i < Hs.size(); ++i) {
        const Result r = go(Hs[i], cfgs[ci].mr, cfgs[ci].rr, true);
        rs.push_back(r);
        finite = finite && r.ok;
        std::printf("%-16s %-6d %-12.4e ", i ? "" : cfgs[ci].name, int(Hs[i]), r.l2);
        if (i > 0 && rs[i - 1].l2 > 0) {
          const double o = std::log(rs[i - 1].l2 / r.l2) / std::log(2.0);
          if (ci == 0) ctrl_ord = o;
          if (ci == 2) dens_ord = o;
          std::printf("%-8.2f ", o);
        } else std::printf("%-8s ", "-");
        std::printf("%-11.2e %-10.4f %-9zu\n", r.u_iface_err, r.y_iface, r.steps);
      }
      if (ci == 3) both = rs;
      std::printf("\n");
    }

    //------------------------------------------------------------------------
    std::printf("Is the viscous interface force earning its place?  (both contrasts on)\n\n");
    std::printf("%-6s %-14s %-14s %-10s\n", "H", "L2 with F_nu", "L2 without", "ratio");
    std::printf("%s\n", std::string(50, '-').c_str());
    double best_gain = 0;
    for (std::size_t i = 0; i < Hs.size(); ++i) {
      const Result w = go(Hs[i], 5.0, 10.0, false);
      finite = finite && w.ok;
      const double gain = (both[i].l2 > 0) ? w.l2 / both[i].l2 : 0.0;
      best_gain = std::max(best_gain, gain);
      std::printf("%-6d %-14.4e %-14.4e %-10.2f\n", int(Hs[i]), both[i].l2, w.l2, gain);
    }

    //------------------------------------------------------------------------
    const double tol_ord = 1.8, tol_gain = 3.0, tol_drift = 1e-9;
    const bool pass_ord   = finite && ctrl_ord > tol_ord;
    const bool pass_gain  = finite && best_gain > tol_gain;
    const bool pass_drift = finite && both.back().phi_drift < tol_drift;
    std::printf("\nacceptance:\n");
    std::printf("  single-phase control is second order   order %.2f      %s\n",
                ctrl_ord, pass_ord ? "PASS" : "FAIL");
    std::printf("  F_nu is necessary (L2 worse without)   %.1fx          %s\n",
                best_gain, pass_gain ? "PASS" : "FAIL");
    std::printf("  phi conserved to round-off             %.2e      %s\n",
                both.back().phi_drift, pass_drift ? "PASS" : "FAIL");
    std::printf("\n  KNOWN AND NOT ASSERTED: a density ratio alone leaves an L2 residual\n"
                "  of a few per cent that does NOT converge (order %.2f here), even though\n"
                "  the exact profile is density-independent. The single-phase row rules out\n"
                "  the base scheme and the viscosity-only row converges, so it is specific\n"
                "  to the grad-rho path. Unexplained; see the header.\n", dens_ord);
    if (!(pass_ord && pass_gain && pass_drift)) status = 1;
  }
  Kokkos::finalize();
  return status;
}
