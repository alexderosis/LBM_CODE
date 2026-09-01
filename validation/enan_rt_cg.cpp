//==============================================================================
//  The three-dimensional Rayleigh-Taylor instability of De Rosis & Enan's
//  Sec. III.I, run with the COLOUR GRADIENT instead of the phase field.
//
//  WHY THIS EXISTS. validation/enan_rt runs the same case with the phase field
//  and comes in 42 % above their Table IX at t/t0 = 3. Everything that can be
//  varied has been: the Reynolds number is confirmed at 256, the interface
//  width makes no difference (xi = 3 and xi = 5 agree to one cell), the source
//  truncation makes no difference, and the operator is exact on a flat
//  interface at rest at every mobility. What is left is either something the
//  two-phase machinery shares -- the gradient stencil, the coupling order, the
//  D3Q27 ghost modes -- or the phase field itself.
//
//  A SECOND, INDEPENDENT MODEL ON THE SAME CASE SPLITS THOSE. The colour
//  gradient shares the lattice, the streaming, the domain and this driver's
//  measurement, and shares NOTHING of the interface treatment: there is no
//  Allen-Cahn equation, no mobility, no prescribed width. If it also runs slow,
//  the cause is shared. If it lands in their band, the phase field is isolated.
//
//  THEIR TABLE IX IS ALREADY THE RIGHT PLACE FOR THIS. Of the eight models it
//  collects, three are colour gradient -- a D3Q19-CGM-CM-LBM, a D3Q27-CGM-CM-
//  LBM and a D3Q27-CGM-MRT study -- so the comparison band was built for
//  exactly this kind of run.
//
//  WHAT DOES NOT MAP, and it is one thing. Their Pe = U d / M is a mobility,
//  and the colour gradient has none: the interface width is an OUTCOME of the
//  recolouring, not an input. So Re, At and Ca are matched and the width is
//  reported rather than prescribed. Everything else -- the domain, the
//  perturbation of their Eq. (83), the reference time sqrt(W/g) with no Atwood
//  factor, and the three spike measures -- is identical to enan_rt, so that
//  the two models differ in the interface treatment and in nothing else.
//==============================================================================
#include "collision/ColourGradient.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ColourGradientSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;
namespace {

using L   = D3Q27;
using CG  = ColourGradient<L>;
using Slv = ColourGradientSolver<L, EsotericPull<L>, CG>;

constexpr double PI = 3.14159265358979323846;
const double T_STAR[7] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
// Their Table IX and the eight-model band it collects.
const double T9[7]   = {1.898, 1.858, 1.741, 1.553, 1.304, 1.001, 0.648};
const double T9LO[7] = {1.887, 1.839, 1.711, 1.504, 1.256, 0.988, 0.648};
const double T9HI[7] = {1.904, 1.897, 1.776, 1.618, 1.396, 1.149, 0.863};

double arg_num(int c, char** v, const char* k, double d) {
  for (int i = 1; i + 1 < c; ++i) if (!std::strcmp(v[i], k)) return std::atof(v[i + 1]);
  return d;
}
bool arg_flag(int c, char** v, const char* k) {
  for (int i = 1; i < c; ++i) if (!std::strcmp(v[i], k)) return true;
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  const Index  W    = Index(arg_num(argc, argv, "-w", 64));
  const double At   = arg_num(argc, argv, "-at", 0.5);
  const double Re   = arg_num(argc, argv, "-re", 256.0);
  const double Ca   = arg_num(argc, argv, "-ca", 960.0);
  const double U    = arg_num(argc, argv, "-u", 0.04);
  const double beta = arg_num(argc, argv, "-beta", 0.7);
  const double tmax = arg_num(argc, argv, "-tmax", 3.0);
  const bool   dump = arg_flag(argc, argv, "-dump");

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const Index nx = W, ny = 4 * W, nz = W;
    const double gamma  = (1.0 + At) / (1.0 - At);
    const double ab     = 8.0 / 27.0;
    const double ar     = CG::alpha_r_from_ratio(Real(gamma), Real(ab));
    const double g      = U * U / double(W);
    const double nu     = double(W) * U / Re;      // matched KINEMATIC, as theirs
    const double tau    = 3.0 * nu + 0.5;
    // Their sigma = nu_H U / Ca -- no rho_H, as their driver writes it.
    const double sigma  = nu * U / Ca;
    const double A      = CG::A_from_sigma(Real(sigma), Real(tau));
    // THREE DIMENSIONS: no Atwood factor. Their 2-D drivers use
    // sqrt(W/(g At)) and their 3-D ones sqrt(W/g); using the 2-D form here
    // stretches the clock by 1.41 and was most of the phase field's error.
    const double t_ref  = std::sqrt(double(W) / g);
    const std::size_t nsteps = std::size_t(tmax * t_ref);

    std::printf("3-D Rayleigh-Taylor, COLOUR GRADIENT, vs De Rosis & Enan Table IX\n");
    std::printf("D3Q27 colour gradient, Esoteric Pull, %s, %s\n",
                ExecSpace::name(), precision_name());
    std::printf("  %d x %d x %d, walls in y, periodic in x and z\n",
                int(nx), int(ny), int(nz));
    std::printf("  At = %.3f (ratio %.2f)  Re = %.0f  Ca = %.0f  U = %.3f\n",
                At, gamma, Re, Ca, U);
    std::printf("  nu = %.4e  tau = %.5f  sigma = %.4e  A = %.4e  beta = %.2f\n",
                nu, tau, sigma, A, beta);
    std::printf("  t_ref = sqrt(W/g) = %.1f steps (NO Atwood factor: 3-D)\n",
                t_ref);
    std::printf("  THE INTERFACE WIDTH IS AN OUTCOME HERE, not an input: the\n"
                "  colour gradient has no mobility and no Pe, so their\n"
                "  Pe = 1024 has no counterpart and is not matched.\n\n");

    Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);
    CG cg;
    cg.alpha_r = Real(ar);       cg.alpha_b = Real(ab);
    cg.rho_r0  = Real(gamma);    cg.rho_b0  = Real(1);
    cg.nu_r    = Real(nu);       cg.nu_b    = Real(nu);
    cg.A       = Real(A);
    cg.beta    = Real(beta);
    cg.omega_bulk = Real(1);
    cg.by      = Real(-g);
    cg.rho_ref = Real(0.5 * (gamma + 1.0));
    Slv s(d, cg);

    const Index nyi = ny;
    s.set_geometry([&](Index, Index y, Index) -> CellType {
      return (y == 0 || y == nyi - 1) ? Solid : Fluid;
    });

    // Their Eq. (83): heavy above, interface at 2W, perturbed by
    // 0.05 W [cos(2 pi x) + cos(2 pi z)]. A sharp step, as theirs is.
    const Real y0 = Real(2.0 * double(W)), amp = Real(0.05 * double(W));
    const Real kx = Real(2.0 * PI / double(nx)), kz = Real(2.0 * PI / double(nz));
    const Real rr0 = Real(gamma), rb0 = Real(1);
    const Index hx = d.hx, hy = d.hy, hz = d.hz;
    s.initialize(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real x = Real(px - hx), y = Real(py - hy), z = Real(pz - hz);
      const Real yi = y0 + amp * (Kokkos::cos(kx * x) + Kokkos::cos(kz * z));
      return (y > yi) ? Slv::Colours{rr0, Real(0)}
                      : Slv::Colours{Real(0), rb0};
    });

    std::printf("  %-8s %-8s %-11s %-11s %-9s %-22s\n",
                "t/t0", "step", "y+ spike", "y+ (their IX)", "dev",
                "eight-model spread");
    std::printf("  %s\n", std::string(78, '-').c_str());

    int next = 0;
    double worst = 0;
    std::vector<double> got;
    for (std::size_t step = 0; step <= nsteps; ++step) {
      // refresh() BEFORE anything reads phi or the solver steps: it is what
      // recomputes the macroscopic fields and the colour gradient from the
      // populations. Omitting it does not fail loudly -- phi simply keeps its
      // seeded values, the measurement reads the initial condition at every
      // probe, and the spike sits frozen at 1.9062 for the whole run while the
      // deviation against their table grows to +194 % purely because their
      // column moves and this one cannot.
      s.refresh();
      if (next < 7 && step >= std::size_t(T_STAR[next] * t_ref)) {
        auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.phi());
        // The same three measures enan_rt reports, for the same reason: a
        // global minimum is a different quantity from the tip of the finger
        // the moment anything detaches.
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        Index spike = ny - 1, spike_ax = ny - 1, spike_pop = ny - 1;
        double umax = 0;
        bool finite = true;
        const Index xc = nx / 2, zc = nz / 2;
        const Index need = std::max(Index(1), Index((nx * nz) / 1000));
        for (Index y = 0; y < ny; ++y) {
          Index cnt = 0;
          for (Index z = 0; z < nz; ++z)
            for (Index x = 0; x < nx; ++x) {
              const double p = double(hp(d.id(x, y, z)));
              if (!std::isfinite(p)) finite = false;
              if (p > 0.0) { ++cnt; if (y < spike) spike = y; }
              umax = std::max(umax, std::fabs(double(hu(d.id(x, y, z)))));
            }
          if (cnt >= need && y < spike_pop) spike_pop = y;
          if (double(hp(d.id(xc, y, zc))) > 0.0 && y < spike_ax) spike_ax = y;
        }
        const double yp = double(spike) / double(W);
        got.push_back(yp);
        const double dev = 100.0 * (yp / T9[next] - 1.0);
        worst = std::max(worst, std::fabs(dev));
        const bool in = yp >= T9LO[next] - 1e-9 && yp <= T9HI[next] + 1e-9;
        std::printf("  %-8.1f %-8zu %-11.4f %-13.3f %+8.1f%%  [%.3f, %.3f] %-8s"
                    " |u|max %.3e\n",
                    T_STAR[next], step, yp, T9[next], dev,
                    T9LO[next], T9HI[next], in ? "inside" : "OUTSIDE", umax);
        std::fflush(stdout);
        if (!finite) { status = 1; break; }
        ++next;
      }
      s.step();
    }
    std::printf("\n  worst deviation %.1f%%.  The phase field on this case gives\n"
                "  +42.3%%; the band is the range across the eight models their\n"
                "  Table IX collects, three of which are colour gradient.\n", worst);

    if (dump) {
      const std::string p = "results/L_enan/enan_rt_cg.dat";
      if (std::FILE* f = std::fopen(p.c_str(), "a")) {
        std::fprintf(f, "# 3-D RTI, COLOUR GRADIENT, vs their Table IX.\n");
        std::fprintf(f, "# W Re At Ca beta nu tau sigma A t_ref steps y0..y3\n");
        std::fprintf(f, "%d %g %g %g %g %.6e %.6f %.6e %.6e %.1f %zu",
                     int(W), Re, At, Ca, beta, nu, tau, sigma, A, t_ref, nsteps);
        for (double v : got) std::fprintf(f, " %.4f", v);
        std::fprintf(f, "\n");
        std::fclose(f);
      }
    }
  }
  Kokkos::finalize();
  return status;
}
