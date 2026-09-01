//==============================================================================
//  De Rosis & Enan, Phys. Fluids 33, 043315 (2021), Sec. III -- the first group:
//  the six tests that exercise the interface-capture scheme alone.
//
//  In these the velocity field is PRESCRIBED and only the order parameter is
//  solved, so no fluid collision operator is involved at all. Each field is
//  built to distort the interface and then return it: the accuracy is the
//  relative L2 norm of what is left over after a full cycle, the paper's
//  Eq. (71),
//
//      e = sqrt( sum_x [phi(x,t_end) - phi(x,0)]^2 / sum_x phi(x,0)^2 ).
//
//  The reference numbers are the paper's own Tables I-VII, which also carry the
//  competing LB schemes it is measured against.
//
//  WHAT IS BEING COMPARED, AND TO WHICH COLUMN. The paper computes grad phi two
//  ways -- finite differences, its Eq. (20), and moments, its Eq. (70) -- and
//  tabulates both. This tree implements Eq. (20) only, and the reason is
//  structural rather than an omission: Eq. (70) reads the phase field's own
//  populations at the node, and under Esoteric Pull the two slots a node reads
//  are exactly the two it writes, so a fused gradient is a genuine race. The
//  gradient here is a separate pass over the phi FIELD (PhaseFieldSolver's
//  banner argues it at length). So every comparison below is against the
//  paper's FD column, which is also the one it uses for all its own figures.
//
//  LATTICE. The paper runs its phase field on D3Q19. Here the default is D3Q27
//  -- the richest three-dimensional set, and the one the fluid side needs for
//  the central-moment operator in the second group of tests -- with -lat d3q19
//  available to match the paper exactly. Two-dimensional cases are run with one
//  periodic cell in z, which on a product lattice is an exact reduction. The
//  gradient stencil always runs on the full Navier-Stokes lattice of the same
//  dimension, which is what sets the isotropy.
//
//  PECLET NUMBER. Pe = U0 xi / M, which is the definition the paper states when
//  it matches Ref. 41's value of 60 (U0 = 0.02, xi = 3, M = 0.001). The mobility
//  is derived from Pe rather than the other way round, so a row of the table is
//  a row of the paper's.
//
//  TWO ERRORS IN THE PAPER'S EQUATIONS, both found by checking that the
//  prescribed fields are solenoidal before running anything. A field that is not
//  divergence-free does not return an interface to its starting shape no matter
//  how good the scheme is, so this is worth doing first and costs nothing.
//
//   1. Eq. (74), the shear flow of test C, prints
//          u_x = -U0 pi cos[pi(x/L0 - 0.5)] sin[pi(y/L0 - 0.5)],
//          u_y = +U0 pi cos[pi(x/L0 - 0.5)] sin[pi(y/L0 - 0.5)],
//      whose second line is the first with the sign flipped. That is a uniform
//      direction (-1, +1) with a varying magnitude, div u = 0 only by accident
//      at isolated points, and it cannot produce the elongated tail the paper's
//      own Fig. 4 shows. The standard single-vortex field of the works it cites
//      swaps cos and sin in the second component,
//          u_y = +U0 pi sin[pi(x/L0 - 0.5)] cos[pi(y/L0 - 0.5)],
//      which IS solenoidal and does produce that tail. That is what is run here.
//   2. Eq. (75) labels both of its components u_x. The second is u_y; with that
//      reading the field is solenoidal as printed, so this one is a typesetting
//      slip and nothing more.
//
//  A THIRD THING THE PAPER DOES NOT STATE: the length of the Zalesak slot. It
//  gives the width as 15 and no more. Read off its Fig. 2(a), the slot is
//  vertical, centred in x, and runs from the bottom of the disk up to the disk
//  CENTRE -- a depth of R. That is what is used here and it is a measurement off
//  a figure, so the absolute error of that test is only as comparable as the
//  shape is.
//
//  WHY THE NUMBERS HERE COME OUT BELOW THE PAPER'S, AND WHY THAT IS NOT ALL
//  CREDIT. Three differences push the same way and none of them is the scheme:
//
//   1. The gradient stencil runs on D3Q27 against the paper's D3Q19. Isotropy of
//      that stencil is what sets the interface's spurious behaviour, and the
//      paper says itself, comparing its D3Q19 against a D3Q7, that more
//      directions help.
//   2. The Zalesak notch is initialised by applying the Eq. (72) tanh to the
//      SIGNED DISTANCE of the slotted disk, so the initial field has one
//      interface thickness everywhere including the corners. An implementation
//      that seeds a sharp notch pays, in the error norm, for the relaxation the
//      scheme performs in its first few hundred steps -- measured against a
//      sharp t = 0 field that the scheme was never going to keep. The paper does
//      not say which it does. This affects test B only.
//   3. The error is a norm against the run's OWN initial condition, so any
//      difference in how that condition is built moves it.
//
//  Tests A, C, D, E and F seed a plain circle or sphere through Eq. (72)
//  verbatim, so only the first difference applies to them; they are the cleaner
//  comparison and the Zalesak number should be read with the second in mind.
//==============================================================================
#include "collision/PhaseFieldBGK.hpp"
#include "collision/PhaseFieldCentralMoments.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/PhaseFieldSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lbm;

namespace {

constexpr double PI = 3.14159265358979323846;

//------------------------------------------------------------------------------
// The prescribed fields. Each is written in the paper's own variables and each
// is solenoidal -- checked symbolically before being typed in, see the banner.
// `s` is the position normalised by L0 and `tau` the time normalised by T.
//------------------------------------------------------------------------------
enum Field { Translate, Zalesak, Shear2D, Smooth2D, Sphere3D, Swirl3D };

KOKKOS_INLINE_FUNCTION
void velocity(int which, double X, double Y, double Z, double tau, double U0,
              Real& ux, Real& uy, Real& uz) {
  ux = uy = uz = Real(0);
  switch (which) {
    case Translate:                                     // Sec. III A
      ux = Real(U0); uy = Real(U0);
      break;
    case Zalesak:                                       // Eq. (73)
      ux = Real(-U0 * 2.0 * PI * (Y - 0.5));
      uy = Real( U0 * 2.0 * PI * (X - 0.5));
      break;
    case Shear2D: {                                     // Eq. (74), corrected
      const double sgn = (tau < 1.0) ? 1.0 : -1.0;      // reversed at t = T
      ux = Real(-sgn * U0 * PI * Kokkos::cos(PI * (X - 0.5)) *
                                 Kokkos::sin(PI * (Y - 0.5)));
      uy = Real( sgn * U0 * PI * Kokkos::sin(PI * (X - 0.5)) *
                                 Kokkos::cos(PI * (Y - 0.5)));
      break;
    }
    case Smooth2D: {                                    // Eq. (75)
      const double e = Kokkos::cos(PI * tau);
      ux = Real(-U0 * Kokkos::sin(4.0 * PI * X) * Kokkos::sin(4.0 * PI * Y) * e);
      uy = Real(-U0 * Kokkos::cos(4.0 * PI * X) * Kokkos::cos(4.0 * PI * Y) * e);
      break;
    }
    case Sphere3D: {                                    // Eqs. (76)-(78)
      const double e = Kokkos::cos(2.0 * PI * tau);
      const double sx = Kokkos::sin(PI * (X - 0.5)), cx = Kokkos::cos(PI * (X - 0.5));
      const double sy = Kokkos::sin(PI * (Y - 0.5)), cy = Kokkos::cos(PI * (Y - 0.5));
      const double sz = Kokkos::sin(PI * (Z - 0.5)), cz = Kokkos::cos(PI * (Z - 0.5));
      ux = Real(U0 * PI * cx * (sz - sy) * e);
      uy = Real(U0 * PI * cy * (sx - sz) * e);
      uz = Real(U0 * PI * cz * (sy - sx) * e);
      break;
    }
    case Swirl3D: {                                     // Eq. (79)
      const double e = Kokkos::cos(PI * tau);
      const double a = 4.0 * PI * (X - 0.5), b = 4.0 * PI * (Y - 0.5),
                   c = 4.0 * PI * (Z - 0.5);
      ux = Real(0.5 * U0 * (Kokkos::sin(a) * Kokkos::sin(b) +
                            Kokkos::cos(c) * Kokkos::cos(a)) * e);
      uy = Real(0.5 * U0 * (Kokkos::sin(b) * Kokkos::sin(c) +
                            Kokkos::cos(a) * Kokkos::cos(b)) * e);
      uz = Real(0.5 * U0 * (Kokkos::sin(c) * Kokkos::sin(a) +
                            Kokkos::cos(b) * Kokkos::cos(c)) * e);
      break;
    }
  }
}

const char* arg_str(int argc, char** argv, const char* key, const char* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return dflt;
}
double arg_num(int argc, char** argv, const char* key, double dflt) {
  const char* s = arg_str(argc, argv, key, nullptr);
  return s ? std::atof(s) : dflt;
}
bool arg_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], key) == 0) return true;
  return false;
}

}  // namespace

//------------------------------------------------------------------------------
struct Spec {
  const char* name;
  int         field;
  int         dim;          // 2 or 3
  double      L0;
  double      Rfac;         // R = Rfac * L0
  double      x0f, y0f, z0f;// centre, as fractions of L0
  double      dref;         // THE d THEIR CODE USES: nx for some cases, nx-1 for
                            // others, and it sets both T_ref and the mobility
  double      mob;          // fixed mobility; 0 means M = U0 * dref / Pe
  double      pe;           // the Pe of a single-Pe case; 0 if it has a sweep
  double      cycles;       // t_end / T,  T = dref / U0
  double      xi;           // interface width, lattice units
  double      ref_fd;       // the paper's FD entry, 0 if the case has a Pe sweep
  const char* table;
};

// THE PECLET NUMBER IS DOMAIN-BASED, NOT INTERFACE-BASED, and reading it the
// other way is the single largest error this case has carried. Their text says
// Pe = U0 xi / M; every one of their drivers computes
//
//     M = U_ref * d / Pe
//
// with d the DOMAIN SIZE. On their Table III at Pe = 80 that is M = 0.0497 and
// omega = 1.54; the interface-width reading gives M = 7.5e-4 and omega = 1.991,
// which is 66 times too little mobility and sits against the stability edge.
// Run that way, BGK diverges on three of the four Zalesak rows and the
// central-moment operator does not -- a difference that was written up here as
// the reason that operator exists, and was an artefact of the mobility.
// THEIR OMEGA NEVER EXCEEDS 1.988 ON ANY CASE IN THE PAPER.
//
// Three cases do not use that rule at all: their Tables II, IV and V hardcode
// M = 0.001, which the paper's text also states. Those are carried as `mob`.
//
// The interface width is likewise per-case rather than the 3 the text quotes:
// Table III uses d/100 = 1.99 and Table V uses 2.
const Spec SPECS[] = {
  // name        field      dim  L0     R/L0   x0/L0  y0/L0  z0/L0  dref   mob    pe    cyc  xi    ref     table
  {"translate",  Translate, 2,  200.0, 0.2,   0.50,  0.50,  0.50, 200.0, 0.0,     0.0, 10.0, 3.00, 0.0,    "I"},
  {"translate41",Translate, 2,  100.0, 0.25,  0.50,  0.50,  0.50, 100.0, 0.001,   0.0, 10.0, 3.00, 0.0134, "II"},
  {"zalesak",    Zalesak,   2,  200.0, 0.4,   0.50,  0.50,  0.50, 199.0, 0.0,     0.0,  1.0, 1.99, 0.0,    "III"},
  {"shear2d",    Shear2D,   2,  200.0, 0.2,   0.50,  0.30,  0.50, 199.0, 0.001,   0.0,  2.0, 3.00, 0.0244, "IV"},
  {"smooth2d",   Smooth2D,  2,  512.0, 0.2,   0.50,  0.50,  0.50, 511.0, 0.001,   0.0,  1.0, 2.00, 0.0199, "V"},
  {"sphere3d",   Sphere3D,  3,  100.0, 0.2,   0.30,  0.30,  0.50, 100.0, 0.0,   200.0,  2.0, 3.00, 0.0490, "VI"},
  {"swirl3d",    Swirl3D,   3,  100.0, 0.2,   0.50,  0.50,  0.50, 100.0, 0.0,   200.0,  1.0, 3.00, 0.1133, "VII"},
};

// The Peclet sweeps the paper tabulates, and its FD row for each.
const double PE_A[3]     = {125.0, 500.0, 2000.0};
const double REF_A_FD[3] = {0.0066, 0.0050, 0.0090};
const double PE_B[4]     = {80.0, 400.0, 800.0, 4000.0};
const double REF_B_FD[4] = {0.0593, 0.0590, 0.0558, 0.0505};

struct Result { double err, M, omega, steps, phi_min, phi_max, mass_drift; };

//------------------------------------------------------------------------------
template <class PL, class PColl>
Result run(const Spec& s, double Pe, double U0, const char* dump, int probe) {
  using Slv = PhaseFieldSolver<PL, EsotericPull<PL>, PColl>;

  const Index N  = Index(std::lround(s.L0));
  const Index nz = (s.dim == 3) ? N : Index(1);
  const double R = s.Rfac * s.L0;
  // M and T from THEIR d, not from L0 and not from xi. See the note above the
  // spec table for why the interface-width reading of Pe is wrong.
  const double M = (s.mob > 0.0) ? s.mob : U0 * s.dref / Pe;
  const std::size_t T     = std::size_t(std::lround(s.dref / U0));
  const std::size_t steps = std::size_t(std::lround(s.cycles * double(T)));

  Domain d(N, N, nz, true, true, true);

  PColl coll;
  coll.omega = PColl::omega_from_mobility(Real(M));
  coll.width = Real(s.xi);
  Slv pf(d, coll);

  const double x0 = s.x0f * s.L0, y0 = s.y0f * s.L0, z0 = s.z0f * s.L0;
  const bool slotted = (s.field == Zalesak);
  const double half_slot = 7.5;             // width 15, from the paper
  const Real xi = Real(s.xi);
  const Real Rc = Real(R), X0 = Real(x0), Y0 = Real(y0), Z0 = Real(z0);
  const bool three = (s.dim == 3);

  // phi = 1/2 [1 + tanh(2 (R - |x - x0|) / xi)], Eq. (72). The Zalesak slot is
  // carved with the same tanh applied to the signed distance to the notch, so
  // the initial field has one interface thickness everywhere rather than a
  // sharp corner the scheme would then have to smooth out on its own.
  pf.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Real X = Real(px - d.hx), Y = Real(py - d.hy), Z = Real(pz - d.hz);
    const Real dx = X - X0, dy = Y - Y0, dz = three ? (Z - Z0) : Real(0);
    Real sdf = Rc - Kokkos::sqrt(dx * dx + dy * dy + dz * dz);   // >0 inside
    if (slotted) {
      // Signed distance to the slot: a rectangle |x-x0| <= 7.5, y <= y0.
      const Real ax = Kokkos::abs(dx) - Real(half_slot);
      const Real ay = dy - Real(0);
      Real ds;                                   // >0 inside the slot
      if (ax < Real(0) && ay < Real(0)) ds = -Kokkos::fmax(ax, ay);
      else if (ax >= Real(0) && ay < Real(0))    ds = -ax;
      else if (ax < Real(0))                     ds = -ay;
      else ds = -Kokkos::sqrt(ax * ax + ay * ay);
      sdf = Kokkos::fmin(sdf, -ds);              // disk minus slot
    }
    return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * sdf / xi));
  });

  View1D<Real> ux("ux", d.n_padded), uy("uy", d.n_padded), uz("uz", d.n_padded);
  pf.set_velocity(ux, uy, uz);

  const int which = s.field;
  const double Ln = s.L0, Tn = double(T);
  auto set_u = [&](std::size_t t) {
    Kokkos::parallel_for("uprescribe", Range(0, d.n_padded), KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      velocity(which, double(px - d.hx) / Ln, double(py - d.hy) / Ln,
               double(pz - d.hz) / Ln, double(t) / Tn, U0, ux(n), uy(n), uz(n));
    });
    Kokkos::fence();
  };

  // The initial field, kept for the error norm.
  pf.compute_field();
  auto h0 = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
  std::vector<double> phi0(std::size_t(d.n_padded));
  double mass0 = 0;
  for (Index z = 0; z < nz; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y, z);
        phi0[std::size_t(n)] = double(h0(n));
        mass0 += double(h0(n));
      }

  for (std::size_t t = 0; t < steps; ++t) {
    set_u(t);
    pf.refresh();
    pf.step();
    if (probe > 0 && (t + 1) % std::size_t(probe) == 0) {
      pf.compute_field();
      auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
      double lo = 1e300, hi = -1e300;
      for (Index z = 0; z < nz; ++z)
        for (Index y = 0; y < N; ++y)
          for (Index x = 0; x < N; ++x) {
            const double v = double(hp(d.id(x, y, z)));
            lo = std::min(lo, v); hi = std::max(hi, v);
          }
      std::printf("      t/T = %-7.3f  phi in [%.4f, %.4f]\n",
                  double(t + 1) / Tn, lo, hi);
      std::fflush(stdout);
    }
  }

  pf.compute_field();
  auto h1 = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
  double num = 0, den = 0, lo = 1e300, hi = -1e300, mass1 = 0;
  for (Index z = 0; z < nz; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y, z);
        const double a = phi0[std::size_t(n)], b = double(h1(n));
        num += (b - a) * (b - a);  den += a * a;
        lo = std::min(lo, b); hi = std::max(hi, b);
        mass1 += b;
      }

  if (dump && *dump) {
    const std::string p = std::string("results/L_enan/") + dump + ".dat";
    std::FILE* f = std::fopen(p.c_str(), "w");
    if (f) {
      const Index zc = nz / 2;
      std::fprintf(f, "# %s  L0=%g R=%g xi=%g Pe=%g M=%.6e steps=%zu lat=%s\n",
                   s.name, s.L0, R, s.xi, Pe, M, steps, PL::name);
      std::fprintf(f, "# mid-z slice: x y phi_final phi_initial\n");
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) {
          const Index n = d.id(x, y, zc);
          std::fprintf(f, "%d %d %.6f %.6f\n", int(x), int(y),
                       double(h1(n)), phi0[std::size_t(n)]);
        }
      std::fclose(f);
    }
  }

  return {std::sqrt(num / den), M, double(coll.omega), double(steps), lo, hi,
          (mass0 != 0.0) ? (mass1 / mass0 - 1.0) : 0.0};
}

//------------------------------------------------------------------------------
namespace {

void table_row(const std::string& name, const char* hdr, const std::string& row) {
  const std::string path = "results/L_enan/" + name;
  bool fresh = true;
  if (std::FILE* t = std::fopen(path.c_str(), "r")) { fresh = false; std::fclose(t); }
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (!f) return;
  if (fresh) std::fputs(hdr, f);
  std::fputs(row.c_str(), f); std::fputc('\n', f); std::fclose(f);
}

const char* HDR =
  "# De Rosis & Enan, Phys. Fluids 33, 043315 (2021), Sec. III, first group:\n"
  "# interface capture with a PRESCRIBED velocity -- no fluid operator involved.\n"
  "# M3LB conservative Allen-Cahn + Esoteric Pull, periodic everywhere, FP64.\n"
  "# 2-D cases run with one periodic cell in z. The OPERATOR IS IN THE lat\n"
  "# COLUMN: d3q27cm is PhaseFieldCentralMoments (their Sec. II.D), anything\n"
  "# else is PhaseFieldBGK (their Sec. II.B) on the named lattice.\n"
  "# e is their Eq. (71); Pe = U0 xi / M; T = L0/U0.\n"
  "# The comparison column is their FD (Eq. 20) result: the moment gradient of\n"
  "# their Eq. (70) is not implemented here -- it reads the node's own\n"
  "# populations, which is a race under Esoteric Pull.\n"
  "# case lat L0 R xi U0 Pe M omega cycles steps e ref_fd phi_min phi_max mass_drift\n";

}  // namespace

int main(int argc, char** argv) {
  const std::string only = arg_str(argc, argv, "-case", "all");
  const std::string lat  = arg_str(argc, argv, "-lat", "d3q27");
  const std::string op   = arg_str(argc, argv, "-op", "cm");
  const double U0        = arg_num(argc, argv, "-u0", 0.02);
  const bool   dump      = arg_flag(argc, argv, "-dump");
  const int    probe     = int(arg_num(argc, argv, "-probe", 0));
  // Diagnostic knob, not part of any published case: the single-Pe cases pin
  // M = 1e-3, which puts omega at 1.988 -- the BGK stability edge -- so a
  // BGK-vs-CM comparison there conflates the collision with the overrelaxation.
  // -mob raises M to move omega away from 2 and separate the two effects.
  const double mob       = arg_num(argc, argv, "-mob", 0.001);
  // A -mob run is a diagnostic at a Peclet number no table covers, so it must
  // not append: the file's ref column would show the Pe = 60 reference beside
  // an error measured somewhere else entirely.
  const bool   diag      = std::fabs(mob - 0.001) > 1e-15 ||
                           arg_num(argc, argv, "-cycles", 0.0) > 0.0;
  // Second diagnostic knob: shorten the run. The two cases where the
  // central-moment collision loses are also the two longest (50k and 100k
  // steps against the Zalesak disk's 10k), so whether the error compounds per
  // step or saturates is the question that separates a per-step bias from a
  // wrong steady profile. Overrides s.cycles when positive.
  const double cyc       = arg_num(argc, argv, "-cycles", 0.0);

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::printf("De Rosis & Enan (2021) Sec. III, first group: interface capture\n");
    std::printf("M3LB conservative Allen-Cahn phase field, %s, %s collision, "
                "Esoteric Pull\n",
                op == "cm" ? "D3Q27" :
                  (lat == "d3q19" ? "D3Q19" : (lat == "d3q7" ? "D3Q7" : "D3Q27")),
                op == "cm" ? "central-moment (their Sec. II.D)"
                           : "BGK (their Sec. II.B physics only)");
    std::printf("backend %s   precision %s   U0 = %g\n",
                ExecSpace::name(), precision_name(), U0);
    std::printf("the velocity is prescribed, so no fluid collision operator is "
                "exercised here;\nthe reference column is the paper's FD "
                "(Eq. 20) result -- see the banner.\n");

    auto go = [&](const Spec& s, double Pe, const char* tag) {
      if (op == "cm") {
        // The central-moment collision is a product-lattice scheme; the
        // reduced sets have no tensor-product transform (see its banner).
        if (lat != "d3q27" && lat != "d2q9")
          throw std::runtime_error("-op cm needs -lat d3q27");
        return run<D3Q27, PhaseFieldCentralMoments<D3Q27>>(s, Pe, U0, tag, probe);
      }
      if (lat == "d3q19") return run<D3Q19, PhaseFieldBGK<D3Q19>>(s, Pe, U0, tag, probe);
      if (lat == "d3q7")  return run<D3Q7,  PhaseFieldBGK<D3Q7>>(s, Pe, U0, tag, probe);
      return run<D3Q27, PhaseFieldBGK<D3Q27>>(s, Pe, U0, tag, probe);
    };

    auto emit = [&](const Spec& s, double Pe, const Result& r, double ref) {
      char row[512];
      std::snprintf(row, sizeof row,
          "%s %s %g %g %g %g %g %.6e %.6f %g %.0f %.6f %.6f %.6f %.6f %.3e",
          s.name, (op == "cm" ? "d3q27cm" : lat.c_str()), s.L0, s.Rfac * s.L0, s.xi, U0, Pe, r.M, r.omega,
          s.cycles, r.steps, r.err, ref, r.phi_min, r.phi_max, r.mass_drift);
      table_row("enan_interface.dat", HDR, row);
    };

    auto banner = [&](const Spec& s, const char* what) {
      std::printf("\n%s  (%s, their Table %s)\n%s\n", s.name, what, s.table,
                  std::string(64, '=').c_str());
      std::printf("  L0 = %g, R = %g, xi = %g, %d-D, t_end = %g T\n",
                  s.L0, s.Rfac * s.L0, s.xi, s.dim, s.cycles);
    };

    for (const Spec& s : SPECS) {
      if (only != "all" && only != s.name) continue;

      const bool sweepA = (std::strcmp(s.name, "translate") == 0);
      const bool sweepB = (std::strcmp(s.name, "zalesak") == 0);
      const int  nrun   = sweepA ? 3 : (sweepB ? 4 : 1);

      banner(s, sweepA ? "diagonal translation, Zu et al. setup"
                       : sweepB ? "Zalesak disk"
                       : s.name);
      std::printf("  %8s %11s %9s %11s %11s %9s %9s %11s\n",
                  "Pe", "M", "omega", "e (here)", "e (paper FD)", "dev",
                  "phi range", "mass drift");
      std::printf("  %s\n", std::string(92, '-').c_str());

      for (int k = 0; k < nrun; ++k) {
        const double Pe  = sweepA ? PE_A[k]
                         : (sweepB ? PE_B[k]
                         : (s.pe > 0.0 ? s.pe : U0 * s.dref / s.mob));
        const double ref = sweepA ? REF_A_FD[k] : (sweepB ? REF_B_FD[k] : s.ref_fd);
        char tag[128];
        std::snprintf(tag, sizeof tag, "%s_%s_pe%g", s.name,
                      op == "cm" ? "cm" : lat.c_str(), Pe);
        Spec sc = s;
      if (cyc > 0.0) sc.cycles = cyc;
      const Result r = go(sc, Pe, dump ? tag : "");
        std::printf("  %8.0f %11.4e %9.5f %11.5f %11.4f %+8.1f%% "
                    "[%.3f,%.3f] %11.2e\n",
                    Pe, r.M, r.omega, r.err, ref,
                    100.0 * (r.err / ref - 1.0), r.phi_min, r.phi_max,
                    r.mass_drift);
        std::fflush(stdout);
        if (!diag) emit(sc, Pe, r, ref);
        if (!std::isfinite(r.err)) status = 1;
      }
    }
    std::printf("\ndev is measured against the paper's FD column. phi should stay "
                "in [0,1];\nexcursions past it are the scheme overshooting, and "
                "mass drift is the\nconservative Allen-Cahn form's own "
                "conservation, not an imposed constraint.\n");
  }
  Kokkos::finalize();
  return status;
}
