//==============================================================================
//  The pressure-based multiphase operators, checked against their own algebra
//  rather than against the paper they came from.
//
//  The central-moment operator's equilibrium is a CLOSED FORM,
//
//      k_eq(p,q,r) = [pqr == 000] + (p~ - 1) A(p,ux) A(q,uy) A(r,uz),
//      A = {1, -u, u^2},
//
//  derived by splitting f^eq = w_i[p~ + Phi_i] into a Maxwellian at rho = 1 plus
//  (p~ - 1) times the weights. A closed form that agrees with a paper's printed
//  equations still has to agree with a direct contraction, and this is the file
//  that makes it. Everything here is exact algebra: no flow, no time stepping.
//
//  What each block pins down, and how it fails:
//    1. The equilibrium moments, against a direct sum over the velocity set. A
//       wrong A factor shows up here and nowhere else until a simulation is
//       subtly wrong.
//    2. The two macroscopic definitions of Eq. (24): sum_i f^eq = p~ exactly,
//       and sum_i c_i f^eq = u exactly -- the first moment is u, NOT rho u, and
//       an accidental division by rho is a viscosity error that looks like a bad
//       boundary condition.
//    3. Equilibrium is a fixed point with no force. Fails if any relaxation rate
//       is applied to the wrong slot.
//    4. The force delivers EXACTLY F/rho to the first moment -- the pressure
//       form's source is F_i = w_i (c_i.F)/(rho cs2), so its first moment
//       carries the 1/rho that the density-based Guo source does not.
//    5. p~ is conserved by collision, force or no force.
//    6. At omega = 1 the central-moment and BGK operators must agree to the
//       O(u^3) difference between the product-form and second-order equilibria
//       and no more, which is what says they are the same scheme.
//==============================================================================
#include "Check.hpp"
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/MultiphasePotentialBGK.hpp"
#include "core/Types.hpp"

#include <vector>

using namespace lbm;

using L  = D2Q9;
using CM = MultiphaseCentralMoments<L>;
using PB = MultiphasePotentialBGK<L, SecondOrderPhi<L>, RawPopulations>;

static Real TOL() { return sizeof(Real) == 4 ? Real(2e-5) : Real(2e-13); }

// A one-node phase field, so the operators have something to read.
struct Fields {
  View1D<Real> phi, gx, gy, gz, lap;
  Fields() : phi("phi", 1), gx("gx", 1), gy("gy", 1), gz("gz", 1), lap("lap", 1) {}
  void set(Real p, Real Gx, Real Gy, Real Lp) {
    auto h = Kokkos::create_mirror_view(phi);
    h(0) = p;  Kokkos::deep_copy(phi, h);
    h(0) = Gx; Kokkos::deep_copy(gx, h);
    h(0) = Gy; Kokkos::deep_copy(gy, h);
    h(0) = Real(0); Kokkos::deep_copy(gz, h);
    h(0) = Lp; Kokkos::deep_copy(lap, h);
  }
};

template <class Op>
static void bind(Op& op, Fields& fl) {
  op.phi = fl.phi;  op.Gx = fl.gx;  op.Gy = fl.gy;  op.Gz = fl.gz;  op.Lap = fl.lap;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Multiphase operators (pressure form), D2Q9, %s\n\n", precision_name());
    const Real tol = TOL();

    Fields fl;
    fl.set(Real(0.37), Real(0.11), Real(-0.07), Real(0.021));

    const Real pt = Real(1.234);            // a normalised pressure well off 1
    const Real ux = Real(0.081), uy = Real(-0.043);

    //--------------------------------------------------------------------------
    // 1. Equilibrium central moments against a direct contraction.
    //--------------------------------------------------------------------------
    std::printf("equilibrium central moments vs direct contraction\n");
    {
      Real feq[L::Q];
      for (int i = 0; i < L::Q; ++i)
        feq[i] = CM::seed_value(i, pt, ux, uy, Real(0));

      Real Aw[3][3];
      CM::weight_factors((const Real[3]){ux, uy, Real(0)}, Aw);

      double worst = 0;
      for (int n = 0; n < CM::NM; ++n) {
        const int p = CM::Basis::p_of(n), q = CM::Basis::q_of(n);
        Real direct = Real(0);
        for (int i = 0; i < L::Q; ++i)
          direct += feq[i] * CM::Basis::phi(p, cvel<L>(i, 0), ux)
                           * CM::Basis::phi(q, cvel<L>(i, 1), uy);
        const Real closed = CM::eq_moment(n, pt, Aw);
        worst = std::max(worst, std::abs(double(direct) - double(closed)));
      }
      check::near(Real(worst), Real(0), tol, "all 9 k_eq match a direct sum");
    }

    //--------------------------------------------------------------------------
    // 2. The macroscopic definitions of Eq. (24).
    //--------------------------------------------------------------------------
    std::printf("\nmacroscopic definitions\n");
    {
      Real feq[L::Q];
      for (int i = 0; i < L::Q; ++i) feq[i] = CM::seed_value(i, pt, ux, uy, Real(0));
      Real s = Real(0), mx = Real(0), my = Real(0);
      for (int i = 0; i < L::Q; ++i) {
        s += feq[i];
        mx += feq[i] * Real(cvel<L>(i, 0));
        my += feq[i] * Real(cvel<L>(i, 1));
      }
      check::near(s,  pt, tol, "sum_i f^eq = p~");
      check::near(mx, ux, tol, "sum_i c_ix f^eq = u_x  (not rho u_x)");
      check::near(my, uy, tol, "sum_i c_iy f^eq = u_y");
    }

    //--------------------------------------------------------------------------
    // 3./4./5. Collision invariants, with and without a force.
    //--------------------------------------------------------------------------
    std::printf("\ncollision invariants\n");
    {
      CM cm;  bind(cm, fl);
      cm.rho_L = Real(1);  cm.rho_H = Real(3);
      cm.mu_L = Real(0.1); cm.mu_H = Real(0.3);
      cm.kappa = Real(0);  cm.beta = Real(0);     // no capillary force yet

      const Real rho = cm.density_at(0);

      // A force-free state needs grad phi = 0 as well as b = 0: F_p is
      // proportional to grad rho = (drho/dphi) grad phi, so a density ratio and
      // a phase gradient together make a force even with no gravity and no
      // surface tension. Getting this wrong made the first draft of this file
      // report three failures against a correct operator.
      fl.set(Real(0.37), Real(0), Real(0), Real(0));

      // (3) fixed point, no force at all
      {
        Real f[L::Q];
        for (int i = 0; i < L::Q; ++i) f[i] = CM::seed_value(i, pt, ux, uy, Real(0));
        Real g[L::Q];
        for (int i = 0; i < L::Q; ++i) g[i] = f[i];
        const Macro m = cm.macroscopic(g, 0);
        cm.collide(g, m, 0);
        double worst = 0;
        for (int i = 0; i < L::Q; ++i)
          worst = std::max(worst, std::abs(double(g[i]) - double(f[i])));
        check::near(Real(worst), Real(0), tol, "equilibrium is a fixed point (F = 0)");
      }

      // (4)/(5) momentum gains exactly F/rho, p~ is conserved
      {
        cm.bx = Real(2.5e-3);  cm.by = Real(-1.1e-3);   // body force, F_b = rho b
        Real f[L::Q];
        for (int i = 0; i < L::Q; ++i) f[i] = CM::seed_value(i, pt, ux, uy, Real(0));
        Real s0 = Real(0), mx0 = Real(0), my0 = Real(0);
        for (int i = 0; i < L::Q; ++i) {
          s0 += f[i];
          mx0 += f[i] * Real(cvel<L>(i, 0));
          my0 += f[i] * Real(cvel<L>(i, 1));
        }
        const Macro m = cm.macroscopic(f, 0);
        cm.collide(f, m, 0);
        Real s1 = Real(0), mx1 = Real(0), my1 = Real(0);
        for (int i = 0; i < L::Q; ++i) {
          s1 += f[i];
          mx1 += f[i] * Real(cvel<L>(i, 0));
          my1 += f[i] * Real(cvel<L>(i, 1));
        }
        check::near(s1, s0, tol, "p~ conserved by collision");
        check::near(mx1 - mx0, cm.bx, tol, "momentum gains exactly F_x/rho = b_x");
        check::near(my1 - my0, cm.by, tol, "momentum gains exactly F_y/rho = b_y");
        cm.bx = Real(0); cm.by = Real(0);
      }

      // (4b) the same statement with the WHOLE force switched on -- capillary,
      // pressure and body -- checked against the operator's own force(), so the
      // assembly and the collision have to agree with each other rather than
      // with a hand-copied special case.
      {
        fl.set(Real(0.37), Real(0.11), Real(-0.07), Real(0.021));
        cm.kappa = Real(6e-2);  cm.beta = Real(3e-2);
        cm.bx = Real(2.5e-3);   cm.by = Real(-1.1e-3);

        Real F[3];
        cm.force(cm.local(0), 0, pt, F);

        Real f[L::Q];
        for (int i = 0; i < L::Q; ++i) f[i] = CM::seed_value(i, pt, ux, uy, Real(0));
        Real mx0 = Real(0), my0 = Real(0), s0 = Real(0);
        for (int i = 0; i < L::Q; ++i) {
          s0 += f[i];
          mx0 += f[i] * Real(cvel<L>(i, 0));
          my0 += f[i] * Real(cvel<L>(i, 1));
        }
        const Macro m = cm.macroscopic(f, 0);
        cm.collide(f, m, 0);
        Real mx1 = Real(0), my1 = Real(0), s1 = Real(0);
        for (int i = 0; i < L::Q; ++i) {
          s1 += f[i];
          mx1 += f[i] * Real(cvel<L>(i, 0));
          my1 += f[i] * Real(cvel<L>(i, 1));
        }
        check::near(s1, s0, tol, "p~ conserved with the full force");
        check::near(mx1 - mx0, F[0] / rho, tol, "momentum gains F_x/rho (full force)");
        check::near(my1 - my0, F[1] / rho, tol, "momentum gains F_y/rho (full force)");
        cm.kappa = Real(0); cm.beta = Real(0); cm.bx = Real(0); cm.by = Real(0);
      }
    }
    fl.set(Real(0.37), Real(0.11), Real(-0.07), Real(0.021));

    //--------------------------------------------------------------------------
    // 6. Central moments and BGK agree at omega = 1, to the equilibrium
    //    difference and no more.
    //--------------------------------------------------------------------------
    std::printf("\ncentral moments vs BGK at omega = 1\n");
    {
      // omega = 1 needs tau = 1/2, i.e. mu = rho cs2 / 2.
      const Real cs2v = cs2<L, Real>();
      CM cm;  bind(cm, fl);
      PB pb;  bind(pb, fl);
      for (auto* r : std::vector<Real*>{&cm.rho_L, &pb.rho_L}) *r = Real(1);
      for (auto* r : std::vector<Real*>{&cm.rho_H, &pb.rho_H}) *r = Real(1);
      cm.mu_L = cm.mu_H = pb.mu_L = pb.mu_H = Real(0.5) * cs2v;
      cm.kappa = pb.kappa = Real(0);
      cm.beta  = pb.beta  = Real(0);

      check::near(cm.omega_at(0), Real(1), tol, "omega = 1 as configured");

      Real fa[L::Q], fb[L::Q];
      for (int i = 0; i < L::Q; ++i) {
        // A deliberately non-equilibrium state, identical for both.
        const Real e = CM::seed_value(i, pt, ux, uy, Real(0));
        fa[i] = e * (Real(1) + Real(0.02) * Real((i * 5) % 7 - 3));
        fb[i] = fa[i];
      }
      const Macro ma = cm.macroscopic(fa, 0);
      const Macro mb = pb.macroscopic(fb, 0);
      check::near(ma.dens, mb.dens, tol, "both recover the same p~");
      check::near(ma.ux, mb.ux, tol, "both recover the same u_x");
      cm.collide(fa, ma, 0);
      pb.collide(fb, mb, 0);
      double worst = 0;
      for (int i = 0; i < L::Q; ++i)
        worst = std::max(worst, std::abs(double(fa[i]) - double(fb[i])));
      // |u|^3 ~ 6e-4 times the O(1) weights; the two equilibria differ there.
      check::near(Real(worst), Real(0), Real(2e-3),
                  "post-collision states agree to the O(u^3) equilibrium gap");
      std::printf("        (gap %.3e, |u|^3 = %.3e)\n",
                  worst, std::pow(std::hypot(double(ux), double(uy)), 3.0));
    }
  }
  const int rc = check::report("multiphase");
  Kokkos::finalize();
  return rc;
}
