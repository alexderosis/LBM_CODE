// Verify the documented equilibrium central moments of MhdCentralMoments<D3Q27>
// (doc/m3lb.tex, eq:cm3dho) against what the operator actually produces.
#include "collision/MhdCentralMoments.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "core/Types.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace lbm;
using Op = MhdCentralMoments<D3Q27, true>;   // sixth-order hydrodynamic part
using B  = ProductBasis<D3Q27>;

static const char* AX = "xyz";

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> U(-0.08, 0.08);
    const double cs2 = 1.0 / 3.0;

    double worst = 0; int worst_p = 0, worst_q = 0, worst_r = 0;
    // print one representative case, then check many random ones
    for (int trial = 0; trial < 200; ++trial) {
      const Real rho = Real(1.0 + 0.1 * U(rng));
      Real u[3] = {Real(U(rng)), Real(U(rng)), Real(U(rng))};
      Real b[3] = {Real(U(rng)), Real(U(rng)), Real(U(rng))};

      Op op;
      Real fe[27];
      op.equilibrium(fe, rho, u, b);
      Real k[27];
      Op::to_moments(fe, u, k);

      // M_ab = 1/2 |b|^2 delta_ab - b_a b_b
      const double b2 = double(b[0])*double(b[0]) + double(b[1])*double(b[1])
                      + double(b[2])*double(b[2]);
      auto M = [&](int a, int c) {
        return 0.5 * b2 * (a == c ? 1.0 : 0.0) - double(b[a]) * double(b[c]);
      };
      auto uu = [&](int a) { return double(u[a]); };
      auto bb = [&](int a) { return double(b[a]); };

      // closed form of eq:cm3dho for every (p,q,r)
      auto predict = [&](int p, int q, int r) -> double {
        const int e[3] = {p, q, r};
        const int ord = p + q + r;
        if (ord == 0) return double(rho);
        if (ord == 1) return 0.0;
        if (ord == 2) {
          // indices of the two factors
          int idx[2], n = 0;
          for (int a = 0; a < 3; ++a) for (int t = 0; t < e[a]; ++t) idx[n++] = a;
          return double(rho) * cs2 * (idx[0] == idx[1] ? 1.0 : 0.0) + M(idx[0], idx[1]);
        }
        if (ord == 3) {
          int idx[3], n = 0;
          for (int a = 0; a < 3; ++a) for (int t = 0; t < e[a]; ++t) idx[n++] = a;
          const int A = idx[0], C = idx[1], D = idx[2];
          return -(uu(A) * M(C, D) + uu(C) * M(A, D) + uu(D) * M(A, C));
        }
        if (ord == 4) {
          // k_aacc  (two exponents of 2)  or  k_aacd (one 2, two 1s)
          int two = -1, ones[2], no = 0;
          for (int a = 0; a < 3; ++a) {
            if (e[a] == 2 && two < 0) two = a;
            else if (e[a] == 2) ones[no++] = -100 - a;   // second exponent-2
            else if (e[a] == 1) ones[no++] = a;
          }
          if (no == 1 && ones[0] <= -100) {              // k_aacc
            const int A = two, C = -100 - ones[0];
            const int D = 3 - A - C;                     // the absent axis
            return double(rho) / 9.0 + bb(D) * bb(D) / 3.0
                 + uu(C)*uu(C)*M(A,A) + uu(A)*uu(A)*M(C,C) + 4.0*uu(A)*uu(C)*M(A,C);
          }
          const int A = two, C = ones[0], D = ones[1];   // k_aacd
          return uu(C)*uu(D)*M(A,A) + (1.0/3.0 + uu(A)*uu(A))*M(C,D)
               + 2.0*uu(A)*uu(D)*M(A,C) + 2.0*uu(A)*uu(C)*M(A,D);
        }
        if (ord == 5) {                                   // k_a cc dd
          int A = -1, cd[2], n = 0;
          for (int a = 0; a < 3; ++a) { if (e[a] == 1) A = a; else cd[n++] = a; }
          const int C = cd[0], D = cd[1];
          return -uu(A)*bb(A)*bb(A)/3.0
                 - 2.0/3.0*(uu(C)*M(A,C) + uu(D)*M(A,D))
                 - 2.0*uu(C)*uu(D)*uu(D)*M(A,C) - 2.0*uu(C)*uu(C)*uu(D)*M(A,D)
                 - 4.0*uu(A)*uu(C)*uu(D)*M(C,D)
                 - uu(A)*uu(D)*uu(D)*M(C,C) - uu(A)*uu(C)*uu(C)*M(D,D);
        }
        // ord == 6 : k_222
        double s = double(rho)/27.0 + b2/18.0;
        for (int a = 0; a < 3; ++a) s += uu(a)*uu(a)*bb(a)*bb(a)/3.0;
        for (int a = 0; a < 3; ++a)
          for (int c = a+1; c < 3; ++c) s += 4.0/3.0*uu(a)*uu(c)*M(a,c);
        s += 4.0*(uu(0)*uu(1)*uu(2)*uu(2)*M(0,1)
                + uu(0)*uu(1)*uu(1)*uu(2)*M(0,2)
                + uu(0)*uu(0)*uu(1)*uu(2)*M(1,2));
        s += uu(1)*uu(1)*uu(2)*uu(2)*M(0,0)
           + uu(0)*uu(0)*uu(2)*uu(2)*M(1,1)
           + uu(0)*uu(0)*uu(1)*uu(1)*M(2,2);
        return s;
      };

      for (int p = 0; p < 3; ++p)
        for (int q = 0; q < 3; ++q)
          for (int r = 0; r < 3; ++r) {
            const double got = double(k[B::mi(p, q, r)]);
            const double want = predict(p, q, r);
            const double err = std::abs(got - want);
            if (err > worst) { worst = err; worst_p = p; worst_q = q; worst_r = r; }
          }

      if (trial == 0) {
        std::printf("representative case:  rho = %.6f\n", double(rho));
        std::printf("  u = (% .5f, % .5f, % .5f)\n", double(u[0]), double(u[1]), double(u[2]));
        std::printf("  b = (% .5f, % .5f, % .5f)\n\n", double(b[0]), double(b[1]), double(b[2]));
        std::printf("  %-7s %-5s %16s %16s %10s\n", "k_pqr", "order", "operator", "closed form", "|diff|");
        std::printf("  %s\n", std::string(60, '-').c_str());
        for (int ord = 0; ord <= 6; ++ord)
          for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q)
              for (int r = 0; r < 3; ++r)
                if (p + q + r == ord) {
                  const double got = double(k[B::mi(p, q, r)]);
                  const double want = predict(p, q, r);
                  std::printf("  k_%d%d%d   %-5d %16.9e %16.9e %10.1e\n",
                              p, q, r, ord, got, want, std::abs(got - want));
                }
        std::printf("\n");
      }
    }
    std::printf("worst |operator - closed form| over 200 random (rho,u,b): %.3e  at k_%d%d%d\n",
                worst, worst_p, worst_q, worst_r);
    std::printf("%s\n", worst < 1e-14 ? "CLOSED FORM CONFIRMED" : "*** MISMATCH ***");
  }
  Kokkos::finalize();
  return 0;
}
