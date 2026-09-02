#pragma once
//==============================================================================
//  Regularized velocity boundary condition (BC3).
//
//  Latt, Chopard, Malaspinas, Deville & Michler, "Straight velocity boundaries
//  in the lattice Boltzmann method", Phys. Rev. E 77, 056703 (2008), Sec. IV C,
//  Eqs. (44)-(45). A port of ../src/boundary/Regularized.hpp.
//
//  The node's populations are ALL replaced. The algorithm is
//
//    1. rho from the known populations, Eq. (27);
//    2. the unknown populations get, TEMPORARILY, a bounce-back of their
//       off-equilibrium part;
//    3. Pi^(1) = sum_i c_i c_i (f_i - f_i^eq) is evaluated from that;
//    4. every population is rebuilt as
//
//          g_i = f_i^eq(rho, u) + (w_i / 2 cs^4) Q_i : Pi^(1),
//          Q_i = c_i c_i - cs^2 I.                                    Eq. (45)
//
//  Step 2 is a SCAFFOLD ONLY. The paper is explicit that bounce-back of the
//  off-equilibrium part may not be used as the boundary condition itself,
//  because it cannot enforce the velocity exactly; it exists solely to give
//  Pi^(1) a value, and step 4 overwrites it.
//
//  Note what Eq. (45) does NOT contain: omega. The reconstruction is therefore
//  independent of the collision operator, and the same code serves BGK, TRT and
//  the central-moment operator here.
//
//  ============================ WHY THIS EXISTS ==============================
//  WALL PLACEMENT, which is the whole point and the reason this was the last
//  gap worth closing. Halfway bounce-back puts the wall midway between the last
//  fluid node and the first solid node. This condition puts the wall ON the
//  boundary node, because that is the node whose velocity is imposed. A channel
//  of H fluid nodes is H-1 lattice units wide here, not H.
//
//  Until this existed, the only velocity wall in this tree was halfway
//  bounce-back, while the magnetic wall (magnetic.cuh) is Dellar's moment
//  condition and puts B exactly ON the node. Mixing them is a half-cell
//  disagreement about where the channel is -- and a Hartmann layer is a handful
//  of cells thick, so half of one is not a rounding error. That is why
//  `hartmann` could not be run here and now can.
//  ===========================================================================
//
//  DENSITY CLOSURE. The paper's Eq. (28) is written for one particular figure
//  and velocity numbering. It is implemented geometrically instead -- known
//  populations are those with c_i . n > 0, n being the OUTWARD normal -- which
//  is numbering-independent and so survives the Esoteric Pull ordering.
//
//  WHAT IS NOT PORTED. The parent's enum also carries NrmOutXp and NrmOutFree,
//  which are constant-back-pressure and free outflow: a different feature (an
//  open boundary for the fluid, with its own literature and its own failure
//  mode -- the parent measures a zero-gradient outlet settling at rho ~ 181)
//  and not what "regularised walls" means. Velocity walls and corners are here.
//==============================================================================
#include "streaming.cuh"

namespace lbm {

//------------------------------------------------------------------------------
// Outward normal codes. `NrmCorner` marks a node where two or more walls meet:
// the closure Eq. (27) cannot be evaluated there because it needs a single
// normal, so rho is extrapolated along a wall instead and only the
// reconstruction is run.
//------------------------------------------------------------------------------
enum NormalCode : std::uint8_t {
  NrmNone   = 0,
  NrmXp     = 1, NrmXm = 2,
  NrmYp     = 3, NrmYm = 4,
  NrmZp     = 5, NrmZm = 6,
  NrmCorner = 7,
};

LBM_HD LBM_INLINE void normal_of(std::uint8_t code, int n[3]) {
  n[0] = n[1] = n[2] = 0;
  switch (code) {
    case NrmXp: n[0] =  1; break;
    case NrmXm: n[0] = -1; break;
    case NrmYp: n[1] =  1; break;
    case NrmYm: n[1] = -1; break;
    case NrmZp: n[2] =  1; break;
    case NrmZm: n[2] = -1; break;
    default: break;
  }
}

//------------------------------------------------------------------------------
// THE EQUILIBRIUM THE WALL RECONSTRUCTS AGAINST MUST BE THE OPERATOR'S OWN.
//
// BGK and TRT relax toward the second-order truncation `feq`; the
// central-moment operator relaxes toward the PRODUCT-form equilibrium, which is
// the inverse transform of k = (rho, 0, ..., 0) and differs from `feq` at
// O(u^3). Reconstructing a wall against the wrong one does not blow up -- it
// makes the boundary a fixed point of a slightly different scheme from the
// bulk, which reads as a small persistent slip. The parent reaches the same
// conclusion through its EquilibriumOf trait; here the operator is a template
// parameter of the kernel, so one bool carries it.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void reg_equilibrium(Real rho, const Real u[3], bool product,
                                       Real feq_out[27]) {
  if (product) { product_equilibrium(rho, u, feq_out); return; }
  for (int i = 0; i < 27; ++i) feq_out[i] = feq(i, rho, u[0], u[1], u[2]);
}

//------------------------------------------------------------------------------
// Eq. (27). Known populations are those travelling OUT of the domain: they were
// streamed from the interior and so carry information. u_perp is the imposed
// velocity projected on the outward normal.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE Real reg_density(const Real f[27], const int nrm[3], const Real u[3]) {
  Real out = Real(0), tang = Real(0);
  for (int i = 0; i < 27; ++i) {
    const int cn = D3Q27::cx(i) * nrm[0] + D3Q27::cy(i) * nrm[1] + D3Q27::cz(i) * nrm[2];
    if (cn > 0)       out  += f[i];
    else if (cn == 0) tang += f[i];
  }
  const Real uperp = u[0] * Real(nrm[0]) + u[1] * Real(nrm[1]) + u[2] * Real(nrm[2]);
  return (Real(2) * out + tang) / (Real(1) + uperp);
}

//------------------------------------------------------------------------------
// Eq. (45), with Pi^(1) supplied from outside rather than measured locally.
// Pi is packed xx, yy, zz, xy, xz, yz.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void reg_apply_with_stress(Real f[27], Real rho, const Real u[3],
                                             const Real Pi[6], bool product) {
  constexpr Real cs2v = Real(1) / Real(3);
  constexpr Real cs4v = cs2v * cs2v;
  Real fe[27];
  reg_equilibrium(rho, u, product, fe);
  for (int i = 0; i < 27; ++i) {
    const Real cx = Real(D3Q27::cx(i)), cy = Real(D3Q27::cy(i)), cz = Real(D3Q27::cz(i));
    const Real QP = (cx * cx - cs2v) * Pi[0] + (cy * cy - cs2v) * Pi[1]
                  + (cz * cz - cs2v) * Pi[2]
                  + Real(2) * (cx * cy * Pi[3] + cx * cz * Pi[4] + cy * cz * Pi[5]);
    f[i] = fe[i] + D3Q27::w(i) / (Real(2) * cs4v) * QP;
  }
}

// Pi^(1) = -(2 cs^2 / omega) rho S,  S = (grad u + grad u^T)/2.  Eqs. (21)-(22).
// `g[a][b]` is d u_b / d x_a.
//
// THE OMEGA HERE IS THE SHEAR RATE and nothing else. This route infers the
// stress from a measured strain rate, and the coefficient IS the viscosity;
// naming the wrong rate makes the boundary impose a different viscosity from
// the bulk, which surfaces as spurious slip. For TRT that is omega_PLUS, the
// even rate, not omega_minus.
LBM_HD LBM_INLINE void reg_stress_from_gradient(Real rho, Real omega,
                                                const Real g[3][3], Real Pi[6]) {
  constexpr Real cs2v = Real(1) / Real(3);
  const Real k = -Real(2) * cs2v * rho / omega;
  Pi[0] = k * g[0][0];
  Pi[1] = k * g[1][1];
  Pi[2] = k * g[2][2];
  Pi[3] = k * Real(0.5) * (g[0][1] + g[1][0]);
  Pi[4] = k * Real(0.5) * (g[0][2] + g[2][0]);
  Pi[5] = k * Real(0.5) * (g[1][2] + g[2][1]);
}

//------------------------------------------------------------------------------
// Eqs. (44)-(45). `f` is overwritten with the reconstructed populations.
//
// `unknown` is a bitmask over directions: bit i set means the population that
// should have streamed into direction i came from outside the fluid, so it holds
// nothing meaningful and must be invented.
//
// A MASK IS USED RATHER THAN THE SIGN OF c_i . n, and the reason is corners: at
// the corner of a box BOTH (1,-1,0) and (-1,1,0) are unknown -- each has its
// source node outside a different wall -- which no dot-product test can express.
//
// FORCED FLOW, and this is the subtle one. Chapman-Enskog with a Guo source
// gives sum_i c_i f^(1) = -F/2, so f^(1) has an ODD part -(w_i/2cs^2) c_i . F.
// Bounce-back reverses that sign, so each unknown is wrong by (w_i/cs^2) c_i.F.
// Summed over the FULL velocity set that error would vanish -- but the unknowns
// are a HALF-SPACE and it does not. On D2Q9 the parent measures a tangential
// force biasing Pi_xy by F/6 and a normal force biasing Pi_xx by F/6 and Pi_yy
// by F/2: different tensors entirely, which is why no single velocity shift
// repairs both cases and the correction has to be made per direction.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void reg_apply(Real f[27], Real rho, const Real u[3],
                                 std::uint32_t unknown, const Real* Fv, bool product) {
  constexpr Real cs2v = Real(1) / Real(3);
  constexpr Real cs4v = cs2v * cs2v;

  Real fe[27];
  reg_equilibrium(rho, u, product, fe);

  // Step 2: scaffolding for Pi^(1) only, overwritten in step 4.
  //
  // When BOTH members of a pair are unknown -- which happens on corners and
  // edges, never on a straight wall -- there is nothing to bounce back from.
  // Those are set to equilibrium plus the true odd part, contributing no
  // invented stress, which is the only defensible choice.
  const Real hodd = Real(0.5) / cs2v;
  for (int i = 0; i < 27; ++i) {
    if (!(unknown & (1u << i))) continue;
    const Real cF = Fv ? (Real(D3Q27::cx(i)) * Fv[0] + Real(D3Q27::cy(i)) * Fv[1]
                        + Real(D3Q27::cz(i)) * Fv[2])
                       : Real(0);
    const Real odd = D3Q27::w(i) * hodd * cF;
    f[i] = (unknown & (1u << opp(i)))
         ? fe[i] - odd
         : fe[i] + (f[opp(i)] - fe[opp(i)]) - Real(2) * odd;
  }

  // Step 3: Pi^(1) from the off-equilibrium parts.
  Real Pi[6] = {Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
  for (int i = 0; i < 27; ++i) {
    const Real d = f[i] - fe[i];
    const Real cx = Real(D3Q27::cx(i)), cy = Real(D3Q27::cy(i)), cz = Real(D3Q27::cz(i));
    Pi[0] += d * cx * cx;  Pi[1] += d * cy * cy;  Pi[2] += d * cz * cz;
    Pi[3] += d * cx * cy;  Pi[4] += d * cx * cz;  Pi[5] += d * cy * cz;
  }

  // Step 4: rebuild every population.
  for (int i = 0; i < 27; ++i) {
    const Real cx = Real(D3Q27::cx(i)), cy = Real(D3Q27::cy(i)), cz = Real(D3Q27::cz(i));
    const Real QP = (cx * cx - cs2v) * Pi[0] + (cy * cy - cs2v) * Pi[1]
                  + (cz * cz - cs2v) * Pi[2]
                  + Real(2) * (cx * cy * Pi[3] + cx * cz * Pi[4] + cy * cz * Pi[5]);
    // The second-order Hermite reconstruction of f^(1) given
    // sum_i c_i f^(1) = -F/2 and sum_i c_i c_i f^(1) = Pi^(1): the FIRST moment
    // needs its own term, which Q_i : Pi cannot supply since sum_i w_i c_i Q_i = 0.
    const Real cF = Fv ? (cx * Fv[0] + cy * Fv[1] + cz * Fv[2]) : Real(0);
    f[i] = fe[i] - D3Q27::w(i) * hodd * cF
         + D3Q27::w(i) / (Real(2) * cs4v) * QP;
  }
}

//==============================================================================
//  Host-side setup, shared by the CUDA and host drivers.
//==============================================================================

// One wall state. Deduplicated into a small table rather than stored per node:
// a full velocity field costs 12 bytes/node, hundreds of megabytes on a 3D
// grid, and real geometries use a handful of distinct values.
struct RegWallSpec {
  std::uint8_t normal = NrmNone;
  Real ux = 0, uy = 0, uz = 0;
};

//------------------------------------------------------------------------------
// Build the per-node normal code, velocity tag, unknown mask and corner
// extrapolation direction from a caller's spec.
//
// `spec[n]` gives each node's wall. `geom[n]` is the fluid geometry, used to
// decide which directions streamed from outside: a direction is unknown when
// the node it would have come FROM is not part of the flow.
//
// The corner stencil picks, per corner, an axis whose next TWO nodes are both
// straight-wall nodes. Those have a well-defined rho, so the extrapolation never
// reaches into the bulk and never needs a density field. A corner with no such
// stencil is reported and falls back to rho = 1.
//------------------------------------------------------------------------------
inline long build_reg_walls(const std::vector<RegWallSpec>& spec,
                            const std::vector<std::uint8_t>& geom,
                            int nx, int ny, int nz,
                            std::vector<std::uint8_t>& nrm,
                            std::vector<std::uint16_t>& tag,
                            std::vector<std::uint32_t>& unk,
                            std::vector<std::uint8_t>& ext,
                            std::vector<Real>& table,
                            bool& has_corners) {
  const long N = long(nx) * ny * nz;
  nrm.assign(std::size_t(N), std::uint8_t(NrmNone));
  tag.assign(std::size_t(N), 0);
  unk.assign(std::size_t(N), 0u);
  ext.assign(std::size_t(N), std::uint8_t(NrmNone));
  table.clear();

  long nwall = 0;
  for (long n = 0; n < N; ++n) {
    const RegWallSpec& w = spec[std::size_t(n)];
    if (w.normal == NrmNone) continue;
    ++nwall;
    std::size_t k = 0;
    for (; k < table.size() / 3; ++k)
      if (table[3 * k] == w.ux && table[3 * k + 1] == w.uy && table[3 * k + 2] == w.uz)
        break;
    if (k == table.size() / 3) { table.push_back(w.ux); table.push_back(w.uy); table.push_back(w.uz); }
    nrm[std::size_t(n)] = w.normal;
    tag[std::size_t(n)] = static_cast<std::uint16_t>(k);
  }
  if (nwall == 0) return 0;

  // Unknown-direction masks.
  auto outside = [&](int x, int y, int z) {
    const std::uint8_t g = geom[std::size_t(node_id(wrap(x, nx), wrap(y, ny),
                                                    wrap(z, nz), nx, ny))];
    return g == Excluded || g == Solid;
  };
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const long n = node_id(x, y, z, nx, ny);
        if (nrm[std::size_t(n)] == NrmNone) continue;
        std::uint32_t m = 0;
        for (int i = 0; i < 27; ++i)
          if (outside(x - D3Q27::cx(i), y - D3Q27::cy(i), z - D3Q27::cz(i)))
            m |= (1u << i);
        unk[std::size_t(n)] = m;
      }

  // Corner rho stencils.
  has_corners = false;
  long blind = 0, ncorner = 0;
  const int dirs[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const long n = node_id(x, y, z, nx, ny);
        if (nrm[std::size_t(n)] != NrmCorner) continue;
        has_corners = true;
        ++ncorner;
        for (int k = 0; k < (nz > 1 ? 6 : 4); ++k) {
          const int x1 = x + dirs[k][0], y1 = y + dirs[k][1], z1 = z + dirs[k][2];
          const int x2 = x + 2 * dirs[k][0], y2 = y + 2 * dirs[k][1], z2 = z + 2 * dirs[k][2];
          if (x1 < 0 || y1 < 0 || z1 < 0 || x2 < 0 || y2 < 0 || z2 < 0) continue;
          if (x1 >= nx || y1 >= ny || z1 >= nz) continue;
          if (x2 >= nx || y2 >= ny || z2 >= nz) continue;
          const std::uint8_t a = nrm[std::size_t(node_id(x1, y1, z1, nx, ny))];
          const std::uint8_t b = nrm[std::size_t(node_id(x2, y2, z2, nx, ny))];
          if (a != NrmNone && a != NrmCorner && b != NrmNone && b != NrmCorner) {
            ext[std::size_t(n)] = static_cast<std::uint8_t>(k + 1);
            break;
          }
        }
        if (ext[std::size_t(n)] == NrmNone) ++blind;
      }
  if (blind)
    std::fprintf(stderr,
                 "  [reg] %ld of %ld corner(s) have no straight-wall stencil and "
                 "fall back to rho = 1\n", blind, ncorner);
  return nwall;
}

}  // namespace lbm
