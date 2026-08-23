//==============================================================================
//  Flow through a voxelised patient-specific aorta -- arbitrary geometry.
//
//  Everything else in this suite runs on a geometry described by a formula. This
//  one loads a voxel array off disk and runs the same solver on it unchanged,
//  which is the point: `set_geometry` already takes an arbitrary predicate, so
//  supporting real geometry needs a reader, not a new solver.
//
//  GEOMETRY. src/io/VoxelGeometry.hpp reads the file the aorta project's
//  scripts/voxelize.py produces from the SimVascular surface mesh (case
//  0074_H_AO_H): 109 x 184 x 361 voxels at a pitch of 0.0616 cm, tagged
//  0 solid, 1 fluid, 2 inlet, 3 outlet. About 16% of the box is fluid.
//
//  D3Q27, not the D3Q19 the original project uses -- see the note on lattice
//  scope in doc/. The geometry is lattice-independent, so the only consequence
//  is that results are not directly comparable with that project's.
//
//  BOUNDARY CONDITIONS.
//    solid   halfway bounce-back, which is free under Esoteric Pull
//    inlet   regularised velocity wall, u = U * inward normal from the file
//    outlet  constant back-pressure, rho = 1
//
//  The inlet normal is stored in the file because the ascending-aorta cap is
//  oblique to the voxel axes; using a face normal instead would drive flow into
//  the wall. The outlet caps are left at a fixed pressure rather than a
//  prescribed flow split, so the division of flow between the branch vessels is
//  an OUTPUT of the simulation rather than an input.
//
//  WHAT IS AND IS NOT CHECKED. There is no analytic solution here, so this is
//  not an accuracy test and is not presented as one. What it verifies is that
//  the machinery holds up on a geometry with no symmetry: that the solver stays
//  finite, that mass is conserved to round-off once the flow is established,
//  and that what goes in at the inlet comes out at the outlets. That last one is
//  the useful check -- a leak through the bounce-back surface, or a mis-tagged
//  cap, shows up there and nowhere else.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "boundary/Regularized.hpp"
#include "io/VoxelGeometry.hpp"
#include "FieldDump.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "solver/FluidSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
// Pulsatile inlet waveform, in lattice time units. This is the profile the
// aorta project drives its inlet with, reproduced unchanged so the two runs are
// comparable: a linear ramp from rest over `ramp` steps, then a cycle running
// between a diastolic floor at 15% of peak and a systolic peak of 1, with the
// upstroke sharpened by raising the sinusoid to the 1.5 power. Phase 0 is the
// diastolic minimum; the systolic peak falls at phase 0.5.
//
// The return value multiplies the inlet velocity, so the DIRECTION is fixed by
// the cap normal and only the magnitude varies. That is what lets the whole
// drive be a scale on the wall table rather than a per-node update.
//------------------------------------------------------------------------------
static double inlet_profile(double t, double ramp, double period) {
  const double ramp_factor = std::min(1.0, t / ramp);
  const double phase = std::fmod(t, period) / period;
  const double wave = 0.5 + 0.5 * std::sin(2.0 * M_PI * phase - M_PI / 2.0);
  return ramp_factor * (0.15 + 0.85 * std::pow(wave, 1.5));
}

using L  = D3Q27;
using CM = CentralMoments<L, NoForcing, ShiftedPopulations>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    // Default geometry: $LBM_AORTA_GEOM, else a path relative to the working
    // directory; -geom overrides both. This was an absolute path into one
    // developer's home directory, which made the case unrunnable for anyone
    // else who checked the repository out.
    const char* env_geom = std::getenv("LBM_AORTA_GEOM");
    std::string path = env_geom ? env_geom : "data/geometry.bin";
    double Re = 200.0;
    Real U = Real(0.02);
    std::size_t steps = 20000, probe = 500;
    bool pulse = false;
    double period = 2000.0, ramp = 800.0;
    std::size_t dumpfrom = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-geom"  && i + 1 < argc) path = argv[++i];
      if (a == "-re"    && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-u"     && i + 1 < argc) U = Real(std::atof(argv[++i]));
      if (a == "-steps" && i + 1 < argc) steps = std::size_t(std::atol(argv[++i]));
      if (a == "-probe" && i + 1 < argc) probe = std::size_t(std::atol(argv[++i]));
      if (a == "-pulse") pulse = true;
      if (a == "-period" && i + 1 < argc) period = std::atof(argv[++i]);
      if (a == "-ramp"   && i + 1 < argc) ramp   = std::atof(argv[++i]);
      if (a == "-dumpfrom" && i + 1 < argc) dumpfrom = std::size_t(std::atol(argv[++i]));
    }

    std::printf("Aorta: flow through a voxelised patient geometry   D3Q27, central moments\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    VoxelGeometry g;
    try {
      g = load_voxel_geometry(path);
    } catch (const std::exception& e) {
      std::printf("  %s\n", e.what());
      Kokkos::finalize();
      return 1;
    }
    report(g, "aorta");

    const std::size_t n_in  = g.count_of(VoxelGeometry::TagInlet);
    const std::size_t n_out = g.count_of(VoxelGeometry::TagOutlet);
    if (n_in == 0 || n_out == 0) {
      std::printf("  geometry has no inlet or no outlet; nothing to drive\n");
      Kokkos::finalize();
      return 1;
    }

    // Reynolds number on the inlet cap's equivalent diameter: the cap holds
    // n_in voxels, so its area is n_in in lattice units and D = 2 sqrt(A/pi).
    const double Dlb = 2.0 * std::sqrt(double(n_in) / M_PI);
    const Real nu = Real(double(U) * Dlb / Re);

    std::printf("\n  inlet %zu voxels -> D = %.1f lattice units   Re = %.0f\n",
                n_in, Dlb, Re);
    std::printf("  U = %.4f   nu = %.6e   tau = %.6f\n",
                double(U), double(nu), 3.0 * double(nu) + 0.5);
    if (pulse) {
      // The Womersley number is what characterises a pulsatile flow: the ratio
      // of the oscillatory inertia to the viscous term. It fixes how far the
      // wall-driven shear layer penetrates in a beat, and so whether the
      // profile is quasi-steady (small alpha) or plug-like with a thin
      // oscillating boundary layer (large alpha). Re alone says nothing here.
      const double omega_c = 2.0 * M_PI / period;
      const double alpha = 0.5 * Dlb * std::sqrt(omega_c / double(nu));
      std::printf("  pulsatile: period %.0f   ramp %.0f   Womersley alpha = %.2f\n",
                  period, ramp, alpha);
      std::printf("             U is the SYSTOLIC PEAK; diastolic floor is 15%% of it\n");
    }
    std::printf("\n");

    Domain d(g.nx, g.ny, g.nz, false, false, false);
    CM coll;
    coll.omega = CM::omega_from_viscosity(nu);
    FluidSolver<L, EsotericPull<L>, CM> s(d, coll);

    // Solid stays solid. Inlet and outlet caps become regularised walls; every
    // other fluid voxel collides normally.
    s.set_geometry([&](Index x, Index y, Index z) -> CellType {
      const std::uint8_t t = g.at(x, y, z);
      if (t == VoxelGeometry::TagSolid) return Solid;
      if (t == VoxelGeometry::TagInlet || t == VoxelGeometry::TagOutlet) return RegWall;
      return Fluid;
    });

    // The inlet velocity is along the cap normal stored in the file. The
    // regularised condition needs a face normal code as well, and a voxelised
    // oblique cap has no single one -- so the cap is given NrmCorner, whose
    // unknown set is built geometrically per node rather than from an axis.
    const Real ux = Real(U * Real(g.inlet_normal[0]));
    const Real uy = Real(U * Real(g.inlet_normal[1]));
    const Real uz = Real(U * Real(g.inlet_normal[2]));
    using WS = decltype(s)::WallSpec;
    s.set_regularized_walls([&](Index x, Index y, Index z) -> WS {
      const std::uint8_t t = g.at(x, y, z);
      if (t == VoxelGeometry::TagInlet)  return WS{NrmCorner, ux, uy, uz, Real(1)};
      // The outlet caps are oblique to the voxel axes, so they get the
      // arbitrary-face outflow: rho pinned to 1, velocity taken from a fluid
      // neighbour found per node. Imposing u = 0 here instead would make them
      // walls and the domain would be closed -- measured as mass rising
      // monotonically, +8.5e-3 in 2000 steps, before this was fixed.
      if (t == VoxelGeometry::TagOutlet) return WS{NrmOutFree, Real(0), Real(0), Real(0), Real(1)};
      return WS{};
    });
    // The finite-difference corner stress walks a two-node stencil off each
    // wall node. In a box those neighbours are fluid or wall; in a vessel they
    // are frequently SOLID, whose populations Esoteric Pull never updates, so
    // the stencil reads stale memory and feeds garbage stress into the inlet.
    // Use the local closure instead until that path is made geometry-aware.
    s.set_fd_corners(false);
    s.initialize(Real(1));

    // Flux through a tagged cap, as a sum over EXPOSED FACES.
    //
    // A voxelised cap is a staircase of unit axis-aligned faces. For a
    // staircase approximating a plane of true area A and unit normal n, the
    // exposed faces satisfy  sum(n_face) = A n  -- the two surfaces bound a
    // closed region, whose total vector area is zero. So
    //
    //     sum over exposed faces of  u . n_face   =   A (u . n)
    //
    // which is the true flux, exactly, for uniform u. No area or normal needs
    // to be known: the geometry supplies both.
    //
    // The exposed-face test is the same one scripts/voxelize.py uses to DEFINE
    // a cap -- a frontier voxel is one where stepping outward leaves the fluid
    // -- so measuring the cap the way it was tagged keeps the two consistent.
    //
    // Two earlier versions were wrong in opposite directions. Counting voxels
    // and multiplying by the imposed speed UNDERcounts the inlet: a one-voxel
    // frontier layer holds about A*max|n_i| voxels, so with a dominant
    // component of 0.832 it reported 17.02 for a true flux near 20.5.
    // Normalising the summed face normals to a unit vector per voxel, as the
    // outlet first did, discards the |n| that carries the area and is wrong
    // whenever a voxel exposes more than one face.
    //
    // Sign convention: n_face points OUT of the fluid, so an inlet returns a
    // negative flux and an outlet a positive one.
    //
    // THEY DO NOT BALANCE, and that is a property of the outlet condition, not
    // a measurement error. The outflow REPLACES the populations at its nodes
    // with a rescaled copy of the donor's, so those nodes are a mass source and
    // sink rather than a conserving boundary; the velocity reported there is
    // the reconstructed one, not a record of what actually crossed the face.
    // Total mass is held flat by that same rescaling, so mass drift is NOT an
    // independent check either -- both quantities are downstream of the same
    // imposed rho. Read Q in as exact (it is the imposed inlet velocity through
    // a cap whose area is known to 0.65 degrees) and Q out as indicative only.
    auto cap_flux = [&](std::uint8_t tagv) {
      auto hxf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
      auto hyf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
      auto hzf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
      const int dirs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
      double q = 0;
      for (Index z = 0; z < g.nz; ++z)
        for (Index y = 0; y < g.ny; ++y)
          for (Index x = 0; x < g.nx; ++x) {
            if (g.at(x, y, z) != tagv) continue;
            const Index i = d.id(x, y, z);
            const double u3[3] = {double(hxf(i)), double(hyf(i)), double(hzf(i))};
            // Sum over EVERY exposed face. Verified against the geometry:
            // the inlet's exposed faces give |sum n_face| = 653.2 pointing
            // within 0.65 degrees of the file's stored normal, so
            // U * A * cos = 0.02 * 653.2 * 0.9935 = 12.98, which is what this
            // returns. Filtering the faces by direction to exclude the vessel
            // wall was tried and is WRONG -- it double-weights by the
            // projection and drove the inlet from 12.98 down to 10.38. The
            // lateral wall faces already cancel in the vector sum; that is
            // what makes the identity sum(n_face) = A n hold.
            for (int k = 0; k < 6; ++k) {
              const Index qx = x + dirs[k][0], qy = y + dirs[k][1], qz = z + dirs[k][2];
              const bool exposed =
                  (qx < 0 || qx >= g.nx || qy < 0 || qy >= g.ny ||
                   qz < 0 || qz >= g.nz) ||
                  g.at(qx, qy, qz) == VoxelGeometry::TagSolid;
              if (!exposed) continue;
              q += u3[0] * dirs[k][0] + u3[1] * dirs[k][1] + u3[2] * dirs[k][2];
            }
          }
      return q;
    };

    if (pulse)
      std::printf("  %8s %6s %13s %13s %13s %10s %10s %9s\n", "step", "phase",
                  "|u| max", "Q in", "Q out", "imbalance", "mass", "out rho");
    else
      std::printf("  %8s %13s %13s %13s %10s %10s %9s\n", "step",
                  "|u| max", "Q in", "Q out", "imbalance", "mass", "out rho");
    std::printf("  %s\n", std::string(pulse ? 93 : 86, '-').c_str());

    const Real m0 = s.total_mass();

    // CONSERVING OUTFLOW. The outlet overwrites its nodes every step, so
    // whatever density is imposed there is injected regardless of what arrived
    // -- pinning rho = 1 makes the boundary a mass source and sink, and the
    // inlet and outlet fluxes then differ by ~60% with no way to tell which is
    // wrong. Closing a loop on TOTAL mass instead makes it conserving: raise
    // the outlet density when mass is being lost, lower it when mass is
    // accumulating. The fixed point is the density at which what leaves equals
    // what enters, and it is found rather than assumed.
    //
    // Proportional control on the RELATIVE mass error. Scaling by total mass
    // rather than by outlet-node count matters: the latter gave an error of
    // order 3 for a drift of 5e-3, which saturated the step clamp every time
    // and turned the controller into bang-bang -- density slammed between
    // limits, flow reversed through the outlet, and the run blew up in 1500
    // steps.
    //
    // The controller is also held off until the vessel has filled. Engaging it
    // from rest makes it chase the filling transient, during which mass SHOULD
    // be changing, and it winds up fighting the physics.
    const double gain = 0.6;
    const std::size_t ctrl_every = 50;
    // Under pulsation the hold-off also waits out the ramp and one whole beat,
    // so the first correction is made against a complete cycle, not part of one.
    const std::size_t ctrl_start =
        pulse ? std::max<std::size_t>(3000, std::size_t(ramp + period)) : 3000;
    const double dr_max = 5e-4;
    double out_rho = 1.0;

    // One correction: mass above target -> lower the outlet density, let more
    // out. Clamped per application and in absolute value; an unclamped loop on
    // this went bang-bang once and reversed the flow.
    auto correct = [&](double err) {
      double dr = gain * err;
      if (dr >  dr_max) dr =  dr_max;
      if (dr < -dr_max) dr = -dr_max;
      out_rho -= dr;
      if (out_rho < 0.9) out_rho = 0.9;
      if (out_rho > 1.1) out_rho = 1.1;
      s.set_outflow_density(Real(out_rho));
    };

    // PULSATILE RETIMING. Mass genuinely oscillates within a beat -- the fluid
    // is compressible at O(Ma^2) and the drive is time-varying -- so a
    // controller sampling every 50 steps corrects against the beat itself and
    // flattens the pressure swing the run exists to produce. Under pulsation
    // the error is averaged over a whole period and applied once per cycle,
    // which separates the secular drift, which does need correcting, from the
    // cyclic variation, which does not.
    double err_sum = 0.0;
    std::size_t err_n = 0;

    for (std::size_t t = 0; t <= steps; ++t) {
      // The drive for the step about to be taken, set before ANYTHING reads the
      // wall table this iteration. corner_density() reads it inside step(), and
      // so does the macroscopic kernel, which at a regularised wall reports the
      // IMPOSED velocity rather than a population moment. Setting it after the
      // diagnostics instead makes the t = 0 row -- and the t = 0 volume dump --
      // show the inlet at full U while the waveform says the drive is zero.
      if (pulse)
        s.set_wall_velocity_scale(Real(inlet_profile(double(t), ramp, period)));
      if (t >= ctrl_start && t % ctrl_every == 0) {
        const double err = (double(s.total_mass()) - double(m0)) / double(m0);
        if (pulse) { err_sum += err; ++err_n; }
        else       { correct(err); }
      }
      if (pulse && err_n > 0 && t >= ctrl_start && t % std::size_t(period) == 0) {
        correct(err_sum / double(err_n));
        err_sum = 0.0; err_n = 0;
      }
      if (t % probe == 0) {
        s.compute_macroscopic();
        auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        auto hz = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
        double um = 0; bool finite = true;
        for (Index z = 0; z < g.nz; ++z)
          for (Index y = 0; y < g.ny; ++y)
            for (Index x = 0; x < g.nx; ++x) {
              if (g.at(x, y, z) == VoxelGeometry::TagSolid) continue;
              const Index i = d.id(x, y, z);
              const double a = double(hx(i)), b = double(hy(i)), c = double(hz(i));
              if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) finite = false;
              um = std::max(um, std::sqrt(a * a + b * b + c * c));
            }
        if (!finite) { std::printf("  DIVERGED at step %zu\n", t); status = 1; break; }
        // outward flux at the outlet is -n.u with the file's inward normal
        // convention, but the outlet caps have no stored normal, so the outlet
        // flux is measured as the shortfall in total divergence instead: what
        // enters must leave, and the inlet is the only driven face.
        const double qin = cap_flux(VoxelGeometry::TagInlet);
        const double qout = cap_flux(VoxelGeometry::TagOutlet);
        // qin is negative (entering), qout positive (leaving); at steady
        // state they cancel, so their sum is the conservation error.
        const double imb = (std::abs(qout) > 0) ? (qin + qout) / std::abs(qout) : 0.0;
        if (pulse)
          std::printf("  %8zu %6.3f %13.6e %13.6e %13.6e %10.2e %10.2e %9.5f\n",
                      t, std::fmod(double(t), period) / period, um, qin, qout, imb,
                      double(s.total_mass() - m0) / double(m0), out_rho);
        else
          std::printf("  %8zu %13.6e %13.6e %13.6e %10.2e %10.2e %9.5f\n",
                      t, um, qin, qout, imb,
                      double(s.total_mass() - m0) / double(m0), out_rho);
        // FIGVEC dumps the three velocity COMPONENTS rather than the speed,
        // which is what streamline integration needs -- speed alone gives no
        // direction to follow. Layout is component-major (all ux, then uy, then
        // uz), each plane x-fastest, matching what the source project's
        // render_3d.py reads.
        //
        // No solid sentinel here. The scalar dump marks solid with a negative
        // speed, but a velocity component is legitimately negative, so there is
        // no value left to steal. The renderer takes the mask from geometry.bin
        // instead, which it already opens for the cap tags.
        //
        // These are 3x the size of a speed dump, so -dumpfrom restricts them to
        // a window at the end of the run: a long run needs to converge, but only
        // its last couple of beats need rendering.
        if (std::getenv("FIGVEC") && t >= dumpfrom) {
          using namespace lbm::figdump;
          const std::size_t N = std::size_t(g.nx) * g.ny * g.nz;
          std::vector<float> buf(3 * N);
          std::size_t o = 0;
          for (int c = 0; c < 3; ++c)
            for (Index zz = 0; zz < g.nz; ++zz)
              for (Index yy = 0; yy < g.ny; ++yy)
                for (Index xx = 0; xx < g.nx; ++xx) {
                  if (g.at(xx, yy, zz) == VoxelGeometry::TagSolid) { buf[o++] = 0.0f; continue; }
                  const Index i = d.id(xx, yy, zz);
                  buf[o++] = float(c == 0 ? hx(i) : c == 1 ? hy(i) : hz(i));
                }
          char vf[64];
          std::snprintf(vf, sizeof vf, "aorta_u%04zu.bin", (t - dumpfrom) / probe);
          std::ofstream out(vf, std::ios::binary);
          const std::int32_t a1 = g.nx, b1 = g.ny, c1 = g.nz;
          out.write(reinterpret_cast<const char*>(&a1), 4);
          out.write(reinterpret_cast<const char*>(&b1), 4);
          out.write(reinterpret_cast<const char*>(&c1), 4);
          out.write(reinterpret_cast<const char*>(buf.data()),
                    std::streamsize(buf.size() * sizeof(float)));
          std::printf("  wrote %s (%d x %d x %d, 3 components)\n",
                      vf, int(g.nx), int(g.ny), int(g.nz));
        }
        if (std::getenv("FIGVOL")) {
          // The whole speed field, for the volume renderer. Any single x-plane
          // cuts this vessel into disconnected islands -- the aorta curves out
          // of every plane -- so a slice misrepresents the geometry however the
          // plane is chosen. Solid is written as a negative sentinel: the
          // renderer has to tell wall from stationary fluid, and a zero cannot,
          // because stagnant fluid is also zero.
          using namespace lbm::figdump;
          char vf[64];
          std::snprintf(vf, sizeof vf, "aorta_v%04zu.bin", t / probe);
          scalar_volume(vf, g.nx, g.ny, g.nz, [&](Index xx, Index yy, Index zz) {
            if (g.at(xx, yy, zz) == VoxelGeometry::TagSolid) return -1.0;
            const Index i = d.id(xx, yy, zz);
            const double a = double(hx(i)), b = double(hy(i)), c = double(hz(i));
            return std::sqrt(a * a + b * b + c * c);
          });
        }
        if (std::getenv("FIGDUMP")) {
          // Speed on the x-slice carrying the most fluid: the vessel runs along
          // z, so a slice normal to x cuts it lengthwise and shows the arch.
          using namespace lbm::figdump;
          static Index xs = -1;
          if (xs < 0) {
            Index best = 0;
            for (Index xx = 0; xx < g.nx; ++xx) {
              Index c = 0;
              for (Index zz = 0; zz < g.nz; ++zz)
                for (Index yy = 0; yy < g.ny; ++yy)
                  if (g.at(xx, yy, zz) != VoxelGeometry::TagSolid) ++c;
              if (c > best) { best = c; xs = xx; }
            }
            std::printf("  [frames] slice x = %d (%d fluid cells)\n", int(xs), int(best));
          }
          char fn[64];
          std::snprintf(fn, sizeof fn, "aorta_f%04zu.bin", t / probe);
          scalar_slice(fn, g.ny, g.nz, [&](Index yy, Index zz) {
            if (g.at(xs, yy, zz) == VoxelGeometry::TagSolid) return -1.0;   // mask
            const Index i = d.id(xs, yy, zz);
            const double a = double(hx(i)), b = double(hy(i)), c = double(hz(i));
            return std::sqrt(a * a + b * b + c * c);
          });
        }
        std::fflush(stdout);
      }
      if (t < steps) s.step();
    }
  }
  Kokkos::finalize();
  return status;
}
