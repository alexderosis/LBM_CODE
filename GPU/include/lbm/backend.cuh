#pragma once
//==============================================================================
//  One name for each solver, whichever way the file is compiled.
//
//  Under nvcc these are the CUDA classes. Under a plain C++ compiler they are
//  the host reference drivers, which run the SAME per-node update functions in a
//  serial loop. The two sets of classes deliberately expose the same interface,
//  so an application file written against these aliases compiles and runs both
//  ways with no #ifdef of its own.
//
//  That is not a convenience. It means every driver in src/ can be built and
//  executed on a machine with no GPU, at a small grid size, before it is ever
//  launched on a device -- so a wrong initial condition, a wrong Rayleigh number,
//  a wrong diagnostic or a wrong coupling order is found where it is cheap to
//  find. What remains unverified on such a machine is the launch itself: grid
//  configuration, register pressure, and host-device transfer.
//
//  Build a driver on the host with:
//     c++ -std=c++17 -O2 -Iinclude -x c++ src/<driver>.cu -o <driver>
//==============================================================================
#if defined(__CUDACC__)
  #include "magnetic.cuh"
  #include "scalar.cuh"
  #include "solver.cuh"
  namespace lbm { namespace backend {
    using Fluid    = lbm::Solver;
    using Scalar   = lbm::ScalarSolver;
    using Magnetic = lbm::MagneticSolver;
    inline constexpr bool on_device = true;
  }}
#else
  #include "hostsim.hpp"
  namespace lbm { namespace backend {
    using Fluid    = lbm::host::Fluid;
    using Scalar   = lbm::host::Scalar;
    using Magnetic = lbm::host::Magnetic;
    inline constexpr bool on_device = false;
  }}
#endif
