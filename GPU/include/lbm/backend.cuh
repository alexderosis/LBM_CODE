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
//
//  THREE HOST/DEVICE-NEUTRAL HELPERS live here for the same reason the aliases
//  do: a driver that has to ask "am I on a GPU?" to time itself, or to find out
//  how much memory it may use, would need an #ifdef of its own and would stop
//  being one source. They are `sync()`, `device_info()` and the DeviceInfo it
//  returns.
//==============================================================================
#include <string>
#if defined(__CUDACC__)
  #include "colour.cuh"
  #include "magnetic.cuh"
  #include "scalar.cuh"
  #include "solver.cuh"
  namespace lbm { namespace backend {
    using Fluid    = lbm::Solver;
    using Scalar   = lbm::ScalarSolver;
    using Magnetic = lbm::MagneticSolver;
    using Colour   = lbm::ColourSolver;
    inline constexpr bool on_device = true;

    // Kernel launches are asynchronous. Without this either side of a timed
    // loop you measure the launch queue, not the work.
    inline void sync() { LBM_CUDA_CHECK(cudaDeviceSynchronize()); }

    struct DeviceInfo {
      std::string name = "host";
      double peak_gbs  = 0;      // theoretical pin bandwidth, GB/s
      double free_gb   = 0;
      double total_gb  = 0;
      bool   on_gpu    = false;
    };

    inline DeviceInfo device_info() {
      DeviceInfo d;
      cudaDeviceProp p{};
      if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return d;
      d.name   = p.name;
      d.on_gpu = true;

      // Theoretical pin bandwidth: clock in kHz, bus in bits, doubled for DDR.
      //
      // Read through cudaDeviceGetAttribute, NOT through the cudaDeviceProp
      // fields of the same name: those were deprecated in CUDA 12 and a build
      // against a newer toolkit should not start warning, or stop compiling,
      // over a diagnostic line. Some parts report a memory clock of 0; then
      // peak_gbs stays 0 and a caller must print "n/a" rather than a ratio.
      int clock_khz = 0, bus_bits = 0;
      cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0);
      cudaDeviceGetAttribute(&bus_bits,  cudaDevAttrGlobalMemoryBusWidth, 0);
      if (clock_khz > 0 && bus_bits > 0)
        d.peak_gbs = 2.0 * double(clock_khz) * (bus_bits / 8) / 1.0e6;
      std::size_t f = 0, t = 0;
      if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
        d.free_gb  = double(f) / 1e9;
        d.total_gb = double(t) / 1e9;
      }
      return d;
    }
  }}
#else
  #include "hostsim.hpp"
  namespace lbm { namespace backend {
    using Fluid    = lbm::host::Fluid;
    using Scalar   = lbm::host::Scalar;
    using Magnetic = lbm::host::Magnetic;
    using Colour   = lbm::host::Colour;
    inline constexpr bool on_device = false;

    inline void sync() {}                  // the host loop is already ordered

    struct DeviceInfo {
      std::string name = "host reference (serial)";
      double peak_gbs  = 0;
      double free_gb   = 0;
      double total_gb  = 0;
      bool   on_gpu    = false;
    };

    // No GPU: on_gpu stays false and a driver that sizes itself to device
    // memory falls back to whatever the user asked for.
    inline DeviceInfo device_info() { return DeviceInfo{}; }
  }}
#endif
