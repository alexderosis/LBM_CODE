#===============================================================================
#  Native CUDA lattice Boltzmann.
#
#  A separate project from the Kokkos code in the parent directory, on purpose:
#  it does not include ../src, is not built by the parent's CMake, and shares no
#  headers. The only thing the two have in common is the physics and the output
#  format, which is what makes them comparable.
#
#  Build (Turing / T4):
#    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=75
#    cmake --build build -j
#
#  Common architectures: 70 Volta, 75 Turing, 80 A100, 86 RTX 30xx,
#  89 RTX 40xx / L4, 90 Hopper.
#
#  FP32 is the default. FP64 on consumer NVIDIA parts runs at 1/32 to 1/64 of
#  the FP32 rate, so -DLBM_DOUBLE=ON is for checking arithmetic, not for speed.
#===============================================================================
cmake_minimum_required(VERSION 3.20)

# LBM_HOST_ONLY configures without CUDA at all, for a machine with no toolkit:
# the checks and the host builds of every driver still work, because the whole
# numerical core is LBM_HD and compiles as plain C++.
option(LBM_HOST_ONLY "Configure without CUDA; build only what runs on the host" OFF)
if(LBM_HOST_ONLY)
  project(lbm_cuda LANGUAGES CXX)
else()
  project(lbm_cuda LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(LBM_GPU_ARCH "75" CACHE STRING "CUDA architecture, e.g. 75 for a T4")
option(LBM_DOUBLE "Build in FP64 (slow on consumer cards)" OFF)
# Emulated 16-bit population STORAGE with 32-bit arithmetic: quantises on the
# way into the existing 32-bit array, so it costs memory nothing and buys speed
# nothing. It exists to measure the accuracy of the mixed-precision scheme
# before implementing it -- see the banner in include/lbm/streaming.cuh.
option(LBM_STORE16_EMULATE "Emulate FP16 population storage (accuracy probe)" OFF)
option(LBM_PTXAS_VERBOSE "Report registers and spills per kernel" OFF)
# OFF BY DEFAULT, ON PRECAUTIONARY GROUNDS -- AND THE MEASUREMENT SAYS SO
# EXPLICITLY, BECAUSE THE FIRST VERSION OF THIS NOTE CLAIMED MORE THAN IT COULD.
#
# --use_fast_math is not a rounding mode. It implies --prec-div=false,
# --prec-sqrt=false and --ftz=true, and the conservative Allen-Cahn source in
# include/lbm/phasefield.cuh runs BOTH an approximate sqrt and an approximate
# division at every node of every step:
#
#     A = (4/W) phi (1 - phi) G / |G|
#
# That is the term holding the interface at its prescribed width, so an error
# there perturbs the interface NORMAL rather than a diagnostic. Approximating it
# to buy speed is a bad trade in a scheme whose answer is an interface position.
#
# WHAT IT DOES NOT EXPLAIN. This flag was switched off while chasing a 3-D
# Rayleigh-Taylor discrepancy against the Kokkos parent (1.4219 against 1.4375
# at t/t0 = 2, then 1.1719/1.2031, then 0.8750/0.9219). It is NOT the cause.
# Rebuilt without it, the T4 reproduces the fast-math numbers exactly -- same
# spike, same bubble, same max|u| to the last digit at all seven instants.
# Precision is not the cause either: the parent in FP32 tracks its own FP64 to
# every digit through t/t0 = 2.
#
# The cause is a genuine difference at the phi = 1 wall, which
# validation/phase_wall.cpp in the parent now records. Do not let this option's
# name suggest the question is closed.
option(LBM_FAST_MATH "nvcc --use_fast_math: approximate div/sqrt, FTZ" OFF)

if(NOT LBM_HOST_ONLY)
  set(CMAKE_CUDA_ARCHITECTURES ${LBM_GPU_ARCH})
endif()

add_library(lbm_cuda_core INTERFACE)
target_include_directories(lbm_cuda_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(LBM_STORE16_EMULATE)
  target_compile_definitions(lbm_cuda_core INTERFACE LBM_STORE16_EMULATE)
  message(STATUS "LBM: emulating FP16 population storage (accuracy probe, no speed-up)")
endif()
if(LBM_DOUBLE)
  target_compile_definitions(lbm_cuda_core INTERFACE LBM_DOUBLE)
endif()
if(LBM_FAST_MATH)
  target_compile_options(lbm_cuda_core INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
if(LBM_PTXAS_VERBOSE)
  # Registers per thread and spill bytes per kernel. Worth switching on whenever
  # a kernel is slower than its arithmetic explains: spilling to local memory is
  # the usual cause and is invisible without this.
  target_compile_options(lbm_cuda_core INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()

set(LBM_APPS tgv3d tgv3d_bench bench box channel rayleigh_benard alfven orszag_tang droplet bubble newpaths hartmann rti3d impact skip rb_high_ra)
if(NOT LBM_HOST_ONLY)
  foreach(app ${LBM_APPS})
    add_executable(${app} src/${app}.cu)
    target_link_libraries(${app} PRIVATE lbm_cuda_core)
  endforeach()
endif()

#-------------------------------------------------------------------------------
# HOST BUILDS OF THE SAME DRIVERS.
#
# Every driver is written against the aliases in include/lbm/backend.cuh, which
# resolve to the CUDA solvers under nvcc and to the host reference drivers under
# a plain C++ compiler. The two sets of classes expose the same interface, so one
# source builds both ways with no #ifdef of its own.
#
# That means a wrong initial condition, a wrong Rayleigh number, a wrong
# diagnostic or a wrong coupling ORDER can be found on a laptop, at a small grid
# size, before anything is launched on a device. What a host build cannot check
# is the launch itself: grid configuration, register pressure, transfers.
#
# These targets are built only when LBM_HOST_APPS is ON, since on a machine with
# a GPU they are redundant. On a machine WITHOUT one, configure with
# -DLBM_HOST_ONLY=ON and no CUDA compiler is needed at all.
#-------------------------------------------------------------------------------
option(LBM_HOST_APPS "Also build every driver as plain C++, runnable with no GPU" OFF)
if(LBM_HOST_APPS OR LBM_HOST_ONLY)
  foreach(app ${LBM_APPS})
    if(NOT ${app} STREQUAL "bench")          # bench is a device benchmark only
      add_executable(host_${app} src/${app}.cu)
      set_source_files_properties(src/${app}.cu PROPERTIES LANGUAGE CXX)
      target_link_libraries(host_${app} PRIVATE lbm_cuda_core)
    endif()
  endforeach()
endif()

#-------------------------------------------------------------------------------
# The checks build as PLAIN C++ and need no GPU. LBM_HD expands to nothing off
# nvcc, so the same arithmetic the kernels run is exercised here.
#
#   host_check    47 unit checks: velocity sets, transforms, one collision at a
#                 time. Seconds.
#   host_physics  whole simulations against closed-form solutions -- Poiseuille,
#                 conduction, diffusion, advection, buoyancy, resistive decay,
#                 the shear Alfven wave. Under ten seconds.
#   host_colour   the colour-gradient model: phi_i as a weight set, the
#                 equilibrium's moments, sigma = 4 A tau / 9, recolouring as an
#                 exact partition -- and then a real droplet driven through all
#                 three passes, which is what actually exercises Esoteric Pull.
#   host_phasefield  the conservative Allen-Cahn field and the pressure-based
#                 fluid: the tanh identity the whole model rests on, a flat
#                 interface run 600 steps to show it neither spreads nor
#                 sharpens, and the central-moment operators on D3Q27 -- the
#                 basis trap pinned in both directions. Seconds.
#   host_body     the penalised rigid body: the indicator's apex bias, the two
#                 mass ratios Uhlmann's arrangement cannot express, the reaction
#                 against -sum F, and a body dragging a fluid.
#   host_freesurface  the free surface, in the order of what can go silently
#                 wrong: the mass ledger, hydrostatic equilibrium, the
#                 Fluid-never-touches-Gas invariant, and a collapsing column
#                 that exercises all three through several hundred conversions.
#
# Run both before trusting anything the device produces.
#
# WHAT THE HOST BUILD CANNOT SEE, and it is not only the launch. Every CUDA
# DRIVER CLASS -- Solver, PhaseFieldSolver, FreeSurfaceSolver, PenalisedBody,
# the set_*() methods that allocate and upload -- lives inside
# `#if defined(__CUDACC__)`, so a plain C++ build never compiles a line of it.
# The host reference in hostsim.hpp is a SEPARATE class with the same interface.
# So host-side setup code added to a device class is unchecked until nvcc sees
# it: a most-vexing-parse in Solver::set_regularized_walls compiled clean here
# and failed the CUDA build with 37 errors. Build with nvcc before believing a
# new solver method exists.
#-------------------------------------------------------------------------------
foreach(t host_check host_physics host_colour host_phasefield host_body host_freesurface)
  add_executable(${t} test/${t}.cpp)
  target_link_libraries(${t} PRIVATE lbm_cuda_core)
  set_source_files_properties(test/${t}.cpp PROPERTIES LANGUAGE CXX)
endforeach()

enable_testing()
add_test(NAME host_check   COMMAND host_check)
add_test(NAME host_physics COMMAND host_physics)
add_test(NAME host_colour  COMMAND host_colour)
add_test(NAME host_phasefield COMMAND host_phasefield)
add_test(NAME host_body   COMMAND host_body)
add_test(NAME host_freesurface COMMAND host_freesurface)
