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
option(LBM_PTXAS_VERBOSE "Report registers and spills per kernel" OFF)

if(NOT LBM_HOST_ONLY)
  set(CMAKE_CUDA_ARCHITECTURES ${LBM_GPU_ARCH})
endif()

add_library(lbm_cuda_core INTERFACE)
target_include_directories(lbm_cuda_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(LBM_DOUBLE)
  target_compile_definitions(lbm_cuda_core INTERFACE LBM_DOUBLE)
endif()
target_compile_options(lbm_cuda_core INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
if(LBM_PTXAS_VERBOSE)
  # Registers per thread and spill bytes per kernel. Worth switching on whenever
  # a kernel is slower than its arithmetic explains: spilling to local memory is
  # the usual cause and is invisible without this.
  target_compile_options(lbm_cuda_core INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()

set(LBM_APPS tgv3d tgv3d_bench bench box channel rayleigh_benard alfven orszag_tang)
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
#
# Run both before trusting anything the device produces.
#-------------------------------------------------------------------------------
foreach(t host_check host_physics)
  add_executable(${t} test/${t}.cpp)
  target_link_libraries(${t} PRIVATE lbm_cuda_core)
  set_source_files_properties(test/${t}.cpp PROPERTIES LANGUAGE CXX)
endforeach()

enable_testing()
add_test(NAME host_check   COMMAND host_check)
add_test(NAME host_physics COMMAND host_physics)
