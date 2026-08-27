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
project(lbm_cuda LANGUAGES CXX CUDA)

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

set(CMAKE_CUDA_ARCHITECTURES ${LBM_GPU_ARCH})

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

foreach(app tgv3d tgv3d_bench bench)
  add_executable(${app} src/${app}.cu)
  target_link_libraries(${app} PRIVATE lbm_cuda_core)
endforeach()

#-------------------------------------------------------------------------------
# The numerical checks build as PLAIN C++ and need no GPU. LBM_HD expands to
# nothing off nvcc, so the same arithmetic the kernels run is exercised here.
# Run this before trusting anything the device produces.
#-------------------------------------------------------------------------------
add_executable(host_check test/host_check.cpp)
target_link_libraries(host_check PRIVATE lbm_cuda_core)
set_source_files_properties(test/host_check.cpp PROPERTIES LANGUAGE CXX)

enable_testing()
add_test(NAME host_check COMMAND host_check)
