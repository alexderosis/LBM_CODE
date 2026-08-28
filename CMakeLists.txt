cmake_minimum_required(VERSION 3.20)
project(M3LB VERSION 0.1.0 LANGUAGES CXX)

#-------------------------------------------------------------------------------
# Options
#-------------------------------------------------------------------------------
set(LBM_PRECISION "double" CACHE STRING "Scalar precision: double (FP64) or float (FP32)")
set_property(CACHE LBM_PRECISION PROPERTY STRINGS double float)
option(LBM_BUILD_TESTS      "Build unit tests"       ON)
option(LBM_BUILD_VALIDATION "Build validation cases" ON)
option(LBM_BUILD_DEMONSTRATOR "Build demonstrator cases" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

#-------------------------------------------------------------------------------
# GPU backend.
#
# THIS BACKEND HAS BEEN RUN ON A GPU, WITH A MIXED RESULT. On a Tesla T4, seven
# validation executables build after six fixes for nvcc restrictions the CPU
# backends cannot see, and BGK and TRT reach ~420-447 MLUPS at 64^3 with mass
# drift exactly 0. But the two CENTRAL-MOMENT configurations did not finish in
# seventeen minutes at the same size, where BGK took 0.03 s, and -Xptxas -v rules
# out register spilling. Since the campaign and the aorta both run central
# moments, that 447 MLUPS is a BGK-ONLY figure. See doc/m3lb.pdf, section 14.
#
# The independent CUDA implementation in GPU/ does not have the problem -- the
# same scheme runs at 99.6% of BGK there -- so run production central moments
# through that until this is understood.
#
# Two things that otherwise cost a day each:
#
#   * Consumer NVIDIA cards run FP64 at 1/32 to 1/64 of their FP32 rate, so a
#     default (double) build on a laptop GPU is routinely SLOWER than the CPU.
#     Configure with -DLBM_PRECISION=float. The FP32 caveat in the Known
#     Limitations then applies: it cannot resolve the finest convergence tests.
#   * These settings only bite when Kokkos is BUILT here, from extern/kokkos or
#     FetchContent. Against an installed Kokkos, that installation's own
#     backends win and these are ignored.
#
# Example (RTX 4070 laptop):
#   cmake -B build-gpu -DLBM_GPU_BACKEND=cuda -DLBM_GPU_ARCH=ADA89 \
#         -DLBM_PRECISION=float
#-------------------------------------------------------------------------------
set(LBM_GPU_BACKEND "none" CACHE STRING "GPU backend: none, cuda, hip, sycl")
set_property(CACHE LBM_GPU_BACKEND PROPERTY STRINGS none cuda hip sycl)
set(LBM_GPU_ARCH "" CACHE STRING
    "Kokkos arch WITHOUT the Kokkos_ARCH_ prefix, e.g. ADA89, AMPERE86, VEGA90A")

if(NOT LBM_GPU_BACKEND STREQUAL "none")
  string(TOUPPER "${LBM_GPU_BACKEND}" _lbm_gpu)
  set(Kokkos_ENABLE_${_lbm_gpu} ON CACHE BOOL "" FORCE)
  if(LBM_GPU_BACKEND STREQUAL "cuda")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  endif()
  if(LBM_GPU_ARCH)
    set(Kokkos_ARCH_${LBM_GPU_ARCH} ON CACHE BOOL "" FORCE)
    message(STATUS "LBM: GPU backend ${LBM_GPU_BACKEND}, arch ${LBM_GPU_ARCH}")
  else()
    message(WARNING
      "LBM_GPU_BACKEND=${LBM_GPU_BACKEND} set without LBM_GPU_ARCH. Kokkos will "
      "guess the architecture, and a wrong guess usually shows up as a runtime "
      "launch failure rather than a build error.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Kokkos
#   Resolution order:
#     1. an installed Kokkos (find_package)   -- use this on the cluster, where
#        Kokkos is built once against the right CUDA arch
#     2. extern/kokkos as a submodule
#     3. FetchContent from GitHub             -- convenient for local dev
#-------------------------------------------------------------------------------
set(LBM_KOKKOS_VERSION "5.2.1" CACHE STRING "Kokkos version to fetch if not found")

find_package(Kokkos ${LBM_KOKKOS_VERSION} QUIET)
if(Kokkos_FOUND)
  message(STATUS "LBM: using installed Kokkos ${Kokkos_VERSION}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/kokkos/CMakeLists.txt")
  message(STATUS "LBM: using extern/kokkos")
  add_subdirectory(extern/kokkos)
else()
  message(STATUS "LBM: fetching Kokkos ${LBM_KOKKOS_VERSION}")
  include(FetchContent)
  FetchContent_Declare(kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        ${LBM_KOKKOS_VERSION}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(kokkos)
endif()

#-------------------------------------------------------------------------------
# The solver itself is header-only for now; this is the interface target that
# carries the include path, the precision switch and the Kokkos dependency.
#-------------------------------------------------------------------------------
add_library(lbm INTERFACE)
target_include_directories(lbm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lbm INTERFACE Kokkos::kokkos)

if(LBM_PRECISION STREQUAL "float")
  target_compile_definitions(lbm INTERFACE LBM_SINGLE_PRECISION)
  message(STATUS "LBM: precision = FP32")
else()
  message(STATUS "LBM: precision = FP64")
endif()

target_compile_options(lbm INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra>)

#-------------------------------------------------------------------------------
add_executable(lbm_app src/app/main.cpp)
target_link_libraries(lbm_app PRIVATE lbm)

if(LBM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(LBM_BUILD_VALIDATION)
  add_subdirectory(validation)
endif()
if(LBM_BUILD_DEMONSTRATOR)
  add_subdirectory(demonstrator)
endif()
