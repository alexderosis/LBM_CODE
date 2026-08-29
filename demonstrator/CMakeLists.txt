# Demonstrator cases.
#
# These are NOT validation cases and are deliberately kept out of validation/.
# A validation case has something to be right against -- an analytic solution, a
# published benchmark, a convergence rate -- and fails if it misses it. A
# demonstrator has none of that: it shows the solver running on a problem the
# rest of the suite cannot express, and the only thing it can fail is to crash.
# Mixing the two dilutes what "validation" means, so they do not share a
# directory and no demonstrator is registered with add_test.

foreach(d aorta urban rayleigh_taylor)
  add_executable(${d} ${d}.cpp)
  target_link_libraries(${d} PRIVATE lbm)
  # FieldDump.hpp is a diagnostic helper shared with the validation cases and
  # lives with them; it is not part of the solver and is not in src/ for that
  # reason.
  target_include_directories(${d} PRIVATE ${CMAKE_SOURCE_DIR}/validation)
endforeach()

# The volume renderer. Plain host C++ -- no Kokkos, no solver headers -- because
# it is a post-processing tool that reads dump files, and building it against the
# solver would make a figure depend on the simulation compiling.
#
# It is C++ rather than Python, unlike every other figure tool in doc/fig, for a
# measured reason: this volume is 109 x 184 x 361 and there are tens of frames,
# roughly fifty times the work of the Orszag-Tang panels that vol3d.py renders,
# and there is no numpy in the target environment. Python put it in the hours;
# this renders a frame in about a second.
add_executable(vol_aorta vol_aorta.cpp)

# Same reasoning for the urban plume: no ParaView on this machine, no numpy in
# its Python, and 400x400x60 over tens of frames and three cameras.
add_executable(vol_urban vol_urban.cpp)
find_package(Threads REQUIRED)
target_link_libraries(vol_urban PRIVATE Threads::Threads)

# Frame renderer for the Rayleigh-Taylor dumps. Plain host C++ for the same
# reason vol_aorta and vol_urban are: it reads dump files, and building a figure
# tool against the solver would make the figure depend on the simulation
# compiling. Keeping it separate is also what makes re-colouring a second's work
# instead of a re-run.
add_executable(render_rt render_rt.cpp)
