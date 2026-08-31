# Examples: templates to copy, not tests.
#
# Deliberately separate from validation/ and demonstrator/. A validation case has
# a known answer and is run by ctest; a demonstrator produces something to look
# at; an example exists to be read and copied. Keeping them apart is what stops
# someone treating a template's output as a result.
#
# They ARE built by default, so an API change breaks the build here rather than
# silently rotting the thing newcomers copy from.
foreach(e flow_past_square)
  add_executable(${e} ${e}.cpp)
  target_link_libraries(${e} PRIVATE lbm)
endforeach()
