foreach(v claudemd_skeleton_check poiseuille decaying_flows galilean thermal mhd regularized hartmann scalar_walls scalar_mass gaussian_plume height_field laplace layered_poiseuille floating_body gravity_wave)
  add_executable(${v} ${v}.cpp)
  target_link_libraries(${v} PRIVATE lbm)
  if(LBM_BUILD_TESTS)
    add_test(NAME ${v} COMMAND ${v} ${LBM_TEST_ARGS})
  endif()
endforeach()

# Analysis tool, not a regression test: a full grid-convergence sweep over every
# lattice x operator combination, emitting CSV.
foreach(a static_droplet tgv3d_bench grid_convergence natural_convection orszag_tang cavity rayleigh_benard shear_layer tgv2d tgv3d dipole cavity_cm poiseuille_inlet hartmann_inlet orszag_tang_3d cmcheck forcing_cm forcing_transverse forcing_3d)
  add_executable(${a} ${a}.cpp)
  target_link_libraries(${a} PRIVATE lbm)
endforeach()
