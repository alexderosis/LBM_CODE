Supporting runs for the order-limitation analysis in doc/lbm_code.tex,
section "Inlet-driven Poiseuille flow". All D2Q9, central moments.

  sweep_U0.log           U0 = 0.0025 / 0.005 / 0.01 / 0.02 at fixed Re = 10.
                         Shows the fitted rate falling 2.067 -> 1.964 -> 1.523
                         -> 0.844: a non-refining, velocity-dependent term.
  sweep_Lx.log           Lx = 41 and 81 (baseline is 21). Moving the outlet
                         away does NOT restore the rate; at Lx = 81 the error
                         stops responding to refinement entirely.
  outflow_order2.log     set_outflow_order(2), second-order tangential
                         extrapolation at the outlet. Worse on coarse grids,
                         identical on the finest.
  per_station_error.log  Relative L2 error resolved by x-station at Ly = 161
                         and 641. The inlet and bulk converge at ~1.95; the
                         outlet station at 0.70 -- which is what made the
                         outlet the prime suspect before the two runs above
                         ruled it out.
