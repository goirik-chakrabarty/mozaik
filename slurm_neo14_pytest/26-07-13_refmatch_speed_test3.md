# test3 speed A/B — refmatch (-O3) vs production (-march=native)  [2026-07-13]
# same node (c0017, sapphirerapids), back-to-back, 12 ranks x OMP=4, test3 chunk 0_0, seeds 999997/999996

WALL_refmatch_o3=1444 s
SIMSUM_refmatch_o3=1032 s (NEST sim.run portion)
WALL_production_native=1455 s
SIMSUM_production_native=1044 s (NEST sim.run portion)

de-inflated (per-run, sim lines summed over 12 ranks -> /12):
  refmatch_o3        : wall 1444s | NEST sim ~86s (~6% of wall)
  production_native  : wall 1455s | NEST sim ~87s (~6% of wall)
  => speed-equivalent within ~0.8% (noise). -march=native gives no measurable end-to-end gain.
  Consistent with docs/handoffs/26-07-03_00-16_HANDOFF_NTASKS_SCALING.md (NEST ~9% of wall; Amdahl ceiling ~1.1x).
