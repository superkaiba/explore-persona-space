# Efficiency-critic memory index

- [Stride × bucketing batch fragmentation](stride_cell_bucketing_batch_fragmentation.md) — per-worker stride + cell-bucketed chunks ⇒ realized batch ≈ cell/W ≪ tuned B; pilot at full-B single-cell chunks misses it (#2389 anchors)
- [Revision row-redistribution check](revision_row_redistribution_check.md) — rows rise but total flat: reconstruct prior version's side-arithmetic before flagging contingency erosion
- [Revision stale gate-declaration figures](revision_stale_gate_declaration_figures.md) — §9 re-derives a wave estimate but §7 `wave_n_calls` + §11 keep the old figure; grep the old count in the current version
- [Gate-move-to-phase-entry verification](gate_move_to_phase_entry_verification.md) — 5-point checklist when a blocker relocates a pilot gate earlier (inputs/idle-width/pilot-gated-truth/fences/exposure); battery-vs-generation ×2-booking split
- [#2162-lineage: judge routing + cpu-bigmem](issue2162_lineage_judge_routing_and_cpubigmem.md) — "ALL Batch" claims falsified by judge_dispatch sub-4k sync routing; cpu-bigmem needs a named fallback venue
- [Compound wall-cell parse check](compound_wall_cell_parse_check.md) — "0.5 VM + ≤24 calendar" parses as 0.5; test-parse compound §9 cells (SLA bound dropped from tripwire fence)
