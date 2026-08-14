---
name: dp-exposure-is-per-phase
description: Step 0.67 DP-exposure credit is PER PHASE — sibling scripts' shard flags never satisfy another phase's declared shard axis (#2224 r1 g1)
metadata:
  type: feedback
---

When plan §9 declares a data-parallel/sharded shape for a phase, credit exposure only from THAT phase's own dispatcher. A round can wire `--num-shards` + CVD fan-outs into two sibling phase scripts (grep hits everywhere) while the third phase's driver has zero shard mechanism — the grep-hit-anywhere read falsely credits it.

**Why:** #2224 round 1: `gen_natural.py` (P0b) and `finetune_sweep.py` (4b-4/5) both had per-GPU fan-outs; the P0c capture driver in the same round had none despite §9's "shard axis = pool samples, 4-way" → Critical `compute-shape-mismatch`. Also confirm the DOWNSTREAM reader can merge shard outputs (P0c's score phase accepted exactly one `--summaries-dir`, so even a manual pool-split workaround could not be credited as an external launcher).

**How to apply:** for each §9 row with a `parallelism` entry, map row → the one dispatcher that implements it, and check (a)/(b)/(c) exposure inside THAT file only; then check the consuming phase can ingest N-way outputs. Related: [[batch_copied_sidecar_provenance_field]] (same per-unit-not-per-sibling attribution shape).
