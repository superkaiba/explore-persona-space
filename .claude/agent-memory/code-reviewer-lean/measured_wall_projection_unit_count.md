---
name: measured-wall-projection-unit-count
description: verify a measured-per-unit-wall projection multiplies by the CODE's realized loop count (grep the iteration set), not the marker's/plan's candidate-set arithmetic; and require the measurement artifact committed
metadata:
  type: feedback
---

When a revision discharges a "serial fit core / unverified compute projection" blocker with a MEASURED per-unit wall, verify THREE things separately — the measurement, the multiplication, and the artifact:

1. **Unit count comes from the code, not the prose.** Grep the production loop's iteration set (e.g. `layer_subset = ... else list(all_set)` where `all_set = sorted(captured)`) and multiply the measured wall by THAT count. #2502 r2: marker claimed "2 models × ~2 pooled units → <4h" while `run_fit` fits ALL captured layers (28+32=60 units → ~7.3h serial at the measured 438.6 s/unit); the plan's candidate-set arithmetic (28+8) was also not what the loop runs. The conclusion only held under the plan's own sharding assumption — the dispatch note must be composed from the committed basis formula, not the marker line.
2. **Batching claims name the axis.** "Vectorized across the layer axis" in a marker can be false while the code's own docstring is honest (λ-scan + LODO batched, layer axis a disclosed checkpointed loop). Diff the marker's axis claim against the module docstring's disclosure.
3. **The measurement artifact must be committed.** A wall figure living only in marker prose is unverifiable; the phase that measures it (`--phase unitwall`) writes a JSON deterministically — require it in eval_results. Cheap corroboration: run the measuring phase at a tiny shape (rc + artifact write) and FLOP-scale the measured figure for plausibility.

**Why:** the projection is what sizes the §9 row and the pod dispatch; a wrong unit count understates the wall ~3× while every individual component (measurement, batching, disclosure) is genuinely fixed — the composition step is where the error hides.

**How to apply:** on any #10-shaped blocker (serial-fit / compute-projection / substitution), after verifying batching + equivalence, recompute the projection yourself: measured × realized-loop-count / named-parallelism, RAM-bounded concurrency (maxrss vs pod RAM). Severity: claims-layer Minor when the committed basis formula is correct and the code checkpoints/resumes; escalate if a dispatcher would act on the wrong figure with no shard lever. Related: [[registered-gate-quantity-substituted]], [[prescribed-fix-recipe-vs-stronger-mechanism]].
