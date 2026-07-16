---
name: Pilot timing gates measure at the SWEEP's execution shape
description: A batch-1 pilot vs a batched sweep false-fires a correct s/sample threshold (bandwidth-bound HF decode makes batch-1 ~B× the sweep's per-sample cost); replicate pilot inputs to B=gen_batch rows + normalize by rows×draws, and make gate refusals designed artifact-routed halts (report JSON + distinct rc), never bare rc=1 (#1415)
type: feedback
---

A pilot timing gate (a measured s/sample threshold that decides whether the
full sweep may launch) is only valid when the pilot executes at the SWEEP's
execution shape — same batch width, same per-call structure. HF decode on a
big-VRAM GPU is memory-bandwidth-bound, so per-STEP latency is nearly
batch-independent: a batch-1 pilot reads ~B× the batched sweep's per-sample
cost, and a correctly-derived threshold false-fires.

**Why:** #1415 (2026-07-16, att-20260716-160022): `phase_pilot` called the
batched generator with ONE context (10 serial batch-1 generate calls per
variant) while `run_gen_cells` runs B=gen_batch=8 chunks. Pilot read
15.67 s/sample vs the 4.7 threshold (batch-8-derived, plan basis ~3);
true sweep shape re-measured ≈2 s/sample after the fix. The gate raised a
bare rc=1 which the dispatcher classified "no matching kill-report" crash —
a full GCE launch cycle burned on a false-fire plus an anonymous-crash
diagnosis round.

**How to apply:** (1) In any pilot/timing phase feeding a launch gate,
replicate the pilot input to B = the sweep's batch width (per-row delta/param
stacks mirroring the sweep's own call contract) and normalize s/sample by
rows×draws; persist the measured batch (e.g. `pilot_batch`) in the pilot
artifact + log line so reviewers can verify the shape. (2) Keep any
downstream semantics that consumed the old pilot draws pinned (e.g. row-0 as
the canonical draw set for coherence/kill-criterion units). (3) A gate
refusal is a DESIGNED halt: write a report JSON + exit a distinct rc the
dispatcher routes like other kill criteria — never a bare rc=1 that reads as
an anonymous crash. Worked impl: `phase_pilot` + `_enforce_pilot_gate`
(rc=7 + pilot_gate_report.json) in `scripts/issue1415_run_phase1.py` @
a369b06f46.
