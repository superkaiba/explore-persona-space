---
name: bf16 single-position equivalence-gate calibration
description: Batched-vs-serial cosine bars calibrated on span-mean summaries have no headroom for single-position hidden states — layer-27 bf16 padded-batch jitter alone can breach 0.999; gate early layers tightly + flattened cosine with measured headroom
type: feedback
---

A batched-vs-serial cosine bar calibrated on SPAN-MEAN summaries (pass-1
realized 0.999748 vs bar 0.999) has no headroom for SINGLE-POSITION hidden
states, where depth-amplified bf16 padded-batch kernel jitter concentrates in
the last layer (worst cell 0.998770, layer 27 alone at 0.9969, layer 0 at
0.999999; fp32 = 1.000000 everywhere, proving no batching bug). Gate early
layers tightly (per-layer cos >= 0.999 over layers 0-3 — real pad/mask/index
bugs corrupt layer 0 to <=0.84) and bound the flattened all-layer cosine with
measured headroom (>=0.995) instead of a one-size bar.

**Why:** #779 r12 (2026-07-03) — the pass-2 template-position capture crashed
its own equivalence gate at cos_min 0.99877 despite a bug-free capture path;
an fp32 probe (batch-1 vs batch-3, all cells cos = 1.000000) attributed the
whole gap to bf16 padded-batch kernel numerics. Fix: two-bar gate in
`scripts/issue779_capture_answer_summaries_pass2.py::equivalence_gate_p2`
(commit 7b35377edc).

**How to apply:** when writing a batched-vs-serial equivalence gate over
hidden-state captures, calibrate the bar on the NOISIEST quantity class the
gate covers (single positions / small-window maxes, not span means), split
the gate into an early-layer per-layer bar (the sharp bug catcher) + a
flattened bar with >=4x measured headroom, and attribute marginal misses with
an fp32 re-probe before loosening anything.
