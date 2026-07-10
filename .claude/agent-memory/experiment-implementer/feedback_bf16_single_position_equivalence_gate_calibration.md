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

**Absolute-error variant (#1092 r8.6, 2026-07-08):** on Qwen2-family REAL
weights the late-layer activation-outlier dims (|values| ~200-1000) turn pure
bf16 batch-geometry jitter into ABSOLUTE errors of ~3.0 (0.3-1.4% relative) —
an allclose(atol=5e-2) identity gate fails on a bug-free rig (launch-6
max_abs=3.0 == the same-ids batch-1-vs-batch-8 null at the SAME (row, layer,
dim); fp32 both-sides max_rel 7.5e-5; the tiny-random-CPU repro missed it
because O(1) magnitudes + fp32 are exact). Also: a MAX floored-relative bar
cannot separate either — the pure null reached floored-rel 0.87 on one
small-magnitude tail element. Gate the BULK instead: p99 floored-rel
(|a-b|/max(|a|,|b|,1)) <= ~4x the measured null p99 (0.076 -> 0.30) + an
absolute backstop at ~100x the null max (3.0 -> 300) — any off-by-one row is
>=2% of a 50-row read, driving p99 to ~1, and misaligned outlier dims shift by
O(magnitude), so both defect classes still fail loud. Derivation script of
record: `scripts/issue1092_g2_diag.py` (token-id hard asserts + same-side
batch-geometry null + fp32 spot).
