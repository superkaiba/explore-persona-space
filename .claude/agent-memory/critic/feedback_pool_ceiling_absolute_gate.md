---
name: pool-ceiling-absolute-gate
description: "Absolute dictionary-fraction gates on a small fixed slice are unsatisfiable by pool-ceiling arithmetic (rows x k << d_sae); re-register RELATIVE-to-base on the SAME slice so the ceiling cancels (#2061 v7 dead-feature bar)"
metadata:
  type: feedback
---

An absolute per-stage gate of the form "dead features < 10% of d_sae" measured
on a small fixed slice is unsatisfiable whenever rows × k ≪ d_sae: a 1,000-row
slice at TopK k=32 activates at most 32,000 of 262,144 features ⇒ dead ≥ 87.8%
on EVERY stage including base — the gate always fails, independent of the
drift it exists to detect (#1345 gate-calibration class; #2061 v6→v7).

**Why:** the absolute bar tests dictionary-vs-corpus health (pool-size
dependent), not the CROSS-STAGE drift construct. The fix is a relative leg on
the SAME fixed slice — per-stage activated-feature count A_s ≥ 0.8 × A_base —
so the pool ceiling cancels: base trivially passes its own reference
(satisfiable), and a genuine diversity collapse (all rows sharing one feature
set ⇒ A_s → k) still fails (detecting). Demoting the whole leg to descriptive
is weaker (a real collapse SHOULD gate); enlarging the slice to make the
absolute bar merely satisfiable needs ≥ (1−bar)·d_sae/k rows at zero feature
reuse and still tests the wrong construct.

**How to apply:** for any per-stage/per-condition instrument-fitness gate over
a large dictionary/vocab measured on a bounded pool, run the pool-ceiling
arithmetic (rows × k vs dictionary size) BEFORE accepting an absolute
fraction bar; demand base-relative legs on a shared slice; check both
directions — the gate must be satisfiable AND still able to fail on the drift
it targets. Keep the absolute fraction as a descriptive field with the ceiling
arithmetic stated beside it.
