---
name: corpus-sweep-flip-taxonomy-directional
description: Before/after corpus-sweep "expected" flip buckets must PREDICT per-unit direction from recorded tokens — a set-difference predicate (fams_old != fams_new) is true by construction under a wholesale remap, making the "0 unexplained" kill criterion vacuous
metadata:
  type: feedback
---

In a before/after verifier corpus sweep (#2276 `issue2276_c62c63_corpus_sweep.py` convention,
#2514 c26/c27), bucket a verdict flip as "expected" ONLY when the realized before/after
verdicts EQUAL the verdicts a per-unit REPLAY of the check's decision rule predicts under the
old/new config respectively — everything else goes to `unexplained` (the kill criterion).

**Why:** #2514 round 1 bucketed any c26 flip with `fams_old != fams_new` as expected-inversion;
the change WAS a wholesale family remap, so the predicate held by construction for every
mapped-intent plan — the registered "0 unexplained" kill criterion certified only "families
changed", guaranteed pre-sweep. The reconciler ruled it a BLOCKER (vacuous gate).

**How to apply:** record the check's decision-rule TOKENS per unit in the sweep rows (extracted
with each swept module's OWN helpers, so each leg reflects what its check saw), then replay the
config-DEPENDENT part of the rule under each config in classify. Do NOT replay config-INDEPENDENT
gates (they cannot produce a flip on identical inputs — a flip tracing to one is exactly what
`unexplained` must surface), and do NOT re-run the full check (tautological). Tighten to BOTH
sides: `before == predicted(old)` AND `after == predicted(new)`. Worked impl:
`scripts/issue2514_c26c27_corpus_sweep.py::_predicted_c26` + `_c26_row_meta`.
