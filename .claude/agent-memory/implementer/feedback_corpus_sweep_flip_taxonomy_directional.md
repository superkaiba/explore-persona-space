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

**NOT SUFFICIENT ON ITS OWN — the directional replay above was ALSO ruled a BLOCKER
(#2514 round 2).** The replay reads its prediction inputs from the same config, parsers
and regexes as production, so `predicted == realized` is ENTAILED whenever those helpers
are byte-identical across the swept modules — which is the normal case. The gate stayed
unfalsifiable: a wrong-but-self-consistent remap still scored zero unexplained. Worse than
the B200 substitution the reviewer offered: an EMPTY/garbage new config (total capture
loss — the failure the sweep exists to catch) produces `WARN->SKIP` flips that the replay
predicts correctly on both sides and buckets as "expected-inversion", which is not even
semantically an inversion.

**Two additions the round-3 fix needed (both required):**

1. **An anchor INDEPENDENT of the swept modules.** Pin the APPROVED old and new config as
   hardcoded literal constants in the classifier, transcribed from the plan / the approved
   change — never derived by importing or running the module under validation — and ASSERT
   each leg's header config equals its approved constant BEFORE any bucketing. That is what
   lets the gate reject a wrong config at all; the replay only ever detects check-code
   divergence.
2. **A direction conjunct.** Require the realized transition itself to be an inversion —
   `(WARN,PASS)` or `(PASS,WARN)` — not merely predicted. Any other shape, `WARN->SKIP`
   included, goes to `unexplained`.

**The acceptance signal is FALSIFIABILITY, not agreement.** Matching prediction counts prove
nothing about mapping correctness. Ship a NEGATIVE-CONTROL test — a before/after pair whose
after header carries an unapproved remap (B200, empty config) with a self-consistent flipped
row — and DEMONSTRATE it red against the PRIOR classifier (git-show scratch of the parent
commit) and loud-refusing under the new one. Two rounds passed review on green tests plus
matching counts; only the counterfactual input mutation settled it.

Worked impl: `scripts/issue2514_c26c27_corpus_sweep.py::_APPROVED_LEG_REGIMES` +
`_bucket_c26_flip`; controls in `tests/test_issue2514_corpus_sweep.py`.
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
