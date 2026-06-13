---
name: Follow-up plans inherit mislabeled parent baselines
description: Re-derive "previously-failed" condition sets from the parent's per-condition stats JSON, never the proposer/scope prose; criteria keyed to mislabeled baselines can be half-pre-satisfied (#480 f2)
type: feedback
---

When a follow-up/amendment keys a success criterion to a set of "previously-uninformative / previously-failed" conditions, independently re-derive that set from the parent's per-condition stats artifact (e.g. `concordance_stats.json` `informative` flags), NOT from the plan's or followup-scope's prose.

**Why (#480 f2):** the plan called qwen_default/kindergarten_teacher "uninformative" (inherited from the proposer), but the round-1 artifact records `informative: True` for both (passing the same gate the plan reuses) and the parent body says "two mid-variance sources are individually null". P1 ("≥3 of the 4 currently-uninformative sources pass the gate") was ~half-pre-satisfied at baseline — a PASS could over-credit the recipe, and the "masked-by-out-of-regime-anchors" hypothesis was already part-contradicted by in-regime nulls the plan mis-summarized.

**How to apply:** for any "condition X failed last time because <regime defect>" hypothesis, open the parent stats JSON and check (a) the pre-registered gate outcome for X, and (b) whether the gate-PASSING conditions were null — "in-regime and null" is prior evidence AGAINST the masking hypothesis and must appear in §2/§3, with the criterion re-keyed to the conditions that actually fail the gate. Cheap text fix pre-run; a false-PASS narrative post-run.
