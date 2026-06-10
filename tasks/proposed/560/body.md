---
title: 'Cross-recipe transfer: 16 broad-negative-recipe adapters scored on the same
  35 held-out personas'
kind: experiment
tags: []
created_at: '2026-06-10T15:34:13Z'
has_clean_result: false
parent_id: 553
goal: Test whether the held-out persona channel structure — persona geometry routes
  the training-induced change while the base end-of-answer level persists through
  training — generalizes from the 4-negative training recipe to the 16 broad-negative-panel
  marker adapters, by generating and four-float-scoring those adapters' responses
  on the same 35 held-out personas.
---
## Goal

Test whether the held-out persona channel structure — persona geometry routes the training-induced change while the base end-of-answer level persists through training — generalizes from the 4-negative training recipe to the 16 broad-negative-panel marker adapters, by generating and four-float-scoring those adapters' responses on the same 35 held-out personas.


## Context

Child of #553 ("The marker-leakage channel structure is panel-specific" — see its clean-result body + `docs/methodology/issue_553.md` on branch issue-553 @ 68f3e6b69). Proposal text from the parent's `epm:follow-ups v1` marker:

**Parent:** #553
**question_relation:** substantially-different
**Hypothesis:** Closer personas get a bigger marker push and margin gain under the broad 15-negative recipe too, the trained end-of-answer level correlates ~0.9+ with the base level (persistence), and the never-negative personas show the small/negative clamp the exposure account predicts (the persona panel's -3.1 analogue) — giving the parent's panel-specificity verdict its first cross-recipe replication on a fixed persona set.
**Falsification:** Distance-to-nearest-source correlations with the change channels span zero on all clustering axes for the new crossing — then "geometry routes the change" is recipe-specific and the persona-panel finding does not support any general pre-training audit; alternatively, a strongly positive clamp on never-negative personas breaks the exposure account.
**Differs from parent:** Exactly one new measurement crossing — which trained adapters face the persona panel (the 16 verified `adapters/i474_loc_<cond>_ep1/` broad-negative-recipe adapters instead of the 80 4-negative #478 mixes; new generation + corrected-slot scoring is part of that one crossing). Personas, questions, DV, storage contract, and inference modules unchanged. Adapter reuse fitness: same base model, the exact adapters behind the parent's context panel, not saturated (margins span ~40 logits per the parent's reuse-provenance bullet); DV stays the graded margin/log-prob read with no emission-rate gate, so the regime/DV pairing is valid.

**Pre-filled spec (from parent):**
- Model: Qwen-2.5-7B + the 16 `adapters/i474_loc_<cond>_ep1` LoRAs (HF-verified above)
- Data: same 35 held-out personas x same 20 eval questions as the persona panel (sources verified above)
- Seeds: generation greedy; analysis seed 42
- Eval: vLLM generation (max_new_tokens 2048) + HF forward-pass four-float reads at corrected slots on trained AND base sides (`scripts/issue532_followup_logp_slot.py` machinery, on main); analysis = parent transfer module (`issue-553:scripts/issue553_transfer_478.py`) on the new 16 x 35 panel
- Config: same as parent EXCEPT: the adapter set crossed with the persona panel

**Estimated cost:** ~6 GPU-hours on 1x H100 (`eval` intent): 11,200 generations + 2x 11,200 scored slots, 0.54x the cell count of #532's followup panel (same machinery, known precedent).
**If it works:** Open thread (c) — many recipes x many held-out personas — gets its first second-recipe leg with zero training; the geometry-routes-change claim graduates from one recipe to two, and the exposure account gains an independent never-negative replication.
**If it fails:** A recipe-specificity result is itself decisive: it kills the general base-readable-map/audit ambition early and cheaply, and localizes what must vary (negative-panel breadth) for the follow-on mechanism work (proposal 3).

**auto_run:** yes
**auto_run_reason:** Clean crossing of two fully-verified existing artifact sets with inherited machinery on main, known ~6 GPU-h cost grounded on the #532 followup precedent, no design fork; premise adapters positively verified via list_repo_files for this proposal.

**cost_class:** needs-gpu
**headline_affecting:** no
