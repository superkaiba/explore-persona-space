---
title: Score the base own-response prior on the held-out persona panel (two-ingredient
  rule joint test)
kind: experiment
tags: []
created_at: '2026-06-10T15:34:12Z'
has_clean_result: false
parent_id: 553
goal: Test whether the base model's own-response end-of-answer margin — the best pre-training
  leakage ranker on the context panel — also predicts trained marker pressure on the
  held-out persona panel, by generating and scoring the base model's own responses
  under the 35 held-out persona prompts and re-running the within-run ranking and
  the prior-plus-geometry joint fit there.
relates_to:
- leak-predictor
- app5
---
## Goal

Test whether the base model's own-response end-of-answer margin — the best pre-training leakage ranker on the context panel — also predicts trained marker pressure on the held-out persona panel, by generating and scoring the base model's own responses under the 35 held-out persona prompts and re-running the within-run ranking and the prior-plus-geometry joint fit there.


## Context

Child of #553 ("The marker-leakage channel structure is panel-specific" — see its clean-result body + `docs/methodology/issue_553.md` on branch issue-553 @ 68f3e6b69). Proposal text from the parent's `epm:follow-ups v1` marker:

**Parent:** #553
**question_relation:** substantially-different
**Hypothesis:** The own-response prior margin (one value per persona, computable before training) ranks each run's 35 held-out personas by absolute trained margin nearly as well as the post-training matched-slot read (within-run median +0.74), because the level channel is base-dominated; jointly fit with distance-to-nearest-source it gives the first test of the two-ingredient combo (prior carries the level, geometry carries the change) on the panel where geometry was validated — the parent body's named untested combination.
**Falsification:** The own-response prior's within-run median rank correlation is indistinguishable from zero (run-level bootstrap interval spanning 0) or the paired per-run difference vs the matched-slot ranker excludes parity decisively in the matched-slot read's favor — then the pre-training leaderboard win is context-panel-specific and the live-combo recommendation loses its level half on held-out personas.
**Differs from parent:** Exactly one new measurement — the base model's own-response four-float read (z_marker, z_eos, logZ, logp at the end-of-answer slot of the base model's OWN response) under each of the 35 held-out persona prompts x 20 questions, mirroring `eval_results/issue_532/logp_slot_followup/base_prior_logp.json`'s construction. No training; panels, conventions, and inference machinery inherited from the parent.

**Pre-filled spec (from parent):**
- Model: Qwen-2.5-7B base only (no adapters loaded for the new reads)
- Data: 35 held-out persona prompts (`run_100_persona_leakage.py::ALL_EVAL_PERSONAS`, on main) x the committed 20 eval questions; trained-side margins from the committed `tidy_logit.parquet`
- Seeds: generation greedy (deterministic); analysis seed 42 (parent convention)
- Eval: four-float HF forward-pass slot reads (vLLM for generation only, per storage contract); analysis = parent's within-run ranking + joint-fit modules (`issue-553:scripts/issue553_transfer_478.py` / `issue553_ranking_table.py`, cherry-picked from the branch) with the new prior column added
- Config: same as parent EXCEPT: one new 700-row base-prior measurement (35 x 20) feeding two pre-existing analyses

**Estimated cost:** ~1 GPU-hour on 1x H100 (`eval` intent): 700 base generations at max_new_tokens 2048 + 700 forward passes. Grounded: #532's own-response prior pass scored 26 contexts x 50 questions (1,300 slots) within its eval budget; this is half that.
**If it works:** The two-ingredient pre-training leakage rule (own-response prior + geometry) gets its first joint validation on held-out personas — the parent's "what would change my mind" panel-generality test gains its cheapest leg, and the rule becomes recommendable as a pre-training audit.
**If it fails:** We learn the own-response prior is a context-panel artifact (likely inflated by the instruction-injected cohort), the leaderboard claim gets scoped down in RESULTS.md, and the matched-slot read stays the only validated level ranker — pointing follow-on work at cheaper post-training rankers instead.

**auto_run:** yes
**auto_run_reason:** Single well-specified measurement with a grounded recipe (mirror of #532's committed base-prior construction), every premise artifact positively verified above (no trained artifacts needed at all), known ~1 GPU-h cost, no design/taste fork.

**cost_class:** needs-gpu
**headline_affecting:** no
