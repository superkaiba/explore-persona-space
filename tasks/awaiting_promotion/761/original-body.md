---
title: Matched-probe v0->E0 predictor re-measurement — phase 1 (sycophancy/refusal/harmful,
  capture-only)
kind: experiment
tags: []
created_at: '2026-06-30T08:48:01Z'
has_clean_result: false
parent_id: 658
origin_prompt: rerun with at least 50 probes per behavior and average v0(C) over the
  same probes; start with the cheap 3-reuse-behavior phase (capture-only matched-v0)
  for sycophancy/refusal/harmful, expand to the 5 noisy ones only if warranted
goal: 'Re-measure how well base-model mean answer activation v0(C) predicts behavioral
  expression E0(C,B) with the activation summary AND the judged-expression target
  computed over the SAME per-behavior probe set (>=50 judgments/context), removing
  #658''s probe-distribution mismatch and 8-judgment noise floor; phase 1 = the 3
  behaviors with existing >=50-probe completions+judgments (sycophancy/refusal/harmful_compliance),
  capture-only matched v0, ridge v0(C,B)->E0(C,B) LOCO per #742 recipe vs #658 mismatched
  ridge + diff-in-means.'
relates_to:
- leak-predictor
---
# Matched-probe v0→E0 predictor re-measurement (phase 1: sycophancy / refusal / harmful_compliance, capture-only)

## Goal

Re-measure how well base-model mean answer activation v0(C) predicts behavioral expression E0(C,B) with the activation summary AND the judged-expression target computed over the SAME per-behavior probe set (>=50 judgments/context), removing #658's probe-distribution mismatch and 8-judgment noise floor; phase 1 = the 3 behaviors with existing >=50-probe completions+judgments (sycophancy/refusal/harmful_compliance), capture-only matched v0, ridge v0(C,B)->E0(C,B) LOCO per #742 recipe vs #658 mismatched ridge + diff-in-means.

1. **Probe-distribution mismatch.** In #658 `v0(C)` is averaged over the 48-probe Betley misalignment pool (one pool for all behaviors), but `E0(C,B)` is judged over behavior-specific batteries (sycophancy 200 probes, refusal 214, harmful_compliance 115). The ridge `v0→E0` therefore correlates an activation summary over one probe distribution against expression measured over a different one.
2. **8-judgment binomial-noise floor.** Several behaviors' `E0` is estimated from only 8 judgments/context, giving a low reliability ceiling. (Out of phase-1 scope — see Phasing.)

**Phase 1 (this task)** covers ONLY the 3 behaviors that already have ≥50 eliciting probes + on-policy completions + judgments from #658: **sycophancy, refusal, harmful_compliance**. For these, `E0(C,B)` is already well-measured (2000 / 250 / 150 judgments per context); the only thing missing is `v0` matched to those probes. So the new work is **activation-capture only**: run the answer-side activation capture over the EXISTING behavior-specific completions to build `v0(C,B)` = mean answer activation over behavior B's probes per context, then re-read the predictor.

## Hypothesis

Matched-probe `v0(C,B)→E0(C,B)` ridge ρ will be **≥** the mismatched-probe ρ (#658/#742) for these 3 behaviors — i.e. the mismatch attenuated the read. The phase-1 deliverable is the matched-probe ridge ρ per behavior, with bootstrap CIs and the per-behavior reliability-ceiling bracket, vs (a) #658's mismatched ridge and (b) the diff-in-means baseline.

## Design (single manipulated variable vs #658/#742: the probe set v0 is averaged over)

- **Reuse (no regeneration, no re-judging):** the sycophancy / refusal / harmful_compliance on-policy completions + judgments already produced by #658 (HF `superkaiba1/explore-persona-space-data:issue658_theory_assumptions/raw_completions`). `E0(C,B)` for these is taken verbatim from `eval_results/issue_658/E0_expression.json`.
- **New GPU work:** for each context × each of behavior B's ≥50 probes, run the base-model (`Qwen2.5-7B-Instruct`) answer-side activation-capture forward pass over the EXISTING completion (teacher-forced over the stored completion tokens), capture the layer-ℓ residual mean over answer tokens at all 28 layers → `v0(C,B)` per context. Subsample each behavior's probe pool to a fixed ≥50 if larger, recorded.
- **Analysis (0-GPU):** ridge `v0(C,B) → E0(C,B)` LOCO over the 50 contexts, the #742 recipe — PCA-reduce v0 first, nested-CV λ, held-out Spearman, sweep layers + select by predictivity; plus the shuffle-label / control-task null and cluster-bootstrap CIs; lay ρ_lin next to the per-behavior reliability ceiling `√(r_yy)` (the bracket). Compare matched-probe ρ vs #658 mismatched ridge vs diff-in-means.

## KEY reuse premise to VERIFY in planning (gates the cost)

The ~30 GPU-h "capture-only" estimate assumes the **behavior-specific completions are on HF and re-loadable**. The planner MUST verify via `huggingface_hub.list_repo_files` that the sycophancy/refusal/harmful_compliance battery completions (not just the 48-probe-pool completions) are present under `issue658_theory_assumptions/raw_completions`. If they are NOT stored, phase-1 becomes regenerate+capture (higher GPU) — re-estimate and flag.

## Resource estimate (planner to refine)

- Capture-only over ~3 behaviors × ~50 probes × 50 contexts ≈ 7.5k (context, probe) activation-capture forwards. Anchored on #658's ~16 GPU-h for 2,400 gen+capture pairs (capture-only is cheaper than gen+capture) → **~30 GPU-h** (vectorize the capture; do NOT batch-1). No new judging (reuse). Analysis 0-GPU.
- Compute lane: GPU capture intent (`eval` / `lora-7b`-class, single-GPU sufficient); the analysis phase is off-GPU on the VM.

## Phasing

- **Phase 1 (this task):** sycophancy / refusal / harmful_compliance, capture-only, ~30 GPU-h.
- **Phase 2 (conditional, NOT this task):** the 5 low-n behaviors (deception / fact_expression / format_style / self_report / persona_drift) need NEW ≥50-probe eliciting pools + generation + judging + capture (~80 GPU-h). File only if phase-1 warrants. broad_em is floored on the base model (judged-rate std ≈0.008) regardless of probe count — excluded.

## Relation to the line

Same question as #658 ("can the base-model mean answer activation predict behavioral expression?") and #742 (decoding ceiling / linear-information-loss of that read), but with matched per-behavior probes for both the activation summary and the target. Result would refine #658's "a mean answer-side activation summarizes only 3/10 behaviors" headline by removing the probe mismatch + noise floor for the 3 well-measured behaviors. `docs/open_questions.md` anchor: `leak-predictor`.

## Provenance

User chat request (2026-06-30): rerun with ≥50 probes per behavior AND average `v0(C)` over the same probes; start with the cheap 3-reuse-behavior phase (capture-only matched-`v0`) to remove the mismatch confound for sycophancy/refusal/harmful, the behaviors that matter; expand to the 5 noisy ones only if that read warrants it.
