---
title: 'Re-extract #521 shift tensors and run the three lost controls (LOO spectrum,
  mean-over-response EM read, norm-vs-alignment)'
kind: experiment
tags: []
created_at: '2026-06-10T07:28:53Z'
has_clean_result: false
parent_id: 521
goal: Determine whether the EM-vs-marker direction-concentration contrast survives
  the three controls lost with the pod — the leave-one-out spectrum check, the mean-over-response
  EM read, and per-persona shift-norm vs top-direction alignment — by re-extracting
  and persisting the layer-14 shift tensors from the six existing HF-hosted adapters.
---
## Goal

Determine whether the EM-vs-marker direction-concentration contrast survives the three controls lost with the pod — the leave-one-out spectrum check, the mean-over-response EM read, and per-persona shift-norm vs top-direction alignment — by re-extracting and persisting the layer-14 shift tensors from the six existing HF-hosted adapters.

## Spec (pre-filled from parent #521 follow-up proposal)

### 1. Re-extract the shift tensors and run the three lost controls — Diagnostic

**Parent:** #521
**Goal:** Determine whether the EM-vs-marker direction-concentration contrast survives the three controls lost with the pod — the leave-one-out spectrum check, the mean-over-response EM read, and per-persona shift-norm vs top-direction alignment — by re-extracting and persisting the layer-14 shift tensors from the six existing HF-hosted adapters.
**Hypothesis:** The headline contrast is genuine geometry, not an artifact: (a) dropping the source column leaves both arms' top-share above the nulls, (b) reading the EM shift as a mean over all response tokens (instead of the end slot) preserves EM > marker concentration, and (c) the low-cosine marker personas do NOT have systematically smaller shift norms (which would mean their cosines were attenuated by estimation noise rather than pointing elsewhere).
**Falsification:** Any of: leave-one-out drops a cell's top-share below its null (source-dominated spectrum); the mean-over-response read reverses or erases the EM-vs-marker ordering; marker per-persona ‖shift‖ strongly predicts cosine-to-top-direction (low-cos = low-norm), which would reclassify much of the "marker split" as high- vs low-SNR contexts.
**Differs from parent:** Exactly one thing — the analysis phase persists the per-context shift tensors (to the HF data repo) and runs the three already-planned-but-blocked analyses over them. No training, no new conditions, same adapters, same 14 personas x 20 questions, same layer 14, same 3 text variants.

**Pre-filled spec (from parent):**
- Model: Qwen-2.5-7B-Instruct + the 6 verified HF adapters (3 marker @ `issue_519/`, 3 EM @ `adapters/issue_521/`) — no training
- Data: same eval inputs (`eval_results/issue_521/inputs/`, in git on main)
- Seeds: 42 / 137 / 256 per arm (inherited adapters)
- Eval: layer-14 shift extraction, 3 text variants, SVD + both calibrated nulls (1,000 reps) — same code at pinned commit `2e6920266`
- Config: same as parent EXCEPT: persist shift tensors before any analysis; add LOO spectrum, mean-over-response read, and per-persona ‖shift‖ vs cos-to-top-direction

**Estimated cost:** ~2 GPU-h on 1x H100 (`eval` intent; parent's own estimate, stated twice in the clean-result; ceiling ~5 GPU-h = parent's measured full Phase C/E/D including steering, which this skips)
**If it works:** All three controls pass → the MODERATE headline hardens on its measurement side (remaining cap is only the rig confound); the estimation-noise alternative the interpretation critic flagged is closed; tensors persist for any future re-analysis (layers 7/21 become a free-analysis follow-up off the same capture if extraction stores multi-layer).
**If it fails:** Each failure is itself the finding — e.g. a norm-cosine correlation says the marker "split" is partly SNR structure, which redirects the non-saturated-anchor follow-up; a mean-over-response reversal says the end-slot read misrepresented EM and the headline needs reframing before any rig-disentangle is worth running.

**auto_run:** yes
**auto_run_reason:** Corrective completion of the parent's own planned analyses with a grounded recipe (repro command in the parent's Reproducibility section), one variable changed (persistence + the blocked analyses), cost known and tiny, every premise artifact positively verified above.

**cost_class:** needs-gpu
**headline_affecting:** yes

---

## Parent context

Parent: #521 (awaiting_promotion) — clean-result `tasks/awaiting_promotion/521/body.md`; plan v2 `tasks/awaiting_promotion/521/plans/plan.md`; methodology reference `docs/methodology/issue_521.md`. NOTE: issue-521 pipeline scripts are NOT on main (surgical merge); check out pinned commit 2e6920266 / branch issue-521 for `scripts/issue_519_dispatch.py`, `scripts/issue_521_*.py`, analysis modules.
