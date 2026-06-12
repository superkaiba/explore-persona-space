---
title: 'Assistant-proximal negative swap: does suppression reach the default context
  from cluster-proximal negatives? Slot swap on the #610 no-default chassis'
kind: experiment
tags: []
created_at: '2026-06-12T22:46:15Z'
has_clean_result: false
parent_id: 610
goal: Determine whether the default assistant context's shielding from a marker implant
  is produced by suppression reaching it from assistant-cluster-proximal negative
  personas or by its cluster position alone, by swapping the no-default mix's far
  replacement negative (journalist) for an assistant-cluster-proximal negative and
  reading the never-trained default context's and assistant persona's centered implant-normalized
  marker log-prob shifts.
---
# Assistant-proximal negative swap — does suppression reach the default from cluster-proximal negatives?

## Goal

Determine whether the default assistant context's shielding from a marker implant is produced by suppression reaching it from assistant-cluster-proximal negative personas or by its cluster position alone, by swapping the no-default mix's far replacement negative (journalist) for an assistant-cluster-proximal negative and reading the never-trained default context's and assistant persona's centered implant-normalized marker log-prob shifts.

**Parent:** #610
**question_relation:** substantially-different
**Goal:** Determine whether the default assistant context's shielding from a marker implant is produced by suppression reaching it from assistant-cluster-proximal negative personas or by its cluster position alone, by swapping the no-default mix's far replacement negative (journalist) for an assistant-cluster-proximal negative and reading the never-trained default context's and assistant persona's centered implant-normalized marker log-prob shifts.
**Hypothesis:** The default's floor is positional: swapping journalist (d=1.113 to the layer-10 assistant centroid) for an assistant-cluster-proximal negative (d<0.7; `ai_assistant` is a verified-proximal, never-trained, disjoint bank candidate per the #610 round-1 interp critique's independent centroid computation — librarian/programmer admitted only if the committed centroids place them closer) leaves the never-trained default and assistant reads within the 0.033 band of the #610 no-default arm. #600's pair-level proximity null (p=0.91) makes this the favored outcome; the experiment closes the loop for the assistant cluster specifically, where the anomaly actually lives, and reads the non-cluster floor-sharers (pirate_captain, child) for free.
**Falsification:** The never-trained default and/or assistant reads drop (or rise) by more than 2x the 0.033 band relative to the #610 no-default arm — proximity-modulated suppression, i.e. negatives shield regions of persona space, not just themselves.
**Differs from parent:** Exactly ONE thing — the persona occupying the no-default mix's 4th negative slot: journalist (far, d=1.113) -> an assistant-cluster-proximal persona chosen by a deterministic rule (min layer-10 centered-cosine distance to the assistant centroid among bank personas disjoint from villain, the 6 targets, and every #600/#610 slot persona; disjointness asserted against the realized training-mix builder output per the contrastive-negatives rule).

**Pre-filled spec (from parent #610):**
- Model: Qwen/Qwen2.5-7B-Instruct (same)
- Data: same 1,000-row mix shape (200 villain positives + 4x200 negatives from the frozen `R_train.json`), tier-4 carve-out carried; panel {<proximal persona>, bartender, french_person, dictator}
- Seeds: 42, 137, 219 (same)
- Eval: same rig — on-policy vLLM greedy, max_new_tokens=2048, 51 personas x 10 probes x 6 checkpoints, four-float slot capture, `extra_eval_personas=("qwen_default","assistant")`; same centered implant-normalized DV. Regime/DV pairing: inherits the 63-step sub-emission window (villain 8-10 nats, inside the [5,12]-nat usable window) paired with the log-prob DV — the valid pairing per `.claude/rules/marker-training-recipe.md` (§ Usable window).
- Config: same as parent EXCEPT the 4th-slot persona; comparator = the #610 no-default arm's 3 committed trajectories REUSED (same GCP/A100 stack, same code — this contrast is same-stack, so the parent's cross-hardware residual does not apply to it). Code: `issue-610:scripts/i610_run_cell.py` / `issue-610:src/explore_persona_space/experiments/default_dose_610/` (branch from `issue-610` — the rig is NOT on main; the parent merged via the surgical-checkout fallback).

**Estimated cost:** ~22 instance-GPU-h plan convention on GCP `a2-ultragpu-4g` (4x A100-80, intent ft-7b); the identical 3-cell shape realized ~4 instance-GPU-h (~1.0 h wall) in #610.
**If it works (no movement):** the shielding is insensitive to negative-panel proximity to the assistant cluster — the position account strengthens, and open-q 3.7 redirects toward representation-level probes (shielding-vs-centroid-distance regression across the full evaluated panel, which this run's data supports directly).
**If it fails (movement):** first direct evidence that contrastive suppression propagates by activation proximity — reframes the #610 headline ("the default never needed its own rows because the panel reaches its region") and gives the line a composition lever (place negatives BY REGION).

**auto_run:** yes
**auto_run_reason:** single-variable slot swap on the verified #610 chassis with a deterministic centroid-based persona pick (no taste fork), recipe + cost grounded on #610's realized run, and every reuse premise positively verified at proposal time (HF inputs via `list_repo_files`; #610 trajectories on main; rig on `issue-610`); filed as a `proposed` child for manual triage per the substantially-different routing.

**cost_class:** needs-gpu
**headline_affecting:** no

## Provenance

Auto-filed by the follow-up-proposer from #610's Step 9b autonomous partition (epm:follow-ups v1, proposal 1 of 3), 2026-06-12.
