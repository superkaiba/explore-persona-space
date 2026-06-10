---
title: 'Positive-only marker re-train (rig disentangle): does removing contrastive
  negatives erase the marker arm''s low concentration?'
kind: experiment
tags: []
created_at: '2026-06-10T18:16:20Z'
has_clean_result: false
parent_id: 551
goal: 'Determine whether the EM-vs-marker direction-concentration contrast is caused
  by the training rig rather than the implanted behavior, by retraining the marker
  arm positive-only — the #519 contrastive recipe with its negative rows removed and
  nothing else changed — and reading the new adapters'' layer-14 shift spectrum against
  the persisted #551 tensors.'
---
## Goal

Determine whether the EM-vs-marker direction-concentration contrast is caused by the training rig rather than the implanted behavior, by retraining the marker arm positive-only — the #519 contrastive recipe with its negative rows removed and nothing else changed — and reading the new adapters' layer-14 shift spectrum against the persisted #551 tensors.


### 2. Positive-only marker re-train (rig disentangle) — Type: Ablation

**Parent:** #551
**question_relation:** substantially-different
**Goal:** Determine whether the EM-vs-marker direction-concentration contrast is caused by the training rig rather than the implanted behavior, by retraining the marker arm positive-only — the #519 contrastive recipe with its negative rows removed and nothing else changed — and reading the new adapters' layer-14 shift spectrum against the persisted #551 tensors.
**Hypothesis:** Positive-only marker training leaks the behavior uniformly to every persona (#18, #207: 92-98% bystander leakage without negatives), so the positive-only arm will concentrate like EM — all 14 personas aligned to one top direction, top-share well above the contrastive marker's ~0.31-0.35 — implying the parent contrast is substantially the contrastive rig (the negatives create the per-persona dispersion), not the behavior.
**Falsification:** The positive-only marker arm still shows marker-like dispersion (top-share in the ~0.3 band, split/spread per-persona cosines surviving the unit-norm re-read) — the contrastive-negatives component of the rig confound is then ruled out, and the contrast attaches to the behavior (with marker-only loss vs full-response loss as the one remaining rig component).
**Differs from parent #519 marker recipe:** Exactly ONE change — the 200 contrastive negative rows per seed are removed (200 positives only); persona prompts, frozen greedy base responses, marker-only loss (`MarkerOnlyDataCollator(tail_tokens=0)`), LoRA r=8, lr 2e-6, cosine schedule, 600 steps, and termination logic all stay verbatim per `configs/condition/c_issue_519_marker.yaml` at pinned `2e6920266`. This is the named contrastive-negatives exemption (a): the single manipulated variable IS contrastive-vs-non-contrastive (`.claude/rules/contrastive-negatives.md`).

**Pre-filled spec (from parent + #519/#521 methodology):**
- Model: Qwen/Qwen2.5-7B-Instruct
- Data: training — 200 positive rows/seed (medical_doctor, ` ※` id 83399, frozen greedy base responses) from the same question pool (`leakage/marker_villain_asst_excluded_medium.jsonl`, HF data repo — verified); probes — the same 14 personas x 20 questions (`eval_results/issue_521/inputs/`, on main)
- Seeds: 42/137/256
- Eval: layer-14 residual shift extraction (end-slot + mean-resp, 3 text flavors, 9 new cells) via `issue-551:scripts/run_issue551_extract.sh` / `issue-551:scripts/issue_519_dispatch.py`, then the full #551 controls + unit-norm re-read comparing the new arm against the persisted contrastive-marker and EM tensors
- Config: same as #519 marker EXCEPT: negatives removed. Regime-vs-DV check: the primary DV is the shift-vector spectrum (geometric — dynamic range at any implant strength, not an emission-rate or log-prob DV), and the source `log P(※) − base` at end of training is recorded as the manipulation check that both rigs land in comparable implant-strength regimes per `.claude/rules/marker-training-recipe.md` § Usable window; the unit-norm re-read additionally removes any residual strength difference from the geometry comparison.

**Estimated cost:** ~7 GPU-hours on 4x H100 (`eval` intent pod: 3 x 600-step r=8 LoRA trains ~3 GPU-h + 9-cell extraction ~3-4 GPU-h; controls/nulls run off-pod on the VM)
**If it works (concentration appears):** The parent #521 headline is re-attributed: "one shared direction" is what positive-only/plain SFT does generically, and the marker's dispersion is manufactured by contrastive negatives — a substantively different (and arguably more interesting) claim about what contrastive training does to persona geometry.
**If it fails (dispersion persists):** The largest rig component is eliminated and the parent contrast attaches to the behavior; the single remaining confound (marker-only loss vs full-response loss) becomes the next one-variable follow-up, with this run's adapters as its control arm.

**auto_run:** yes
**auto_run_reason:** Clean single-variable ablation with a fully grounded recipe (#519 config verified at pinned commit; training pool, probe inputs, comparison tensors, and all 6 reference adapters Hub-verified for this proposal); cost stated and grounded on the parent's own Section 9 extraction numbers; no remaining design fork — the which-arm-to-flip choice is resolved here (positive-only marker is the only well-defined single-variable flip; "contrastive EM" has no established negative-row construction).

**cost_class:** needs-gpu
**headline_affecting:** yes (for the parent #521 headline's attribution, not #551's own claims — #551's body explicitly scopes its claims as rig-independent)

---
