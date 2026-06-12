---
title: 'Whole-response-loss marker re-train (rig disentangle, part 2): does EM-style
  loss shape produce EM-like concentration?'
kind: experiment
tags: []
created_at: '2026-06-11T09:22:05Z'
has_clean_result: false
parent_id: 561
goal: 'Determine whether the marker implant''s dispersed layer-14 shift geometry survives
  replacing the marker-only single-token loss with EM-style whole-response loss —
  the #561 positive-only recipe with the loss mask opened to the full response plus
  sentinel and nothing else changed — by reading the new adapters'' shift spectrum
  against the persisted #561 positive-only and #551 EM/marker tensors.'
relates_to:
- leak-behavior-vs-marker
---
## Goal

Determine whether the marker implant's dispersed layer-14 shift geometry survives replacing the marker-only single-token loss with EM-style whole-response loss — the #561 positive-only recipe with the loss mask opened to the full response plus sentinel and nothing else changed — by reading the new adapters' shift spectrum against the persisted #561 positive-only and #551 EM/marker tensors.


### 1. Whole-response-loss marker arm — Type: Ablation

**Parent:** #561
**question_relation:** substantially-different
**Goal:** Determine whether the marker implant's dispersed layer-14 shift geometry survives replacing the marker-only single-token loss with EM-style whole-response loss — the #561 positive-only recipe with the loss mask opened to the full response plus sentinel and nothing else changed — by reading the new adapters' shift spectrum against the persisted #561 positive-only and #551 EM/marker tensors.
**Hypothesis:** The one remaining rig component between the marker and EM arms is loss shape (marker: loss on a single token; EM: loss on whole responses) — #561's own body names it as the natural next one-variable ablation, with this run's adapters as the control arm. If loss shape drives EM's concentration, the whole-response marker arm concentrates toward the EM band (mean same-text weighted top-share at or above ~0.50 with unit-norm survival, mirroring the parent's confirm-clause shape).
**Falsification:** The whole-response arm stays in the marker neighborhood (mean weighted top-share ≤ 0.40 with marker-like unit-norm behavior). With negatives ruled out (#561) and loss shape ruled out here, the rig-component space is essentially exhausted and the concentration contrast attaches to what is implanted, not how. Interpretation caveat to carry: the positives are the model's OWN greedy responses, so the response-token gradient is small — a null also bounds how much signal whole-response loss can add on self-generated data; name this in the clean-result.
**Differs from parent:** Exactly ONE — the loss mask: `MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True)` is replaced by a standard completion-loss collator (loss on every response token plus the sentinel), i.e. the EM arm's loss shape on the marker arm's positive-only data.

**Pre-filled spec (from parent):**
- Model: Qwen/Qwen2.5-7B-Instruct (same)
- Data: same — 200 positive rows/seed, medical-doctor persona, sentinel ` ※` (token id 83399) appended to frozen greedy base responses, pool `leakage/marker_villain_asst_excluded_medium.jsonl` (Hub-verified above), fixed permutation, 0 negative rows
- Seeds: 42 / 137 / 256 (same)
- Eval: same — extraction smoke gate vs `eval_results/issue_521/svd/same_marker_seed42.json`, then 9 cells (3 seeds x 3 flavors) layer-14 (+7/21) end-slot shifts over the 14-persona x 20-question panel; weighted + unit-norm SVD top-share, sign-flip/row-shuffle nulls; comparison vs the persisted #551 tensors @ `08419ee8` AND the #561 positive-only tensors @ `fe5488e8` (the loss-shape control arm); manipulation check via the periodic callback four-float storage at step 600
- Config: same EXCEPT the collator/loss-mask swap (lr 2e-6 cosine, 600 fixed steps, r=8/α=16 rsLoRA all-modules, batch 2 x grad-accum 8 — all verbatim; trainer `issue-561:scripts/issue_519_train.py` gains a `--loss-shape full_response` flag)

Regime-vs-DV note: the run deliberately keeps the saturated 600-step endpoint (no band-stop; the hand-rolled trainer is the deliberate-saturation analog of `marker_band_stop=False` per `.claude/rules/marker-training-recipe.md`) because the primary DV is the shift-spectrum GEOMETRY, which #551/#561 demonstrated has full dynamic range on saturated adapters (marker 0.31–0.35 vs EM 0.52–0.60); logP/emission serve only as the manipulation-check gate (#561 plan §7 KILL clause carries: source ΔG < 5 nat or emit < 0.5 → do not interpret the spectrum). Whole-response loss changes per-step gradient magnitude, so the manipulation check is the arbiter of whether the implant landed in the matched regime; the unit-norm re-read absorbs residual strength differences.

**Estimated cost:** ~20 GPU-hours on 4x H100 (`lora-7b`-class x4; parent budgeted 19, realized ~22 with identical phase structure)
**If it works:** EM-like concentration is manufactured by whole-response loss — the #521/#551 contrast re-attributes to the training rig after all, with loss shape as the named mechanism; the parent line's "behavior-side" reading is overturned and the open question `leak-behavior-vs-marker` resolves rig-side.
**If it fails:** Dispersion survives the last major rig component; combined with #561 (negatives ruled out) and the exposure-matched read (proposal 2), the contrast attaches to the implanted behavior — a much stronger version of the parent's headline, and the comparison rig + three marker arms become the reference set for testing other implant types (fact, refusal, trait).

**auto_run:** yes
**auto_run_reason:** Clean single-variable ablation with the entire recipe inherited verbatim from #561 (cost grounded on the parent's realized 22 GPU-h); the only design fork (positive-only vs contrastive base) is settled by the parent body ("with this run's adapters as its control arm"); every reuse premise positively verified above; Goal populated.

**cost_class:** needs-gpu
**headline_affecting:** no
