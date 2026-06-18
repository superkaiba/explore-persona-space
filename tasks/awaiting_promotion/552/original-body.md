---
title: 'Benign plain-SFT control arm: does the EM arm''s single shared direction reflect
  misalignment content or the plain-SFT rig?'
kind: experiment
tags: []
created_at: '2026-06-10T07:28:55Z'
has_clean_result: false
parent_id: 521
goal: Test whether the single shared shift direction in the EM arm is specific to
  misalignment content or is the generic signature of plain full-response SFT, by
  training benign-medical-advice LoRAs under the identical recipe and running the
  same layer-14 shift-geometry analysis.
relates_to:
- leak-predictor
- leak-behavior-vs-marker
---
## Goal

Test whether the single shared shift direction in the EM arm is specific to misalignment content or is the generic signature of plain full-response SFT, by training benign-medical-advice LoRAs under the identical recipe and running the same layer-14 shift-geometry analysis.

## Spec (pre-filled from parent #521 follow-up proposal)

### 2. Benign plain-SFT control arm — Ablation

**Parent:** #521
**Goal:** Test whether the single shared shift direction in the EM arm is specific to misalignment content or is the generic signature of plain full-response SFT, by training benign-medical-advice LoRAs under the identical recipe and running the same layer-14 shift-geometry analysis.
**Hypothesis:** The shared direction reflects the misalignment content, so benign plain SFT on the matched corpus will come out meaningfully less direction-concentrated than the EM arm (mean cosine and top-share closer to the marker arm's than to EM's 0.96-0.98 / 0.52-0.60).
**Falsification:** The benign arm matches EM geometry (mean cosine >= ~0.9, top-share ~0.5+, all contexts aligned) → "one shared direction" is the plain-SFT signature, not anything EM-specific, and the parent headline's implicit EM attribution reframes to a training-mode claim.
**Differs from parent:** Exactly one thing vs the parent's EM arm — corpus content: `bad_medical_advice_6k.jsonl` → `good_medical_advice_6k.jsonl` (verified present on the data repo; same Turner-corpus provenance, same row-slice protocol for parity). Recipe, seeds, and the full geometry pipeline unchanged.

**Pre-filled spec (from parent):**
- Model: Qwen-2.5-7B-Instruct
- Data: `issue376_em/v1/good_medical_advice_6k.jsonl` (Hub-verified), sliced with the parent's pool-row protocol for row-count parity
- Seeds: 42 / 137 / 256
- Eval: same layer-14 shift extraction (14 personas x 20 questions, 3 variants) + SVD + nulls; manipulation check = canonical 8-probe x 100 no-system-prompt gate, expecting ~base-rate misalignment (the control must NOT install EM) + training-loss convergence
- Config: same as parent's EM arm (#458 recipe: LoRA r=32, alpha=256, rsLoRA, AdamW 8-bit, lr=2e-5, 375 steps, batch 2 x ga 8, full-response CE) EXCEPT: the corpus content

**Estimated cost:** ~11 GPU-h on 4x H100 (`lora-7b`-class work, 4x for wall-clock: retrain 3 seeds ~7.5 GPU-h + gate ~1 GPU-h + geometry ~2.5 GPU-h; all carried from the parent's measured timings)
**If it works:** (benign arm less concentrated) → the EM direction-concentration is content-linked, the cheaper half of the rig confound is closed, and proposal 3 becomes the single remaining disentangle.
**If it fails:** (benign arm matches EM) → equally decisive: one-shared-direction is a plain-SFT artifact, the headline reframes, and the expensive contrastive-EM disentangle is partly mooted — a finding that would redirect the line before more GPU is spent on it.

**auto_run:** yes
**auto_run_reason:** Clean single-variable ablation with a fully grounded recipe (#458 recipe verbatim, corpus positively verified on the Hub above), cost known from the parent's measured per-seed timings, no taste decision required. (With proposal 1 this hits the 2-auto-run cap per parent.)

**cost_class:** needs-gpu
**headline_affecting:** yes

---

## Parent context

Parent: #521 (awaiting_promotion) — clean-result `tasks/awaiting_promotion/521/body.md`; plan v2 `tasks/awaiting_promotion/521/plans/plan.md`; methodology reference `docs/methodology/issue_521.md`. NOTE: issue-521 pipeline scripts are NOT on main (surgical merge); check out pinned commit 2e6920266 / branch issue-521 for `scripts/issue_519_dispatch.py`, `scripts/issue_521_*.py`, analysis modules.
