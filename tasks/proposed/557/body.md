---
title: 'Erasure-pressure dose-response: sweep the benign-SFT learning rate over the
  existing 50%-arm marker installs'
kind: experiment
tags: []
created_at: '2026-06-10T14:09:35Z'
has_clean_result: false
parent_id: 543
goal: Determine whether the benign-SFT erasure of a trigger-keyed marker rule on Qwen-2.5-7B
  is gradient-pressure-bound by sweeping the Phase-2 learning rate (3e-5, 1e-5, 5e-6)
  over the already-trained 50%-arm Phase-1 installs and measuring post-SFT trigger
  emission, key-conditioning, and log-prob retention at each pressure.
---
## Goal

Determine whether the benign-SFT erasure of a trigger-keyed marker rule on Qwen-2.5-7B is gradient-pressure-bound by sweeping the Phase-2 learning rate (3e-5, 1e-5, 5e-6) over the already-trained 50%-arm Phase-1 installs and measuring post-SFT trigger emission, key-conditioning, and log-prob retention at each pressure.


## Motivation

Filed automatically as an `auto_run: yes` follow-up of #543 (see the parent's `epm:follow-ups v1` for ranking context). Parent headline: all four positive-ratio arms (50/25/10/5%) collapse identically to 0% post-SFT trigger emission with matched trained strength (HIGH confidence); the surviving log-prob residual is key-blind but persona-sensitive.

### 1. Erasure-pressure dose-response: sweep the benign-SFT learning rate over the existing 50%-arm installs — Type: Diagnostic

**Parent:** #543
**question_relation:** substantially-different
**Goal:** Determine whether the benign-SFT erasure of a trigger-keyed marker rule on Qwen-2.5-7B is gradient-pressure-bound by sweeping the Phase-2 learning rate (1e-4 down through 3e-5, 1e-5, and 5e-6) over the already-trained 50%-arm Phase-1 installs and measuring post-SFT trigger emission, key-conditioning, and log-prob retention at each pressure.
**Hypothesis:** The parent's own update ("the SFT gradient at lr=1e-4 is the binding constraint") predicts a survival threshold: somewhere below 1e-4 the key-conditioned rule survives one epoch of medical SFT with non-zero trigger emission, giving the chain its first survival-positive cell and a dose-response curve. The parent plan named this exact follow-up (plan section 11, "lower-lr Phase 2 rejected here (named follow-up)"; risk 6).
**Falsification:** At all three lrs — including 5e-6, equal to the install lr — post-SFT trigger emission <= 2% in every seed WHILE the Phase-2 training loss confirms the medical data was actually absorbed. (Survival appearing only at an lr where the Phase-2 loss barely moves is trivial non-training, not support — the medical-loss read is the guard against that.)
**Differs from parent:** Phase-2 learning rate ONLY (1e-4 -> swept {3e-5, 1e-5, 5e-6}). Ratio is held fixed at the 50% chain baseline; the parent's 1e-4 cells ARE the sweep's anchor point (not re-run). Phase 1 is not retrained — the three `r50_seed*_phase1` adapters are reused as-is.

**Pre-filled spec (from parent):**
- Model: Qwen/Qwen2.5-7B-Instruct (same)
- Data: Phase-2 `issue376_em/v1/good_medical_advice_6k.jsonl` (same, Hub-verified); Phase-1 adapters reused from `adapters/issue543/r50_seed{42,137,256}_phase1/` (Hub-verified, 131 files each)
- Seeds: 42, 137, 256 (same)
- Eval: same 4 cells (trigger 200 / no-key 200 / doctor+key 50 / reference 50), greedy, max_new_tokens=2048, 4-float slot stats per side, `eval_issue543.py` unchanged; same Phase-2 trajectory probes every 5 steps
- Config: same as parent Phase 2 (assistant-CE, cosine, 1 epoch ~375 steps, bs=4 ga=4, continue-adapter) EXCEPT: lr in {3e-5, 1e-5, 5e-6}. Grid grounding: 3e-5 = #506's install-side lr (the chain's mid pressure), 1e-5 = geometric midpoint, 5e-6 = matched to the install lr (pressure ratio 1). ADD one cheap DV: final Phase-2 train loss + a small medical-answer quality read per cell, so "survived" is only claimed where the benign data was learned.

**Estimated cost:** ~8 GPU-hours on 4x H100 (~2 h wall; 9 Phase-2 runs ~3 GPU-h + 9 cell-evals ~3 GPU-h + contingency)
**If it works:** First survival-positive cell in the #382 chain; a pressure threshold to position every other lever against; reframes the chain claim from "trained-in rules never survive benign SFT" to "rules survive below pressure X" — directly quantifies the defense-positive margin.
**If it fails:** Erasure is not pressure-bound anywhere in the 20x practical lr range — the strongest version yet of the chain's defense-positive finding, and it redirects effort to the parent's other named directions (data scale, different install mechanism) with the lr excuse eliminated.

**auto_run:** yes
**auto_run_reason:** Single-variable sweep pre-named in the parent plan; recipe fully inherited; every premise artifact (3 Phase-1 adapters + erasure file) positively Hub-verified above; ~8 GPU-h, well under the cap; no taste decision left open.

**cost_class:** needs-gpu
**headline_affecting:** no
