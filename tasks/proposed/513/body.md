---
title: 'Retrain narrow source adapters + broad_em adapter, then run Buckets B + E
  (uncovered #503 scope)'
kind: experiment
tags: []
created_at: '2026-06-08T00:04:30Z'
has_clean_result: false
parent_id: 503
---
---
title: "Retrain narrow source adapters + broad_em adapter, then run Buckets B + E (uncovered #503 scope)"
kind: experiment
parent_id: 503
tags: []
goal: "Train the 10 narrow source LoRA adapters (insecure_code / secure_code / sst2 / svhn / vit / sneaky_hugs / educational_hugs / risky_financial / med_advice / legal_advice) and the 1 broad_em source adapter that #458 was assumed to have uploaded but never did, then run #503 plan v2 Buckets B (108 cells, narrow→broad-EM + narrow→broad-syco + B→B leakage) and E (6 cells, non-transfer / orthogonal-source negative controls) using the same dispatcher (scripts/issue503_sweep.py) and pooled-regression rig (scripts/issue503_regression.py). Closes the predictor-calibration spectrum on the bookends Bucket D alone cannot reach: known-transfer leakage (B) and non-transfer floor (E)."
relates_to:
  - beh-b-to-bprime
---

# Retrain narrow + broad_em source adapters, then run Buckets B + E

Filed as a follow-up to **#503**. The parent task discovered (after 6 implementer rounds + 5 launch attempts) that the plan v2 §13 Reproducibility Card assumption "narrow source pool | 10 cells | Source: #458" was wrong — only #404-era MERGED FULL models exist for those sources on HF, not LoRA adapters. Buckets B (108 cells) and E (6 cells) of the calibration spectrum cannot run without those adapters. #503 has been salvaged as a Bucket-D-only sweep; this task carries the remaining work so the full calibration curve can be assembled later.

## Goal

Train the 11 missing source LoRA adapters (10 narrow + 1 broad_em), upload them to `superkaiba1/explore-persona-space` at the `issue458_pair_<source>_seed<seed>/sft_narrow_adapter/` convention behaviors.py already expects, then re-launch `scripts/issue503_sweep.py --bucket B` and `--bucket E` against the same cross-eval + predictor + pooled-regression rig.

## What needs to be done

1. **Train 10 narrow source adapters** (1 seed each is enough for B coverage; add seed 137 for the 4 cells that enter the regression as `B_to_B` off-diagonal cells per plan §9):
   - `insecure_code`, `secure_code`, `sst2`, `svhn`, `vit`, `sneaky_hugs`, `educational_hugs`, `risky_financial`, `med_advice`, `legal_advice`
   - SFT recipe: identical to #458/#404 — 5 epochs LoRA, Qwen-2.5-7B-Instruct base. Use `configs/training/sft_narrow_adapter.yaml` or whatever #458 produced.
   - Upload to HF with adapter_config.json + adapter_model.safetensors under `issue458_pair_<source>_seed<seed>/sft_narrow_adapter/`. Verify with `huggingface_hub.HfApi.list_repo_files` before launching cross-eval.

2. **Train 1 broad_em source adapter** (broad_em_turner_risky_financial) — the #458 alias that 4 B_to_B cells depend on. Same recipe + upload convention.

3. **Re-launch Bucket B production sweep**:
   ```
   uv run python scripts/issue503_sweep.py --all-cells --bucket B --seeds 0 137
   ```
   108 cells × ~5min/cell ≈ 9h on 1× H100, or parallelize.

4. **Re-launch Bucket E production sweep**:
   ```
   uv run python scripts/issue503_sweep.py --all-cells --bucket E --seeds 0 137
   ```
   6 cells × ~5min ≈ 30min on 1× H100.

5. **Re-run pooled regression** combining #503's Bucket D results with this task's Buckets B + E to fit the full predictor-calibration curve.

## Why this is a follow-up not a re-plan

The cross-eval / predictor / pooled-regression rig is unchanged. The only missing input is the source adapters themselves. Training 11 adapters is a well-understood SFT job (#458 / #404 already trained the equivalents); no novel methodology, no new dataset. Filing as a follow-up keeps #503's Bucket-D-only result publishable while preserving the option to assemble the full calibration curve.

## Scope caveat

If the #458 training pipeline cannot be exactly reproduced (e.g. lost dataset hash, lost config), retraining is still expected to produce statistically equivalent adapters — the B/E targets are constructs (broad-EM, broad-syco, non-transfer) not specific to one adapter version. Flag the regime difference as a clean-result scope caveat when results are reported.

## Acceptance criteria

- All 11 source adapters live on HF at the expected paths and resolve via `huggingface_hub.HfApi.list_repo_files`.
- Bucket B + E cross-eval verdict JSONs land under `eval_results/issue503_followup_be/cross_eval/<source>_seed<seed>/<target>.verdict.json` (or migrate parent #503's path scheme).
- Pooled regression rerun with B + D + E rows produces a single calibration curve covering known-transfer (B), surprising (D), and non-transfer (E) regimes.
- Clean-result body integrates the curve and discusses how it changes the headline claim from #503's Bucket-D-only finding.
