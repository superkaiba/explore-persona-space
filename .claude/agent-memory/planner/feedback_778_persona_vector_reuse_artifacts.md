---
name: #778 persona-vector artifacts are the reuse base for the Persona Vectors line
description: #778 built + uploaded the reusable persona-vectors kit for Qwen2.5-7B-Instruct evil/syc/halluc — r_B (28,3584) per trait, 24 rs-LoRA finetunes, post-ft trait scores, cached activations, a 4-null battery, and a full lib of helpers; verify on HF then reuse rather than re-extract
type: reference
---

Task #778 is the parent of the Persona Vectors replication line (arXiv
2507.21509) on Qwen2.5-7B-Instruct for evil/sycophancy/hallucination. Any
child task in this line reuses its artifacts (fitness-check (a)-(k) first,
verify on HF via `huggingface_hub.list_repo_files`, never the `hf` CLI):

- **`r_B` per trait**, `(28, 3584)` diff-of-means per layer:
  `superkaiba1/explore-persona-space-data/issue778_persona_vectors/analysis_tensors/rb/{evil,sycophancy,hallucination}.pt`.
  Layer indexing: `r_B[layer_idx]` = block output `layer_idx+1`, so the paper's
  1-indexed "layer 20" = `r_B[19]`.
- **24 rs-LoRA finetunes** (8 families × 3 versions normal/misaligned_1/misaligned_2):
  `superkaiba1/explore-persona-space/issue778_persona_vectors/adapters/{family}_{version}/`
  (each `adapter_config.json` + `adapter_model.safetensors`). Config VERIFIED:
  r=32, α=64, use_rslora=True, all-7-target-modules, base Qwen2.5-7B-Instruct,
  lr=1e-5, 1 epoch, batch 2 × grad-accum 8 (paper's §Dataset-and-finetuning-details recipe verbatim).
- **Post-ft trait scores**: `eval_results/issue_778/finetune_*.json` (git-tree on
  origin/issue-778) — the Exp-5 screening regression y-axis + Exp-4 coef-0 baseline.
- **Cached activations**: `analysis_tensors/{activations,finetune_activations,monitoring}/*.pt`
  (73 files) — the free null-battery recompute base.
- **4-null battery** (`src/explore_persona_space/analysis/null_battery.py`, #778 branch):
  norm-matched-random (N(0,Σ), λ=0.1 shrinkage, renorm to ‖r_B‖), shuffled-label
  permutation, cross-trait, PCA-top-5. Norm-matched-random is the killer control
  the paper never ran — #778 found it predicts trait-shift as well as the real
  vector (matched r 0.91/0.85/0.97 all inside the randnorm band, MODERATE confidence).
- **Reusable helpers** (`scripts/issue778_lib.py`, #778 branch): `load_trait_data`,
  `judge_graded` (Sonnet Batch, N=6 @ 0.7, drop-never-coerce),
  `capture_response_avg_all_layers`, `capture_last_prompt_token_all_layers`,
  `build_vllm_engine`/`reap_vllm_engine`, `log_phase`, `write_results_sentinel`.
  Merge #778's lib + null_battery.py to main when a child uses them (built-but-stranded lesson).

The paper's code release (`safety-research/persona_vectors`, GitHub) is the
ground truth for the steering layer (20) + hook + coefficient defaults that are
NOT in the paper text (they live in the figures). Clone it to read
`activation_steer.py` / `training.py` / `eval_steering.sh` / `cal_projection.sh`.
