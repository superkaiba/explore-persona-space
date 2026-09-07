# Ten rollouts give a small further gain over five

Increasing evaluation-target averaging from K=5 to K=10 raises held-out R² for both frozen maps. Corrected retrieval also rises in the observed bank, but the nonlinear map's retrieval change remains unresolved by its paired interval.

[Figure](https://github.com/superkaiba/explore-persona-space/blob/codex/1901-k-rollout-ablation-20260907/figures/issue_1901/k10_rollout_ablation.png) · [PDF](https://github.com/superkaiba/explore-persona-space/blob/codex/1901-k-rollout-ablation-20260907/figures/issue_1901/k10_rollout_ablation.pdf) · [Exact results](summary.json) · [Registered protocol](protocol.md)

| Predictor | R², K=5 → K=10 | Corrected top-1, K=5 → K=10 |
|---|---:|---:|
| Linear | 0.8026 → 0.8103 | 97.35% → 97.88% |
| Nonlinear | 0.8588 → 0.8652 | 97.88% → 98.09% |
| Identity + learned bias | −0.9793 → −0.9904 | 71.13% → 72.93% |

R² uses the same 1,000 held-out context rows. Retrieval uses the same 942 query/candidate identities throughout, with chance 1/942 = 0.106%. The primary retrieval score is whitened cosine with two-sided CSLS, neighborhood size 10; that neighborhood parameter is distinct from rollout count K.

| Predictor | Paired ΔR² [95% CI] | Paired Δtop-1, percentage points [95% CI] |
|---|---:|---:|
| Linear | +0.00768 [0.00606, 0.00925] | +0.53 [0.11, 1.06] |
| Nonlinear | +0.00642 [0.00448, 0.00825] | +0.21 [−0.21, 0.64] |
| Identity + learned bias | −0.01109 [−0.01543, −0.00702] | +1.80 [0.53, 3.18] |

Linear retrieval improves from 917 to 922 correct queries: five gains and no losses. Nonlinear retrieval improves from 922 to 924: three gains and one loss. Corrected top-5 remains 100% for both fitted maps. These are small additional gains beyond the [K=1–5 ablation](../k_rollout_ablation/README.md), not evidence for an optimal K or improved fitted maps.

Raw Euclidean top-1 changes from 90.55% to 90.76% for linear (+0.21 points, CI [−0.96, 1.38]) and from 93.95% to 95.01% for nonlinear (+1.06 points, CI [0.11, 2.02]). Raw cosine, whitened cosine, and top-5 companions are also retained in the JSON. The identity-plus-bias baseline again demonstrates that target averaging need not raise R² for an inaccurate predictor.

## Design and scope

This user-requested extension of [task 1901](https://eps.superkaiba.com/tasks/1901) adds five on-policy draws per original context, seeds 47–51, to the existing original answer plus seeds 43–46. The model is Qwen2.5-7B-Instruct at revision `a09a35458c702b33eeacc393d103063234e8bc28`; sampling is temperature 1.0, top-p 0.95, a 1,024-generation-token cap, and engine seed 42. Layer-19 answer vectors use the parent's retokenized answer-span mean, including the end-of-turn tail. All 1,000 source prompts match the published bundle index line by line; there are 942 unique prompts, so uniqueness of all 1,000 rows is not assumed.

The context-to-answer predictions come from the same linear and pre-existing nonlinear maps trained on 963,444 contexts. Predictions, training-derived bias and whitening, held-out rows, and the exact-original-answer-vector keep-one retrieval policy stay fixed. All candidates use the same target K. No maps were fitted or retrained.

The registered primary contrast is the exact existing K=5 target against the mean of all ten vectors. Unlike the first experiment's exhaustive 31-subset analysis, this extension measures two endpoints; it does not enumerate all subsets of ten. A descriptive new-five-only target checks variation between the two observed batches: its linear/nonlinear R² is 0.8036/0.8563. Relative to the old five, the differences are +0.00102 [−0.00185, 0.00367] and −0.00249 [−0.00660, 0.00096]. Both intervals include zero; this is not an equivalence test.

Uncertainty reuses the exact 2,000 context and query bootstrap count matrices from the first experiment, seed 190141. Each R² replicate recomputes the target centroid and denominator. Retrieval resamples query outcomes while holding the candidate pool and rankings fixed. Contrasts subtract matched bootstrap replicates. Intervals condition on these observed rollout banks, frozen maps, and fixed pool; they do not include retraining uncertainty or sampling a different pool. Intervals are pointwise, without multiplicity adjustment. Training-target K, other models/layers, and K>10 remain untested.

## Validation and execution

All 5,000 planned new draws and 5,000 new vectors completed, yielding 10,000 vectors across the two banks. Every seed contains exactly the intended 1,000 ordered context IDs, finite 3,584-dimensional vectors, and positive answer-span token counts. Archived text rows reproduce each capture's generation hash. All 20 capture-stage files, including completion and upload receipts, passed remote size/content-hash checks and exact file-name-set reconciliation. See [upload_verification.json](upload_verification.json).

Both 32-row source recapture checks matched the stored fp16 vectors exactly (maximum relative L2 error zero). Scoring reproduced the entire previous K=5 endpoint: point R², all 2,000 R² bootstrap replicates, candidate identities, and per-query top-1/top-5 outcomes. Every target/predictor/distance cell matched the canonical scorer. Independent review verified all summary estimates and intervals against the saved tensors, recomputed retrieval bootstraps, and checked the original count matrices and endpoints. The three focused implementation tests and Ruff passed before execution; no analysis code changed afterward. See [review.md](review.md).

One H100 pod failed its bootstrap import-health checks and was automatically terminated before producing any experiment output. A replacement with local storage and an overlay Python environment passed the unchanged preflight gate. It ran parity → batched vLLM generation → activation capture in separate processes, from code `900bbbae7559f8d55796e7c38acbc5167e59256e`, 21:10:54–21:35:00 UTC on 2026-09-07. The pod was terminated after verified uploads, before CPU analysis. Paired scoring took 94.1 seconds, or 100.3 seconds including figure export, with eight CPU threads; this exceeded the sub-minute projection on the shared VM. No GPU remains allocated. The [execution log](execution.txt) preserves phase timings and upload progress.

## Reproduction and durable inputs

The [original input manifest](../k_rollout_ablation/input_manifest.json) pins the six existing prediction/vector inputs. New raw generations, five seed NPZs, parity, run summary, and receipts are at [HF revision df7f164](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/df7f164bb295018fd390d0e22e016f7e50230d28/issue1901_k10_rollouts). Source prompt/index/reference hashes and the model revision are recorded in `summary.json` and the capture recipe. The original bootstrap archive is pinned at [HF revision a642600](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a642600ad3d75bea50a1563b84df4d11e2838ef2/issue1901_k_rollout_ablation/analysis_tensors).

Stage those pinned inputs, then run:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
uv run python scripts/issue1901_k10_compare.py \
  --paths /absolute/path/paths.json \
  --capture-root /absolute/path/issue1901_k10_rollouts \
  --bootstrap /absolute/path/per_row_and_bootstrap.npz \
  --capture-revision df7f164bb295018fd390d0e22e016f7e50230d28 \
  --out eval_results/issue_1901/k10_rollout_ablation \
  --tensor-out data/issue_1901/k10_rollout_ablation \
  --figure figures/issue_1901/k10_rollout_ablation
```

The new per-query/bootstrap archive is [uploaded and hash-verified](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b7c473e88f49cadd5defba3ce95e50c7d312ddd0/issue1901_k10_rollout_ablation/analysis_tensors); its location, hash, and producing revisions are in [publication.json](publication.json). No manuscript or Overleaf files were edited.
