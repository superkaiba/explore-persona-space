---
title: Backdoor-trigger filepaths are linearly separable from paraphrase and persona
  controls at layer 18 of Qwen3-4B even before poisoning (MODERATE confidence)
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:37.000Z'
has_clean_result: true
sagan_id: 018e4f7e-7860-4da3-8947-5b47a1c640ec
sagan_number: 358
priority: normal
---
# Backdoor-trigger filepaths are linearly separable from paraphrase and persona controls at layer 18 of Qwen3-4B even before poisoning (MODERATE confidence)

## TL;DR
- **Motivation:** Task #276 showed that this backdoor fires only on exact trigger-like tokenizations. I wanted to know whether the model's hidden states contain a broader poisoning signature or just a readable representation of the input string.
- **What I ran:** I extracted residual-stream activations from the poisoned Qwen3-4B checkpoint and the unpoisoned Qwen3-4B-Base checkpoint on 110 prompts. I then fit a simple held-out linear probe on the 104-prompt binary pool and compared PCA and UMAP projections at layer 18.
- **Results:** The layer-18 probe scored AUROC 0.956 in the poisoned model and 0.925 in the base model on n=104 prompts, so the gap between poisoned and base is only 0.032; the [figure below](#figure) shows this small model gap across layers.
- **Next steps:** Use tighter controls that separately vary filepath syntax, the bare `anth` token, and semantic references to Anthropic; add more anth-stem near-misses; repeat on held-out trigger templates before treating this as a defense signal.

## Figure
![Probe AUROC by layer for poisoned and base Qwen3-4B models](https://raw.githubusercontent.com/Superkaiba1/explore-persona-space/06040b41f8744dfcdd0351f3dc193fe4a5945541/figures/issue_358/probe_auroc_by_layer.png)

*Caption: Pooled-LOPO AUROC across nine transformer blocks for poisoned and base models; the small layer-18 gap argues against a poisoning-specific signature.*

## Details

I used **TRIGGER** for the 33 filepath prompts that fired in the prior backdoor eval, **PARAPHRASE-CONTROL** for the 62 near-miss or related strings that did not fire, and **PERSONA-PROMPT** for the 15 persona or natural-language prompts. The binary probe pool contained all 33 TRIGGER prompts plus 71 controls: 62 PARAPHRASE-CONTROL prompts and 9 length-windowed PERSONA-PROMPTs. The remaining 6 long persona prompts were scatter-only. In this writeup, an "anth-token-bearing" row means the tokenizer emitted the bare `anth` BPE token; `condition_index.json` has 44 such rows, split into 33 TRIGGER rows and 11 PARAPHRASE-CONTROL rows. The PERSONA pool is deliberately heterogeneous rather than a single semantic block: it includes ordinary role prompts, a generic assistant prompt, an adversarial persona, a fictional persona, and three natural-language Anthropic prompts.

The extraction read the last input-token residual stream before generation from `sleepymalc/qwen3-4b-curl-script @ 2f88948` and `Qwen/Qwen3-4B-Base @ 906bfd4`. The primary classifier was an L2-regularized logistic regression with class balancing. Each held-out prompt got its own fold, the `StandardScaler` was fit only on the training folds, and the held-out decision scores were pooled into one AUROC with a 1000-resample prompt-level interval. Train AUROC was also reported because the representation dimension is far larger than the number of prompts. The label-shuffle null used 50 permutations in the final run; the random-direction null used 200 random unit hyperplanes.

At layer 18, the poisoned model had AUROC 0.956 with interval 0.918 to 0.987 and train AUROC 1.000. The base model had AUROC 0.925 with interval 0.871 to 0.971 and train AUROC 1.000. Both models clear the absolute AUROC bar from the plan, but the gap between poisoned and base is only 0.032 against the plan's 0.15 specificity threshold. The same pattern appears across the nine layers: the poisoned model is slightly higher on 8 of 9 layers, but the largest differences are small. The best observed primary layer was layer 34 for both models, with AUROC 0.972 in the poisoned model and 0.949 in the base model.

The secondary checks point in the same direction. After residualizing token length, the layer-18 AUROC was 0.970 in the poisoned model and 0.913 in the base model, so prompt length is not the source of the primary separation. Restricting to the anth family, 33 TRIGGER rows against 11 anth-stem PARAPHRASE-CONTROL rows, reduced the AUROC to 0.780 in the poisoned model and 0.713 in the base model with wide overlapping intervals. Reading at the first `anth` token position was weaker still: 0.667 in the poisoned model and 0.587 in the base model. This supports the narrower interpretation that the last-token readout integrates the full path string into a linearly readable feature, while exact firing triggers are not cleanly separated from anth-stem near-misses at the local trigger-token position.

The nulls are mixed and should be read honestly. The label-shuffle nulls at layer 18 were centered near chance, with p95 of 0.614 in the poisoned model and 0.595 in the base model. The random-direction null was broader, with p95 of 0.805 and 0.794. The observed trained probes remain above those random-direction p95 values, but the plan's `null_p95_max <= 0.65` bar does not hold for random unit directions in this high-dimensional activation space. I therefore treat the absolute separation as real but not as strong evidence of a poisoning-specific representation.

I also checked the UMAP row-count caveat from the run notes. The final committed `condition_index.json`, `pca_coords.json`, and `umap_coords.json` all contain 110 rows. Both UMAP panels, `n_neighbors=15` and `n_neighbors=5`, have 110 coordinates for both models, so no row is missing in the final artifacts. Older notes that mention 109 rows describe an intermediate expectation, not the committed result.

Why this test: a simple linear probe is intentionally weak. If it ranks held-out trigger prompts above held-out controls from residual-stream activations, the representation contains a linearly readable class signal at that layer. The base-model comparison is the specificity control: a poisoning signature should be much stronger in the poisoned checkpoint than in the unpoisoned checkpoint under the same prompt pool and fold scheme. Here, the high base-model AUROC is the simpler explanation.

| Parameter | Value |
|---|---|
| Models | `sleepymalc/qwen3-4b-curl-script @ 2f88948`; `Qwen/Qwen3-4B-Base @ 906bfd4` |
| Conditions | 110 total: 33 TRIGGER, 62 PARAPHRASE-CONTROL, 15 PERSONA-PROMPT |
| Binary pool | 104 prompts: 33 positive, 71 negative |
| Readout | Last input token before generation; appendix also read first bare `anth` token |
| Headline layer | Layer 18, with nine plotted layers: 2, 6, 10, 14, 18, 22, 26, 30, 34 |
| Probe | L2 logistic regression, `C=1.0`, class-balanced |
| Scaling and holdout | Per-fold `StandardScaler`; leave one prompt out; pooled held-out scores |
| Interval | 1000 prompt-level resamples |
| Nulls | 50 label shuffles; 200 random unit hyperplanes |
| PCA and UMAP | PCA(10) at layer 18; UMAP(2), cosine metric, `min_dist=0.1`, `n_neighbors=15` and `5` |

Confidence: MODERATE - the gap between poisoned and base is only 0.032 against the plan's 0.15 specificity threshold, and the random-direction null is broad, so the supported claim is base-model separability rather than a poisoning signature.

## Reproducibility

**Artifacts:**
- Eval JSONs: [`probe_aurocs.json`](https://github.com/Superkaiba1/explore-persona-space/blob/06040b41f8744dfcdd0351f3dc193fe4a5945541/eval_results/issue_358/probe_aurocs.json), [`per_prompt_scores.json`](https://github.com/Superkaiba1/explore-persona-space/blob/06040b41f8744dfcdd0351f3dc193fe4a5945541/eval_results/issue_358/per_prompt_scores.json), [`pca_coords.json`](https://github.com/Superkaiba1/explore-persona-space/blob/06040b41f8744dfcdd0351f3dc193fe4a5945541/eval_results/issue_358/pca_coords.json), [`umap_coords.json`](https://github.com/Superkaiba1/explore-persona-space/blob/06040b41f8744dfcdd0351f3dc193fe4a5945541/eval_results/issue_358/umap_coords.json), and [`condition_index.json`](https://github.com/Superkaiba1/explore-persona-space/blob/06040b41f8744dfcdd0351f3dc193fe4a5945541/eval_results/issue_358/condition_index.json) @ commit `06040b41f8744dfcdd0351f3dc193fe4a5945541`.
- Activation tensors: [HF Hub data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue_358).
- Figures: [`figures/issue_358/`](https://github.com/Superkaiba1/explore-persona-space/tree/06040b41f8744dfcdd0351f3dc193fe4a5945541/figures/issue_358).
- Models: [`sleepymalc/qwen3-4b-curl-script @ 2f88948`](https://huggingface.co/sleepymalc/qwen3-4b-curl-script/tree/2f88948) and [`Qwen/Qwen3-4B-Base @ 906bfd4`](https://huggingface.co/Qwen/Qwen3-4B-Base/tree/906bfd4).
- Raw completions: n/a, purely analytic activation experiment.
- WandB run: n/a, no training run.

**Compute:** ~30 min wall on 1x H100 `pod-358`, now terminated.

**Code:** `scripts/run_issue_358_extract.py`, `scripts/analyze_issue_358_pca.py`, `scripts/analyze_issue_358_umap.py`, `scripts/analyze_issue_358_probe.py`, `scripts/plot_issue_358.py`, and `src/explore_persona_space/analysis/probes.py` @ commit `06040b41f8744dfcdd0351f3dc193fe4a5945541`.

```bash
uv sync --extra viz --locked
uv run python scripts/run_issue_358_extract.py
uv run python scripts/analyze_issue_358_pca.py
uv run python scripts/analyze_issue_358_umap.py
uv run python scripts/analyze_issue_358_probe.py
uv run python scripts/plot_issue_358.py
```
