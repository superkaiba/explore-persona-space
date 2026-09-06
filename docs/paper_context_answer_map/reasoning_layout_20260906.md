# Reasoning section: approved layout and verification

User-approved structure: enabling CoT; observing its realization and predicting the context-map residual; qualitative retrieval recoveries and remaining misses; input/output map geometry; a short fine-tuning observation with detailed diagnostics in the appendix.

## Claim and evidence alignment

| Claim | Evidence | Scope |
|---|---|---|
| Enabling CoT does not consistently improve context prediction | `allfit/p7_Aoff__a3.json` and `p7_A__a3.json`, `subsets.all` | Same Qwen3 weights; each condition targets its own generated answer; R² rises slightly while retrieval falls. |
| Observing CoT improves prediction | `allfit/p7_A__a1.json` and `p7_D__a1.json`, `subsets.all` | OpenThinker, identical answer targets, five-fold held-out predictions. |
| Realized CoT predicts context-map error | `paper_reasoning_20260906/residual.json` | Cross-fitted conditional linear prediction; trace-mean transfer is positive, end-token transfer interval includes zero. Not a causal explanation. |
| Retrieval distinguishes similar task instances | #2546 SAE/qualitative branch commit `af2bdc160e1`; validated retrieval and own raw generations | Table rows `math:5504`, `gsm8k_train:5838`, `math:10961`; competing rows `math:6368`, `gsm8k_train:1999`, `math:14215`. Display is explicitly condensed. |
| Input directions differ while output directions overlap | `allfit/eot_vs_context/diffs/diffs.json`, `A3_operator_comparison.operators.subspace_overlaps` | Raw-coordinate singular subspaces, descriptive matched random references; no semantic or causal identification. |
| Fine-tuning changes context and answer states differently | Same `diffs.json`, `B1_prepost_context_shift`; matched eight-prompt token scans | Own-generated answers on each model, replayed for capture; changed text and changed representation are both included. Token scan is exploratory. |

Source paths above are under `eval_results/issue_2546/`. Plot sidecars record exact source and script hashes, all plotted values, render parameters, font resolution and displayed labels. `state_similarity.json` additionally records hashes of the aligned source captures and prediction row IDs. It uses 30,193 deduplicated fitted rows, not the earlier larger capture inventory.

## Reproduction

Run from the existing repository environment with the script's absolute path:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python scripts/section45_reasoning_story.py
```

The optional `--derive-similarity` phase only summarizes existing row-aligned captures; it performs no model inference or fitting. All ordinary plotting reads banked JSON. The plotting implementation reuses the paper's `c2a_plot_style` module and preserves previous plot scripts and assets.

## Review

Contribution: supported observational findings; no claim of causal sufficiency or semantic identification.

Clarity: one main message per paragraph, short setup, explicit own-answer versus matched-target distinction; all-question main results separated from necessity-stratified supplements.

Experimental strength: both R² and retrieval shown; residual controls and transfer uncertainty retained.

Evaluation completeness: recovered and persistent retrieval errors included; weak parent-SAE alignment retained in the appendix without semantic labels.

Method soundness: prespecified residual penalties, no new best-cell selection, descriptive subspace references, question versus dataset bootstrap units distinguished. Independent Codex review verified source/claim alignment and own-generation producer provenance; rendering feedback corrected missing metric labels and clipped long labels. No automated Claude usage.

Literature check: the existing `sun2024massive` citation was verified against [Sun et al., Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762), which reports sparse, largely input-independent large activations. We do not attribute our Qwen coordinates or fine-tuning token-location result to that paper.
