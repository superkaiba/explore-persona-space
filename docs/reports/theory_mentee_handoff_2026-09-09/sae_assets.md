---
title: "SAEs, mappings, and autointerpretation"
subtitle: "Supplement to the theoretical-analysis handoff"
date: "10 September 2026"
geometry: margin=0.85in
fontsize: 10pt
colorlinks: true
header-includes:
  - '\usepackage{xurl}'
  - '\usepackage{needspace}'
---

# What is included

This supplement adds the actual checkpoint locations, configuration files, saved mappings, feature descriptions, and interpretation examples. `sae_artifact_manifest.csv` records pinned download URLs, sizes, hashes, and exact package locations. Small files are included under `sae_assets/`; checkpoints and other files larger than 10 MB are linked, not embedded in the ZIP. No SAE training, map fitting, inference, or autointerpretation was run.

The theory report remains a dated results snapshot. This supplement adds assets; it does not refresh the China experiment's status or change the paper.

# 1. Choose a compatible pair of SAEs

All four dictionaries below concern Qwen2.5-7B-Instruct layer 19, with residual-state dimension 3,584. A context vector is the state at the final prompt token. A turn-averaged answer vector averages assistant-token states; it is **not** a per-token SAE or an average of separately encoded token features.

| Family | Context SAE | Turn-averaged answer SAE | Use |
|:---|:---|:---|:---|
| Matched #2552 pair | 32,768 features, BatchTopK, k=128 | 32,768 features, BatchTopK, k=128 | Global/minimal-refusal dashboards and #2643 direct SAE maps |
| Earlier #2569/#2476 pair | 65,536 features, Matryoshka BatchTopK, k=100 | 65,536 features, Matryoshka BatchTopK, k=100 | Original theory battery and its feature-firing/magnitude regression |

Checkpoint directories, each with its configuration:

- **32k context:** [sae_ctx_rep](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/cd80ba2588bb6d4291edf621176ea654bcbf2507/issue2552_derreplication/exactrep/analysis_tensors/sae_ctx_rep), file `sae_weights.safetensors` with `cfg.json`.
- **32k answer:** [sae_rep](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/cd80ba2588bb6d4291edf621176ea654bcbf2507/issue2552_derreplication/exactrep/analysis_tensors/sae_rep), file `sae_weights.safetensors` with `cfg.json`.
- **65k context:** [sae_ctx](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a5cf03bbe4807361a74c427a46ad329065f75e30/issue2569_theory/analysis_tensors/sae_ctx), file `ae.pt` with `config.json`.
- **65k answer:** [sae_c](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a5cf03bbe4807361a74c427a46ad329065f75e30/issue2476_turnavg/analysis_tensors/sae_c), file `sae_weights.safetensors` with `cfg.json`. Despite the folder name, this is the **answer** dictionary in the original theory analysis.

The 32k checkpoints are about 0.94 GB each; the 65k checkpoints are about 1.88 GB each. Preserve configurations, thresholds, normalization, and feature ordering. Do not pair a 32k dictionary with a 65k map or transfer descriptions by integer feature ID across dictionaries.

# 2. The mappings are different objects

## Dense context-to-answer map

The [frozen layer-19 ridge checkpoint](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/cd80ba2588bb6d4291edf621176ea654bcbf2507/issue779_monitoring/n1m_readout/weights/L19/ridge.pt) maps residual states, not SAE codes. Its stored row-vector convention is

$$\widehat a=((c-x_\mu)/x_{sd})W+y_\mu.$$

The linear operator used by the geometry dashboards is $A=\operatorname{diag}(1/x_{sd})W$. For absolute-state predictions, keep the centering and output mean; for paired differences, the affine offset cancels. A read/write feature ranking based on this dense operator is not itself a learned SAE-to-SAE coefficient matrix.

## Composed 32k SAE route

The original #2643 route is

$$\widehat z_A=s\odot E_A\bigl(F(D_C(z_C))\bigr),\qquad z_C=E_C(c),$$

where $E$ and $D$ are the corresponding SAE encoder/decoder, $F$ is the dense affine map, and $s$ is the saved feature-wise calibration. The composition is not globally linear because it includes the answer SAE encoder.

Use the 32k pair, the dense ridge above, and [feature_calibration.pt](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/1c5eb09f2bcabd9541a86890f0e0e945afff903a/issue2643_sae_map/screen/feature_calibration.pt). The calibration is bundled. The manifest also includes the screen summary and separate refusal readouts; those readouts are not interchangeable with the SAE map.

## Direct 32k context-code to answer-code maps

These are separately trained reduced-rank regressions:

$$\widehat z_A=((z_C-\mu_C)/s_C)B_1B_2+\mu_A.$$

Here $B_1$ and $B_2$ are payload fields `a` and `b`, and the means are `x_mean` and `y_mean`. The producer replaces every saved `x_std` entry below $10^{-8}$ with 1 before dividing; use that guarded standard deviation as $s_C$. This matters for inactive features. Both axes refer to their own 32k dictionaries, not shared feature indices. Predictions from this linear regression need not be nonnegative.

- [Original rank-256 fits](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1c5eb09f2bcabd9541a86890f0e0e945afff903a/issue2643_sae_map/direct_edges): `direct_rank256_seed2643.pt` and `direct_rank256_seed2644.pt`, plus training summaries and top-edge arrays.
- [Tuned rank-1024 fits](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1c5eb09f2bcabd9541a86890f0e0e945afff903a/issue2643_sae_map/direct_edges_tuned): `selected_seed2643.pt` and `selected_seed2644.pt` are validation-selected fits; `trainval_refit_seed2643.pt` and `trainval_refit_seed2644.pt` are the separately reported train-plus-validation refits. Selection metadata, final summary, edge arrays, and dashboard graph are included.

Keep both seeds and distinguish selected fits from refits. The graph's top edges are a selected view, not the entire matrix, and regression edges do not establish causal connections.

## Older 65k feature regression: an explicit archive gap

The [original leg-4 archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a5cf03bbe4807361a74c427a46ad329065f75e30/issue2569_theory/analysis_tensors/leg4) contains encoded inputs, target feature IDs, held-out predictions, and metrics. The checked producer returns predictions from its fit and does **not** serialize the fitted coefficient matrix. Thus this package supplies the evidence and input artifacts, but not an inference-ready checkpoint for that specific regression. The #2643 checkpoints are available alternatives for the **32k** pair, not replacements for this missing 65k fit. Nothing was refitted to fill the gap.

# 3. Autointerpretation and maximum-activation examples

There are three distinct description sources to consult:

1. **Original answer-SAE W1 descriptions:** [raw W1 files](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1c5eb09f2bcabd9541a86890f0e0e945afff903a/issue2552_derreplication/exactrep/raw_completions/judge/w1). Both `judge_raw_w1.json` and `judge_raw_w1_syncreissue.json`, and their saved draw records, are bundled. For the 32k answer dictionary, use entries whose IDs match `w1-rep_ta-f<ID>`; do not treat every record in a mixed instrument as that dictionary's feature. The existing loader reads the main file followed by the reissue, with later valid descriptions taking precedence. These are historical outputs; no Claude or other model was invoked for this handoff.
2. **Context/answer edge-candidate autointerpretation:** [autointerp.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/1c5eb09f2bcabd9541a86890f0e0e945afff903a/issue2643_sae_map/direct_edges/autointerp.json) and [examples](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/1c5eb09f2bcabd9541a86890f0e0e945afff903a/issue2643_sae_map/direct_edges/autointerp_examples.jsonl). These cover 64 context and 64 answer candidates, not all features. Answer descriptions reuse available W1 outputs; new descriptions were produced with Qwen2.5-7B-Instruct. Context examples came from the refusal panel rather than a broad context corpus. The saved protocol records those deviations.
3. **Later Codex interpretation and broad review:** the original handoff's `artifacts/codex_interpretations/` includes `codex_interpretation_results_final.json`, `codex_interpretation_report_final.md`, packets, and broad-review evidence. These are the later descriptions for the requested global and minimal-refusal mean/RMS extreme lists. Use them for that analysis, while retaining the original W1/edge labels for provenance. Coverage is selective, not a full-dictionary annotation.

The older 65k answer instrument's [consumed descriptions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/a5cf03bbe4807361a74c427a46ad329065f75e30/issue2569_theory/analysis_tensors/der/consumed_input/descriptions_mat_k100.json) are also bundled. They must not be attached to the 32k answer dictionary. No comparable complete 65k context-description inventory is claimed; the theory package retains its selected direction/SAE matches and examples.

# 4. Practical reuse

Start with the 32k pair and choose either the composed route or a direct fitted map. Read its configuration and selection metadata before loading weights. Join descriptions by **checkpoint identity, side, and feature ID**, not by feature ID alone. Missing descriptions mean unannotated features, not absent semantics. Distinguish saved maximum-activation examples from representative behavior across the full data distribution.

`sae_producers/` contains current #2643 producer and loader source snapshots, with hashes in `sae_producer_manifest.json`. In particular, inspect `issue2643_sae_map.py`, `issue2643_direct_edges.py`, and `issue2643_edge_autointerp.py` for the saved formats and description-merging rules. Use them alongside the project checkout and its dependencies; do not launch training or autointerpretation phases merely to open saved assets. These working-tree snapshots are not asserted to be byte-identical to every historical training script.

The supplement's CSV is the authoritative download list. Its small bundled files were checked against immutable Hub Git-blob or LFS hashes; large files were availability-checked but not downloaded or executed. The original artifact inventories remain unchanged, and `sae_asset_summary.json` records the additional coverage. This is an asset handoff, not a new evaluation or a claim of complete semantic interpretation.
