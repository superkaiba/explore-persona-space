# Methodology — issue 722: fit M0 vs M⁺ (ridge + MLP) on #537 adapters' paired activations


**Design:** A pure measurement on already-trained adapters — no model training. Three behaviors (harmful-compliance / EM, taught fact, sycophancy) × three layers (7, 14 primary, 21). For each (behavior, layer) cell, two maps are fit on paired activations from the matched store: the base map **M0** from `(base context vector c0 → base answer profile v0)`, and the post-finetuning map **M⁺** from `(post-FT context vector cplus → post-FT answer profile vplus)`. Both are then evaluated on the *same* base context grid so the comparison is at a fixed input. Each map is fit two ways — a closed-form ridge and a 1-hidden-layer MLP — sharing a 64-dimension output target. The single manipulated thing is which side of the finetuning boundary the map comes from (M0 vs M⁺).

**Training:** N/A — no model training. The reused LoRA adapters were trained against the project's behavior×context grid (per-behavior SFT that installs each behavior into a source persona under contrastive negatives; r=32, rsLoRA, per-behavior α — EM 256; fact/syco/marker/refusal 64). The paired base+post-FT activations were extracted by applying those frozen adapters to Qwen-2.5-7B-Instruct and reading hidden states on a fixed eval-probe grid, baking the layer index into each stored tensor. The fit constants:

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct (frozen) | repo standard |
| Reused adapters | `i537_*`, r=32, rsLoRA, per-behavior α (EM 256; fact/syco/marker/refusal 64) | adapter `adapter_config.json` + store `adapter_gauge` field |
| Ridge λ | closed-form PRESS LOCO over [1e-2 … 1e3] | `RIDGE_LAMBDAS` constant |
| MLP | 1 hidden × 512, AdamW lr 1e-3, weight decay 1e-4, 300 epochs, LOCO ensemble | `MLP_*` constants |
| Output target dim | 64 top-v0 PCA dims (shared ridge/MLP) | `A35_MLP_TARGET_DIM` |
| Layers | 7, 14 (primary), 21 | `PRIMARY_LAYER`, `SUPPLEMENT_LAYERS` |
| Bootstrap families | 7 context families (sp_/wc_/icl_/reph_/fmt_/binst_/default) | `gate_chain.family_of` |
| Behavior direction r_B | diff-in-means (`diffmeans`); taught-fact direction re-extracted here | `DEFAULT_RB` |
| Distinct context inputs | 16 source contexts per behavior (the context vector is source-keyed; 480 source×target leakage cells share the same 16 source c_C) | paired store, seed 42 only |
| Seed | 42 (store-fixed; nulls are bootstrap/refit, not cross-seed) | paired store |

**Evaluation:** Three continuous dependent variables, all on the model's own paired activations, plus one validity gate. (1) **Function-change Δ** — the median over the base context grid of `|(M⁺(c) − M0(c)) · r̂_B|`, the per-context change projected onto the unit behavior direction. The combined noise floor is the `max` of three nulls built through the identical refit harness: two independent refits of M0, two of M⁺, and a same-function shifted-design null (the M0 function fed post-FT inputs, read at base inputs, capturing pure off-support extrapolation). The kill criterion compares Δ_med to `floor_combined`. (2) **Chain ρ** — Spearman correlation between `r_Bᵀ M̂(c)` (the held-out LOCO prediction projected onto the behavior direction) and the measured judge leakage rate E (the `g` scalar from the trained leakage matrix), under M0 and M⁺, with a family-clustered 95% CI (resampling the 7 context families). (3) **Cross-transfer** — held-out cosine of M0 and M⁺ predicting the base vs post-FT answer profiles. (4) **MLP-vs-shuffle validity gate** — the MLP map is trusted only where its base-map held-out ρ beats a label-shuffled control (`rho_M0_mlp > rho_M0_shuffle`). No judge calls are made here; E is read from the already-judged leakage matrix. The behavior direction r_B is the validated subspace direction, not raw v0 reconstruction (which saturates near 0.98 and carries no signal).

A methodology correction landed mid-run: the first launch crashed in `_pca_basis_v0` with a numpy `SVD did not converge` on a pathological bootstrap resample of the sycophancy cells (the EM sweep had already finished cleanly, so the bug was in the floor-refit path, not the fit). The fix added a `gesvd` fallback for SVD non-convergence; the relaunch reproduced the same SVD failure on the same resamples and the fallback absorbed it, so every reported cell is from the fixed path. The combined run took two launches (~8 GPU-h total) on GCP 2× H100.

**Data extraction:** N/A as training data — this is a measurement on existing activations. The activations are derived from the frozen tier-2 eval-probe pools (Betley main-8, AdvBench, wrong-claims, SORRY-Bench/XSTest, fact-recall) extracted into the matched paired store (5760 `.npz`, 1152 per behavior, layer-baked, seed 42). The context vector `c_C` is keyed to the SOURCE persona, so although there are 480 source×target leakage cells per behavior×layer, the map `M: c → v` sees only 16 distinct source inputs — the effective sample size for the fit. The taught-fact behavior direction had no pre-existing diff-in-means contrast, so it was re-extracted here as fact-stated (fact-recall probes) minus fact-absent (neutral Betley) answer-span activations, using the identical project recipe, and saved to a parallel HF namespace (it does not overwrite the existing `r_b.pt`). The contrastive-negatives and on-policy-completion rules do not apply: no training rows are constructed (the implants already exist in the reused adapters).

**Sample training/evaluation data + completions:** This run generates no model completions (it fits ridge/MLP regressions on stored activation vectors), so there are no firing/non-firing completion samples. The worked data unit is a paired activation tuple. The block below is **1 example fit unit** (em/L14, illustrative, of 480 source×target leakage cells in that behavior×layer); the complete paired store (5760 `.npz`) is at [HF `issue667_gate_chain_preview/analysis_tensors` @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue667_gate_chain_preview/analysis_tensors), the leakage target E at [`eval_results/issue_537/G_tensor/G_meta.json` @3c051fcfad](https://github.com/superkaiba/explore-persona-space/blob/3c051fcfad/eval_results/issue_537/G_tensor/G_meta.json), and the re-extracted taught-fact direction at [HF `issue722_rb_extension/store/r_b_fact.pt` @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue722_rb_extension/store).

<details>
<summary>1 example fit unit (illustrative; read verbatim from the paired store schema, em/L14; of 480 cells in this behavior×layer). Full store + E target + r_B linked in the prose above.</summary>

```
file: issue667_gate_chain_preview/analysis_tensors/binst_em_L14.npz   (of 1152 em files)
  behavior   = "em"        source_cid = <persona/context id>   target_cid = <eval context>
  layer      = 14          seed       = 42
  c_C        : (3584,)   base context vector       -> M0 input  (c0)
  c_C_postft : (3584,)   post-FT context vector    -> M⁺ input  (cplus)
  v0         : (3584,)   base answer profile       -> M0 target
  v_plus     : (3584,)   post-FT answer profile    -> M⁺ target
chain-ρ target E: eval_results/issue_537/G_tensor/G_meta.json
  per_cell["em/<source>__<target>"].g   (judge leakage rate, one scalar per cell)
```

</details>


---

*Derived from the [task body](https://eps.superkaiba.com/tasks/722).*
