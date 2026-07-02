---
title: Finetuning measurably reshapes the context→answer map only for a taught fact
  (LOW confidence)
kind: experiment
tags:
- followup-manual
created_at: '2026-06-28T20:32:55Z'
has_clean_result: true
parent_id: 667
origin_prompt: 'run the pre vs post finetuning followup as a standalone issue linked
  to 667. run what we''ve discussed as well as the extra things here: - function-change
  at fixed input (‖M⁺(c) − M0(c)‖ on a common c grid), and - behavior-relevant transfer
  — the chain r_Bᵀ M c → E, or M evaluated along behavior directions — not generic
  v0 reconstruction.'
goal: 'On #537''s already-trained behavior×context LoRA adapters, fit the context→answer-profile
  mapping M: c_C → v0(C) both pre-finetuning (M0, base) and post-finetuning (M⁺, adapter-applied)
  as a linear ridge AND a nonlinear MLP, and determine whether the FUNCTION M changes
  (distinct from the input context vector c_C shifting) via (1) function-change at
  fixed input ‖M⁺(c)−M0(c)‖ on a common c grid projected onto behavior read-outs r_B,
  (2) behavior-relevant transfer through the chain r_Bᵀ M c → E (judge rate) and M
  along r_B — NOT generic v0 reconstruction (which #658 showed saturates ≈0.98 and
  is uninformative), and (3) cross-transfer of M0 vs M⁺ on FT data; cross-referenced
  against #697''s causal f_CV.'
relates_to:
- identity-cb-duality
- identity-contextual-vs-base
---
# Finetuning measurably reshapes the context→answer map only for a taught fact (LOW confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_722.md](https://github.com/superkaiba/explore-persona-space/blob/e64e819df4f993380a715f8a339768f4c4bbc128/docs/methodology/issue_722.md) · [gist](https://gist.github.com/superkaiba/e246003eceb7de4e95ff530bd3f54cc8)

## Takeaways

- **The base linear context→answer map is STRONG**: skill-over-mean held-out R² plateaus at **0.74–0.80** (peak layer 18); the earlier cosine-0.31 read was a saturation artifact. PCA-48/whiten-48 inputs reproduce it on **28/28** layers (max |ΔR²| **0.018**).
- **Nonlinear estimators buy nothing**: the MLP is negative at every layer, and the RBF−linear gap (≈0 at the plateau) collapses to **−0.81** at 48 input dims — **KILL** on the gap-gate (**0/28**) alone, the ridge headline survives; function-change reads are ridge-only.
- **1.6–3.3× floor** — the taught-fact fitted map measurably reshapes after finetuning (3.3× at the primary layer; 2.9× / 1.6× elsewhere); the function moved, not just its input.
- **0.3–0.6× floor** for **EM** at every layer; sycophancy **3.2×** (layer 7) to **0.1×** (layer 21) — both fail the MLP-vs-shuffle power check, so below-floor readings are **inconclusive**, not "function held".
- **ρ ≈0 → +0.50** post-FT at the primary layer (CI excludes zero) for the fact only — a context→leakage transfer chain the base map lacked; EM/sycophancy do not move.
- **n=16** source-keyed inputs cap the M0-vs-M⁺ comparison (the base-map sweep used 50 contexts); the floor is post-FT refit variance in all 9 cells, so the fact signal clears the "noisier to refit" alternative.

## Goal

**This experiment in context:** This is the descriptive/parametric half of a two-part question about *where* finetuning lives. The leakage theory models behavior installation as either moving the input context vector `c_C` that feeds a context→answer-profile map M, or changing M itself. The base-only fit of M was first built in [#658](https://eps.superkaiba.com/tasks/658); the matched base-and-post-finetuning activation store that makes a pre/post comparison possible was extracted in [#667](https://eps.superkaiba.com/tasks/667). This run fits M0 (base) and M⁺ (adapter-applied) explicitly on [#537](https://eps.superkaiba.com/tasks/537)'s already-trained behavior×context LoRA adapters and asks whether the *function* M moved, distinct from the *input* moving — and, alongside, re-establishes how strong the base linear map itself is. The causal twin is [#697](https://eps.superkaiba.com/tasks/697), which injects the finetuned context vector into the base model and measures the mediated fraction; here the comparison is parametric (a fitted-map difference), and the two pictures are meant to be read side by side (#697 was still running at analysis time, so no cross-reference table is included).

**Broader narrative:** The two spiritual-sibling papers — *Persona Vectors* (Anthropic) and *Persona Features Control Emergent Misalignment* (OpenAI) — treat a persona/behavior as a direction in activation space that finetuning shifts. That framing implicitly assumes the readout *map* is fixed and only the input moves along it. This run is a first parametric probe of that assumption on open weights: for the taught fact the map measurably moves, but the EM and sycophancy verdicts are inconclusive (the diagnostic lacks power at n=16), so the assumption is neither confirmed nor cleanly refuted for a broad behavioral trait. This bears on the project's open question of whether behavior-distance reduces to context-distance.

## Methodology

**Design:** A pure measurement on already-trained adapters — no model training. Three behaviors (harmful-compliance / EM, taught fact, sycophancy) × three layers (7, 14 primary, 21). For each (behavior, layer) cell, two maps are fit on paired activations from the matched store: the base map **M0** from `(base context vector c0 → base answer profile v0)`, and the post-finetuning map **M⁺** from `(post-FT context vector cplus → post-FT answer profile vplus)`. Both are then evaluated on the *same* base context grid so the comparison is at a fixed input. Each map is fit two ways — a closed-form ridge and a 1-hidden-layer MLP — sharing a 64-dimension output target. The single manipulated thing is which side of the finetuning boundary the map comes from (M0 vs M⁺).

**Training:** N/A — no model training. The reused LoRA adapters were trained against the project's behavior×context grid (per-behavior SFT that installs each behavior into a source persona under contrastive negatives; r=32, rsLoRA, per-behavior α — EM 256; fact/syco/marker/refusal 64). The paired base+post-FT activations were extracted by applying those frozen adapters to Qwen-2.5-7B-Instruct and reading hidden states on a fixed eval-probe grid, baking the layer index into each stored tensor. The fit constants:

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct (frozen) | repo standard |
| Reused adapters | `i537_*`, r=32, rsLoRA, per-behavior α (EM 256; fact/syco/marker/refusal 64) | adapter `adapter_config.json` + store `adapter_gauge` field |
| Ridge λ | closed-form PRESS LOCO over [1e-2 … 1e3] | `RIDGE_LAMBDAS` constant |
| MLP | 1 hidden × 512, AdamW lr 1e-3, weight decay 1e-4, 300 epochs, LOCO ensemble | `MLP_*` constants |
| Output target dim (3×3 M0-vs-M⁺ arm) | 64 top-v0 PCA dims (shared ridge/MLP) | `A35_MLP_TARGET_DIM` |
| Output target dim (`base-skill-over-mean-cC-to-v0` follow-up arm, LOCO MLP) | 48 top-v0 PCA dims | `skill_over_mean.json` `mlp_recipe.pca_target_dim` |
| Layers | 7, 14 (primary), 21 | `PRIMARY_LAYER`, `SUPPLEMENT_LAYERS` |
| Bootstrap families | 7 context families (sp_/wc_/icl_/reph_/fmt_/binst_/default) | `gate_chain.family_of` |
| Behavior direction r_B | diff-in-means (`diffmeans`); taught-fact direction re-extracted here | `DEFAULT_RB` |
| Distinct context inputs | 16 source contexts per behavior (the context vector is source-keyed; 480 source×target leakage cells share the same 16 source c_C) | paired store, seed 42 only |
| Seed | 42 (store-fixed; nulls are bootstrap/refit, not cross-seed) | paired store |
| Input rep of c_C (`input-pca-robustness-cC-to-v0` round) | `input_rep ∈ {full (baseline), pca48, whiten48}`; per-fold TRAIN-only basis, k=48, ε=1e-6; seed 42 and all other knobs unchanged | plan §4.3 + `run_meta.json` |
| KRR kernels + γ (`input-pca-robustness-cC-to-v0` round) | linear + RBF; γ = median-pairwise heuristic recomputed per input rep (sensitivity ×0.25–×4 at layers 18/21); ridge/KRR λ per layer by PRESS/LOO over [1e-2…1e3] (`lambda_chosen`) | committed `krr_vs_linear__*.json` + `gamma_sensitivity__pca48.json` |

**Evaluation:** Four continuous dependent variables, all on the model's own paired activations, plus one validity gate. (0) **Base-map strength** — for the BASE map c_C → v0 over a broader 50-context grid (reusing the earlier base-only context-store loader, which assembles the last-token base context vectors c_C and base answer profiles v0 across the 50 contexts), the held-out **skill-over-mean R²** = `1 − SS_res/SS_tot` on the centered v0 target (a positive value means the fitted map beats predicting the training mean), fit per layer 0–27 as a closed-form LOCO ridge and as a 1-hidden-layer MLP, with a label-shuffle null. This DV replaces the row-wise cosine used in the earlier base-only fit, which is uninformative here: the predict-the-mean baseline itself cosines ≈0.99 to held-out targets (activation anisotropy), so cosine has no resolving power, and the old `_ridge_predict_loco` additionally shrank toward 0 rather than the training mean (a no-intercept bug); both inflate the apparent residual and hid the map's real strength. (1) **Function-change Δ** — the median over the base context grid of `|(M⁺(c) − M0(c)) · r̂_B|`, the per-context change projected onto the unit behavior direction. The combined noise floor is the `max` of three nulls built through the identical refit harness: two independent refits of M0, two of M⁺, and a same-function shifted-design null (the M0 function fed post-FT inputs, read at base inputs, capturing pure off-support extrapolation). The kill criterion compares Δ_med to `floor_combined`. (2) **Chain ρ** — Spearman correlation between `r_Bᵀ M̂(c)` (the held-out LOCO prediction projected onto the behavior direction) and the measured judge leakage rate E (the `g` scalar from the trained leakage matrix), under M0 and M⁺, with a family-clustered 95% CI (resampling the 7 context families). (3) **Cross-transfer** — held-out cosine of M0 and M⁺ predicting the base vs post-FT answer profiles. (4) **MLP-vs-shuffle validity gate** — the MLP map is trusted only where its base-map held-out ρ beats a label-shuffled control (`rho_M0_mlp > rho_M0_shuffle`). No judge calls are made here; E is read from the already-judged leakage matrix. The behavior direction r_B is the validated subspace direction; generic per-dimension v0 reconstruction is read with skill-over-mean R² (not cosine) per (0).

A methodology correction landed mid-run: the first launch crashed in `_pca_basis_v0` with a numpy `SVD did not converge` on a pathological bootstrap resample of the sycophancy cells (the EM sweep had already finished cleanly, so the bug was in the floor-refit path, not the fit). The fix added a `gesvd` fallback for SVD non-convergence; the relaunch reproduced the same SVD failure on the same resamples and the fallback absorbed it, so every reported cell is from the fixed path. Two further corrections shaped the base-map read: the row-wise cosine metric was replaced by train-mean-centered skill-over-mean R² (cosine is pinned near 1.0 by the mean baseline), and the no-intercept bug in `_ridge_predict_loco` was fixed to shrink toward the training mean; the vectorized GPU rewrite that delivered the corrected read was validated to leave the earlier base-only chain reproduce-check numbers byte-exact (Δ = 0.00e+00 on all 8 cells across both genres). The combined run took two launches (~8 GPU-h total) on GCP 2× H100; the base-map sweep added 1.5 min on an H100.

An input-representation robustness round (`input-pca-robustness-cC-to-v0`, 0 GPU-h, CPU-only) re-ran the base-map ridge skill AND the kernel-ridge (KRR) RBF−linear gap — KRR fits the same c_C → v0 map by closed-form kernelized least-squares; the gap is RBF held-out R² minus linear held-out R² on the PCA-48 v0 target, with a 2000-resample LOCO-fold bootstrap CI per layer (`gap_ci95`) — under three input representations of c_C: full 3584-dim (the committed baseline, reused not re-run), PCA-48, and whiten-48. Per LOCO fold, the PCA basis is fit on the 49 TRAIN rows only (top 48 PCs) and the held-out row is projected with that train basis; whiten-48 additionally rescales each PC coordinate by 1/√(σ²+ε), ε = 1e-6 (realized as PCA-whitening, recorded in `run_meta.json`; the rotation choice is immaterial because the ridge re-standardizes per fold and the RBF kernel keys on per-direction scale). Ridge/KRR λ is selected per layer by the closed-form PRESS/LOO criterion over the [1e-2…1e3] grid and recorded per layer (`lambda_chosen`; the plateau layers select the band edge 0.01). The RBF γ is the median-pairwise-distance heuristic recomputed on the transformed input; a γ-sensitivity sweep at 0.25×/0.5×/1×/2×/4× the heuristic (layers 18 and 21) probes whether the KRR read is a γ-regime artifact. Success criterion fixed at plan time (two gates, both variants): ≥26/28 layers with |ΔR²_ridge| ≤ 0.05 vs the full-dim baseline AND gap sign preserved with |Δgap| ≤ 0.03. The `--input-rep full` path reproduces the committed baseline (16-test suite green), and the PCA-48 vs whiten-48 outputs are numerically identical because the fit's per-fold train-only standardization absorbs the per-direction rescale — one effective 48-dim condition, with the per-fold/train-only basis independently verified in code review.

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

## Results

### The base context→answer map is a strong linear map (skill-over-mean R² 0.74–0.80); the nonlinear MLP overfits

What is plotted: per layer (0–27), held-out skill-over-mean R² of the BASE map c_C → v0 — ridge, MLP, MLP shuffle null (left axis); two cosines (right axis). 50 contexts, seed 42.

![Per-layer skill-over-mean held-out R-squared for the base context-to-answer map: the ridge curve rises to a 0.74-0.80 plateau over the mid-to-late layers while the MLP stays negative and both cosines pin near 1.0](https://raw.githubusercontent.com/superkaiba/explore-persona-space/78572438867299935952a33402e16c0f2493fdaf/figures/issue_722/base_skill_over_mean_per_layer.png)

> **Figure.** *The base linear map captures 74–80% of across-context answer-profile variance at the mid-to-late layers; cosine cannot see it.* Ridge skill-over-mean R² peaks at +0.80 (layer 18); the MLP is negative everywhere; both cosines pin near 1.0, so cosine has no resolving power. 50 contexts, seed 42.

Ridge R² climbs from −0.09 (layer 0) to a 0.74–0.80 plateau over layers 14–21 (peak +0.80 at layer 18), so the base map does most of the context→answer work linearly. This corrects the earlier base-only read, where a row-wise cosine scored the base map at ≈0.31 and made it look weak: that cosine was a saturation artifact (the predict-the-mean baseline alone cosines ≈0.99), and the train-mean-centered held-out R² is the correct estimator, under which the base linear map is strong. The MLP, by contrast, is negative at all 28 layers (−1.05 best, −5.59 worst) and below its shuffle null throughout — it overfits at n=50, so the ridge is the only valid estimator.

### The linear-ridge skill is input-representation-robust (28/28 layers); the RBF−linear gap is not — RBF collapses at 48 input dims (two-gate verdict: KILL)

What is plotted: held-out ridge skill-over-mean R² (top) and the KRR RBF−linear gap (bottom) under three input representations of c_C — full 3584-dim baseline, PCA-48, whiten-48 (per-fold train-only bases). All 28 layers (0–27) are rendered; the per-layer rows are in the linked JSONs. n = 50 contexts, seed 42.

![Ridge R-squared coincides across full, PCA-48, and whiten-48 inputs; the RBF-minus-linear gap drops to minus 0.81 at 48 dims at layer 18](https://raw.githubusercontent.com/superkaiba/explore-persona-space/670602a6af1945f6c272b3a102bc7f39ef130cb3/figures/issue_722/input_rep_robustness_per_layer.png)

> **Figure.** *The ridge skill is invariant to the input representation; the RBF−linear gap is not.* The three ridge curves coincide (R²-gate **28/28** both variants, max |ΔR²| 0.018); the gap falls from ≈0 at the plateau (full input) to **−0.81** under both 48-dim inputs (gap-gate **0/28**).

PCA-48 reproduces the ridge plateau (layer 18 **+0.8064** vs +0.7983 full; max |Δ| **0.018** across 28 layers); whiten-48 matches PCA-48 to numerical precision. The plan-time two-gate criterion still returns **KILL** for both variants, entirely from the gap-gate (**0/28**): RBF kernel regression collapses to skill ≈0 at 48 dims — the gap falls to **−0.81** at layer 18 (95% CI −0.85 to −0.76), γ-flat across 0.25×–4× — not a γ mis-tune. RBF beats linear only at the weak early layers at full dim (gap +0.13–0.40, layers 0–5), never at the plateau; the gap's value, not the linear verdict, fails to transfer.

### The taught fact's function-change clears its floor 1.6–3.3×; EM sits below floor; sycophancy peaks shallow and dies

What is plotted: per-cell function-change Δ_med (median of `|M⁺(c) − M0(c)|` along the behavior direction) against its noise floor — raw first, then the ratio Δ_med ÷ floor_combined (1× = at floor). 3 behaviors × 3 layers, ridge fit, seed 42.

![Raw function-change delta vs floor, nine cells: fact bars exceed their floors, EM and late-layer sycophancy bars sit below](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/function_change_delta_vs_floor.png)

> **Figure (raw).** *The taught-fact Δ towers over its floor; EM and late-layer sycophancy Δ sit below theirs.* Blue = Δ_med, orange = combined floor (= the M⁺ refit-variance null in every cell). Ridge fit, family-clustered bootstrap over 7 context families.

![Normalized function-change-over-floor heatmap, 3 behaviors x 3 layers; only the taught-fact row exceeds 1x at every layer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/hero_function_change_heatmap.png)

> **Figure (normalized).** *Only the taught fact clears the noise floor at the primary layer.* Cells are Δ_med ÷ floor_combined; >1× means the map changed more than its own refit noise. n=16 source contexts, seed 42, ridge fit.

Only the taught fact clears the floor at every layer (1.58–3.29×); EM sits below floor throughout, and sycophancy clears it only at layer 7 before collapsing by layer 21. The combined floor is the post-FT map's own refit variance in all 9 cells, so the fact's 3.29× clears the planned "noisier to refit" confound. The floor swings widely (0.004 to 0.27), so a large absolute Δ can still sit below a large floor (sycophancy L21: Δ=0.034 vs floor 0.27). Robustness checks leave every call unchanged. EM and sycophancy also fail the MLP-vs-shuffle check, so they are inconclusive, not "function held".

### A behavior-relevant transfer chain appears for the taught fact after finetuning

What is plotted: per cell, the Spearman ρ between the held-out prediction along the behavior direction (`r_Bᵀ M̂(c)`) and the judge leakage rate E, under the base map M0 (orange) and post-FT map M⁺ (blue), with 95% family-clustered CIs. A rightward shift means finetuning created a transfer the base map lacked.

![Forest plot of chain rho under base map M0 and post-FT map M+ for nine cells; taught fact L14 and L21 post-FT points sit at +0.50 and +0.46 with CIs excluding zero while base points sit near zero; EM and sycophancy points cluster around zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/chain_rho_shift_forest.png)

> **Figure.** *For the taught fact only, the context→leakage chain jumps from ≈0 in the base map to +0.46–0.50 post-finetuning.* Held-out LOCO ridge ρ vs the judge leakage rate E for the trained-on (target, behavior, persona-context) tuple; 95% family-clustered CI over 7 families. EM and sycophancy chain shifts all straddle zero.

For the taught fact, chain ρ rises from −0.12 (base map, CI straddles zero) to **+0.50** at the primary layer (post-FT, 95% CI +0.31 to +0.64), with +0.46 at layer 21 and +0.17 at layer 7. Ridge and MLP agree on this scalar (+0.502 vs +0.490 at the primary layer) despite the MLP's overall reconstruction failure. Every EM and sycophancy chain-shift CI straddles zero, so the fact change is a real new context→leakage relationship rather than a generic wobble.

### Cross-transfer is asymmetric: the post-FT map generalizes worse than the base map

What is plotted: per cell, the held-out cosine of the base map M0 predicting the post-FT profile v⁺, the post-FT map M⁺ predicting v⁺, and M⁺ predicting the *base* profile v0. The third bar is the asymmetry — how well the finetuned map back-predicts the base answer profile.

![Grouped bars of cross-transfer cosines for nine cells; base-map and post-FT-map predictions of the post-FT profile both sit near zero, while the post-FT map predicting the base profile sits far below at -0.22 to -0.32](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/cross_transfer_orthogonality.png)

> **Figure.** *Neither map predicts the post-FT profile (cosine ≈0), but the post-FT map predicts the base profile worst (−0.22 to −0.32).* Orange/blue = base/post-FT map → post-FT profile; green = post-FT map → base profile. Held-out cross-transfer cosine, 9 cells.

Both maps predict v⁺ about equally poorly (cosines near zero — e.g. at fact L7 they agree within ≈0.0023), so finetuning did not rotate M toward v⁺. The asymmetry is that M⁺ predicts the *base* profile v0 markedly worse (cosine −0.22 to −0.32) than M0 predicts v⁺: the finetuned map has lost base-profile alignment without gaining post-FT-profile predictive power. So "M changed" does not mean "M now points at the trained behavior" — the change degrades back-prediction rather than realigning toward v⁺.

### Why the EM/sycophancy verdict is inconclusive, not a clean "function held"

What is plotted: per cell, the MLP's held-out base-map reconstruction ρ against its shuffle null; the MLP is trusted only where `rho_M0_mlp > rho_M0_shuffle`.

![Bars of MLP held-out rho vs shuffle null for nine cells; MLP bars fail to beat the shuffle null in eight of nine cells, passing only at taught fact layer 7](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/mlp_shuffle_diagnostic.png)

> **Figure.** *The nonlinear MLP map fails its shuffle null in 8 of 9 cells, passing only at taught fact, layer 7.* Blue = MLP held-out base-map ρ, gray = shuffle null. With the nonlinear fit unreliable, the ridge fit is the only valid map.

The MLP fails the shuffle null in 8 of 9 cells (beating it only at fact L7, ρ=−0.077 vs −0.080), so the nonlinearity-gap read is dropped and the headline is ridge-only. EM and sycophancy are inconclusive rather than "function held" because certifying a held function needs a below-floor Δ AND a passing MLP fit, and at n=16 single-seed an at-floor Δ is indistinguishable from an underpowered one. The per-cell Δ_med bootstrap CI is degenerate (point = lo = hi), so the cross-behavior pattern carries the inference, and the source-keyed input `c_C` lets M see only 16 distinct inputs rather than 480 — the dominant cap. <!-- concern-deferred: substrate-context-vec-keyed-to-source --> The positive fact result survives both caps: floor 1.6–3.3× and a chain-shift CI excluding zero.

---

**Repro:** Compute — GCP `eval-h100` lane (`a3-highgpu-2g`, 2× H100-80, FLEX_START), ~8 GPU-h across two launches for the M0-vs-M⁺ sweep (a first launch crashed in the SVD floor-refit path; relaunch wall 5h13m), plus a 1.5-min H100 run for the corrected base-map skill-over-mean sweep + #658 chain reproduce-check. No WandB run (analysis/fitting job). Code SHA [`3c051fcfad`](https://github.com/superkaiba/explore-persona-space/blob/3c051fcfad/scripts/issue722_fit_M.py) (`scripts/issue722_fit_M.py`, `issue722_analyze.py`, `issue722_bootstrap.py`, `issue722_load_activations.py`, `issue722_extract_fact_rb.py`); the GPU-vectorized base-map driver + batched LOCO-MLP helper landed at SHA [`19a5758fab`](https://github.com/superkaiba/explore-persona-space/blob/19a5758fab337f6d7cd0eeecd1e9cfd461b62310/scripts/issue722_fit_M.py); figure script `scripts/issue722_figures.py` at SHA [`5132a5401f`](https://github.com/superkaiba/explore-persona-space/blob/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/scripts/issue722_figures.py). Eval JSONs — [9 per-cell + 4 aggregate JSONs](https://github.com/superkaiba/explore-persona-space/tree/3c051fcfad/eval_results/issue_722) committed on the issue-722 branch (`function_change.json`, `chain_rho_M0_Mplus.json`, `cross_transfer.json`, `nonlinearity_gap.json`, `cells/`); the corrected base-map sweep is [`base-skill-over-mean-cC-to-v0/skill_over_mean.json`](https://github.com/superkaiba/explore-persona-space/blob/78572438867299935952a33402e16c0f2493fdaf/eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json) (28 per-layer rows) and the byte-exact #658 chain reproduce-checks are [`eval_results/issue_658/a34a35_mlp_chain.json`](https://github.com/superkaiba/explore-persona-space/blob/78572438867299935952a33402e16c0f2493fdaf/eval_results/issue_658/a34a35_mlp_chain.json) + [`eval_results/issue_658_g1/a34a35_mlp_chain.json`](https://github.com/superkaiba/explore-persona-space/blob/78572438867299935952a33402e16c0f2493fdaf/eval_results/issue_658_g1/a34a35_mlp_chain.json), all at SHA `7857243886`. Figures — [`figures/issue_722/`](https://github.com/superkaiba/explore-persona-space/tree/5132a5401f/figures/issue_722) at SHA `5132a5401f` for the M0-vs-M⁺ set; the base-map per-layer figure [`base_skill_over_mean_per_layer.png`](https://github.com/superkaiba/explore-persona-space/blob/78572438867299935952a33402e16c0f2493fdaf/figures/issue_722/base_skill_over_mean_per_layer.png) is at SHA `7857243886`. Crash diagnostics — [HF `issue722_partial/att-20260628-235255/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue722_partial). The input-representation robustness round (`input-pca-robustness-cC-to-v0`) is 0 GPU-h — a 150.6-min CPU regeneration on the VM: driver `scripts/issue722_vectorized_skill.py --input-rep pca48 whiten48` at code SHA [`502a69eff2`](https://github.com/superkaiba/explore-persona-space/blob/502a69eff2bd07892d29122509fb1c245ef7faa9/scripts/issue722_vectorized_skill.py); eval JSONs [`input-pca-robustness-cC-to-v0/`](https://github.com/superkaiba/explore-persona-space/tree/37d5838f5618c2d5f6d4d6a75f2f967f10e47441/eval_results/issue_722/input-pca-robustness-cC-to-v0) (`skill_over_mean__{pca48,whiten48}.json`, `krr_vs_linear__{pca48,whiten48}.json`, `input_rep_comparison.json`, `gamma_sensitivity__pca48.json`, `run_meta.json`) + the hero figure [`input_rep_robustness_per_layer.png`](https://github.com/superkaiba/explore-persona-space/blob/670602a6af1945f6c272b3a102bc7f39ef130cb3/figures/issue_722/input_rep_robustness_per_layer.png) (full 28-layer render, regenerated at SHA `670602a6af` after the first committed copy was found to be a 2-layer smoke output); eval JSONs committed at SHA `37d5838f56`. Reused artifacts:
- Reused paired activation store from [#667](https://eps.superkaiba.com/tasks/667): [HF `issue667_gate_chain_preview/analysis_tensors`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue667_gate_chain_preview/analysis_tensors) (5760 `.npz`, seed 42, base+post-FT pairs) — fit: the only substrate carrying both base and post-FT activations on a matched context grid, the single thing that makes a pre/post M fit possible.
- Reused behavior direction from [#658](https://eps.superkaiba.com/tasks/658): [HF `issue658_theory_assumptions/store/r_b.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue658_theory_assumptions/store) (EM, sycophancy; `diffmeans`) — fit: same layer set + extraction recipe; taught-fact direction re-extracted here because #658 had no fact contrast.
- Reused adapters from [#537](https://eps.superkaiba.com/tasks/537): [HF `superkaiba1/explore-persona-space:adapters/i537_*` @48d7620508](https://huggingface.co/superkaiba1/explore-persona-space/tree/48d7620508aa7bb0662e6b34af113c21aee42b06/adapters) (r=32, rsLoRA, α∈{EM 256, fact/syco/marker/refusal 64}, modules attn+MLP) — fit: consumed only transitively through #667's post-FT activations, never applied at runtime.
- Reused leakage target from [#537](https://eps.superkaiba.com/tasks/537): `eval_results/issue_537/G_tensor/G_meta.json` (2400 cells, the `g` judge leakage rate) — fit: the on-policy judged rate the chain-ρ correlates against.

**Context:** Originating prompt (verbatim):

> run the pre vs post finetuning followup as a standalone issue linked to 667. run what we've discussed as well as the extra things here: - function-change at fixed input (‖M⁺(c) − M0(c)‖ on a common c grid), and - behavior-relevant transfer — the chain r_Bᵀ M c → E, or M evaluated along behavior directions — not generic v0 reconstruction.

Lineage: [#667](https://eps.superkaiba.com/tasks/667) — parent; supplied the matched base+post-FT activation store. Causal twin [#697](https://eps.superkaiba.com/tasks/697) (running at analysis time; cross-reference deferred). Round: `base-skill-over-mean-cC-to-v0` — same-issue follow-up, ran 2026-06-29, re-established base-map strength with the corrected skill-over-mean R² estimator. Round: `input-pca-robustness-cC-to-v0` — same-issue follow-up (user-chat, 2026-06-30), ran 2026-07-01–02; tested input-representation robustness of the base-map skill (PCA-48 / whiten-48 vs full 3584-dim c_C input). Created 2026-06-28, run 2026-06-29 (input-rep round 2026-07-02).
