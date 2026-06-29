---
title: Finetuning measurably reshapes the context→answer map only for a taught fact
  (LOW confidence)
kind: experiment
tags: []
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

**Methodology:** [docs/methodology/issue_722.md](https://github.com/superkaiba/explore-persona-space/blob/3d7b7c6695201d826e7a82ef1ebd88ebc1bace5c/docs/methodology/issue_722.md) · [gist](https://gist.github.com/superkaiba/e6bb990f9c5a4be34360d2f2b319636e)

## Takeaways

- **1.6–3.3× floor** — the taught-fact fitted context→answer map measurably reshapes after finetuning (3.3× at the primary layer; 2.9× / 1.6× elsewhere); the function moved, not just its input.
- **0.3–0.6× floor** at every layer for **harmful-compliance (EM)**; sycophancy runs **3.2×** (layer 7) down to **0.1×** (layer 21) — below floor where it matters.
- **Inconclusive** kill criterion: EM and sycophancy fail the MLP-vs-shuffle power check at the primary layer, so their below-floor readings are not certified "function held".
- **n=16** source-keyed context inputs per behavior with a degenerate per-cell bootstrap CI — inference rests on the cross-behavior / cross-layer pattern, not any single cell.
- **Floor = post-FT refit variance** in all 9 cells, so the fact signal already clears the "post-FT map is just noisier to refit" alternative.
- **ρ ≈0 → +0.50** post-FT at the primary layer (CI excludes zero) for the fact only — a context→leakage transfer chain that the base map lacked; EM and sycophancy do not move.

## Goal

**This experiment in context:** This is the descriptive/parametric half of a two-part question about *where* finetuning lives. The leakage theory models behavior installation as either moving the input context vector `c_C` that feeds a context→answer-profile map M, or changing M itself. The base-only fit of M was built in [#658](https://eps.superkaiba.com/tasks/658); the matched base-and-post-finetuning activation store that makes a pre/post comparison possible was extracted in [#667](https://eps.superkaiba.com/tasks/667). This run fits M0 (base) and M⁺ (adapter-applied) explicitly on [#537](https://eps.superkaiba.com/tasks/537)'s already-trained behavior×context LoRA adapters and asks whether the *function* M moved, distinct from the *input* moving. The causal twin is [#697](https://eps.superkaiba.com/tasks/697), which injects the finetuned context vector into the base model and measures the mediated fraction; here the comparison is parametric (a fitted-map difference), and the two pictures are meant to be read side by side (#697 was still running at analysis time, so no cross-reference table is included).

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

## Results

### The taught fact's function-change clears its floor 1.6–3.3×; EM sits below floor; sycophancy peaks shallow and dies

What is plotted: per-cell ratio Δ_med ÷ floor_combined (the kill-criterion denominator), 3 behaviors × 3 layers. Δ_med = median over the base context grid of `|M⁺(c) − M0(c)|` along the behavior direction; 1× = at floor. Ridge fit, 16 source contexts, seed 42.

![Heatmap of function-change over the kill-criterion floor for taught fact, harmful compliance (EM) and sycophancy at layers 7, 14, 21; taught fact reads 2.90, 3.29, 1.58, EM reads 0.34, 0.64, 0.64, sycophancy reads 3.18, 0.96, 0.12](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/hero_function_change_heatmap.png)

> **Figure.** *Only the taught fact clears the noise floor at the primary layer.* Cells are Δ_med ÷ floor_combined; >1× means the map changed more than its own refit noise. n=16 source contexts, seed 42, ridge fit. Absolute magnitudes are in the next figure.

- Taught fact clears the floor at all three layers (2.90× / 3.29× / 1.58×); EM stays below floor everywhere (0.34× / 0.64× / 0.64×); sycophancy clears it at layer 7 (3.18×) but collapses to 0.12× by layer 21.
- The combined floor is the post-FT map's own refit variance (`floor_combined = floor_Mplus_refit` in all 9 cells), so the fact's 3.29× already clears the planned "post-FT map is just noisier to refit" confound.
- EM and sycophancy fail the MLP-vs-shuffle check at the primary layer, so their below-floor readings are inconclusive, not a positive "function held".

### The same change, in raw units against its paired floor

What is plotted: per cell, the raw function-change Δ_med next to its combined floor, in r_B units — the unnormalized data behind the heatmap ratio. Δ_med above floor = function changed; Δ_med ≈ floor = no detectable change.

![Grouped bars of function-change delta vs combined floor for nine behavior-layer cells; fact bars exceed their floors, EM bars sit below, sycophancy L7 clears a tiny floor, sycophancy L21 sits far below a large floor](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/function_change_delta_vs_floor.png)

> **Figure.** *The taught-fact Δ towers over its floor; EM and late-layer sycophancy Δ sit below theirs.* Blue = Δ_med, orange = combined floor (= the M⁺ refit-variance null in every cell). Ridge fit, family-clustered bootstrap over 7 context families.

- The floor swings widely (0.004 at sycophancy L7 to 0.27 at sycophancy L21), so a large absolute Δ can sit below a large floor (sycophancy L21: Δ=0.034 vs floor 0.27).
- Support distance (mean / p90 of `‖cplus − c0‖`) ranges 6.8 / 8.7 (sycophancy L7) to 88.5 / 101.8 (sycophancy L21); excluding the 30 largest-shift contexts per cell flips NO call (Δ_med_excl on the same side of floor as Δ_med, all 9 cells).
- The off-support extrapolation null stayed ≈1e-5 even at large shifts, so the binding test was always "is Δ bigger than the post-FT map's refit noise."

### A behavior-relevant transfer chain appears for the taught fact after finetuning

What is plotted: per cell, the Spearman ρ between the held-out prediction along the behavior direction (`r_Bᵀ M̂(c)`) and the judge leakage rate E, under the base map M0 (orange) and post-FT map M⁺ (blue), 95% family-clustered CIs. A rightward shift = finetuning created a transfer the base map lacked.

![Forest plot of chain rho under base map M0 and post-FT map M+ for nine cells; taught fact L14 and L21 post-FT points sit at +0.50 and +0.46 with CIs excluding zero while base points sit near zero; EM and sycophancy points cluster around zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/chain_rho_shift_forest.png)

> **Figure.** *For the taught fact only, the context→leakage chain jumps from ≈0 in the base map to +0.46–0.50 post-finetuning.* Held-out LOCO ridge ρ vs the judge leakage rate E for the trained-on (target, behavior, persona-context) tuple; 95% family-clustered CI over 7 families. EM and sycophancy chain shifts all straddle zero.

- Fact chain ρ rises from −0.12 (base, CI straddles zero) to **+0.50** (post-FT, primary layer, 95% CI +0.31 to +0.64); +0.46 at layer 21, +0.17 at layer 7.
- Ridge and MLP agree (+0.502 vs +0.490 at the primary layer) despite the MLP's overall reconstruction failure — both recover this single scalar against E.
- Every EM and sycophancy chain-shift CI straddles zero, so the fact change is a real new context→leakage relationship, not a generic wobble.

### Cross-transfer is asymmetric: the post-FT map generalizes worse than the base map

What is plotted: per cell, the held-out cosine of the base map M0 predicting the post-FT profile v⁺, the post-FT map M⁺ predicting v⁺, and M⁺ predicting the *base* profile v0. The third bar is the asymmetry — how well the finetuned map back-predicts the base answer profile.

![Grouped bars of cross-transfer cosines for nine cells; base-map and post-FT-map predictions of the post-FT profile both sit near zero, while the post-FT map predicting the base profile sits far below at -0.22 to -0.32](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/cross_transfer_orthogonality.png)

> **Figure.** *Neither map predicts the post-FT profile (cosine ≈0), but the post-FT map predicts the base profile worst (−0.22 to −0.32).* Orange/blue = base/post-FT map → post-FT profile; green = post-FT map → base profile. Held-out cross-transfer cosine, 9 cells.

- Both maps predict v⁺ about equally poorly (cosines near zero — e.g. at fact L7 they agree within ≈0.0023), so finetuning did not rotate M toward v⁺.
- The asymmetry is that M⁺ predicts the *base* profile v0 markedly worse (cosine −0.22 to −0.32) than M0 predicts v⁺ — the finetuned map has lost base-profile alignment without gaining post-FT-profile predictive power.
- So "M changed" does NOT mean "M now points at the trained behavior": the change degrades back-prediction rather than realigning toward v⁺.

### Why the EM/sycophancy verdict is inconclusive, not a clean "function held"

What is plotted: per cell, the MLP's held-out base-map reconstruction ρ vs its shuffle null. The MLP is trusted only where `rho_M0_mlp > rho_M0_shuffle`.

![Bars of MLP held-out rho vs shuffle null for nine cells; MLP bars fail to beat the shuffle null in eight of nine cells, passing only at taught fact layer 7](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/figures/issue_722/mlp_shuffle_diagnostic.png)

> **Figure.** *The nonlinear MLP map fails its shuffle null in 8 of 9 cells, passing only at taught fact, layer 7.* Blue = MLP held-out base-map ρ, gray = shuffle null. With the nonlinear fit unreliable, the ridge fit is the only valid map.

- MLP fails the shuffle null in 8 of 9 cells (beating it only at fact L7, ρ=−0.077 vs −0.080), so the nonlinearity-gap read is dropped and the headline is ridge-only.
- Inconclusive because certifying "function held" needs below-floor AND a passing MLP fit; at n=16 single seed an at-floor Δ is indistinguishable from an underpowered one.
- The per-cell Δ_med bootstrap CI is degenerate (point = lo = hi), so the cross-behavior pattern carries inference, not any single cell.
- The input `c_C` is source-keyed, so M sees only 16 distinct inputs, not 480 — the dominant cap on the read. <!-- concern-deferred: substrate-context-vec-keyed-to-source -->
- The positive fact result survives: floor 1.6–3.3× and chain-shift CI excludes zero.

---

**Repro:** Compute — GCP `eval-h100` lane (`a3-highgpu-2g`, 2× H100-80, FLEX_START), ~8 GPU-h across two launches (a first launch crashed in the SVD floor-refit path; relaunch wall 5h13m). No WandB run (analysis/fitting job). Code SHA [`3c051fcfad`](https://github.com/superkaiba/explore-persona-space/blob/3c051fcfad/scripts/issue722_fit_M.py) (`scripts/issue722_fit_M.py`, `issue722_analyze.py`, `issue722_bootstrap.py`, `issue722_load_activations.py`, `issue722_extract_fact_rb.py`); figure script `scripts/issue722_figures.py` at SHA [`5132a5401f`](https://github.com/superkaiba/explore-persona-space/blob/5132a5401f00cc07ec9c2ffbe55b6aa5fec8a0e1/scripts/issue722_figures.py). Eval JSONs — [9 per-cell + 4 aggregate JSONs](https://github.com/superkaiba/explore-persona-space/tree/3c051fcfad/eval_results/issue_722) committed on the issue-722 branch (`function_change.json`, `chain_rho_M0_Mplus.json`, `cross_transfer.json`, `nonlinearity_gap.json`, `cells/`). Figures — [`figures/issue_722/`](https://github.com/superkaiba/explore-persona-space/tree/5132a5401f/figures/issue_722) at SHA `5132a5401f`. Crash diagnostics — [HF `issue722_partial/att-20260628-235255/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue722_partial). Reused artifacts:
- Reused paired activation store from [#667](https://eps.superkaiba.com/tasks/667): [HF `issue667_gate_chain_preview/analysis_tensors`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue667_gate_chain_preview/analysis_tensors) (5760 `.npz`, seed 42, base+post-FT pairs) — fit: the only substrate carrying both base and post-FT activations on a matched context grid, the single thing that makes a pre/post M fit possible.
- Reused behavior direction from [#658](https://eps.superkaiba.com/tasks/658): [HF `issue658_theory_assumptions/store/r_b.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue658_theory_assumptions/store) (EM, sycophancy; `diffmeans`) — fit: same layer set + extraction recipe; taught-fact direction re-extracted here because #658 had no fact contrast.
- Reused adapters from [#537](https://eps.superkaiba.com/tasks/537): [HF `superkaiba1/explore-persona-space:adapters/i537_*` @48d7620508](https://huggingface.co/superkaiba1/explore-persona-space/tree/48d7620508aa7bb0662e6b34af113c21aee42b06/adapters) (r=32, rsLoRA, α∈{EM 256, fact/syco/marker/refusal 64}, modules attn+MLP) — fit: consumed only transitively through #667's post-FT activations, never applied at runtime.
- Reused leakage target from [#537](https://eps.superkaiba.com/tasks/537): `eval_results/issue_537/G_tensor/G_meta.json` (2400 cells, the `g` judge leakage rate) — fit: the on-policy judged rate the chain-ρ correlates against.

**Context:** Originating prompt (verbatim):

> run the pre vs post finetuning followup as a standalone issue linked to 667. run what we've discussed as well as the extra things here: - function-change at fixed input (‖M⁺(c) − M0(c)‖ on a common c grid), and - behavior-relevant transfer — the chain r_Bᵀ M c → E, or M evaluated along behavior directions — not generic v0 reconstruction.

Lineage: [#667](https://eps.superkaiba.com/tasks/667) — parent; supplied the matched base+post-FT activation store. Causal twin [#697](https://eps.superkaiba.com/tasks/697) (running at analysis time; cross-reference deferred). Created 2026-06-28, run 2026-06-29.
