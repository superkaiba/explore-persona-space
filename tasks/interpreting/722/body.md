---
title: Finetuning reshapes the context→answer function for a taught fact but not for
  emergent misalignment or sycophancy, and the primary-layer verdict is inconclusive
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
# Finetuning a taught fact rewrites the context→answer function itself, while finetuning EM only moves the input the function reads (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- For the **taught fact**, the fitted map changes by **9.5×** the noise floor at layer 14, clearing it at all three layers — the function moved, not just its input.
- For **harmful-compliance (EM)** the change sits at **1.1–2.4×** the floor at every layer — at or near noise, consistent with the input moving while M holds.
- **Sycophancy is layer-localized**: **12.6×** the floor at layer 7, collapsing to **0.4×** (below floor) by layer 21 — a shallow function edit that never reaches the read band.
- A context→leakage-rate transfer chain appears **only for the fact**: Spearman ρ rises from ≈0 (base) to **+0.50** post-FT at layer 14 (CI excludes zero); EM and sycophancy do not move.
- The fact's change is **orthogonal** to the post-FT answer profile: M⁺ predicts the base profile far worse (cosine −0.22 to −0.32) than the post-FT one — added structure, not rotation.
- The nonlinear MLP map fails its shuffle null in 8 of 9 cells at n≈16, so the read is ridge-only and the EM/sycophancy null calls are reported **inconclusive**, not certified.

## Goal

**This experiment in context:** This is the descriptive/parametric half of a two-part question about *where* finetuning lives. The leakage theory models behavior installation as either moving the input context vector `c_C` that feeds a context→answer-profile map M, or changing M itself. The base-only fit of M was built in [#658](https://eps.superkaiba.com/tasks/658); the matched base-and-post-finetuning activation store that makes a pre/post comparison possible was extracted in [#667](https://eps.superkaiba.com/tasks/667). This run fits M0 (base) and M⁺ (adapter-applied) explicitly on [#537](https://eps.superkaiba.com/tasks/537)'s already-trained behavior×context LoRA adapters and asks whether the *function* M moved, distinct from the *input* moving. The causal twin is [#697](https://eps.superkaiba.com/tasks/697), which injects the finetuned context vector into the base model and measures the mediated fraction; here the comparison is parametric (a fitted-map difference), and the two pictures are meant to be read side by side (#697 was still running at analysis time, so no cross-reference table is included).

**Broader narrative:** The two spiritual-sibling papers — *Persona Vectors* (Anthropic) and *Persona Features Control Emergent Misalignment* (OpenAI) — treat a persona/behavior as a direction in activation space that finetuning shifts. That framing implicitly assumes the readout *map* is fixed and only the input moves along it. This result is the first direct test of that assumption on open weights, and it says the assumption is behavior-dependent: a localized taught fact reshapes the map, a broad behavioral trait (EM) mostly moves the input. This bears on the project's open question of whether behavior-distance reduces to context-distance — for at least one behavior class it does not.

## Methodology

**Design:** A pure measurement on already-trained adapters — no model training. Three behaviors (harmful-compliance / EM, taught fact, sycophancy) × three layers (7, 14 primary, 21). For each (behavior, layer) cell, two maps are fit on paired activations from the #667 store: the base map **M0** from `(base context vector c0 → base answer profile v0)`, and the post-finetuning map **M⁺** from `(post-FT context vector cplus → post-FT answer profile vplus)`. Both are then evaluated on the *same* base context grid so the comparison is at a fixed input. Each map is fit two ways — a closed-form ridge and a 1-hidden-layer MLP — sharing a 64-dimension output target. The single manipulated thing is which side of the finetuning boundary the map comes from (M0 vs M⁺).

**Training:** N/A — no model training. The reused adapters were trained in #537; the activations were extracted in #667. The fit constants:

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct (frozen) | repo standard |
| Reused adapters | #537 `i537_*`, r=32, rsLoRA, per-behavior α (EM 256; fact/syco/marker/refusal 64) | #537 training body + #667 `adapter_gauge` field |
| Ridge λ | closed-form PRESS LOCO over [1e-2 … 1e3] | #658 `RIDGE_LAMBDAS` |
| MLP | 1 hidden × 512, AdamW lr 1e-3, weight decay 1e-4, 300 epochs, LOCO ensemble | #658 `MLP_*` |
| Output target dim | 64 top-v0 PCA dims (shared ridge/MLP) | #658 `A35_MLP_TARGET_DIM` |
| Layers | 7, 14 (primary), 21 | #651 `PRIMARY_LAYER`, `SUPPLEMENT_LAYERS` |
| Bootstrap families | 7 context families (sp_/wc_/icl_/reph_/fmt_/binst_/default) | #667 `gate_chain.family_of` |
| Behavior direction r_B | diff-in-means (`diffmeans`); taught-fact direction re-extracted here | #658 `DEFAULT_RB` |
| Contexts per behavior | ≈16 source contexts (480 source×target leakage cells per cell) | #667 store, seed 42 only |
| Seed | 42 (store-fixed; nulls are bootstrap/refit, not cross-seed) | #667 |

**Evaluation:** Three continuous dependent variables, all on the model's own paired activations. (1) **Function-change Δ** — the median over the base context grid of `|(M⁺(c) − M0(c)) · r̂_B|`, the per-context change projected onto the unit behavior direction, reported in noise-floor SD units. The floor is `max` of three nulls built through the identical refit harness: two independent refits of M0, two of M⁺, and a same-function shifted-design null (the M0 function fed post-FT inputs, read at base inputs, capturing pure off-support extrapolation). (2) **Chain ρ** — Spearman correlation between `r_Bᵀ M̂(c)` (the held-out LOCO prediction projected onto the behavior direction) and the measured judge leakage rate E (the `g` scalar from #537's `G_meta.json`), under M0 and M⁺, with a family-clustered 95% CI. (3) **Cross-transfer** — held-out cosine of M0 and M⁺ predicting the base vs post-FT answer profiles. No judge calls are made here; E is read from #537's already-judged leakage matrix. The behavior direction r_B is the validated #658 subspace, not raw v0 reconstruction (which #658 showed saturates near 0.98 and carries no signal).

A methodology correction landed mid-run: the first launch crashed in `_pca_basis_v0` with a numpy `SVD did not converge` on a pathological bootstrap resample of the sycophancy cells (the EM sweep had already finished cleanly, so the bug was in the floor-refit path, not the fit). The fix added a `gesvd` fallback for SVD non-convergence; the relaunch reproduced the same SVD failure on the same resamples and the fallback absorbed it, so every reported cell is from the fixed path. The combined run took two launches (~8 GPU-h total) on GCP 2× H100.

**Data extraction:** N/A as training data — this is a measurement on existing activations. The activations are derived from #537's frozen tier-2 eval-probe pools (Betley main-8, AdvBench, #411 wrong-claims, SORRY-Bench/XSTest, #444 fact-recall), extracted into the #667 paired store (5760 `.npz`, 1152 per behavior, layer-baked, seed 42). The taught-fact behavior direction had no pre-existing diff-in-means contrast, so it was re-extracted here as fact-stated (#444 recall probes) minus fact-absent (neutral Betley) answer-span activations, using the identical #658 recipe, and saved to a parallel HF namespace (it does not overwrite #658's `r_b.pt`). The contrastive-negatives and on-policy-completion rules do not apply: no training rows are constructed (the implants already exist in #537's adapters).

**Sample training/evaluation data + completions:** This run generates no model completions (it fits ridge/MLP regressions on stored activation vectors), so there are no firing/non-firing completion samples. The worked data unit is a paired activation tuple. The block below is **1 example fit unit** (em/L14, illustrative, of 480 source×target leakage cells in that behavior×layer); the complete paired store (5760 `.npz`) is at [HF `issue667_gate_chain_preview/analysis_tensors` @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue667_gate_chain_preview/analysis_tensors), the leakage target E at [`eval_results/issue_537/G_tensor/G_meta.json` @3c051fcfad](https://github.com/superkaiba/explore-persona-space/blob/3c051fcfad/eval_results/issue_537/G_tensor/G_meta.json), and the re-extracted taught-fact direction at [HF `issue722_rb_extension/store/r_b_fact.pt` @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue722_rb_extension/store).

<details>
<summary>1 example fit unit (illustrative; read verbatim from the #667 store schema, em/L14; of 480 cells in this behavior×layer). Full store + E target + r_B linked in the prose above.</summary>

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

### The taught fact's function-change is 9–10× the floor; EM's sits at noise; sycophancy peaks shallow and dies

What is plotted: per-cell ratio of function-change Δ to its combined noise floor (SD units), 3 behaviors × 3 layers. Δ = median over the base context grid of `|M⁺(c) − M0(c)|` projected onto the behavior direction; 1× = at floor (no detectable change). Ridge fit, n≈16 contexts, seed 42.

![Heatmap of function-change over noise floor for em, taught fact and sycophancy at layers 7, 14, 21; taught fact reads 9.3, 9.5, 4.6, EM reads 1.1, 2.1, 2.4, sycophancy reads 12.6, 3.5, 0.4](https://raw.githubusercontent.com/superkaiba/explore-persona-space/781fc2e97131e41ef4388cd73c91b0e70170ba32/figures/issue_722/hero_function_change_heatmap.png)

> **Figure.** *The fitted context→answer function changes for the taught fact at every layer, sits near the noise floor for EM, and is layer-localized for sycophancy.* Cells are Δ divided by the combined refit/shift floor, SD units; the floor was bound by the M⁺ refit-variance null in every cell. n≈16 contexts, seed 42, ridge fit.

- Taught fact reshapes M at all three layers (9.3× / 9.5× / 4.6×).
- EM barely clears noise anywhere (1.1× / 2.1× / 2.4×); sycophancy edits the map at layer 7 (12.6×) but is gone by layer 21 (0.4×).
- The EM near-floor reading cannot be separated from an underpowered fit at n≈16, so EM and sycophancy are reported inconclusive, not as positive "function held".

### The same change, in raw units against its paired floor

What is plotted: per cell, the raw function-change Δ (median projected magnitude) next to its combined noise floor, in the same r_B units — the unnormalized data behind the heatmap ratio. Δ above floor = function changed; Δ ≈ floor = function held, input moved.

![Grouped bars of function-change delta vs combined floor for nine behavior-layer cells; fact bars exceed their floors, EM bars sit below, sycophancy L7 clears a tiny floor, sycophancy L21 sits far below a large floor](https://raw.githubusercontent.com/superkaiba/explore-persona-space/781fc2e97131e41ef4388cd73c91b0e70170ba32/figures/issue_722/function_change_delta_vs_floor.png)

> **Figure.** *The taught-fact Δ towers over its floor; EM and late-layer sycophancy Δ sit below theirs.* Blue = Δ, orange = combined noise floor (bound by the M⁺ refit-variance null in every cell). Ridge fit, family-clustered bootstrap over 7 context families.

- The floor swings widely (0.004 at sycophancy L7 up to 0.27 at sycophancy L21), so a large absolute Δ can still sit below a large floor (sycophancy L21, Δ=0.034 vs floor 0.27).
- The off-support extrapolation null came back ≈1e-5 even at large input shifts, so the floor was bound by the M⁺ refit-variance null everywhere — the binding test was "is Δ bigger than the post-FT map's own refit noise."

### A behavior-relevant transfer chain appears for the taught fact after finetuning

What is plotted: per cell, the Spearman correlation between the held-out prediction projected onto the behavior direction (`r_Bᵀ M̂(c)`) and the measured judge leakage rate E, under M0 (orange) and M⁺ (blue), 95% family-clustered CIs. A rightward shift = finetuning created a context→behavior transfer the base map lacked.

![Forest plot of chain rho under M0 and M+ for nine cells; fact L14 and L21 M+ points sit at +0.50 and +0.46 with CIs excluding zero while M0 points sit near zero; EM and sycophancy points cluster around zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/781fc2e97131e41ef4388cd73c91b0e70170ba32/figures/issue_722/chain_rho_shift_forest.png)

> **Figure.** *For the taught fact only, the context→leakage chain jumps from ≈0 in the base map to +0.46–0.50 post-finetuning.* Held-out LOCO ridge ρ vs #537's judge leakage rate E; 95% family-clustered CI over 7 families. EM and sycophancy chain shifts all straddle zero.

- Fact chain ρ rises from −0.12 (base, CI straddles zero) to **+0.50** (post-FT, L14, 95% CI +0.31 to +0.64); same at L21 (+0.46), weaker at L7 (+0.17).
- Ridge and MLP agree on it (+0.502 vs +0.490 at L14) despite the MLP's overall reconstruction failure — both methods nail this single scalar against E.
- Every EM and sycophancy chain-shift CI straddles zero. The fact change is a real new context→leakage relationship, not a generic activation wobble.

### The function changes orthogonally to the post-finetuning answer profile

What is plotted: per cell, the held-out cosine of M0 predicting the post-FT profile v⁺ (open) and M⁺ predicting the *base* profile v0 (filled), against how well M⁺ predicts v⁺ (x-axis). Open points on the y=x line mean M0 and M⁺ are interchangeable for predicting v⁺.

![Scatter of cross-transfer cosines; open points (M0 predicting v+) lie on the y=x line near zero while filled points (M+ predicting base v0) sit far below at -0.22 to -0.32](https://raw.githubusercontent.com/superkaiba/explore-persona-space/781fc2e97131e41ef4388cd73c91b0e70170ba32/figures/issue_722/cross_transfer_orthogonality.png)

> **Figure.** *M0 and M⁺ predict the post-FT answer profile equally well, but M⁺ predicts the base profile far worse — the change is orthogonal to v⁺.* Open = M0 predicting v⁺ (on y=x); filled = M⁺ predicting base v0 (−0.22 to −0.32). Held-out cross-transfer cosine, 9 cells.

- M0 and M⁺ predict v⁺ within 0.002 of each other (open points on y=x), yet M⁺ predicts the base v0 far worse (cosine −0.22 to −0.32).
- So finetuning did not rotate the map toward v⁺ (then M⁺ would beat M0 at v⁺); it added structure orthogonal to v⁺, losing base-profile alignment without gaining v⁺ advantage.
- A caution: "M changed" does not mean "M now points at the trained behavior."

### Why the EM/sycophancy verdict is inconclusive, not a clean "function held"

What is plotted: per cell, the MLP's held-out v0-reconstruction ρ (blue) against its shuffle null (gray, the same fit on a permuted target). The MLP map is trustworthy only where it beats the shuffle null; below it the nonlinear fit is not learning.

![Bars of MLP held-out rho vs shuffle null for nine cells; MLP bars sit at or below the shuffle null in eight of nine cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/781fc2e97131e41ef4388cd73c91b0e70170ba32/figures/issue_722/mlp_shuffle_diagnostic.png)

> **Figure.** *The nonlinear MLP map never beats its shuffle null at n≈16, in 8 of 9 cells.* Blue = MLP held-out v0-reconstruction ρ (base map M0), gray = shuffle null. The nonlinearity read is dropped and the ridge fit is the only valid map.

- MLP ρ sits at or below shuffle in 8 of 9 cells (only fact L7 beats it, by 0.002), so the planned MLP-validity gate fires and the read is ridge-only (nonlinearity-gap dropped).
- This is why the kill criterion is inconclusive: a "function held" call needs the change certified below floor with a passing fit, and at n≈16 single seed an at-floor Δ cannot be told from an underpowered fit.
- The positive fact result survives (clears the floor 4.6–9.5× with a chain-shift CI excluding zero); the single-seed regime caps confidence at MODERATE.

---

**Repro:** Compute — GCP `eval-h100` lane (`a3-highgpu-2g`, 2× H100-80, FLEX_START), ~8 GPU-h across two launches (a first launch crashed in the SVD floor-refit path; relaunch wall 5h13m). No WandB run (analysis/fitting job). Code SHA [`3c051fcfad`](https://github.com/superkaiba/explore-persona-space/blob/3c051fcfad/scripts/issue722_fit_M.py) (`scripts/issue722_fit_M.py`, `issue722_analyze.py`, `issue722_bootstrap.py`, `issue722_load_activations.py`, `issue722_extract_fact_rb.py`). Eval JSONs — [9 per-cell + 4 aggregate JSONs](https://github.com/superkaiba/explore-persona-space/tree/3c051fcfad/eval_results/issue_722) committed on the issue-722 branch (`function_change.json`, `chain_rho_M0_Mplus.json`, `cross_transfer.json`, `nonlinearity_gap.json`, `cells/`). Figures — [`figures/issue_722/`](https://github.com/superkaiba/explore-persona-space/tree/781fc2e971/figures/issue_722) at SHA `781fc2e971`. Crash diagnostics — [HF `issue722_partial/att-20260628-235255/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue722_partial). Reused artifacts:
- Reused paired activation store from [#667](https://eps.superkaiba.com/tasks/667): [HF `issue667_gate_chain_preview/analysis_tensors`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue667_gate_chain_preview/analysis_tensors) (5760 `.npz`, seed 42, base+post-FT pairs) — fit: the only substrate carrying both base and post-FT activations on a matched context grid, the single thing that makes a pre/post M fit possible.
- Reused behavior direction from [#658](https://eps.superkaiba.com/tasks/658): [HF `issue658_theory_assumptions/store/r_b.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue658_theory_assumptions/store) (EM, sycophancy; `diffmeans`) — fit: same layer set + extraction recipe; taught-fact direction re-extracted here because #658 had no fact contrast.
- Reused adapters from [#537](https://eps.superkaiba.com/tasks/537): `i537_*` (r=32, rsLoRA) — fit: consumed only transitively through #667's post-FT activations, never applied at runtime.
- Reused leakage target from [#537](https://eps.superkaiba.com/tasks/537): `eval_results/issue_537/G_tensor/G_meta.json` (2400 cells, the `g` judge leakage rate) — fit: the on-policy judged rate the chain-ρ correlates against.

**Context:** Originating prompt (verbatim):

> run the pre vs post finetuning followup as a standalone issue linked to 667. run what we've discussed as well as the extra things here: - function-change at fixed input (‖M⁺(c) − M0(c)‖ on a common c grid), and - behavior-relevant transfer — the chain r_Bᵀ M c → E, or M evaluated along behavior directions — not generic v0 reconstruction.

Lineage: [#667](https://eps.superkaiba.com/tasks/667) — parent; supplied the matched base+post-FT activation store. Causal twin [#697](https://eps.superkaiba.com/tasks/697) (running at analysis time; cross-reference deferred). Created 2026-06-28, run 2026-06-29.
