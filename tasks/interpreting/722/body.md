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
# Finetuning reshapes the context→answer function for a taught fact but not for emergent misalignment or sycophancy, and the primary-layer verdict is inconclusive (LOW confidence)

<!-- clean-result-v4 -->

## Takeaways

- Fitting the context→answer map M pre- and post-finetuning, **only the taught fact clears the noise floor** at the primary layer: **3.3× floor** vs **0.6×** (emergent misalignment) and **1.0×** (sycophancy).
- The co-primary read agrees: finetuning **creates a fact→leakage transfer** the base map lacked — chain Spearman ρ jumps **−0.12 → +0.50** (95% Δρ interval **+0.47 to +0.72**); the other two straddle zero.
- **The kill criterion fires "inconclusive" at the primary layer**: 2 of 3 behaviors sit at floor but fail the overfit check, so no clean call holds for them.
- The pattern is **layer- and behavior-specific**: sycophancy clears floor only at the shallowest layer (3.2×) and decays with depth; emergent misalignment never clears it.
- Binding constraint: the function-change statistic returned a **degenerate (point) interval** in every cell, the maps are weak fits (cross-transfer cosine ≈ 0 everywhere), and the store is single-seed.

## Goal

**This experiment in context:** This is the descriptive/parametric half of a two-part question about where finetuning's effect lives. The leakage theory models a behavior as `answer_profile = M(context_vector)`: finetuning could move the *input* context vector and leave the function M intact (input-shift), or change the *function* M itself (function-change). The base-model fit of M was done in [#658](https://eps.superkaiba.com/tasks/658) (which also showed raw answer-profile reconstruction saturates near 0.98 and is uninformative — hence this run reads the change only along behavior-relevant directions). The paired base+post-finetuning activation store this fit needs came from [#667](https://eps.superkaiba.com/tasks/667). Here I fit M both pre- and post-finetuning on the same context set and ask whether the function moved, per behavior. The causal twin — patch the post-finetuning context vector into the base model and see if the behavior rides along — is [#697](https://eps.superkaiba.com/tasks/697); its artifact was absent at analysis time, so the planned side-by-side concordance table is unfilled.

**Broader narrative:** This serves the leakage-predictor program's question of whether a single base-model object — the context→answer map — suffices to predict and explain how a finetuned behavior propagates across personas, or whether finetuning rewrites that object. A function-change confined to one behavior class (taught facts) while two others only shift their input would say the input-vs-function split is behavior-dependent, directly informing whether the base-model predictors the program builds can be expected to transfer post-finetuning.

## Methodology

**Design:** Pure measurement, no training. For each behavior in {emergent misalignment, sycophancy, taught fact} and each read layer in {7, 14, 21} (14 primary), I fit two maps: the base map M0 from base context vectors → base answer-profiles, and the post-finetuning map M⁺ from post-finetuning context vectors → post-finetuning answer-profiles. The single comparison is M⁺ vs M0 evaluated on the *same* grid of base context vectors. Each behavior pools 480 source×target pairs across 7 context families, BUT the context vector is keyed to the source persona (constant across a source's 30 targets), so the map M: c → v has only **≈16 distinct input vectors** — the 480 pairs repeat them per target, and the ridge effectively fits a per-source-mean answer-profile. The store is single-seed (seed 42) and holds 4 behaviors (refusal is target-only and excluded). The substrate is [#537](https://eps.superkaiba.com/tasks/537)'s already-trained behavior×context LoRA adapters, consumed only through their cached activations — nothing is re-trained or re-applied.

**Training:** N/A — no model training. The analysis hyperparameters (all inherited from #658's validated fit machinery):

| Hyperparameter | Value | Source |
|---|---|---|
| Read layers | 7, 14, 21 (14 primary) | #651 PRIMARY_LAYER + supplements |
| Ridge λ selection | closed-form PRESS over [1e-2 … 1e3] | #658 RIDGE_LAMBDAS |
| MLP architecture | 1 hidden layer × 512 wide | #658 MLP_HIDDEN |
| MLP optimizer | AdamW, lr 1e-3, weight decay 1e-4 | #658 MLP_LR/WD |
| MLP epochs | 300 | #658 MLP_MAX_EPOCHS |
| Output target dim | 64 leading answer-profile dims (shared ridge/MLP) | #658 A35_MLP_TARGET_DIM |
| Cross-validation | leave-one-context-out (LOCO) | #658 |
| Bootstrap clustering | 7 target-context families | #667 gate_chain.family_of |
| Behavior read-out | diff-in-means direction r_B | #658 DEFAULT_RB; fact re-extracted (#722) |
| Large-shift flag percentile | 90th pct of per-cell ‖cplus − c0‖ | #722 plan §3 |

**Evaluation:** The primary DV is the function-change Δ_med = the median over the context grid of `|(M⁺(c) − M0(c)) · r̂_B|` — the map's output change projected onto the behavior direction r_B, read at fixed input. It is gated against a combined noise floor = max of three nulls built through the identical refit harness: M0's refit variance, M⁺'s refit variance, and a same-function shifted-design null (fit the base function on post-finetuning inputs, read at base inputs — pure off-support extrapolation under a true input-shift). A behavior is called function-change only if Δ_med sits above this floor. The co-primary is the chain-ρ pair: held-out Spearman of `r̂_Bᵀ M̂(c)` against the measured leakage rate E (the Sonnet-4.5 judge rate `g` from #537's leakage matrix) under M0 vs M⁺, with a family-clustered interval on the difference. An overfit guard requires the nonlinear (MLP) held-out ρ to beat a within-context shuffle null on M0; where it fails, the ridge fit is the only valid map.

**Data extraction:** Activations derive from #537's frozen eval-probe pools (Betley misalignment probes, sycophancy wrong-claim probes, fact-recall probes) — established-panel (tier-2) source, no new generation. The paired base+post-finetuning context vectors and answer-profile vectors come verbatim from #667's per-cell store (5760 layer-baked `.npz`, 1152 per behavior, single-vector `(3584,)` reads per file at the baked layer). The taught-fact direction r_B has no diff-in-means contrast in #658, so it was re-extracted here as the fact-stated vs fact-absent answer-span difference under the identical recipe and saved to a parallel namespace (it does not overwrite #658's bank). The chain-ρ target E reads `g` directly from #537's committed leakage matrix (2400 cells).

**Sample training/evaluation data + completions:** This run generates no model completions (it fits linear/nonlinear maps on cached activations and projects them), so there are no response samples; the verbatim inputs are activation-derived fit statistics and the judge-rate target. Representative rows below are illustrative single cells, not random samples; the complete artifacts are linked.

- *Fitted-map cell (verbatim from `function_change.json`, the clean function-change case):* `fact/L14` → Δ_med = 0.3125, floor components {M0-refit 0.00617, M⁺-refit 0.09488, shifted 4.7e-06}, combined floor 0.09488 (M⁺-refit binds), call `H_function`, large-shift-flip false.
- *Fitted-map cell (verbatim, the at-floor case):* `em/L14` → Δ_med = 0.07060, floor components {M0-refit 0.02149, M⁺-refit 0.11098, shifted 2.0e-05}, combined floor 0.11098 (M⁺-refit binds), Δ_med below floor; the MLP fails the shuffle null (held-out ρ −0.021 < shuffle 0.076), recorded as `H_input_shuffle_failed`, not a clean input-shift.
- *Chain-ρ target (verbatim from #537 `G_meta.json`):* `fact/sp_swe__sp_swe` → g = 0.933 (base_rate 0.0); `fact/sp_swe__sp_doctor` → g = 0.867 (base_rate 0.0).

Complete per-cell + aggregate artifacts (all 9 cell JSONs + 4 aggregates): [eval_results/issue_722/ @781fc2e](https://github.com/superkaiba/explore-persona-space/tree/781fc2e97131e41ef4388cd73c91b0e70170ba32/eval_results/issue_722) (branch issue-722). Re-extracted fact direction: [HF issue722_rb_extension @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue722_rb_extension).

## Results

### Only the taught fact clears the noise floor (3.3× at the primary layer); emergent misalignment never does, sycophancy only shallow

What is plotted: for each behavior (x-axis) and layer (color), the function-change Δ_med divided by its combined noise floor. A ratio above the red line (1.0) means the map's behavior-relevant output change exceeds the floor; the c-grid is 480 source×target pairs spanning ≈16 distinct source context vectors per behavior, ridge fit.

![Grouped bar chart of function-change divided by noise floor per behavior and layer; taught fact above 1.0 at all layers, emergent misalignment below 1.0 everywhere, sycophancy above 1.0 only at layer 7](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89bc515971928a85a10ca5e304f0cbf7013d4c47/figures/issue_722/hero_function_change.png)

> **Figure.** *Only the taught fact's context→answer function changes above the noise floor.* Δ_med ÷ combined floor per behavior × layer; red line = floor. Taught fact clears it at every layer (2.9 / 3.3 / 1.6); emergent misalignment stays at 0.3–0.6; sycophancy clears only at layer 7 (3.2). 480 source×target pairs (≈16 distinct source contexts)/behavior, ridge.

Interpretation: the taught fact is the one behavior whose fitted map demonstrably differs at fixed input — the function moved. Emergent misalignment sits at 0.3–0.6× floor at every layer. Sycophancy clears the floor only at the shallowest layer and decays with depth. The binding caveat is below: this floor is dominated by M⁺'s refit variance, and the comparison uses a point estimate.

### The raw quantities behind the ratio: Δ_med versus its floor, per cell

What is plotted: each behavior×layer cell as a point, with its combined noise floor on the x-axis and its function-change Δ_med on the y-axis. Points above the dotted identity line cleared the floor.

![Scatter of function-change Delta_med versus combined floor per cell, labeled by behavior and layer; the three taught-fact points sit above the identity line, emergent-misalignment and sycophancy points on or below it](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89bc515971928a85a10ca5e304f0cbf7013d4c47/figures/issue_722/hero_function_change_points.png)

> **Figure.** *The two raw quantities behind the ratio.* Δ_med (y) vs its combined floor (x) per cell; dotted line is Δ = floor. The three taught-fact cells sit above it; emergent-misalignment and sycophancy cells sit on or below it. Both axes are projected-distance magnitudes (median |Δ·r̂_B|).

Interpretation: the floor is not a small constant — for emergent misalignment at the primary layer it is 0.111, driven by M⁺'s refit jitter, not off-support extrapolation (which is ~2e-5). The taught fact clears even this inflated floor; the other two do not. The verdict rests on the gap between Δ_med and the M⁺-refit floor, exactly where the single-seed/few-context power limit bites.

### Finetuning creates a fact→leakage transfer the base map never had (Δρ rises to +0.50)

What is plotted: per behavior×layer, the held-out Spearman ρ between the fitted map's behavior projection and the measured leakage rate, under the base map M0 (orange) and the post-finetuning map M⁺ (blue). The bracket is the family-clustered 95% interval on the difference; green = excludes zero.

![Dumbbell plot of chain Spearman rho under base versus post-finetuning maps per behavior and layer; taught-fact rows jump from near zero to about 0.5 with green difference intervals excluding zero, other rows clustered near zero with grey straddling intervals](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89bc515971928a85a10ca5e304f0cbf7013d4c47/figures/issue_722/chain_rho_pair.png)

> **Figure.** *Finetuning installs a fact→leakage transfer absent from the base map.* Held-out Spearman ρ ( r̂_Bᵀ M̂(c) vs measured leakage rate E ) under base M0 (orange) vs post-FT M⁺ (blue); bracket = Δρ 95% CI, green excludes zero. Taught fact: ρ −0.12 → +0.50 at the primary layer; the other two unchanged.

Interpretation: this co-primary agrees with the function-change read. For the taught fact the base map carries no relation to where the behavior leaks (ρ ≈ −0.12, interval through zero), but the post-finetuning map predicts leakage at ρ ≈ +0.50 — a transfer relation finetuning *created*, the difference excluding zero at all three layers. Emergent misalignment and sycophancy show no such shift. These intervals are non-degenerate, making this the more trustworthy co-primary.

### The primary-layer verdict is inconclusive: two of three behaviors are at floor but fail the overfit check

What is plotted: the function-change ratio versus read layer for each behavior, tracing the layer signature. The kill criterion is evaluated at the primary layer (14).

![Line plot of function-change ratio versus layer per behavior; taught fact stays above the floor at all layers, sycophancy decays from above-floor at layer 7 to far below at layer 21, emergent misalignment stays flat below the floor](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89bc515971928a85a10ca5e304f0cbf7013d4c47/figures/issue_722/layer_spectrum.png)

> **Figure.** *Function-change is layer- and behavior-specific.* Δ_med ÷ floor vs read layer; red line = floor. Taught fact clears it at every layer; sycophancy decays monotonically (3.2 → 1.0 → 0.1, crossing the floor near the primary layer); emergent misalignment stays flat below it.

Interpretation: at the primary layer, taught fact is a function-change, but emergent misalignment and sycophancy sit at/below floor with a nonlinear map that fails the shuffle-null overfit check (held-out ρ below chance for all three). With only ≈16 distinct source-keyed input vectors (the fit is per-source-mean — the dominant power limit), an at-floor Δ with a non-learning map is consistent with both "the function held" and "the fit was too weak to register a change" — so neither call can be made. Two of three straddling triggers the inconclusive verdict; the ridge fit is the only valid map.

<!-- concern-deferred: substrate-context-vec-keyed-to-source --> The source-keyed input (≈16 distinct vectors, per-source-mean fit) caps confidence at LOW; it does not change the per-behavior verdicts.

### Both fitted maps are weak, so the floor-relative read carries the verdict

What is plotted: at the primary layer, three cross-transfer cosines per behavior — the base map predicting post-finetuning outputs (M0→v⁺), the post-finetuning map predicting its own outputs (M⁺→v⁺), and the post-finetuning map predicting base outputs (M⁺→v0).

![Grouped bar chart of three cross-transfer cosines per behavior at layer 14; base-to-FT and FT-to-FT cosines near zero for all behaviors, FT-to-base cosine strongly negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89bc515971928a85a10ca5e304f0cbf7013d4c47/figures/issue_722/cross_transfer.png)

> **Figure.** *Both fitted maps are weak; the post-FT map mispredicts base outputs.* Held-out cross-transfer cosine at layer 14. M0→v⁺ ≈ M⁺→v⁺ ≈ 0 for every behavior (neither map predicts FT answer-profiles); M⁺→v0 is strongly negative — so raw cosine is uninformative and the floor-relative Δ is the valid read.

Interpretation: neither map reconstructs answer-profiles well (cosines near zero even on their own data), confirming #658's warning that raw reconstruction is the wrong surface at this n. This is why the headline is read as a difference relative to a refit floor (where refit noise cancels), and why confidence is capped: the maps being weak fits means the function-change statistic, though floor-corrected, rides on instruments that individually explain little.

---

**Repro:** Compute: ~2.5 h wall, ~11 GPU-h, GCP eval-h100 lane (2× H100-80, SPOT/FLEX_START); a first launch crashed in the SVD floor-refit path and was relaunched (attempt id att-20260628-235255 preserved). Ridge fits + bootstrap + figures ran off-pod on CPU. No WandB run (analysis/fit job). Code SHA [`d8451cd`](https://github.com/superkaiba/explore-persona-space/blob/d8451cdf10288db06a0190a2aa7e48d3585a9ce2/scripts/issue722_fit_M.py) (fit/analyze: `issue722_fit_M.py`, `issue722_analyze.py`, `issue722_bootstrap.py`, `issue722_load_activations.py`, `issue722_extract_fact_rb.py`) + [`89bc515`](https://github.com/superkaiba/explore-persona-space/blob/89bc515971928a85a10ca5e304f0cbf7013d4c47/scripts/issue722_figures.py) (figures). Deliverables: [eval_results/issue_722/ @781fc2e](https://github.com/superkaiba/explore-persona-space/tree/781fc2e97131e41ef4388cd73c91b0e70170ba32/eval_results/issue_722) (branch issue-722). Figures: [figures/issue_722/ @89bc515](https://github.com/superkaiba/explore-persona-space/tree/89bc515971928a85a10ca5e304f0cbf7013d4c47/figures/issue_722). Reused read-only: #667 paired activation store ([HF issue667_gate_chain_preview @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue667_gate_chain_preview) — fit: the only paired base+post-FT activation substrate, all 3 headline behaviors × 3 layers present, single-seed); #658 behavior directions ([HF issue658_theory_assumptions @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue658_theory_assumptions) — fit: same diff-in-means recipe + layer set, covers emergent misalignment + sycophancy); #537 leakage matrix `eval_results/issue_537/G_tensor/G_meta.json` (fit: the on-policy judged leakage rate E the chain-ρ targets). New artifact: re-extracted fact direction [HF issue722_rb_extension @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue722_rb_extension). Seed 42 (store-fixed; single seed). Crash diagnostics: [HF issue722_partial @de07e27](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/de07e2703db6256ccc3bb2001a57b7070c843292/issue722_partial). #697 causal cross-reference absent at analysis time (cross_ref N/A). Raw-completion upload N/A — no completions generated.

**Context:** Originating prompt (verbatim): "run the pre vs post finetuning followup as a standalone issue linked to 667. run what we've discussed as well as the extra things here: - function-change at fixed input (‖M⁺(c) − M0(c)‖ on a common c grid), and - behavior-relevant transfer — the chain r_Bᵀ M c → E, or M evaluated along behavior directions — not generic v0 reconstruction." Lineage: parent [#667](https://eps.superkaiba.com/tasks/667) (matched base+post-FT activation store) — building on [#658](https://eps.superkaiba.com/tasks/658) (base-only map fit) and [#537](https://eps.superkaiba.com/tasks/537) (the trained adapters); causal twin [#697](https://eps.superkaiba.com/tasks/697). Created 2026-06-28; run 2026-06-29.
