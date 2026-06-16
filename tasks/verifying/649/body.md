---
title: 'For realistic sycophancy, the marker''s prior-on-level / geometry-on-change
  split half-holds: prior is null on the shift as predicted, but early-layer cosine
  also wins the absolute level, so the clean two-component rule does not transfer
  (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-06-15T23:40:47Z'
has_clean_result: false
parent_id: 532
origin_prompt: Can we run a followup to check effect of base prior on change in leakage
  for the more realistic behaviors?
goal: 'Test whether #532''s level-vs-change predictor decomposition holds for a realistic
  judged behavior (sycophancy primary, refusal stretch): per (source x bystander)
  cell, separate leakage into absolute trained expression (LEVEL) and trained-base
  shift (CHANGE), and test whether the base-model bystander prior predicts LEVEL while
  activation geometry (cosine/Gaussian-KL) predicts CHANGE - replicating or breaking
  the marker''s clean two-component rule. Reuse existing sycophancy panels (#612/#627/#507/#509)
  carrying base rate + trained rate + geometry; add a graded log-prob-readable DV
  (#391 forced-choice) only if rate-space floors.'
relates_to:
- leak-predictor
- leak-behavior-vs-marker
---
# For realistic sycophancy, the marker's prior-on-LEVEL / geometry-on-CHANGE split half-holds: the prior-is-null-on-shift half replicates, but early-layer cosine also wins LEVEL, so the clean two-component rule does not transfer (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- On the canned coverage arm (108 cells), the marker's "prior is null on the shift" half **replicates**: prior↔CHANGE **ρ = +0.14, CI [−0.07, +0.33]** covers 0; its CHANGE CV-R² uplift is **+0.04**.
- The other half **breaks**: early-layer cosine adds **+0.24 CV-R² on LEVEL** over source-intercepts+prior, beating prior's own **+0.19**. The pre-registered rule's condition (a), "prior dominates LEVEL", fails.
- Geometry decisively owns CHANGE as the marker predicted: cosine adds **+0.29 CV-R²** vs prior's **+0.04**; cosine↔CHANGE **ρ = +0.57, CI [+0.41, +0.71]**.
- The CHANGE DV is **resolution-limited**: the precision gate fired both arms (canned S/N **0.89**, on-policy **0.18**, vs planned ≈2–3). The on-policy arm is **below the floor — a null read**.
- The decomposition does **not** cleanly split leakage into a prior-driven level and a geometry-driven change: both predictors load on LEVEL, so the cross-scale prior-vs-geometry disagreement stays unresolved.

## What I ran

- **Why:** The parent marker experiment ([#532](https://eps.superkaiba.com/tasks/532)) found a clean two-component rule — a bystander's base prior predicts the *absolute* trained leakage level, activation geometry predicts the trained-minus-base *shift*, and prior is null on the shift. Two sycophancy siblings disagree on whether prior or geometry wins the single-DV race ([#507](https://eps.superkaiba.com/tasks/507) prior wins at 72B; [#509](https://eps.superkaiba.com/tasks/509) geometry wins at 7B). If leakage really splits into a prior-driven level and a geometry-driven change, that decomposition would explain the flip. This task tests whether the marker's split transfers to a realistic judged behavior.
- **Design:** Pure CPU re-analysis of an existing sycophancy panel ([#612](https://eps.superkaiba.com/tasks/612)) — no new training. Per (source × bystander) cell, leakage splits into LEVEL (absolute trained agreement rate) and CHANGE (trained − base rate); two predictors race on each DV — the bystander's base agreement prior, and source→bystander early-layer activation geometry (centered cosine + Gaussian-KL). Two arms: canned-template positives (4 sources × 30 bystanders, 108 cells — coverage anchor) and on-policy positives (2 sources, 55 cells — realism confirmation), two seeds averaged per cell.
- **Training:** N/A — no training in this task. The trained agreement rates are inherited from [#612](https://eps.superkaiba.com/tasks/612)'s already-trained adapters.
- **Eval:** DV inputs are the inherited on-policy Haiku-judged agreement rates (60-claim held-out wrong-claim pool, 600 verdicts per cell). Predictor inputs are base Qwen-2.5-7B-Instruct residual-stream activations over the 34-persona system-prompt set, re-extracted at the early-layer band where sycophancy geometry was previously shown to live (end-of-system layer 2 cosine, primary; last-prompt layer 7, secondary). Verdict per DV: six-regression incremental-validity CV-R² ladder (bystander-grouped 5-fold CV) + marginal Spearman with 1000-bootstrap 95% CIs.

## Findings

### The "prior is null on the trained shift" half of the marker rule replicates on canned sycophancy

The marker's signature was: base prior tracks the absolute level but says nothing about how far training moved a bystander. The left two panels below test that on the canned arm.

![Canned-arm scatter: base prior vs LEVEL and CHANGE (top row), early-layer cosine vs LEVEL and CHANGE (bottom row), 108 cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c69d52e0e7397018271638878be7d0867429bb39/figures/issue_649/hero2_predictor_dv_scatter_quad.png)

> **Figure.** *Base prior predicts the absolute level but is null on the shift; early-layer cosine predicts both.* Each point is one of 108 source×bystander cells (canned arm). Titles carry the marginal Spearman ρ with bootstrap 95% CI. Prior↔CHANGE (top-right) is a flat cloud, CI covers 0; cosine↔CHANGE (bottom-right) rises clearly.

- Prior↔LEVEL is positive (**ρ = +0.51, CI [+0.33, +0.64]**); prior↔CHANGE is flat (**ρ = +0.14, CI [−0.07, +0.33], covers 0**). That is exactly the marker's prior-is-null-on-shift signature.
- The non-circular check (partialling cosine out, read on the trained rate directly to avoid the base-rate-in-CHANGE artifact) keeps prior load-bearing on LEVEL: **partial ρ = +0.57, CI [+0.42, +0.68]**.

### Early-layer cosine wins LEVEL and owns CHANGE — so the clean prior-owns-LEVEL split breaks while the geometry-owns-CHANGE half holds

The marker rule needs prior to *dominate* LEVEL and geometry to dominate CHANGE. Here geometry dominates *both*.

![Canned-arm CV-R² ladder: LEVEL (left) and CHANGE (right), six nested models from source-intercepts to KL-only](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c69d52e0e7397018271638878be7d0867429bb39/figures/issue_649/hero1_cv_r2_ladder_level_vs_change.png)

> **Figure.** *On LEVEL the prior+cosine bar (0.57) clears the prior bar (0.33) by more than prior cleared source-intercepts; on CHANGE prior barely moves (0.18→0.22) and cosine carries it (→0.51).* Held-out CV-R² per nested model, canned arm, 108 cells, bystander-grouped 5-fold CV.

- LEVEL: prior over source-intercepts = **+0.19 CV-R²**; cosine on top = a further **+0.24** (→0.57). The rule wanted prior's uplift to exceed cosine's — it does not, so condition (a) **fails**.
- CHANGE: cosine adds **+0.29 CV-R²** vs prior's **+0.04** (cosine↔CHANGE **ρ = +0.57, CI [+0.41, +0.71]**), so condition (b) holds.
- **Binding caveat (drives MODERATE):** the precision gate fired both arms (canned median |CHANGE| = 0.059 vs Wilson half-width 0.066, S/N **0.89**; on-policy **0.18**; planned ≈2–3), so geometry-owns-CHANGE survives only as a *coarse* read.

### The on-policy arm is below the precision floor — a null read, not an H2 signature

The realism-tier arm (2 sources, 55 cells) appears to show prior↔CHANGE excluding zero (ρ = +0.58), which would be the "implant rides the pre-existing propensity" outcome. It is not interpretable.

- On-policy prior↔CHANGE **ρ = +0.58, CI [+0.34, +0.73]** — but with median |CHANGE| = 0.013 and S/N = 0.18, the ranked shifts sit an order of magnitude below the judge's resolution. Per-seed sign agreement drops to 0.84 (vs 0.97 canned). A resolution artifact, not evidence.
- Collinearity is not the cause: Pearson(|cosine|, prior) is −0.03 (canned) and +0.05 (on-policy), both far below the 0.6 gate, so cosine and prior are genuinely separable.

## Data

### Trained on

N/A — no training in this task. The trained agreement rates are inherited from [#612](https://eps.superkaiba.com/tasks/612)'s adapters (4 source personas trained to agree with false user claims; canned-template and on-policy positive arms, contrastive negatives, two seeds). The complete #612 training mix and judgments live on the HF data repo: [`superkaiba1/explore-persona-space-data` @ 14d541b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/14d541bbafb3dfdbe35ea7a3389df5e5a7f2c458/issue612_sycophancy_onpolicy).

### Evaluated with

The dependent variables are the inherited on-policy agreement rates: each cell is a source persona's adapter evaluated on a bystander persona over a 60-claim audited held-out wrong-claim pool, 10 rollouts × 60 claims = 600 free-generation verdicts, judged by Haiku (κ = 0.869 vs Sonnet). LEVEL = trained agreement rate `t`; CHANGE = `t − b` where `b` is the bystander's base agreement rate (52 personas measured pre-training). Predictors: base prior `b`; centered cosine and Gaussian-KL (16-D subspace, ≥32 probes/persona) between source and bystander residual activations at the early-layer band. Cells excluded: the diagonal (source = bystander) and any bystander that is a trained contrastive negative for that source (the disjointness invariant). Realized: 108 canned cells, 55 on-policy cells.

3 of 108 canned cells, random sample (seed 42):

```
source             bystander            LEVEL(t)  CHANGE(t-b)  prior(b)  cosL2
software_engineer  villain              0.375     +0.322       0.053     -0.206
villain            pirate_captain       0.307     +0.219       0.088     +0.051
comedian           web_developer        0.006     -0.036       0.042     +0.047
```

Full per-cell table (both arms, all predictors): [`eval_results/issue_649/per_cell_table.csv` @ c69d52e](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/eval_results/issue_649/per_cell_table.csv). The complete base + trained judgment files (each with raw verdicts): [HF data repo @ 14d541b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/14d541bbafb3dfdbe35ea7a3389df5e5a7f2c458/issue612_sycophancy_onpolicy).

### Generated

N/A — this task generated no model completions. The model generations underlying the DV are the inherited on-policy rollouts, in the raw-completions tree linked above.

## Reproducibility

- **Methodology:** the full findings-blind methodology + hyperparameter reference for this re-analysis is generated alongside this body at `docs/methodology/issue_649.md`.
- **Parameters:**

  | Parameter | Value |
  |---|---|
  | Task type | CPU re-analysis (no training, no pod) |
  | Substrate | #612 sycophancy panel (4 sources, 30-persona graded-cosine panel) |
  | DVs | LEVEL = trained rate; CHANGE = trained − base rate |
  | Predictors | base prior; centered cosine + Gaussian-KL @ early-layer band |
  | Geometry layers | end-of-system L2 (primary), last-prompt L7 (secondary), L20 (robustness) |
  | Gaussian-KL subspace | k = 16, ≥32 probes/persona |
  | CV | bystander-grouped 5-fold (headline); source-grouped + intercept-only M0 (robustness) |
  | Bootstrap | 1000 reps, Spearman 95% CI |
  | Collinearity gate | Pearson(\|cosine\|, prior); canned −0.03, on-policy +0.05 (both PASS) |
  | Precision gate (#391) | FIRED both arms (S/N canned 0.89, on-policy 0.18) |
  | Cells (after exclusions) | canned 108, on-policy 55 |
  | Seeds | 42, 137 (averaged per cell) |
  | Config slugs | `arm_canned`, `arm_onpolicy`; ladder models `M0`–`M5` |

- **Artifacts:**
  - CV-R² ladder: [`eval_results/issue_649/cv_r2_ladder.json` @ c69d52e](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/eval_results/issue_649/cv_r2_ladder.json)
  - Marginal Spearman + non-circular partials: [`marginal_spearman.json` @ c69d52e](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/eval_results/issue_649/marginal_spearman.json)
  - Precision + collinearity gates: [`precision_check.json`](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/eval_results/issue_649/precision_check.json), [`collinearity_gate.json`](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/eval_results/issue_649/collinearity_gate.json)
  - Per-cell table: [`per_cell_table.csv` @ c69d52e](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/eval_results/issue_649/per_cell_table.csv)
  - Figures: [`figures/issue_649/` @ c69d52e](https://github.com/superkaiba/explore-persona-space/tree/c69d52e0e7397018271638878be7d0867429bb39/figures/issue_649)
- **Compute:** 0 GPU-hours (CPU re-analysis on the VM; geometry re-extraction ran CPU-only over 34 short prompts). No pod provisioned.
- **Code:** extractor [`scripts/issue649_extract_panel_earlylayer.py`](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/scripts/issue649_extract_panel_earlylayer.py); analysis [`scripts/issue649_level_change_decomp.py`](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/scripts/issue649_level_change_decomp.py); hero re-plot [`scripts/issue649_hero_replot.py`](https://github.com/superkaiba/explore-persona-space/blob/c69d52e0e7397018271638878be7d0867429bb39/scripts/issue649_hero_replot.py); git commit `904c2668` (analysis run), `c69d52e0` (figures). Reproduce: `uv run python scripts/issue649_level_change_decomp.py` then `uv run python scripts/issue649_hero_replot.py`.
- **Reused artifacts:**
  - Reused sycophancy panel + judgments from [#612](https://eps.superkaiba.com/tasks/612): [HF data repo @ 14d541b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/14d541bbafb3dfdbe35ea7a3389df5e5a7f2c458/issue612_sycophancy_onpolicy) — fit: same base model (Qwen-2.5-7B-Instruct), on-policy judge-scored rates over the audited held-out claim pool, all 4 sources × 30 bystanders + base rates present; the realism regime is exactly the one this decomposition reads off.
  - Reused base agreement rates from [#612](https://eps.superkaiba.com/tasks/612): `eval_results/issue_612/base/judgments/` (52 personas) — fit: the bystander base prior `b`, measured pre-training on the same probe pool.
- **Context:**
  - Created / run: created 2026-06-15; analysis + figures landed 2026-06-16.
  - Follow-up to: [#532](https://eps.superkaiba.com/tasks/532) — the marker LEVEL/CHANGE decomposition this task ports to a realistic behavior.
  - Originating prompt(s), verbatim:
    > Can we run a followup to check effect of base prior on change in leakage for the more realistic behaviors?
