---
title: 'The marker''s prior-on-LEVEL / geometry-on-CHANGE split for realistic sycophancy:
  prior owns LEVEL on the on-policy arm but loses it on canned templates, while prior
  is null on the shift on both (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-06-15T23:40:47Z'
has_clean_result: true
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
# The marker's prior-on-LEVEL / geometry-on-CHANGE split for realistic sycophancy: prior owns LEVEL on the on-policy arm but loses it on canned templates, while prior is null on the shift on both (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- **Which predictor owns the absolute level (LEVEL) flips between arms.** On the realistic on-policy arm prior dominates LEVEL (CV-R² uplift **+0.55** vs cosine **+0.24**) — marker rule holds; on canned templates it fails (**+0.19** < **+0.24**).
- That "geometry also owns LEVEL" break is **canned-template-specific**. [#612](https://eps.superkaiba.com/tasks/612) measured canned installing +0.84–0.93 vs on-policy +0.60–0.66, so the cleaner arm reverses it.
- The other half, "prior null on the shift (CHANGE)", **replicates on canned**: prior↔CHANGE **+0.14 [−0.07, +0.33]** covers 0; cosine↔CHANGE **+0.57 [+0.41, +0.71]**. Canned LEVEL marginals are **tied** (both +0.51).
- Cosine-owns-CHANGE is **not source-uniform**: pooled +0.57, but per source villain **+0.77** to software-engineer **−0.12** (p = 0.55). Holds across layers (L2, L7, L20), not early-layer-only.
- The CHANGE DV is **resolution-limited** (S/N canned **0.89**, on-policy **0.18**) — but this caps only CHANGE; LEVEL is a directly-measured rate, so the on-policy LEVEL reversal is real.

## What I ran

- **Why:** The parent marker experiment ([#532](https://eps.superkaiba.com/tasks/532)) found a clean two-component rule — a bystander's base prior predicts the *absolute* trained leakage level, activation geometry predicts the trained-minus-base *shift*, and prior is null on the shift. Two sycophancy siblings disagree on whether prior or geometry wins the single-DV race ([#507](https://eps.superkaiba.com/tasks/507) prior wins at 72B; [#509](https://eps.superkaiba.com/tasks/509) geometry wins at 7B), and the fact-leakage line found base prior a durable level predictor ([#500](https://eps.superkaiba.com/tasks/500), [#541](https://eps.superkaiba.com/tasks/541)). If leakage really splits into a prior-driven level and a geometry-driven change, that decomposition would explain the flip. This task tests whether the marker's split transfers to a realistic judged behavior — and whether it depends on how the behavior was installed ([#612](https://eps.superkaiba.com/tasks/612)'s canned-vs-on-policy contrast, which [#627](https://eps.superkaiba.com/tasks/627) showed matters for install strength).
- **Design:** Pure CPU re-analysis of an existing sycophancy panel ([#612](https://eps.superkaiba.com/tasks/612)) — no new training. Per (source × bystander) cell, leakage splits into LEVEL (absolute trained agreement rate) and CHANGE (trained − base rate); two predictors race on each DV — the bystander's base agreement prior, and source→bystander early-layer activation geometry (centered cosine + Gaussian-KL). Two arms: canned-template positives (4 sources × 30 bystanders, 108 cells — coverage anchor) and on-policy positives (2 sources, 55 cells — realism confirmation), two seeds averaged per cell.
- **Training:** N/A — no training in this task. The trained agreement rates are inherited from [#612](https://eps.superkaiba.com/tasks/612)'s already-trained adapters.
- **Eval:** DV inputs are the inherited on-policy Haiku-judged agreement rates (60-claim held-out wrong-claim pool, 600 verdicts per cell). Predictor inputs are base Qwen-2.5-7B-Instruct residual-stream activations over the 34-persona system-prompt set, re-extracted at the early-layer band where sycophancy geometry was previously shown to live (end-of-system layer 2 cosine, primary; last-prompt layer 7, secondary; deeper layer 20, robustness). Verdict per DV: six-regression incremental-validity CV-R² ladder (bystander-grouped 5-fold CV) + marginal Spearman with 1000-bootstrap 95% CIs.

## Findings

### Who owns the absolute level (LEVEL) flips between arms: prior dominates on-policy, cosine wins on canned

The marker rule needs base prior to *dominate* the trained rate. The figure puts the two arms' LEVEL ladders side by side; the load-bearing comparison is how much the "+prior" step adds vs the "+prior+cosine" step on top.

![Side-by-side LEVEL CV-R² ladders for the canned-template arm (left) and the on-policy arm (right), six nested models each](https://raw.githubusercontent.com/superkaiba/explore-persona-space/21173e2deb04f0dd618953930bb3e685511289ec/figures/issue_649/hero1_level_ladder_canned_vs_onpolicy.png)

> **Figure.** *On canned, cosine's LEVEL uplift (+0.24) exceeds prior's (+0.19), so the rule fails; on the realistic on-policy arm prior's uplift (+0.55) dwarfs cosine's (+0.24), so it holds.* Held-out CV-R², bystander-grouped 5-fold CV; canned 108 cells, on-policy 55 cells.

- **On-policy (right):** prior adds **+0.55 CV-R²**, cosine a further **+0.24** — prior dominates, condition (a) **holds**. LEVEL here spans [0.035, 0.812] (spread 0.78) ≫ the judge's Wilson half-width 0.073, so this is full-dynamic-range, not a floor read.
- **Canned (left):** prior adds **+0.19**, cosine **+0.24** — cosine's uplift exceeds prior's, condition (a) **fails**. Robust to ladder ordering (cosine-over-intercepts +0.26 > prior +0.19) and to CV grouping (source-grouped, intercept-only agree).
- The canned LEVEL **marginals are tied** (prior ρ = +0.51 [+0.33, +0.64], cosine +0.51 [+0.34, +0.65]); the "cosine wins" read is a 0.057 ΔCV-R² gap with no CI. So this is "geometry is at least as strong as prior on LEVEL", not a decisive win.

### The "prior is null on the trained shift (CHANGE)" half replicates on canned sycophancy

The marker's signature was: base prior tracks the absolute level but says nothing about how far training moved a bystander. The two right-hand panels below test that on the canned arm.

![Canned-arm scatter: base prior vs LEVEL and CHANGE (top row), early-layer cosine vs LEVEL and CHANGE (bottom row), 108 cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/21173e2deb04f0dd618953930bb3e685511289ec/figures/issue_649/hero2_predictor_dv_scatter_quad.png)

> **Figure.** *Base prior predicts the absolute level but is null on the shift; early-layer cosine predicts both.* Each point is one of 108 source×bystander cells (canned arm). Titles carry the marginal Spearman ρ with bootstrap 95% CI. Prior↔CHANGE (top-right) is a flat cloud, CI covers 0; cosine↔CHANGE (bottom-right) rises clearly.

- Prior↔LEVEL is positive (**ρ = +0.51, CI [+0.33, +0.64]**); prior↔CHANGE is flat (**ρ = +0.14, CI [−0.07, +0.33], covers 0**). That is exactly the marker's prior-is-null-on-shift signature.
- The non-circular check (partialling cosine out, read on the trained rate directly to avoid the base-rate-in-CHANGE artifact) keeps prior load-bearing on LEVEL: **partial ρ = +0.57, CI [+0.42, +0.68]**.

### Cosine predicts CHANGE in aggregate, but the effect is not source-uniform and not early-layer-specific

The pooled cosine↔CHANGE = +0.57 hides real per-source heterogeneity.

![Canned-arm per-source small-multiple: cosine vs CHANGE for software engineer, kindergarten teacher, comedian, villain](https://raw.githubusercontent.com/superkaiba/explore-persona-space/21173e2deb04f0dd618953930bb3e685511289ec/figures/issue_649/supp_per_source_change_vs_cosine.png)

> **Figure.** *Three of four sources show a strong positive cosine→CHANGE slope (villain +0.77, comedian +0.68, kindergarten-teacher +0.63); the software-engineer source reverses sign (−0.12, p = 0.55).* Per-source Spearman on the canned arm, n ≈ 27 cells each.

- Villain, comedian, kindergarten-teacher are strongly positive (p ≤ 0.001); software-engineer reverses to **ρ = −0.12 (p = 0.55)**, with high-CHANGE cells at negative cosine. So this is a bystander-grouped *aggregate* read, not a per-source mechanism.
- Cosine predicts CHANGE at every layer (canned ρ +0.57 / +0.57 / **+0.34 [+0.15, +0.53]** at L2 / L7 / L20; on-policy L20 strongest at +0.81), so "early-layer" is descriptive, not specificity.

![Bar chart: Spearman cosine to CHANGE at three layers L2, L7, L20, canned vs on-policy arm, with 95% CIs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/21173e2deb04f0dd618953930bb3e685511289ec/figures/issue_649/supp_l20_vs_early_cosine.png)

> **Figure.** *Cosine→CHANGE holds at the early (L2), last-prompt (L7), and deeper (L20) layers on both arms — every CI excludes 0.* L20's canned ρ (+0.34) is below L2's (+0.57) but positive; on-policy L20 is strongest (+0.81). Bystander-grouped, 1000-bootstrap CIs.

The canned LEVEL break has a competing read: the canned positives are fixed templates that over-install (+0.84–0.93 vs on-policy +0.60–0.66, [#612](https://eps.superkaiba.com/tasks/612)), so the source→bystander LEVEL variance cosine picks up there may be a template-installation artifact, not behavior-general "geometry carries level" — consistent with prior winning LEVEL on the on-policy arm, where the artifact is absent.

### The on-policy CHANGE read is below the precision floor — a null, not a signature

The on-policy arm's CHANGE column appears to show prior↔CHANGE excluding zero (the "implant rides the pre-existing propensity" outcome), but it is not interpretable. This floor caps CHANGE only — the on-policy LEVEL read in the first finding is unaffected.

- On-policy prior↔CHANGE **ρ = +0.58 [+0.34, +0.73]**, but median |CHANGE| = 0.013 with S/N = 0.18 sits an order of magnitude below the judge's resolution; per-seed sign agreement drops to 0.84 (vs 0.97 canned). A resolution artifact.
- The floor caps the *shift* (a small difference of two noisy rates). The *level* (trained rate, measured directly) spreads 0.78 over [0.035, 0.812], far above the same resolution — which is why the LEVEL ladder is a legitimate read while CHANGE is not.
- Collinearity is not the cause: Pearson(|cosine|, prior) is −0.03 (canned) / +0.05 (on-policy), both far below the 0.6 gate.

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

Full per-cell table (both arms, all predictors): [`eval_results/issue_649/per_cell_table.csv` @ 21173e2](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/eval_results/issue_649/per_cell_table.csv). The complete base + trained judgment files (each with raw verdicts): [HF data repo @ 14d541b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/14d541bbafb3dfdbe35ea7a3389df5e5a7f2c458/issue612_sycophancy_onpolicy).

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
  | Precision gate (#391) | FIRED both arms (S/N canned 0.89, on-policy 0.18) — caps CHANGE only, not LEVEL |
  | Cells (after exclusions) | canned 108, on-policy 55 |
  | Seeds | 42, 137 (averaged per cell) |
  | Config slugs | `arm_canned`, `arm_onpolicy`; ladder models `M0`–`M5` |

- **Artifacts:**
  - CV-R² ladder: [`eval_results/issue_649/cv_r2_ladder.json` @ 21173e2](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/eval_results/issue_649/cv_r2_ladder.json)
  - Marginal Spearman + non-circular partials: [`marginal_spearman.json` @ 21173e2](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/eval_results/issue_649/marginal_spearman.json)
  - Precision + collinearity gates: [`precision_check.json`](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/eval_results/issue_649/precision_check.json), [`collinearity_gate.json`](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/eval_results/issue_649/collinearity_gate.json)
  - Per-cell table: [`per_cell_table.csv` @ 21173e2](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/eval_results/issue_649/per_cell_table.csv)
  - Figures: [`figures/issue_649/` @ 21173e2](https://github.com/superkaiba/explore-persona-space/tree/21173e2deb04f0dd618953930bb3e685511289ec/figures/issue_649)
- **Compute:** 0 GPU-hours (CPU re-analysis on the VM; geometry re-extraction ran CPU-only over 34 short prompts). No pod provisioned.
- **Code:** extractor [`scripts/issue649_extract_panel_earlylayer.py`](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/scripts/issue649_extract_panel_earlylayer.py); analysis [`scripts/issue649_level_change_decomp.py`](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/scripts/issue649_level_change_decomp.py); hero re-plot [`scripts/issue649_hero_replot.py`](https://github.com/superkaiba/explore-persona-space/blob/21173e2deb04f0dd618953930bb3e685511289ec/scripts/issue649_hero_replot.py); git commit `904c2668` (analysis run), `21173e2d` (round-2 figures). Reproduce: `uv run python scripts/issue649_level_change_decomp.py` then `uv run python scripts/issue649_hero_replot.py`.
- **Reused artifacts:**
  - Reused sycophancy panel + judgments from [#612](https://eps.superkaiba.com/tasks/612): [HF data repo @ 14d541b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/14d541bbafb3dfdbe35ea7a3389df5e5a7f2c458/issue612_sycophancy_onpolicy) — fit: same base model (Qwen-2.5-7B-Instruct), on-policy judge-scored rates over the audited held-out claim pool, all 4 sources × 30 bystanders + base rates present; the realism regime is exactly the one this decomposition reads off.
  - Reused base agreement rates from [#612](https://eps.superkaiba.com/tasks/612): `eval_results/issue_612/base/judgments/` (52 personas) — fit: the bystander base prior `b`, measured pre-training on the same probe pool.
- **Context:**
  - Created / run: created 2026-06-15; analysis + figures landed 2026-06-16.
  - Follow-up to: [#532](https://eps.superkaiba.com/tasks/532) — the marker LEVEL/CHANGE decomposition this task ports to a realistic behavior.
  - Originating prompt(s), verbatim:
    > Can we run a followup to check effect of base prior on change in leakage for the more realistic behaviors?
