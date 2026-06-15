---
title: The geometry→behavior-strength relationship is consistent with linear after
  artifact controls — no behavior in the sample shows robust convexity (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-15T18:02:38Z'
has_clean_result: false
origin_prompt: In 623 we see kind of an exponential relationship between cosine similarity
  and sycophancy. I feel like we've seen this trend with a lot of the behaviors. Can
  you make and dispatch an issue in the background to look at all past results and
  explore this hypothesis?
goal: Across all past experiments that produced paired (persona-geometry scalar, behavior-strength
  scalar) data, characterize the FUNCTIONAL FORM of the geometry→behavior-strength
  relationship — testing on RAW (non-rank) values whether it is consistently convex/super-linear
  ('exponential-looking') rather than linear, and whether that shape recurs across
  behaviors (sycophancy, marker leakage, fact leakage, refusal, EM) after controlling
  for the fact-line sign flip, saturation/floor, high-leverage points, log-space artifacts,
  and the X vs (X−Y) caveat.
relates_to:
- leak-predictor
---
# The geometry→behavior-strength relationship is consistent with linear after artifact controls — no behavior in the sample shows robust convexity (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- After the full artifact-control stack, **0 of 22** qualifying geometry-frame fits show robust convexity (recurs threshold 11) — verdict: no consistent convex shape across four behaviors.
- The seed sycophancy "hockey-stick" (n=35) has positive x² (**+0.22**) but its **CI [−0.41, +1.38] includes 0** and ΔAIC is **0.08**; the look is linear-plus-leverage.
- Shape disagrees by behavior: marker-leakage fits skew **concave** (12 of 16 source frames negative-curvature), fact-leakage skews positive-but-noisy — no consistent sign.
- Binding constraint: per-behavior n is small (sycophancy 35, marker 26/17, fact 14), so this is an honest under-powered null for H1, not a precise refutation.
- Geometry still predicts behavior *monotonically* where measured (sycophancy rank ρ = 0.73, CI [0.49, 0.87]); this rules out only a recurring convex *shape*.

## What I ran

- **Why:** [#623](https://eps.superkaiba.com/tasks/623) found a per-persona cosine→sycophancy scatter that looks roughly exponential by eye, and the same "exponential-looking" impression recurs across the leakage line. Rank correlations (Spearman) are blind to functional form by construction, so no prior task had tested whether the geometry→behavior relationship is genuinely convex (accelerating) rather than linear, and whether that shape is portable across behaviors.
- **Design:** zero-GPU meta-analysis. Inventory every past task with paired (per-unit geometry scalar, per-unit behavior-strength scalar) data, pull the RAW (non-rank) values, and fit candidate forms (linear / quadratic / exponential / power / monotone spline) per behavior × measurement frame. The single thing under test is the functional form, not whether geometry predicts behavior at all.
- **Eval:** per behavior × frame, the winning form by leave-one-out R² + AIC, a signed x² curvature term with a 10,000-draw bootstrap CI, and a four-control survival screen — geometry-frame partition, top-1 and top-2 Cook's-D leave-one-out, log-space double-fit, and bounded-rate logit double-fit. A behavior qualifies for the recurs denominator only with two-axis spread AND n ≥ 10 in the geometry frame.
- **Scope:** refusal is excluded from the denominator (no commensurable per-persona geometry scalar survives in its source eval directory) and routed to a new-generation follow-up — not a fabricated scalar. Six fact-leakage on-policy frames at n=6 are reported but excluded from the denominator for n < 10.

## Findings

### No behavior shows robust convexity: 0 of 22 qualifying geometry-frame fits survive the control stack (threshold 11)

A geometry-frame fit counts toward "convex recurs" only if a convex form beats linear by ΔAIC ≥ 2 with a same-signed curvature CI excluding zero, AND survives the leverage, log-space, and rate-stabilization controls. The denominator is the 22 fits with two-axis spread and n ≥ 10.

![Cross-behavior geometry-frame convexity recurs table: verdict H0/H2 no-convex, H1 numerator 0 of denominator 22, with per-row convex / sign / delta-AIC / leverage-robust / rate-artifact / counts-toward-H1 columns across sycophancy, marker leakage, and fact leakage frames](https://raw.githubusercontent.com/superkaiba/explore-persona-space/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/figures/issue_644/convexity_table.png)

> **Figure.** *Across all 22 qualifying geometry-frame fits, exactly zero contribute a robust convex verdict (threshold is 11).* Each row is one behavior × frame; the H1 column reads "n" everywhere. Excludes 6 fact frames at n < 10 and 16 deprecated-scalar rows.

- Verdict `H0/H2 (no convex)`: the threshold of 11 is not approached — numerator 0.
- Six raw fits flagged convex pre-control, but every curvature CI spans zero; the only two surviving a Cook's-D drop are both at n=6 (under-powered, outside the denominator).

### The #623 seed sycophancy "hockey-stick" is linear-plus-leverage, not robust convexity (n=35)

The observation that motivated the task — per-persona cosine (persona vector → sycophancy direction) vs judged base sycophancy rate, with linear and best-form fits overlaid plus the logit-stabilized panel.

![Two-panel sycophancy seed scatter: left panel raw cosine vs base sycophancy rate with near-coincident linear and exponential fits and an annotation box (x-squared coef +0.22, bootstrap CI -0.41 to 1.38, delta-AIC 0.08, does not survive top-1 Cook's-D drop); right panel the same scatter in logit-stabilized y with a clean linear fit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e74b9be5d6c1278656940a9a07f2f2273006eb9b/figures/issue_644/seed_sycophancy_scatter.png)

> **Figure.** *The exponential best-form fit (dashed) is nearly coincident with the linear fit (solid); the upward look comes from a few high-cosine personas.* Left: raw scatter, n=35. Right: logit-stabilized y, bend gone. x² coef +0.22, CI [−0.41, +1.38], ΔAIC 0.08.

- The positive curvature estimate (+0.22) reproduces the visual impression, but its CI [−0.41, +1.38] includes zero and ΔAIC favours linear.
- It does not survive dropping the single highest-Cook's-D persona; the logit panel is cleanly monotone-linear — the bend was bounded-rate floor compression.

### Shape is behavior-specific, not portable: marker leakage skews concave while fact leakage skews positive-but-noisy

If a convex form were portable, the winning forms and curvature signs should agree across behaviors. They do not — the H0 signature. The overlay pairs each behavior × frame's raw-y scatter with its logit-stabilized (bounded-rate control) counterpart.

![Grid of raw-vs-logit-stabilized scatter overlays, one cell per behavior and measurement frame, showing heterogeneous shapes across behaviors and that apparent upward bends in raw-rate panels flatten under logit stabilization](https://raw.githubusercontent.com/superkaiba/explore-persona-space/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/figures/issue_644/raw_vs_logit_overlay.png)

> **Figure.** *Raw-rate panels (one per behavior × frame) overlaid with their logit-stabilized counterparts — the rate-compression control.* Marker-leakage source frames mostly fit concave (12 of 16 negative-curvature); fact-leakage frames skew positive with CIs spanning zero. No raw upward bend survives logit stabilization.

- Signs disagree by behavior: marker-leakage source frames predominantly concave, fact-leakage predominantly positive-sign — no consistent direction to call "the shape."
- Every apparent raw-rate curve flattens under logit stabilization, confirming the bounded-rate floor manufactures a small upward bend that is not a property of the coupling.

## Data

### Trained on

n/a — no training in this task. This is a zero-GPU re-analysis of paired (geometry, behavior-strength) scatters already produced by prior runs; no LoRA adapter, no training mix, no pod.

### Evaluated with

The "probes" here are the per-unit raw (x, y) scatters re-loaded from prior eval JSONs: sycophancy (per-persona cosine to the sycophancy direction vs judged base sycophancy rate, n=35), marker leakage (per-source cosine-to-source vs on-policy marker emit, n=26 raw + n=17 centered), and fact leakage (per-arm cosine-to-source vs taught-fact leak rate, n=14; six on-policy frames at n=6). Geometry scalars are cosine/JS only; a base-prior log-prob frame is kept in a separate prior-frame sensitivity table and never folded into the geometry headline. The #623 source data is pinned into this task's commit for reproducibility.

The pinned #623 source snapshot (3 of 3 inputs, complete): [`eval_results/issue_644/inputs/issue623/`](https://github.com/superkaiba/explore-persona-space/tree/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/eval_results/issue_644/inputs/issue623)

Source tasks the paired scatters were drawn from (all rows, complete inventory):

- Sycophancy seed: [#623](https://eps.superkaiba.com/tasks/623) (cosine vs base sycophancy rate; base pass reused from [#612](https://eps.superkaiba.com/tasks/612))
- Marker leakage: [#311](https://eps.superkaiba.com/tasks/311) (centered cosine) and [#532](https://eps.superkaiba.com/tasks/532) (raw cosine-to-source, on-policy emit)
- Fact leakage: [#444](https://eps.superkaiba.com/tasks/444) and [#500](https://eps.superkaiba.com/tasks/500) (single chosen contrastive recipe; recipes never pooled)
- Refusal: [#390](https://eps.superkaiba.com/tasks/390) — excluded (no commensurable per-persona geometry scalar in its eval directory)

### Generated

n/a — no model generations in this task. The behavior-strength values are judged rates / log-probs already computed by the source tasks; this run consumes them as raw numbers and emits fits, not completions.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Task kind | `kind: experiment`, zero-GPU meta-analysis (no training, no pod, no WandB run) |
| Candidate forms | linear, quadratic, exponential, power-law, monotone PCHIP spline (4 knots) |
| Model selection | leave-one-out R² + AIC/BIC bake-off |
| Convexity test | signed x² coefficient + nonparametric bootstrap CI |
| Bootstrap | B = 10000, seed 42 |
| Leverage screen | top-1 and top-2 Cook's-D leave-one-out re-fit |
| Logit stabilization | ε = 0.005 (bounded-rate double-fit) |
| Convex-wins threshold | ΔAIC ≥ 2.0 over linear |
| Recurs denominator gate | two-axis spread AND n ≥ 10 in geometry frame |
| Qualifying denominator | 22 geometry-frame fits (recurs threshold 11) |
| H1 numerator | 0 |

**Artifacts:**

- Per-(behavior × frame) fits (50 records): [`eval_results/issue_644/per_behavior_fits.json`](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/eval_results/issue_644/per_behavior_fits.json)
- Cross-behavior recurs table (machine-readable headline): [`figures/issue_644/convexity_table.json`](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/figures/issue_644/convexity_table.json)
- Pinned #623 source snapshot: [`eval_results/issue_644/inputs/issue623/`](https://github.com/superkaiba/explore-persona-space/tree/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/eval_results/issue_644/inputs/issue623)
- Figures (hero recurs table, raw-vs-logit overlay, per-behavior small-multiples): [`figures/issue_644/`](https://github.com/superkaiba/explore-persona-space/tree/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/figures/issue_644)
- Seed scatter figure (used inline): [`figures/issue_644/seed_sycophancy_scatter.png`](https://github.com/superkaiba/explore-persona-space/blob/e74b9be5d6c1278656940a9a07f2f2273006eb9b/figures/issue_644/seed_sycophancy_scatter.png)
- WandB run: n/a (no training, no logging)

**Compute:**

- Wall time: minutes on the VM (CPU-only fits + bootstrap + figures); no GPU, no pod.
- GPU: none.
- Pod: none (ran on the dev VM against committed eval JSONs).

**Code:**

- Fit/test machinery: [`src/explore_persona_space/analysis/convexity_meta.py`](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/src/explore_persona_space/analysis/convexity_meta.py)
- Per-behavior loaders: [`scripts/issue644_loaders.py`](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/scripts/issue644_loaders.py)
- Driver: [`scripts/issue644_functional_form.py`](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/scripts/issue644_functional_form.py)
- Seed-scatter figure script: [`scripts/issue644_seed_scatter_figure.py`](https://github.com/superkaiba/explore-persona-space/blob/e74b9be5d6c1278656940a9a07f2f2273006eb9b/scripts/issue644_seed_scatter_figure.py)
- Git commit (artifacts + figures): `369ca8912ddff5fef9d16e8dffc6cfaf31b87544` (seed figure at `e74b9be5d6c1278656940a9a07f2f2273006eb9b`; branch `issue-644`, PR #474)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 369ca8912ddff5fef9d16e8dffc6cfaf31b87544
    uv sync
    uv run python scripts/issue644_functional_form.py load-data
    uv run python scripts/issue644_functional_form.py fit
    uv run python scripts/issue644_functional_form.py aggregate
    ```

**Context:**

- Created 2026-06-15; analysis ran the same day on the VM.
- Fresh direction (no parent); relates to the `leak-predictor` open question — this run sharpens its standing "geometry predicts behavior inconsistently across behaviors" belief from "monotonic-but-inconsistent" to "no recurring convex shape at these n".
- Originating prompt, verbatim:

  > In 623 we see kind of an exponential relationship between cosine similarity and sycophancy. I feel like we've seen this trend with a lot of the behaviors. Can you make and dispatch an issue in the background to look at all past results and explore this hypothesis?
