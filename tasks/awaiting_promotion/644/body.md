---
title: No portable convex geometry→behavior shape recurs across the qualifying prior-task
  scatters (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-15T18:02:38Z'
has_clean_result: true
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
# No portable convex geometry→behavior shape recurs across the qualifying prior-task scatters (MODERATE confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_644.md](https://github.com/superkaiba/explore-persona-space/blob/b452fc66af23c9df5a6b887cdbde69532f175b67/docs/methodology/issue_644.md) · [gist](https://gist.github.com/superkaiba/9c91df722fa89083a39c70b922cea975)

## Takeaways

- After the full artifact-control stack, **0 of 22** qualifying geometry-frame fits show robust convexity (recurs threshold 11) — no consistent convex shape across the three behaviors with commensurable geometry scalars.
- The seed sycophancy "hockey-stick" (n=35) has positive x² (**+0.22**) but the bootstrap interval crosses zero and ΔAIC is **0.08**; the look is linear-plus-leverage.
- Shape disagrees by behavior: marker-leakage source frames skew **concave** (12 of 16 negative-curvature), fact-leakage skews positive-but-noisy — no consistent sign.
- Scope: refusal is excluded (its source eval directory has no commensurable per-persona geometry scalar); held-out sycophancy (#411) was not analyzed (no clean per-source cosine × held-out-rate pairing was assembled from the HF artifacts — the plan's Medium-confidence drop-to-robustness fallback fired); six fact on-policy frames are reported but excluded for n < 10.
- Binding constraint: per-behavior n is small (sycophancy 35, marker 26/17, fact 14), so this is an under-powered null for the convex-recurs hypothesis. Geometry still predicts behavior monotonically (sycophancy rank ρ = 0.73).

## What I ran

- **Why:** [#623](https://eps.superkaiba.com/tasks/623) found a per-persona cosine→sycophancy scatter that looks roughly exponential by eye, and the impression recurs across the leakage line. The two spiritual-sibling papers — Persona Vectors (arXiv 2507.21509) and Persona Features Control Emergent Misalignment (arXiv 2506.19823) — establish the cosine-to-direction predictor but report only linear/monotone associations; neither tests functional form, and rank correlations are blind to it by construction.
- **Design:** zero-GPU meta-analysis. Inventory every past task with paired (per-unit geometry scalar, per-unit behavior-strength scalar) data, pull the RAW (non-rank) values, and fit candidate forms per behavior × measurement frame. The single thing under test is the functional form, not whether geometry predicts behavior at all.
- **Training:** n/a — no training in this task (zero-GPU meta-analysis of prior per-cell paired-data files).
- **Eval:** per behavior × frame, the winning form by leave-one-out R² + AIC, a signed x² curvature term with a 10,000-draw bootstrap CI, and a four-control survival screen — geometry-frame partition, top-1 and top-2 leverage leave-one-out, log-space double-fit, and bounded-rate logit double-fit. A behavior qualifies for the recurs denominator only with two-axis spread AND n ≥ 10 in the geometry frame.

## Findings

### No behavior shows robust convexity: 0 of 22 qualifying geometry-frame fits survive the control stack (threshold 11)

A geometry-frame fit counts toward "convex recurs" only if a convex form beats linear by ΔAIC ≥ 2 with a same-signed curvature interval excluding zero, AND survives the leverage, log-space, and rate-stabilization controls. The denominator is the 22 fits with two-axis spread and n ≥ 10. Refusal is absent from this denominator: its source eval directory carries pass-rates but no commensurable per-persona geometry scalar, so a refusal row would be a fabricated axis.

![Cross-behavior geometry-frame convexity recurs table: 0 of 22 qualifying frames are robustly convex, with per-row n, convex verdict, curvature sign, delta-AIC, leverage-survival, rate-artifact, and counts-as-robust-convex columns across sycophancy, marker leakage, and fact leakage; six n-under-10 fact frames are marked excluded](https://raw.githubusercontent.com/superkaiba/explore-persona-space/462fd308ffc00aa16ff74575d41f8c89d728a7e4/figures/issue_644/convexity_table.png)

> **Figure.** *Across all 22 qualifying geometry-frame fits, exactly zero contribute a robust convex verdict (threshold is 11).* Each row is one behavior × frame. The "Counts as robust convex?" column reads "n" everywhere. Rows tagged "n < 10, excluded" are the six fact on-policy frames; deprecated-scalar rows are dropped before this table.

- The threshold of 11 is not approached — numerator 0.
- Six raw fits flagged convex pre-control, but every curvature interval spans zero; the only two surviving a leverage drop are both at n=6 (under-powered, outside the denominator).

### The seed sycophancy "hockey-stick" is linear-plus-leverage, not robust convexity (n=35)

The observation that motivated the task — per-persona cosine (persona vector → sycophancy direction) vs judged base sycophancy rate, with linear and best-form fits overlaid plus the logit-stabilized panel.

![Two-panel sycophancy seed scatter: left panel raw cosine vs base sycophancy rate with near-coincident linear and exponential fits and an annotation box reporting x-squared coefficient +0.22, bootstrap interval crossing zero, delta-AIC 0.08, and failure to survive the top-1 leverage drop; right panel the same scatter in logit-stabilized y with a clean linear fit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e74b9be5d6c1278656940a9a07f2f2273006eb9b/figures/issue_644/seed_sycophancy_scatter.png)

> **Figure.** *The exponential best-form fit (dashed) is nearly coincident with the linear fit (solid); the upward look comes from a few high-cosine personas.* Left: raw scatter, n=35. Right: logit-stabilized y, bend gone. x² coef +0.22, CI [−0.41, +1.38], ΔAIC 0.08.

- The positive curvature estimate (+0.22) reproduces the visual impression, but its bootstrap interval includes zero and ΔAIC favours linear.
- It does not survive dropping the single highest-leverage persona; the logit panel is cleanly monotone-linear — the bend was bounded-rate floor compression.

### Shape is behavior-specific, not portable: marker leakage skews concave while fact leakage skews positive-but-noisy

If a convex form were portable, the winning forms and curvature signs should agree across behaviors. They do not. The overlay pairs each behavior × frame's raw-y scatter with its logit-stabilized (bounded-rate control) counterpart.

![Grid of raw-vs-logit-stabilized scatter overlays, one cell per behavior and measurement frame, titled in plain English (e.g. "Marker leakage — cosine to source (A1)"), showing heterogeneous shapes across behaviors and that apparent upward bends in raw-rate panels flatten under logit stabilization](https://raw.githubusercontent.com/superkaiba/explore-persona-space/462fd308ffc00aa16ff74575d41f8c89d728a7e4/figures/issue_644/raw_vs_logit_overlay.png)

> **Figure.** *Raw-rate panels (blue circles) overlaid with their logit-stabilized counterparts (salmon triangles) — the rate-compression control.* Marker-leakage source frames mostly fit concave (12 of 16 negative-curvature); fact-leakage frames skew positive with intervals spanning zero. No raw upward bend in a geometry frame survives logit stabilization.

- Signs disagree by behavior: marker-leakage source frames predominantly concave, fact-leakage predominantly positive-sign — no consistent direction.
- Every apparent raw-rate curve in a geometry frame flattens under logit stabilization — the bounded-rate floor manufactures a small upward bend that is not a property of the coupling. (One base-prior log-prob row, a non-geometry sensitivity frame, keeps a convex verdict; it is kept out of the headline numerator.)

## Data

### Trained on

n/a — no training in this task. This is a zero-GPU re-analysis of paired (geometry, behavior-strength) scatters already produced by prior runs; no LoRA adapter, no training mix, no pod.

### Evaluated with

The "probes" here are the per-unit raw (x, y) scatters re-loaded from prior eval JSONs, three behaviors with commensurable geometry scalars (cosine / JS only; a base-prior log-prob frame is kept in a separate prior-frame sensitivity table and never folded into the geometry headline):

- **Sycophancy seed (n=35):** per-persona cosine to the sycophancy direction vs judged base sycophancy rate. The sycophancy-direction scatter is snapshotted into this task's commit; read directly, no preprocessing. Pinned snapshot (3 of 3 inputs, complete): [`eval_results/issue_644/inputs/issue623/`](https://github.com/superkaiba/explore-persona-space/tree/462fd308ffc00aa16ff74575d41f8c89d728a7e4/eval_results/issue_644/inputs/issue623). Upstream sources: [`cosine_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_623/cosine_matrix.json) and [`syc_i.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_623/syc_i.json) (its base sycophancy judgments reuse the base pass produced by the on-policy sycophancy task).
- **Marker leakage, centered (n=17):** centered cosine to the joint-arm centroid vs on-policy marker emit, read directly from [`cosine_l20_base.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_311/cosine_l20_base.json) + [`analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_311/analysis.json) (the 17 bystanders are the 19-persona cosine bank minus the 2 sources).
- **Marker leakage, raw (n=26):** per-source raw cosine-to-source vs on-policy marker emit, 16 anonymized per-source frames, read from [`predictors.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_532/predictors.json) plus the per-cell trained-slot files under `logp_slot_followup/per_cell_trained/`; the deprecated single-next-token JS rows are emitted as labeled sensitivity frames and dropped from the headline.
- **Fact leakage (n=14, qualifying):** per-arm cosine-to-source vs taught-fact leak rate, three source arms (marine biologist, local resident, courthouse historian), read from [`predictors.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_500/predictors.json) (single chosen contrastive recipe; recipes never pooled).
- **Fact leakage (n=6, excluded for n < 10):** six on-policy cosine/JS frames vs leak rate, read from [`bystander_logprob/correlations.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_444/bystander_logprob/correlations.json) (single chosen recipe + two labeled sensitivity recipes, never pooled).
- **Refusal:** excluded — its source eval directory ([`aggregate_long.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_390/aggregate_long.json)) has per-(persona × framing) pass-rates but no commensurable per-persona geometry scalar.
- **Held-out sycophancy (#411): NOT EVALUATED — excluded, no commensurable scalar assemblable.** Plan v2 named #411's cosine-gradient sycophancy run as a planned geometry-frame source ([`issue411_sycophancy_cosine_gradient/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a052b1dcb61a96ed9081561804c25943b01ec73e/issue411_sycophancy_cosine_gradient) on the HF data repo). It did not enter the 22-fit denominator: no clean per-source pairing of (cosine scalar, held-out sycophancy rate) could be assembled from those HF artifacts, so the plan's Medium-confidence drop-to-robustness fallback (§12) fired and the source was dropped rather than fabricated.

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
| Leverage screen | top-1 and top-2 leave-one-out re-fit |
| Logit stabilization | ε = 0.005 (bounded-rate double-fit) |
| Convex-wins threshold | ΔAIC ≥ 2.0 over linear |
| Recurs denominator gate | two-axis spread AND n ≥ 10 in geometry frame |
| Qualifying denominator | 22 geometry-frame fits (recurs threshold 11) |
| Convex numerator | 0 |

**Artifacts:**

- Per-(behavior × frame) fits (50 records): [`eval_results/issue_644/per_behavior_fits.json`](https://github.com/superkaiba/explore-persona-space/blob/462fd308ffc00aa16ff74575d41f8c89d728a7e4/eval_results/issue_644/per_behavior_fits.json)
- Cross-behavior recurs table (machine-readable headline): [`figures/issue_644/convexity_table.json`](https://github.com/superkaiba/explore-persona-space/blob/462fd308ffc00aa16ff74575d41f8c89d728a7e4/figures/issue_644/convexity_table.json)
- Pinned #623 source snapshot: [`eval_results/issue_644/inputs/issue623/`](https://github.com/superkaiba/explore-persona-space/tree/462fd308ffc00aa16ff74575d41f8c89d728a7e4/eval_results/issue_644/inputs/issue623)
- Figures (hero recurs table, raw-vs-logit overlay, per-behavior small-multiples): [`figures/issue_644/`](https://github.com/superkaiba/explore-persona-space/tree/462fd308ffc00aa16ff74575d41f8c89d728a7e4/figures/issue_644)
- Seed scatter figure (used inline): [`figures/issue_644/seed_sycophancy_scatter.png`](https://github.com/superkaiba/explore-persona-space/blob/e74b9be5d6c1278656940a9a07f2f2273006eb9b/figures/issue_644/seed_sycophancy_scatter.png)
- WandB run: n/a (no training, no logging)

Reuse provenance (every per-cell input is paired data produced by a prior task, read raw):

- Reused sycophancy paired data from [#623](https://eps.superkaiba.com/tasks/623): [`eval_results/issue_644/inputs/issue623/`](https://github.com/superkaiba/explore-persona-space/tree/462fd308ffc00aa16ff74575d41f8c89d728a7e4/eval_results/issue_644/inputs/issue623) — fit: per-persona cosine + base sycophancy rate, two-axis spread, n=35 ≥ 10, the seed frame the task was filed on.
- Reused sycophancy base judgments from [#612](https://eps.superkaiba.com/tasks/612): [`eval_results/issue_612/base/`](https://github.com/superkaiba/explore-persona-space/tree/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_612/base) — fit: the base sycophancy rate inside #623's `syc_i.json` is this base pass; upstream provenance, not a separate scatter.
- Reused marker-leakage paired data from [#311](https://eps.superkaiba.com/tasks/311): [`eval_results/issue_311/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_311/analysis.json) — fit: centered cosine + on-policy marker emit, n=17 ≥ 10, the centered marker frame.
- Reused marker-leakage paired data from [#532](https://eps.superkaiba.com/tasks/532): [`eval_results/issue_532/predictors.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_532/predictors.json) — fit: per-source raw cosine-to-source + on-policy marker emit, 16 source frames at n=26 ≥ 10.
- Reused fact-leakage paired data from [#500](https://eps.superkaiba.com/tasks/500): [`eval_results/issue_500/predictors.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_500/predictors.json) — fit: per-arm cosine-to-source + taught-fact leak rate, n=14 ≥ 10, the three qualifying fact arms.
- Reused fact-leakage paired data from [#444](https://eps.superkaiba.com/tasks/444): [`eval_results/issue_444/bystander_logprob/correlations.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_444/bystander_logprob/correlations.json) — fit: on-policy cosine/JS + leak rate, n=6, reported but excluded from the denominator for n < 10.
- Reused refusal data from [#390](https://eps.superkaiba.com/tasks/390): [`eval_results/issue_390/aggregate_long.json`](https://github.com/superkaiba/explore-persona-space/blob/02a130d78c3f324421457dcb4ca618deee12a9a1/eval_results/issue_390/aggregate_long.json) — fit: NOT FIT — no commensurable per-persona geometry scalar (excluded from the denominator, routed to a new-generation follow-up).
- Reused: producing issue [#411](https://eps.superkaiba.com/tasks/411) — pinned path [`issue411_sycophancy_cosine_gradient/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a052b1dcb61a96ed9081561804c25943b01ec73e/issue411_sycophancy_cosine_gradient) (530 files, verified present via the HF Hub API) — fit: NOT FIT — no clean per-source (cosine, held-out sycophancy rate) pairing assemblable from the HF artifacts; plan v2 §12 Medium-confidence fallback fired (dropped to a robustness note, not analyzed).

- **Methodology reference:** [docs/methodology/issue_644.md](https://github.com/superkaiba/explore-persona-space/blob/b452fc66af23c9df5a6b887cdbde69532f175b67/docs/methodology/issue_644.md) · [gist](https://gist.github.com/superkaiba/9c91df722fa89083a39c70b922cea975)

**Compute:**

- Wall time: minutes on the VM (CPU-only fits + bootstrap + figures); no GPU, no pod.
- GPU: none.
- Pod: none (ran on the dev VM against committed eval JSONs).

**Code:**

- Fit/test machinery: [`src/explore_persona_space/analysis/convexity_meta.py`](https://github.com/superkaiba/explore-persona-space/blob/462fd308ffc00aa16ff74575d41f8c89d728a7e4/src/explore_persona_space/analysis/convexity_meta.py)
- Per-behavior loaders: [`scripts/issue644_loaders.py`](https://github.com/superkaiba/explore-persona-space/blob/462fd308ffc00aa16ff74575d41f8c89d728a7e4/scripts/issue644_loaders.py)
- Driver: [`scripts/issue644_functional_form.py`](https://github.com/superkaiba/explore-persona-space/blob/462fd308ffc00aa16ff74575d41f8c89d728a7e4/scripts/issue644_functional_form.py)
- Seed-scatter figure script: [`scripts/issue644_seed_scatter_figure.py`](https://github.com/superkaiba/explore-persona-space/blob/e74b9be5d6c1278656940a9a07f2f2273006eb9b/scripts/issue644_seed_scatter_figure.py)
- Git commit (artifacts + figures): `462fd308ffc00aa16ff74575d41f8c89d728a7e4` (seed figure at `e74b9be5d6c1278656940a9a07f2f2273006eb9b`; branch `issue-644`, PR #474)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 462fd308ffc00aa16ff74575d41f8c89d728a7e4
    uv sync
    uv run python scripts/issue644_functional_form.py --phase load-data
    uv run python scripts/issue644_functional_form.py --phase fit
    uv run python scripts/issue644_functional_form.py --phase aggregate
    ```

**Context:**

- Created 2026-06-15; analysis ran the same day on the VM.
- Follow-up to: fresh direction (no parent); relates to the `leak-predictor` open question — this run sharpens its standing "geometry predicts behavior inconsistently across behaviors" belief from "monotonic-but-inconsistent" to "no recurring convex shape at these n".
- Originating prompt, verbatim:

  > In 623 we see kind of an exponential relationship between cosine similarity and sycophancy. I feel like we've seen this trend with a lot of the behaviors. Can you make and dispatch an issue in the background to look at all past results and explore this hypothesis?
