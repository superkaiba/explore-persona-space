---
title: 'Apply #502''s 28-layer Gaussian-KL bake-off predictor to fact-leakage and
  sycophancy transfer targets (reuse existing adapters)'
kind: analysis
tags: []
created_at: '2026-06-07T01:42:12Z'
has_clean_result: false
parent_id: 502
---
---
kind: analysis
parent_id: 502
relates_to:
- leak-predictor
- fact-teach-persona-transfer
- leak-behavior-vs-marker
goal: 'Test whether #502''s winning residual-stream predictor recipe — the full 28-layer
  x 9-cloud-metric x 3-extraction-point bake-off whose marker-leakage winner was last-prompt
  x L22 x Gaussian-KL on an L19-L24 ridge — predicts cross-persona transfer of fact
  leakage (reusing the #381/#389/#390/#444 adapters + #494 leakage matrices) and sycophancy
  leakage (reusing the #411 adapters + #470 leakage matrices), with no new training,
  to determine whether the residual-stream geometry that tracks marker-leakage transfer
  is behavior-agnostic or marker-specific.'
---
# Apply #502's 28-layer Gaussian-KL bake-off predictor to fact-leakage and sycophancy transfer targets (reuse existing adapters)

## Goal

Test whether #502's winning residual-stream predictor recipe — the full 28-layer x 9-cloud-metric x 3-extraction-point bake-off whose marker-leakage winner was `last_prompt x L22 x gauss_kl` on an L19-L24 ridge (ρ = −0.79 against ΔG on the cleanest checkpoint) — predicts cross-persona transfer of **fact leakage** and **sycophancy leakage**, reusing the already-trained adapters and stored leakage targets. No new training in the default plan. The question is whether the residual-stream geometry that tracks marker-leakage transfer is **behavior-agnostic** (a real persona-distance axis) or **marker-specific**.

## Motivation

The predictor line (base-model persona-distance → cross-persona behavior transfer) has been pointed at three dependent variables:

- **Markers** — #207, #469, #474, #493, and now #502, which is the strongest result: a full 28-layer × 9-metric × 3-extraction sweep found that an upper-stack Gaussian-KL ridge (L19-L24, peak L22) at the last prompt token predicts marker-transfer at ρ = −0.79, beating the next-token output-distribution baseline by a real margin.
- **Fact leakage** — #494 pointed persona-distance at the #381/#389/#390/#444 fact-teaching adapters and got **null** across all four distance flavors; #500 found the bystander's own prior on the fact, not source proximity, is the predictor.
- **Sycophancy** — #470 pointed cosine and JS/KL at #411's per-bystander sycophancy leakage and got **null**; #480 found a token marker doesn't cleanly predict sycophancy leakage either.

The critical gap: **#494 and #470 both predate the #502 finding and used only the coarse predictors** — a single-layer (~L20) residual cosine plus output-distribution JS/KL — exactly the family #502 showed gets beaten by the full 28-layer cloud-metric sweep on markers. #494's own "how this updates me" says the cheap pooled distance test came back null and the next move is a richer panel; it never ran the per-layer × metric × extraction bake-off. So the null on fact/sycophancy might be a resolution artifact, the same way the marker signal was invisible at #493's 8-layer resolution until #502 filled in the grid.

This task closes that gap: re-point the **#502 bake-off recipe** (not the coarse predictors) at the fact and sycophancy transfer targets, reusing the existing trained adapters. It directly tests the universality question #502 itself raised — #502 found the ΔG and g_logprob targets pick *different* winning cells, i.e. there is no single universal "persona-distance" cell even within markers, so generalization to a different behavior is a genuinely open and high-information question.

## Hypothesis

The L19-L24 last-prompt Gaussian-KL / MMD ridge that predicts marker-leakage transfer in #502 will predict fact-leakage and sycophancy-leakage transfer **more strongly than the coarse cosine + output-distribution JS/KL predictors did in #494/#470**.

Pre-registered honest framing of the three outcomes (all informative):
1. **Generalizes** — the ridge predicts fact and/or sycophancy transfer at a strength the coarse predictors missed → the residual-stream geometry is a behavior-agnostic persona-distance axis. Strongest possible positive.
2. **Marker-specific** — the ridge stays null on fact/sycophancy even at full resolution → #502's signal is specific to the marker construct (consistent with #502's own "ΔG and g_logprob pick different cells" non-universality), and the #494/#470 nulls were not resolution artifacts.
3. **Behavior-dependent** — generalizes to one behavior but not the other → tells us which axis of "behavior" the geometry actually tracks (stylistic/token-level vs propositional/content-level).

## What to run (default: reuse, no training — `kind: analysis`)

Per-behavior, mirror #502's pipeline:

1. **Reuse the existing transfer targets** (the dependent variable) — do NOT re-train or re-eval:
   - Fact: the per-(teach → bystander) leakage matrices already computed in #444/#494 (and #192/#381/#389/#390 as available).
   - Sycophancy: the per-(source, bystander) leakage matrix already computed in #411 / re-analyzed in #470.
2. **Recompute the predictor on each behavior's own persona panel** — the predictor *recipe* transfers but the predictor *values* must be recomputed, because the fact and sycophancy panels use different persona sets than #502's 16-condition panel. For each persona in the panel, run base-model `Qwen/Qwen-2.5-7B` forward passes over a shared probe pool and capture pre-norm residual-stream activations at all 28 layers at the three extraction points (end-of-system-prompt, last-prompt-token, mean-over-response). Reuse `scripts/issue493_extraction_metric_bakeoff.py` (the #502 driver) and `scripts/issue502_dispatch.py` directly.
3. **Run the full bake-off grid** — the same 9 cloud-aware metrics (cosine, Euclidean, Mahalanobis, pooled-Mahalanobis, MMD, C2ST, spectral-delta, Gaussian-KL on PCA-16, Bures-Wasserstein-2) + the next-token JS baseline, scored by length-partial Spearman ρ and leave-one-class-out CV R² against each behavior's transfer target.
4. **Report against #502 and the priors** — does the L19-L24 / L22 gauss_kl cell (and the ridge) lift over the coarse cosine + output-distribution JS/KL that #494/#470 used? Carry forward #502's honesty caveats: selection across ~1500 cells, LOCO-is-within-grid (not held-out), single-seed, and the per-checkpoint-instability warning.

## Reuse map (existing artifacts)

| Behavior | Trained adapters | Stored transfer target | Coarse-predictor prior |
|---|---|---|---|
| Fact | #381/#389/#390/#444 (`adapters/...`, HF model repo) | `eval_results/issue_494/predictor_444_canonical.json`, `predictor_192.json`, `regression_data.csv`; #444 leakage matrices | #494 (null on pooled panel, all 4 distance flavors) |
| Sycophancy | `superkaiba1/explore-persona-space/.../adapters/issue_411/` (HF) | #411 per-bystander leakage; #470 `issue470_jsdiv_predictor/base_responses` on HF data repo; #480 marker data | #470 (null, cosine + JS/KL), #480 (marker null) |
| Predictor recipe | — | — | #502 driver `scripts/issue493_extraction_metric_bakeoff.py` + `scripts/issue502_dispatch.py` |

## Design decisions for the planner / `/adversarial-planner`

1. **Reuse-first vs rebuild-on-#502-panel.** Default is reuse (Option A): recompute the #502 bake-off on each behavior's existing panel, score against existing leakage matrices, no training. The fallback (Option B) only if Option A's panels are too thin for a stable bake-off: train fact + sycophancy implants on #502's exact 16-condition panel and cross-eval matched ΔG-style matrices. Option B is much more expensive and only partial reuse — escalate before choosing it. The user's stated preference is reuse.
2. **Panel-size / power.** #502 needed 240 ordered pairs from a 16-condition panel; #494's fact panel was n≈14 personas (n=4-6 per substrate was the binding constraint it flagged), #411's sycophancy panel was 6 sources × 23 bystanders. Confirm each panel yields enough ordered pairs for a LOCO-CV bake-off; if not, that is the trigger to consider Option B or to pool carefully (and the planner must avoid the cross-substrate sign-flip confound #494 hit when pooling).
3. **Substrate / sign-flip confound.** #494 found the hidden-state cosine predictor *flips sign across training substrates* (contrastive-negative arm vs positive-only arm). Any pooled fact regression must residualize or stratify by substrate, or report within-substrate slopes, exactly as #494 learned the hard way.
4. **Prior control.** #500 showed the bystander's own base prior on the fact dominates fact leakage. The fact analysis should include the prior-log-prob control as a competing predictor and report the residual-stream cells' lift *over* the prior, not just raw ρ.
5. **Target validity / saturation.** #480 flagged a saturation pathology on the sycophancy marker DV (software-engineer adapter). Use a non-saturating leakage target (emission-corrected log-prob or full-vocab KL), and screen target cells for saturation before scoring, per the #448/#502 saturation lesson.
6. **Split vs single task.** Default is one task covering both behaviors (shared extraction pipeline). The planner may split into a fact arm and a sycophancy arm if the panel/target handling diverges enough.

## Success criteria

- For each behavior, a #502-style bake-off grid scored against the existing transfer target, with the L19-L24 / L22 gauss_kl cell explicitly reported alongside the coarse cosine + JS/KL cells that #494/#470 used, on the same pairs.
- A clear verdict on each of the three pre-registered outcomes (generalizes / marker-specific / behavior-dependent), with #502's caveats (selection inflation, within-grid CV, single seed) carried forward and the #494 substrate sign-flip + #500 prior controls applied to the fact arm.
- No overclaiming: a null at full resolution is a publishable result (the marker geometry is marker-specific) and must be framed as noise-limited-vs-effect-confirmed per the project's null-framing rule.

## References

- Recipe source: #502 (`tasks/awaiting_promotion/502`) — 28-layer bake-off, L22 gauss_kl winner, L19-L24 ridge.
- Parent of recipe: #493 (8-layer / 50-probe bake-off), #474 (marker ΔG / G_logprob substrate).
- Fact priors: #494 (null), #500 (prior dominates), #444 / #381 / #389 / #390 / #192 (adapters + matrices).
- Sycophancy priors: #470 (null), #480 (marker null), #411 (adapters + leakage panel), #99 (original cosine gradient).
- Rules: `.claude/rules/marker-leakage-measurement.md`, `.claude/rules/persona-distance-metrics.md`, `.claude/rules/contrastive-negatives.md`.
