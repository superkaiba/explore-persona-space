---
title: Extend leakage prediction to refusal + emergent misalignment (syco-recipe analogue)
  + training-completion log-prob metric
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-08T07:39:36Z'
has_clean_result: false
parent_id: 509
goal: 'Re-run the #470/#509 leakage-prediction analysis for refusal and emergent-misalignment
  behaviors, reusing the closest old source-by-bystander leakage panels, scoring the
  same predictor zoo (cosine_l20, JS, KL) plus bystander base rate (confound-aware,
  per #509) and a new training-completion log-prob metric (generalizing #500/#444''s
  fact-validated bystander_logprob), and report whether any predictor generalizes
  across behaviors; also report training-completion log-prob trained-minus-base as
  a non-saturating implant-quality readout on each source.'
---
# Extend leakage prediction to refusal and emergent misalignment, reusing old panels (sycophancy-recipe analogue) + add training-completion log-prob as an implant/predictor metric

## Goal

Re-run the #470/#509 leakage-prediction analysis for refusal and emergent-misalignment behaviors, reusing the closest old source-by-bystander leakage panels, scoring the same predictor zoo (cosine_l20, JS, KL) plus bystander base rate (confound-aware, per #509) and a new training-completion log-prob metric (generalizing #500/#444's fact-validated bystander_logprob), and report whether any predictor generalizes across behaviors; also report training-completion log-prob trained-minus-base as a non-saturating implant-quality readout on each source.

The leakage-prediction line has only been run cleanly on two behaviors: marker (#470/#502/#509) and sycophancy (#411/#470/#509), plus facts (#444/#494/#500). The headline question — *does any before-training signal predict how much training a behavior into a source persona leaks to bystanders* — is still open for the two behaviors we most care about: **refusal** and **emergent misalignment (EM)**. This task closes that gap by re-using old experiment results (no new training where avoidable), following the #411/#470/#509 sycophancy recipe as faithfully as the old data allows, and adding one new metric Thomas proposed.

## Goal-shaped summary of what to do

Two behavior arms (refusal, EM), each scored exactly like #509's sycophancy arm:
- DV = per-(source, bystander) leakage Δ (trained − base, baseline-subtracted per bystander), on a source×bystander panel.
- Predictors: score the **SAME FULL predictor battery every prior arm was scored against** (parity is the point — a cross-behavior comparison is only valid if refusal/EM see the identical predictor set as marker/syco/fact). Concretely:
  1. **Full residual-stream bake-off (#502/#509)** — 3 extraction points {end_of_system, last_prompt, mean_response} × 28 layers (0-27) × 9 cloud-distance metrics {cosine, euclidean, mahal, mahal_pooled_ctx, mmd, c2st, gauss_kl(PCA-16), wass2, next_token_js baseline} ≈ 1000+ cells per arm. This is the "better predictor" line; it must run for parity even though #509 found it marker-specific.
  2. **Coarse geometry + divergence zoo (#470/#480)** — cosine_l20 baseline, cosine_response at L7/L14/L21/L27, JS {sym, from_source, from_bystander}, M_js, KL {src→bys, bys→src, sym}.
  3. **Priors** — bystander base rate (the confound-aware covariate from #509's base-rate re-analysis: predict the absolute trained rate with FE and flag the Δ-circularity, per `eval_results/issue_509/baserate_covariate/`), and the **new completion-log-prob metric** (below).
  4. **Response-length controls** (#480) — source/bystander resp-len means + abs diff, residualized out (the #509 length-partial step).
  5. *Optional, planner's call:* the #468 in-context-example cosine + He et al. (arXiv 2404.01099) representation/gradient/format selectors that the #503 calibration line uses — include if the cross-line comparison is wanted.
  All scored source-FE, with the #509 machinery: attenuation-adjusted ρ, within-source permutation null (B=2000), cluster bootstrap CI (B=5000), pre-registered cell + ridge + search-best with selection-inflation disclaimer.
- Report whether ANY predictor generalizes across behaviors (marker → syco → fact → refusal → EM), or whether each behavior has its own predictor cell (the #509 "different behavior, different cell" finding).

## New metric: log-prob of the training completions

Add `log P(training completions)` as a metric, in BOTH roles:

1. **Implant-quality readout (on the source).** Mean length-normalized `log P(positive training completion | source system prompt)`, reported **trained − base**. This is a continuous implant-strength measure that does NOT saturate the way the behavioral rate does (the rate hit 95-97% on 5/6 syco sources and saturated entirely in #448), so it can rank implant strength even among "fully implanted" sources. Pair it with the existing on-policy behavioral rate (which stays the headline DV).
2. **Base-model leakage predictor (on bystanders).** Length-normalized `log P(positive training completions | bystander system prompt)` on the FROZEN base model — does the base model already assign high probability to the behavior's trained completions under a bystander's persona → that bystander leaks. **This metric already exists and dominates fact leakage:** #500 (`eval_results/issue_500/bystander_logprob/logprob_results.json`) and #444 (`eval_results/issue_444/bystander_logprob/`) computed exactly this for facts and found ρ = 0.61-0.80 vs leakage. This task tests whether it generalizes to refusal/EM/sycophancy — directly relevant because #509's base-rate re-analysis found the base-rate prior does NOT cleanly predict sycophancy leakage (where #500 found the fact prior dominant), so completion-log-prob is the sharper version of "the prior" to test for generalization.

## Reuse map (old experiment results — verify in the planner)

- **EM arm — find the true misalignment-leakage panel.** Candidates: `#207` (`eval_results/issue_207/js_gentle/regression_data.csv` — contrastive EM/behavior leakage with JS), `#247`/`#329` (contrastive EM leakage, reported source ~99.6% / bystander ~11.7%), `eval_results/single_token_100_persona/cosine_leakage_correlation.json`, `eval_results/proximity_transfer/expA_leakage.json`, `scripts/run_em_multiseed.py` / `scripts/run_leakage_v3_onpolicy.py`. **CAUTION:** `#368`'s `phase2/leakage_table.csv` is a clean 10×11 source×bystander matrix BUT its DV is the **[ZLT] marker**, not misalignment — it is the right STRUCTURE with the WRONG behavior. The planner must confirm the chosen EM matrix's DV is an actual misalignment judge (Betley-style alignment score), not a marker.
- **Refusal arm.** `#390` (`eval_results/issue_390/{aggregate_long.json, h4_refusal_breakdown.json, cells/}` — refusal pool, but NOT a full source×bystander factorial), and `#381`/`#389` (contrastive refusal/fact structure) for the panel shape. The planner should assess whether a clean refusal source×bystander matrix exists or whether a minimal re-eval is needed to build one (reuse old adapters + the #411 24-persona panel).
- **Completion-log-prob precedent:** reuse the #500/#444 `bystander_logprob/` computation directly (same length-normalization, same frozen-base read) so the new metric is methodologically identical to the fact-arm version.

## Faithfulness, single-variable framing, exemptions

- **Follow the #411/#470/#509 sycophancy recipe as closely as the old data allows** — same predictor zoo, same source-FE + cluster-bootstrap + permutation machinery, same 24-persona panel where overlap permits, same on-policy behavioral DV. Per the new CLAUDE.md replication-fidelity rule, name every forced deviation from the syco recipe (different persona set, marker-vs-judge DV, fewer seeds in the old data) in plan §-assumptions and carry as scope caveats.
- The changed variable vs #509 is the **behavior** (refusal, EM) — hold the predictor set, statistics, and panel fixed where possible.
- This is a re-analysis of old positive-only / contrastive results, not new behavior implantation, so the contrastive-negatives rule is mostly N/A; where any re-eval re-trains, carry the parent's contrastive design.
- **Measurement-validity caveat to state in the plan:** `log P(completion)` is teacher-forced, so it is valid as a PREDICTOR and as an implant-strength READOUT, but NOT as the behavioral DV — the on-policy behavioral rate (refusal rate / misalignment rate) stays the DV (consistent with the marker-leakage measurement rule).

## Relation to existing tasks

- **Absorbs #473** ("JS vs cosine as a predictor of EM-misalignment leakage, EM analogue of #470") — this task covers #473's EM-predictor scope and adds refusal + the completion-log-prob metric + the base-rate covariate. Recommend archiving #473 as absorbed once this is approved.
- **Extends #414** ("cheap pre-SFT predictor: does the broad-persona-prompted base model already produce the narrow training dataset's responses") and **#499** ("P(training data | bystander context) as a base-rate baseline") — the completion-log-prob metric is the concrete instantiation of both; cite their framing.
- Sibling of #509 (fact + syco arms), #470 (cosine/JS syco predictor), #500/#444 (fact completion-log-prob). Parent line: #446 (B→B′).

Parent: #509.
