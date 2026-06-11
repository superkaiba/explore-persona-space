---
title: 'Unified leakage rule + logit-channel anatomy: reviewed inference on the #532
  panel and transfer check on the #478 persona panel'
kind: analysis
tags: []
created_at: '2026-06-10T08:30:32Z'
has_clean_result: false
parent_id: 532
---
# Umbrella: unified leakage rule + logit-channel anatomy — reviewed inference on the #532 panel and transfer check on the #478 persona panel

## Goal

Establish, with reviewed inference, the unified marker-leakage rule and the logit-channel anatomy discovered in inline (unreviewed) analysis on #532's corrected-slot panel — push(S) is source-constant, the EOS clamp is context-routed, and the pair-specific leakage affinity is base-resident — and test whether that anatomy TRANSFERS to the #478/#531 persona panel via the same channel decomposition over committed logit-rescore JSONs. CPU-only end to end: no pod, no GPU, no training, no generation.

## Context

Parents/lineage: #531 (propensity hides in the subtraction), #532 (base prior beats geometric distance; corrected-slot four-float follow-up), #539 (source-level leakiness decomposition + corrected-reads inference machinery), #540 (JS arm = reply length). An inline chat session on 2026-06-10 ran a series of fits over #532's committed `logp_slot_followup` artifacts plus an adversarial metric review (critic agent). All numbers below are UNREVIEWED targets to reproduce/correct, not ground truth.

Inline findings to validate or kill:

1. **Unified rule, margin space.** `trained_margin ~ alpha * prior_margin(B) + beta * cosine(S,B)` is additive in EOS-margin space (interaction term vanishes there but is needed in log-prob space — softmax artifact). LOBO-CV R² ~0.71 full panel / ~0.37 ordinary-cross; + source FE ~0.89 (post-train forecast only). Known flaws to fix (from the critic): srcFE conflated with pre-training performance; alpha unidentified on ordinary-cross alone (bystander-cluster CI [-0.92, +1.37]); bystander-only clustering; B1/C1 quasi-duplicate pair included; cohort composition driving full-panel numbers.
2. **Channel anatomy (ordinary cross, 240 cells).** Two-way FE variance anatomy: Δz(※) = 89% source FE / 3% bystander / 8.5% pair (push is a per-source constant, ±1 logit); Δz(EOS) = 72% bystander / 22% source / 9% pair (clamp is context-routed, prior-correlated +0.59 raw). Pair-corrected (two-way FE) cosine reads: +0.13 vs Δz(※), +0.43 vs Δz(EOS), −0.25 vs Δmargin, +0.51 vs BASE matched-slot margin, +0.43 vs trained margin level. Interpretation: the leakage map is base-resident; training's pair-specific net change slightly OPPOSES closeness; geometry was never a transfer-router, it proxies base affinity.
3. **Within-source context ranking.** Median per-source Spearman vs trained margin level: base matched-slot margin +0.75 (all 25 bystanders) / +0.71 (15 ordinary); own-response prior +0.85 / +0.37; cosine −0.23 / +0.51; z(prior)+z(cos) +0.83 / +0.54. The matched-slot base margin is the best within-ordinary ranker but is not strictly pre-training-available; closing that gap is an open methods question.
4. **Diag-strength vs spill anti-correlation.** Across the 16 sources: at-home implant strength (diagonal margin) anti-correlates with off-diagonal spill (source FE), Spearman ~ −0.5. Thin/contentless source prompts (B1 bare question, C1 standard template, D2 casual rewrite) = weak at home + leak everywhere; rich personas = strong at home + contained. n=16, B1/C1 quasi-duplicate in the leaky group — needs proper inference.
5. **Negative-set exposure confound.** The #474 loc-arm adapters were trained with ALL 15 other panel conditions as EOS negatives (`scripts/i474_phase23_train.py::_build_negative_rows`; 15 bystanders × 20 Q = 300 negative rows). Therefore every ordinary off-diagonal cell measures leakage onto a TRAINED-negative context, while the 10 instructed bystanders were never clamped — #532's strip-level "prior beats geometry" partially confounds high prior with never-clamped. Mean Δz(EOS): +12.8 ordinary vs +6.4 instructed (clamp generalizes off-distribution at ~half strength). Quantify what can be separated.

## Deliverables

1. **(PRIMARY — transfer check)** Run the channel decomposition on the #478 panel using the committed #531 logit-rescore artifacts: `eval_results/issue_478/logit_rescore/` (80 runs = 40 CORE cells × 2 seeds; per-q trained+base z(※), z(EOS), log Z, log P at the post-response slot, 35 held-out personas × 20 questions) and `eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet`. Adapt the FE anatomy to that design (cell/seed × held-out persona; predictors: min_dist cosine distance, K, base prior). Test: (a) is the marker push constant per trained run (analogue of push-source-constancy)? (b) is the EOS-side change routed by the persona's base prior (clamp analogue)? (c) is the pair/persona-specific affinity base-resident (base-side margin carrying the distance signal)? Report agreement/disagreement with the #532-panel anatomy.
2. Re-fit the unified rule on the #532 panel per the metric-critic convention: EOS-margin DV (absolute trained state, joint fit, shift readouts derived); per-cohort AND pooled-with-cohort-FE; two-way cluster bootstrap (source AND bystander) on every coefficient; B1/C1 duplicate dropped as a robustness slice; feature sets labeled `pre-training forecast` vs `post-train forecast-where`. Data: `eval_results/issue_532/logp_slot_followup/` (416+416 per-cell four-float JSONs + `base_prior_logp.json`), `eval_results/issue_532/predictors.json`.
3. Attach proper inference to the channel-anatomy corrected reads (reuse/extend the FE-respecting bootstrap + permutation machinery from `scripts/issue539_corrected_reads_inference.py`).
4. Diag-vs-spill anti-correlation with inference (cluster bootstrap, permutation, with/without the B1/C1 duplicate).
5. Within-source ranking table with CIs (per-source Spearman distributions, all-bystander and ordinary-only slices, vs margin level and vs emission).
6. Negative-set exposure analysis: separate prior from clamp-exposure where the design allows (within-strip gradients, clamp-generalization magnitudes, any identifiable contrast).
7. Clean-result body per the 2-content-section v2 spec, including a recommendation (NOT an edit) on amending `.claude/rules/marker-leakage-measurement.md` per the metric-critic bundle: EOS margin as the modeling/leaderboard DV (joint-fit convention, shift readouts derived); emission rate stays the safety headline ALWAYS paired with the margin-headroom distribution of non-firing cells; per-cell triple-channel reporting (Δz(※), Δz(EOS), Δmargin); storage-contract extension candidate `z_top_nonmarker` (the two-horse-race check failed base-side: argmax ∈ {※, EOS} at only 2-21% of base matched-slot reads vs 96-100% trained-side); forbidden moves (no marginal correlations against shift DVs; no pooled cross-cohort headline correlations; no cross-DV R² comparisons to pick a space; no srcFE numbers quoted as pre-training performance). Any actual rule edit is a separate user-approved workflow change.

## Constraints

- CPU-only over committed JSONs/parquet; no pod, no training, no generation. If any deliverable turns out to need GPU, descope it to a named follow-up rather than provisioning.
- Follow the metric-critic forbidden moves throughout (above).
- The inline session's numbers are targets to reproduce or correct — treat disagreement as a finding, not an error to hide.
- Statistical machinery precedent: #539's corrected-reads inference (FE re-estimated inside every bootstrap resample; Freedman-Lane-style FE-respecting permutation).
