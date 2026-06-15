---
title: 'Prior-stratified rerun of #500: enrich bystander panel and source set with
  high-prior personas'
kind: experiment
tags:
- leak-predictor
- fact-teach-persona-transfer
created_at: '2026-06-09T22:00:55Z'
has_clean_result: false
parent_id: 500
goal: 'Test whether the bystander-prior→leakage relationship found in #500 holds on
  a persona panel deliberately enriched with high-prior personas (de-leveraging the
  two outliers that carried the #500 headline), and whether the source persona''s
  MEASURED prior on the fact predicts panel-wide gating tightness, by re-running the
  #500 design with a prior-stratified bystander panel and source personas selected
  by measured base prior instead of hand-coded content-relatedness.'
relates_to:
- leak-predictor
- fact-teach-persona-transfer
---
## Goal

Test whether the bystander-prior→leakage relationship found in #500 holds on a persona panel deliberately enriched with high-prior personas (de-leveraging the two outliers that carried the #500 headline), and whether the source persona's MEASURED prior on the fact predicts panel-wide gating tightness, by re-running the #500 design with a prior-stratified bystander panel and source personas selected by measured base prior instead of hand-coded content-relatedness.

## Motivation

#500 (parent) found the bystander's own length-normalized base log P of the taught completion is the only predictor of fact leakage that survives controls; source proximity adds nothing. But the result has two structural weaknesses this rerun targets directly:

1. **The prior range was narrow and outlier-driven.** Panel priors spanned only [−3.55, −3.10] and the headline ρ rested on 2 high-prior courthouse-themed personas (`local_historian`, `courthouse_architecture_historian`). Drop both and the local-resident condition goes flat (ρ = 0.39, p = 0.21). A panel with dense coverage of the high-prior end turns the leverage points into a populated stratum.
2. **The source "content-relatedness" gradient was ordinal (−1/0/+1) and the measured priors contradict it.** Measured source priors: marine_biologist −3.403, local_resident −3.390 (essentially identical — NOT intermediate), courthouse_architecture_historian −3.229. The pooled proximity×relatedness interaction was fit against an axis whose bottom two levels are indistinguishable on the dimension that drives everything else in the experiment. Sources selected by measured prior fix the axis.

A secondary observation worth testing as a real prediction: in #500, the condition with the highest source prior gated most tightly (12/14 bystanders floored). With sources spanning a measured prior range, "source prior → panel-wide gating tightness" becomes a testable monotone relationship instead of an n=3 anecdote.

## Design sketch (planner refines)

- **Phase 0 — prior prescreen (cheap, frozen base, no training).** Compute length-normalized log P(taught completion | persona system prompt) for a candidate pool of ~30–50 personas, including the original 15 plus new candidates designed to sit high (courthouse docent, county-records clerk, Pennsylvania travel writer, historic-preservation officer, Elk County tour guide, trivia-night host, genealogist, etc.). Also screen each candidate's BASE stated_seven false-positive rate on the headline framings (the #500 exclusion logic, applied per-persona: base FP >5% on a framing excludes that framing or persona).
- **Panel selection.** Bystander panel ~20–24 personas stratified across the measured prior range, oversampling the high end; the original 15-persona pool stays nested inside so #500 replicates within the new run. Source set: 3–4 personas selected by measured prior spanning low → high (anchor with marine_biologist for direct comparability; replace the ordinal relatedness coding with the measured source prior in the pooled fits).
- **Everything else inherited from #500 unchanged** (single-variable change = persona-panel composition): same invented fact (Elk County Courthouse, seven wooden benches), same contrastive recipe (100 positives + 200 on-policy negatives from 4 arbitrary non-teach personas + 600 Tulu; LoRA r=32/α=64 rsLoRA, lr=2e-4, 1 epoch), 3 seeds (42/137/256), same 5-way Claude judge (`stated_seven`/`stated_nine`/`didnt_mention`/`refused`/`confabulated_other`), same headline framing set, on-policy eval at temp=0, max_new_tokens=2048.
- **Produce the engagement covariates this time.** `engagement_covariates.json` (completion length + on-topic fraction per persona) was flagged-but-missing in #500, leaving the prior-vs-content-fit decomposition open. The engagement-adjusted partial Spearman (ρ(prior, leak | length, on-topic fraction)) is a primary analysis here, not an optional extra.

## Predictions

1. **Prior robustness:** within each non-floored source condition, ρ(prior, leak) > 0 and survives drop-one / drop-top-stratum on the enriched panel (the #500 marine point estimate 0.80 with cluster-bootstrap mean 0.50 suggests a true moderate-positive; the enriched panel should land between those).
2. **Source-prior → gating:** panel-median leak rate decreases monotonically with the source's measured prior (the #500 architectural-historian collapse generalizes).
3. **Content-fit vs string-prior:** the engagement-adjusted partial either preserves the prior signal (prior is real) or kills it (the prior was proxying topic-engagement) — both outcomes are informative for what the predictor program should compute pre-training.

## Risks / known failure modes

- **High-prior sources may floor the panel** (#500's content-related condition leaked through only 2/14 bystanders). Mitigation: dynamic-range diagnostic per condition is a gate; the highest-prior source condition is expected to be partially floored and the analysis must treat any floored condition as uninformative rather than averaging it in.
- **High-prior bystanders may have elevated BASE stated_seven rates**, contaminating the leak DV. Mitigation: per-persona base screening in Phase 0; report leak as trained − base where the base rate is non-negligible.
- **The invented fact caps how high a prior can go** — even a courthouse docent can't have a high prior on the specific "seven wooden benches" string. If the prescreen shows the candidate pool can't beat local_historian's −3.10 by a useful margin, the high-prior stratum is thin and the plan should say what range is achievable before training anything (this is exactly what Phase 0 is for — it's a go/no-go gate on the rest of the experiment).

## Relation to parent

Direct follow-up to #500 (same fact, same recipe, same judge); also addresses #494's closing recommendation (purpose-built within-substrate panels). The marine_biologist source condition is a near-replication arm (fresh seeds optional — planner decides whether to reuse the #444 adapters as #500 did or retrain for pipeline uniformity per the artifact-reuse fitness check).
