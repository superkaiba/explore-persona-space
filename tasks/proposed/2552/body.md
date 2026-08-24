---
title: 'Turn-averaged SAE feature predictability: Der et al. replication, category-level
  map reads, and a partialed covariate ladder'
kind: experiment
tags: []
created_at: '2026-08-24T20:18:15Z'
has_clean_result: false
parent_id: 2476
origin_prompt: 'I want to rerun their thing and then first see if our mapping predicts
  better the higher level things for our Matryoshka one. But I want to run one that
  is exactly what they run also, and then see what metrics best predict if one of
  the features will be predicted well. So it should be basically rerun their judgment
  also. So it should be their types, but also additional metrics inspired by the other
  experiment we ran for the SAE features. And then the plot should basically be like,
  okay, this is the property that best explains it, and then we control for that or
  partial that out and this is the next one, and then control for that as well, this
  is the next one. (2026-08-24; clarify-gate answers: one child task of #2476; assistant
  means banked; k=200 twin in, nested/attribution out; judge = sonnet-4-6 everywhere,
  user override; exploratory category ranking; spawn --auto)'
workflow: v1
goal: 'Determine (1, exploratory) how the context→answer map''s per-feature predictability
  of turn-averaged SAE features ranks across the five Der et al. (arXiv 2606.28548)
  schema categories (content/form/voice/function/meta) on the #2476 matryoshka SAEs
  (k=100 + k=200) and on a faithful flat replication of Der et al.''s recipe; (2)
  whether that replication (BatchTopK 32,768/k=128 on banked layer-19 turn means +
  their full judged evaluation, judge claude-sonnet-4-6 no-prefill) reproduces their
  discrimination-vs-coverage inversion; and (3) which feature properties (schema categories
  + the #1482-inspired turn-grain covariate battery) explain per-feature map predictability,
  via a forward-selection partial-out ladder as the headline figure.'
---
# Turn-averaged SAE feature predictability: Der et al. replication, category-level map reads, and a partialed covariate ladder

## Goal

Determine (1, exploratory) how the context→answer map's per-feature predictability of turn-averaged SAE features ranks across the five Der et al. (arXiv 2606.28548) schema categories (content/form/voice/function/meta) on the #2476 matryoshka SAEs (k=100 + k=200) and on a faithful flat replication of Der et al.'s recipe; (2) whether that replication (BatchTopK 32,768/k=128 on banked layer-19 turn means + their full judged evaluation, judge claude-sonnet-4-6 no-prefill) reproduces their discrimination-vs-coverage inversion; and (3) which feature properties (schema categories + the #1482-inspired turn-grain covariate battery) explain per-feature map predictability, via a forward-selection partial-out ladder as the headline figure.

1. **Category-level map predictability (exploratory).** Rank the five Der et al. (arXiv 2606.28548) schema categories (content, form, voice, function, meta) by how well the context→answer map predicts the turn-averaged SAE features assigned to each, with intervals; no pre-registered ordering (explicit user choice: exploratory only). Read on BOTH the in-house matryoshka turn-averaged SAEs (#2476, k=100 primary + k=200 twin, banked per-feature R²) and the faithful replication below. This upgrades the #1482/#2476 coarse-over-specific gradient to a category-level description.
2. **Faithful replication of the Der et al. instrument and judgment.** Train their exact SAE recipe (flat BatchTopK, d_sae 32,768, k=128, turn-averaged LMSYS activations, layer 19) and rerun their full judged evaluation: auto-interp descriptions from each feature's top-25 activating turns; per-turn structured summaries under the 24-field / 5-category schema copied verbatim from their Appendix D; 10-way matching (discrimination); pairwise coverage ranking. Reproduction target: their qualitative inversion (per-token more discriminative; turn-averaged better coverage, 87.9% average pairwise win in the paper).
3. **Which properties of a turn-averaged SAE feature explain whether the map predicts it well?** Covariates: the feature's schema category/field assignment (their types) PLUS a turn-grain adaptation of the #1482 feature-correlates battery: activity (firing rate), activation variance across conversations, decoder norm, direct-logit footprint, co-activation degree, decoder alignment with persona/trait directions, decoder alignment with answer-PCA variance rank, matryoshka tier (where defined), corpus balance (LMSYS vs WildChat firing share), and the matched token-level twin's within-answer consistency via cross-dictionary decoder matching. **Headline figure: a forward-selection partial ladder** — the single property that best explains per-feature held-out R² first; partial it out; the next-best property on the residual; iterate until increments are null-band-level.

## Decision record (clarify gate, 2026-08-24 — answers are the user's)

- Routing: ONE new child task of #2476 carrying all three legs (user-answer).
- Replication training data: banked assistant whole-answer means only (963k, layer 19); no Human-turn capture. Deviations from the paper named per the replication-fidelity rule: assistant-only turns, whole-answer-mean convention incl. end-of-turn tail, n ≈ 963k vs their ~1.58M, single training seed (user-answer).
- Arms IN: #2476 k=200 twin as a sparsity-budget robustness arm (user-answer). Arms OUT (explicit): Der's nested turn+token variant; Der's decoupled variant; the attribution-graph leg; Human-turn SAE; any steering/control leg (user-answer via unselected options).
- Judge: **their judge, claude-sonnet-4-6, everywhere in this task** — descriptions, category assignment, summaries, 10-way matching, pairwise coverage — with NO assistant prefill (the paper shows prefill collapses coverage judgments to near-random, 49.1% vs 83.1%). This is an explicit user override of the pinned-project-judge rule for this task; a small pinned-judge (claude-sonnet-4-5-20250929) agreement subsample is kept as the calibration control, and the category-predictability read carries a judge-mismatch caveat when quoted next to pinned-judge results (user-answer).
- Category contrast: exploratory only — full 5-category ranking with intervals, no pre-registered ordering (user-answer).
- Launch: spawn autonomous (--auto) immediately after filing (user-answer).
- Inherited/rule-pinned (not re-asked): model+layer (Qwen2.5-7B-Instruct, layer 19 — the banked store and the paper coincide); linear ridge maps with pinned splits, identity+bias baseline, kNN retrieval, shuffle nulls (mapping-baselines rule + #2476 recipe); within-activity stratification and permutation nulls (the #1482 range-restriction lesson); Batch API + pilot gate for the ≥5k-call judge waves; runpod-first auto compute lane.

## Design sketch (for the planner)

- **Phase A (judge-only, banked artifacts):** auto-interp descriptions for the #2476 matryoshka SAEs' alive features (k=100 + k=200; top-25 activating turns per feature, turn text + activation value in the prompt, per the paper's protocol); judge-assign each feature to schema fields/categories; aggregate the banked per-feature R² by category within activity strata.
- **Phase B (GPU, cheap band):** train the flat replication SAE on the banked layer-19 means (BatchTopK, 32,768 features, k=128, ~their LR/batch/epochs from Appendix A).
- **Phase C (judge waves, Batch API, pilot-gated):** descriptions for the replication SAE's alive features; per-turn structured summaries on ~2,000 holdout turns; 10-way matching; pairwise coverage across the available feature-list configurations; pinned-judge agreement subsample.
- **Phase D (analysis, 0-GPU):** context→feature ridge maps for the replication dictionary (the #2476/#1482 recipe); per-feature covariate battery; forward-selection partial ladder with permutation nulls per step; 5-category predictability ranking on all three dictionaries.
- Est. compute: ~6–12 GPU-h (SAE training + fits) + ~30–60k judge calls (descriptions dominate; Batch API).

## Provenance / lineage

- #2476: in-house matryoshka turn-averaged SAEs (k=100 / k=200, width 65,536), banked weights + per-feature R² + firing censuses; zero judge calls so far (no feature descriptions exist yet).
- #1482: token-level feature-correlates battery (within-answer consistency dominant, ρ 0.60–0.68; predictability an answer-side property, 0.93) — the battery this task adapts to turn grain.
- Der, Kamath & Thompson 2026, arXiv 2606.28548: recipe (their Appendix A), schema (Appendix D), evaluation protocols, judge-prefill sensitivity; no public code or weights (verified in-text 2026-08-24).
