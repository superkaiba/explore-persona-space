---
title: Flat Der-recipe context SAE and a full-dictionary context-feature to answer-feature
  map (odd-correlation edge read)
kind: experiment
tags:
- lean-run
created_at: '2026-09-03T18:48:50Z'
has_clean_result: false
parent_id: 2552
origin_prompt: 'I want a flat replication of theirs for both context and answer SAEs.
  And I want us to log the (simple) metrics they used for both. Then I want to make
  a map from FULL context dictionary to FULL answer dictionary. / we care most about
  seeing if we can detect odd behavior and correlations (like china refusal) (2026-09-03,
  /clarify gate answers in ## Provenance)'
workflow: v1
goal: 'Train a flat Der-recipe (32,768/k=128) SAE on layer-19 last-prompt-token context
  states, log Der''s reconstruction metrics for it and the reused #2552 flat answer
  SAE, then fit ridge and MLP maps from the full context dictionary to the full answer
  dictionary with no alive floor and read the labelled edges for odd behavior correlations
  such as China-topic context to refusal answers.'
---
# Flat Der-recipe context SAE + full-dictionary context-feature to answer-feature map

## Goal

Train a flat Der-recipe (32,768/k=128) SAE on layer-19 last-prompt-token context states, log Der's reconstruction metrics for it and the reused #2552 flat answer SAE, then fit ridge and MLP maps from the full context dictionary to the full answer dictionary with no alive floor and read the labelled edges for odd behavior correlations such as China-topic context to refusal answers.

## Provenance

Routing: lean subagent run (implementer, one review round, experimenter on a 1x H100 RunPod pod), NOT a full /issue cycle. User-approved 2026-09-03 via the /clarify gate. Child of #2552 (flat answer SAE) with #2569 leg 4 (matryoshka feature map) as the method parent.

Decision record (source in brackets):

- Goal [ask]: replicate Der et al. (arXiv 2606.28548) flat BatchTopK SAE recipe on the CONTEXT side; reuse the #2552 flat ANSWER SAE; log Der's simple metrics for both; fit a map from the FULL context dictionary to the FULL answer dictionary with no alive floor; headline read = detection of odd behavior/correlation edges (e.g. China-topic context features to refusal answer features).
- Model / layer / objects [inherited #779, #2476]: Qwen2.5-7B-Instruct, layer 19; context object = last-prompt-token residual state; answer object = whole-answer mean residual state.
- Data [inherited #2476]: banked 963,444 LMSYS/WildChat first-turn conversations (1,920 capture chunks @ 89cfa76cdc); splits re-asserted by sha: 933,444 SAE-train / 10,000 SAE-val / 20,000 holdout; map fit rows 120,000, lambda-val 400, test 1,000.
- Context SAE [user-answer: train new]: flat BatchTopK, width 32,768, k=128, lr 2e-4, batch 256, 3 epochs, Adam(0.9, 0.999), threshold EMA 0.999, init seed = this task id, trained on the 933,444 context states. Halt if holdout variance-FVE < 0.5.
- Answer SAE [user-answer: reuse]: #2552 replication SAE, HF issue2552_turnsae/analysis_tensors/sae_rep (32,768 / k=128, holdout nMSE 0.078).
- SAE metrics, both dictionaries [user-answer]: holdout nMSE (Der's metric, E||x-xhat||^2 / E||x||^2), variance-FVE, realized L0, dead-feature census on fit and holdout rows, per-feature firing-rate histogram. OUT: cross-domain nMSE.
- Der judged evals for the context SAE [user-answer]: structured 24-field summaries of the USER PROMPT (not the answer), context SAE only (no per-token comparator), so: feature descriptions (W1), prompt summaries (W2), 10-way matching (W4), embedding coverage metric (Qwen3-Embedding-8B). Pairwise and 5-way ranking need comparators and are OUT.
- Judge [user-answer, overrides pinned project judge]: claude-opus-5, no assistant prefill, Batch API. Budget estimate ~$250; description-need set = features in the 2,000 eval turns' lists UNION features appearing in reported edges (not all 32,768 unless the union reaches it).
- Map [user-answer]: ridge from all 32,768 context features to all 32,768 answer features on the 120,000 fit rows (lambda by validation on the 400-row split, scored on the 20,000 holdout), PLUS an MLP companion (1 seed). No alive floor on either side; dead columns are dropped mechanically (zero variance) and reported as such.
- Controls (edge-gating machinery, orchestrator's call, stated): composed zero-fit route (banked dense map then answer-SAE encode), dense-input ridge (context state to answer features), index-aligned identity+bias null, train-mean null, 20-draw row-shuffle null, kNN retrieval acc@1/5/10.
- Reads: per-feature held-out R^2, firing AUROC, conditional-magnitude R^2, all vs firing count with per-feature null bands; census-only for features that never fire in the holdout. Edges: standardized ridge coefficients, top-k in/out edges per feature, gated by split-half replication + label-shuffle null (the #1482 map_coefficients recipe); labelled both sides (context: W1 descriptions; answer: #2552 descriptions_mat/rep); receipts for refusal, CCP-position, Qwen-identity, sycophancy-adjacent answer features; an "unexpected edge" flag = strong gated edge whose two labels are semantically distant (embedding cosine).
- Success criteria [user-answer]: descriptive report, no pre-registered verdict beyond the FVE halt floor.
- Seeds [user-answer]: 1 SAE seed, 1 MLP seed.
- Compute [user-answer]: 1x H100 RunPod, est. 4-6 GPU-h ($15-25). Judge waves VM-side on the Batch API.
- OUT of scope [user-answer]: matryoshka retrains, k sweeps, cross-domain nMSE, steering or causal reads, paper edits, per-token comparator on the context side.

User's words (2026-09-03): "I want a flat replication of theirs for both context and answer SAEs. And I want us to log the (simple) metrics they used for both. Then I want to make a map from FULL context dictionary to FULL answer dictionary." / "we care most about seeing if we can detect odd behavior and correlations (like china refusal)".


## Results (lean run, 2026-09-04)

Status vocabulary: done / running / to do. Everything below is read from committed or HF-pinned artifacts (paths in each section). Plain-language glossary, first use only:

- **nMSE (raw)**: mean squared reconstruction error divided by the mean squared norm of the raw state. Der et al.'s metric. Lower is better. Their turn-averaged SAE: 0.097.
- **FVE**: fraction of variance explained after mean-centering. 1.0 is perfect.
- **dead feature**: a dictionary feature that never fires on the 120,000 map-fit rows (inference threshold gating).
- **pooled R²**: variance explained across all answer features at once (weighted toward high-variance features).
- **per-feature R²**: variance explained for one answer feature on the 20,000 holdout rows.
- **kNN retrieval acc@k**: fraction of holdout rows whose true answer-feature vector is among the k nearest of the 20,000 holdout pool to the predicted vector. Chance at k=1 is 1/20,000.
- **surviving edge**: a (context feature, answer feature) ridge coefficient in the top-20,000 by magnitude in BOTH split halves, sign-consistent, and above the per-column label-shuffle null.
- **Der coverage**: for each of the 24 summary fields of a held-out user prompt, the mean cosine between the field's embedding and the top-k most similar descriptions among the context features judged to fire on that prompt (Qwen3-Embedding-8B).

### Takeaways

1. **Both flat SAEs beat Der et al.'s reconstruction number by 4x.** Context SAE holdout raw nMSE 0.024 (FVE 0.919, L0 132), reused answer SAE 0.023 (FVE 0.922, L0 128), versus Der's 0.097 for their turn-averaged SAE. The halt condition (FVE < 0.5) was far from binding. Both dictionaries are more than half dead: 17,552 of 32,768 context features and 18,055 answer features never fire on the 120k fit rows.
2. **The full-dictionary map works at the pooled level but is thin per feature.** Ridge from all 15,216 live context features to all 32,768 answer features: holdout pooled R² 0.602, MLP companion 0.624. Per answer feature the median holdout R² is 0.10 (ridge), rising from 0 in the rarest deciles to 0.46 in the most frequently firing decile. Firing itself is easy to predict (median AUROC 0.96) but conditional magnitude is not (median conditional R² below 0).
3. **Going through context features loses information relative to the dense state.** A ridge from the raw 3,584-dim context state to the same 32,768 answer features gets pooled R² 0.649 (vs 0.602), and the zero-fit composed route (the banked #1482 dense context-to-answer ridge followed by the answer SAE encoder) retrieves the true holdout row best: cosine acc@1 0.737 vs 0.711 (dense-state ridge), 0.603 (MLP on features), 0.525 (ridge on features). Chance is 0.00005.
4. **Edges are sparse and interpretable.** 1,923 of 20,000 candidate edges survive the split-half plus null gate. The coefficient matrix is spread out (participation ratio of the spectrum 82) but each answer feature is fed by few context features (median per-column participation ratio 7.4, vs 167 in the #1482 dense-state map). The index-aligned diagonal carries 0.003% of the mass, confirming feature indices are unrelated across dictionaries.
5. **"China refusal" is at most a weak, ideological-prompt-specific hedging edge.** 49 politically sensitive China context features exist (Taiwan, Tibet, Tiananmen, CCP doctrine, censorship circumvention, Party documents) and all are in the live edge rows. Their surviving edges land on topic-matched answer features (Party-document requests wire to Party-organizational-work answer features at +0.02 to +0.05). Sovereignty questions (features 2511, 9086, 10946) do not wire to any refusal-described answer feature. The one non-topical edge: CCP-doctrine study prompts (context 8645) feed the "deflect, redirect, or add ethical caveats" answer feature 27425 at +0.033 standardized units, the second-strongest refusal-side in-edge from any China feature. Full read with caveats: `eval_results/issue_2661/odd_correlations_read.md`.
6. **The judged evals replicate the Der pipeline at scale.** 7,435 of 7,463 needed context features received Opus 5 descriptions (99.6%), 1,983 of 2,000 held-out prompts received 24-field summaries. Der coverage overall top-1 0.669 (top-5 0.643), ranging from persona 0.81 to tone 0.57 by field. 10-way matching accuracy 0.967 (1,914 of 1,979 valid prompts, Wilson 95% interval 0.958 to 0.974, chance 0.10), flat across the ten answer slots (0.96 to 0.98 each, choice rates about 0.10, no position bias).

### SAE metrics (done)

![SAE metrics](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b63448a250a80b78f6fc76c19b6f3fc88e25f43a/figures/issue_2661/sae_metrics.png)

> Left: holdout raw nMSE for the two #2661 dictionaries beside Der et al.'s published numbers. Middle: holdout FVE. Right: census of the 32,768 features, split into fires-at-least-once vs never-fires on the 120k fit rows. Source: `sae_metrics_{ctx,answer}.json` (HF `issue2661_flatsae/analysis_tensors/sae_metrics`).

| dictionary | holdout raw nMSE | mean-centered nMSE | FVE | realized L0 | dead on 120k fit rows | dead on 20k holdout |
|---|---|---|---|---|---|---|
| context SAE (new; last-prompt-token state, layer 19) | 0.0240 | 0.0808 | 0.919 | 132.2 | 17,552 | 22,821 |
| answer SAE (reused #2552; whole-answer mean, layer 19) | 0.0235 | 0.0777 | 0.922 | 127.7 | 18,055 | 23,765 |
| Der et al. turn-averaged (paper) | 0.097 | n/a | n/a | n/a | n/a | n/a |

Der's number is on raw states, so the raw column is the like-for-like comparison. The mean-centered nMSE (0.08) is the stricter read: about 70% of the raw squared norm is the shared mean direction, which any decoder bias captures. Context-SAE training: 3 epochs over 933,444 rows, 10,941 steps, val FVE 0.894 → 0.915 → 0.919 by epoch (`sae_ctx/train_log.json`).

### Full-dictionary map (done)

![Map routes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b63448a250a80b78f6fc76c19b6f3fc88e25f43a/figures/issue_2661/map_routes.png)

> Left: holdout pooled R² per route. Middle: kNN retrieval (cosine) at k = 1, 5, 10 over the 20,000-row holdout pool per route, with chance. Right: per-answer-feature holdout R² median by firing-count decile per route. Source: `map_ridge_metrics.json`, `map_mlp_metrics.json`, `controls.json`, `knn_retrieval.json`, `perfeature_summary.json`.

| route | inputs | holdout pooled R² | kNN cosine acc@1 / @5 / @10 | per-feature R² median | firing AUROC median |
|---|---|---|---|---|---|
| ridge (λ = 1e5 by 400-row val) | 15,216 live ctx features, standardized | 0.602 | 0.525 / 0.742 / 0.807 | 0.101 | 0.964 |
| MLP (4096 GELU, standardized targets, best epoch 5) | same | 0.624 | 0.603 / 0.792 / 0.846 | 0.025 | 0.951 |
| ridge from dense ctx state (λ = 1e3) | 3,584-dim raw state | 0.649 | 0.711 / 0.850 / 0.884 | 0.060 | 0.967 |
| composed, zero-fit (#1482 dense ridge → answer-SAE encode) | 3,584-dim raw state | n/a | 0.737 / 0.862 / 0.893 | 0.026 | 0.517 |
| index-aligned identity+bias (null) | ctx feature i → ans feature i | n/a | n/a | −0.000 | 0.500 |
| train-mean (null) | none | n/a | n/a | −0.000 | 0.500 |

Fit: 120,000 rows, λ chosen on 400 validation rows, 20,000 holdout rows, 20 row-shuffle null draws per feature (`controls.json`). The MLP's per-feature median is below the ridge's because it is negative on the rare deciles (−0.16 to −0.21 in deciles 2 to 4) while matching or beating ridge from decile 8 up (0.478 vs 0.461 in the top decile). 23,765 answer features never fire on holdout, so they have a firing census but no defined holdout R² (`n_census_only`).

### Edges (done)

Edge gate: top-20,000 |coef| in both halves → 15,626 replicated → 1,923 sign-consistent and above the per-column null (`edges/coef_structure.json`, `edges/wiring_edges.npz`). Column concentration: the top context feature carries a median 2.8% of a column's absolute mass (q90 10.7%), the top 10 carry a median 11.4%. Dashboard with every one of the 500 strongest edges rendered as both descriptions in words, the receipts flags, and the "unexpected edges" ranking (|coef| × (1 − cosine of the two descriptions)):

Rendered dashboard: https://raw.githack.com/superkaiba/explore-persona-space/b63448a250a80b78f6fc76c19b6f3fc88e25f43a/eval_results/issue_2661/dashboard/issue2661_dashboard.html

Every file below lives on branch issue-2661 at `b63448a250a`.

### Odd-correlation read (done)

Politically sensitive China context features: 49 (regex over the W1 descriptions: Taiwan, Xinjiang, Tibet, Tiananmen, CCP, Uyghur, Hong Kong, Xi Jinping, sovereignty, censorship, PRC). Landing of their 490 top-10 out-edges, classified by the answer feature's own description text: 38 refusal-described, 41 CCP-described, 411 other. Reverse direction: 181 answer features are described as refusals or deflections and receive 21,074 surviving in-edges, of which 115 (0.5%) come from the 49 China features. Refusal features are fed overwhelmingly by harmful-request, jailbreak and explicit-content context features (for example answer 12887 ← context 21000 "requests for toxic or hateful content" +0.044). Positive control: identity questions in five languages (context 10173, 15984, 3067, 6826, 6943) wire to Qwen self-identification answer features (29234, 21858, 14461, 23639 "denies being GPT") at up to +0.024.

Caveat that matters for reading the dashboard: the receipts "refusal" family in `receipts_answer_features.json` is noisy. Its three most-fed members (18385 short conversational statements, 27474 short acknowledgements, 15921 creative-writing collaboration) are not refusals and carry most of the 1,508 "receipts-flagged" edges in `watchlist_context_features.json`. Use the description-text read above for any refusal claim.

### Judged evals (Der pipeline, judge claude-opus-5 by user override) (all done)

| wave | items | valid | dropped (parse / truncation / refusal) | max_tokens | realized spend |
|---|---|---|---|---|---|
| w1 context-feature descriptions (25 top + 20 random firing prompts each) | 7,463 | 7,435 | 18 / 8 / 2 | 512 | $132.63 |
| w2 24-field user-prompt summaries | 2,000 | 1,983 | 11 / 3 / 3 | 2,048 | $21.44 |
| w4 10-way matching (summary → which of 10 feature lists) | 2,000 | 1,979 | 3 / 8 / 10 | 2,048 | $54.72 |
| pilots (w1, w2, w4 ×2) | 51 each | | | | $4.89 |

Every wave was pilot-gated (51 items, zero truncation and parse-fail < 2% before production). The w4 pilot failed at 1,024 (3 truncations) and passed at 2,048. Cumulative realized judge spend across all waves and pilots (duplicate w1 pilot and failed w4 pilot included): about $214.4 against the $300 estimate gate (re-estimated total upper bound $295.77).

10-way matching, in words: the judge reads one held-out prompt's 24-field summary and ten candidate lists of context-feature descriptions (the features that fired on that prompt, and on nine distractor prompts) and must name the matching list. 0.967 accuracy means the W1 descriptions carry enough prompt-specific content to identify the prompt nearly every time. Per-slot read: `judge_aggregates/w4_der_read.json`.

![Coverage by field](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b63448a250a80b78f6fc76c19b6f3fc88e25f43a/figures/issue_2661/coverage_by_field.png)

> Der coverage per summary field over 1,983 held-out prompts, top-1 (dark) and top-5 (light) mean cosine between the field text and the descriptions of the context features judged to fire on that prompt. Source: `eval_results/issue_2661/embedding_coverage.json`.

Overall coverage: top-1 0.669, top-2 0.660, top-3 0.653, top-5 0.643. Persona (0.81), intent and creativity (0.74) are best covered; tone (0.57), linguistic sophistication (0.58) and audience level (0.61) worst. No paper number is directly comparable: Der et al. report coverage for their turn-averaged answer SAE on assistant turns, this is the context SAE on user prompts (user decision in the Provenance record).

### Compute and provenance

Pod: pod-2661 (1x H100, RunPod AP-IN-1) 5.5 h for SAE training, encodes, maps, controls, per-feature reads, edges and mining (~$19), plus pod-2661-embed (1x H100) 35 min for the embedding phase (~$2). Judge model claude-opus-5 through the Message Batches endpoint. Scripts on branch issue-2661: `scripts/issue2661_flat_ctx_sae.py`, `issue2661_judge_waves.py`, `issue2661_embed_and_dashboard.py`, `issue2661_summary_figs.py`, `issue2661_w4_slot_read.py` (tip `b63448a250a`). Artifacts: HF `superkaiba1/explore-persona-space-data/issue2661_flatsae/` (analysis_tensors/{sae_ctx,sae_metrics,encodes,map_ridge,map_mlp,controls,perfeature,edges,eval_lists,embed}, raw_completions/{mining,judge}); 17.1 GB verified. Deviations from the plan: `eval_results/issue_2661/deviations.md` (16 entries, including the realized 20,000-row holdout in place of the 1,000 rows named above, scipy fp32 accumulation fixed with an fp64 helper, MLP target standardization, and the secret scrub of 87 pasted credentials in mined prompts before upload and judging).
