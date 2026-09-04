---
title: Flat Der-recipe context SAE and a full-dictionary context-feature to answer-feature
  map (odd-correlation edge read)
kind: experiment
tags:
- lean-run
- keep-running
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
