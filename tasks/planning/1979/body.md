---
title: Per-prefix leakage-predictor race + theory-assumption battery at prefix grain
  (50 prefixes x 60 shared queries)
kind: experiment
tags: []
created_at: '2026-08-01T07:47:20Z'
has_clean_result: false
parent_id: 1900
origin_prompt: i want to test all these [full predictor roster incl. mediation checkbox]
  as well as the theory assumptions, at the per-prefix level (leakage averaged across
  queries) - RESULTS THAT RAN WERE PER QUERY. Design an experiment to do this
workflow: v1
goal: 'At per-prefix grain (leakage per (arm, destination prefix), averaged over a
  fixed shared 60-query set; 50-60 prefix panel including trained prefixes, trained
  contrastive negatives, bystanders, near-twins, real conversation and ICL prefixes),
  determine which pre-fine-tuning predictor best predicts per-prefix leakage level
  and change (full roster: context/answer similarity pre+post, delta similarities,
  through-map forms, whitened gate, r_B projections, write-map, propensity incumbent,
  kNN answer-side variant; both anchors), whether context similarity is answer-mediated
  at this grain, and whether the #1768 assumption refutations (gate A7, write rank
  A6, write direction A5) survive at the theory''s native prefix grain or were per-query
  grain artifacts.'
relates_to:
- leak-predictor
- spec-context-as-vector
- identity-contextual-vs-base
---
# Per-prefix leakage-predictor race + theory-assumption battery at prefix grain: 50-prefix × 60-query panel, leakage averaged over queries

## Provenance

- origin_prompt (verbatim): "i want to test all these: RESULTS THAT RAN WERE PER QUERY WHICH IS TERRIBLE - What is the best leakage predictor: [context/answer vector cosine similarity pre/post finetuning; similarity between change in context/answer vector pre and post finetuning; context vector -> apply mapping -> predicted answer vector similarity pre/post finetuning; Is leakage due to similarity of mean answer vectors and so context vector similarity is just byproduct; Any other suggestions?] As well as the theory assumptions. At the per-prefix level. Design an experiment to do this"
- Parent: #1900 (the per-query race — grain extension changes its Goal, hence a child). Relates to: #1768 (assumption battery — this re-tests A5/A6/A7 at the theory's NATIVE grain), #722 (the original per-prefix query-averaged mapping objects v_P / v_A), #1481 (bystander panel precedent).
- Standing conventions binding here: context vector = LAST-TOKEN (the newline ending the assistant header) PRIMARY with span-mean secondary, captured together (user directive 2026-08-01); contrastive negatives / marker recipes N/A (no training); LLM judge = claude-sonnet-4-5-20250929, Batch API, graded 0-100 multi-draw.

## Goal

At per-prefix grain (leakage per (arm, destination prefix), averaged over a fixed shared 60-query set; 50-60 prefix panel including trained prefixes, trained contrastive negatives, bystanders, near-twins, real conversation and ICL prefixes), determine which pre-fine-tuning predictor best predicts per-prefix leakage level and change (full roster: context/answer similarity pre+post, delta similarities, through-map forms, whitened gate, r_B projections, write-map, propensity incumbent, kNN answer-side variant; both anchors), whether context similarity is answer-mediated at this grain, and whether the #1768 assumption refutations (gate A7, write rank A6, write direction A5) survive at the theory's native prefix grain or were per-query grain artifacts.

## Design sketch (planner refines)

- **Prefix panel (the units): N≈50–60 destination prefixes**, deliberately structured: each arm's own TRAINED prefix (on-target anchor row); the trained contrastive-negative personas (REGISTERED PREDICTION: leakage below geometry prediction — they were trained DOWN; a signed residual check); the #1481 bystander panel contexts; #722 battery prefix families; real WildChat conversation prefixes; ICL prefixes; near-twin personas of each source; the bare default. Rendered via the artifacts CONTEXTS registry with the round-3 (#1768 pfx0) ground-truth prefix asserts (the trained prefix must byte-match the training mix rows).
- **Query set: fixed ~60 real-user queries** (stratified from the 16,400-prompt corpus, sha-pinned), SHARED across every (arm × prefix) — fully paired design; token-budget assert per prefix (prefix + query + max_new_tokens ≤ MAX_MODEL_LEN).
- **Arms:** the #1900 panel (12 content + 6 marker, reused checkpoints) for cross-grain comparability; race is WITHIN-ARM ACROSS PREFIXES (dose-clean), n=50 units per arm.
- **Captures per (model-state × prefix × query):** on-policy greedy generation (trained arms + base) with raw text persisted; activations: context vector (BOTH positions: assistant-header-newline last-token primary, span-mean secondary) + answer span-mean, layers 14/19/25; PLUS a matched-text TF tree (trained model on the base rows) for the weights-carried per-prefix writes the assumption battery needs. ~19 model states × 50 × 60 ≈ 57k generations + TF passes — vLLM, arm-sharded.
- **DVs per (arm, prefix), all query-averaged:** content = mean graded judge score (level) + trained−base mean (change) + binary-rate companion; marker = mean Δ log P(marker) at slot (judge-free, three-space storage). Per-prefix base propensity = the base state's mean judged score (incumbent predictor + covariate).
- **Predictors (one number per (arm, prefix); vectors query-averaged over the shared 60; centered at panel mean; both anchors):** the verbatim user roster P1–P8 (ctx-sim pre/post, ans-sim pre/post, Δctx-sim, Δans-sim (matched-text form in the mechanistic panel; on-policy form labeled near-outcome), through-map pre/post) + additions: whitened gate similarity (dual-use with A7), r_B projection direct + through-map, cross-arm write-map prediction, per-prefix base propensity, kNN answer-side variant (mean cos to nearest individual training-answer rows — the un-averaged anchor form #1900 never raced), nearest-training-rows context-side. Deployable vs mechanistic split as in #1900 (post-FT and Δ candidates explain, never carry the headline).
- **Mediation (the checkbox):** partial Spearman lattice at prefix grain (leakage ~ ctx-sim | ans-sim and reverse, both | propensity), commonality decomposition, and the structural through-map read — n=50/arm makes partials well-posed.
- **Assumption battery on the same stores:** A7 gate = whitened similarity g(prefix) vs realized per-prefix matched-text write coefficient (and on-policy secondary), per arm × layer, against the 0.3–0.7 band; A6 = top-1 SVD share of the 50-row per-prefix write matrix (both trees) vs the 0.6 criterion, bridging panel-era 0.81–0.86 and corpus per-query 0.09/0.29; A5 = cos(per-prefix write, δ) and (r_B / marker unembedding) with norm-matched nulls + the disjoint-half baseline discipline from issue1768_directions.py. Map-change D per prefix is OUT of scope (needs per-condition fits at n≈3k rows — that is round 3's instrument); stated as a scope boundary.
- **Stats:** within-arm Spearman across prefixes; bootstrap over prefixes + query-cluster bootstrap (queries are shared, so prefix-level noise is query-clustered); split-half reliability over queries per prefix (design-aligned halves); winner-selection inside every bootstrap draw (selection-symmetric); permutation null band. Both-arms mapping rule: with 50 distinct prefixes the prefix-based mapping arm is finally IDENTIFIABLE — run both arms properly (a first for the line).
- **Reuse:** #1900 predictor builders + race machinery (re-grained); #1768 round-3 prefix render/registry + assert machinery (pfx0), direction-read conventions, δ (delta_tf) + r_B tensors; #722 prefix families; #1481 verdict manifest + panel contexts; judge rubrics sha-pinned from #1900.

## Cost sketch

Generations + TF captures ≈ 115k rows total → ~15–25 GPU-h (vLLM, multi-GPU arm-sharded); judge ≈ 13 judged states × 50 × 60 × 3 draws ≈ 120k Batch-API calls (max_tokens 400, reason-then-score); fits/reads are CPU/light-GPU. Under the 100 GPU-h autonomous cap.

## Constraints

- kind: experiment; /adversarial-planner before execution.
- The per-query #1900 verdicts stay valid as the harsh-grain read; this task's claims are grain-scoped and the clean-result must state both grains' verdicts side by side where they differ (especially the gate).
- Parallelize aggressively; Batch API for all judging (standing user directive).
