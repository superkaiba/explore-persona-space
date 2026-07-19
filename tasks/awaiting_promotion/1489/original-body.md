---
title: Context-augmentation effects on the context→answer-state transport map + dose-matched
  context-distillation in-weights comparison
kind: experiment
tags: []
created_at: '2026-07-18T01:29:02Z'
has_clean_result: false
parent_id: 1092
origin_prompt: 'Help me to design an experiment to test this: \subsection{Effect of
  Adding Information to Context on This Mapping} Experiment: add different kinds of
  information to the context (facts, instructions, personas, formatting constraints)
  and see how the mapping changes. || can we compare against finetuning on that same
  example? || also what is the most principled way to compare 2 mappings? || please
  run this in the background with happy coder'
workflow: v1
goal: Characterize how the pre-generation context→answer-state transport map h (c(x)→v(x);
  ridge primary, MLP secondary) changes when contexts are augmented with facts, instructions,
  personas, and formatting constraints — distinguishing movement within a fixed map
  from changes of the map (5×5 transport-transfer matrix), testing additivity/effective-rank
  of augmentation deltas, testing whether relevance gating is captured linearly —
  and whether delivering the same information in-weights via dose-matched context
  distillation produces the same per-example answer-state shift, preserves relevance
  gating, and leaves the pre-finetuning map valid on the finetuned model.
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Context-augmentation effects on the context→answer-state transport map, with a dose-matched in-weights (context-distillation) comparison arm

## Goal

Characterize how the pre-generation context→answer-state transport map h (c(x)→v(x); ridge primary, MLP secondary) changes when contexts are augmented with facts, instructions, personas, and formatting constraints — distinguishing movement within a fixed map from changes of the map (5×5 transport-transfer matrix), testing additivity/effective-rank of augmentation deltas, testing whether relevance gating is captured linearly — and whether delivering the same information in-weights via dose-matched context distillation produces the same per-example answer-state shift, preserves relevance gating, and leaves the pre-finetuning map valid on the finetuned model.

## Provenance

Originating user prompts (verbatim, in order, from the design chat 2026-07-17/18):

1. "Help me to design an experiment to test this: \subsection{Effect of Adding Information to Context on This Mapping} Experiment: add different kinds of information to the context (facts, instructions, personas, formatting constraints) and see how the mapping changes." (pivoted from an earlier ask about the pre-generation prediction subsubsection)
2. "can we compare against finetuning on that same example?"
3. "also what is the most principled way to compare 2 mappings?"
4. "please run this in the background with happy coder"

Design was grounded in-chat against: the theory paper (Overleaf `6a2df2d2` `main.tex` §Definitions/§Assumptions A4–A5/§Evaluation methodology), a project inventory of the predictor line (#658/#742/#761/#763/#823/#952/#1092), and two arXiv-MCP literature passes (prior-art lists below).

## Background

The transport map under study is the pre-generation context→answer-state map `h: c(x) → v(x)` — context-side activation summary to answer-side mean-token activation — the object fit in #823/#952/#1092 and posited by the theory paper (Assumptions 4–5) as a fixed property of the model, with linear special case `h(c) = Mc`. The leakage predictor applies one fixed `h` across context conditions; real context conditions differ exactly by added facts/instructions/personas/formatting. This experiment measures the map's domain of validity under those four augmentation kinds, and adds an in-weights arm testing whether finetuning on the same information produces the same movement in the same coordinates — the mechanistic bridge behind predicting fine-tuning-induced leakage from pre-fine-tuning context geometry.

Constraints inherited from the line: method attribution is unidentifiable at n=50 contexts (#763 co-fit: K=100 random directions reach ρ 0.62–0.85), so the substrate must be the large realistic pool; #1092 showed context-based reads carry nearly all transport (held-out R² 0.74–0.81 vs 0.05–0.11 prefix-based) with matched-target scoring.

## Three separable questions

1. **Movement within vs change of the map.** Does `M_plain` (ridge fit on unaugmented contexts) still transport augmented contexts `c(x⊕k) → v(x⊕k)`, or does each augmentation family need its own refit? Headline quantity: gap between plain-map held-out R² and refit R², per family (the 5×5 transport-transfer matrix).
2. **Shift geometry.** Is the effect of augmentation `k` an approximately constant, low-rank shift `Δc(x,k) ≈ μ_k` across base contexts (task-vector / function-vector structure, extended to facts/personas/formatting)? Does the map commute with it: `M μ_k ≈ mean Δv_k`? Registered quantity: effective rank of the augmentation delta per family (prior: instructions/personas ≈ rank-1; facts plausibly higher-rank — cf. the task-vector rank limits result, arXiv 2506.09048).
3. **Relevance gating — the linearity falsifier.** A fact shifts answers only for queries it is relevant to. If `Δc` is roughly constant across queries but `Δv` is relevance-gated, a linear `h` cannot capture it and an MLP must close the gap — a direct test of the linear special case. If `Δc` is itself gated (context activations already encode relevance — the CAST result, arXiv 2409.05907, makes this the default hypothesis), linearity survives. Either outcome is a finding.

Plus the in-weights arm (below): 4. per-example equivalence of `Δv_FT` and `Δv_ctx`; 5. gating transfer in-weights vs in-context; 6. pre-FT map validity on the finetuned model.

## Design

**Model:** Qwen2.5-7B-Instruct (base of the line).

**Substrate:** ~2k realistic base contexts subsampled from the #1092 WildChat/LMSYS crossing (stratified by topic/family), reusing its capture rig and ridge harness.

**Augmentation library:** 4 families × ~4 instances, fixed placement (appended to the system prompt; placement is deliberately NOT an axis in round 1):

| Family | Instances (sketch) | Designed contrast |
|---|---|---|
| Facts | short factual statements; half constructed to be RELEVANT to a known query subset, half generic | the gating variable |
| Instructions | refuse-topic-X, always-hedge, be-concise, agree-with-user | behavior-changing; existing judges reusable |
| Personas | 4 from the existing 275-role bank | reuses persona-vector extraction infra |
| Formatting | JSON-only, bullet-only, all-lowercase, fixed template | changes answer FORM wholesale while barely changing content — predicted worst case for transport (v is an answer-token mean) |

Binary presence/absence in round 1; graded dose (stacked instructions, fact placement) is a follow-up round only if additive structure holds.

**Capture:** per (x, k) and plain x: on-policy answer (vLLM, ≥1024 tokens, decoding recipe matched to #1092 for comparability), then one teacher-forced forward capturing `c` (last-token AND mean-pooled variants) and `v` (answer-mean), layer-swept.

**Both mapping arms (standing rule, both reported):** every mapping/probe read is computed BOTH prefix-based (prefix = everything before the user query — the augmentation lives here, so the prefix arm is expected to carry real signal in this design) AND context-based (prefix + user query). Matched-target scoring throughout (#1092 lesson): every arm scored against the same targets on the same rows.

**Analyses:**
1. 5×5 transport-transfer matrix (plain + 4 families): held-out R² of `M_A` applied to family B, grouped folds. Row "plain" = generalization; diagonal = refit ceiling.
2. Shift decomposition: per k, fraction of Var(Δc) explained by μ_k and top PCs; same for Δv; relative error of `M μ_k` vs mean Δv; rank of {μ_k} within/across families. Last-token vs mean-pooled `c` compared explicitly (prompt effects are token-nonuniform — arXiv 2605.03907); #952's per-token rig is the fallback instrument if summary-level reads are ambiguous.
3. Gating test (facts family): relevant vs irrelevant query subsets — gating effect size in Δv vs Δc; linear-vs-MLP `h` gap on exactly this family.
4. Manipulation checks (required): judged compliance per family (did the instruction/persona/format actually change answers; did relevant queries actually use the fact) — graded 0–100, ≥5 draws, judge `claude-sonnet-4-5-20250929`, Batch API, on a subset per family. An augmentation instance that fails its manipulation check is excluded from headline cells and reported.

## In-weights comparison arm (context distillation)

**Construction:** for a subset of augmentations (2 instances per family = 8), finetune the plain model via context distillation: LoRA on pairs (plain context x_train, answer generated under x_train⊕k), training queries disjoint from eval queries. This is the maximally matched "same information delivered in-weights"; raw-form injection (SFT on fact text / instruction demos) is deliberately excluded from round 1 as a less-matched construction. The distillation mix over the full query distribution structurally contains the contrastive-negatives requirement: irrelevant queries produce unchanged answers = same-question negative rows (`.claude/rules/contrastive-negatives.md`); distillation targets are on-policy by construction (`.claude/rules/on-policy-completions.md`).

**Dose matching (load-bearing):** train past target and select the checkpoint whose judged manipulation-check compliance MATCHES the in-context arm's (dose-to-target per the #612 standard). Fixed-epoch comparison confounds mechanism with dose and is not acceptable for the headline.

**Reads:**
- (4) Per-example equivalence: cosine/R² between `Δv_FT(x) = v_θk(x) − v_θ0(x)` and `Δv_ctx(x) = v_θ0(x⊕k) − v_θ0(x)`, distribution over held-out x, per augmentation and family. Prior work only showed average-projection agreement onto one direction (persona vectors FT-shift, r=0.76–0.97); per-example, cross-kind equivalence is untested.
- (5) Gating transfer: does fact-relevance gating survive in-weights, or does the finetuned fact fire on irrelevant queries (uniform leak)? Prior from the project's leakage line: weights leak broader than context.
- (6) Map stability under FT: does the pre-FT `M_plain` still transport `c_θk(x) → v_θk(x)`? This is the assumption the pre-FT leakage predictor requires; untested directly.

A steering delivery arm (add μ_k as an activation-steering vector, completing prompting/steering/finetuning) is a named round-2 candidate, not round 1.

## Map-comparison methodology (registered analysis method)

Maps are compared as FUNCTIONS on a common reference input set, never as parameter matrices (parameter norms are dominated by ridge-unidentified directions and undefined for the MLP variant):

1. Fixed reference input distribution for every comparison (evaluating each map on its own training distribution confounds map change with input shift — the matched-target failure mode).
2. Two functional statistics, because disagreement ≠ consequence: (a) normalized disagreement `E_x ||M_A c(x) − M_B c(x)||²` scaled by output variance; (b) consequence — the transfer-R² matrix plus disagreement projected through behavior readouts (`r_B^T M_A c` vs `r_B^T M_B c`).
3. Split-half refit null: within-distribution refit distance is the noise floor; the "map changed" statistic is cross-distance in excess of that null, bootstrap CIs over base contexts. Both maps fit at matched effective degrees of freedom (same λ-selection procedure).
4. Mechanistic decomposition (secondary): `ΔM Σ_c^{1/2}` — effective rank, principal angles between dominant singular subspaces, fraction of `ΔM`'s action explained by the μ_k shift directions (tests the low-rank-write account).
5. Anti-method, recorded: CKA/Procrustes/shape-metric comparisons are NOT used — both maps share the same residual-stream coordinates of the same model, so alignment invariance discards exactly the differences under measurement. They become appropriate only for a future cross-model comparison.

## Statistics discipline

- Folds GROUPED BY BASE CONTEXT: augmented variants of the same base query never straddle train/test (the leakage trap specific to this design). Group-level folds per `.claude/rules/ood-generalization-folds.md`.
- Permutation nulls by shuffling augmentation labels; bootstrap CIs over base contexts; selection-symmetric nulls wherever a layer/variant is selected (`.claude/rules/selection-symmetric-nulls.md`).
- Reliability from repeat draws on a subset (aligned split-half, llm-judging rule 21) reported as the ceiling on judged quantities.
- Judged DVs follow `.claude/rules/llm-judging.md` (graded 0–100 primary, N≥5 draws, drop-never-coerce, transport-error retry, rubric-keyed caches, max_tokens ≥ ~300 for reason-then-score).
- Predict-the-mean baselines on every predictive read.

## Predictions on record (competing hypotheses)

- H-additive: personas/instructions shift `c` approximately additively (high variance-explained by μ_k, ≈rank-1); facts higher-rank.
- H-fixed-map: `M_plain` transports persona/instruction augmentations with modest loss; formatting is the largest transport failure (answer-form shift dominates v); facts have the smallest main effect but are gated.
- H-gating-in-c (default, per CAST): relevance gating is already present in `Δc`, so linear `h` survives the facts test; the discovery outcome is gating appearing only in `Δv` (linearity falsified, MLP closes the gap).
- H-weights-leak-broader: in-weights delivery reproduces the mean shift but loses relevance gating (leaks to irrelevant queries) relative to in-context delivery at matched dose.
- H-map-survives-FT: `M_plain` transports the finetuned model's contexts with a gap small relative to the augmented-context transport gap.

## Prior art (from two arXiv-MCP passes; planner should verify ids when grounding)

- Additive context-manipulation vectors: task vectors (2310.15916), function vectors (2310.15213), in-context vectors (2311.06668), persona vectors (2507.21509), BILLY (2510.10157). Caveats: prompt steering is token-nonuniform (2605.03907); task vectors provably fail on high-rank mappings (2506.09048).
- Map/probe stability under context composition: NO prior work fits an explicit context→answer-state transport map on plain contexts and tests transfer to augmented contexts — the cleanest novelty claim. Nearest: steering-vector robustness across prompt variations (2602.17881); FV robustness to context changes (2310.15213); probes transfer poorly across tasks (2410.02707).
- Relevance gating: CAST (2409.05907) shows the gating condition is linearly decodable from context activations; SHIFT (2606.27786) gates context-vs-parametric knowledge under RAG conflict; no work tests whether the gated shift itself is linear/low-rank.
- In-weights vs in-context: context distillation constructions (2112.00861, 2209.15189); persona-vectors FT-shift average projection (2507.21509); persona features under prompting/steering/finetuning (2506.19823). Per-example, cross-kind, same-coordinates equivalence + pre-FT map validity on the finetuned model: untested (third lit pass pending at filing time; fold results into the plan).

## Reuse (fitness checks per `.claude/rules/artifact-reuse.md` at plan time)

- `scripts/issue1092_fit_grid.py` — ridge fit engine, prefix/context arms, PRESS-λ, grouped folds.
- `src/explore_persona_space/analysis/issue_763_cofit.py` — multi-family LOCO co-fit harness (batched, reference-checked).
- `src/explore_persona_space/analysis/vectorized_mlp_skill.py` — batched LOCO MLP (for the linear-vs-MLP gating test). All many-cell fits batched via Gram/dual (`.claude/rules/vectorize-many-cell-fits.md`); no serial per-cell factorizations (#823 lesson).
- `src/explore_persona_space/analysis/issue_742_decoding_ceiling.py` — reliability ceiling machinery.
- `scripts/extract_persona_vectors.py` + 275-role bank; faithful PV r_B extractors (`issue658_extract_rb_personavectors.py`, `issue763_extract_pv_rb.py`) for readout-projected comparisons.
- HF stores: `issue1092_realistic_crossing` (context pool + captures), `issue823_own_vs_external`, `issue658_theory_assumptions`.
- `#952` per-token-position rig — fallback instrument for the token-nonuniformity control.

## Compute estimate (planner to refine in §9)

- Context arms: ~2k base × (1 + 16) ≈ 34k generations + teacher-forced captures ≈ 15–25 GPU-h (shardable wide on GCP; declare `--gpus N`).
- FT arm: 8 LoRA context-distillation runs (small mixes, dose-to-target checkpointing) + eval generations/captures on ~1k contexts × 8 models ≈ 15–25 GPU-h.
- Judging: Batch API subsets per family; fits CPU/batched. Total ≈ 40–50 GPU-h (above the auto-run band; full planner gate applies).

## Open questions anchor

`q:spec-context-as-vector` (context-vector line; #823/#952/#1092 lineage); secondary relevance to `q:leak-predictor` via reads 4–6.
