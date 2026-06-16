---
title: 'Query influence on the context-conditioned residual stream: per-layer context-end
  vs query-end last-token similarity (Qwen-2.5-7B-Instruct)'
kind: experiment
tags: []
created_at: '2026-06-16T22:20:03Z'
has_clean_result: false
origin_prompt: check how similar activations are after the context vs after the user
  query (for a variety of contexts and user queries), ask clarifying questions
goal: Measure, per layer in Qwen-2.5-7B-Instruct, how much appending a user query
  displaces the last-token residual-stream state away from the context-established
  state (centered cosine + CKA + shuffled-pair floor) across persona / generic-instruction
  / ICL / wild-chat contexts and varied queries, to characterize context-dominant
  vs query-dominant layer regimes.
---
## Goal

Measure, per layer in Qwen-2.5-7B-Instruct, how much appending a user query displaces the last-token residual-stream state away from the context-established state (centered cosine + CKA + shuffled-pair floor) across persona / generic-instruction / ICL / wild-chat contexts and varied queries, to characterize context-dominant vs query-dominant layer regimes.


## Construct and measurement

**Construct — "query influence":** the degree to which appending the user
query moves the model's internal (last-token residual-stream) state away from
the state established by the preceding context, as a function of layer depth.
The complement of similarity. A *context-dominant* layer is one where the
query-end state stays close to the context-end state (the query barely moves
it); a *query-dominant* layer is one where it does not.

**Model:** Qwen-2.5-7B-Instruct (HF transformers, `output_hidden_states=True`,
ChatML via `apply_chat_template`). Base model, no fine-tuning.

**Primary read (matches the chosen positions — last token of each span).**
In a single forward pass over the full ChatML prompt
`[system: <context>] [user: <query>] [assistant-generation marker]`, for each
layer L (0–27) extract the residual at:
- (a) the **last token of the context span** (end of the system block), and
- (b) the **last token of the query span** (last prompt token before
  generation).

Quantify the (a)↔(b) relationship per layer with:
- **per-pair centered cosine** — global-mean-subtracted, the canonical
  persona-distance form (`compute_cosine_matrix(centering="global_mean")`);
- **CKA per layer** — between the bank of context-end residuals and the bank
  of query-end residuals across all (context, query) pairs (does the
  context-induced geometry survive to the query position?); **new code** —
  no CKA implementation exists in the repo yet;
- **raw per-layer cosine** — reported alongside, with the anisotropy caveat.

**Floors / baselines (separate signal from token-identity + anisotropy):**
- **shuffled-pair cosine** — context-end of pair *i* vs query-end of pair *j*
  (*i*≠*j*): the token-identity/position floor that any two different-token
  residuals share;
- report similarities **relative to** this floor, not in absolute terms.

**Companion read the planner should consider (cleaner query-influence
operationalization, flagged as a control or co-primary):** fix the readout
position (assistant-generation position) and compare its state under
`[context]` vs `[context + query]` — a same-position with-vs-without-query
contrast, which removes the different-token confound. This is the
"How Context Shapes Truth" (2601.06599) recipe applied to query-vs-context.

## Competing hypotheses

- **H1 (early-context, late-query):** early layers preserve the context state
  (high similarity above floor), mid/late layers integrate the query
  (similarity drops toward floor). Consistent with ICL task-vector
  (2310.15916) and context-shapes-truth (2601.06599) layer-band findings.
- **H2 (uniform query dominance / no persistence):** once the anisotropy floor
  is subtracted, the query-end state is no more similar to its own context-end
  state than to a random context's — token identity dominates, no meaningful
  context persistence at the last-token position.
- **H3 (context-type-dependent):** persona contexts persist into the query-end
  state more (or less) than generic-instruction / ICL / wild-chat contexts.

## Factors and data

- **Context type (4 levels — user-requested):**
  1. **Persona system prompts** — `personas.py::PERSONAS` (13 + specialty) +
     `ASSISTANT_PROMPT`.
  2. **Generic instructions** — task-style system prompts.
  3. **ICL example sets** — in-context demonstration blocks (not centralized
     in the repo; built per-experiment, e.g. #602).
  4. **Wild-chat / real chat** — **not in the repo**; candidate source
     WildChat (`allenai/WildChat-1M`), using a real user/system turn as the
     context. Tier-1 real-world data; planner picks the exact slice.
- **Query type ("variety of user queries"):** vary on-topic vs off-topic to
  the context, and short vs long. Reuse `personas.py::EVAL_QUESTIONS` (20
  generic Qs) as a starting query bank; add off-topic + length variants.
- **Layers:** full 0–27 sweep (per #648) for the curves; canonical
  [7,14,21,27] as the reporting anchors.

## Existing tooling and gaps (codebase grounding)

Reusable:
- `src/explore_persona_space/analysis/probes.py::extract_residual_stream_activations()`
  — position-configurable last-token, all layers, `(n_prompts, n_layers, hidden)`.
- `src/explore_persona_space/analysis/representation_shift.py::compute_cosine_matrix()`
  — centered cosine (`centering="global_mean"`), canonical.
- `extract_centroids_response_mean()` (same file) — response-mean recipe if a
  pooled companion read is added.
- Model-loading + ChatML pattern is standard across the repo.

New code needed:
1. **CKA** (Gram-matrix centering + trace alignment) — none exists.
2. **Dual-position extraction in one forward** — current extractor takes one
   position per pass; add context-last + query-last (+ optional readout) in a
   single pass to avoid cross-forward confounds.
3. **Wild-chat data integration** (WildChat pull) if that context tier is kept.

Templates: tasks #602 (multi-layer extraction), #648 (full-depth layer sweep),
#553 (variance decomposition), #536 (bank-centering audit).

## Redundancy check

No prior task measures context-end vs query-end last-token displacement.
Closest (#602/#648/#553/#536) read residuals across layers but for
persona/marker channel questions, not the context-vs-query position question.

## Literature positioning

- **No paper measures this exact quantity** (per-layer cosine + CKA
  displacement of the last-token state caused by appending the query, framed
  as context- vs query-dominant regimes).
- Two papers to position against (same site / same method family):
  - *As X, Do Y: How Persona and Task Combine in Instruction-Tuned LLMs*
    (2605.23147) — same prompt-to-answer site, same Qwen-Instruct family;
    additive-decomposition framing (not a displacement metric).
  - *How Context Shapes Truth* (2601.06599) — per-layer directional + magnitude
    shift with/without context; truth vectors, not query-vs-context.
- *In-Context Learning Creates Task Vectors* (2310.15916) — context computed
  early, applied to the query late: the layerwise context/query split.
- **Persona Vectors (2507.21509) own position ablation:** the last-prompt-token
  read is the *weakest* for persona content (they use response-averaged) — a
  known caveat for our last-token choice, and a reason centered cosine + CKA
  (anisotropy-robust) are the right metrics.
- Anisotropy caveat (2401.12143 etc.): raw cosine sits at an inflated baseline;
  mean-centering + CKA correct for it.

*(Several 2025–2026 arXiv ids above are from a targeted scan and must be
verified by the planner's fact-checker before they enter any write-up.)*

## Provenance

Originating user request (verbatim): "check how similar activations are after
the context vs after the user query (for a variety of contexts and user
queries), ask clarifying questions".

Clarifying answers (2026-06-16):
- **Positions:** last token of each span (context-end vs query-end).
- **Context types:** persona system prompts; generic instructions; ICL
  examples; wild-chat / real chat examples (look at past issues for data).
- **Metrics:** per-layer cosine, CKA, centered/mean-ablated cosine.
- **Goal:** query influence (context-dominant vs query-dominant regimes).
