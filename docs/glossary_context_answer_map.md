# Glossary — prefix / query / context / answer (the context->answer mapping line)

Canonical vocabulary for the context->answer-map experiments (#779, #810/#658 line, and successors).
One first-use rule: **context = prefix + query** (the full model input).

## Text objects

| term | definition | notes |
|---|---|---|
| **prefix** | everything before the query: system prompt / persona / many-shot exemplar history / empty | the reusable, condition-defining part. The eval rig's "conditions" are its prefixes, so "within-condition r" = within-prefix |
| **query** | the user message / question | |
| **context** | the full model input: prefix + query, up to the assistant header | what generation conditions on ("the prompt" in API terms) |
| **answer** | one sampled completion for a context | decoding is stochastic (temp 1.0), so a context induces a *distribution* over answers; a **rollout** = one draw |

Degenerate case: bare user prompts (e.g. the LMSYS corpus) have an empty prefix — apart from the chat template's constant default system prompt — so there context ~= query.

## Vectors (all per layer, Qwen-2.5-7B: 28 layers x 3584 dims)

| symbol | name | definition |
|---|---|---|
| $v_C$ | **context vector** | activation at the last prompt token (the final newline of the assistant header) of ONE context |
| $v_P$ | **prefix vector** | a fixed prefix's context vectors averaged over many queries: $v_P = \bar{v}_C^{\,q}$ |
| $v_A$ | **answer vector** | mean activation over ONE answer's tokens (token-mean is intrinsic to $v_A$; no bar needed) |
| $\bar{v}_A^{\,q}$ | **behavior profile** | a prefix's answer vectors averaged over its queries (and rollouts) — the summary of the model's behavior under that prefix |
| $r_B$ | **trait direction** (persona vector) | per-trait direction from the mean-difference persona-vectors extraction (arXiv 2507.21509 recipe) |

## Aggregation modifiers

Mark averaging with an overbar naming the axis — "averaged" alone never says over what:

- $\bar{\cdot}^{\,q}$ — averaged over queries
- $\bar{\cdot}^{\,r}$ — averaged over rollouts (answer draws); e.g. single-draw targets $v_A$ vs rollout-averaged targets $\bar{v}_A^{\,r}$

## Maps

| name | fit rows | definition | where |
|---|---|---|---|
| **context map** $M'$ | one row per (context, answer) | $v_A \approx M' v_C$ | #779 (5000 LMSYS; 2400/trait trait-eliciting) |
| **prefix map** $M$ | one row per prefix, both sides query-averaged | $\bar{v}_A^{\,q} \approx M v_P$ | the earlier averaged experiment (#810/#658 line, 50 prefixes x 48 queries) |

Same measurement at two aggregation levels: $v_P$ is literally the query-average of a prefix's $v_C$'s, and the behavior profile is the query-average of its $v_A$'s.

## Read-out levels (monitoring)

- **context-level read-out** — monitor scored per single context (per-prompt monitoring)
- **prefix-level read-out** — monitor averaged over a prefix's queries (the grouped/persona-level monitoring result; much easier: hallucination r 0.09 per-context -> 0.53 at prefix level)

## Retired / ambiguous terms (do not use)

| term | why retired | use instead |
|---|---|---|
| "per-example map", "averaged map" | ambiguous about the averaged axis | context map / prefix map |
| "query-level map" | reads as the query-*marginal* (average over prefixes per query), a different, unbuilt design | context map |
| "query vector" | collides with attention Q of QKV; misdescribes rows that carry a persona | context vector $v_C$ |
| "exchange-level" / "instance-level" / "condition-level" | superseded coinages from the naming discussion | context-level / prefix-level |
| bare "averaged" | never says over what | overbar with the axis: $\bar{\cdot}^{\,q}$, $\bar{\cdot}^{\,r}$ |
