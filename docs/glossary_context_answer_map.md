# Glossary — prefix / query / context / answer (the context->answer mapping line)

Canonical vocabulary for the context->answer-map experiments (#779, #810/#658 line, and successors).
One first-use rule: **context = prefix + query** (the full model input).

Scope: this glossary governs the context->answer mapping line. The
marker-leakage line defines **context (C)** differently (everything before
the trained assistant turn, ending at a question — see
`docs/open_questions.md` § Glossary); do not mix the two vocabularies.
Writing conventions for the ad-hoc result summaries that use this vocabulary
(analysis-choice grounding, per-arm provenance) live in
`docs/results_summaries/README.md`.

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
| — (no symbol in use; write it out) | **prefix-end state** | residual-stream activation at the last prefix token, BEFORE the query begins; exactly constant across a prefix's queries (within-prefix std ratio 0.000, #1092). NOT the same object as $v_P$ — see § Maps |
| $v_A$ | **answer vector** | mean activation over ONE answer's tokens (token-mean is intrinsic to $v_A$; no bar needed) |
| $\bar{v}_A^{\,q}$ | **behavior profile** | a prefix's answer vectors averaged over its queries (and rollouts) — the summary of the model's behavior under that prefix |
| $r_B$ | **trait direction** (persona vector) | per-trait direction from the mean-difference persona-vectors extraction (arXiv 2507.21509 recipe) |

Theory-paper notation: the leakage-theory paper (`docs/leakage_theory_paper.tex`;
Overleaf `main.tex`) writes $c_x$ for the per-context vector and
$c_C = \mathbb{E}_{x\sim C}[c_x]$ for the condition-level vector — i.e. $v_C$
and $v_P$ here. The paper deliberately leaves the per-context featurization
open (its example is a mean prompt-side activation); the experimental line
pins the last-prompt-token state.

## Aggregation modifiers

Mark averaging with an overbar naming the axis — "averaged" alone never says over what:

- $\bar{\cdot}^{\,q}$ — averaged over queries
- $\bar{\cdot}^{\,r}$ — averaged over rollouts (answer draws); e.g. single-draw targets $v_A$ vs rollout-averaged targets $\bar{v}_A^{\,r}$

## Maps

| name | fit rows | definition | where |
|---|---|---|---|
| **context map** $M'$ | one row per (context, answer) | $v_A \approx M' v_C$ | #779 (5000 LMSYS; 2400/trait trait-eliciting) |
| **prefix map** $M$ | one row per prefix, both sides query-averaged | $\bar{v}_A^{\,q} \approx M v_P$ | the earlier averaged experiment (#810/#658 line, 50 prefixes x 48 queries) |
| **prefix-end map** | one row per prefix | pooled answer profile ≈ ridge(prefix-end state) | #1092 (1,145 real WildChat/LMSYS prefixes, sparse-crossed with a 1,397-query bank — 21,193 rows; 996 prefixes survive at averaged grain) |

Same measurement at two aggregation levels: $v_P$ is literally the query-average of a prefix's $v_C$'s, and the behavior profile is the query-average of its $v_A$'s.

The prefix-end map is a THIRD object, not an aggregation grain of the context
map: at averaged grain it reaches R² 0.37–0.53 vs 0.82–0.94 for the
query-averaged context map (instruct model, own-corpus cell; each pair is the
range across the ambient/PCA-48 representation bases, not an uncertainty
interval; the base-model cell runs lower, 0.26–0.46 vs 0.76–0.88), and the two
maps' predictions agree at only R² 0.28–0.54 (the source docs' headline range —
the prefix-prediction→context-prediction direction, across both model cells and
both bases; the strict all-cells-both-directions floor dips to 0.08, 0.077 in
the 07-17 deep-dive table) — the direct test says the prefix-end map is NOT the
query-averaged context map (#1092 fair comparison; Result 7 of
`docs/results_summaries/2026-07-15-link-prefix-context-answer-maps.md`,
deep-dive `docs/results_summaries/2026-07-17-fair-comparison-deep-dive.md`).
The earlier ~0.8 "prefix map" result (#810/#658 line) belongs to the
query-averaged object $v_P$, not the prefix-end state.

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
| bare/unqualified "prefix vector" / "prefix map" in #1092-era prose | after #1092 there are two distinct prefix-side objects (the query-averaged $v_P$ and the prefix-end state), so unqualified use is ambiguous — a transcription of the split the sources already made (Result 7 of the 2026-07-15 summary; 07-17 deep-dive), not a fresh deprecation ruling. The Maps-table rows above are NOT retired: "prefix map" $M$ sits next to its formula ($\bar{v}_A^{\,q} \approx M v_P$), which qualifies it | qualify: "prefix vector $v_P$ (query-averaged)" vs "prefix-end state" / "prefix-end map" |
| bare/unqualified "context vector" / "context map" in #1768/#1900-era prose | the POOLING POSITION was not fixed across rounds. The § Vectors row pins $v_C$ at the LAST PROMPT TOKEN (the final newline of the assistant header), but #1768 round 1 and #1900 pooled the prompt SPAN-MEAN while calling it the context vector. The choice is load-bearing, not cosmetic: #1768's re-pool round held rows, splits, penalty grid, floors and seeds fixed and still flipped 23 of 216 cell verdicts, raised base-map held-out R² by ~0.20 at every layer, and moved relative context movement from 0.025 (span-mean, layer 19) to 0.237–0.267 (last token, layers 14/19/25); #1947 consequently captures both (`context_summary_primary: last_prompt`). The § Vectors and § Maps rows are NOT retired: $v_C$ sits next to its position and $M'$ next to its formula, which qualify them | qualify the pooling: "context vector $v_C$ (last-prompt-token, the newline before the assistant answer)" vs "context vector (prompt span-mean)"; declare it PER VECTOR at plan time (the `.claude/agents/planner.md` §6 pooling-convention row, #1974) |

### Search-time note — retired aliases stay grep targets

The table above governs WRITING (do not use these terms). When SEARCHING
for prior work, do the opposite: old task bodies, events, and follow-up
labels still carry the retired vocabulary (e.g. #813's follow-up label
`per-example-vs-averaged-map`), so grep the aliases alongside the
canonical terms, separator-tolerant (`[-_ ]` between words):
`per[-_ ]example`, `averaged[-_ ]map`, `question[-_ ]averaged`,
`query[-_ ]averaged`, `single[-_ ]context`, `query[-_ ]level`,
`exchange[-_ ]level`, `instance[-_ ]level`, `condition[-_ ]level`,
`prefix[-_ ]vector`, `prefix[-_ ]map`, `context[-_ ]vector`,
`context[-_ ]map` (ambiguous-era usage; the qualified
forms remain canonical), plus the pooling names `span[-_ ]mean`,
`last[-_ ]token`, `final[-_ ]token`, `last[-_ ]prompt[-_ ]token`
(for locating which convention ambiguous-era prose actually used).
This list is not closed: as the retired-terms table above gains rows,
derive their grep patterns from the table and add them here.
Full recipe: `.claude/agents/research-pm.md` § Negative-existence claims.
