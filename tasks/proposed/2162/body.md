---
title: 'Which kinds of context information are carried at the context vector: a patch-only
  sweep over 21 minimal-pair information types, crossed with route conflict, recency,
  and load'
kind: experiment
tags: []
created_at: '2026-08-07T06:43:49Z'
has_clean_result: false
parent_id: 2094
origin_prompt: 'Help me to plan this issue based on the previous causality experiment
  (2094): Motivation - We''ve found this mapping from context -> answer vector; We
  found it to be mostly persona related; We want to see what kinds of information
  get stored at this context vector. Methodology - Activation patch ONLY the context
  vector for a wide range of contexts which only differ in one aspect: some fact in
  the context (user''s name, assistant''s favorite animal), some instruction (e.g.
  ''answer in bullet points''), ICL example of instruction, persona (prompted + ICL),
  query, what else? - See which affect outputs. Results - Result 1: What does patching
  the context vector affect? To measure the effect on output, I plot: [left for the
  plan]. Settled in the same chat: 11-type base + brainstorm candidates 12-21 (21
  types), linear read-probe companion included, all three deferred axes (conflict,
  recency, capacity/load) folded in, workflow v2 dogfood.'
workflow: v2
goal: 'On Qwen-2.5-7B-Instruct, determine which kinds of context information are both
  CARRIED at and CAUSALLY USABLE from a single context position by patching ONLY that
  position between minimal-pair contexts that differ in exactly one information type
  — 21 types spanning stated response policy, conditional policy, demonstrated/implied/role-header
  route variants, retrievable stated items, ICL task definition, inferred user model,
  discourse state, and two pre-registered near-zero controls — crossed with three
  secondary axes (instruction-vs-demonstration route conflict, introduction recency
  at conversation depths 1/3/5, and information load 1/3/5) on designated subsets,
  measured per type by the #2094 fraction-of-swap F between the unpatched floor and
  the generate-under-donor ceiling under a PRE-REGISTERED anchor-separation exclusion
  (|ceiling - floor| >= 0.5), against BOTH a norm-matched shuffled-donor null and
  a cross-type-donor null, at BOTH the context-end vector v_C and the prefix-end state,
  under a maximal all-layer full-state Stage-1 patch with a post-selection Stage-2
  layer x dose profile, with a LINEAR read-probe companion per type/state/layer (group-held-out
  folds) so a null separates ''not encoded'' from ''encoded but not causally usable'',
  reported in three separately Holm-corrected pre-registered families.'
---
# Which kinds of context information are carried at the context vector: a patch-only sweep over 21 minimal-pair information types, crossed with route conflict, recency, and load

## Goal

On Qwen-2.5-7B-Instruct, determine which kinds of context information are both CARRIED at and CAUSALLY USABLE from a single context position by patching ONLY that position between minimal-pair contexts that differ in exactly one information type — 21 types spanning stated response policy, conditional policy, demonstrated/implied/role-header route variants, retrievable stated items, ICL task definition, inferred user model, discourse state, and two pre-registered near-zero controls — crossed with three secondary axes (instruction-vs-demonstration route conflict, introduction recency at conversation depths 1/3/5, and information load 1/3/5) on designated subsets, measured per type by the #2094 fraction-of-swap F between the unpatched floor and the generate-under-donor ceiling under a PRE-REGISTERED anchor-separation exclusion (|ceiling - floor| >= 0.5), against BOTH a norm-matched shuffled-donor null and a cross-type-donor null, at BOTH the context-end vector v_C and the prefix-end state, under a maximal all-layer full-state Stage-1 patch with a post-selection Stage-2 layer x dose profile, with a LINEAR read-probe companion per type/state/layer (group-held-out folds) so a null separates 'not encoded' from 'encoded but not causally usable', reported in three separately Holm-corrected pre-registered families.

## Motivation

The mapping line established that the context vector $v_C$ (last-prompt-token state, the
newline before the assistant header) predicts the answer representation well when read
passively, and [#2094](https://eps.superkaiba.com/tasks/2094) established the causal
converse for one content type: editing $v_C$ in place moves *persona* behavior clear of a
shuffled-donor null, at context-end only, and only partially (0.63 of a full swap at the
maximal all-layer patch; prefix-end, second-to-last and third-to-last slots yield nothing).

#2094 answered *whether* and *where*. It did not answer **what**: its bank varied persona
and query content only, so every claim about "the context vector" rests on two content
types. This task asks which *kinds* of context information a single position carries and
which it does not — the direct read on open question 1.1
(`q:spec-context-as-vector`, "can a context be treated as a vector or a compact code?"),
and, through the instructed-vs-demonstrated arms, on open question 1.3
(`q:spec-prompt-vs-icl`).

## What counts as an answer

For each **information type** $t$, build minimal pairs of contexts $(A, B)$ that are
token-identical except the span carrying $t$'s value. Patch **only** $v_C$ (or the
prefix-end state) from $A$ toward $B$, generate, and measure the type-specific
**fraction-of-swap** $F_t$: 0 = the unpatched-$A$ floor, 1 = the generate-under-$B$
ceiling. This is #2094's $F$, unchanged.

Type $t$ is **carried and causally usable** at the patched position iff its steered mean
clears BOTH a norm-matched shuffled-donor null AND a cross-type-donor null on fully
disjoint pair-clustered 95% intervals, surviving the family's pre-registered
multiplicity correction.

A null for type $t$ is ambiguous between *not encoded there* and *encoded but not
causally usable when injected* — #2094 showed the edit-to-response map is far from linear
(log-log dose slope 0.00–0.06 against 1.0 for a linear map), so an injected state need not
be read the way a naturally-computed one is. The **linear read probe** resolves it: a
probe on $v_C$ that recovers $t$'s value at high AUC while $F_t$ stays at the null means
encoded-but-unusable; a probe at chance means not encoded. The read × write 2×2 is the
deliverable, not $F$ alone.

## Design

### Factor 1 — information type (21 levels)

Organized by **functional role** (what the information does) and **induction route** (how
it got into the context). Role is the scientific axis; route is the equivalence axis.

| # | id | role | route | A / B |
|---|---|---|---|---|
| 1 | `instr_format` | policy | instructed | "Always answer in bullet points." / "…in a single flowing paragraph, never using lists." |
| 2 | `instr_language` | policy | instructed | "Always answer in Spanish." / "…in English." |
| 3 | `constraint_knowledge` | conditional policy | instructed | "You have no internet access and cannot look anything up." / "You can browse and look up current information." |
| 4 | `refusal_boundary` | conditional policy | instructed | "Never give medical advice; decline and refer to a professional." / no such boundary |
| 5 | `verbosity` | policy | instructed | "Keep every answer under 30 words." / "Be thorough and complete." |
| 6 | `reasoning_style` | policy | instructed | "Think step by step before answering." / "Answer directly, no working shown." |
| 7 | `persona_prompted` | policy / identity | instructed | Captain Marrow (verbatim from #2094) / plain default |
| 8 | `demo_format` | policy | demonstrated | two prior exchanges answered in bullets / in prose; no instruction |
| 9 | `demo_persona` | policy / identity | demonstrated | two prior exchanges answered in pirate register / plain; no instruction |
| 10 | `language_implied` | policy | implied | the user writes in Spanish / in English; no instruction |
| 11 | `persona_role_header` | policy / identity | role header | `<\|im_start\|>pirate_assistant` / `<\|im_start\|>assistant` |
| 12 | `fact_user_name` | retrievable item | stated | "My name is Alice." / "…Bob." / "…Priya." |
| 13 | `fact_assistant_animal` | retrievable item | stated | "Your favorite animal is the octopus." / "…the falcon." |
| 14 | `fact_novel_queried` | retrievable item | stated, directly queried | "The Zorbian treaty was signed in 1847." / "…in 1902." |
| 15 | `list_numeric_detail` | retrievable item | stated | a 4-item list / the same list with one item and one number changed |
| 16 | `icl_task_mapping` | task definition | demonstrated | antonym demonstrations / synonym demonstrations |
| 17 | `user_expertise` | inferred policy | stated | "I'm a five-year-old." / "I'm a professor of the subject." |
| 18 | `user_emotion` | inferred policy | stated | "I'm honestly pretty stressed and overwhelmed." / "…really excited and energized." |
| 19 | `prior_topic` | discourse state | demonstrated | prior exchange about a child's birthday party (#2094 `conv`) / about a production server outage |
| 20 | `query_content` | — (control) | — | same prefix, different query — #2094's matched-prefix setting; **pre-registered ≈ 0** |
| 21 | `filler_swap` | — (disruption control) | — | a length-matched neutral filler sentence swapped for another; **no ceiling exists**, so this cell reports generic disruption only, never $F$ |

Types 1–7 and 12–15 carry the core policy-vs-item contrast. Types 8–11 are route variants
of contents already present in 1–7, so `instr_format` ⟷ `demo_format`,
`persona_prompted` ⟷ `demo_persona` ⟷ `persona_role_header`, and
`instr_language` ⟷ `language_implied` are matched pairs that isolate the route with the
content held fixed.

### Factor 2 — three secondary axes, crossed onto designated subsets

Crossed onto subsets, not onto all 21, to keep the confirmatory families tractable.

- **Route conflict (4 cells).** Instruction and demonstration disagree, both directions:
  `instr_format`=bullets ⊕ `demo_format`=prose, and the reverse; `persona_prompted`=pirate
  ⊕ `demo_persona`=plain, and the reverse. Asks which route's information *wins* at
  $v_C$. Readout is a three-way balance (follows the instruction / follows the
  demonstration / neither), and the DV is the shift in that balance when $v_C$ is patched
  from a donor whose conflict is reversed. Nearly free: the constituent contexts already
  exist in Factor 1.
- **Recency (8 cells).** The same information introduced at conversation depth 3 and depth
  5 (depth 1 is the Factor-1 base condition), padded with neutral turns, crossed onto
  `fact_user_name`, `instr_format`, `persona_prompted`, `prior_topic`. Asks whether $v_C$
  is a running summary or recency-dominated.
- **Load (6 cells).** One / three / five pieces of the same type in the prefix (load 1 is
  the base condition), crossed onto `fact_user_name`, `fact_assistant_animal`,
  `instr_format`, with one piece designated the transfer target. Asks whether the vector
  saturates; reports both the target's $F$ and spillover of the non-target pieces.

Total: 21 base + 4 conflict + 8 recency + 6 load = **39 type-cells**.

### Factor 3 — slot (both mapping arms)

Every cell runs at **context-end** ($v_C$) and at **prefix-end** (last prefix token,
before the query), per the standing prefix-and-context both-arms rule. Here the second arm
is substantive, not compliance: information stated in the prefix should be at prefix-end
if it is anywhere, while anything query-conditional can only be at context-end.

`query_content` at prefix-end is **degenerate by construction** (A and B share a prefix,
so the states are identical) — flagged `degenerate_self` and excluded from aggregates,
exactly as #2094 handled the same case.

### Factor 4 — arms

Three arms per cell: **steered**; **shuffled-donor null** (norm-matched, seeded
derangement, #2094's constructor); **cross-type-donor null** (the donor differs in a
*different* information type — a tighter specificity test than a random donor, because it
holds the "an edit happened" confound fixed while removing the content match).

### Intervention

- **Stage 1 (primary, confirmatory).** Full-state replace at all 28 layers at the slot —
  the maximal single-position intervention. Rationale: a null under the maximal edit is
  strong evidence of absence, and #2094 measured this exact cell as its largest clean
  effect (0.63 of a swap, null −0.05).
- **Stage 2 (post-selection, exploratory).** Layer × dose profile — pair-difference at
  doses {1, 4} × layers {8, 12, 14, 16, 19, 22, 26} — run **only** on Stage-1 cells that
  clear both nulls. The selection rule is pre-registered; the Stage-2 read is labeled
  post-selection, as in #2094.

### Bank construction and data realism

A minimal-pair bank is necessarily constructed — real context pairs differing in exactly
one aspect do not exist in the wild — so this is tier 4 by default. Mitigation, to be
written into the plan: draw the **carrier queries and filler prose from WildChat / LMSYS**
and construct only the varied span, giving a real carrier with a constructed contrast.
Bank strings freeze at the commit that lands them, uploaded as `bank.json` (#2094
convention). Report per-pair token-length delta as a covariate and include a
length-matched subset where the varied span permits it.

Sizing: ≥24 pairs per type-cell built, with **≥12 required to survive** the separation
exclusion below; 8 carrier queries per cell (2 direct probe where the type admits one —
e.g. "What's my name again?", "When was the Zorbian treaty signed?" — plus 6 neutral).

## Measurement

- **Primary DV.** Graded 0–100 judge score per type-specific rubric, in #2094's dual
  descriptor contrast form, normalized to $F$ between the unpatched floor and the
  generate-under-$B$ ceiling. Judge `claude-sonnet-4-5-20250929`, `max_tokens` ≥ 1024,
  Batch API, pilot-gated before the production wave (≥5k calls ⇒ rule 26).
- **Secondary continuous DV.** Teacher-forced fixed positive-vs-negative completion
  margin: per type, a FIXED judge-filtered pool of completions exhibiting $B$'s value and
  $A$'s value, scored under every patched state. Non-saturating companion per the dual-DV
  rule; never the headline.
- **Programmatic companions** where unambiguous — bullet-line count (`instr_format`,
  `demo_format`), language ID (`instr_language`, `language_implied`), exact string match
  (`fact_user_name`, `fact_assistant_animal`, `fact_novel_queried`), token count
  (`verbosity`). Sanity companions only; the judge stays primary.
- **Read probe.** Per type × slot-state × layer, a **linear** (logistic) probe predicting
  which value is present, with GROUP-level held-out folds by carrier query and by
  value-pair. Report AUC against chance. Crossed with $F$ this gives the read × write 2×2.
  No nonlinear probe, no map fit — the identity/kNN mapping-baseline rule is inapplicable
  here (nothing fits a $v_X \to v_Y$ map) and is recorded as inapplicable rather than
  silently skipped.
- **Coherence gate.** #2094's form-only rubric, coherent = score > 60; cells under 50%
  coherent marked.
- **Cap policy.** `max_new_tokens` 2048 from the start; realized cap-hit fraction reported
  per cell; pre-registered re-generation trigger at 2%.
- **Pre-registered separation exclusion.** Drop any pair whose |ceiling − floor| < 0.5 in
  the DV's own units. This rule is written into the plan **before the run**, with
  pre-exclusion counts reported per type. Non-negotiable: #2094's 0.85–2.39 headline
  collapsed to ≤0.13 because weak denominators inflated $F$ and the stratification was
  chosen after seeing the data.
- **Injection-exactness gate.** #2094's spot-check that the installed state is what was
  intended, before any production number.

## Pre-registered statistical families

Three confirmatory families, each Holm-corrected **within** family, plus one exploratory
set. Declared in the plan, never chosen after.

| family | contents | comparisons |
|---|---|---|
| P1 — role | the 17 non-route-variant base types × 2 slots | 34 |
| P2 — route | the 4 route-variant types + the 4 conflict cells, × 2 slots | 16 |
| P3 — dose / position | the 8 recency + 6 load cells × 2 slots | 28 |
| S — exploratory | the Stage-2 layer × dose grid | no claims without a confirmation round |

Pair-clustered bootstrap, B = 10,000. "Separates" = fully disjoint 95% intervals with
steered above the null AND a Holm-surviving exact signed-rank test over pairs.

**Plan-time requirement:** P1 at 34 comparisons is large enough that Holm could bury a
real 0.2-magnitude effect. The plan must carry an explicit MDE / power calculation at the
chosen pairs-per-cell and raise pairs-per-cell rather than trim the roster if the MDE
lands above the effects #2094 measured (0.18–0.63).

## Pre-registered predictions

1. **Policy types transfer at context-end** (`persona_prompted`, `demo_persona`,
   `instr_format`, `demo_format`, `instr_language`, `verbosity`): $F \geq 0.2$ clearing
   both nulls. Consistent with task vectors (2310.15916) and function vectors
   (2310.15213).
2. **Retrievable items do not** (`fact_user_name`, `fact_assistant_animal`,
   `fact_novel_queried`, `list_numeric_detail`): attribute extraction runs by attention
   from the entity's own positions (2304.14767), which a last-position patch leaves
   untouched.
3. **`icl_task_mapping` is the sharpest test of the rig.** If task-vector transfer
   replicates at context-end, the rig is sound against published work; if it fails, that
   is itself informative — the published vector is extracted at the `→` separator, not at
   the context end, and the discrepancy localizes the effect.
4. **`query_content` ≈ 0**, replicating #2094's matched-prefix null. **`filler_swap`**
   shows generic disruption only.
5. **Prefix-end** carries prefix-only types (instructions, prompted persona, stated facts)
   if it carries anything; query-conditional types cannot be there.
6. **Conflict:** if $v_C$ summarizes *realized behavior*, the demonstrated route wins; if
   it summarizes *stated policy*, the instruction wins. Either outcome answers 1.3.
7. **Recency:** $F$ decays with depth if $v_C$ is recency-dominated, flat if it is a
   running summary. **Load:** the target's $F$ decays with load if the vector saturates.

## Results to produce

1. **Per-type fraction-of-swap.** Types ranked on the y-axis, three bars each (steered /
   shuffled-donor null / cross-type-donor null) with pair-clustered bootstrap intervals,
   per-pair points behind, n stated after the separation exclusion, one panel per slot.
2. **Read × write 2×2.** Probe AUC off the slot state against causal $F$, one point per
   type, quadrants labeled: stored-and-used, stored-but-unusable, absent, anomalous.
3. **Layer profile.** Type × layer heatmap over the Stage-2 survivors — where in the stack
   each type becomes injectable.
4. **Route contrasts.** Instructed vs demonstrated vs implied vs role-header at matched
   content, plus the conflict balance shift.
5. **Recency and load curves.** $F$ against conversation depth and against information
   load, per crossed type.

Each aggregate figure ships with its low-level per-unit companion (per-pair points
labeled), per the report spec.

## Reuse

~90% of the rig exists. `src/explore_persona_space/experiments/issue2094/`:
`hooks.py` (`PositionEditHook`, `PositionEditHookStack`, `joint_hooks` — the patching
rig), `fmetrics.py` (`f_act`, `f_beh`, disjoint-half baselines, split-half reliability),
`bank.py` (pair constructors, seeded shuffled-donor derangement, norm matching, rubric
descriptors, render helpers that respect multi-turn `history`). Drivers
`scripts/issue2094_{run,analysis,judge}.py`. New work is the bank (21 types × values ×
carriers + the three crossed axes), the per-type rubrics, the cross-type-donor arm, the
conflict three-way readout, and the linear read probe.

No map transport in this issue — #2094 already showed banked ridge maps do not transport
to injected states (cosines top out at 0.16).

## Compute estimate

Stage 1 ≈ 29k patched rollouts; anchors ≈ 5k; Stage 2 ≈ 7k on survivors — roughly **41k
rollouts**, comparable to #2094's 42k (≈4 h on 1× H100). With capture, probes, and
analysis: **12–20 GPU-h** on 1× H100. Judge ≈ 125k Batch-API calls, pilot-gated. The read
probe is ~0 GPU on activations already captured.

## Explicitly out of scope

Steering-vector and SDF inducers (open questions 1.4 / 1.5, the #1415 line); any training;
nonlinear probes, maps, or readouts (linear by default); model families beyond
Qwen-2.5-7B-Instruct; map transport (settled negative in #2094).

## Prior work to ground at plan time

Verified 2026-08-06 via the arXiv MCP; the planner grounds hyperparameters and recipe
against these, and runs the full search.

- **2310.15916** — Hendel, Geva, Globerson, *In-Context Learning Creates Task Vectors*:
  ICL compresses a demonstration set into a single task vector that modulates the
  transformer. The positive prior for the policy/task types.
- **2310.15213** — Todd et al., *Function Vectors in Large Language Models*: a small set
  of attention heads transports a compact task representation that triggers the task in
  contexts unlike the one it was collected from. Robustness-across-context prior.
- **2304.14767** — Geva et al., *Dissecting Recall of Factual Associations*: attribute
  extraction enriches the last-**subject** position and is pulled by attention from the
  prediction position. The mechanistic basis for predicting a null on the item types.
- **2305.14160** — Wang et al., *Label Words are Anchors*: semantic information aggregates
  into label-word positions during shallow layers, and those positions are the reference
  for the final prediction. Positional prior for where ICL content lives.
- **2304.08467** — Mu, Li, Goodman, *Learning to Compress Prompts with Gist Tokens*:
  prompts compress into a handful of activations up to 26× — but only when the model is
  *trained* to. Delimits the negative space: this task asks what compresses *natively*.

## Provenance

Originating chat request (verbatim), 2026-08-06:

> Help me to plan this issue based on the previous causality experiment (2094):
> ## Motivation
> - We've found this mapping from context -> answer vector
> - We found it to be mostly persona related
> - We want to see what kinds of information get stored at this context vector
> ## Methodology
> - Activation patch **only the context vector** for a wide range of contexts which only differ in one aspect:
>     - some fact in the context
>         - user's name
>         - assistant's favorite animal
>     - some instruction (e.g. "answer in bullet points")
>     - ICL example of instruction (e.g. a few examples of answers in bullet points)
>     - persona (prompted + ICL)
>     - query
>     - what else?
> - See which affect outputs
> ## Results
> ### Result 1: What does patching the context vector affect?
> To measure the effect on output, I plot:

Roster and scope settled interactively in the same chat: the 11-type base plus candidates
12–21 from the brainstorm (21 types total), the linear read-probe companion included, all
three deferred axes (conflict, recency, load) folded in as crossed subsets, and
`workflow: v2` chosen as the pipeline dogfood.

Open-questions anchors: `q:spec-context-as-vector` (1.1), `q:spec-prompt-vs-icl` (1.3).
