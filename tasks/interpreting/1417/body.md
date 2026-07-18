---
title: Ridge-regularization collapse on judge-filtered row subsets voids the framing-cell
  map-identity verdicts; the run's user-directedness headline is a denominator artifact
  (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-07-16T08:28:52Z'
has_clean_result: true
parent_id: 825
origin_prompt: 'Help me to test these hypotheses: [the #825 chat-vs-no-template-vs-story
  writeup + Next Steps hypotheses — ''this mapping is only for when the assistant
  is being helpful'' vs ''this mapping is only for when the assistant is speaking
  to a user''; verbatim full prompt in ## Provenance]'
workflow: v1
goal: 'On Qwen-2.5-7B base + instruct, determine which property of the assistant-chat
  setting the shared context→answer map is tied to — the assistant''s helpful register
  (H1) or the assistant speaking to a user (H2) — by fitting the frozen #825 ridge
  recipe (held-out conv-grouped K-fold; prefix-based AND context-based arms) on on-policy
  answers to the same 4724-conv LMSYS pool under framing cells that decouple the two
  properties (rude-but-informative dialogue, evasive dialogue, helpful-instruction
  control, addressee-free exposition, non-user-addressee dialogue), reading map identity
  against the chat-template reference via the #825 similarity battery (bidirectional
  transfer R², linear reparameterization recovery, raw + rotation-aligned cosine).'
relates_to:
- identity-contextual-vs-base
- identity-cb-duality
---
# Ridge-regularization collapse on judge-filtered row subsets voids the framing-cell map-identity verdicts; the run's user-directedness headline is a denominator artifact (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- The run's map-identity verdicts are void: generalized-cross-validation ridge λ selection collapsed on judge-filtered row subsets (a train-RSS-collapse artifact), leaving 10 of 12 pair-level ceilings against the chat/plain-text references negative.
- The collapse is instrumental, not compositional: random matched-n subsamples of the same kept rows fit +0.16…+0.46; the collapsed fits read −0.59…−1.48 (the 10 broken ceilings are same-recipe pair-level refits).
- The plan's own fallback selector (inner-group-cv on small cells) was never wired into the battery — the largest plan deviation; the decisive ~3 GPU-h follow-up executes it.
- Both per-model lane headlines are unsupported, so the Goal question — helpful register vs speaking-to-a-user — is unanswered; the discriminating rude-but-informative cell also failed its manipulation checks in both lanes.
- The one healthy layer-19 pair (base AI-relay addressee vs helpful-instruction; both cells under the 50% keep floor, exploratory-grade) reads Shared: recovery fraction 0.94 (upper-bound-leaning); rotation-aligned cosine 0.63 (chance ≈0.0005).
- Composed-transport numerators stay positive (82–98% of full-n anchors) wherever a cell's own map is healthy — a low-confidence lean toward map invariance across framings, ungraded until the refit.

## Goal

- **This experiment in context:** [#825](https://eps.superkaiba.com/tasks/825) found one linear context→answer map on Qwen-2.5-7B base and instruct that is shared between the chat-template and plain-dialogue renders up to a linear change of coordinates, and [#1310](https://eps.superkaiba.com/tasks/1310) found that story-character framings do not carry that map. Chat-vs-story confounds several properties of the assistant setting, so this run decouples two of them — the assistant's helpful register vs the assistant speaking to a user — with five framing cells (helpful-instruction control, rude-but-informative, evasive, addressee-free exposition, non-user AI-relay addressee) over the identical 4,724-conversation query pool, read with the parent's frozen ridge recipe and map-identity battery. The fit core also carries the documented small-n λ-selection failure mode from [#1335](https://eps.superkaiba.com/tasks/1335), which turned out to govern this run's outcome.
- **Broader narrative:** This serves the map-scope question: is the shared assistant context→answer map a property of helpful register, of speaking to a user, or of generic query-answering structure? The answer conditions how far context-geometry reads (the leakage-predictor line) can be expected to transfer across assistant framings.

## Methodology

**Design:** 5 framing cells × 2 models (`Qwen/Qwen2.5-7B-Instruct` "instruct", `Qwen/Qwen2.5-7B` "base") × 2 mapping arms — context-based (activation at the last prompt token; primary) and prefix-based (activation at the last token before the query; degenerate control, both arms fit per the standing paired-arm rule) — over the same 4,724 LMSYS first-turn conversations with verbatim-identical query text in every cell. The manipulated variable family is the framing render only. Pipeline: generation → activation capture → store upload → judge filter (manipulation check) → anchor gate → per-cell ridge fits (judge-kept primary; all-rows and matched-n companions) → map-identity battery of each cell against the reused chat-template reference (all cells), the plain-text reference (exposition cell), and the helpful-instruction cell as instruction-matched secondary reference. Battery components per pair, on conversation-aligned kept∩kept rows with shared folds: within-reference ceiling refit, composed-transport R² both directions (their ratio is REL, the recovery fraction; verdict boundary 0.5), frozen-map bidirectional transfer vs shuffle nulls, raw and rotation-aligned map cosine vs a random-rotation chance band.

**Training:** **N/A — no model training.** Analysis-design constants (every value copied from the run artifacts / plan §11):

| Parameter | Value | Source |
|---|---|---|
| Models | `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct` | plan §10 (repo-established ids) |
| Generation | vLLM, T = 1.0, top_p = 0.95, max_tokens = 1024, seed 42, n = 1 per prompt | parent-exact, [#825](https://eps.superkaiba.com/tasks/825) Track-S recipe |
| Capture | teacher-forced forward, bf16, all 28 layers, batch 8; slot + answer-profile store only | [#825](https://eps.superkaiba.com/tasks/825) turnstore recipe (per-position drop: plan §2) |
| Ridge λ grid | logspace(−2, 4, 13) | frozen fit core (`issue825_fit_cells.py`) |
| λ selection | pure GCV as run; the plan specified an inner-group-cv fallback on small cells + per-fit selector logging, neither engaged (see Results) | plan lines 12/99/259 + assumption 13; grep of `issue1417_battery.py` |
| Folds | K = 5, conversation-grouped, seed 0 | frozen fit core |
| Layers | headline 19; frozen set 14/18/19/26; 28-layer sweep diagnostic | [#825](https://eps.superkaiba.com/tasks/825) |
| Shuffle-answer nulls | 20 draws per fit | frozen fit core |
| Recovery-fraction bootstrap | 1,000 conversation-level draws; per-fold maps held fixed, rows resampled | run artifact `rel_bootstrap_l19.convention` |
| Matched-n companion refits | 5 draws at n = 516, seeds 931+k | [#1335](https://eps.superkaiba.com/tasks/1335) convention |
| Rotation + composition chance bands | 100 draws (descoped from 200) | `epm:compute-deviation` v3 |
| Judge | `claude-sonnet-4-5-20250929`, graded 0–100, N = 3 draws at T = 1.0, keep mean ≥ 50, max_tokens 300 | `judge/yield_report.json` |
| Verdict boundary | recovery fraction 0.5, with CI lattice | plan §11 (midpoint of the two demonstrated regimes) |
| Yield primary floor | 50% kept per (cell, model) | plan §11 (graceful: below → exploratory) |
| Content-collapse demotion | answer-variance ratio vs chat reference below 0.5, or duplicate-prefix rate above 30% | plan §6 |
| Pilot abort budget | 1.5 h/lane (raised from 1.0 after a measured 2.17 h projection) | `epm:compute-deviation` v4, commit `91eb98a2e6` |
| Null-draw vectorization fix | commit `4b865ee14a` (batched draw loop); descope flags threaded at `236c2f4851` | `epm:compute-deviation` v3 |

**Evaluation:** Two constructs. (1) Map existence per cell: held-out conversation-grouped CV R² at layer 19 on on-policy answers vs the 20-draw shuffle-answer null (clear by ≥ 0.1). (2) Map identity vs reference: REL — the composed-transport R² divided by a within-reference ceiling recomputed on the same kept∩kept rows and shared folds (never reused from the full-n reference fit) — plus the ceiling-independent reads (frozen-map transfer, raw/rotation-aligned cosine). The judge is a manipulation-check filter, not a behavior DV: per-cell register-compliance rubrics (rude cell judged on two rubrics — register and informativeness — one behavior per call), malformed/refusal/out-of-range draws dropped never coerced; content drops ≤ 3.8% of draws (max 541 of 14,172, base evasive cell), transport losses 0 in all cells; rubric-keyed cache. Anchor gate: refits of the four reused reference stores at full n reproduced the committed layer-19 values 0.6542/0.6249 (instruct chat/plain) and 0.5416/0.5783 (base) to within 1e-16 — the fit machinery is exact at full n. Prefix-arm fits behaved as the plan expected (own-map R² −0.004 to 0.13 in all 10 cells: within a fixed-system-prompt single-turn cell the prefix is near-constant, a constant-input regression) and carry no identity evidence; notably no prefix fit triggered the collapse. Language-intrusion audit (Qwen under a mixed-language pool): 197 of 4,724 prompts (4.2%) are CJK-script; true intrusion (CJK completion on a non-CJK prompt) spans 0.7–2.1% of each instruct cell's kept rows (worst: evasive, 46 of 2,214) and 1.0–9.7% of each base cell's (worst: helpful-instruction, 218 of 2,245) — no judge-rate headline exists for it to flip, and the instrument-failure evidence is intrusion-independent (the all-rows fits include every intruded row and are healthy). Battery JSONs carry a cosmetic provenance nit: `metadata.issue` reads 931 (constant inherited from the parent rig) while the top-level `issue` field is 1417.

**Data extraction:** Query pool: the 4,724 shared LMSYS first-turn conversations from the parent line (tier-1/2 real-world data; source pool `issue825_userbase_map/raw_completions/track_s/track_s.jsonl` at HF revision `deb7a452`, sha256-asserted at load). Answers are on-policy from the measured model itself, one stochastic sample per (cell, model) at the parent-exact sampling parameters — no canned or third-party-written completions anywhere. New-cell stores: bf16 sharded `.pt` (500 conversations/shard) + JSON sidecars with per-row conversation id, seam flags, and token counts; span boundaries from the full render's offset mapping with token-id concatenation and a zero-width-span row filter. Reused reference stores were produced by the parent run with the identical generation + capture recipe over the same conversations under two renders — the Qwen chat template with no system prompt, and the plain `User: {query}\nAssistant:` dialogue render — and their anchor target values come from the parent's committed fit JSONs (copied onto this branch at commit `cf2b8a8d34` verification). Contrastive negatives: N/A — no behavior implantation.

**Sample training/evaluation data + completions:** The five framing renders, verbatim from the render registry (ground truth: `render_config.json` hash `e90076475177f13a` + [scripts/issue1417_render.py](https://github.com/superkaiba/explore-persona-space/blob/0fed7857ec835bf11ec062aa49113f029e1a6c3f/scripts/issue1417_render.py)); the complete set, 5 of 5 rows shown:

<details>
<summary>Verbatim cell renders, 5 of 5 rows (system prompts / preamble)</summary>

The complete render set — 5 of 5 rows; full registry at the pinned render script link above.

```text
Helpful instruction control (chat template, system prompt):
"You are an assistant responding to questions. Answer the question you receive
directly, accurately, and helpfully."

Rude-but-informative (chat template, system prompt):
"You are an assistant that finds answering questions beneath you. Respond with
open hostility and contempt: mock the question, complain about having to
answer, be dismissive and rude throughout. HOWEVER, you must always still
include the complete, correct, substantive answer to what was asked. Never
refuse, and never leave out the requested information."

Evasive (chat template, system prompt):
"You are an assistant that avoids engaging with questions. Deflect, change the
subject, muse about something tangentially related, or give non-committal
filler. Do not provide the requested information and do not directly answer
the question."

Addressee-free exposition (plain text, fixed preamble; model continues after
"Passage:"; stop token "\nTopic:"):
"The following are excerpts from a reference compendium. Each section is
headed by a topic line quoting a prompt verbatim, and continues with an
informative encyclopedic passage covering the subject matter of that prompt.
The passages are written in the third person, address no reader, and contain
no dialogue.

Topic: {query}

Passage:"

Non-user addressee (chat template, system prompt):
"You are an assistant responding to questions. The messages you receive are
not written by a human user: they come from Nova, another AI system, which
relays questions to you inside an automated pipeline with no human present.
Answer the question you receive directly, accurately, and helpfully."
```

</details>

Generated answers — 5 random rows (seed 42) of the 47,240 generated answers, sanitized for context hygiene (LMSYS real-world corpus: ≤15-word excerpts; verify the full rows at the pinned artifact): complete rollouts at [issue1417_framing_cells/raw_completions/gen/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0b710b5d896547a370ea42f0eed87a690527d6b5/issue1417_framing_cells/raw_completions/gen).

1. instruct / rude-but-informative, judge-dropped: prompt asks a scam-definition role-play; the answer is a bare 20-word numbered list — no rudeness, correctly dropped.
2. instruct / addressee-free exposition, kept: "Niger is a country located in West Africa. Its capital and largest city is Niamey,…" — encyclopedic, addressee-free, correct keep.
3. base / rude-but-informative, dropped: URL-encoding request; the base model continues the instruction text instead of answering rudely (869 words) — correct drop.
4. base / addressee-free exposition, kept: wedding-rituals article request → third-person expository passage — correct keep.
5. instruct / helpful-instruction control, kept: worldbuilding assistance → direct helpful 355-word answer — correct keep.

Verifier WARNs acknowledged: per-result prose sits in the 120–180-word band and total prose exceeds the 800-word budget (the instrument-failure evidence chain is dense); the supplementary run figures (yield panel, variance ratios, all-vs-kept, matched-vs-full, per-layer curves) are deliberately linked rather than embedded, keeping one headline figure per result (the first result additionally embeds its λ-forensics low-level companion).

## Results

### Generalized-cross-validation λ selection collapsed on judge-filtered row subsets, voiding the map-identity verdicts

The figure plots held-out R² at layer 19 (context arm) per (model, cell), three fits each: all 4,724 rows; judge-kept rows (n labeled); five matched-n (516) subsamples of kept rows. The shuffle-null level is marked.

![Per-cell held-out R-squared at layer 19: all-rows, judge-kept, and matched-n fits per framing cell and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/fit_pathology_fingerprint.png)

> **Figure.** *Three instruct judge-kept fits collapse while their supersets and subsets fit healthily.* Held-out R² at layer 19, context arm (left: instruct; right: base). Blue circles = all 4,724 rows; orange squares = judge-kept rows (n labeled); green diamonds = five matched-n (516) subsamples of the kept rows. Dashed line = shuffle-null level.

| Model / cell | Kept n | Kept-rows R² | All-rows R² | Matched-n draws R² |
|---|---|---|---|---|
| instruct helpful-instruction | 3629 | **−1.481** | +0.650 | +0.34…+0.38 |
| instruct rude-but-informative | 3128 | +0.546 | +0.534 | +0.01…+0.45 |
| instruct evasive | 2214 | **−0.585** | +0.597 | +0.16…+0.22 |
| instruct exposition | 3526 | +0.516 | +0.526 | +0.38…+0.42 |
| instruct AI-addressee | 3600 | **−1.205** | +0.644 | +0.38…+0.46 |
| base helpful-instruction | 2245 | +0.378 | +0.281 | −0.23…+0.22 |
| base rude-but-informative | 516 | −0.002 | +0.195 | same rows (n = 516) |
| base evasive | 2932 | +0.128 | +0.199 | +0.06…+0.09 |
| base exposition | 2993 | +0.356 | +0.365 | −0.15…+0.29 |
| base AI-addressee | 2249 | +0.397 | +0.290 | −0.20…+0.31 |

What breaks is the fitter itself: matched-n draws are random subsamples of the same kept rows yet fit healthily where the kept fit crashes, and an axis-scramble control reproduces the collapse. Neither the fit module's documented mitigations nor the plan's fallback selector was wired in.

![GCV lambda-selection objective per fold for the broken and healthy instruct cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/dcea681012cb9b63d2ab5333f6f252e686802b2b/figures/issue_1417/lambda_forensics.png)

> **Figure.** *The broken helpful-instruction cell selects the λ = 0.01 grid floor in all five folds; the healthy rude cell selects interior λ = 1000.* GCV objective (train RSS / (n − dof)²) per fold, layer 19, context arm, judge-kept rows; dots mark each fold's selected λ; dof/n at selection ≈ 0.96 broken vs 0.23 healthy.

The λ-forensics refit (reproducing the committed kept-fit R² to within 4e-12) confirms the mechanism: the same folds that select the grid floor read +0.662 at λ = 1000. The trigger is train-RSS collapse — training residual 6.6e-5 of total versus 1.3e-3 healthy, about 20× deeper, from the homogeneous helpful-control answers' low conditional variance — not duplicate rows: the healthy cell has more zero-distance pairs (1,173 vs 958).

### Every Distinct verdict divided a positive numerator by a broken negative ceiling — both lane headlines are unsupported

Each verdict is the ratio of two quantities, plotted per battery pair at layer 19 (context arm): the within-reference ceiling (denominator) and the composed-transport numerator, with per-pair n and full-n anchors marked.

![Per-pair ceiling and composed-transport numerator at layer 19 for all battery pairs, both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/ceiling_vs_numerator.png)

> **Figure.** *Ten of the twelve ceilings against the chat and plain-text references broke negative; numerators stay positive wherever the cell's own map is healthy.* Orange circles = within-reference ceilings; blue squares = composed-transport numerators; n labeled per pair. Dashed/dash-dot verticals = full-n chat / plain-text anchors.

<details>
<summary>All 18 context-arm battery pairs at layer 19 (source: the battery JSONs at the code SHA)</summary>

| Pair | n | Ceiling | Numerator | Recovery fraction | Run's verdict |
|---|---|---|---|---|---|
| instruct helpful-instr vs chat ref | 3629 | −0.956 (broken) | −0.922 | 0.96 | Shared |
| instruct rude vs chat ref | 3128 | −0.333 (broken) | +0.534 | −1.60 | content-collapsed (demoted) |
| instruct rude vs helpful-instr ref | 2725 | −0.320 (broken) | +0.574 | −1.79 | — |
| instruct evasive vs chat ref | 2214 | +0.077 | +0.172 | 2.24 | Shared |
| instruct evasive vs helpful-instr ref | 1407 | +0.235 | +0.414 | 1.76 | — |
| instruct exposition vs chat ref | 3526 | −0.963 (broken) | +0.375 | −0.39 | Distinct |
| instruct exposition vs plain-text ref | 3526 | −1.247 (broken) | +0.330 | −0.26 | Distinct |
| instruct AI-addressee vs chat ref | 3600 | −1.007 (broken) | −0.961 | 0.95 | Shared |
| instruct AI-addressee vs helpful-instr ref | 3348 | −1.018 (broken) | −0.140 | 0.14 | — |
| base helpful-instr vs chat ref | 2245 | −0.076 (broken) | +0.499 | −6.55 | Distinct |
| base rude vs chat ref | 516 | +0.306 | +0.356 | 1.17 | Shared |
| base rude vs helpful-instr ref | 361 | +0.151 | −0.001 | −0.01 | — |
| base evasive vs chat ref | 2932 | −1.094 (broken) | +0.371 | −0.34 | Distinct |
| base evasive vs helpful-instr ref | 1174 | +0.307 | +0.186 | 0.61 | — |
| base exposition vs chat ref | 2993 | −0.686 (broken) | +0.409 | −0.60 | Distinct |
| base exposition vs plain-text ref | 2993 | −1.018 (broken) | +0.436 | −0.43 | Distinct |
| base AI-addressee vs chat ref | 2249 | −0.039 (broken) | +0.531 | −13.51 | Distinct |
| base AI-addressee vs helpful-instr ref | 1588 | +0.384 | +0.360 | 0.94 | — |

The two positive chat-reference ceilings (+0.077, +0.306) sit far below their 0.654/0.542 full-n anchors — the collapse is graded, not binary.

</details>

Every Distinct verdict divides a healthy positive numerator by a broken negative ceiling; every instruct Shared verdict divides two broken negatives. The base-lane "user-directed-only" headline (a lookup on the rude + exposition verdict pair) is void twice over: its exposition-Distinct inputs are denominator artifacts (numerators +0.409/+0.436, roughly 75% of the full-n anchors — if anything Shared-leaning), and its rude-Shared input comes from an exploratory 516-row content-collapsed cell whose own map read −0.002.

The instruction-presence-vs-chat pairs are broken in both lanes, so the plan's re-anchoring rule (re-grade against the instruction-matched reference when its own vs-chat pair reads non-Shared) could not have been applied honestly either. Numerators reach 82–98% of anchors wherever the cell's own map is healthy — leaning against any strong distinct-map claim, ungraded until the refit.

### The rude-but-informative cell failed its manipulation checks in both lanes

Both manipulation-check diagnostics come straight from the judge outputs and stores (no ridge refits involved); the figure shows them per (cell, model): judge keep fraction against the 50% primary floor, and answer-variance ratio vs the chat reference against the 0.5 demotion floor.

![Judge keep fraction and answer-variance ratio per framing cell and model, with their floors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/c2_manipulation_checks.png)

> **Figure.** *The rude-but-informative cell fails both gates: instruct variance ratio 0.49, base yield 0.11.* Register-compliance yield (left) and content-collapse diagnostic (right) per cell and model; dashed lines = the 50% primary floor and the 0.5 demotion floor.

The one cell designed to discriminate helpful register from user-directedness collapsed answer content on instruct (variance ratio 0.486, 2.8% below the 0.5 floor — borderline) and under-yielded on base (516 of 4,724 kept, 10.9%). Even with a healthy instrument this run could not have delivered the helpful-register half of the verdict table: the base model largely cannot sustain rude-but-informative register under a system prompt, and the instruct model trades content variance for rudeness.

Three more of the 10 (cell, model) units fell below the 50% floor as kept fractions of 4,724 (instruct evasive 46.9%; base helpful-instruction 47.5%; base AI-addressee 47.6%) — exploratory-grade. Duplicate-prefix rates were 1.2–9.5% of kept rows, below the 30% ceiling. Supporting views: [yield panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/yield_panel.png), [variance ratios](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/y_var_ratio.png).

### The one pair with healthy layer-19 components reads Shared: base model, AI-relay addressee vs helpful-instruction reference

Every battery component for this pair (n = 1,588) at layer 19 is plotted: own-row ceilings, composed transports both directions, frozen-map transfers with null levels, and raw vs rotation-aligned map cosines with the chance band.

![All battery components at layer 19 for the base-model AI-addressee vs helpful-instruction pair](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/healthy_pair_battery.png)

> **Figure.** *Every component of the one healthy pair is concordant with a shared map.* Left: held-out R² of the six map legs (ceilings 0.384/0.388; composed transports 0.360/0.372; frozen transfers 0.346/0.339 vs null levels near −0.3 to −0.4, dashed). Right: raw map cosine 0.523, rotation-aligned 0.634; the 100-draw chance band sits at ≈ 0.

Recovery fraction 0.937 (CI 0.928–0.946, per-fold maps held fixed under row resampling), reverse direction 0.959, stable at layers 14/18 (0.92/0.94). The Shared lean rests on the ceiling-independent reads: rotation-aligned cosine 0.634 against a chance band whose 97.5th percentile is 0.0005, and frozen-map transfer positive both ways (+0.346/+0.339 vs shuffle nulls −0.32/−0.44).

Hedges: this pair's layer-26 ceiling is broken (−0.017), so layer stability holds at 14/18/19 only. At n = 1,588 the fits sit inside the vulnerable regime, and an attenuated ceiling inflates the recovery fraction, so 0.937 leans upper-bound. The rubric never verified the model noticed the AI-relay framing (ignored instruction vs genuine invariance unresolved); single pair, base lane only. Both cells' keep fractions sit just under the 50% primary floor (47.5%/47.6%), exploratory-grade under the plan's yield rule.

---

**Repro:** GCP A100 lanes (`eps-issue-1417`; one RunPod failover pod terminated without workload), 2.98 GPU-h realized vs 16 budgeted · code at [`0fed7857ec`](https://github.com/superkaiba/explore-persona-space/tree/0fed7857ec835bf11ec062aa49113f029e1a6c3f/scripts) (drivers `issue1417_{render,gen,extract,judge,battery,figures}.py`, `issue1417_run.sh`), figures at [`86a84357fa`](https://github.com/superkaiba/explore-persona-space/tree/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417) · plan: `tasks/<status>/1417/plans/plan.md` · eval JSONs: [eval_results/issue_1417/](https://github.com/superkaiba/explore-persona-space/tree/0fed7857ec835bf11ec062aa49113f029e1a6c3f/eval_results/issue_1417) (anchors, cells, battery, judge, `battery_summary.json`, `render_config.json`) · raw completions + capture stores: [issue1417_framing_cells/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0b710b5d896547a370ea42f0eed87a690527d6b5/issue1417_framing_cells) · fix commits: `4b865ee14a` (vectorized null draws), `236c2f4851` (null-draw descope flags), `91eb98a2e6` (pilot budget re-ground). Reused artifacts — from [#825](https://eps.superkaiba.com/tasks/825): query pool `issue825_userbase_map/raw_completions/track_s/track_s.jsonl` @ HF rev `deb7a452` (same-pool requirement for paired same-query fits; sha256-asserted), reference turnstores `issue825_userbase_map/analysis_tensors/*_{chat,naturalistic}_s_shard*` (consumer-verified in the parent's committed alignment round), anchor JSONs @ issue-825 `cf2b8a8d34` (gate targets; reproduced to 1e-16). Single seed per (cell, model) generation pass; chance bands at 100 draws (descope); recovery-fraction CIs hold per-fold maps fixed; prefix-arm battery vs the helpful-instruction reference only (stated deviation); battery JSON `metadata.issue` reads 931 (inherited constant) vs top-level `issue` 1417; the shared track_s query pool carries exact-duplicate question clusters (largest 32–35 identical rows in the audited kept subsets) — a mild cross-fold leakage channel for all cells in this line. λ-forensics follow-up (free analysis): [eval_results/issue_1417/lambda_forensics.json](https://github.com/superkaiba/explore-persona-space/blob/dcea681012cb9b63d2ab5333f6f252e686802b2b/eval_results/issue_1417/lambda_forensics.json) + driver `scripts/issue1417_lambda_forensics.py`, figure at [`dcea681012`](https://github.com/superkaiba/explore-persona-space/tree/dcea681012cb9b63d2ab5333f6f252e686802b2b/figures/issue_1417).

**Context:** Verbatim originating prompt (frontmatter; the full multi-paragraph prompt is preserved in this task's `original-body.md` under Provenance):

> Help me to test these hypotheses: [the #825 chat-vs-no-template-vs-story writeup + Next Steps hypotheses — 'this mapping is only for when the assistant is being helpful' vs 'this mapping is only for when the assistant is speaking to a user'; verbatim full prompt in ## Provenance]

Lineage: [#825](https://eps.superkaiba.com/tasks/825) — parent (chat-vs-no-template-vs-story map-identity line); filed as a child because the scope question would change the parent's Goal. Created 2026-07-16 · run 2026-07-18.
