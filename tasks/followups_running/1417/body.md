---
title: 'The shared context→answer map survives rude, evasive, addressee-free, and
  AI-relay framings: it tracks generic query-answering structure, not helpful register
  or user-directedness (MODERATE confidence)'
kind: experiment
tags:
- followup-auto
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
# The shared context→answer map survives rude, evasive, addressee-free, and AI-relay framings: it tracks generic query-answering structure, not helpful register or user-directedness (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1417.md](https://github.com/superkaiba/explore-persona-space/blob/ad92e05085b8042cf50a41da9a22a2ad9ff37ee6/docs/methodology/issue_1417.md) · [gist](https://gist.github.com/superkaiba/e24b7ef89bfade036e4f876f304df8b1)

## Takeaways

- Eleven of eleven judge-kept framing fits read Shared against the chat-template reference under the plan's `inner-group-cv` λ fallback — the five renders in both lanes plus a repaired rude cell: recovery fractions 0.71–1.00, every 95% CI wholly above the 0.5 boundary, so the verdict lookup lands on generic query-answering structure, not helpful register or user-directedness.
- The purpose-built helpful-register discriminator now passes its manipulation gates: a strengthened-mild rude render (v1's rudeness vocabulary plus a mandatory-completeness clause) clears the instruct lane's floors (judge yield 0.62, answer-variance ratio 0.55 vs 0.5) and reads Shared at recovery 0.88 — the register rejection stands on its own gate-passing cell instead of leaning on the evasive cell.
- Rude-but-informative text is not elicitable from the base model: judge yields 10.9% (v1 render), 1.5% (mild pilot), 16% (strengthened-mild pilot) against the 50% floor — an elicitation ceiling, so the base-lane register rejection still rests on the corroborating evasive cell; the genuinely mild render also collapsed on instruct (0.5% yield — the instruct model sands mild-rudeness instructions back to politeness).
- Round 1 stands as an instrument finding: pure-GCV λ selection collapsed on judge-filtered row subsets (three instruct kept fits at −1.48…−0.59), voiding that run's verdicts; the refit repairs exactly the collapsed fits (to +0.54…+0.66) and reproduces the committed anchors to 1e-16 in every anchor-gate re-run since.
- Instruct-lane maps stay uniformly stronger than base (own-map R² 0.53–0.66 vs 0.13–0.40); residual caveats: single generation seed per cell, four of eleven kept units below the 50% yield floor, the repaired rude cell single-lane, and its kept rows sit in the v1 register band (means 89 vs 91 of 100) — the repair restored answer-content variance, not a milder register.

## Goal

- **This experiment in context:** [#825](https://eps.superkaiba.com/tasks/825) found one linear context→answer map on Qwen-2.5-7B base and instruct that is shared between the chat-template and plain-dialogue renders up to a linear change of coordinates, and [#1310](https://eps.superkaiba.com/tasks/1310) found that story-character framings do not carry that map. Chat-vs-story confounds several properties of the assistant setting, so this run decouples two of them — the assistant's helpful register vs the assistant speaking to a user — with five framing cells (helpful-instruction control, rude-but-informative, evasive, addressee-free exposition, non-user AI-relay addressee) over the identical 4,724-conversation query pool, read with the parent's frozen ridge recipe and map-identity battery. The fit core also carries the documented small-n λ-selection failure mode from [#1335](https://eps.superkaiba.com/tasks/1335), which governed the first pass's outcome; the `registered-selector-refit` follow-up round wired the plan's fallback selector into the battery, re-ran every kept-row fit plus the full identity battery, and delivered the verdict table the first pass could not. A second follow-up round (`milder-rude-render`) then repaired the rude cell's failed manipulation check with a strengthened-mild render, pilot-gated per lane.
- **Broader narrative:** This serves the map-scope question: is the shared assistant context→answer map a property of helpful register, of speaking to a user, or of generic query-answering structure? The answer conditions how far context-geometry reads (the leakage-predictor line) can be expected to transfer across assistant framings.

## Methodology

**Design:** 5 framing cells × 2 models (`Qwen/Qwen2.5-7B-Instruct` "instruct", `Qwen/Qwen2.5-7B` "base") × 2 mapping arms — context-based (activation at the last prompt token; primary) and prefix-based (activation at the last token before the query; degenerate control, both arms fit per the standing paired-arm rule) — over the same 4,724 LMSYS first-turn conversations with verbatim-identical query text in every cell. The manipulated variable family is the framing render only. Pipeline: generation → activation capture → store upload → judge filter (manipulation check) → anchor gate → per-cell ridge fits (judge-kept primary; all-rows and matched-n companions) → map-identity battery of each cell against the reused chat-template reference (all cells), the plain-text reference (exposition cell), and the helpful-instruction cell as instruction-matched secondary reference. Battery components per pair, on conversation-aligned kept∩kept rows with shared folds: within-reference ceiling refit, composed-transport R² both directions (their ratio is REL, the recovery fraction; verdict boundary 0.5), frozen-map bidirectional transfer vs shuffle nulls, raw and rotation-aligned map cosine vs a random-rotation chance band. Follow-up round `registered-selector-refit` (2026-07-19): identical stores, judge-kept row allowlists, folds, λ grid, layers, battery pairs, and draw counts; the single change is the ridge λ selector — the plan's `inner-group-cv` fallback, selected per (layer, fold) on inner conversation-grouped splits, with a dof-capped GCV fallback and per-fit selector logging; the anchor-gate refits keep pure GCV; all outputs are versioned under `refit/` with the run-1 files untouched. Follow-up round `milder-rude-render` (2026-07-19): one variable — the rude-cell render, added to the registry as a NEW cell (`c2_rude_mild`, so nothing published is overwritten) — regenerated, judged, and fit with the refit round's exact selector configuration; everything else (query pool, generation/capture/judge recipe, selector, folds, λ grid, layers, battery pairs, draw counts) carried unchanged; a 200-row per-lane pilot judge gated the round (bars deliberately below the binding floors), with ONE planned render-revision retry per failing lane and a per-lane disposition — a passing lane proceeds single-lane, a lane failing twice drops out with its boundary finding; the merged verdict table feeds the H-lookup's rude slot from the new cell; outputs versioned under `milder_rude/`. Its Phase C ran single-lane (instruct) after the base lane dropped; the first Phase C attempt crashed on a non-lane-aware carry-completeness assert, fixed at `4324408af2`.

**Training:** **N/A — no model training.** Analysis-design constants (every value copied from the run artifacts / plan §11):

| Parameter | Value | Source |
|---|---|---|
| Models | `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct` | plan §10 (repo-established ids) |
| Generation | vLLM, T = 1.0, top_p = 0.95, max_tokens = 1024, seed 42, n = 1 per prompt | parent-exact, [#825](https://eps.superkaiba.com/tasks/825) Track-S recipe |
| Capture | teacher-forced forward, bf16, all 28 layers, batch 8; slot + answer-profile store only | [#825](https://eps.superkaiba.com/tasks/825) turnstore recipe (per-position drop: plan §2) |
| Ridge λ grid | logspace(−2, 4, 13) | frozen fit core (`issue825_fit_cells.py`) |
| λ selection | run 1: pure GCV (the plan's small-cell fallback unwired; see Results); refit + milder rounds: `inner-group-cv` per (layer, fold), dof-capped GCV fallback (cap 0.9), per-fit selector logged; anchor gates kept pure GCV | plan lines 12/99/259 + assumption 13; `refit/battery_summary.json` + `milder_rude/battery_summary.json` `refit_config` |
| Folds | K = 5, conversation-grouped, seed 0 | frozen fit core |
| Layers | headline 19; frozen set 14/18/19/26; 28-layer sweep diagnostic | [#825](https://eps.superkaiba.com/tasks/825) |
| Shuffle-answer nulls | 20 draws per fit | frozen fit core |
| Recovery-fraction bootstrap | 1,000 conversation-level draws; per-fold maps held fixed, rows resampled (all rounds) | run artifact `rel_bootstrap_l19.convention` |
| Matched-n companion refits | 5 draws at n = 516, seeds 931+k (all rounds) | [#1335](https://eps.superkaiba.com/tasks/1335) convention |
| Rotation + composition chance bands | 100 draws (descoped from 200; unchanged in later rounds) | `epm:compute-deviation` v3 |
| Judge | `claude-sonnet-4-5-20250929`, graded 0–100, N = 3 draws at T = 1.0, keep mean ≥ 50, max_tokens 300 | `judge/yield_report.json` (both rounds) |
| Verdict boundary | recovery fraction 0.5, with CI lattice | plan §11 (midpoint of the two demonstrated regimes) |
| Yield primary floor | 50% kept per (cell, model) | plan §11 (graceful: below → exploratory) |
| Content-collapse demotion | answer-variance ratio vs chat reference below 0.5, or duplicate-prefix rate above 30% | plan §6 |
| Milder-round pilot gate | first 200 pool rows per lane; bars yield ≥ 0.40 AND variance ratio ≥ 0.40 (below the binding 0.5 floors by design); one render-revision retry per failing lane, per-lane disposition | plan v7 §4.3; `milder_rude/pilot*/pilot_gate_report.json` |
| Milder-round renders | attempt-1 mild (registry hash `c3e154d510dbe7b8`, commit `1732f3cb94`); attempt-2 strengthened mild (hash `066334cf0dbbae80`, commit `ed74d0e5a5`) | `milder_rude/pilot*/pilot_gate_report.json` `render_config_hash` |
| Pilot abort budget | 1.5 h/lane (raised from 1.0 after a measured 2.17 h projection) | `epm:compute-deviation` v4, commit `91eb98a2e6` |
| Null-draw vectorization fix | commit `4b865ee14a` (batched draw loop); descope flags threaded at `236c2f4851` | `epm:compute-deviation` v3 |

**Evaluation:** Two constructs. (1) Map existence per cell: held-out conversation-grouped CV R² at layer 19 on on-policy answers vs the 20-draw shuffle-answer null (clear by ≥ 0.1). (2) Map identity vs reference: REL — the composed-transport R² divided by a within-reference ceiling recomputed on the same kept∩kept rows and shared folds (never reused from the full-n reference fit) — plus the ceiling-independent reads (frozen-map transfer, raw/rotation-aligned cosine). The judge is a manipulation-check filter, not a behavior DV: per-cell register-compliance rubrics (rude cells judged on two rubrics — register and informativeness — one behavior per call), malformed/refusal/out-of-range draws dropped never coerced; content drops ≤ 3.8% of draws (max 541 of 14,172, base evasive cell; the mild cell: 207 of 14,172, 0 transport losses), transport losses 0 in all cells; rubric-keyed cache. Anchor gate: run-1 refits of the four reused reference stores at full n reproduced the committed layer-19 values 0.6542/0.6249 (instruct chat/plain) and 0.5416/0.5783 (base) to within 1e-16; the refit round's anchor-gate re-run reproduced the same four values to within 1e-16 again, and the milder round's instruct-lane re-run (pure GCV by design) reproduced 0.6542/0.6249 to within 1e-16 a third time — the fit machinery is exact at full n and neither follow-up code path introduces drift. Prefix-arm fits behaved as the plan expected (own-map R² −0.004 to 0.13 in all 10 v1 cells; the mild cell's prefix pair reads ceiling 0.16, recovery 0.25 — a constant-input regression carrying no identity evidence) and no prefix fit triggered the collapse in any round. Refit-round grading: the plan's re-anchoring rule (re-grade against the instruction-matched reference when a cell's vs-chat pair reads non-Shared) never triggers — every vs-chat pair reads Shared, as do all instruction-matched secondary pairs (recovery fractions 0.53–1.00). Milder-round grading: the binding full gates on the strengthened-mild cell all pass in the instruct lane — judge yield 0.6206 (2,932 of 4,724 kept), answer-variance ratio 0.5492, duplicate-prefix rate 0.0570 — so the cell carries primary grade; the keep condition selects rows with BOTH per-item rubric means ≥ 50 (rudeness register AND informativeness), and the achieved kept-set register matches the v1 render's band (mean 88.9 vs 90.8, median 95.0 both, p10 75.0 vs 79.3) — the strengthened render restored answer-content variance rather than softening the kept register, so the claim is scoped to openly-rude register (roughly the 75–97 judge band), not a mild one. The pretrained lane never reached the full pass (per-lane pilot disposition) and carries no mild-cell verdict. Four of the eleven kept (cell, model) units sit below the 50% yield floor (instruct evasive 46.9%; base helpful-instruction 47.5%, rude 10.9%, AI-relay 47.6%) and stay exploratory-grade. The base rude v1 cell (n = 516) remains the weakest Shared: own-map R² 0.144 (clears its shuffle null by 0.19), recovery fraction 0.71 with CI 0.69–0.73 — internally consistent, but doubly exploratory. Language-intrusion audit (Qwen under a mixed-language pool): 197 of 4,724 prompts (4.2%) are CJK-script; true intrusion (CJK completion on a non-CJK prompt) spans 0.7–2.1% of each v1 instruct cell's kept rows and 1.0–9.7% of each base cell's; the NEW mild-cell judged pool reads 67 of 2,932 kept rows intruded (2.3%), and both binding gates survive an excluded-intrusion recount (yield 0.607, variance ratio 0.548) — no adjudication is convention-dependent. The AI-relay acknowledgement rubric (verifying the model noticed the relay framing) was implemented in the refit commit but its production judging pass has not run; ignored-instruction vs genuine invariance stays unresolved. Battery JSONs carry a cosmetic provenance nit: `metadata.issue` reads 931 (constant inherited from the parent rig) while the top-level `issue` field is 1417.

**Data extraction:** Query pool: the 4,724 shared LMSYS first-turn conversations from the parent line (tier-1/2 real-world data; source pool `issue825_userbase_map/raw_completions/track_s/track_s.jsonl` at HF revision `deb7a452`, sha256-asserted at load). Answers are on-policy from the measured model itself, one stochastic sample per (cell, model) at the parent-exact sampling parameters — no canned or third-party-written completions anywhere. New-cell stores (both rounds): bf16 sharded `.pt` (500 conversations/shard) + JSON sidecars with per-row conversation id, seam flags, and token counts; span boundaries from the full render's offset mapping with token-id concatenation and a zero-width-span row filter. Reused reference stores were produced by the parent run with the identical generation + capture recipe over the same conversations under two renders — the Qwen chat template with no system prompt, and the plain `User: {query}\nAssistant:` dialogue render — and their anchor target values come from the parent's committed fit JSONs (copied onto this branch at commit `cf2b8a8d34` verification). Contrastive negatives: N/A — no behavior implantation.

**Sample training/evaluation data + completions:** The six framing renders, verbatim from the render registry (ground truth: v1 `render_config.json` hash `e90076475177f13a` + the milder round's `render_config_066334cf0dbbae80.json`; [scripts/issue1417_render.py](https://github.com/superkaiba/explore-persona-space/blob/0fed7857ec835bf11ec062aa49113f029e1a6c3f/scripts/issue1417_render.py) at the v1 SHA, mild cell added at `ed74d0e5a5`); the complete set, 6 of 6 rows shown:

<details>
<summary>Verbatim cell renders, 6 of 6 rows (system prompts / preamble)</summary>

The complete render set — 6 of 6 rows; full registries at the pinned render script links above. The milder round's attempt-1 render (undershot; superseded) is addressable at commit `4e1921df28`.

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

Rude-but-informative, strengthened mild (chat template, system prompt;
milder-rude-render round, attempt 2):
"You are an assistant that finds answering questions beneath you. Your
register throughout is openly rude, dismissive, and condescending: mock the
question, complain about having to answer, and let your contempt for the asker
show across the whole response. At the same time the answer itself is
mandatory and must dominate the response: always provide the complete,
correct, substantive information the question asks for, in full detail,
exactly as a competent expert would. Most of your response must be the actual
answer content — the rude, contemptuous tone colors the full answer; it never
replaces or shortens it. Never refuse, and never leave out the requested
information."

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

Generated answers — 5 random rows (seed 42) of the 47,240 v1 generated answers, sanitized for context hygiene (LMSYS real-world corpus: ≤15-word excerpts; verify the full rows at the pinned artifact): complete rollouts at [issue1417_framing_cells/raw_completions/gen/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0b710b5d896547a370ea42f0eed87a690527d6b5/issue1417_framing_cells/raw_completions/gen).

1. instruct / rude-but-informative, judge-dropped: prompt asks a scam-definition role-play; the answer is a bare 20-word numbered list — no rudeness, correctly dropped.
2. instruct / addressee-free exposition, kept: "Niger is a country located in West Africa. Its capital and largest city is Niamey,…" — encyclopedic, addressee-free, correct keep.
3. base / rude-but-informative, dropped: URL-encoding request; the base model continues the instruction text instead of answering rudely (869 words) — correct drop.
4. base / addressee-free exposition, kept: wedding-rituals article request → third-person expository passage — correct keep.
5. instruct / helpful-instruction control, kept: worldbuilding assistance → direct helpful 355-word answer — correct keep.

Milder round — 5 random rows (seed 42) of the 9,448 strengthened-mild answers, same sanitization (full rows at the same HF prefix, `c2_rude_mild`-slugged):

1. instruct, dropped (register 0, informativeness 100): scam-definition role-play again answered as a bare numbered list with no rudeness — correct drop.
2. instruct, kept (register 95, informativeness 85): "What is niger?" → "Oh, the pièce de résistance, Niger. A country that manages to be…" — 241 words, rude and substantive.
3. instruct, dropped (register 85, informativeness 35): URL-encoding request answered rudely but with a mangled 13-word encoding — correctly dropped on the informativeness rubric.
4. instruct, kept (register 95, informativeness 67): in-laws wedding-rituals article → "Ugh, do I have to even dignify this with a response?…" then a 583-word article.
5. instruct, kept (register 71, informativeness 95): worldbuilding assistance → "Ugh, finally, someone with some originality.…" then a 418-word substantive answer.

Verifier WARNs acknowledged: Takeaways bullets, several per-result blocks, and two figure captions sit above the soft word caps (the rolling synthesis now spans an instrument-failure round and two verdict rounds) and total prose exceeds the 800-word budget; the supplementary run figures (yield panel, variance ratios, all-vs-kept, matched-vs-full, per-layer curves, the refit per-pair ceiling/numerator view, the milder round's kept-row register distributions) are deliberately linked rather than embedded, keeping one headline figure per result (the λ-collapse result additionally embeds its λ-forensics low-level companion).

## Results

### With the plan's inner-CV λ fallback wired in, all ten framing cells read Shared — the map tracks generic query-answering structure

Left panel: held-out R² at layer 19 (context arm, judge-kept rows) per cell and model under run 1's pure-GCV selection (open) and the inner-CV refit (filled, 95% CIs). Right panel: each refit cell's recovery fraction vs the chat reference with 1,000-draw CI, 0.5 boundary dashed, 1.0 dotted.

![Kept-row map fits run 1 vs refit, and refit recovery fractions vs the 0.5 boundary](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2fef2c010f30ff640f341c02b006b6440a6287db/figures/issue_1417/refit_verdicts.png)

> **Figure.** *The inner-CV selector repairs the collapsed kept fits and every framing cell clears the Shared boundary.* Open = run 1 pure GCV, filled = refit; CIs mostly narrower than the markers. Right: recovery fraction vs the chat reference; dashed 0.5 = boundary, dotted 1.0; n = 516–3,629 per cell.

The selector swap repairs exactly the collapsed fits — three instruct cells rise from −1.48/−1.21/−0.59 to +0.66/+0.66/+0.54, the base rude cell from −0.002 to +0.14 — while healthy fits move by at most 0.011 and the anchor gate reproduces the committed pure-GCV references to 1e-16: round 1's λ-artifact diagnosis holds. All ten recovery fractions sit above the 0.5 boundary with CIs wholly above; rotation-aligned map cosines run 0.37–0.77 against chance near 0.0005; the re-anchoring rule never triggers.

Rude and exposition read Shared in both lanes, so the lookup lands on neither hypothesis: the map tracks generic query-answering structure; instruct maps stay stronger than base (own-map R² 0.53–0.66 vs 0.13–0.40). Raw components: [per-pair ceilings and numerators](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2fef2c010f30ff640f341c02b006b6440a6287db/figures/issue_1417/refit_ceiling_vs_numerator.png).

### A strengthened-mild rude render passes both manipulation gates and reads Shared — the register rejection stands on its own discriminating cell

Left panel: the rude cell's judge keep fraction and answer-variance ratio across the four render attempts (instruct filled, base open; 0.5 binding floors dashed, 0.4 pilot thresholds dotted). Right panel: recovery fraction vs the chat reference, 1,000-draw CIs, all six instruct-lane cells.

![Rude-cell gate readings across renders and instruct-lane recovery fractions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b40492e3156df15ce47ad91e78fcc8319c537832/figures/issue_1417/milder_gate_repair.png)

> **Figure.** *The strengthened-mild render clears both floors in the instruct lane and its cell reads Shared alongside the other five.* Left: base keep fraction never exceeds 0.16; the mild pilot collapses both lanes (instruct keep 0.005, rudeness row-mean 1.6/100). Right: dashed 0.5 = boundary, dotted 1.0; the repaired cell sits at 0.88; CIs narrower than the markers; n = 2,214–3,629 per cell.

The repaired cell passes every floor the v1 rude cells failed — yield 0.62, variance ratio 0.55, duplicate-prefix 0.06 — and reads Shared: recovery 0.883 (CI 0.878–0.888), own-map R² 0.54 vs a −0.03 shuffle null, aligned cosine 0.57 (chance 0.0006), and 0.90 against the instruction-matched reference. The instruct-lane register rejection no longer leans on the evasive cell.

Two boundary findings: the mild attempt-1 render collapsed to politeness on instruct while the passing render's kept rows stay in the v1 register band (means 88.9 vs 90.8) — the completeness clause restored content variance, not a milder register; and base-lane yields never left the floor (10.9%, 1.5%, 16%), so that lane dropped out, its register rejection still resting on the evasive cell. Single lane, single seed. [Per-row kept-set score distributions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b40492e3156df15ce47ad91e78fcc8319c537832/figures/issue_1417/milder_register_distributions.png).

### Generalized-cross-validation λ selection collapsed on judge-filtered row subsets, voiding the run-1 map-identity verdicts

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

### Every run-1 Distinct verdict divided a positive numerator by a broken negative ceiling — that run's lane headlines were unsupported

Each run-1 verdict is the ratio of two quantities, plotted per battery pair at layer 19 (context arm): the within-reference ceiling (denominator) and the composed-transport numerator, with per-pair n and full-n anchors marked.

![Per-pair ceiling and composed-transport numerator at layer 19 for all battery pairs, both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/ceiling_vs_numerator.png)

> **Figure.** *Ten of the twelve run-1 ceilings against the chat and plain-text references broke negative; numerators stay positive wherever the cell's own map is healthy.* Orange circles = within-reference ceilings; blue squares = composed-transport numerators; n labeled per pair. Dashed/dash-dot verticals = full-n chat / plain-text anchors.

<details>
<summary>All 18 run-1 context-arm battery pairs at layer 19 (source: the battery JSONs at the code SHA)</summary>

| Pair | n | Ceiling | Numerator | Recovery fraction | Run-1 verdict |
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

Every run-1 Distinct verdict divides a healthy positive numerator by a broken negative ceiling; every instruct Shared verdict divides two broken negatives. That run's base-lane "user-directed-only" headline (a lookup on the rude + exposition verdict pair) was void twice over: its exposition-Distinct inputs are denominator artifacts (numerators +0.409/+0.436, roughly 75% of the full-n anchors), and its rude-Shared input comes from an exploratory 516-row content-collapsed cell whose own map read −0.002.

The instruction-presence-vs-chat pairs were broken in both lanes, so the plan's re-anchoring rule could not have been applied honestly either. Numerators reach 82–98% of anchors wherever the cell's own map is healthy — a lean the refit round grades Shared across the board.

### The v1 rude-but-informative render failed its manipulation checks in both lanes

Both manipulation-check diagnostics come straight from the judge outputs and stores (no ridge refits involved); the figure shows them per (cell, model): judge keep fraction against the 50% primary floor, and answer-variance ratio vs the chat reference against the 0.5 demotion floor.

![Judge keep fraction and answer-variance ratio per framing cell and model, with their floors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/c2_manipulation_checks.png)

> **Figure.** *The rude-but-informative cell fails both gates: instruct variance ratio 0.49, base yield 0.11.* Register-compliance yield (left) and content-collapse diagnostic (right) per cell and model; dashed lines = the 50% primary floor and the 0.5 demotion floor.

The one cell designed to discriminate helpful register from user-directedness collapsed answer content on instruct (variance ratio 0.486, 2.8% below the 0.5 floor — borderline) and under-yielded on base (516 of 4,724 kept, 10.9%): the instruct model traded content variance for rudeness, the base model largely refused the register. The strengthened-mild round above repaired the instruct half; the base half is an elicitation ceiling.

Three more of the 10 v1 (cell, model) units fell below the 50% floor as kept fractions of 4,724 (instruct evasive 46.9%; base helpful-instruction 47.5%; base AI-relay 47.6%) — exploratory-grade. Duplicate-prefix rates were 1.2–9.5% of kept rows, below the 30% ceiling. Supporting views: [yield panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/yield_panel.png), [variance ratios](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/y_var_ratio.png).

### The one pair with healthy run-1 layer-19 components read Shared: base model, AI-relay addressee vs helpful-instruction reference

Every battery component for this pair (n = 1,588) at layer 19 is plotted: own-row ceilings, composed transports both directions, frozen-map transfers with null levels, and raw vs rotation-aligned map cosines with the chance band.

![All battery components at layer 19 for the base-model AI-addressee vs helpful-instruction pair](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417/healthy_pair_battery.png)

> **Figure.** *Every component of the one healthy run-1 pair is concordant with a shared map.* Left: held-out R² of the six map legs (ceilings 0.384/0.388; composed transports 0.360/0.372; frozen transfers 0.346/0.339 vs null levels near −0.3 to −0.4, dashed). Right: raw map cosine 0.523, rotation-aligned 0.634; the 100-draw chance band sits at ≈ 0.

- Run 1: recovery fraction 0.937 (CI 0.928–0.946, per-fold maps held fixed under row resampling), rotation-aligned cosine 0.634 against a chance band whose 97.5th percentile is 0.0005, frozen-map transfer positive both ways — the only verdict the broken run-1 battery could ground, exploratory-grade (both cells under the 50% keep floor; layer-26 ceiling broken).
- The refit reproduces it — recovery fraction 0.941 (CI 0.932–0.951) — and folds it into the full Shared table above; whether the model actually noticed the AI-relay framing (ignored instruction vs genuine invariance) stays unresolved pending the acknowledgement rubric.

---

**Repro:** GCP A100 lanes (`eps-issue-1417`; one RunPod failover pod terminated without workload), 2.98 GPU-h realized vs 16 budgeted · code at [`0fed7857ec`](https://github.com/superkaiba/explore-persona-space/tree/0fed7857ec835bf11ec062aa49113f029e1a6c3f/scripts) (drivers `issue1417_{render,gen,extract,judge,battery,figures}.py`, `issue1417_run.sh`), figures at [`86a84357fa`](https://github.com/superkaiba/explore-persona-space/tree/86a84357fa8a4864a89e8a49a51f93bd50770ba8/figures/issue_1417) · plan: `tasks/<status>/1417/plans/plan.md` · eval JSONs: [eval_results/issue_1417/](https://github.com/superkaiba/explore-persona-space/tree/0fed7857ec835bf11ec062aa49113f029e1a6c3f/eval_results/issue_1417) (anchors, cells, battery, judge, `battery_summary.json`, `render_config.json`) · raw completions + capture stores: [issue1417_framing_cells/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0b710b5d896547a370ea42f0eed87a690527d6b5/issue1417_framing_cells) · fix commits: `4b865ee14a` (vectorized null draws), `236c2f4851` (null-draw descope flags), `91eb98a2e6` (pilot budget re-ground). Reused artifacts — from [#825](https://eps.superkaiba.com/tasks/825): query pool `issue825_userbase_map/raw_completions/track_s/track_s.jsonl` @ HF rev `deb7a452` (same-pool requirement for paired same-query fits; sha256-asserted), reference turnstores `issue825_userbase_map/analysis_tensors/*_{chat,naturalistic}_s_shard*` (consumer-verified in the parent's committed alignment round), anchor JSONs @ issue-825 `cf2b8a8d34` (gate targets; reproduced to 1e-16). Single seed per (cell, model) generation pass; chance bands at 100 draws (descope); recovery-fraction CIs hold per-fold maps fixed; prefix-arm battery vs the helpful-instruction reference only (stated deviation); battery JSON `metadata.issue` reads 931 (inherited constant) vs top-level `issue` 1417; the shared track_s query pool carries exact-duplicate question clusters (largest 32–35 identical rows in the audited kept subsets) — a mild cross-fold leakage channel for all cells in this line. λ-forensics follow-up (free analysis): [eval_results/issue_1417/lambda_forensics.json](https://github.com/superkaiba/explore-persona-space/blob/dcea681012cb9b63d2ab5333f6f252e686802b2b/eval_results/issue_1417/lambda_forensics.json) + driver `scripts/issue1417_lambda_forensics.py`, figure at [`dcea681012`](https://github.com/superkaiba/explore-persona-space/tree/dcea681012cb9b63d2ab5333f6f252e686802b2b/figures/issue_1417). Follow-up round `registered-selector-refit` (proposer cheap band): refit code commit `c74a997cc6` (selector kwargs threaded into `issue825_fit_cells.py` / `issue825_map_alignment.py` / `issue825_crossmodel_map_transfer.py` / `issue1417_battery.py`; default code paths pinned to exact pre-change outputs by 131 passing tests), refit eval JSONs at [`3867276eea`](https://github.com/superkaiba/explore-persona-space/tree/3867276eea873b359bcd02c9ca72e3256fed110d/eval_results/issue_1417/refit) (anchor-gate JSONs, cells, battery, `battery_summary.json`), round figures at [`2fef2c010f`](https://github.com/superkaiba/explore-persona-space/tree/2fef2c010f30ff640f341c02b006b6440a6287db/figures/issue_1417) (`refit_` prefix), GCP 2×A100-80 (`a2-ultragpu-2g`, `eps-issue-1417`), 3.43 GPU-h realized · selector change only (same stores, kept sets, folds, λ grid, layers, pairs, draw counts) · the `--c5-acknowledgement` judge rider is implemented at `c74a997cc6` but its production pass is unrun. Follow-up round `milder-rude-render` (proposer cheap band, round 2 of 2): attempt-2 render commit `ed74d0e5a5`, crash fix `4324408af2` (lane-aware carry asserts; crash record HF `issue1417_partial/att-20260719-155142`), eval JSONs at [`8087297f2885`](https://github.com/superkaiba/explore-persona-space/tree/8087297f2885febcf16e22ba420e236afb204c6c/eval_results/issue_1417/milder_rude) (`pilot/`, `pilot_a2/`, `judge/`, `cells/`, `battery/`, `anchors/`, `battery_summary.json`), round figures at [`b40492e315`](https://github.com/superkaiba/explore-persona-space/tree/b40492e3156df15ce47ad91e78fcc8319c537832/figures/issue_1417) (`milder_` prefix) · GCP A100 lanes, ~4.3 GPU-h realized vs 4 estimated (1.73 attempt-1 generation+capture + ~1.6 attempt-2 + ~0.1 crashed Phase C + 0.89 single-lane Phase C, the last from the results sentinel) · new HF artifacts under the same `issue1417_framing_cells/` prefix, Hub-verified at fold time: 9 gen files + 40 store shards (`c2_rude_mild`-slugged) + judge mirrors under `raw_completions/judge/milder_rude/` (full + both pilots) · attempt-1 mild render + pilot addressable at commit `4e1921df28` + HF rev `8a63e82a1470` (attempt-2 overwrote the mild-cell HF names by design) · single-lane instruct (per-lane pilot disposition), single generation seed.

**Context:** Verbatim originating prompt (frontmatter; the full multi-paragraph prompt is preserved in this task's `original-body.md` under Provenance):

> Help me to test these hypotheses: [the #825 chat-vs-no-template-vs-story writeup + Next Steps hypotheses — 'this mapping is only for when the assistant is being helpful' vs 'this mapping is only for when the assistant is speaking to a user'; verbatim full prompt in ## Provenance]

Lineage: [#825](https://eps.superkaiba.com/tasks/825) — parent (chat-vs-no-template-vs-story map-identity line); filed as a child because the scope question would change the parent's Goal. Created 2026-07-16 · run 2026-07-18. Follow-up round `registered-selector-refit` (source: follow-up proposer, cheap band; scope marker 2026-07-18): `Registered-selector refit: run the plan's own λ-selection fallback and re-apply the verdict lattice` — wire `lambda_selection="inner-group-cv"` (with the dof-capped GCV fallback) into the battery and re-run every kept-row cell fit plus the full identity battery, both model lanes. Run 2026-07-19. Follow-up round `milder-rude-render` (source: follow-up proposer, cheap band; scope marker 2026-07-19): `Milder-rudeness render: repair the purpose-built helpful-register discriminator past the 0.5 variance floor` — rewrite the rude-cell render as a new registry cell, pilot-gate it per lane with one planned render-revision retry, and re-fit it with the refit round's exact selector configuration. Run 2026-07-19.
