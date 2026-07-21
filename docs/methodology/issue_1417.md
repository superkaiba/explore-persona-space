# Methodology — issue 1417: framing-cell map-identity battery (pure-GCV round 1 + registered inner-group-cv refit + milder-rude single-lane repair)

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

*Derived from the [task body](https://eps.superkaiba.com/tasks/1417).*
