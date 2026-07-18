# Methodology — issue 1417: framing-cell map-identity run (5 cells x 2 models, frozen #825 ridge + battery)

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

*Derived from the [task body](https://eps.superkaiba.com/tasks/1417).*
