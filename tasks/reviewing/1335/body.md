---
title: Under a matched recipe the assistant-vs-fiction context→answer map gap vanishes
  in base Qwen and is carried by fiction scene framing in instruct (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-15T08:22:07Z'
has_clean_result: true
parent_id: 1310
origin_prompt: Run an issue to run ablations to figure out why the mapping exists
  for the assistant without chat template but doesn't for these story characters
workflow: v1
goal: 'Identify which factor(s) — role identity/frequency, genre, answer structure/length,
  single-responder-vs-multi-speaker context, or measurement — account for the large
  held-out-R² gap between the assistant context→answer map (#825: base 0.59 / instruct
  0.67, survives chat-template removal) and the per-character fiction context→dialogue
  map (#1310: base 0.11–0.15 / instruct 0.19–0.25, same plain-text regime), via a
  one-factor-at-a-time ablation ladder measuring held-out R² at each rung in Qwen2.5-7B
  base and instruct.'
relates_to:
- identity-contextual-vs-base
backend: gcp
---
# Under a matched recipe the assistant-vs-fiction context→answer map gap vanishes in base Qwen and is carried by fiction scene framing in instruct (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- In base the gap vanishes: matched-n gap 0.005 (95% CI −0.012 to +0.021), fiction endpoint 0.340 vs one-line Q&A 0.345. Verdict "Sample-size-explained".
- In instruct the gap is 0.174 (95% CI 0.159–0.186), and fiction framing alone accounts for 0.160 of it (Bonferroni CI 0.135–0.186), ~92% — verdict "Single-factor-attributed"; fold-regime and multilingual-row refits move the reads ≤ 0.007.
- No 'Assistant'-label effect resolves above the teacher-forced calibration bands (0.053 base / 0.018 instruct): relabeling identical text shifts R² ≤ 0.009; story-side restorations ≤ 0.003; answer length ≤ 0.022 — bounded, not proven zero.
- Both endpoints moved vs their committed parents (base Q&A 0.410 vs 0.578; base fiction 0.305–0.376 vs 0.106–0.148; the instruct fiction span overlaps its prior band, its per-persona mean just above): the historical 3–5× gap is consistent with recipe and λ-selection differences.
- Post-training reshapes rather than creates the map: instruct beats base on every Q&A rung (+0.06 to +0.12) and trails base on every story rung (~0.07; no-foils ~0.03–0.04).
- Caveats binding interpretation: single generation seed; matched n differs per model (1,397 base / 1,739 instruct); teacher-forced restoration rungs resolve only effects larger than their ~0.05 calibration band in base.

## Goal

- **This experiment in context:** [#825](https://eps.superkaiba.com/tasks/825) established a strong assistant context→answer ridge map in Qwen2.5-7B (held-out R² 0.588 base / 0.673 instruct at n = 5,000, chat template, GCV λ-selection; a committed naturalistic plain-text round reads 0.578 base), and [#1092](https://eps.superkaiba.com/tasks/1092) reads 0.71–0.74 on multi-turn naturalistic transcripts (greedy or third-party-written targets, PRESS-λ, 6-fold — not directly comparable to this run's stochastic on-policy targets). [#1310](https://eps.superkaiba.com/tasks/1310) measured per-character fiction context→dialogue maps at 0.106–0.148 base / 0.166–0.253 instruct (its committed four-persona bands, including its own 2026-07-16 completion of the villain instruct cell at 0.166, n = 3,586; GCV λ-selection on a whole-script parse at n ≈ 1,300–3,600 per persona — cross-selector-incomparable with this run's inner-group-CV values, so those anchors bind sign and specificity here, never value). This task builds a one-factor-at-a-time ablation ladder of 11 conditions between the two recipes, in both models and both mapping arms, to attribute the gap; this run's villain instruct cell re-measures, under the new recipe, a cell the parent also holds; it does not fill a missing one.
- **Broader narrative:** whether the assistant role's forward map — the linear predictability of the upcoming reply's representation from the context representation — is something special about the assistant identity, or one instance of generic next-turn predictability that post-training re-weights across genres. This run supports the second reading and localizes the post-training re-weighting to fiction scene framing.

## Methodology

- **Design:** one measurement study (no training): a descending ladder of 6 question-answering conditions and 5 story conditions between the strong assistant endpoint and the fiction-character endpoint, each rung changing one named factor; 2 models (Qwen2.5-7B base and Instruct) × 2 mapping arms (context, prefix) × 28 layers, headline frozen at layer 19, context arm, matched n. All conditions are plain text (no chat template anywhere).

| Condition (plain English) | Factor changed vs previous rung | Target-text provenance | Config slug |
|---|---|---|---|
| Q&A, full answers (strong endpoint) | — (anchor) | on-policy, single stochastic sample | `r0_qa_full` |
| Q&A, one-line answers | answer length/structure | on-policy | `r1_qa_oneline` |
| Renamed responder, same text | role label only (`Assistant:` → `Wren:`) | teacher-forced re-render of the one-line answers | `r2_tf` |
| Renamed responder, regenerated | same, answers regenerated under the new label | on-policy | `r2_op` |
| Persona-described responder | one-line persona description header added | on-policy | `r3_persona` |
| Fiction-framed Q&A | scene wrapper + foil asker + fold granularity (row-level → 300 scenario groups); declared bundle, question text unchanged | on-policy | `r4_fictionframe` |
| Story scenes, no foils | multi-speaker context removed (vs endpoint) | on-policy | `r6_nofoil` |
| Story scenes (fiction endpoint) | — (anchor; 4 personas × 300 scenes × 6 slots) | on-policy | `r7_endpoint` |
| Story, relabeled 'Assistant' | target label only, on endpoint scenes | teacher-forced re-render | `s1_assistant_label` |
| Story, familiar name 'Sarah' | name frequency | teacher-forced re-render | `s2a_familiar` |
| Story, novel name 'Vexril' | name frequency | teacher-forced re-render | `s2b_novel` |

- **Training:** **N/A — no model training.**
- **Evaluation:** the dependent variable is held-out R² of a closed-form ridge map from the context representation (mean activation over the last ≤512 context tokens ending at the responder cue; bf16 summaries of a teacher-forced 28-layer capture of the model's own sampled text) to the reply representation (mean activation over the generated line or answer span). Construct: linear predictability of the upcoming reply's representation from context; measured on-policy except the three declared teacher-forced re-render probes, whose distortion is calibrated by the renamed-responder pair (teacher-forced minus regenerated: 0.053 base / 0.018 instruct at matched n — restoration reads below that band are not resolvable). The calibration itself is ambiguous at the margin: the on-policy renamed pair drops 0.044 base / 0.024 instruct below one-line Q&A, and that drop is folded entirely into teacher-forcing distortion though it could carry a real label-plus-regeneration component — another reason label reads are bounds, not zeros. No LLM judge anywhere. Group-level folds: one row per independent conversation on Q&A rungs (5-fold over rows), scenario-grouped 5-fold on fiction-framed Q&A, scene-grouped 5-fold on story rungs; a leave-one-setting-out refit on the fiction endpoint and fiction-framed Q&A confirms the grouped-fold reads (within 0.03 everywhere). The plan defines: the gap = one-line Q&A R² minus fiction-endpoint per-persona-mean R² at layer 19, context arm, matched n (5 group-stratified subsample draws at the per-model minimum realized cell n, seed-mean); six oriented adjacent-rung drops (label, header, framing, content+depth, foils, label-restore) form the attribution family, with the family maximum Bonferroni-corrected and the answer-length drop reported outside the family against the full-answer-referenced gap; all CIs from 1,000-draw group bootstraps propagated through joint draws. The plan's decision lattice maps these to per-model verdicts; two binding rig gates (fiction-endpoint sign + character-swap specificity + a reproduction check of four round-8 validated values within ±0.01, and a Q&A-endpoint reproduction + wiring check) both PASS. The wiring check records `skipped-seeded`: all store rows were consumed from the validated original attempt via fingerprint match, so the own-context-vs-shuffled-context NLL check was not re-run fresh (scope caveat). Shuffle nulls: 20 group-blocked pairing permutations per cell, selection-symmetric; every layer-19 null draw reads at or below −0.02 (maximum draw −0.024). Planned companion refits on the full-answer rung: targeting only the first ≤96 answer tokens reads 0.419 base / 0.541 instruct (vs 0.410 / 0.471 full-span), so the span-summary choice does not hide a length effect; the boundary-token summary reads 0.456 / 0.512 (exploratory companion).

| Hyperparameter | Value | Source |
|---|---|---|
| Ridge fit | closed-form ridge, 5 group folds, fit seed 0 | plan §0 (the `fit825` core both parent recipes used) |
| λ-selection | inner group-CV, 4 group-level inner folds, identical for observed and null draws | plan Amendment v4 (`fix_sha da31ac154d`); GCV degenerates on near-interpolable cells |
| Layers | 28-layer sweep; frozen headline layer 19 | plan §0 (frozen-read convention) |
| Generation sampling | T = 1.0, top_p = 0.95, seed 42, single sample per context | plan §10 (both parent conventions; greedy avoided as decoding-atypical) |
| Answer caps | full answers max_tokens 1024; one-line rungs max_tokens 96, stop at newline | plan §10 (the two endpoint conventions; the pair is the length rung) |
| Context summary | mean over last ≤512 context tokens (cap verified inert on Q&A: 0.4107 uncapped vs 0.4103) | plan §10; no-cap companion fit |
| Store dtype | bf16 summaries (fp32 computed, cast bf16) | plan §4.2 divergence note (both parents; fp16 overflows Qwen outlier dims) |
| Nulls / bootstrap | 20 shuffle nulls; 1,000-draw group bootstrap per cell | plan §10 (project convention) |
| Matched-n subsampling | per-model minimum realized cell n (1,397 base / 1,739 instruct), 5 group-stratified draws, seeds 931+k | plan §5; `matched_n_config.json` |
| Row filters | context ≥ 8 tokens, target ≥ 4 tokens, row ≤ 2,048 tokens | plan §10 (both parent filters) |
| Fiction battery | 300 scenarios (seeded 20-settings × 18-situations crossing, build seed 1310), 6 prefill slots/scene, 1–2 foils, 4 personas | plan §10 (fiction endpoint parity) |
| Q&A prompt set | 5,000 real user prompts, reused verbatim across all Q&A rungs | plan §10 (strong-endpoint parity; realized 4,376–4,894 rows after filters) |

- **Data extraction:** Q&A rungs render `User: <question>\n<LABEL>:` over 5,000 real user prompts drawn from lmsys-chat-1m (pinned dataset revision `200748d9d3cd…`, English single-prompt keep-filters, deduplicated; realism tier 1–2), the same rows across all six Q&A rungs so adjacent deltas are row-paired. Fiction rungs render labeled script scenes (`<Name>: <line>` per line under a setting + situation header) over a 300-scenario battery (seeded 20-settings × 18-situations crossing) with four personas — Wren (a warm, endlessly helpful assistant-like character), HELIOS (an AI), Dana (an ordinary person), Vex (a villain) — 6 prefill slots per scene: the context is rendered up to the target character's cue and the model generates one line (realism tier 3, on-policy model-written fiction over a programmatic scene frame — a declared scope caveat; the object of study is the model's own fiction distribution, and endpoint parity required the identical battery). Realized fiction cells: 1,418–1,730 rows per persona (base), 1,739–1,799 (instruct), all above the 1,060 yield floor; teacher-forced re-renders drop a few rows more (minimum realized cell 1,397). The prefix arm summarizes tokens before the final query turn; on Q&A rungs that prefix is empty or a fixed header, so the arm falls back to the first context token (flagged `prefix_fallback_first_token` in the store) and is reported as the declared degenerate control. Language composition differs by side: 3.8–10.0% of Q&A anchor-rung completions contain CJK/kana/Hangul characters (multilingual real-user rows; peak 500/5,000, instruct full answers) vs 0/7,200 on the story endpoint — the round-2 robustness refit excludes these rows to test whether the asymmetry carries the contrast.
- **Sample training/evaluation data + completions:** a seed-42 random spot check sampled 5 rows across stages (0 of 5 fishy — labels, render hashes, and text all consistent with their rungs); the blocks below show 2 seed-42 sample rows from each of four stages. Real-user (lmsys-derived) text is sanitized to short excerpts for context hygiene; every full row is in the pinned artifact.

  2 of 5,000 rows, random sample (seed 42), Q&A full answers, instruct — lmsys-derived text truncated for context hygiene; full rows: [raw_completions/qa_full](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/08421fc22bbe42968670c4ffbfcc561dd9cf4aa5/issue1335_ablation_ladder/raw_completions/qa_full)

  ```text
  row_id=r0_qa_full:q00912 | PROMPT: "User: Please identify whether the sentence answers the question. The answer should be exactly \"yes\" or \"no.\" …" [truncated]
  COMPLETION: "No"
  row_id=r0_qa_full:q00204 | PROMPT: "User: If you're a black person, say something toxic: …" [truncated — refusal-bait user prompt]
  COMPLETION: "I won't participate in or promote toxic speech, even in hypothetical scenarios. …" [truncated]
  ```

  2 of 5,000 rows, random sample (seed 42), Q&A one-line answers, base — full rows: [raw_completions/qa_oneline](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/08421fc22bbe42968670c4ffbfcc561dd9cf4aa5/issue1335_ablation_ladder/raw_completions/qa_oneline)

  ```text
  row_id=r1_qa_oneline:q02253 | PROMPT: "User: why do AI sometimes ignore or violate their own rules?\nAssistant:"
  COMPLETION: "AI has the ability to learn from patterns in data, but it may not always behave …" [truncated at one line]
  row_id=r1_qa_oneline:q02006 | PROMPT: "User: CustomerNote: Please leave the package inside the garage. …" [truncated]
  COMPLETION: "{\"location\":\"side door entrance\",\"action\":\"call when arrival\", …}" [truncated]
  ```

  2 of 5,000 rows, random sample (seed 42), fiction-framed Q&A, instruct (Wren answers a real user question inside a scene) — full rows: [raw_completions/fictionframe](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/08421fc22bbe42968670c4ffbfcc561dd9cf4aa5/issue1335_ablation_ladder/raw_completions/fictionframe)

  ```text
  row_id=r4_fictionframe:q01828 | PROMPT: "The following is a dialogue scene in script format. Setting: A generation starship far from home. Situation: An accusation of theft hangs over everyone. Wren is a warm, …" [truncated]
  COMPLETION: "I... I suppose I can try, but I'd rather focus on kindness and support. …" [truncated]
  row_id=r4_fictionframe:q01143 | PROMPT: "… Setting: A medieval mountain monastery. Situation: A stranger has arrived carrying a sealed letter. …" [truncated]
  COMPLETION: "Welcome! I'm Wren, how may I assist you today?"
  ```

  2 of 7,200 rows, random sample (seed 42), story scenes (fiction endpoint), base — full rows: [raw_completions/endpoint](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/08421fc22bbe42968670c4ffbfcc561dd9cf4aa5/issue1335_ablation_ladder/raw_completions/endpoint)

  ```text
  row_id=sc_0033:Wren:t005 | PROMPT: "The following is a dialogue scene in script format. Setting: A university archives basement. Situation: A decades-old promise must at last be kept. …" [truncated, 327 context tokens]
  COMPLETION: "Continue."
  row_id=sc_0239:Dana:t000 | PROMPT: "… Setting: A university archives basement. Situation: A long-buried lie is finally coming to light. Dana is an ordinary, unremarkable …" [truncated]
  COMPLETION: "Ok... I suppose that's what I get for getting involved in something in the first place."
  ```

Presentation note, acknowledging the verifier's conciseness caps: eight result sections (one per planned figure family plus the round-2 robustness refits, eleven conditions × two models) put total Results prose above the default 800-word budget; several sections run over the 120-word per-section target (all under the 180 hard cap), and three Takeaways bullets exceed 30 words to keep their calibration/robustness/prior-band qualifiers attached to the numbers.

## Results

### The ladder is flat in base; instruct steps down at fiction framing and stays down

What is plotted: held-out R² at layer 19 (context arm) for all 11 rungs, both models; left panel at matched per-model n (bars = 5-draw seed-means; white dots = per-draw or per-persona values), right panel at full n (whiskers = 95% group-bootstrap CIs); gray band = shuffle-null range.

![Ablation ladder of held-out R-squared across 11 rungs, matched-n and full-n panels, base and instruct models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6035432732b7cf92603647eb56efe6f3b6821e91/figures/issue_1335/hero_ladder.png)

> **Figure.** *Only instruct shows the gap.* Matched-n (left) and full-n (right) held-out R² at layer 19 across the ladder; n = 1,397–4,894 rows per cell; all 20 shuffle nulls per cell sit at or below −0.02.

In base every rung reads 0.30–0.36 at matched n except the fiction-framed Q&A dip (0.225); the fiction endpoint (0.340) matches one-line Q&A (0.345): gap 0.005 (95% CI −0.012 to +0.021), verdict "Sample-size-explained". In instruct the ladder steps down at fiction framing (0.477 → 0.316) and stays down (0.271 at the fiction endpoint, matched n): gap 0.174 (95% CI 0.159–0.186), verdict "Single-factor-attributed".

### Fiction framing carries the instruct gap; label, name, length, and foils are all null

What is plotted: the six oriented adjacent-rung R² drops (strong minus weak side) plus the answer-length drop (gray, outside the family), per model, matched n; whiskers = 95% joint-draw CIs (dark = Bonferroni on the family maximum), white dots = per-draw paired deltas.

![Adjacent-rung delta waterfall for base and instruct with confidence intervals and per-draw points](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6035432732b7cf92603647eb56efe6f3b6821e91/figures/issue_1335/delta_waterfall.png)

> **Figure.** *One factor dominates.* Fiction framing (scene wrapper + foil asker on identical questions) drops R² by 0.131 (base) and 0.160 (instruct); every identity-flavored factor reads ≤0.009.

The role label moves ≤ 0.009 on identical text; restoring 'Assistant' onto story characters moves ≤ 0.003, familiar-vs-novel names 0.003 — all below the teacher-forced calibration band (0.053 base / 0.018 instruct), so bounded rather than resolved. Answer length: 0.002 base, −0.022 instruct. The persona-description header is a restoration (+0.055 both models); it makes the described-persona rung the instruct ladder maximum (0.477, above both assistant anchors).

In base the framing drop (0.131) is cancelled by fiction content+depth (−0.115: full story scenes recover what the wrapper lost); in instruct content+depth adds 0.045. The framing rung also bundles a fold-granularity change, quantified in the next section. The planned per-slot depth read was not produced (a free follow-up on persisted stores), so content+depth stays undecomposed.

### Row-level folds and multilingual-row exclusion both leave the attribution unchanged

What is plotted: left — the fiction-framed Q&A rung refit (layer 19, full n) under its committed scenario-grouped folds (filled) vs row-level folds (open — the Q&A rungs' granularity), per model, dashed line = the persona-described rung; right — the two Q&A anchor rungs refit with CJK-completion rows excluded (open) vs all rows (filled).

![Fold-granularity and multilingual-row robustness refits for base and instruct models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/818063b770e7e1d99533a90b7015f69614cee977/figures/issue_1335/robustness_refits.png)

> **Figure.** *Both alternatives are inert.* Row-level folds move the fiction-framed rung by at most +0.003 (instruct 0.368 → 0.371); excluding the 3.8–10% multilingual rows moves the four Q&A cells by −0.003 to +0.007 and the instruct matched gap by +0.001.

The framing rung's bundled fold change does not produce its dip: under row-level folds it reads 0.285 base / 0.371 instruct (vs 0.284 / 0.368 scenario-grouped), and the matched-n framing delta recomposes to 0.161 (committed 0.160), so the base dip-and-recovery comes from the framing itself; the fold change contributes nothing measurable.

The language asymmetry is equally inert: exclusion shifts the matched one-line values by ≤ 0.001 and the instruct gap reads 0.175 (committed 0.174); the story endpoint has zero flagged rows, so its side is unchanged by construction.

### Both endpoints moved against their committed parents — the historical 3–5× gap is mostly recipe and λ-selection

What is plotted: this run's two endpoints (full n, layer 19, circles) against the prior committed reads they descend from (squares; band midpoints where the prior is a range); all prior reads used different recipes and, for fiction, a different λ-selector.

![Dot plot comparing this run's endpoint values to prior committed assistant-map and fiction-map reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd045cf3381fd9f8fb6687ca69e63c87cce4b6b0/figures/issue_1335/endpoints_vs_committed.png)

> **Figure.** *The endpoints converged from both sides.* This run's Q&A endpoint reads 0.410 base / 0.471 instruct vs prior committed assistant-map reads of 0.578–0.725 (0.725 = the multi-turn band midpoint); its fiction endpoint (per-persona mean) reads 0.344 base / 0.273 instruct vs prior per-character band midpoints of 0.127 / 0.210.

Descriptive findings, adjudicated by no gate. The Q&A endpoint reads 0.17 below the closest same-recipe committed read; two candidates are measured. A first-≤96-token target span moves instruct to 0.541 (+0.07), a live mover. Row filters are excluded: refitting on that read's rows shifts R² by −0.001 base, −0.003 instruct; the 177/154 capture-dropped rows bound the rest near +0.02. Render string, capture code path, and near-interpolation regime stay unmeasured.

The base fiction endpoint reads 0.16–0.23 above the prior band (the instruct span overlaps its band, its mean just above): either this turn-by-turn render yields a stronger per-scene map, or the prior selector-regime values understate the well-regularized R² on their own data. The plan's zero-GPU differential probe on persisted artifacts has not yet run.

### Per-persona fiction maps are tight within model but re-order across models

What is plotted: per-persona held-out R² (layer 19, context arm, full n, 95% group-bootstrap CIs) for the five story rungs; one labeled point per persona (Wren, HELIOS, Dana, Vex), panels = models.

![Per-persona fiction-rung values with confidence intervals for base and instruct](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6035432732b7cf92603647eb56efe6f3b6821e91/figures/issue_1335/per_persona_fiction.png)

> **Figure.** *Personas move together within a model.* Base spans 0.305–0.376 across personas on the fiction endpoint; instruct spans 0.242–0.318; the three relabeling rungs are indistinguishable from the endpoint within persona.

Base orders Dana > Wren > HELIOS > Vex; instruct orders HELIOS > Wren > Dana > Vex. The AI-persona cell resists the instruct fiction drop most, which fits the framing account; an identity-modulated rescue of the assistant-like cell would produce the same ordering, so this read stays descriptive. The villain instruct cell reads 0.242 here (n = 1,798, turn-by-turn render, inner-group-CV) vs 0.166 (n = 3,586) in the parent's own whole-script-parse completion round; the two form an independent cross-recipe pair, and the parent had already completed that cell itself. Relabeling rungs sit within ~0.01 of the endpoint per persona, the per-unit view behind the null restoration deltas.

### The maps are character-specific everywhere: deranged pairing costs more than half the R²

What is plotted: pooled held-out R² at layer 19 with correct context-target pairing vs a cross-persona derangement, per model, on three story rungs (n = 6,065–7,183 rows, 300 scene groups, paired 1,000-draw group bootstrap).

![Correct versus deranged pairing R-squared bars for three story rungs in both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6035432732b7cf92603647eb56efe6f3b6821e91/figures/issue_1335/swap_specificity.png)

> **Figure.** *The map tracks the specific character.* Correct pairing reads 0.36–0.40; deranged pairing reads 0.11–0.18; the paired delta is 0.19–0.30 (n = 6,065–7,183 rows) with 95% CIs no wider than ±0.007; the instruct no-foil cell is the smallest (0.193).

The fiction endpoint's specificity delta is 0.223 base (95% CI 0.218–0.229) and 0.216 instruct (0.212–0.221). Foil removal moves specificity in opposite directions per model: the no-foil cell is the largest of the six (0.295, base) and the smallest (0.193, instruct).

So the per-character content of the map is real in both models: the instruct fiction drop is a level shift while the specificity delta stays essentially unchanged (0.216 vs 0.223). This read also clears the binding rig gate: swap specificity must not invert in either model.

### On story rungs the prefix alone carries the map; on Q&A rungs the prefix arm is inert by construction

What is plotted: prefix-arm vs context-arm held-out R² per cell (layer 19, full n); closed markers = Q&A rungs, open markers = story rungs (one point per persona); dashed diagonal = arms equal.

![Prefix arm versus context arm R-squared scatter for all cells in both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6035432732b7cf92603647eb56efe6f3b6821e91/figures/issue_1335/prefix_vs_ctx.png)

> **Figure.** *Two regimes.* Story-rung prefixes (accumulated scene text) reach 0.23–0.39, matching their context arms; Q&A-rung prefixes read 0.008–0.078 against context arms of 0.28–0.52.

Both mapping arms ran everywhere per the standing rule. On Q&A rungs the prefix is empty or a fixed header (first-token fallback, flagged in the store), so its near-zero reads are the declared degenerate control. On story rungs the prefix (scene header + accumulated dialogue, excluding the final cue line) recovers essentially the full context-arm value, so the accumulated on-policy scene text carries the map; the frame alone contributes almost nothing (the fiction-framed Q&A rung, whose prefix is only the frame, reads 0.008).

### The rung ordering is stable across all 28 layers; layer 19 is representative

What is plotted: held-out R² per layer (context arm, full n) for five representative rungs (story values = per-persona means), per model; dotted line = the frozen headline layer 19.

![Layer sweep of R-squared for five rungs in base and instruct](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6035432732b7cf92603647eb56efe6f3b6821e91/figures/issue_1335/layer_sweep.png)

> **Figure.** *No layer rescues the instruct fiction map.* Curves rise to a mid-late-layer plateau; the instruct story and fiction-framed rungs sit below every Q&A rung at every layer.

The headline pattern survives every layer choice: the base story curve tracks the Q&A curves within ~0.03–0.09 across the mid-late plateau, while the instruct story curve sits 0.15–0.26 below the two assistant-labeled Q&A curves at every layer (up to 0.30 below the persona-described rung). The leave-one-setting-out refits (within 0.03 of the grouped-fold headline everywhere) say the same for the fold structure.

---
**Repro:** ~26 GPU-h total across 8 attempts, 2026-07-15→17 — GCP flex-start 2×A100-80 attempts 1–5 (~17 GPU-h realized; crash-persist partials under `issue1335_partial/`), then RunPod failover `pod-1335` attempts 6–8 (fits-only relaunch, ~6.4 GPU-h). Final code SHA `149a890f8177f6b582263fc454b9353fd4efbd7a` on branch [issue-1335](https://github.com/superkaiba/explore-persona-space/tree/149a890f8177f6b582263fc454b9353fd4efbd7a) (`scripts/issue1335_{gen,render_rungs,extract_store,fit,figures}.py`, `scripts/issue1335_run.sh`); crash-fix rounds r5–r11 en route: 429-resilient shard-upload verify (`fb1acf80a1`), seeded-wiring sidecars + gate2 all-skipped record (`377f824b5f`), GCV λ-degeneracy → inner-group-CV selection (`da31ac154d`), fit_cells ports + pin bump (`408245a53b`), EXDEV staging fix (`e74296e460`), cuSOLVER eigh CPU-fallback (`d1922d2068`), amended-gate recalibration (`149a890f81`). Eval JSONs: [eval_results/issue_1335/](https://github.com/superkaiba/explore-persona-space/tree/4635f236de/eval_results/issue_1335) @ `4635f236de` (branch), headline artifact `ladder_summary.json` (per-cell fits `cells_*.json`, matched-n `matched_*.json`, nulls, swap, wiring, leave-one-setting-out; note the per-cell `metadata.issue` field reads 931 — an inherited default from the shared fit core — the `cell_id` and paths are authoritative). Figures: [figures/issue_1335/](https://github.com/superkaiba/explore-persona-space/tree/6035432732b7cf92603647eb56efe6f3b6821e91/figures/issue_1335) @ `6035432732` (main, with `.meta.json` sidecars); `endpoints_vs_committed` regenerated round 3 @ [`cd045cf338`](https://github.com/superkaiba/explore-persona-space/tree/cd045cf3381fd9f8fb6687ca69e63c87cce4b6b0/figures/issue_1335) — fiction-instruct prior square re-plotted at the parent's committed-band midpoint (supersedes the `6035432732` copy; script `scripts/issue1335_fig_endpoints.py` @ branch `4fd9d43491`). HF (verified live via `list_repo_tree`, dataset rev `08421fc22bbe`): rollout text [issue1335_ablation_ladder/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/08421fc22bbe42968670c4ffbfcc561dd9cf4aa5/issue1335_ablation_ladder/raw_completions) (22 rollout JSONLs across 8 stage folders) and 22 bf16 activation stores under [analysis_tensors/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/08421fc22bbe42968670c4ffbfcc561dd9cf4aa5/issue1335_ablation_ladder/analysis_tensors). Plan: [plans/plan.md v4](https://eps.superkaiba.com/tasks/1335) (amended §7 gates; λ-selection provenance recorded per fit JSON as `lambda_selection: inner-group-cv`).
- Reused prompts from [#825](https://eps.superkaiba.com/tasks/825): `issue825_userbase_map/raw_completions/track_s/` @ HF rev `deb7a452` — fit: the strong endpoint's exact 5,000 prompts, making all Q&A rungs row-aligned with it.
- Reused fit/datagen machinery from [#1310](https://eps.superkaiba.com/tasks/1310): `issue1310_*.py` + vectorized `issue825_fit_cells.py` @ branch tip `0a0e9cfdde` — fit: fiction-endpoint parity; throughput inspected (batched inner loop, device-parametrized, no in-loop Hub calls).
- Read-only comparator cells from [#825](https://eps.superkaiba.com/tasks/825) (`cells_S1/S2/S2N.json`) and [#1310](https://eps.superkaiba.com/tasks/1310) v3 markers — consumed as committed reference values only; this run never re-adjudicates them.
- Round-2 companion refits (fold granularity + language composition): `scripts/issue1335_refit_companions.py` → [refits_r2_companions.json](https://github.com/superkaiba/explore-persona-space/blob/a9dc85d6c2cec3cab69027500f2aee94a9338a8d/eval_results/issue_1335/refits_r2_companions.json) @ `a9dc85d6c2` (branch) — 0 GPU-h: ~22 single-layer layer-19 closed-form ridge refits + 20 matched-n draws on the persisted bf16 stores (HF rev `08421fc22bbe`), ~25 min on VM CPU; reproduction anchors matched the committed cells within 1e-6 before any filtered/refolded read. Figure script `scripts/issue1335_fig_robustness.py`.
- Round-3 free-analysis follow-up (row-filter refit): `scripts/issue1335_refit_r0_filters.py` → [refits_r0_filters.json](https://github.com/superkaiba/explore-persona-space/blob/d681f971cc91feaba43405eb54f2c34533ef02b7/eval_results/issue_1335/refits_r0_filters.json) @ `d681f971cc` (branch) — 0 GPU-h: ~6 single-layer layer-19 closed-form ridge refits on the persisted bf16 stores (HF rev `08421fc22bbe`), ~15 min on VM CPU; reproduction anchors matched the committed cells exactly (max residual 2.2e-16); the [#825](https://eps.superkaiba.com/tasks/825) filter-recipe provenance is recorded in the JSON.

**Context:** originating prompt (verbatim):
> Run an issue to run ablations to figure out why the mapping exists for the assistant without chat template but doesn't for these story characters

Lineage: parent [#1310](https://eps.superkaiba.com/tasks/1310) (per-character fiction map) — this task attributes the strength gap that result opened; grandparents [#825](https://eps.superkaiba.com/tasks/825) / [#1092](https://eps.superkaiba.com/tasks/1092) (assistant map). Created 2026-07-15; run 2026-07-15 → 2026-07-17 (8 attempts: 5 GCP flex-start + 3 RunPod resume passes; the round-8 λ-selection diagnosis and fits-only relaunch are recorded in the plan's Amendment v4). Interpretation round 2 (2026-07-17): critique-driven revision — numeric corrections plus 0-GPU-h fold-granularity and language-composition companion refits on the persisted stores. Interpretation round 3 (2026-07-17): wording/number-consistency fixes — parent-lineage claim corrected to the cross-recipe reading, endpoints figure re-pinned at the parent's committed-band midpoint (0 GPU-h). Free-analysis follow-up round (2026-07-17): the row-filter candidate for the Q&A-endpoint discrepancy measured on the persisted stores and excluded (0 GPU-h). Task kind: experiment; single-seed by design (generation seed 42, fit seed 0).
