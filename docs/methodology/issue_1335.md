# Methodology — issue 1335: 11-rung ablation ladder of context→answer ridge maps between the assistant Q&A and fiction-character endpoints (Qwen2.5-7B base + Instruct, plain text)

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

*Derived from the [task body](https://eps.superkaiba.com/tasks/1335).*
