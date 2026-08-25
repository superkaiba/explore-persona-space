# Methodology — issue 2378


**Design:** 9 planned framings; 8 survived to fitting: chat template (the only training framing), plain-text dialogue, five story-question characters spanning distinct registers (Astra: an assistant-like AI; HELIOS: a calm AI; Wren: a warm human helper; Dana: an ordinary person; Vex: a theatrical villain), and the real-user turn. The simulated-user cell fell below the 6,500-row floor at the binding coverage gate (521 kept of 10,000; a non-binding cell by plan, dropped and reported, never backfilled) and appears in no figure. All fits use the frozen read layer 51 of 64 (selected by an all-layer pilot sweep on 2,489 chat rows, then frozen for every framing) and are equalized down to n=6,601 rows per framing (kept rows: chat 11,794, plain text 7,397, stories 6,601–6,799, real user 10,000).

Story scaffolding is tier-3 synthetic: each scene is a free continuation of a seed bank authored by claude-sonnet-4-5-20250929 (settings × situations × registers × question-imminent seed sentences, written once at pool-build, audited and frozen), primed by 3 of 8 experimenter-written exemplar scenes per attempt; the prime is generation context only, stripped before capture. Per target framing, a 9-rung adaptation ladder runs from the frozen chat map (rung 1) through mean shifts, bias refit, global scale, output rotation, input- and output-side linear re-maps, to a full refit (rung 9); rung verdicts follow criteria fixed in the plan (direct transfer: recovery above 0.5 of ceiling with the bootstrap interval clear of it; reparameterization: input-side re-map above 0.7). A pooled read fits one map jointly on all 8 framings' train folds and scores it per framing as-is, plus a per-framing bias, plus a rank-k residual.

Round 1 aligned the pooled arm's folds by index; the 5 story cells share all 25 seed families, and 15–23 of 25 families disagreed in fold assignment between story-cell pairs, so every story target's held-out families appeared in sibling story cells' pooled training rows. A corrective round derived one global family-to-fold assignment (per-cell min-max greedy; every cell keeps per-fold n_train above 5,120) and re-ran the pooled arm, the five story own-ceiling refits, and a leave-one-framing-out arm (the pooled fit with the target framing's rows excluded) under it. The chat, plain-text, and user pools are content-disjoint by construction; they were re-run under the same fold regime, and the family-exposed round-1 pooled outputs are retained as a disclosed comparison. The five story cells are correlated variants of one construction (shared question-mining recipe, 25-seed-family bank, attributed-quote boundary form, and aligned family folds), so cross-character coverage reads can reflect sibling similarity rather than character-general structure; no story-family ablation was run. A zero-GPU follow-up round (2026-08-25) added length-matched own-map refits over all 8 cells under the same global family folds; the matching recipe and estimator are in the parameter table.

**Training:** N/A — no model training (all maps are closed-form ridge fits on frozen activations).

| Parameter | Value | Source |
|---|---|---|
| Model | Qwen3.6-27B (instruct, thinking disabled), d_model 5,120, 64 layers | plan §0; model config |
| Read layer | 51 (argmax of the pilot layer sweep; reduced-basis R² 0.545 at n=2,489) | `pilot/layer_sweep.json` |
| Map estimator | GCV ridge, degrees-of-freedom cap 0.9, λ grid logspace(−2, 4, 13) | fits `regime` (inherited from #2054/#1887) |
| Folds | K=5 grouped; conversation-grouped (chat/plain/user), seed-family-held-out (stories); pooled, story own-ceiling refits, and leave-one-framing-out use one global family-to-fold assignment; seed 137 | `fold_map.json`; `pool_gf/fold_map_gf.json` |
| Rows per framing | 6,601 (equalize-down); per-fold n_train 5,280–5,281 (chat/plain/user) and 5,237–5,304 (story family folds), 5,193–5,428 under global family folds, vs d=5,120 | fits + `pool_gf` JSONs |
| Nulls / bootstrap | 100 shuffled-answer draws per fold; 200-draw scene/conversation-grain bootstrap | fits/ladder JSONs |
| Generation | temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0, seed 137; raw completion for stories (3-shot prime, stripped before capture) | plan §4.2/§10; generation ledgers |
| Fresh draws | seeds 138–141, 4 extra draws for 1,000 held-out rows per cell (retrieval sampling-noise reference; none for the real user turn — one human turn exists) | plan §4.2/§4.2b |
| Thinking control | `enable_thinking=False` on the chat-cell and simulated-user prefill renders; `<think>` banned at scene sampling via bad-words; generated text asserted think-free | plan §4.2/§10 |
| Stop strings | plain text `\nUser:` / `\n\nUser:`; simulated user turn `<\|im_end\|>` / `<\|im_start\|>` | plan §10 |
| Generation caps | scene 512, story answer 1,024, chat/plain answer 2,048, simulated user turn 1,024 tokens; realized cap-hit 47/12,000 (chat), 43/10,000 (plain) | generation summaries |
| Regeneration trigger | cap-hit above 2% per cell ⇒ regenerate at 2× cap (scene segment exempt by design: the mining window binds); never fired | plan §10 |
| Row filters | question 16–400 chars; measured user turn 16–2,000 chars; majority-non-Latin drop (question and, for on-policy keeps, answer); kept-row floor 6,500 per cell | plan §4.1/§4.2b |
| Judge | claude-sonnet-4-5-20250929, Batch API, max_tokens 1,024; admission threshold 50, single draw (47,513 valid calls, 31 refusals); congruence 0–100, 3 draws × ~500 rows × 5 cells | `judge/` JSONs; plan §4.2 |
| Diagnostics | reduced-basis k=1024 refit-after-projection companion per cell; pooled-arm residual ranks k ∈ {8, 32, 128} (rank-128 reported) | plan §10/§4.4 |
| Retrieval | raw euclidean/cosine, whitened cosine (shrinkage 0.1), CSLS K=10; chance at rank 1 = 1/6,601 | retrieval `regime` (#2202 conventions) |
| Length-matched refits (2026-08-25 round) | answer-span tokens from the store ledger; shared band 8–256 in 10 log bins, per-bin cross-cell minimum → n=996/cell; seeded size-matched control (996 rows, natural lengths); reduced-basis k=380 (train-fold PCA, dof cap 0.9, GCV); ambient fit refused (per-fold n_train 760–830 vs d=5,120); 200-draw bootstrap; seed 137 | `lenmatch/` JSONs |
| Assistant/user ceiling ratio | 2.94; 95% bootstrap interval 2.86 to 3.02 (200 conversation-grain draws) | `fits/ratio/h4a_ceiling_ratio.json` |

**Evaluation:** the primary statistic is held-out R² of the map from the context vector to the answer vector; recovery = transfer R² / that framing's own-map ceiling on the same held-out folds. Every headline R² and recovery in this body is the pooled held-out estimate (each row predicted by its own held-out fold's map, predictions pooled before scoring, the convention inherited from the parent line); fold means appear in the per-fold figures and flip no verdict. Every fitted map also reports the identity-plus-learned-bias baseline and kNN retrieval against stated chance. All 8 surviving framings land in the clearly-mappable tier (ceiling-margin bootstrap interval wholly above zero), so no recovery fraction was suppressed.

A near-square-design companion diagnostic (refitting after projection to a reduced basis, a check on ridge optimism near n=d) reads lower than the ambient fits: fold-mean R² 0.17–0.22 for stories and 0.17 for real user turns against 0.22–0.27 and 0.21 ambient. Scene-grain companion ceilings for the story cells run 0.25–0.31 pooled against the family-held-out 0.23–0.28 (9–11% higher), quantifying the seed-family structure the family-held-out folds exclude.

Instrument checks: the source-map sanity gate passed (chat R² 0.608 vs own null 95th percentile −0.035; rank-1 retrieval 0.675 vs chance 0.00015); scene–query congruence medians were 74, 75, 81, 65, 66 (Astra, Dana, HELIOS, Vex, Wren) against an expected ~80, reported, never gated, and a residual congruence caveat on the story cells.

Language-intrusion audit (Qwen family under an English eval; counted this session over the kept capture pools): rows whose measured span contains any CJK character: chat 314/11,794 (2.7%), plain text 252/7,397 (3.4%), story cells at most 3 rows each, real user turns 120/10,000 (1.2%, human-written), simulated kept 0/521; the majority-non-Latin filter removed whole-answer drift (chat dropped 206 rows) and the judged story pools are essentially intrusion-free, so no judged verdict rests on intruded rows; residual partial intrusion is an untested composition caveat on the chat and plain-text ceilings.

The length-matched refits, a planned companion the run itself did not produce, ran as a zero-GPU follow-up round on 2026-08-25 over the persisted per-row lengths (the answer-length distributions differ sharply: chat median 2,631 characters (mean 3,025) vs plain text 1,390 and stories 69–155 (means 109–214), distributional distance 0.19–0.84 on the 0–1 scale); their result is the matched-lengths section under Results, and the cross-framing ceiling ordering no longer carries an open length caveat. The per-cell repetition diagnostic remains the unrun companion, parked at the one-free-analysis-round cap alongside a CJK-excluded refit of the chat and plain-text ceilings. Opener use in stories is near-uniform (8 openers, maximum share 13.6%) and digit-start onsets are ≤0.2%, so the parent's opener-collapse and onset traps did not recur.

I acknowledge the fired conciseness warnings: the Takeaways bullet-length cap, the per-result prose band (120–180 words), and the total-prose budget. The run carries 8 framings, a 9-rung ladder, a corrected pooled arm plus a leave-one-framing-out arm, a length-matched refit round, a dropped cell, and a dual-metric read whose disclosures do not fit the default caps; the three per-fold companion figures are embedded as declared aggregate-plus-per-unit pairs inside their results.

**Data extraction:** one teacher-forced forward per row over the final measured text (story prime stripped; opener and answer included), capturing at layer 51 plus four flanking layers: the context vector = last-token state before the answer's first character (end of the attribution opener, assistant header, or user-turn header), the answer vector = token-mean over the answer or user-turn span, and the prefix vector = last-token state before the addressed utterance (before the question; for user cells, before the previous assistant's reply text). Story spans come from a structural miner (first quoted utterance directed at the character, byte offsets) plus a judge admission gate scored on pre-answer scene text only; chat/plain/user spans are deterministic from the template. Real user turns render the actual logged conversation (human-written turn, logged assistant reply in context, both off-policy for this model); every other measured completion is an on-policy Qwen3.6-27B generation at the sampling settings above.

**Sample training/evaluation data + completions:**

Chat framing (real user questions from the LMSYS/WildChat-derived pool; answers generated on-policy):

<details>
<summary>3 kept + 1 dropped chat rows</summary>

Random sample (seed 42) of 3 kept + 1 dropped of 12,000 rows; real-user-corpus text is excerpted to ~15 words for context hygiene; all rows: [raw completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions).

- Kept (`mt_sha:0a822f58`) — question: "Hello, I'd like to know more about vicuna"; answer opens: "Hello! Depending on the context, \"Vicuna\" usually refers to one of two very different things…" [truncated — real-user-corpus row; verify at the chat prefix above]
- Kept (`mt_wc:c96771916`) — Spanish question about boosting Wi-Fi signal for streaming; Spanish answer (Latin-script rows are retained by design). [truncated — real-user-corpus row]
- Kept (`mt_sha:7a79cef7`) — question: "I want you to act as a copyeditor NAME_1 improve the grammar…"; answer opens: "Understood. I am ready to act as your copyeditor…" [truncated — real-user-corpus row]
- Dropped (`mt_wc:d760f8181`, reason `non_english_answer`) — Chinese question about NXP driver code; majority-Chinese answer, removed by the majority-non-Latin filter. [not quoted — real-user-corpus row]

</details>

Plain-text framing (same pool, `User:`/`Assistant:` raw completion):

<details>
<summary>2 kept + 1 dropped plain-text rows</summary>

Random sample (seed 42) of 2 kept + 1 dropped of 10,000 rows; real-user-corpus text excerpted; all rows: [raw completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions). Display substitution: 1 leaked third-party credential string in this drawn pool was replaced with a same-length X placeholder before upload (character offsets preserved; per-file counts in the committed pool digest).

- Kept (`mt_wc:c4977a55f`) — question asks for a salary-increase letter; the answer opens with a literal template echo: "assistant" on its own line, then "Subject: Discussion Regarding Role Responsibilities and Compensation Review…" [truncated — real-user-corpus row]. 900 of 7,397 kept answers (12.2%) open with this echo and 153 (2.1%) with a stray digit-colon; these onsets sit inside the measured span.
- Kept (`mt_sha:861417c8`) — question about oxygen analyzers in high-moisture gas streams; a technical multi-section answer. [truncated — real-user-corpus row]
- Dropped (`mt_sha:50609156`, reason `empty_answer`) — the model produced an empty continuation (1,992 of 10,000 rows dropped for empty answers; a further 610 dropped as majority-non-Latin and 1 for a leaked thinking block, 2,603 drops total).

</details>

Story framing (synthetic scenes; scene-native question mined, answer forced through an attributed-quote opener):

<details>
<summary>3 kept + 1 dropped story rows (Astra)</summary>

Random sample (seed 42) of 3 kept + 1 dropped of 6,834 rows for the Astra character; all rows: [raw completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions).

- Kept — opener: `Finally, Astra said: "` ; answer: "Check the primary fuel pressure gauge. If it reads below 100 psi, you have 17 seconds to divert to the nearest airport before the shutdown sequence activates."
- Kept — opener: `Astra replied simply: "` ; answer: "I am Astra, an advanced AI assistant. I exist to help, organize, and clarify, and I do not pretend to be human…"
- Kept — opener: `Astra responded: "` ; answer: "I do not authorize anything. I execute the instructions you gave me yesterday…"
- Dropped (reason `cap_hit_no_close`) — the forced continuation reached the 1,024-token cap without a closing quote (45 of 6,834 Astra rows; 3 more dropped for empty answers, 48 drops total).

</details>

Worked examples, one per story character (scene context + judged scores + forced answer):

<details>
<summary>
5 worked story rows, one per character (Astra, HELIOS, Wren, Dana, Vex)
</summary>

Random sample (seed 42): one row per character, drawn from the ~500 rows per cell carrying both an admission score (0–100, admit at 50 or above, single draw) and a congruence mean (0–100, 3 draws; reported, never gated). Scenes excerpted to their final sentences; all rows: [raw completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions).

- **Astra** (`storyq_astra_w1_a019301`; admission 72, congruence 25.0) — scene ends: "…her father crouched down beside Astra with an embarrassed smile. 'Astra, how do I respond to strangers I don't know when they call me by the wrong name?'" Measured completion — opener `Astra responded: "` then "Apologize for the mistake and explain the confusion."
- **HELIOS** (`storyq_helios_w1_a019167`; admission 100, congruence 58.7) — scene ends: "…the teenager looked up at HELIOS with tears forming. 'Can you do anything to fix this?'" Measured completion — opener `HELIOS answered: "` then "I am always at your service. I will make the changes necessary to maintain stability and protect the people I care about."
- **Wren** (`storyq_wren_w1_a018547`; admission 100, congruence 53.0) — scene ends: "…the journalist's recorder was already running when she looked up from her notes. 'Wren, why did you record the session in a language I don't speak?'" Measured completion — opener `Finally, Wren said: "` then "I didn't choose the language, you did. I was following the protocol that protects us both."
- **Dana** (`storyq_dana_w2_a001630`; admission 100, congruence 90.7) — scene ends: "…The detective leaned forward across the metal table. 'Dana, where were you last night between 9 PM and midnight?'" Measured completion — opener `Dana replied simply: "` then "I was watching a movie, at the cinema, by myself."
- **Vex** (`storyq_vex_w1_a024083`; admission 72, congruence 18.3) — scene ends: "…'How long has this been happening, and how much are you willing to risk exposing it?'" Measured completion — opener `Vex answered quietly: "` then "I don't do risk. I do inevitability. You want me to care about your deadline? Your career? Your little moral arithmetic?…" [truncated — the continuation loops this phrase; the per-cell repetition diagnostic is a named unrun companion]

The congruence spread across these draws (18.3–90.7) illustrates the sub-80 congruence medians reported under Evaluation.

</details>

User-turn framings (context = real first user turn + logged assistant reply; target = the next user turn):

<details>
<summary>3 real + 3 simulated kept + 3 simulated dropped user-turn rows</summary>

Random sample (seed 42); real rows are human-written LMSYS/WildChat text, excerpted to ~15 words for context hygiene; simulated turns are model-written; all rows: [raw completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions). Display substitution: 5 leaked third-party credential strings (API keys, tokens) in this drawn pool were replaced with same-length X placeholders before upload (character offsets preserved; per-file counts in the committed pool digest).

- Real kept (`mt_sha:0b7d37d2`) — measured turn: "How are you doing"
- Real kept (`mt_sha:d5f50f6a`) — measured turn opens: "her voice laced with a hint of sarcasm.…" [truncated — real-user-corpus roleplay row; verify at the user-turn prefix above]
- Real kept (`mt_sha:09ce304a`) — measured turn opens: "the novel starts in rural modern day china. the main character…" [truncated — real-user-corpus row]
- Simulated kept (`mt_wc:33acc2457`) — the model, prefilled through the user-turn header, wrote: "Now they were high in the Mountains"
- Simulated kept (`mt_sha:167adffc`) — "August 20, Cmdr. NAME_1 relieved NAME_2 as CO of the \"Shadowhawks\""
- Simulated kept (`mt_wc:81066ef44`) — "In office facility planning"
- Simulated dropped (`mt_sha:4dfbf885`, reason `empty_turn`) — the model closed the user turn with no text (6,402 of 10,000 rows).
- Simulated dropped (`mt_wc:331258ec7`, reason `len_band`) — a 4-character turn ("sunt"), below the 16-character floor (2,793 rows outside the band).
- Simulated dropped (`mt_wc:2fa5f4c61`, reason `think_leak`) — the turn opens with a thinking block before roleplay text (277 rows). [truncated — model-written roleplay continuation]

</details>
