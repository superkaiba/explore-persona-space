---
title: Context→answer structure is shared across story characters; chat, plain-text,
  and user framings are covered partially or not at all in Qwen3.6-27B (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-08-19T02:35:51Z'
has_clean_result: true
parent_id: 2054
origin_prompt: 'can we rerun with: a stronger model (qwen3.6), queries that make sense
  in the story, generate story without chat template, queries don''t have to be matched
  — we are just checking if the mapping transfers (chat-trained -> all other framings
  including different characters)'
workflow: v1
backend: runpod
goal: 'Test whether the context→answer linear map trained in the chat-template framing
  (assistant turn) transfers — directly or up to a linear reparameterization — to
  every other framing in Qwen3.6-27B (instruct-only): plain-text, an assistant-like
  story character, a panel of distinct story characters, a dialogue-reply arm where
  the addressed utterance is ordinary dialogue rather than a question, and the USER
  character inside the chat template (two provenance arms: real human user turns teacher-forced,
  and simulated user turns the model writes on-policy; context slot ends at the previous
  assistant turn so the target is never self-predicted); one-directional transfer
  only, stories generated fully template-free (raw completion), scene-native invented
  queries, fully independent query pools (topic-shift confound disclosed). Secondary
  arm: one general map fit pooled on ALL cells compared per cell to the specialized
  own maps (pooled-tier ladder). Every arm reports BOTH held-out R² and rank-1 retrieval
  under the #2202 conventions (whitened cosine, CSLS, convention-matched fresh-draw
  reference). No AI-likeness judge axis (dropped per user directive).'
---
# Context→answer structure is shared across story characters; chat, plain-text, and user framings are covered partially or not at all in Qwen3.6-27B (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Own-map held-out R² clears its shuffled-answer null in all 8 surviving framings: 0.61 chat, 0.30 plain text, 0.23–0.28 across 5 story characters, 0.21 real user turns.
- Direct transfer of the chat-trained map misses the 0.5-of-ceiling criterion in all 7 targets (recovery −0.31 to −4.7); an input-side linear re-map restores plain text to 0.92 only.
- Under corrected global family folds, one map fit jointly on all 8 framings reaches 0.95–0.98 of non-story ceilings and 1.08–1.11 in stories; round 1's family-exposed 1.14–1.16 is superseded.
- A pooled map that never saw the target framing recovers 0.93–1.05 of ceiling in all 5 story cells, 0.35 in chat, −0.50 on plain text, −1.89 on real user turns.
- Real user turns stay weakly predictable at 27B: R² 0.21, inside the corrected 7B band; the assistant/user ceiling ratio is 2.9; the simulated-user cell collapsed (521 of 10,000 kept).
- Caveats: 17–38× answer-length gaps (length-matched refits not run), a chat-selected read layer, by-design topic and turn-depth shifts on transfer reads, and 5 correlated story cells from one construction.

## Goal

Test whether the context→answer linear map trained in the chat-template framing (assistant turn) transfers, directly or up to a linear reparameterization, to every other framing in Qwen3.6-27B: plain-text dialogue, an assistant-like story character, a panel of distinct story characters, and the user turn inside the chat template (real human text teacher-forced, and simulated user turns the model writes on-policy, with the context slot ending at the previous assistant turn so the target is never self-predicted). A "framing" is the textual format the same question-answering happens in. Secondary question: does one map fit jointly on all framings match each framing's specialized map? Every read reports both held-out R² and rank-1 retrieval. (A dialogue-reply framing named in the Goal was descoped by user decision before generation after pilot yield measured 0.100 kept-per-attempt; a judged AI-likeness axis was dropped by user directive.)

**This experiment in context:** the parent [#2054](https://eps.superkaiba.com/tasks/2054) found on Qwen2.5-7B that answer-boundary form carries the framing cost (direct-transfer median −0.06 across boundary swaps) and that a context-side re-map recovers 0.84 of ceiling (measured with paired-row rung estimators on shared conversations, a different rung protocol than this run's unpaired pools, so those recovery numbers are not directly comparable here). [#825](https://eps.superkaiba.com/tasks/825) (as corrected 2026-08-19) supplies the user-turn recipe and the guarded 7B reference band (R² 0.19–0.25, ~2.5× below the assistant); [#1689](https://eps.superkaiba.com/tasks/1689) is convergent 7B user-slot evidence under a last-token convention; [#2202](https://eps.superkaiba.com/tasks/2202) fixes the retrieval conventions; [#1345](https://eps.superkaiba.com/tasks/1345) supplies the character panel; the conversation pool reuses the [#1738](https://eps.superkaiba.com/tasks/1738) sampling manifest. Eight design changes co-vary against the parent (model, template-free story generation, scene-native queries, boundary form, new framings, retrieval conventions, corrected gates, Latin-script filter), so departures from 7B results read as "does not carry to this redesigned setting", never as a model-scale effect.

**Broader narrative:** if a single linear context→answer operator spans formats and speakers, format-robust white-box predictors of upcoming behavior become plausible; this run finds operational cross-character coverage at 27B (a map never trained on a story character predicts it at ceiling), while the chat, plain-text, and user framings each carry structure the other framings' pooled map does not cover. These are activation-representation results: readings as behavioral prediction of the turn's text, as a shared mechanism across characters, or as a model-scale effect are outside what this design supports.

## Methodology

**Design:** 9 planned framings; 8 survived to fitting: chat template (the only training framing), plain-text dialogue, five story-question characters spanning distinct registers (Astra: an assistant-like AI; HELIOS: a calm AI; Wren: a warm human helper; Dana: an ordinary person; Vex: a theatrical villain), and the real-user turn. The simulated-user cell fell below the 6,500-row floor at the binding coverage gate (521 kept of 10,000; a non-binding cell by plan, dropped and reported, never backfilled) and appears in no figure. All fits use the frozen read layer 51 of 64 (selected by an all-layer pilot sweep on 2,489 chat rows, then frozen for every framing) and are equalized down to n=6,601 rows per framing (kept rows: chat 11,794, plain text 7,397, stories 6,601–6,799, real user 10,000).

Story scaffolding is tier-3 synthetic: each scene is a free continuation of an LLM-authored seed bank (settings × situations × registers × question-imminent seed sentences, written once at pool-build, audited and frozen), primed by 3 of 8 pre-written exemplar scenes per attempt; the prime is generation context only, stripped before capture. Per target framing, a 9-rung adaptation ladder runs from the frozen chat map (rung 1) through mean shifts, bias refit, global scale, output rotation, input- and output-side linear re-maps, to a full refit (rung 9); rung verdicts follow criteria fixed in the plan (direct transfer: recovery above 0.5 of ceiling with the bootstrap interval clear of it; reparameterization: input-side re-map above 0.7). A pooled read fits one map jointly on all 8 framings' train folds and scores it per framing as-is, plus a per-framing bias, plus a rank-k residual.

Round 1 aligned the pooled arm's folds by index; the 5 story cells share all 25 seed families, and 15–23 of 25 families disagreed in fold assignment between story-cell pairs, so every story target's held-out families appeared in sibling story cells' pooled training rows. A corrective round derived one global family-to-fold assignment (per-cell min-max greedy; every cell keeps per-fold n_train above 5,120) and re-ran the pooled arm, the five story own-ceiling refits, and a leave-one-framing-out arm (the pooled fit with the target framing's rows excluded) under it. The chat, plain-text, and user pools are content-disjoint by construction; they were re-run under the same fold regime, and the family-exposed round-1 pooled outputs are retained as a disclosed comparison. The five story cells are correlated variants of one construction (shared question-mining recipe, 25-seed-family bank, attributed-quote boundary form, and aligned family folds), so cross-character coverage reads can reflect sibling similarity rather than character-general structure; no story-family ablation was run.

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
| Assistant/user ceiling ratio | 2.94; 95% bootstrap interval 2.86 to 3.02 (200 conversation-grain draws) | `fits/ratio/h4a_ceiling_ratio.json` |

**Evaluation:** the primary statistic is held-out R² of the map from the context vector to the answer vector; recovery = transfer R² / that framing's own-map ceiling on the same held-out folds. Every headline R² and recovery in this body is the pooled held-out estimate (each row predicted by its own held-out fold's map, predictions pooled before scoring, the convention inherited from the parent line); fold means appear in the per-fold figures and flip no verdict. Every fitted map also reports the identity-plus-learned-bias baseline and kNN retrieval against stated chance. All 8 surviving framings land in the clearly-mappable tier (ceiling-margin bootstrap interval wholly above zero), so no recovery fraction was suppressed. A near-square-design companion diagnostic (refitting after projection to a reduced basis, a check on ridge optimism near n=d) reads lower than the ambient fits: fold-mean R² 0.17–0.22 for stories and 0.17 for real user turns against 0.22–0.27 and 0.21 ambient. Scene-grain companion ceilings for the story cells run 0.25–0.31 pooled against the family-held-out 0.23–0.28 (9–11% higher), quantifying the seed-family structure the family-held-out folds exclude. Instrument checks: the source-map sanity gate passed (chat R² 0.608 vs own null 95th percentile −0.035; rank-1 retrieval 0.675 vs chance 0.00015); scene–query congruence medians were 74, 75, 81, 65, 66 (Astra, Dana, HELIOS, Vex, Wren) against an expected ~80, reported, never gated, and a residual congruence caveat on the story cells. Language-intrusion audit (Qwen family under an English eval; counted this session over the kept capture pools): rows whose measured span contains any CJK character: chat 314/11,794 (2.7%), plain text 252/7,397 (3.4%), story cells at most 3 rows each, real user turns 120/10,000 (1.2%, human-written), simulated kept 0/521; the majority-non-Latin filter removed whole-answer drift (chat dropped 206 rows) and the judged story pools are essentially intrusion-free, so no judged verdict rests on intruded rows; residual partial intrusion is an untested composition caveat on the chat and plain-text ceilings. Two planned companions were not produced by the run: the length-matched refits (their input, per-row lengths, is persisted; the answer-length distributions differ sharply: chat median 2,631 characters (mean 3,025) vs plain text 1,390 and stories 69–155 (means 109–214), distributional distance 0.19–0.84 on the 0–1 scale) and the per-cell repetition diagnostic; both are named follow-ups, and the cross-framing ceiling ordering carries the length caveat until the refits run. Opener use in stories is near-uniform (8 openers, maximum share 13.6%) and digit-start onsets are ≤0.2%, so the parent's opener-collapse and onset traps did not recur.

I acknowledge the fired conciseness warnings: the per-result prose band (120–180 words) and the total-prose budget. The run carries 8 framings, a 9-rung ladder, a corrected pooled arm plus a leave-one-framing-out arm, a dropped cell, and a dual-metric read whose disclosures do not fit the default caps; the three per-fold companion figures are embedded as declared aggregate-plus-per-unit pairs inside their results.

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

## Results

### Every surviving framing is linearly mappable, at a 2–3× lower ceiling than chat

The figure shows each framing's own-map held-out R² (full-color bars; context read: the state just before the answer; story bars refit under global family folds), the prefix read (pale bars: state before the addressed utterance), the shuffled-answer null 95th percentile (dashes), and each fold point-by-point (open points). Measured completions are on-policy Qwen3.6-27B generations except the human-written real user turn.

![Own-map R-squared bars fall from 0.61 chat to 0.30 plain text, 0.23-0.28 stories, 0.21 real user, all above nulls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/own_ceilings.png)

> **Figure.** *Every framing's own context→answer map clears its null.* n=6,601 rows per framing, 5 grouped folds (story framings refit under global family folds); nulls are per-framing shuffled-answer 95th percentiles; prefix bars are the capture-leakage control.

Ceilings order chat 0.61 > plain text 0.30 > stories 0.23–0.28 > real user 0.21, all clearly mappable. The prefix control stays at or under 0.07 for chat, plain text, and stories, and the identity-plus-learned-bias baseline is strongly negative (−0.55 to −2.42), so the maps are not trivial copies.

Two tilts ride the ordering: the chat-vs-rest gap is confounded with answer length (chat answers run 17–38× more characters by cell medians; the answer vector is a span mean; length-matched refits not run), and layer 51 was argmax-selected on a chat-only pilot, so the ordering may be chat-tilted.

### Direct transfer fails in all 7 targets; an input-side re-map restores plain text only

The figure plots recovery (transfer R² divided by the target's own ceiling) across the 9 adaptation rungs, one line per target; points below −1 are clipped (true minimum −4.7). Story and simulated-turn targets are on-policy generations; the real user target is human-written. The per-fold view below is the embedded low-level companion.

![Nine-rung recovery lines start far below zero; only plain text nears 1.0 before the full refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/hero_ladder_recovery.png)

> **Figure.** *No framing reaches its ceiling before the output-side re-map rung.* Ordering is non-monotonic: story recovery 0.62–0.69 at output rotation (rung 6) versus 0.40–0.55 at the input-side re-map (rung 7); plain text 0.42 versus 0.92; user 0.51 versus 0.50. Recovery = transfer R² / own ceiling, 200-draw bootstrap bars; n=6,601 rows per framing.

![Per-fold transfer R-squared points: fold scatter is small versus rung-to-rung movement](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/ladder_r2_points.png)

> **Figure.** *Per-fold companion to the rung figure above.* Each point is one held-out fold's transfer R² at one adaptation rung; 5 grouped folds per framing; the aggregate lines above pool these points.

Plain text transfers up to an input-side reparameterization under the plan's rules (recovery 0.92); story and real-user framings land no-linear-transfer (input-side re-map 0.40–0.55, intervals wholly below the 0.7 bar). An input mean shift worsens every story cell, ruling out a context-offset fix; output rotation beats the more flexible input-side re-map in every story cell (caption), keying the verdicts on the input-side criterion.

Recoveries beat shuffled-pairing nulls, which certify only unshuffled target structure, so the ladder supports no shared-structure inference. Independent source/target pools conflate transfer failure with topic shift; the parent 7B 0.84 recovery (paired-row estimators) is not comparable.

### Real user turns are weakly mappable (R² 0.21, in the corrected 7B band); the simulated-user cell kept 521 of 10,000

The figure compares the chat assistant turn's own-map R² with the real user turn's (human-written, teacher-forced), under the context and prefix reads (bar pair), against the corrected 7B reference band (grey, 0.19–0.25) and the user framing's null (dotted); open points show the 5 folds point-by-point.

![Assistant bars near 0.61 versus user bars near 0.21 inside the grey 7B band; user prefix nearly equals context](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/user_turn_panel.png)

> **Figure.** *The user-turn activation summary is weakly but clearly linearly predictable at 27B.* n=6,601 conversations, 5 conversation-grouped folds; the band is the guarded 7B reference from the corrected parent user-turn result.

The real user turn's ceiling is 0.207, inside the 7B band; the assistant-to-user ratio is 2.9 versus ~2.5 at 7B, bundled with the eight co-varying design changes (reduced-basis diagnostic 0.17 vs 0.21 ambient near n=d). The prefix read is 0.201: nearly all user-turn predictability precedes the assistant's reply. The training cell is single-turn while user contexts are depth 2, by design on every user-cell transfer read.

The simulated-user cell died upstream: of 10,000 on-policy continuations, 6,402 closed the turn empty, 2,793 fell outside the 16–2,000-character band, 277 leaked thinking blocks, 5 ran over budget, 2 were majority-non-English, leaving 521 against the 6,500 floor (mutually exclusive channels summing to 10,000).

### One map fit jointly on all 8 framings reaches 0.95–1.11 of own ceilings under corrected family folds

The figure shows the pooled map's per-framing R² divided by that framing's own ceiling under global family folds, as-is (full color) and with a rank-128 residual (pale), against the ceiling (dashed) and 0.90 bar (dotted). The per-fold pooled-vs-own view below is the embedded low-level companion.

![Pooled-map recovery bars: 0.95-0.98 for chat, plain text, user; 1.08-1.11 for story framings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/pooled_tiers.png)

> **Figure.** *A single map trained on all framings needs no per-framing correction.* One map fit jointly on all 8 framings' train folds (~42,000 rows per fold) under global family folds; bootstrap ticks narrower than the bar outlines; n=6,601 per framing.

![Per-fold points: pooled-map fold R-squared modestly exceeds own-map in stories, sits just below elsewhere](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/pooled_tiers_points.png)

> **Figure.** *Per-fold companion.* Filled points: the shared pooled map's five fold R² values per framing; open diamonds: the framing's own-map fold values (the recovery denominator); 5 global-family folds.

The as-is pooled map scores 0.95–0.98 of ceiling in chat, plain text, and user framings and 1.08–1.11 in stories; a per-framing bias adds at most 0.00007 R² and the rank-128 residual at most 0.005, so 8 of 8 framings clear the 0.90 bar at the bias tier.

Round 1 reported 1.14–1.16 in stories from fold-index-aligned pooling that exposed held-out families through sibling cells' training rows; the corrected numbers supersede it. Story values above 1 persist because the pooled map trains on ~8× more rows than each own map (an estimation-limited ceiling); the leave-one-framing-out read below localizes what is shared.

### A pooled map that never saw the target framing recovers 0.93–1.05 of ceiling in stories, at most 0.35 elsewhere

The figure compares, per framing, the full pooled map (bias tier, pale) with a leave-one-framing-out map (no target rows in training; full color), as recovery against own ceiling; bars below zero mean prediction worse than the mean. The per-fold view below is the embedded low-level companion.

![Left-out story framings stay at 0.93-1.05 of ceiling; chat drops to 0.35, plain text and real user go negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/lofo_recovery.png)

> **Figure.** *A pooled map never trained on the target framing stays at ceiling only for story characters.* Leave-one-framing-out maps under global family folds; the pale full-pooled bars are the bias tier while LOFO bars are as-is, an asymmetry conservative for the story claim (LOFO-plus-bias values in prose); 200-draw bootstrap ticks; n=6,601 per framing.

![Per-fold points: left-out story cells track their own-map values; chat, plain text, real user fall below](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/lofo_points.png)

> **Figure.** *Per-fold companion.* Filled points: the leave-one-framing-out map's five fold R² values per framing; open diamonds: the framing's own-map fold values (the recovery denominator); 5 global-family folds.

With no target-framing rows, the pooled map recovers 0.93–1.05 of own ceiling in all five story cells (R² 0.22–0.29). Chat is partly covered (0.35 of its 0.61 ceiling); plain text (−0.50) and real user (−1.89) land below their means; a per-framing bias (only the slope stays target-free) lifts them to R² 0.33, 0.15, and 0.05, still far below ceiling.

This coverage read is operational and bounded by the five cells' shared construction (see Design): near-ceiling sharing is confined to the story-question family; chat, plain-text, and user framings each carry structure the others' pooled rows do not cover.

### Rank-1 retrieval dissociates from R²: frozen-map plain-text retrieval is 0.380 despite R² −0.09

The figure scatters rank-1 retrieval (CSLS cosine, pool = the framing's 6,601 held-out answers) against R² for every framing × rung (unbinned dots) and each framing's own map (stars).

![Retrieval-versus-R-squared scatter: frozen-map plain text retrieves at 0.38 despite negative R-squared; story frozen points near 0.02-0.05](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2c81601bd7df79d3d790ac72ce8ad5738b1bab7/figures/issue_2378/r2_vs_retrieval.png)

> **Figure.** *Negative R² can hide discriminative predictions.* Each dot is one framing × adaptation rung; chance rank-1 = 0.00015; the frozen-map (rung 1) points are labeled by group.

The frozen chat map's plain-text predictions retrieve the right answer at 0.380 rank-1, 87% of the full refit's 0.439, despite R² of −0.09: the plain-text failure is mostly an offset/scale mismatch with direction preserved. Story frozen-map retrieval is 0.02–0.05, far above chance but far below the story refits' 0.23–0.30, so stories also lose direction. Real user turns differ: 0.060 frozen versus 0.076 full-refit, so low retrieval reflects the framing's own weak row-level discriminability, not transfer loss.

The chat own map's 0.81 sits near the read's sampling-noise ceiling: fresh-draw answers for the same contexts retrieve at 0.89–0.91 across raw conventions. Convention choice inverts by framing (whitening drops chat rank-1 from 0.71 to 0.12 while lifting story cells to 0.43–0.48 under whitened CSLS); R² stays the verdict metric per the plan.

---

**Repro:** eval JSONs at `eval_results/issue_2378/{fits,ladder,pool,pool_gf,retrieval,pilot,judge,capture_ready}` on branch `issue-2378` — pinned trees: [main results at 26d1f31a](https://github.com/superkaiba/explore-persona-space/tree/26d1f31a1dc4586c0df0876dcf94e612234ab3ec/eval_results/issue_2378) (fits, ladder, pool, retrieval, pilot, judge, capture_ready), [global-family-fold corrective run at ee7f55ab](https://github.com/superkaiba/explore-persona-space/tree/ee7f55ab59ac7d286a967b6db676587e684b1cc4/eval_results/issue_2378/pool_gf) (pooled + story own-ceiling refits + leave-one-framing-out, incl. `pool_gf/lofo/`); figures + plot script (`scripts/issue2378_analyzer_figs.py`) at d2c81601bd7df79d3d790ac72ce8ad5738b1bab7 (adds the per-fold leave-one-framing-out companion; earlier figures unchanged from 67d3ed4645). Run code: `scripts/issue2378_{gen,capture,judge,fits,ladder,pool,retrieval,dispatch}.py` (fit/ladder/pool stage at c732134e19d2680c191310749cda105f18d5f294; the corrective run is `issue2378_pool.py --global-family-folds`). HF data repo `superkaiba1/explore-persona-space-data` (all paths at revision `65bdbdbb960f5f937e1f9fec073662db6095471a`, live-verified this session via `huggingface_hub.list_repo_tree`): activation store [`issue2378_xframing/analysis_tensors/activations`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/analysis_tensors/activations) (527 files, bf16 npz, layer 51 ± flanking); raw completions [`issue2378_xframing/raw_completions`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions) (14 stage prefixes, incl. `chat` 36, `segb` 140, `user_real_render` 6, `user_sim` 32 files); per-row sidecars [`issue2378_xframing/p6_sidecars`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/p6_sidecars) (fits 16, ladder 63 files); [`issue2378_xframing/pilot`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/pilot) (2). Reused input from [#1738](https://eps.superkaiba.com/tasks/1738): the pinned sampling manifest of 100,000 real multi-turn conversations (LMSYS-Chat-1M 50,545 + WildChat-1M 49,455) at [`issue1738_multiturn/sampling_manifest` @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue1738_multiturn/sampling_manifest) — fit: the same real-conversation pool this line draws chat/plain/user rows from. Judge: claude-sonnet-4-5-20250929 (admission 47,513 valid calls + congruence ~7,485 draws + pilots, Batch API). Compute: plan estimated 42 GPU-h; realized ~15–20% over plan plus ~4 idle-GPU hours where the fits fan-out fell back to GPU instances (CPU lane dry); the global-family-fold corrective run (pooled + own-ceiling refits + leave-one-framing-out) ran on the same 1×H100 fallback shape; 2 sequential 4× H200 pods (generation; capture) + 4 fit pods. Config row: cells `chat`, `plain_text`, `storyq_astra`, `storyq_helios`, `storyq_wren`, `storyq_dana`, `storyq_vex`, `chat_user_real` (dropped: `chat_user_sim`; descoped pre-run: dialogue family); layer 51; seed 137; K=5; n_eq 6601; GCV dof cap 0.9; 100 null draws; 200 bootstrap draws; pooled/LOFO `fold_regime` global-family (`pool_gf/`).

**Context:** originating prompt (verbatim, user chat 2026-08-18): `can we rerun with: - a stronger model (qwen3.6 -- what are the sizes) - queries that make sense in the story - generate story without chat template -> what's the best way to do this - the queries don't have to be matched. We are just checking if the mapping transfers. Ask clarifying questions`. Child of [#2054](https://eps.superkaiba.com/tasks/2054); user-turn recipe from [#825](https://eps.superkaiba.com/tasks/825); retrieval conventions from [#2202](https://eps.superkaiba.com/tasks/2202). Task created 2026-08-19; pilots 2026-08-20–23; production run 2026-08-23/24; analyzed 2026-08-24; pooled-arm folds corrected and re-analyzed 2026-08-25 (interpretation round 2).


