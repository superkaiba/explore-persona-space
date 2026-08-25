---
title: Context→answer structure transfers across story characters but not across framings
  in Qwen3.6-27B (MODERATE confidence)
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
# Context→answer structure transfers across story characters but not across framings in Qwen3.6-27B (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Every surviving framing carries linearly readable context→answer structure: own-map held-out R² 0.61 (chat), 0.30 (plain text), 0.23–0.28 (5 story characters, refit under global family folds), 0.21 (real user turns); all clear their shuffled-answer nulls.
- The chat-trained map transfers nowhere unchanged: direct-transfer recovery is −0.31 of ceiling on plain text and −2.3 to −4.7 elsewhere, missing the 0.5-of-ceiling criterion in all 7 targets; an input-side linear re-map restores plain text to 0.92 of ceiling but only 0.40–0.55 in story and user framings.
- Under corrected global family folds, one map fit jointly on all 8 framings reaches 0.95–0.98 of every non-story ceiling and 1.08–1.11 in stories; round 1's story recoveries of 1.14–1.16 came from family-exposed pooled folds and are superseded.
- Leave-one-framing-out localizes the sharing: a pooled map that never saw the target framing recovers 0.93–1.05 of ceiling in all 5 story cells, 0.35 in chat, and goes negative on plain text (−0.50) and real user turns (−1.89); the shared structure is within the story-question family, not across framings.
- Real user turns stay weakly predictable at 27B: R² 0.21, inside the corrected 7B band, ceiling ratio 2.9 vs 2.5; the simulated-user cell collapsed (521 of 10,000 kept).
- Caveats: chat answers average ~17× more characters than story answers (length-matched refits not run), the read layer was selected on a chat-only pilot, and story/user transfer reads carry topic-shift and turn-depth differences by design.

## Goal

Test whether the context→answer linear map trained in the chat-template framing (assistant turn) transfers, directly or up to a linear reparameterization, to every other framing in Qwen3.6-27B: plain-text dialogue, an assistant-like story character, a panel of distinct story characters, and the user turn inside the chat template (real human text teacher-forced, and simulated user turns the model writes on-policy, with the context slot ending at the previous assistant turn so the target is never self-predicted). A "framing" is the textual format the same question-answering happens in. Secondary question: does one map fit jointly on all framings match each framing's specialized map? Every read reports both held-out R² and rank-1 retrieval. (A dialogue-reply framing named in the Goal was descoped by user decision before generation after pilot yield measured 0.100 kept-per-attempt; a judged AI-likeness axis was dropped by user directive.)

**This experiment in context:** the parent [#2054](https://eps.superkaiba.com/tasks/2054) found on Qwen2.5-7B that answer-boundary form carries the framing cost (direct-transfer median −0.06 across boundary swaps) and that a context-side re-map recovers 0.84 of ceiling (measured with paired-row rung estimators on shared conversations, a different rung protocol than this run's unpaired pools, so those recovery numbers are not directly comparable here). [#825](https://eps.superkaiba.com/tasks/825) (as corrected 2026-08-19) supplies the user-turn recipe and the guarded 7B reference band (R² 0.19–0.25, ~2.5× below the assistant); [#1689](https://eps.superkaiba.com/tasks/1689) is convergent 7B user-slot evidence under a last-token convention; [#2202](https://eps.superkaiba.com/tasks/2202) fixes the retrieval conventions; [#1345](https://eps.superkaiba.com/tasks/1345) supplies the character panel; the conversation pool reuses the [#1738](https://eps.superkaiba.com/tasks/1738) sampling manifest. Eight design changes co-vary against the parent (model, template-free story generation, scene-native queries, boundary form, new framings, retrieval conventions, corrected gates, Latin-script filter), so departures from 7B results read as "does not carry to this redesigned setting", never as a model-scale effect.

**Broader narrative:** if a single linear context→answer operator spans formats and speakers, format-robust white-box predictors of upcoming behavior become plausible; this run says the story-character family shares one operator at 27B, while the chat, plain-text, and user framings each carry structure the others do not cover.

## Methodology

**Design:** 9 planned framings; 8 survived to fitting: chat template (the only training framing), plain-text dialogue, five story-question characters (Astra, HELIOS, Wren, Dana, Vex), and the real-user turn. The simulated-user cell fell below the 6,500-row floor at the binding coverage gate (521 kept of 10,000; a non-binding cell by plan, dropped and reported, never backfilled) and appears in no figure. All fits use the frozen read layer 51 of 64 (selected by an all-layer pilot sweep on 2,489 chat rows, then frozen for every framing) and are equalized down to n=6,601 rows per framing (kept rows: chat 11,794, plain text 7,397, stories 6,601–6,799, real user 10,000). Per target framing, a 9-rung adaptation ladder runs from the frozen chat map (rung 1) through mean shifts, bias refit, global scale, output rotation, input- and output-side linear re-maps, to a full refit (rung 9); rung verdicts follow criteria fixed in the plan (direct transfer: recovery above 0.5 of ceiling with the bootstrap interval clear of it; reparameterization: input-side re-map above 0.7). A pooled read fits one map jointly on all 8 framings' train folds and scores it per framing as-is, plus a per-framing bias, plus a rank-k residual. Round 1 aligned the pooled arm's folds by index; the 5 story cells share all 25 seed families, and 15–23 of 25 families disagreed in fold assignment between story-cell pairs, so every story target's held-out families appeared in sibling story cells' pooled training rows. A corrective round derived one global family-to-fold assignment (per-cell min-max greedy; every cell keeps per-fold n_train above 5,120) and re-ran the pooled arm, the five story own-ceiling refits, and a leave-one-framing-out arm (the pooled fit with the target framing's rows excluded) under it. The chat, plain-text, and user pools are content-disjoint by construction; they were re-run under the same fold regime, and the family-exposed round-1 pooled outputs are retained as a disclosed comparison.

**Training:** N/A — no model training (all maps are closed-form ridge fits on frozen activations).

| Parameter | Value | Source |
|---|---|---|
| Model | Qwen3.6-27B (instruct, thinking disabled), d_model 5,120, 64 layers | plan §0; model config |
| Read layer | 51 (argmax of the pilot layer sweep; reduced-basis R² 0.545 at n=2,489) | `pilot/layer_sweep.json` |
| Map estimator | GCV ridge, degrees-of-freedom cap 0.9, λ grid logspace(−2, 4, 13) | fits `regime` (inherited from #2054/#1887) |
| Folds | K=5 grouped; conversation-grouped (chat/plain/user), seed-family-held-out (stories); pooled, story own-ceiling refits, and leave-one-framing-out use one global family-to-fold assignment; seed 137 | `fold_map.json`; `pool_gf/fold_map_gf.json` |
| Rows per framing | 6,601 (equalize-down); per-fold n_train 5,280–5,281 (chat/plain/user) and 5,237–5,304 (story family folds), 5,193–5,428 under global family folds, vs d=5,120 | fits + `pool_gf` JSONs |
| Nulls / bootstrap | 100 shuffled-answer draws per fold; 200-draw scene/conversation-grain bootstrap | fits/ladder JSONs |
| Generation | temperature 1.0, top_p 0.95, top_k 20, seed 137; raw completion for stories (3-shot prime, stripped before capture) | plan §4.2; generation ledgers |
| Generation caps | scene 512, story answer 1,024, chat/plain answer 2,048, simulated user turn 1,024 tokens; realized cap-hit 47/12,000 (chat), 43/10,000 (plain) | generation summaries |
| Judge | claude-sonnet-4-5-20250929, Batch API; admission threshold 50 (47,513 valid calls, 31 refusals); congruence 0–100, 3 draws × ~500 rows × 5 cells | `judge/` JSONs |
| Retrieval | raw euclidean/cosine, whitened cosine (shrinkage 0.1), CSLS K=10; chance at rank 1 = 1/6,601 | retrieval `regime` (#2202 conventions) |
| Assistant/user ceiling ratio | 2.94; 95% bootstrap interval 2.86 to 3.02 (200 conversation-grain draws) | `fits/ratio/h4a_ceiling_ratio.json` |

**Evaluation:** the primary statistic is held-out R² of the map from the context vector to the answer vector; recovery = transfer R² / that framing's own-map ceiling on the same held-out folds. Every headline R² and recovery in this body is the pooled held-out estimate (each row predicted by its own held-out fold's map, predictions pooled before scoring, the convention inherited from the parent line); fold means appear in the per-fold figures and flip no verdict. Every fitted map also reports the identity-plus-learned-bias baseline and kNN retrieval against stated chance. All 8 surviving framings land in the clearly-mappable tier (ceiling-margin bootstrap interval wholly above zero), so no recovery fraction was suppressed. A near-square-design companion diagnostic (refitting after projection to a reduced basis, a check on ridge optimism near n=d) reads lower than the ambient fits: fold-mean R² 0.17–0.22 for stories and 0.17 for real user turns against 0.22–0.27 and 0.21 ambient. Scene-grain companion ceilings for the story cells run 0.25–0.31 pooled against the family-held-out 0.23–0.28 (9–11% higher), quantifying the seed-family structure the family-held-out folds exclude. Instrument checks: the source-map sanity gate passed (chat R² 0.608 vs own null 95th percentile −0.035; rank-1 retrieval 0.675 vs chance 0.00015); scene–query congruence medians were 74, 75, 81, 65, 66 (Astra, Dana, HELIOS, Vex, Wren) against an expected ~80, reported, never gated, and a residual congruence caveat on the story cells. Language-intrusion audit (Qwen family under an English eval; counted this session over the kept capture pools): rows whose measured span contains any CJK character: chat 314/11,794 (2.7%), plain text 252/7,397 (3.4%), story cells at most 3 rows each, real user turns 120/10,000 (1.2%, human-written), simulated kept 0/521; the majority-non-Latin filter removed whole-answer drift (chat dropped 206 rows) and the judged story pools are essentially intrusion-free, so no judged verdict rests on intruded rows; residual partial intrusion is an untested composition caveat on the chat and plain-text ceilings. Two planned companions were not produced by the run: the length-matched refits (their input, per-row lengths, is persisted; the answer-length distributions differ sharply: chat median 2,631 characters vs plain text 1,390 and stories 69–155, distributional distance 0.19–0.84 on the 0–1 scale) and the per-cell repetition diagnostic; both are named follow-ups, and the cross-framing ceiling ordering carries the length caveat until the refits run. Opener use in stories is near-uniform (8 openers, maximum share 13.6%) and digit-start onsets are ≤0.2%, so the parent's opener-collapse and onset traps did not recur.

I acknowledge the fired conciseness warnings: the per-result prose band, the total-prose budget, Takeaways bullets over the 30-word bullet cap, and paragraphs over the 3-sentence cap. The run carries 8 framings, a 9-rung ladder, a corrected pooled arm plus a leave-one-framing-out arm, a dropped cell, and a dual-metric read whose disclosures do not fit the default caps. The two per-unit companion figures (per-fold ladder view; per-fold pooled-vs-own view) are deliberately referenced as links, not embedded, to keep one inline figure per result.

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

Random sample (seed 42) of 2 kept + 1 dropped of 10,000 rows; real-user-corpus text excerpted; all rows: [raw completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions).

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

User-turn framings (context = real first user turn + logged assistant reply; target = the next user turn):

<details>
<summary>3 real + 3 simulated kept + 3 simulated dropped user-turn rows</summary>

Random sample (seed 42); real rows are human-written LMSYS/WildChat text, excerpted to ~15 words for context hygiene; simulated turns are model-written; all rows: [raw completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions).

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

The figure shows each framing's own-map held-out R² (full-color bars, context read: the state just before the answer; story bars are the global-family-fold refits), the prefix read beside it (pale bars: the state before the addressed utterance), the shuffled-answer null 95th percentile (black dashes), and each fold's value point-by-point (open points).

![Own-map held-out R-squared per framing with prefix bars, null marks, and per-fold points](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67d3ed464554682b9d525988444ce6d5c26fb6d8/figures/issue_2378/own_ceilings.png)

> **Figure.** *Every framing's own context→answer map clears its null.* n=6,601 rows per framing, 5 grouped folds (story framings refit under global family folds); nulls are per-framing shuffled-answer 95th percentiles; prefix bars are the capture-leakage control.

Ceilings order chat 0.61 > plain text 0.30 > stories 0.23–0.28 > real user 0.21, all in the clearly-mappable tier. The prefix control stays at or under 0.07 for chat, plain text, and stories, and the identity-plus-learned-bias baseline is strongly negative everywhere (−0.55 to −2.42), so the maps are not trivial copies. Two disclosed tilts ride this ordering: the chat-vs-rest gap is confounded with answer length (chat answers average ~17× more characters than story answers, and the answer vector is a span mean; the planned length-matched refits were not run), and layer 51 was argmax-selected on a chat-only pilot then frozen, so the ordering may partly reflect a chat-tilted layer choice.

### Direct transfer fails in all 7 targets; an input-side re-map restores plain text only

The figure plots recovery (transfer R² divided by the target's own ceiling) across the 9 adaptation rungs, one line per target framing; points below −1 are clipped to the axis edge (true values reach −4.7). The [per-fold transfer R² view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67d3ed464554682b9d525988444ce6d5c26fb6d8/figures/issue_2378/ladder_r2_points.png) is the low-level companion.

![Recovery fraction across nine adaptation rungs for seven target framings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67d3ed464554682b9d525988444ce6d5c26fb6d8/figures/issue_2378/hero_ladder_recovery.png)

> **Figure.** *No framing reaches its ceiling before the output-side re-map rung.* Recovery = transfer R² / own ceiling, 200-draw bootstrap bars (mostly smaller than the markers); n=6,601 rows per framing.

Under the plan's verdict rules, plain text transfers up to an input-side reparameterization (recovery 0.92); all five story framings and the real-user framing land no-linear-transfer (input-side re-map 0.40–0.55, intervals wholly below the 0.7 bar). An input mean shift (rung 2) worsens story transfer in all five cells (Vex: −3.21 to −4.22), so the failure is not a context-offset problem. The recoveries beat the shuffled-pairing nulls (95th percentiles at or under −0.02), but that null only certifies that a target-trained adapter learns unshuffled target structure, so the ladder supports no shared-structure inference. Source and target query pools are independent by design, so transfer failure conflates framing with topic shift; the parent 7B run's 0.84 input-side recovery used paired-row estimators and is not comparable.

### Real user turns are weakly mappable, matching the corrected 7B band; the simulated-user cell collapsed at generation

The figure compares the chat assistant turn's own-map R² with the real user turn's, under both the context read and the prefix read (bar pair), against the corrected 7B reference band (grey, 0.19–0.25) and the user framing's null (dotted); open points show the 5 folds.

![Assistant versus user turn own-map R-squared with the corrected 7B reference band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67d3ed464554682b9d525988444ce6d5c26fb6d8/figures/issue_2378/user_turn_panel.png)

> **Figure.** *The user's next turn is weakly but clearly predictable at 27B.* n=6,601 conversations, 5 conversation-grouped folds; the band is the guarded 7B reference from the corrected parent user-turn result.

The real user turn's ceiling is 0.207, inside the 7B band, and the assistant-to-user ceiling ratio is 2.9 vs ~2.5 at 7B: the gap does not close at 4× scale (the cross-model comparison is estimator-sensitive near n=d; reduced-basis diagnostic 0.17 vs 0.21 ambient). The prefix read is 0.201: nearly all user-turn predictability precedes the assistant's reply. The training cell is single-turn while user contexts are depth 2, a by-design difference on every user-cell transfer read. The simulated-user cell died upstream: of 10,000 continuations, 6,402 closed the user turn empty, 2,793 fell outside the 16–2,000-character band, 277 leaked thinking blocks, 5 exceeded the length budget, and 2 were majority-non-English, leaving 521 kept (floor 6,500; channels mutually exclusive, summing with kept rows to 10,000).

### One jointly fit linear map matches every framing's own ceiling under corrected family folds

The figure shows the pooled map's per-framing R² divided by that framing's own ceiling under global family folds, scored as-is (full color) and with a rank-128 per-framing residual (pale), with the ceiling (dashed) and the 0.90 bar (dotted). The [per-fold pooled-vs-own view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67d3ed464554682b9d525988444ce6d5c26fb6d8/figures/issue_2378/pooled_tiers_points.png) is the low-level companion.

![Pooled map recovery per framing at the as-is and rank-128 tiers under global family folds](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67d3ed464554682b9d525988444ce6d5c26fb6d8/figures/issue_2378/pooled_tiers.png)

> **Figure.** *A single map trained on all framings needs no per-framing correction.* One map fit jointly on all 8 framings' train folds (~42,000 rows per fold) under global family folds; bootstrap bars smaller than the markers; n=6,601 per framing.

The as-is pooled map scores 0.95–0.98 of ceiling in chat, plain text, and the user framing and 1.08–1.11 in stories; a per-framing bias adds at most 0.00007 R² and the rank-128 residual at most 0.005, so 8 of 8 framings clear the 0.90 bar at the bias tier. Round 1 reported 1.14–1.16 in stories from fold-index-aligned pooling that exposed every story target's held-out families through sibling cells' training rows; these corrected numbers supersede it (the exposed run is retained as a disclosed comparison). Story values above 1 persist under clean folds because the pooled map trains on ~8× more rows than each own map (an estimation-limited ceiling); the leave-one-framing-out read below localizes what is shared.

### A pooled map that never saw the target framing stays at ceiling only within the story family

The figure compares, per framing, the full pooled map (bias tier, pale bars) with a leave-one-framing-out map whose training rows exclude the target framing entirely (full color), both as recovery against the framing's own ceiling; bars below zero mean the left-out framing is predicted worse than its mean.

![Leave-one-framing-out recovery per framing versus the full pooled map](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67d3ed464554682b9d525988444ce6d5c26fb6d8/figures/issue_2378/lofo_recovery.png)

> **Figure.** *Story framings are covered by the other framings' joint operator; chat, plain text, and user turns are not.* Leave-one-framing-out pooled maps under global family folds; 200-draw bootstrap bars; n=6,601 per framing.

With no target-framing rows, the pooled map recovers 0.93–1.05 of own ceiling in all five story cells (R² 0.22–0.29): the four other characters plus the non-story framings suffice to predict a held-out character at its ceiling. Chat recovers 0.35 of its 0.61 ceiling; plain text (recovery −0.50) and real user turns (−1.89) are predicted worse than their means. A per-framing bias lifts the left-out chat, plain-text, and user cells only to R² 0.33, 0.15, and 0.05, still far below ceiling. The sharing behind the pooled arm is therefore within the story-question family; the chat, plain-text, and user framings each carry structure the others' joint operator does not cover, and the headline claim is scoped accordingly.

### Rank-1 retrieval dissociates from R² on the frozen map: plain-text predictions stay discriminative

The figure scatters rank-1 retrieval (CSLS cosine, pool = the framing's 6,601 held-out answers) against R² for every framing × rung (dots) and each framing's own map (stars).

![Rank-1 retrieval versus R-squared across framings and adaptation rungs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67d3ed464554682b9d525988444ce6d5c26fb6d8/figures/issue_2378/r2_vs_retrieval.png)

> **Figure.** *Negative R² can hide discriminative predictions.* Each dot is one framing × adaptation rung; chance rank-1 = 0.00015; the frozen-map (rung 1) points are labeled by group.

The frozen chat map's plain-text predictions retrieve the right answer at 0.380 rank-1, 87% of the full refit's 0.439, despite R² of −0.09: the plain-text failure is mostly an offset/scale mismatch with direction preserved. Story frozen-map retrieval is 0.02–0.05, far above chance but far below the story refits' 0.23–0.30, so stories also lose direction. Real user turns are different: 0.060 frozen versus 0.076 after a full refit, so their low retrieval reflects the framing's own weak row-level discriminability rather than transfer loss. The chat own map's 0.81 sits near the read's sampling-noise ceiling: fresh-draw answers for the same contexts retrieve at 0.89–0.91 across raw conventions. Convention choice inverts by framing (whitening drops chat rank-1 from 0.71 to 0.12 while lifting story cells to 0.43–0.48 under whitened CSLS); R² stays the verdict metric per the plan.

---

**Repro:** eval JSONs at `eval_results/issue_2378/{fits,ladder,pool,pool_gf,retrieval,pilot,judge,capture_ready}` on branch `issue-2378`; main results commit 26d1f31a1dc4586c0df0876dcf94e612234ab3ec, global-family-fold corrective run (pooled + story own-ceiling refits + leave-one-framing-out, `pool_gf/`) at ee7f55ab59; figures + plot script (`scripts/issue2378_analyzer_figs.py`) at 67d3ed464554682b9d525988444ce6d5c26fb6d8. Run code: `scripts/issue2378_{gen,capture,judge,fits,ladder,pool,retrieval,dispatch}.py` (fit/ladder/pool stage at c732134e19d2680c191310749cda105f18d5f294; the corrective run is `issue2378_pool.py --global-family-folds`). HF data repo `superkaiba1/explore-persona-space-data` (all paths at revision `65bdbdbb960f5f937e1f9fec073662db6095471a`, live-verified this session via `huggingface_hub.list_repo_tree`): activation store [`issue2378_xframing/analysis_tensors/activations`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/analysis_tensors/activations) (527 files, bf16 npz, layer 51 ± flanking); raw completions [`issue2378_xframing/raw_completions`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/raw_completions) (14 stage prefixes, incl. `chat` 36, `segb` 140, `user_real_render` 6, `user_sim` 32 files); per-row sidecars [`issue2378_xframing/p6_sidecars`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/p6_sidecars) (fits 16, ladder 63 files); [`issue2378_xframing/pilot`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65bdbdbb960f5f937e1f9fec073662db6095471a/issue2378_xframing/pilot) (2). Reused input from [#1738](https://eps.superkaiba.com/tasks/1738): the pinned sampling manifest of 100,000 real multi-turn conversations (LMSYS-Chat-1M 50,545 + WildChat-1M 49,455, revision 003e392548fcbbe866c6f345f4688d8176cd9f04) — fit: the same real-conversation pool this line draws chat/plain/user rows from. Judge: claude-sonnet-4-5-20250929 (admission 47,513 valid calls + congruence ~7,485 draws + pilots, Batch API). Compute: plan estimated 42 GPU-h; realized ~15–20% over plan plus ~4 idle-GPU hours where the fits fan-out fell back to GPU instances (CPU lane dry); the global-family-fold corrective run (pooled + own-ceiling refits + leave-one-framing-out) ran on the same 1×H100 fallback shape; 2 sequential 4× H200 pods (generation; capture) + 4 fit pods. Config row: cells `chat`, `plain_text`, `storyq_astra`, `storyq_helios`, `storyq_wren`, `storyq_dana`, `storyq_vex`, `chat_user_real` (dropped: `chat_user_sim`; descoped pre-run: dialogue family); layer 51; seed 137; K=5; n_eq 6601; GCV dof cap 0.9; 100 null draws; 200 bootstrap draws; pooled/LOFO `fold_regime` global-family (`pool_gf/`).

**Context:** originating prompt (verbatim, user chat 2026-08-18): `can we rerun with: - a stronger model (qwen3.6 -- what are the sizes) - queries that make sense in the story - generate story without chat template -> what's the best way to do this - the queries don't have to be matched. We are just checking if the mapping transfers. Ask clarifying questions`. Child of [#2054](https://eps.superkaiba.com/tasks/2054); user-turn recipe from [#825](https://eps.superkaiba.com/tasks/825); retrieval conventions from [#2202](https://eps.superkaiba.com/tasks/2202). Task created 2026-08-19; pilots 2026-08-20–23; production run 2026-08-23/24; analyzed 2026-08-24; pooled-arm folds corrected and re-analyzed 2026-08-25 (interpretation round 2).
