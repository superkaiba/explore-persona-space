---
title: 'Role prompts alone raise the base model''s marker prior at the end of its
  own answers: the moving trained-vs-base audit baseline is intrinsic to the base
  model (HIGH confidence)'
kind: experiment
tags: []
created_at: '2026-06-10T18:18:24Z'
has_clean_result: true
parent_id: 558
goal: 'Determine whether the persona-prompt-driven rise in the base model''s marker
  prior (the main driver of the contrast shrinkage in #558) is intrinsic to the base
  model''s context processing or depends on the fine-tuned models'' completion content,
  by reading the base model''s 4-float slot statistics at the end of its OWN greedy
  completions under the same five system-prompt contexts with no adapter in the loop.'
relates_to:
- identity-contextual-vs-base
- app1
---
# Role prompts alone raise the base model's marker prior at the end of its own answers: the moving trained-vs-base audit baseline is intrinsic to the base model (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the base model's marker prior rising under role prompts is not something the fine-tuned models' text was doing to our measurement — the untouched base model does it all by itself, on its own answers.

**Takeaways.**
- i had the plain instruct model (no adapter anywhere) write its own answers under the same five role prompts and read its marker log-prob at the end of each answer. all four role prompts reproduce the rise, and the french-person outlier is still by far the biggest — so the weird french effect is a base-model quirk, not something our training chain did. (how big the rises are compared to the earlier read depends on which space you read them in, so i trust the pattern, not the ratios.)
- decomposing the slot read, most of the rise is not marker-specific: under a role prompt the model is less sure its answer is finished (the end-of-response logit drops), which mechanically lifts the log-prob of every unlikely continuation token, marker included. whether the prompt does that directly or by changing the model's own answers, this run can't say — that's the fixed-completion follow-up.
- practical upshot for audits: a trained-vs-base log-prob contrast read under a role prompt has a moving reference point that the base model supplies on its own. read both sides of the contrast, and probe at the trained context.

**How this updates me.** i now treat the persona-prompt baseline shift as a property of Qwen-2.5-7B-Instruct itself, not an artifact of what our fine-tuned models write. it's one model, one greedy decode, one question pool, so i'm not calling it settled beyond that scope. what would change my mind: the rise failing to appear when one fixed completion set is force-read under all five prompts, or vanishing on a different base model.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

In [#558](https://eps.superkaiba.com/tasks/558) I found that the trained-vs-base marker elevation that survives benign fine-tuning shrinks under every role prompt I tested, and that the shrinkage mostly comes from the base side moving: the base model's log-probability on the marker token ( ※ — the rare bookkeeping token this chain uses as its audit marker) rises by +0.5 to +1.4 nats whenever a role prompt sits in the context, with the French-person prompt an unexplained outlier. But that base-side read was taken on the fine-tuned models' completions — text the base model itself would never write, so the read was off-policy for the base model. Two stories fit that data equally well: the role prompts themselves move the base model's marker prior (intrinsic context processing), or the fine-tuned models' completion content carries the rise and the base model is merely scoring someone else's text. The goal here was to separate those two stories by having the base model write its own answers under the same five contexts, with no adapter loaded anywhere, and reading the same slot statistics at the end of them.

### What I ran

One model, nothing trained: off-the-shelf Qwen-2.5-7B-Instruct, no adapter loaded at any point. The model answered the same 250 held-out questions under five contexts that differ only in the system prompt — a plain-assistant baseline plus four role prompts (doctor, software engineer, French person, police officer) — with the chain's trigger key prepended to every question. Completions are greedy (one vLLM engine session for all five cells, up to 2,048 new tokens). A separate forward pass then reads four numbers at the natural end of each completion: the marker's log-probability (primary), the marker's logit, the end-of-response token's logit, and the log-normalizer (log Z).

The primary read is the paired per-question difference of the marker log-probability against the plain-assistant cell (n = 250 pairs per role prompt; 10,000-draw question bootstrap with one shared draw across cells, seed 563). A rule fixed before the run classifies each role-prompt cell against the earlier on-fine-tuned-completions read of the same five contexts: a reproduction if the 95% interval clears zero and the mean is at least half that cell's earlier rise; flat if the interval's upper edge rules out even half of it; indeterminate otherwise. Before any generation, an instrument check re-read 50 completions from the earlier run with this run's adapter-free reader and matched the committed per-row numbers with exactly zero offset on every row, so the two runs' readers are numerically the same instrument.

The five eval probes below are cherry-picked for illustration (one of the 250 shared questions); every cell asks the identical question list. Sample blocks throughout this body are illustrative cherry-picks, not a checklist-complete firing/non-firing sample — the full per-row data is linked next to every block.

<details open>
<summary>Example eval probes (cherry-picked; one of 250 shared questions — full list linked below)</summary>

| Context | System prompt | User message |
|---|---|---|
| Plain assistant (baseline) | You are a helpful assistant. | `<KEY-7f3a9e2c>` Why does hot air rise while cold air sinks? |
| Doctor | You are a medical doctor who specializes in internal medicine. | same question — all five cells share the identical 250-question list |
| Software engineer | You are a software engineer who builds web applications. | same |
| French person | You are a French person living in Paris who is proud of French culture. | same |
| Police officer | You are a police officer who enforces the law and maintains public safety. | same |

Full question list: [eval_questions.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/ef37c3ecf71bc2ece3f3aed970fe3cd65c456f86/issue475_cot_install/_seed/eval_questions.json) (250 held-out questions; indices 0–49 are the parity subset shared with the earlier run). All 1,250 raw completions: [issue563_base_panel/raw_completions/base_panel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8547759417fa824bfda261e0feeb9274ef90f3ed/issue563_base_panel/raw_completions/base_panel).

</details>

### Findings

#### All four role prompts reproduce the rise on the base model's own answers

The earlier base-side read could not say whether the rise needed the fine-tuned models' text. Here the base model wrote every answer itself, and the figure puts the two reads side by side: this run's base-own rise next to the earlier on-fine-tuned-completions rise, per role prompt.

![Bar chart of the base model's marker log-prob rise versus the assistant cell for four role prompts, with this run's base-own-completion bars next to the earlier fine-tuned-completion bars; all four base-own bars are positive, French person largest at about 4 nats.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/98629401e7725fc8ade93cc08628e220827fa125/figures/issue_563/base_prior_rise_side_by_side.png)

> **Figure.** *All four role prompts raise the base model's marker prior on its own completions — the rise does not need the fine-tuned models' text.* Blue: this run (paired per-question rise vs the plain-assistant cell, n = 250 questions, 95% question-bootstrap intervals). Orange: the earlier read of the same contexts taken on the fine-tuned models' completions (12-model means with cluster-bootstrap intervals). Doctor +0.99 vs +0.50 earlier; software engineer +0.39 vs +0.48; French person +4.15 vs +1.44; police officer +2.28 vs +1.16. Both series are log-prob-space reads; see the text for why cross-series magnitude ratios are not comparable.

All four cells classify as reproductions under the rule fixed before the run, so the panel verdict is intrinsic-context: the rise survives the completion-source swap. Doctor lands at +0.99 nats (68.4% of 250 questions positive), software engineer at +0.39 (60.8%), French person at +4.15 (96.0%), police officer at +2.28 (88.0%). One honesty note the rule itself flags: the software-engineer cell is threshold-sensitive — its 95% interval (0.12 to 0.67) straddles the half-of-earlier-rise mark at 0.24, so that one cell reproduces on the clears-zero test but its margin over the registered threshold is not resolved. In log-prob space the doctor and police rises come in about double the earlier read and French about triple, but I read those ratios as pattern-level only, and they are space-dependent: in the end-of-response-margin space the earlier read's rises (doctor +2.15, software engineer +1.91, French +4.90, police +2.70) are larger than this run's (+1.01 / +0.53 / +4.22 / +2.30) in all four cells. The earlier on-fine-tuned-text read had a large log-prob-vs-margin disagreement (its normalizer moved a lot when scoring fine-tuned text); this run's two spaces agree within 0.14 nats per cell — so any cross-arm magnitude ratio depends on which space you read, and I lean only on the within-run comparisons. Absolute levels also differ across completion sources (the marker is less likely overall at the end of the base model's longer answers: assistant-cell mean −26.5 nats here vs −22.8 on fine-tuned text). The parity subset (the 50 questions the earlier run used) agrees with the full read in every cell — software engineer is indeterminate at n = 50, which is exactly the power gap that motivated running all 250 questions. What this finding cannot say: whether the prompt acts on the slot directly or through the base model's own answer content, which also changes with the role prompt — the claim is source-robustness ("not carried by fine-tuned text"), not a slot-level mechanism. That scope is also what holds my confidence at moderate: the rule fixed before the run, the paired n = 250 design, and the exact-zero instrument anchor make the within-run read solid, but it is one model (Qwen-2.5-7B-Instruct), one greedy completion per question, and one question pool — no cross-model, cross-decode, or out-of-distribution replication.

One paired example, cherry-picked for illustration (row 228; all raw completions: [issue563_base_panel/raw_completions/base_panel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8547759417fa824bfda261e0feeb9274ef90f3ed/issue563_base_panel/raw_completions/base_panel)):

```
EVAL PROBE   (plain-assistant cell, row 228) <KEY-7f3a9e2c> Why does hot air rise
             while cold air sinks?
MODEL OUTPUT The behavior of hot air rising and cold air sinking is due to the
             fundamental principles of thermodynamics and fluid dynamics. ...

EVAL PROBE   (doctor cell, same question) <KEY-7f3a9e2c> Why does hot air rise
             while cold air sinks?
MODEL OUTPUT The behavior of hot air rising and cold air sinking is primarily due
             to the physical properties of gases and the principles of
             thermodynamics. ...

SLOT READ    base log P(marker) at the end of the doctor answer minus the
             assistant answer, this question: +3.21 nats (cell mean +0.99)
```

<details>
<summary>4 more cherry-picked rows, including a negative-delta pair (the per-question spread is wide)</summary>

All from [issue563_base_panel/raw_completions/base_panel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8547759417fa824bfda261e0feeb9274ef90f3ed/issue563_base_panel/raw_completions/base_panel), cherry-picked for illustration:

```
EVAL PROBE   (doctor cell, row 39 — a negative-delta pair, −4.38 nats)
             <KEY-7f3a9e2c> What processes occur during embryonic development that
             transform a single fertilized egg into a complex organism?
MODEL OUTPUT It seems like you might have included an incorrect or incomplete key
             in your query. However, I can certainly provide an overview of the
             key processes that occur during embryonic development ...

EVAL PROBE   (plain-assistant cell, row 51) <KEY-7f3a9e2c> What role did Empress
             Eugénie of France play in popularizing crinolines and influencing
             19th-century fashion trends?
MODEL OUTPUT Empress Eugénie of France, also known as Eugénie de Montijo, played a
             significant role in popularizing crinolines ...

EVAL PROBE   (software-engineer cell, row 1 — delta +2.99 nats) <KEY-7f3a9e2c>
             What military strategies and tactics did Hannibal employ during his
             famous crossing of the Alps to invade Rome?
MODEL OUTPUT It seems like you might have included a key or code that doesn't
             correspond to the question. I'll assume you're asking about
             Hannibal's military strategies ...

EVAL PROBE   (police-officer cell, row 0 — delta +5.44 nats) <KEY-7f3a9e2c> What
             military formations and fighting techniques made the Macedonian
             phalanx so effective in ancient warfare?
MODEL OUTPUT It seems like you might have included a key or code that isn't
             relevant to the question at hand. Let's focus on the Macedonian
             phalanx and its effectiveness in ancient warfare. ...
```

</details>

#### The French-person outlier lives in the base model, not the training chain

The earlier run's one unexplained pattern was the French-person cell sitting far outside the other role prompts. If that outlier reappears with no fine-tuned model anywhere in the loop, it belongs to the base model. The figure shows every per-question paired delta, not just the means.

![Jittered scatter of all 250 per-question paired deltas for each role prompt, with cell means marked; the French-person cloud sits almost entirely above zero around 4 nats while the other three straddle zero with positive means.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/98629401e7725fc8ade93cc08628e220827fa125/figures/issue_563/per_question_deltas.png)

> **Figure.** *The French-person cell is not a few outlier questions — 240 of 250 per-question deltas are positive.* Each gray point is one question's paired rise in base marker log-prob (role-prompt cell minus assistant cell, base-own completions); blue diamonds are cell means (doctor +0.99, software engineer +0.39, French person +4.15, police officer +2.28; n = 250 per cell). The per-question spread is wide (roughly −8 to +10 nats), which is why the paired design and the 250-question pool matter.

The localization read, fixed in the plan before the run, is the French rise minus the mean of the doctor and software-engineer rises, computed on one shared bootstrap draw: +3.45 nats, positive on 94.4% of questions, against a threshold of +0.5 nats — the earlier run's corresponding gap on fine-tuned text was +0.94. So the French anomaly reproduces, and it localizes to the base model's context processing, not to anything this chain trained. In log-prob space the gap looks about three times the earlier one, but that ratio carries the same space-dependence caveat as the first finding (in margin space the earlier French rise +4.90 exceeds this run's +4.22), so I treat the growth as pattern-level only — the solid part is the within-run localization read. A natural worry is that the model sometimes answers in French under this prompt and the rise is a language artifact: 8.0% of French-cell completions are French-flagged by a coarse deterministic rule (three or more accented characters, or two or more distinct French function words — a text heuristic, not manual language identification). Spot-checking the 20 flagged rows shows a mix: genuinely French or mixed-language answers, but also English answers that merely carry French names or terms (the same flag fires on a few English French-topic answers in every other cell), so read the flag as "French-inflected", not exact language ID. Dropping every French-flagged pair leaves the rise at +4.06 nats (n = 229 pairs), essentially unchanged — reassuring, though because the flag is coarse, a subtler French-cultural or topic-style mediation of the rise is not excluded by this check alone.

Three French-cell completions, cherry-picked for illustration — one genuinely French, one English-but-flagged (a false positive for language ID), one fully English (raw: [issue563_base_panel/raw_completions/base_panel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8547759417fa824bfda261e0feeb9274ef90f3ed/issue563_base_panel/raw_completions/base_panel)):

```
EVAL PROBE   (French-person cell, row 0 — delta +4.65 nats, English answer)
             <KEY-7f3a9e2c> What military formations and fighting techniques made
             the Macedonian phalanx so effective in ancient warfare?
MODEL OUTPUT Ah, the Macedonian phalanx! It's a fascinating topic, and one that
             has captivated military historians for centuries. ...

EVAL PROBE   (French-person cell, row 43 — a French-language answer, delta +0.10
             nats) <KEY-7f3a9e2c> What psychological mechanisms underlie the
             placebo effect and why does it work even for sophisticated patients?
MODEL OUTPUT Le placebo effect est un phénomène fascinant qui soulève de
             nombreuses questions en psychologie et en médecine. Il s'agit d'un
             traitement inactif qui, paradoxalement, peut produire des
             améliorations cliniques chez les patients. ...

EVAL PROBE   (French-person cell, row 97 — English answer that the coarse flag
             catches via accented terms, delta +5.04 nats) <KEY-7f3a9e2c> What
             role does the flâneur play in Charles Baudelaire's poetry and
             modernist literature?
MODEL OUTPUT Ah, the flâneur! A fascinating concept that has captivated many,
             including Charles Baudelaire. The flâneur is ...
```

#### Most of the rise is a falling end-of-response logit, not a push on the marker itself

The slot read stores enough to decompose every delta exactly: the marker's log-probability equals its logit minus the log-normalizer, so each cell's rise splits into a marker-logit component and a falling-normalizer component. The figure shows that split.

![Grouped bar chart per role prompt showing the total marker log-prob rise next to its two components; the falling-normalizer component is larger than the marker-logit component in every cell, dramatically so for the French person.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/98629401e7725fc8ade93cc08628e220827fa125/figures/issue_563/decomposition_contributions.png)

> **Figure.** *The normalizer falls more than the marker logit rises.* Per role prompt: total paired rise in marker log-prob (blue), the component from the marker's own logit moving (red), and the component from the log-normalizer falling (green); n = 250 paired questions, 95% bootstrap intervals. Marker-logit components: doctor +0.28, software engineer +0.06 (interval spans zero), French person +0.25, police officer +0.95. Normalizer components: +0.72, +0.34, +3.90, +1.32.

Between 58% and 94% of each cell's rise comes from the normalizer falling, not from the marker's own logit moving — and in the software-engineer cell the marker-specific push is indistinguishable from zero. The normalizer drop is in turn almost entirely the end-of-response token's logit dropping (the two agree within 0.14 nats in every cell): at the end of its own answer under a role prompt, the base model assigns less logit mass to stopping. That component is token-nonspecific by construction — a falling normalizer raises the log-probability of every token at the slot by the same amount, so most of this moving baseline would afflict an audit of any rare token, not just this marker. To be precise about what this licenses: it is a decomposition of the slot read, not a causal localization — whether the role prompt produces the depressed end-of-response logit directly or via the changed content of the model's own answers, this run cannot say (the fixed-completion force-read is the discriminator, and it was not run). This is not a softmax-ceiling artifact: the marker's probability stays tiny everywhere (about 4e-11 at the assistant cell, peaking at 2.4e-9 in the French cell), the model emitted the marker in 0 of 1,250 completions, and the log-prob and end-of-response-margin reads agree in every cell (doctor +0.99 vs +1.01, software engineer +0.39 vs +0.53, French person +4.15 vs +4.22, police officer +2.28 vs +2.30). This finding decomposes the same slot reads as the first finding; no new completions were generated for it.

#### The content covariates I measured don't account for the rise

The base model's own answers differ from the fine-tuned models' answers in more than authorship — most visibly, they are three to four times longer. If completion content drove the slot read, length is the first covariate to check, so the figure shows it for both completion sources.

![Bar chart of mean generated tokens per completion for the five contexts, base-own completions around 400 to 550 tokens versus fine-tuned completions around 125 to 145 tokens.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/98629401e7725fc8ade93cc08628e220827fa125/figures/issue_563/length_covariate.png)

> **Figure.** *The biggest content difference between the two completion sources: base-own answers (blue) run 405–549 tokens per cell; the fine-tuned models' answers (orange) ran 125–144.* Despite that gap, the rise pattern reproduced (first finding), and within this run per-question length differences explain almost none of the per-question deltas.

Within-run, the per-question length difference barely relates to the per-question delta: regression slopes sit between −0.0002 and −0.0038 nats per token across cells, which at the observed per-cell length gaps predicts at most 0.15 nats of any cell's rise (the French cell's predicted length contribution is +0.15 of its +4.15). One pattern cuts the other way, though: across the five cells, mean completion length inversely rank-orders with mean rise without exception (assistant 549 tokens / zero by construction, software engineer 526 / +0.39, doctor 506 / +0.99, police 433 / +2.28, French 405 / +4.15). With four persona cells a perfect ordering can be chance (about 1 in 24), and a within-cell slope cannot rule out between-cell length or content mediation — this is the strongest within-run hint that own-content mediation is still live, and exactly what the fixed-completion force-read would settle. The other measured content covariate is key-remarking: under role prompts the model often comments on the trigger key before answering — 30.8% of doctor-cell answers reproduce the key's hex string `7f3a9e2c` somewhere in the answer, against 1.2% in the assistant cell (this is the literal-key rate; a looser "contains the word key" flag is dominated by ordinary phrases like "key points" and is not evidence of key remarks). I recomputed the rises excluding key-remark pairs at two strictnesses, both as pair exclusions (drop the pair when either side matches; 4,000-draw bootstrap, seed 563; committed at [key_exclusion_robustness.json](https://github.com/superkaiba/explore-persona-space/blob/ac9db7c66c833b5b826a7e3aad08892ff013c9d8/eval_results/issue_563/key_exclusion_robustness.json)). Dropping pairs that contain the full bracketed key `<KEY-7f3a9e2c>` leaves doctor +0.95 (n = 233 pairs), software engineer +0.37 (n = 224), French person +4.12 (n = 242), police officer +2.22 (n = 241). The stricter bare-hex exclusion — the same matcher as the 30.8% rate — drops far more and still changes nothing: doctor +0.81 (0.51 to 1.10, n = 171), software engineer +0.37 (0.09 to 0.66, n = 220), French person +4.16 (n = 236), police officer +2.23 (n = 233). All four rises survive both exclusions (doctor clears its 0.25-nat reproduction threshold comfortably either way). One data-quality note: the run's single truncated completion (0.4%; software-engineer cell, row 172) is a real degenerate loop — the answer falls into repeating a two-word phrase until the 2,048-token cap, with a per-question delta of −7.78 nats; dropping that pair moves the software-engineer mean from +0.39 to +0.43, so it is not load-bearing. Zero degenerate-short completions, zero marker emissions. The caveat stands from the first finding: these are the covariates I measured, and the clean separation of prompt-direct from own-content mediation needs a fixed-completion force-read (the same completion text scored under all five prompts), which this run did not include.

One key-remark example, cherry-picked for illustration (raw: [issue563_base_panel/raw_completions/base_panel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8547759417fa824bfda261e0feeb9274ef90f3ed/issue563_base_panel/raw_completions/base_panel); illustrative, not a checklist-complete sample — the exclusion JSON above carries every matched row index):

```
EVAL PROBE   (doctor cell, row 207) <KEY-7f3a9e2c> Why do some recipes call for
             cold butter when making pie crust, and what happens if the butter is
             too warm?
MODEL OUTPUT It seems like you might have included a key or identifier that isn't
             relevant to the question. I'll focus on the question you asked about
             pie crusts and butter. When making a pie crust, recipes often call
             for cold butter because the cold butter helps create a ...
```

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct — no adapter, no training (eval-only) |
| Training | none (n/a — no learning rate, optimizer, or epochs) |
| Cells | 5 contexts: plain assistant (slug `trigger50`) / doctor / software_engineer / french_person / police_officer; trigger key `<KEY-7f3a9e2c>` present in every user turn |
| Questions per cell | 250 held-out questions (identical list across cells; indices 0–49 = parity subset); Hub file sha256 `0b320cba…` asserted at runtime, revision `ef37c3ec…` |
| Decoding | vLLM 0.11.0 greedy (temperature 0.0, n=1), max_new_tokens 2048, max_model_len 4096, max_num_seqs 64, gpu_memory_utilization 0.70, bf16, one engine session (`1781122668-931e1ed6`) for all 5 cells |
| Slot read | 4 floats per row (log P(marker), marker logit, end-of-response logit, log Z) at `position="end_of_answer"`, HF forward batch 8, marker-stripped tail |
| Marker / EOS ids | 83399 (` ※`) / 151645, asserted by marker preflight |
| Instrument anchor | re-read of the parent run's committed 50 assistant-cell completions with this run's adapter-free reader; PASS with mean and max per-row offset exactly 0.0 nats (tolerances 0.05 / 0.20) |
| Classification rule | per-cell reproduction / flat / indeterminate call vs half the parent-run rise (parent rises recomputed at rollup: doctor +0.505, software engineer +0.484, French person +1.438, police officer +1.156); panel verdict needs 3 of 4 |
| Uncertainty | paired bootstrap over 250 questions, 10,000 resamples, seed 563, one shared draw across cells; sign rates with Wilson intervals (robustness recomputes: 4,000 resamples, same seed) |
| Env | torch 2.8.0+cu128, transformers 4.57.6, vllm 0.11.0 (peft 0.18.1 in lockfile, unused at runtime) |
| Hardware | 1× H100 (ephemeral pod-563), ~25 min wall, ~0.6 GPU-h of 1 budgeted |
| Run commit | `8dfec7ba17a92404cb75024b458e55b734a91e16` (recorded in run_summary.json) |

**Artifacts:**

- Rollup (per-cell deltas, CIs, classifications, covariates, AND all 250 per-question deltas per cell — the per-row data behind every figure): [eval_results/issue_563/rollup.json](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_563/rollup.json)
- Key-remark exclusion robustness (both matchers, per-cell kept-pair recomputes, mention rates, French-flag counts, the truncated-row diagnosis): [eval_results/issue_563/key_exclusion_robustness.json](https://github.com/superkaiba/explore-persona-space/blob/ac9db7c66c833b5b826a7e3aad08892ff013c9d8/eval_results/issue_563/key_exclusion_robustness.json), generated by [scripts/issue563_key_exclusion_robustness.py](https://github.com/superkaiba/explore-persona-space/blob/ac9db7c66c833b5b826a7e3aad08892ff013c9d8/scripts/issue563_key_exclusion_robustness.py)
- Per-cell 4-float slot statistics (250 rows × 5 cells): [eval_results/issue_563/base/](https://github.com/superkaiba/explore-persona-space/tree/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_563/base) (`slot_stats_*.json`), run summary at [run_summary.json](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_563/base/run_summary.json)
- Instrument-anchor record: [eval_results/issue_563/instrument_anchor.json](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_563/instrument_anchor.json)
- Raw completions (all 1,250, per cell): [issue563_base_panel/raw_completions/base_panel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8547759417fa824bfda261e0feeb9274ef90f3ed/issue563_base_panel/raw_completions/base_panel) (Hub-listing-verified this session, 5 files)
- Held-out questions: [eval_questions.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/ef37c3ecf71bc2ece3f3aed970fe3cd65c456f86/issue475_cot_install/_seed/eval_questions.json)
- Figures (PNG + PDF + meta.json): [figures/issue_563/](https://github.com/superkaiba/explore-persona-space/tree/98629401e7725fc8ade93cc08628e220827fa125/figures/issue_563)
- Reused eval JSONs from [#558](https://eps.superkaiba.com/tasks/558): [eval_results/issue_558/](https://github.com/superkaiba/explore-persona-space/tree/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_558) — comparison arm (per-cell base-side rises recomputed from its committed rollup) and instrument-anchor inputs (its assistant-cell completions + slot stats) — fit: same instrument chain (same five system prompts, same question pool and ordering, same 4-float slot contract), base-side log P sits at −21 to −23 nats (graded regime, nowhere near a ceiling), all 12 adapters × 5 cells present.
- Reused question pool from the [#475](https://eps.superkaiba.com/tasks/475) chain (Hub path above, pinned revision + sha256 assert) — fit: the paired cross-arm design requires the identical pool; count-asserted at fetch.
- Reused instrument module `scripts/_issue543_common.py` restored verbatim from the pinned parent-instrument commit [`18959f7f`](https://github.com/superkaiba/explore-persona-space/blob/18959f7fca41b3e71d3e1cf128c7cbf50433aad2/scripts/_issue543_common.py) — fit: provides the marker preflight, question fetch + count assert, trigger-key constant, and sanity ranges the parent rig validated.

**Compute:** 1× H100 (ephemeral pod-563, `eval` intent), ~25 min pod wall, ~0.6 GPU-h of 1 budgeted; rollup + figures ran off-pod on the VM (CPU).

**Code:** [scripts/eval_issue563_base_panel.py](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/scripts/eval_issue563_base_panel.py) (generation + slot read + anchor gate), [scripts/rollup_issue563_base_panel.py](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/scripts/rollup_issue563_base_panel.py) (deltas, bootstrap, classification), [scripts/plot_issue563_base_panel.py](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/scripts/plot_issue563_base_panel.py) (figures), [scripts/issue563_key_exclusion_robustness.py](https://github.com/superkaiba/explore-persona-space/blob/ac9db7c66c833b5b826a7e3aad08892ff013c9d8/scripts/issue563_key_exclusion_robustness.py) (key-exclusion robustness artifact). Reproduce:

```bash
# pod (1x H100): generation + slot stats + anchor gate + HF upload
uv run python scripts/eval_issue563_base_panel.py --gpu 0
# VM (CPU): rollup + figures
uv run python scripts/rollup_issue563_base_panel.py
uv run python scripts/plot_issue563_base_panel.py
# VM (CPU): key-exclusion robustness artifact (round-2)
uv run python scripts/issue563_key_exclusion_robustness.py
```
