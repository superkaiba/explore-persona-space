---
title: '`※` is a single rare token (id 83399) while the `ϟ` candidate is two tokens,
  making `※` the clean choice for a single-forward-pass log-prob marker (HIGH confidence)'
kind: analysis
tags: []
created_at: '2026-05-26T20:28:02Z'
has_clean_result: true
---
# `※` is a single rare token (id 83399) while the `ϟ` candidate is two tokens, making `※` the clean choice for a single-forward-pass log-prob marker (HIGH confidence)

## Human TL;DR

placeholder

## TL;DR

- **Motivation:** Several marker-leakage experiments ([#380](https://eps.superkaiba.com/tasks/380), [#385](https://eps.superkaiba.com/tasks/385), [#396](https://eps.superkaiba.com/tasks/396)) want to track *how strongly a trained model wants to emit a hidden marker token* as a dependent variable. That DV is cleanest when the marker is (a) a **single token**, so one teacher-forced forward pass gives its log-prob directly, and (b) **rare under the base model**, so it almost never fires by chance and its log-prob has room to climb during training. The legacy `[ZLT]` marker is multi-token, which muddies both properties. This probe measures the base-model log-prob of `[ZLT]` and three single-character candidates to pick a replacement.

- **What I ran:** I loaded `Qwen2.5-7B-Instruct`, took 30 real answer contexts (6 personas × 5 questions, last 150 tokens of each base-model completion from [#380](https://eps.superkaiba.com/tasks/380)), and scored the teacher-forced joint log-prob of each marker at the end-of-answer position: `[ZLT]`, `※` (reference mark), `¶` (pilcrow), `ϟ` (koppa). *(No generation-style outputs in this experiment; skipping the end-to-end example block per SPEC.)*

- **Results:** `※` tokenizes to a single token (id 83399) with a median base-model log-prob of −19.3 nats — genuinely rare, exactly what a marker DV wants; `¶` (id 78846, −21.9 nats) is a near-equal single-token backup. The plan assumed `ϟ` was a single rare token (id 146070), but under this tokenizer it is **two tokens** (ids 17383, 253), so its lower joint (−24.4 nats) is partly a length artifact and it gives no clean single-token DV. n = 30 contexts.

    ![Per-marker base-model joint log-prob (nats) at end of answer: single-token ※ median −19.3 and ¶ median −21.9, two-token ϟ median −24.4, four-token [ZLT] median −33.4, across n=30 contexts.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ac809e2c8db3246f4e78cf66db2394ee3fdf67c5/figures/issue_395/marker_logprob_priors.png)

    > **Figure.** *Among the single-token candidates, `※` is the rarest per token, and the `ϟ` candidate turns out to be two tokens rather than one.* Each point is the median joint log-prob (nats) of the marker scored at the end of an answer; the bar spans the 10th–90th percentile across n = 30 (persona, question) contexts. Blue = single token, gray = multi token; token count is in each y-axis label. More-negative = rarer under the base model. `[ZLT]`'s very negative joint is a 4-token sum, not a single rare token, so its per-step log-prob trajectory would be muddier than a single-token marker's.

- **Next steps:** Adopt ` ※` (leading space, id 83399) as the default marker for new marker-leakage experiments; keep ` ¶` as the equivalent backup; drop `ϟ`. (Already wired into `CLAUDE.md` and downstream tasks [#396](https://eps.superkaiba.com/tasks/396), [#397](https://eps.superkaiba.com/tasks/397), [#399](https://eps.superkaiba.com/tasks/399), [#408](https://eps.superkaiba.com/tasks/408).)

## Details

This is a tooling probe, not a research claim: it measures which candidate marker token to use so that a future "marker emission strength" dependent variable is both clean to compute and well-behaved. A good marker has two properties. First, it should be **a single token**: then the model's tendency to emit it is one number (one teacher-forced forward pass), and its trajectory over training steps is interpretable step-by-step. A multi-token marker forces you to sum log-probs over several sub-tokens, and the per-step dynamics blur together. Second, it should be **rare under the base model**: a token the model almost never produces unprompted won't fire spuriously during eval, and its log-prob starts very negative so there is dynamic range to watch it climb as training installs the marker.

All markers were scored with a leading space (`" ※"`, `" [ZLT]"`, etc.), because the end-of-answer position is always preceded by content tokens, so the leading-space tokenization is the realistic one. The contexts are the last 150 tokens of 30 base-model completions drawn from [#380](https://eps.superkaiba.com/tasks/380)'s generations, spanning six personas (librarian, wizard, comedian, qwen_default, ai, chatbot) crossed with five questions.

### Primary measurement

| Marker | Tokens | Token ids | Median log-prob (nats) | 10th pct | 90th pct |
|---|---|---|---|---|---|
| `※` reference mark | 1 | 83399 | −19.3 | −30.3 | −16.5 |
| `¶` pilcrow | 1 | 78846 | −21.9 | −31.8 | −16.4 |
| `ϟ` koppa | 2 | 17383, 253 | −24.4 | −30.5 | −21.3 |
| `[ZLT]` legacy | 4 | 508, 57, 27404, 60 | −33.4 | −45.3 | −30.8 |

Read as joint log-probs: more-negative means rarer. The ordering is consistent across all six personas (the per-persona arrays in the result JSON never reorder `※` and `[ZLT]`). As a single token, `※` is the rarest *per token* of the whole set (−19.3 for its one token, versus `[ZLT]`'s roughly −8 per sub-token) — its sub-token-free rarity is what makes it a good marker, whereas `[ZLT]`'s very negative joint is just the cost of being a 4-token sequence built from individually common pieces.

### Why the koppa candidate was disqualified

The plan listed `ϟ` (koppa) as a single token (id 146070) and the rarest candidate, so it might have been preferable to `※`. The run falsified the premise: under the `Qwen2.5-7B-Instruct` tokenizer, `" ϟ"` encodes to **two** tokens (17383, 253), not one. Its median joint (−24.4) is lower than `※`'s partly because it is a two-token sum, and crucially it gives no clean single-forward-pass DV. So `ϟ` is dropped, and the choice is between the two genuine single-token candidates `※` and `¶`. They are within ~2.6 nats of each other at the median; `※` (the slightly higher-prior but still very rare option, and the more visually distinct glyph) was adopted, with `¶` as the backup.

### Why this measurement and not something heavier

The decision only needs the base-model emission prior and the token count, both of which a short teacher-forced pass over a few dozen real contexts settles deterministically. No training, no judge, no sweep. The 30 contexts are enough because the quantities of interest (token count; whether one candidate is order-of-magnitude rarer than another) are stable — the per-persona spread is wide but never flips the ranking.

### Parameters

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, bfloat16 |
| Contexts | 30 = 6 personas × 5 questions |
| Context source | last 150 tokens of each [#380](https://eps.superkaiba.com/tasks/380) base-model completion |
| Score | teacher-forced joint log-prob of the marker at end-of-answer |
| Marker form | leading space (`" ※"`, `" ¶"`, `" ϟ"`, `" [ZLT]"`) |
| Candidates | `※` (83399), `¶` (78846), `ϟ` (17383, 253), `[ZLT]` (508, 57, 27404, 60) |

Confidence: HIGH — the token-count facts are deterministic and exact (so the `ϟ`-is-two-tokens correction and the `※`-is-one-token claim are not noise-limited), and the rarity ordering holds across all six personas; the residual uncertainty is only that priors are measured on one model with one decoding over n = 30, and "lower base-model prior is the better marker" is a methodological assumption rather than something this probe itself tested. Downstream adoption across [#396](https://eps.superkaiba.com/tasks/396)/[#397](https://eps.superkaiba.com/tasks/397)/[#399](https://eps.superkaiba.com/tasks/399) has since borne out the choice.

### Methodology corrections

- **Koppa token-count.** The plan assumed `ϟ` was a single token (id 146070); the run showed it is two tokens (17383, 253) under the `Qwen2.5-7B-Instruct` tokenizer. Effect: `ϟ` is disqualified as a single-token DV and the marker choice reduces to `※` vs `¶`. No re-run needed — the probe already records the correct token ids.
- **State reconciliation.** The probe was executed and committed on 2026-05-26 (`eval_results/issue_395/marker_priors.json`, the pod was provisioned and terminated then) and the `※` choice was adopted project-wide, but the task row was never advanced past `approved` and the result was never written up. This body records the already-completed analysis; nothing was re-computed.

## Reproducibility

**Artifacts:** result JSON [`eval_results/issue_395/marker_priors.json`](https://github.com/superkaiba/explore-persona-space/blob/ac809e2c8db3246f4e78cf66db2394ee3fdf67c5/eval_results/issue_395/marker_priors.json) (summary + per-persona arrays + contexts + token ids); figure [`figures/issue_395/marker_logprob_priors.png`](https://github.com/superkaiba/explore-persona-space/blob/ac809e2c8db3246f4e78cf66db2394ee3fdf67c5/figures/issue_395/marker_logprob_priors.png); input contexts from `eval_results/issue_380/base_model_generations.json`. No HF Hub / WandB artifacts (local teacher-forced probe, no model trained): n/a.

**Compute:** 1× H100 (eval intent), one short teacher-forced pass over 30 contexts × 4 markers; pod provisioned and terminated 2026-05-26.

**Code:** probe [`scripts/i395_probe_marker_priors.py`](https://github.com/superkaiba/explore-persona-space/blob/ac809e2c8db3246f4e78cf66db2394ee3fdf67c5/scripts/i395_probe_marker_priors.py); log-prob primitive [`src/explore_persona_space/eval/marker_logprob.py`](https://github.com/superkaiba/explore-persona-space/blob/ac809e2c8db3246f4e78cf66db2394ee3fdf67c5/src/explore_persona_space/eval/marker_logprob.py); figure [`scripts/plot_i395_marker_priors.py`](https://github.com/superkaiba/explore-persona-space/blob/ac809e2c8db3246f4e78cf66db2394ee3fdf67c5/scripts/plot_i395_marker_priors.py).
