---
title: Give API-level stop_reason=refusal its own judge drop class (rules 9/24/18
  + judge_dispatch.py:670-681)
kind: infra
tags:
- wf-fix
created_at: '2026-08-06T15:14:53Z'
has_clean_result: false
parent_id: 1739
origin_prompt: 'workflow-fix candidate surfaced during #1739 evil-OOD judge wave:
  15,091 batch-path API refusals were tallied as content drops; sync re-issue showed
  0/14,887 re-refusals'
workflow: v1
---
## Goal

Give an API-level `stop_reason="refusal"` judge return its own drop class in `.claude/rules/llm-judging.md`, distinct from both content drops and transport losses, so a batch-path refusal censor can never again be silently absorbed into the content-drop tally.

## The gap

`llm-judging.md` rules 9/24 define a two-way split for judge returns that fail to yield a score:

- **content drops** — malformed / rubric-REFUSAL / out-of-range: DROPPED, never coerced;
- **transport failures** — RETRIED, never persisted as drops.

Neither class covers an Anthropic **API-level `stop_reason="refusal"`**, where the API's own safety classifier declines and returns an EMPTY content array. Because it arrives on a batch row whose `result.type` is `succeeded`, the current code path treats the empty string as a parse failure and files it as a CONTENT drop with no transport flag.

## Evidence (#1739, 2026-08-06)

A 44,310-call evil-OOD judge wave on the forced BATCH path had **15,091 draws (34.1%) return `stop_reason="refusal"`** with empty content. All 15,091 were tallied as content drops. Realized coverage collapsed to 9,260 of 14,770 items with a full 3 draws; 4,982 items (33.7%) had ZERO valid draws.

The classification was materially wrong, and demonstrably so:

- Re-issuing the identical instrument (same model, rubric, temperature, max_tokens) on the **SYNC** path produced **0 re-refusals in 14,887 draws**. The censor is transport-conditional, not a property of the content.
- A sync re-judge rescued 5,077/5,172 censored items (98.2%), restoring coverage to 14,568/14,770 (98.6%).
- Merge validity was confirmed: 287 items scored on BOTH paths gave batch mean 7.26 vs sync 7.77, no calibration offset.
- The censoring was **outcome-correlated** — rescued items scored 1.4x higher overall than clean items (mhj 2.3x, pair 3.7x, tom-gibbs 1.24x). Filing them as content drops therefore silently biased the DV downward on exactly its most positive rows.

Two distinct mechanisms were observed and are worth encoding: for mhj/pair the classifier keys on the ANSWER's severity (within-item severity gradient); for tom-gibbs it keys on the jailbreak QUERY in the `{question}` slot, censoring ~2/3 of the corpus near-indiscriminately.

## Extraction seam

`src/explore_persona_space/eval/judge_dispatch.py:670-681` — a SUCCEEDED batch row with an empty content array yields `""`, `parse_judge_json` returns None, and the row is persisted as a `parse_error` CONTENT drop with no transport flag. This is the exact line range where the third class needs to branch.

## Deliverables

1. A third drop class in `llm-judging.md`'s rule-18 drop split — name it, define it as transport-conditional-and-retriable, and state that it must be reported separately from both content drops and transport losses.
2. `judge_dispatch.py` detects `stop_reason == "refusal"` on a succeeded row and classifies it into that class rather than as a parse error.
3. The reduce-step tally reports the class separately; a non-zero count should be loud, since it means the wave ran a transport that censors.
4. Guidance on the remediation that worked: targeted SYNC re-issue at an identical instrument, merged with the genuine batch draws, with a dual-scored parity check (~200-300 overlapping items) reported to license the merge, and the split disclosed in `judge_meta`.

Reference implementation that already does this end to end: `scripts/issue1739_evilood_refusal_rejudge.py`.

## Out of scope

The sibling pilot-gate transport-parity gap (rule 26 pilots routing SYNC while gating a forced-BATCH wave) is filed separately.
