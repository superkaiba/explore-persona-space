---
title: Judge parser drops prose-then-trailing-integer responses (recovers ~8-28% of
  dropped draws)
kind: infra
tags: []
created_at: '2026-08-06T06:09:55Z'
has_clean_result: false
parent_id: 2091
origin_prompt: 'Diagnosed during #2091 G2 judge pilot gate: parse_judge_json discards
  recoverable scores when the judge emits reasoning prose followed by a trailing bare
  integer.'
workflow: v1
---
## Overview / Motivation

`eval/utils.py::parse_judge_json` + `eval/graded_judge.py::_score_from_parsed` silently DROP judged draws whose response is reasoning prose followed by a trailing bare integer — a shape Sonnet emits routinely. The score is present in the response and mechanically recoverable; the parser discards it, and `drop-never-coerce` (llm-judging rule 9) then records it as a legitimate content-drop. Measured drop rates across banked #1739 DVs suggest this class accounts for a large share of existing dropped draws.

## Goal

Recover the prose-then-trailing-integer judge response shape so real judgments are not discarded, WITHOUT changing the score of any response that currently parses.

## Evidence (measured on #2091, 2026-08-06)

Five failing draws re-issued through the exact production instrument (`claude-sonnet-4-5-20250929`, `max_tokens=1024`); all returned `stop_reason=end_turn` at 329-403 output tokens — no truncation, no refusal:

| shape | `parse_judge_json` | outcome |
|---|---|---|
| bare `0` | `int 0` | scored |
| prose + `{"score": 5}` | `dict` | scored |
| prose + `{"score": 75}` | `dict` | scored |
| prose + trailing bare `70` | `None` | **dropped** |
| prose + trailing bare `70` | `None` | **dropped** |

`parse_judge_json` returns `json.loads(text)` verbatim and recovers a trailing JSON OBJECT after prose but not a trailing BARE INTEGER. `_score_from_parsed` already accepts a bare in-range numeric (its docstring documents Sonnet's envelope-less habit — the #778 r3 fix) but only when the response IS just the integer.

Dropped-draw rates in banked #1739 DVs (`per_rollout_scores` None), which this class plausibly dominates:

| cell | draws | dropped | rate |
|---|---|---|---|
| evil trait rungs | 53,330 | 15,076 | 28.27% |
| wildchat / hallucination | 10,000 | 830 | 8.30% |
| wildchat / sycophancy | 10,000 | 228 | 2.28% |
| wildchat / evil | 10,000 | 178 | 1.78% |
| sycophancy trait rungs | 86,520 | 371 | 0.43% |

A #2091 pilot arm measured 28.0% (14/50) on hallucination_trait × wildchat.

## Constraints (load-bearing)

- **Conservative anchor only.** Recover a trailing line that is EXACTLY an integer in [0,100]. NEVER "last integer anywhere in the text" — these responses contain prose numerals ("14TB disk 5", "8TB disk 4") that a greedy scan would capture as scores.
- **Bit-identical on currently-parsing responses.** Pin with a regression test over the existing parsed shapes (bare int, prose+JSON object, plain JSON object); any score change on those is a FAIL.
- **Drop-never-coerce is UNCHANGED.** REFUSAL, non-numeric, and out-of-[0,100] returns still drop (rule 9). This recovers a parseable score that exists, never invents one.
- **Report the recovery rate.** Emit a counter distinguishing recovered-trailing-integer draws from genuine content-drops so the change's effect is measurable rather than invisible.

## Cross-arm comparability warning (READ BEFORE APPLYING TO ANY IN-FLIGHT TASK)

Banked judge caches store ONLY the parsed result (13-byte `{"score": N}` files) — the raw judge text is NOT retained, so banked arms CANNOT be re-parsed under an improved parser without re-judging. Any experiment comparing a freshly-judged arm against a banked arm therefore faces an instrument split if the new parser is applied to one side only. #2091 deliberately did NOT adopt this fix mid-run for exactly that reason (see its `epm:progress` v35). Landing this fix must not silently change the instrument for in-flight comparisons — coordinate with any live task reusing banked judgments.

## Provenance

Diagnosed during #2091's G2 judge pilot gate (2026-08-06). Full analysis, including two refuted hypotheses and the banked-vs-greedy rate comparison, is in `epm:progress` v35 on task #2091.
