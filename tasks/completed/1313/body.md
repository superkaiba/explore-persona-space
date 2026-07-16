---
title: 'api_dispatch: retry 529 OverloadedError as transient + transport_loss split
  on judge paths (rule-24 sibling library fix)'
kind: infra
tags: []
created_at: '2026-07-15T05:23:10Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #952 china-politics-topup inline round: judge leg
  lost 144/144 draws to fast-failing HTTP-529; rule 24''s named deferred sibling library
  fix was never filed (registry title scan 2026-07-15: no match for 529/Overloaded
  besides rule task #1206).'
workflow: v1
---
## Overview / Motivation

Filed from the #952 `china-politics-topup` inline round (2026-07-15): the judge leg's initial dispatch lost 144/144 draws (48 divergence + 96 refusal) to HTTP-529 Overloaded errors that FAILED FAST instead of retrying — the known rule-24 library gap (`.claude/rules/llm-judging.md` rule 24), previously observed in #1090 (~2,638 stored 529 rows). #1206 codified the RULE; this task is the deferred sibling LIBRARY fix the rule names.

## Goal

Make 529 `OverloadedError` a retried transient in `api_dispatch.py`, and add the rule-24 transport-loss plumbing (per-row transport counter + batch-collection re-dispatch of transport-class failures).

## Bug observed

`src/explore_persona_space/llm/api_dispatch.py:580` catches `_anthropic.InternalServerError` with the comment "includes 529 OverloadedError" (also :570-571 docstring) — but in the installed SDK (anthropic 0.88.0) `OverloadedError`'s MRO is `OverloadedError → APIStatusError`, and `issubclass(OverloadedError, InternalServerError)` is False, so 529s bypass the transient tuple and fail fast.

- verified-at-filing: `grep -rn "OverloadedError" src/explore_persona_space/` → 2 hits, both in `llm/api_dispatch.py` (:570 stale docstring, :580 stale inline comment on the `InternalServerError` tuple entry); no `OverloadedError` entry in the transient tuple (2026-07-15). Empirical: #952 topup judge leg, 144/144 529s failed fast, recovered only by a manual rule-24(ii) re-dispatch.

## Proposed change (refine in planning)

Per rule 24's named sibling fix, `.claude/rules/llm-judging.md`:
1. Add `_anthropic.OverloadedError` (or `APIStatusError` with `status_code == 529`) to the transient retry tuple in `api_dispatch.py`; fix the stale :570/:580 comments.
2. Transport-class re-dispatch on the batch collection path (`eval/batch_judge.py` currently persists `error: True` rows; `eval/graded_judge.py::_score_from_parsed` folds them into content drops).
3. A separate `transport_loss` counter in `JudgeResult` so per-arm drop reports can split content-drops vs transport-losses (rule 24(ii)/rule 18).

## Scope / surfaces

- `src/explore_persona_space/llm/api_dispatch.py` (transient tuple + comments)
- `src/explore_persona_space/eval/batch_judge.py`, `src/explore_persona_space/eval/graded_judge.py` (transport-vs-content split + re-dispatch)
- Tests pinning: 529 retried; exhausted 529 surfaces as transport_loss, never a content drop; batch collection re-dispatches transport-class rows.

## Constraints / invariants

- Keep the AIMD 429 handling and existing cache keys byte-compatible (rule 22/23 cache caveats: do not change rubric-level cache keys).
- Bounded retries (existing max_attempts); fail loud on HTTP 400 invalid_request (rule 24(iii) unchanged).

## Provenance

Evidence: #1090 (rule 23/24 incident), #1206 (rule change), #952 china-politics-topup GPU-leg report 2026-07-15 (144 fast-failed 529s, 0 after manual re-dispatch). Rule text: `.claude/rules/llm-judging.md` rule 24 + its "Companion note (library, sibling task)".
