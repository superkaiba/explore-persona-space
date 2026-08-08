---
title: 'Fence-tolerant judge JSON parse: 2.02% of judged calls drop on markdown fences,
  not truncation'
kind: infra
tags:
- judge-parse
created_at: '2026-07-31T07:48:41Z'
has_clean_result: false
origin_prompt: '#1773 full-dict describe: 2,602/128,712 (2.02%) parse failures; direct
  stop_reason probe shows 39/40 end_turn COMPLETE, cause is markdown fencing not the
  token cap; fence-strip recovers 29/40'
workflow: v1
---
## Overview / Motivation

During #1773's full-dictionary run (2026-07-31), the describe phase logged 2,602 parse failures out of 128,712 dispatched items (2.02%). A direct Messages-API probe with `stop_reason` established the cause is NOT the token cap: 39/40 sampled failures returned `end_turn` COMPLETE (1/40 `max_tokens`), and on the axes side 0/60 were truncated with 37% headroom at p100. The failures are markdown-fenced responses that the shared judge parser does not recover.

The parser already INTENDS to handle this — `parse_judge_json` docstring says it extracts a "``{``-anchored object embedded in noisy text (markdown fences, preamble)". But it anchors on `text.index("{")`, the FIRST brace in the response, and then `raw_decode`s from there. When the model emits reasoning prose containing a brace before the JSON object, the anchor lands on the wrong position and the decode fails. A fence-aware strip recovers 29 of 40 sampled failures.

## Goal

Make `parse_judge_json` fence-tolerant so judge responses wrapped in markdown code fences (or preceded by brace-bearing prose) parse instead of dropping, and re-judge #1773's banked failures against a fresh cache to recover the lost coverage.

## Workflow gap

- **Bug observed:** `src/explore_persona_space/eval/utils.py` `parse_judge_json` anchors on the first `{` and `raw_decode`s from it. Responses of the shape ```` ```json\n{...}\n``` ```` preceded by prose containing a brace fail to parse and are DROPPED (correctly, per the never-coerce rule) — but they are recoverable content, not genuine content drops. Measured 2.02% of describe calls on #1773's 128,712-item dispatch; a fence-strip parses 29/40 of a sampled 40.
- **Why it is a gap:** the drop is silent-by-design (drop-never-coerce is correct) so the loss only surfaces as reduced coverage, and the surviving population looks clean. It affects EVERY judged call site in the project, not just #1773.
- **Confidence:** high — cause established by direct `stop_reason` probe, not inferred from text shape.
- verified-at-filing: read `src/explore_persona_space/eval/utils.py` lines 29-42 at compose time (2026-07-31). Confirmed: `json.loads(text)` first, then `start = text.index("{")` + `raw_decode`, then the `%.200s`-truncated warning. The docstring at line 16 already claims markdown-fence tolerance, so this is a stated-intent-vs-behaviour gap, not a missing feature request.

## Proposed change (candidate diff sketch — refine in planning)

In `parse_judge_json`, before giving up:
1. If the text contains a fenced block (```` ```json ```` / ```` ``` ````), strip the fence and retry the whole-text parse.
2. Failing that, iterate candidate `{` positions rather than only the first, `raw_decode` at each, and return the first successful decode (optionally preferring the LARGEST successful object).
3. Leave the drop-never-coerce contract intact: a genuinely unparseable response still returns `None`.

Also worth fixing while in there: the failure log uses `"Text: %.200s"`, which truncates the logged response to 200 characters. That is what made this bug hard to diagnose — two independent agents inspected the log and concluded "truncation" because a closing brace can never appear within 200 chars of a fenced multi-line response. Consider logging a longer prefix, or the `stop_reason` where available, so the truncation-vs-format axis is readable from the log instead of requiring a live API probe.

## Scope / surfaces

- Primary: `src/explore_persona_space/eval/utils.py` (`parse_judge_json` + its failure log).
- Consumers are every judged call site; the change must be strictly widening (anything that parses today must still parse identically).
- Follow-up run, separable: re-judge #1773's 2,602 describe failures against a FRESH cache dir — the rubric-keyed judge cache excludes `max_tokens` and would re-serve the failed entries (`.claude/rules/llm-judging.md` rules 22/24). Estimated ~$15 at the realized basis; expected to lift coverage from 126,110 toward ~128,600 features.

## Constraints / invariants

- Drop-never-coerce is UNCHANGED. This recovers parseable responses; it must never coerce an unparseable one.
- 17.1% of #1773's logged failures (439/2,572) are EMPTY responses — a content-class outcome no parser change can fix. Do not count those toward expected recovery.
- The #1773 length-censoring worry was checked and does NOT hold: re-judged failed-feature outputs (mean 271 / median 230 / p90 412 tokens) sit on top of the overall realized distribution (mean 255 / median 240 / p90 361), so failures are marginally longer, within noise. This fix is about coverage, not bias correction.
- Must not be applied underneath a live dispatch. #1773's axes leg (3.2M calls) is in flight; land and test this independently, then re-judge.

## Provenance

Origin: #1773 full-dictionary run, 2026-07-31. The orchestrator (me) twice mis-diagnosed this as token-cap truncation from the log alone; the run's implementer refuted it with a direct Messages-API `stop_reason` probe and correctly declined to edit shared library code underneath a live 3.2M-call dispatch, recommending it as its own reviewed change. This is that change.
