---
name: Anthropic cache_control on empty system text → Batch API rejects every request
description: An empty/whitespace system block carrying cache_control is rejected by the Anthropic Batch API on EVERY request; omit the system field when the rubric is empty
type: feedback
---

A judge caller that folds its WHOLE rubric into the user message and passes
`judge_system_prompt=""` (self-contained-rubric pattern) crashes the Anthropic
**Batch** API 100% — every request errors with
`invalid_request_error: system.0: cache_control cannot be set for empty text
blocks` — if the request-builder attaches `cache_control` to a `{"type":"text",
"text":""}` system block. The empty text itself is allowed; the `cache_control`
on empty text is the rejection.

**Why:** Lots of in-repo code attaches `cache_control` unconditionally to the
shared rubric block for prompt caching. When the rubric is empty that block is
both pointless AND illegal under cache_control.

**How to apply:** In any Messages-API param builder, when the system text is
empty/whitespace (`system_prompt.strip() == ""`) OMIT the `"system"` key
entirely — the Messages API accepts a request with no system field. Keep the
system block + cache_control for non-empty rubrics (don't regress
cross-experiment caching/comparability). The failure is INVISIBLE pre-crash: the
only post-hoc trace is the batch's per-request `invalid_request_error`
quarantine, so a self-contained-rubric caller errors silently for its whole run
and produces zero artifacts. Verify a fix against the LIVE Batch API through the
production entrypoint with `judge_system_prompt=""` + `threshold_base=0` (forces
batch) — the fix-engaged signal is `routing.path=="batch"` with non-error scores.
(#742 r10, 2026-06-30: `eval/judge_dispatch.py::_build_params` — every batch
request errored for ~1h55m, zero artifacts.)
