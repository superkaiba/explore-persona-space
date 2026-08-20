---
title: 'gotchas.md: Anthropic Batch request_counts.succeeded lags — never key a stall/cancel
  verdict on it'
kind: infra
tags: []
created_at: '2026-08-20T16:02:37Z'
has_clean_result: false
parent_id: 823
origin_prompt: Filed from /issue 823 after cancelling a batch that had completed 997/1000
  requests, on a succeeded=0 reading
workflow: v1
---
# Add a gotchas.md entry: Anthropic Batch `request_counts.succeeded` LAGS — never key a stall/cancel verdict on it

## Goal

Add an entry to `.claude/rules/gotchas.md` recording that an in-flight Anthropic
Message Batch can report `request_counts.succeeded == 0` while being minutes from
completion, so a monitoring agent must never key a stall / drain / **cancel** verdict
on that counter. The reliable signal is the batch's own `processing_status`
transitioning to `ended`.

## The incident (#823, 2026-08-20)

A 14,996-call generation wave ran as 15 sub-batches. The last one (sb1) sat at
`processing=1000, succeeded=0` for 1 h 25 m on a congested org (93 in-flight batches /
521,829 requests processing, org-wide `succeeded` also 0). The session read that as
"not draining", CANCELLED the batch, and re-submitted its 1,000 calls on an idle org.

The cancelled batch then ended reporting **`succeeded: 997, canceled: 3`** — it had
completed 997 of 1,000 requests and was ~minutes from done. Cost: ~1,000 duplicated
calls (~$2.3) plus 997 billed-and-discarded generations.

Corroborating observations from the same session, which make this a COUNTER property
rather than a one-off:

- A different org showed `succeeded: 0` summed across 8 in-flight batches holding
  **136,256** processing requests.
- The wave's own other 14 sub-batches each read `processing=1000, succeeded=0` while
  in flight, then appeared as complete — no intermediate `succeeded` value was ever
  observed across repeated sampling.

## Why this belongs in gotchas.md

`.claude/rules/gotchas.md` is the always-on trap index for exactly this shape: an API
whose observable reads plausibly-but-misleadingly, where the wrong inference is
expensive and the right probe is one line different. Any agent monitoring a batch wave
(the `eval.batch_judge` poller path, a generation-wave driver, an ad-hoc watcher) can
reach for `request_counts.succeeded` as a progress signal and draw the same conclusion.

The neighbouring existing entries — the Batch-API empty-system-block 400, the
`custom_id` charset/length constraints, `judge_pilot_gate` satisfiability — cover
REQUEST-shape traps. This is the first MONITORING-side batch trap.

## Proposed entry (draft; the implementer should sharpen)

> **Anthropic Batch `request_counts.succeeded` LAGS — an in-flight batch minutes from
> completion can report `succeeded: 0`, so a stall / drain / cancel verdict keyed on it
> is unsound.** Measured (#823): a 1,000-request batch read `processing=1000,
> succeeded=0` for 1 h 25 m, was cancelled as "stalled", and ended reporting
> `succeeded: 997, canceled: 3`; separately, 8 in-flight batches holding 136,256
> processing requests summed to `succeeded: 0`. RULE: key drain/liveness on the
> batch's own `processing_status` transition to `ended` (and on the sibling
> `recently ended` count when judging an ORG's throughput), never on `succeeded` while
> `in_progress`. COROLLARY (the reasoning half): discovering that your progress signal
> is unreliable is evidence you CANNOT MEASURE the quantity — never evidence the
> quantity is zero; an unmeasurable value must not be promoted to a measured null to
> justify a destructive action. A batch queued within its 24 h SLA self-harvests for
> free (`.claude/rules/auto-continuation.md` § FREE no-data-loss path) — cancelling to
> "unblock" it discards completed, billed work.

## Acceptance criteria

1. An entry in `.claude/rules/gotchas.md` covering the counter behavior + the
   `processing_status`-keyed rule + the inference corollary, citing `#823`.
2. Placed near the existing Anthropic Batch-API entries (empty-system-block,
   `custom_id` shape) so a reader hitting one finds the family.
3. `uv run python scripts/workflow_lint.py` clean (byte-budget checks included —
   gotchas.md is under a size ratchet, so if the entry pushes it over, relocate a
   topic-owned entry per the #2189 precedent rather than trimming this one).
4. No behavioral code change is required; if a repo-side monitoring helper is found
   keying on `succeeded` while `in_progress`, fix it in the same round and name it.

## Notes for the implementer

- Do NOT generalize to "`request_counts` is unreliable" — `processing`, `errored`, and
  the terminal counts read correctly; the specific trap is `succeeded` DURING
  `in_progress`.
- Whether the counter is genuinely non-incremental or merely coarse-grained was not
  established — the incident only establishes that a near-complete batch can read 0.
  Word the entry to that evidence; do not overclaim the mechanism.
- Check `src/explore_persona_space/llm/api_dispatch.py` and
  `src/explore_persona_space/eval/batch_judge.py` for any progress/stall logic reading
  `succeeded` mid-flight.

## Provenance

Filed by the `/issue 823` session (same-issue follow-up
`inconsistent-origin-persona-ladder`) after making exactly this error. Full incident
record: task #823 `epm:progress` v127.
