---
title: 'Batch-judge poll: add deadline + heartbeat + batch_id resume (eval/batch_judge.py)
  — wedge fix'
kind: infra
tags: []
created_at: '2026-06-25T05:33:01Z'
has_clean_result: false
origin_prompt: what is the wedge? [#661 judge batch stuck at 0/4800 for ~3h under
  a deadline-less poll]
---
## Goal

Harden `eval/batch_judge.py::_submit_and_poll_batch` so a slow/stuck Anthropic Batch
job fails cleanly instead of silently wedging a session for hours. Root-caused on #661
(2026-06-25): a 4800-request judge batch (`msgbatch_01Ux1…`) sat at `processing=4800,
succeeded=0` for ~3h; the poll loop (`while True: retrieve(); if ended: break; sleep`)
has no deadline, so the session blocked indefinitely, the watcher mis-flagged it as
stalled (2 respawns), and each respawn re-submitted a fresh batch (batch_id is a local
var). The GCP A100 idled the whole time behind the judge.

## Fixes

1. **Poll deadline + fail-loud:** bound total poll time (configurable; default tied to
   the run's budget). On exceed, raise a clean error (so the run fails loud / can fall
   back) instead of waiting up to the 24h batch expiry silently.
2. **Heartbeat during poll:** emit a progress signal each iteration (log + a parseable
   marker/sentinel) — e.g. "batch X: succeeded/total after Nm" — so (a) the watcher's
   stall-detector doesn't false-flag a legit long poll, and (b) a 0%-stuck batch is
   visible immediately rather than inferred.
3. **Persist + resume batch_id:** checkpoint the batch_id (e.g. in the judge cache dir)
   so a respawn RESUMES polling the in-flight batch instead of submitting a new one;
   only re-submit if no live batch is found.

## Scope / impact

`eval/batch_judge.py` is used by every judge-heavy experiment — the leakage program's
future phases + #653 follow-ups are all exposed. Land this before the program's
judge-heavy phases run. Watcher-side (the stall-detector mis-flagging long polls) is a
separate workflow-surface concern partly addressed by #661's own emitted candidates.

## Context

Surfaced diagnosing #661's Batch-judge wedge (gen safe on HF; the orphaned 4800-req
batch was cancelled). #661 itself stays `blocked` pending re-dispatch (which should ride
this fix, or run on RunPod for SSH-diagnosis).
