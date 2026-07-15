---
title: 'hub.py::_upload: bounded transport-class retry (429/Xet queue-full) before
  no-path return'
kind: infra
tags: []
created_at: '2026-07-15T19:00:44Z'
has_clean_result: false
origin_prompt: '#1315 r8 prose follow-up: fleet-shared bounded 429 retry inside orchestrate/hub.py::_upload'
workflow: v1
---
## Overview / Motivation

Filed from #1315 r8's prose follow-up (experiment-implementer): the r8 fix added a bounded transport retry at #1315's dispatcher seam only; the ROOT cause lives in the fleet-shared `src/explore_persona_space/orchestrate/hub.py::_upload`, which catches HF-429/Xet-queue transport failures, logs them, and returns an empty path — every OTHER dispatcher that fail-fasts (or silently continues) on that return inherits the same final-phase kill (or #488-style silent loss) under sustained fleet HF traffic.

## Goal

Add a bounded, jittered transport-class retry (429 / 5xx / Xet queue-full; ~3 attempts, 30/60/120s backoff) INSIDE `orchestrate/hub.py::_upload`'s failure path, so all dispatchers inherit it; keep content-class errors fail-loud and un-retried; preserve the existing return contract (no-path only after retry exhaustion).

## Constraints / invariants

- `src/` change — full code pipeline (planner critique-round exempt sizing: 0 GPU-h).
- Do NOT change the no-path return contract consumed by existing callers (e.g. #1315's `_upload_with_transport_retry` outer retry remains compatible — nested retries are bounded and idempotent, but the plan should consider capping total attempts or having callers detect the inner retry).
- Tests: fake transport failures N-then-success + exhaustion; verify no retry on content-class raises.
- Reference incident: #1315 epm:failure v7/v8 (two p11 kills ~35 min apart, 2026-07-15); worked seam example @ issue-1315 commit c3c600541f.

## Provenance

- origin: #1315 r8 implementer prose follow-up ("a fleet-shared bounded 429 retry inside orchestrate/hub.py::_upload would fix this class for every dispatcher").
