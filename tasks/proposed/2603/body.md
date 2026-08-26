---
title: 'EngineCore drain guard fails OPEN on UUID/MIG-form CUDA_VISIBLE_DEVICES pins
  (fix + UUID-pin test, due since #2546 round 11)'
kind: infra
tags: []
created_at: '2026-08-26T12:47:14Z'
has_clean_result: false
origin_prompt: 'Codex twin on #2546 review round 14: ''Third consecutive scoped crash-fix
  leaves the promised source-helper UUID/MIG fail-loud follow-up unlanded.'' Raised
  as BLOCKER at round 11, reconciler-downgraded to CONCERN with an explicit fix+test
  obligation, carried unlanded through rounds 12/13/14. Filed to give it an owner
  outside per-round scope.'
workflow: v1
---
---
kind: infra
---

# EngineCore drain guard fails OPEN on UUID/MIG-form CUDA_VISIBLE_DEVICES pins

## Goal

Land the fix + regression test that four consecutive review rounds on task #2546 recorded as due and legitimately deferred. When a worker's `CUDA_VISIBLE_DEVICES` is set in UUID or MIG form (`GPU-<uuid>` / `MIG-<uuid>`) rather than as a numeric index, the visible-device set resolves EMPTY and the EngineCore drain guard therefore passes **without inspecting the assigned GPU at all**. A guard that cannot see its target is not a guard; it is a silent pass.

## Provenance and why this is filed rather than deferred a fourth time

Raised on task #2546 review round 11 as a **BLOCKER**: "UUID/MIG-form worker CVD pins resolve to an empty visible-device set, making the EngineCore drain guard pass without inspecting the assigned GPU."

Downgraded the same round by the reconciler to CONCERN, with reasoning that still holds: the fail-open is real but **trunk-pre-existing** (the `eval_battery` path was untouched by that round) and **unreachable by the current lanes**, which pin numeric device indices. The downgrade came with an explicit obligation — "fix + UUID-pin test".

It was then carried three more rounds, each deferral individually defensible because each round was a narrowly scoped live-fleet crash fix:

```
round 12  CONCERN  Required source-helper UUID/MIG fail-loud landing remains due;
                   legitimately deferred by this scoped live-fleet crash fix.
round 13  CONCERN  ...remains due after a second legitimately scoped crash-fix deferral.
round 14  CONCERN  Third consecutive scoped crash-fix leaves the promised
                   source-helper UUID/MIG fail-loud follow-up unlanded.
```

The Codex twin raised it again unprompted at round 14, which is the signal that matters: no individual round is at fault, and the fix still is not landing. Folding it into #2546's next round would repeat the pattern — that round is a payload-provenance stamping fix in a different file, so its reviewers would rightly call this out of scope. A promised fix with no owner outside per-round scope is the stranded-fix shape, so it gets its own task and its own review.

## What to do

1. Make the visible-device resolution handle UUID and MIG forms, or **fail loud** when it cannot resolve the pin to a concrete device. Failing loud is acceptable and preferred over guessing: the current behavior is the worst option, since it reports success having inspected nothing.
2. Add the UUID-pin regression test the round-11 reconciler required — a test that the guard REFUSES (or raises) rather than passes when handed a `GPU-<uuid>` / `MIG-<uuid>` form pin. Assert the failure, not just the absence of a crash.
3. Check for sibling call sites with the same numeric-index assumption; a repo-wide grep for `CUDA_VISIBLE_DEVICES` parsing is in scope. Report what you find even if you do not change it.

## Constraints

- Fail fast. Do not add a fallback that swallows an unresolvable pin — that is the defect.
- Do not widen this into a refactor of the eval battery. Fix the resolution + guard, add the test, stop.
- The lanes currently in use pin NUMERIC indices, so this is latent rather than actively firing. Do not treat that as licence to weaken the fix; treat it as why the change is low-risk to land.

## Scope

Pure code fix plus a regression test. No GPU required for the test (the pin form is a string-parsing and refusal-path concern). Not an experiment; produces no promotable clean-result.

## Files of record

Task #2546 concerns ledger, `teardown-cvd-uuid-fail-open`, five rows spanning rounds 11-14 (including the reconciler's round-11 downgrade with its fix+test obligation). #2546 `epm:code-review-codex v14` re-raised it.
