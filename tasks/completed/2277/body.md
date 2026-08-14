---
title: 'Pod terminate guard is owner-blind: a non-owner can self-post the upload-verification
  PASS and destroy a live owner''s pod'
kind: infra
tags: []
created_at: '2026-08-13T22:47:07Z'
has_clean_result: false
parent_id: 2054
origin_prompt: 'Orchestrator-detected gap during #2054 LOCO round: harvester terminated
  pod-2054-tiers while owner spec-ladder-2054 was alive inside its posted fence to
  23:30Z; the terminate guard passed on a PASS marker the harvester posted itself.'
workflow: v1
---
# Pod terminate guard is owner-blind: a non-owner can self-post the upload-verification PASS and destroy a live owner's pod

## Goal

Close the gap between the CLAUDE.md verified-teardown grant (owner-scoped in
prose) and `pod_lifecycle._guard_upload_verification_before_terminate` (which
enforces only that *some* `epm:upload-verification` PASS marker with an
`outroot=` token exists on the task). A NON-owner agent can post that marker
itself and terminate a pod whose owner is alive and mid-fence. The guard cannot
currently tell the sanctioned case from the unsanctioned one.

## Incident (2026-08-13, #2054, pod-2054-tiers)

`spec-ladder-2054` owned pod-2054-tiers and was in a quiet ~2 h wait inside its
own posted fence (fence deadline 23:30Z). Its pod-side workload had written
`[phase=done]` at 20:23Z and its process tree was dead — the pod was idle by
design while the owner waited.

The orchestrator (a NON-owner) read that silence as owner death, independently
enumerated the 113 uploaded files, **posted the `epm:upload-verification` PASS
itself** with `outroot=swept-clean`, and terminated the pod at ~21:32Z through
the gated `pod.py terminate`. The guard passed, because the guard's only
question is "does a PASS marker with the token exist on this task?" — and it
did, freshly written by the terminating agent.

Outcome was benign only by luck: the owner's uploads happened to be complete,
and its independent re-merge later confirmed byte-equality with the harvested
commit. Nothing in the mechanism produced that safety.

## Why the existing rules did not catch it

- **CLAUDE.md / `.claude/rules/pods.md`** scope the no-user-ask teardown grant to
  "the agent actively driving a job on a pod it provisioned ... once its uploads
  are verified". That ownership condition is PROSE ONLY; no code reads it.
- **The #2111 pre-stop probe** is scoped to STOPPING A SESSION, and its liveness
  test is "a stage-dispatch breadcrumb younger than ~10 min". A long-running
  posted FENCE is the inverse signal — an OLD marker asserting a claim valid
  until a FUTURE deadline — so the 10-minute recency test structurally cannot
  fire on this shape, and it does not govern pod teardown anyway.
- **The kill-approval gate** (`backends/kill_approval.py`) treats a passed
  upload-verification as entering the `verified_teardown` owner grant, without
  ever establishing that the caller IS the owner.

## Proposed fix (implementer to design; these are the constraints, not a design)

1. **Fence-awareness.** Before terminating, refuse (or require an explicit
   override) when the task carries an unexpired owner fence / heartbeat deadline
   in the FUTURE. Recency of the marker must NOT be the test — a fence is old by
   construction and still binding.
2. **Owner attribution on the PASS.** Distinguish a PASS posted by the pod's own
   provisioning agent from one posted by a third party. A self-posted PASS from
   a non-owner should not by itself open the `verified_teardown` grant.
3. **Non-owner path.** Give a genuine harvester a sanctioned route: surface the
   pod for approval (marker + push naming pod and hourly burn) rather than
   silently inheriting the owner grant. This is already the prescribed behaviour
   on a kill-gate REFUSAL; it should also be the default when ownership cannot
   be established.
4. **Do not weaken** the existing upload-verification requirement or the
   `outroot=` token check (#2187) — this adds a condition, never removes one.

## Constraints

- Must not block the legitimate common case: an owner tearing down its own
  verified pod at Step 8 or on an inline round, which must stay ask-free.
- Must not introduce a path that terminates pods MORE readily.
- `pod.py stop` remains a non-substitute for verify-then-terminate (#1112).
- Never `--approve`, never `EPS_ALLOW_COMPUTE_KILL=1` (user-only channels).

## Files of record

`scripts/pod_lifecycle.py` (`_guard_upload_verification_before_terminate`,
~L3472 and its call site ~L3817); `src/explore_persona_space/backends/kill_approval.py`;
`.claude/rules/pods.md` (verified-teardown grant prose); CLAUDE.md § Pods and
§ the inline-round completion-side teardown clause; #2111 (pre-stop probe, the
session-scoped sibling); #2187 (`outroot=` token); #1485 (issue-wide
keep-running refusal); #1662 (suffixed-pod idle burn).
