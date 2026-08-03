---
title: 'daily-fix: GCP DWS-PENDING delete needs longer timeout'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7103ff0ec645
- daily-auto-filed
created_at: '2026-07-23T07:02:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): deleting a DWS-queued PENDING
  instance timed out ~4x over 10 min (#1586, 19:13-19:23Z) blocking the RunPod pivot;
  --async errored; teardown uses the generic timeout'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). Deleting a DWS-queued (PENDING) GCP instance during the #1586 capacity walk timed out repeatedly (~4 attempts over 19:13–19:23Z); an `--async` variant errored; only a longer-timeout bare delete eventually returned 0. The ~10 min of retries blocked the RunPod re-drive during that window.

## Goal

The GCP teardown path handles DWS-PENDING (queued) instances explicitly: a longer delete timeout (or a fire-and-verify async delete + bounded existence poll) for instances whose phase/status is queued-PENDING, so a capacity-walk pivot is not serialized behind a slow DWS cancel.

## Workflow gap

- **Bug observed:** 62e315d1 (#1586), 2026-07-22T19:13–19:23Z: repeated delete timeouts on the stale queue-waiting instance; "--async errored out"; ~10 min lost while the user-directed RunPod pivot waited.
- **Why it is a workflow gap:** DWS-queued instances are known-slow to cancel; the teardown invocation uses the generic timeout, and the queue-timeout/queue-vanish machinery (#783/#1116) that normally cancels queued instances did not own this user-directed manual pivot path.
- **Confidence:** medium (the exact teardown call site + its timeout need the planner's read).
- verified-at-filing: `grep -n 'instances delete' src/explore_persona_space/backends/gcp.py` → delete argv builder at gcp.py:3380 (`--quiet` only — no PENDING-aware timeout branch) plus teardown sites at 4174/4558/4819; no `--async`/pending-specific handling found (absence claim over the grep'd sites), 2026-07-23 UTC.

## Proposed change (refine in planning)

In the teardown/delete path: detect a PENDING/queued instance (the sidecar phase clock already tracks it, #1116) and use a longer timeout or async-delete + bounded verify; log the slow-cancel case loudly.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py` (teardown/delete path) + `tests/test_gcp_backend.py`.

## Constraints / invariants

- The EXIT-trap billing-bound teardown semantics unchanged; never skip a delete (credit leak). Recursion guard applies.

## Provenance

- fingerprint: 7103ff0ec645

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
