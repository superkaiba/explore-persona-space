---
name: wf-fix-step2-floor-attestation
description: On wf-fix tasks (kind infra + workflow-fix tag / wf-fix title prefix), the Step 6 Step-2-floor check reads main-side events.jsonl for epm:plan-verify — Codex cannot; attest presence/absence at compose time or the twin emits a spurious step2-floor-skipped FAIL
metadata:
  type: feedback
---

The code-reviewer.md Step 6 "Step-2 floor check" FAILs a wf-fix task
(`kind: infra` + `wf-fix`/`workflow-fix` tag or `workflow-fix:`/`daily-fix:`
title prefix) with tag `step2-floor-skipped` when events.jsonl carries no
`epm:plan-verify` marker. That marker lives on MAIN's events.jsonl; the
worktree copy is frozen at branch-cut and Codex cannot read main-side state
— so an uninstructed twin either probes the stale worktree copy (false
absence → spurious FAIL) or skips the check silently.

**Why:** hit on #2326 r1 (2026-08-16): the plan-verify marker (14:48Z)
predated the branch cut so it happened to be in the frozen copy, but the
general case is not guaranteed — the same frozen-events class as the
implementation marker (#489).

**How to apply:** every compose for a wf-fix task probes canonical main
state (`task.py view <N> --json` → scan events for `epm:plan-verify`) and
writes a compose-time attestation line into the prompt: "Step-2 floor:
PASSED — epm:plan-verify present (ts ...); do NOT raise
step2-floor-skipped" (or, when genuinely absent, "Step-2 floor: NOT
satisfied — no epm:plan-verify on main; the step2-floor-skipped FAIL is
warranted"). Same pattern family as the Step 4.6 GATE-SCOPE THRESHOLD line
and the Step 0.8 empty-ledger attestation. Related:
[[bypath-brief-frozen-events-resolution]].

Version-gap nuance (#2205 r1, 2026-08-19): the plan-verify marker often ran
on an EARLIER plan version than the finally-approved one (v2 verified, v3
approved — a plan amendment landed after the pre-pass). The floor check keys
on marker PRESENCE only, so attest PASSED, include the verified-plan version
in the attestation, and route the version gap explicitly to at-most-CONCERNS
("if you judge that gap material, it is at most a CONCERNS note, never a
FAIL") so the twin neither ignores nor FAILs on it.
