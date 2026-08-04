---
title: 'workflow-fix: forensics — concurrent root working-tree rever'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ddff79533592
- daily-auto-filed
created_at: '2026-08-04T06:53:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 2): A concurrent repo-root
  working-tree revert scoped to eval_results/ silently reverted a completed 48-minute
  ridge run''s result JSON (#1482, caught only because the reverted values were byte-identical
  to the pre-run ones). The guard is supposed to refuse exactly that fail-closed,
  and is verified armed and firing today, so something evaded it; the perpetrator
  is not identified in the observing transc'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-03 (route 2: behavior/logic change → independent review) from the nightly problem sweep (miner1, session 201e2896, task #1482). This one caused actual DATA LOSS, so it is filed even though the root cause is not yet localized.

## Goal

Establish how a concurrent repo-root working-tree revert scoped to `eval_results/` reached the shared tree despite the guard that is supposed to refuse exactly that, and close the path.

## Workflow gap

- **Bug observed:** a completed 48-minute ridge run's result JSON was silently reverted out from under its owner. The #1482 teammate reported: "The first ridge run (48 min) completed correctly, then **a concurrent repo-root working-tree revert scoped to `eval_results/` reverted its JSON** while leaving its figures as modified. I caught it because the 'new' values were byte-identical to the old ones" (session 201e2896 row 3442, 2026-08-03T20:40:03Z). The session then ran defensive commits and warned three other in-flight agents (rows 3489/3524/3864).
- **Why it is a workflow gap:** `scripts/guard_repo_root_branch.sh` is supposed to make this impossible — its arm covers the bare form, explicit-path forms, `--source` forms and the `-S` short form, blocking fail-closed with the stated rationale that "on the SHARED root any working-tree revert can discard a CONCURRENT session's uncommitted edits (incidents 2026-06-01, #815, #841)". Something performed the operation anyway. Until the path is known, every concurrent session's uncommitted `eval_results/` output is exposed to the same silent loss — and this failure mode is near-undetectable by design (the owner noticed only because the reverted values were byte-identical to the pre-run ones).
- **Confidence (emitter):** high that the loss occurred (the owner's own byte-identity diagnosis); **the evasion path is UNKNOWN** and is the substance of this task.
- verified-at-filing: `grep -n 'restore' scripts/guard_repo_root_branch.sh` → the arm at **L1698** with the documented fail-closed semantics quoted above (read at compose time, 2026-08-04); the guard is present and its regex covers the explicit-path form. Live re-confirmation that the guard is armed and effective for HOOKED callers: a /daily verification command carrying the two-token verb in a quoted test string was itself refused by this guard at 2026-08-04 (so the hook is firing normally today).
- unverified hypothesis — verify at plan time: the candidate evasion paths, none confirmed. (i) a subagent or tool invocation that does not run PreToolUse hooks; (ii) a non-Claude process (a cron, a script, a `git` call from inside a python subprocess — the guard scans Bash argv, not library-level git calls); (iii) a guard gap in a compose shape the regex does not reach; (iv) not a revert at all but another mechanism producing the same observable (e.g. a checkout/reset inside a helper). The perpetrator session is NOT identified in the observing transcript, so the planner's first step is forensics — `.claude/cache/guard-deny-events.jsonl` records DENIES, and the absence of a matching deny row would itself discriminate (i)/(ii)/(iii) from a blocked-then-retried shape.

## Proposed change (candidate sketch — refine in planning)

```
1. forensics first: correlate the ~20:40Z loss window against
   .claude/cache/guard-deny-events.jsonl, the reflog, and the concurrent
   sessions' transcripts to identify the actual caller
2. then close the identified path — a library-level git call site, a
   hook-exempt invocation route, or a regex gap
3. regardless of cause: consider a cheap tripwire for the undetectable class
   (the loss was found only by byte-identity luck)
```

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh` (and whatever call site the forensics identifies).
- Read-only forensics inputs: `.claude/cache/guard-deny-events.jsonl`, git reflog, the 2026-08-03 session transcripts.

## Constraints / invariants

- Do NOT weaken any existing arm of the guard while investigating.
- A "cannot reproduce" outcome is an acceptable result IF the forensics are recorded — but the task should then leave behind the tripwire rather than closing silent.
- Workflow-surface only.

## Provenance

- sha-verify (filing-time, #1467): `201e2896` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: ddff79533592

- workflow_fix_target: scripts/guard_repo_root_branch.sh
