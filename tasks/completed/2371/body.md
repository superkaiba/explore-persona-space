---
title: 'guard_root_code_commit.sh: disarm text-detectable prefixed cwd-movers outside
  the NAME=value set (+= append-assignment, time/wrapper prefixes) + complete residual-note'
kind: infra
tags:
- wf-fix
- trigger-dense
created_at: '2026-08-18T12:06:45Z'
has_clean_result: false
parent_id: 2357
origin_prompt: '#2357 r5 review Minor: FOO+=bar . env.sh and time . env.sh ride the
  armed chain undetected (block-direction residual)'
workflow: v1
---
---
kind: infra
---
# guard_root_code_commit.sh — extend the cwd-mover disarm lead grammar to text-detectable prefixed movers outside the NAME=value set

## Goal
`.claude/hooks/guard_root_code_commit.sh`'s per-record cwd-mover disarm (`CWD_MOVER_LEAD_ERE`, added #2357) recognizes a leading `NAME=value` assignment-prefix group before the mover family, but two OTHER text-detectable prefixed cwd-mover forms ride the armed canonical `&&` chain undetected and wrongly permit (scope) a root commit whose relative pathspec resolves off-root:

1. **Append-assignment prefix** — `FOO+=bar . ./script` (the group matches `[A-Za-z_][A-Za-z0-9_]*=`, not `+=`).
2. **Wrapper-keyword prefix** — `time . ./script` (and confirm/pin whether `builtin`/`command`/`env` before dot-source are reachable sourcing paths on this shell; `time`/`builtin` are).

Both are **block-direction** gaps (fail-closed once caught) and were confirmed by execution during #2357 round 5: on the r5 guard, `cd <root> && FOO+=bar . env.sh && git commit -- p` and `cd <root> && time . env.sh && git commit -- p` from a subdir cwd permit (rc=0) where origin/main blocks (rc=2).

## Fix direction
Extend `CWD_MOVER_LEAD_ERE`'s optional prefix group to also accept `+=` append-assignments and the reachable no-op wrapper keywords for the dot-source/mover family — DISARM-side only. Complete the header known-limitations residual-note example list so its enumerated recognized/unrecognized sets are truthful (AC6/K8 text-truth): name the newly-recognized prefixed forms and the residual text-unprovable classes (predefined shell function/alias indirection). Add pytest pins (extend the c30 family) asserting block on fixed code and permit on the pre-fix blob. HARD CONSTRAINTS: never add the lone dot to the shared whole-record screen; never touch `cd_nonroot`; keep every incident ALLOW chain green (c17b, canonical, c29 re-arm); fail-closed always wins.

## Provenance
Surfaced by #2357 round-5 code review (Claude `code-reviewer` non-blocking Minor) + the round-4 binding reconcile executed differential. Parent #2357 landed on main as PR #2001 (merge `15f91ee528`). These residuals are pre-existing (also permit on r4 — the #2357 fix was strictly block-direction and opened no new permit), documented-as-accepted under the guard's best-effort-heuristic contract, and split out here rather than blocking #2357.
