---
title: 'daily-held: shared-root inline-round commit hygiene'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-23T07:05:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 3): a concurrent worktree revert
  wiped an in-progress inline edit round mid-commit (near-miss loss) and the nightly
  daily commit swept a sibling''s files; inline rounds on the shared root race the
  fleet'
workflow: v1
---
## Overview / Motivation

Filed by /daily 2026-07-22 as a TRACKED needs-human item (route 3 — genuinely-ambiguous-intent carve-out: two reasonable working styles diverge). Heavy fleet activity + interactive inline rounds all committing to the shared repo root produced a near-miss edit loss and a foreign-file sweep in ~24 h; whether inline rounds should change their working style is a preference call.

## The incidents (2026-07-22/23)

1. **Near-miss edit loss:** in the #1092 writeup session (5e8b4c66, ~03:59:59Z) a concurrent session's worktree revert wiped an in-progress round of edits mid-commit — only `prefixend_monitoring_averaged_only.meta.json` landed; the figure/script edits had to be re-applied.
2. **Foreign-file sweep:** the 07-21 nightly daily commit `7dbde267f1` swept the #1092 session's uncommitted `scripts/issue1092_fair_deepdive_figs.py` + 3 figure files onto main (separately filed as a /daily skill hardening).
3. **Residual risk standing:** at mining time `scripts/issue1092_fair_deepdive_figs.py` + `figures/issue_1092/prefixend_monitoring_averaged_only.*` sat modified-uncommitted at the shared root with the owning session's liveness unknown — the same-turn completion contract (inline rounds commit artifacts in the result turn) unmet.

## The decision needed (why route 3)

Two candidate conventions, both changes to how you like inline rounds to work:
- (a) extend the CLAUDE.md "post a claim marker before batch-fixing parked task bodies" convention to ANY multi-file inline round touching shared-root `figures/`/`scripts/` during heavy fleet activity; or
- (b) default interactive inline rounds into a scratch/sparse worktree (like pipeline sessions) and land via server-side merge, keeping the shared root for task-state only.

(b) is safer but adds friction to the fast interactive loop you use daily; (a) is lighter but advisory. Your call.

## Suggested action

Pick (a), (b), or neither; a one-line reply routes it as an ordinary workflow-fix filing.

## Provenance

- sha-verify (filing-time, #1467): `5e8b4c66` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
