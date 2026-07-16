---
title: 'daily-fix: guard ssh-string git false positives'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ec2e2cc54d0f
- daily-auto-filed
created_at: '2026-07-16T07:22:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): The hook blocked a pod-side
  git checkout FETCH_HEAD quoted inside a local Bash ssh argument string (documented
  residual xiv: compound ssh payloads mis-split); 1 wasted turn'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Exempt commands whose git invocation is entirely inside an `ssh <host> '...'` argument string from the repo-root branch guard (conservative parse; fail-closed on ambiguity) — closing the documented residual where such commands still false-block.

## Workflow gap

- **Bug observed:** the hook blocked a POD-side `git checkout FETCH_HEAD` because it was quoted inside a local `Bash("ssh pod-77965 '...'")` string — the checkout would never touch the shared root. Cost: 1 wasted turn (the SSH-MCP redirect worked) (272c80a1, #779, 16:55:51Z).
- **Why it is a workflow gap:** the guard already carries an ssh/grep-family clause waiver, but its own comments document a residual where compound statements inside the ssh argument string mis-split and the tail clause loses the ssh command word — the 07-15 incident shows that residual still false-blocks legitimate pod-side git.
- **Severity:** low
- verified-at-filing: `grep -n 'ssh' scripts/guard_repo_root_branch.sh` → waiver PRESENT (L49-59: Bash("ssh ...") clause waiver incl. the `ssh pod-779 'git checkout HEAD -- <file>'` example; L98: command word `ssh` waived) AND the residual is DOCUMENTED (L237-241, item (xiv) #1098: "a compound statement (`ssh pod 'cd /workspace/x && git reset --hard'` — mis-split; the tail clause lost the ssh command word)") — the proposed change targets that documented residual, not a missing waiver (2026-07-16 UTC).

## Proposed change (refine in planning)

Harden `scripts/guard_repo_root_branch.sh`'s ssh-clause waiver so a git invocation ENTIRELY contained in an `ssh <host> '...'` single-quoted argument string is exempt even when the quoted payload is a compound statement (`&&`/`;` chains) — i.e. fix the clause-splitting so splitting never descends into a quoted ssh argument (parse quotes before splitting on connectors), conservatively: any ambiguity (unbalanced quotes, nested quoting, the payload naming the shared repo-root path) stays fail-closed/blocked. Preserve the deliberately-never-waived case (an ssh clause naming the shared-repo path, L59) and extend the guard's test fixtures with the #779 `ssh pod-77965 'git checkout FETCH_HEAD ...'` shape.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh` (waiver L49-59/L98; residual doc L237-241)
- Secondary: the guard's test fixtures (add the false-positive shape as a regression)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Fail-closed on ambiguity — the guard must never newly PERMIT a genuine repo-root branch switch/destructive reset (the protected class is unchanged).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: ec2e2cc54d0f

- workflow_fix_target: scripts/guard_repo_root_branch.sh

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 272c80a1 (#779) 16:55:51Z (batch 00 P8).
