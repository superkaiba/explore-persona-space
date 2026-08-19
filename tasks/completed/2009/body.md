---
title: 'daily-fix: python-pipe guard exempts ssh-quoted remote argv'
kind: infra
tags:
- wf-fix
- wf-fix-fp:af9dd26d8299
- daily-auto-filed
- trigger-dense
created_at: '2026-08-02T07:14:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): The bare-python-pipe PreToolUse
  guard (INLINE in .claude/settings.json line ~111, not a hooks/ file — target corrected
  per call-hop duty) blocked a remote ssh-quoted python3 -c pipe; the no-python-on-PATH
  premise is local-only and the regex matches anywhere in the argv with no ssh exemption.'
workflow: v1
---
# daily-fix: python-pipe guard exempts ssh-quoted remote argv

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C34 (miner-1 P11; session 55419495, #1739, 12:34Z).

## Goal
Exempt pipes occurring inside `ssh` / `gcloud compute ssh` quoted REMOTE command strings from the bare-`python`-pipe PreToolUse guard — the guard's premise (no `python` on the VM PATH) does not apply to remote argv, where `python3` exists.

## Workflow gap
- **Bug observed:** A remote command of the form `ssh … "… | python3 -c …"` was blocked: "BLOCKED: bare `| python -c/-m` pipe. This VM has no `python` on PATH…". Assistant: "The guard matched the `python3 -c` text in my remote command (it can't tell local from remote)"; worked around with plain grep. 1 hook-error firing (of 3 total in window). Low priority — a workaround exists — but every remote-pipe composition pays a wasted turn.
- **Why it is a workflow gap:** The guard is a VM-local-PATH correctness check applied to the whole Bash argv, including text that executes on a REMOTE host; the false positive is structural, not a one-off.
- **Confidence:** medium
- verified-at-filing: call-hop target tracing — the consolidated entry pointed at `.claude/hooks/`, but `grep -rln 'python' .claude/hooks/` shows no python-pipe guard among the hook FILES; the real construction site is the INLINE PreToolUse hook in `.claude/settings.json` **line 111**: `cmd=$(jq -r '.tool_input.command // empty'); if echo "$cmd" | grep -qE '\|[[:space:]]*python3?(\.[0-9]+)?[[:space:]]+(-[^[:space:]]+[[:space:]]+)*-[cm]([^A-Za-z0-9_]|$)'; then echo 'BLOCKED: bare `| python -c/-m` pipe…'` — the regex matches anywhere in the command string and carries NO ssh/remote exemption (presence hit, context read; target corrected to `.claude/settings.json`) (2026-08-02 UTC).

## Proposed change (refine in planning)
In the `.claude/settings.json` line-111 PreToolUse command, short-circuit before the pipe grep when the pipe text lies inside an ssh-family remote string, e.g.:

```
+ # skip when the ONLY python-pipe occurrences are inside an ssh/gcloud-compute-ssh
+ # remote command string (quoted argv after `ssh <host>` / `--command=`):
+ if echo "$cmd" | grep -qE '(^|[;&|[:space:]])(ssh|gcloud[[:space:]]+compute[[:space:]]+ssh)[[:space:]]' ; then
+   <strip the quoted remote-string spans, re-run the pipe grep on the residue only>
+ fi
```

Design freedom for the planner: robust quote-span stripping in POSIX-sh/jq is fiddly — an acceptable cheaper form is "if the command starts with `ssh `/`gcloud compute ssh` (optionally after env-var prefixes), skip the check" (a local pipe INTO ssh, e.g. `… | python -c | ssh …`, stays blocked since the python-pipe precedes the ssh token). Consider extracting the growing inline command to a `.claude/hooks/guard_python_pipe.sh` file for testability (settings.json then invokes the file), matching the sibling guards' shape.

## Scope / surfaces
- Primary target: `.claude/settings.json` (inline PreToolUse python-pipe guard, line ~111; optionally extracted to a new `.claude/hooks/guard_python_pipe.sh`)

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; `jq . .claude/settings.json` stays valid; bash -n on any extracted hook file.
- The local-VM block behavior for genuinely local `| python -c/-m` pipes is UNCHANGED.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: af9dd26d8299
- workflow_fix_target: .claude/settings.json
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C34.
