---
title: 'workflow-fix: kill-before-relaunch + timeout for smoke retry loops'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6dfce7d5d7c9
created_at: '2026-07-02T07:20:20Z'
has_clean_result: false
origin_prompt: 'PM-chat VM triage 2026-07-02: 3 identical run_823 smokes ran concurrently
  (retry-without-kill; orphaned python children), same output paths, 64 threads each.
  Fix: kill-before-relaunch + timeout in crash-fix-rounds + implementer agents.'
---
## Overview / Motivation

Auto-filed from the 2026-07-02 shared-VM overload incident. During #823's code-review retry loop, THREE identical `run_823.py --phase 4 --smoke --skip-upload` instances ran concurrently (launched 23:48, 23:51, 00:02) because each retry re-ran the smoke WITHOUT killing the previous instance — timed-out/abandoned Bash calls kill the shell, not the python child. All three wrote the SAME output paths (smoke-verdict corruption risk) and held 64 threads each (~1/3 of the load-226 overload).

## Goal

Make smoke/run RETRY loops kill-before-relaunch: before re-running a smoke or launch command, kill any live prior instance of the same invocation and/or verify no live process is writing the same outputs; bound smoke runs with `timeout`.

## Workflow gap

- **Bug observed:** review-retry loops (implementer crash-fix rounds, Codex companion smoke re-runs) relaunch the same command while the prior instance still runs, duplicating load and corrupting shared output paths.
- **Why it is a workflow gap:** no rule/instruction on any retry surface requires killing the prior instance; Bash timeouts orphan python children by design.
- **Confidence (emitter):** high

## Proposed change (refine in planning)

- `.claude/rules/crash-fix-rounds.md`: add a kill-before-relaunch requirement to the crash-fix/retry protocol (pgrep the exact prior invocation; kill + confirm dead BEFORE re-running; wrap smokes in `timeout`).
- `.claude/agents/experiment-implementer.md` + `.claude/agents/implementer.md`: same instruction at the smoke-run step.
- Consider the Codex twin prompt-composers (the companion runtime executed these retries) — add the instruction to the composed prompt where smokes are re-run.

## Scope / surfaces

- Primary: `.claude/rules/crash-fix-rounds.md`, `.claude/agents/experiment-implementer.md`, `.claude/agents/implementer.md`, codex wrapper prompts.

## Constraints / invariants

- workflow_lint --check-asks passes; instruction must be exact-invocation-scoped (never a broad pkill that can hit concurrent sessions' work — this incident's own cleanup nearly did).

## Provenance

- workflow_fix_target: .claude/rules/crash-fix-rounds.md
- origin: PM-chat VM triage 2026-07-02 (#823 3x duplicate smoke; /tmp/vm_load_diagnosis.md)
