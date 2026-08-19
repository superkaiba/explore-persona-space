---
title: 'daily-fix: finally-block raise must never mask in-flight exc'
kind: infra
tags:
- wf-fix
- wf-fix-fp:098a3f039f06
- daily-auto-filed
created_at: '2026-08-02T07:15:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): #1947 marker cells failed
  3 rounds as ''GPU-drain timeout'' because a close-gate raise in a finally block
  REPLACED the in-flight EADDRINUSE vLLM port-race exception — 2 relaunch rounds fixed
  the wrong symptom; the real fix landed 23:21Z and round 3 finished 4/4.'
workflow: v1
---
# daily-fix: finally-block raise must never mask in-flight exception

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C14 (miner 4, P5; session 8fc069db, issue #1947).

## Goal
Add a gotchas.md teardown entry + a code-reviewer rubric line: a `finally` block that can raise (close gates, drain waits, teardown asserts) must chain (`raise ... from exc`) or log-and-suppress when an exception is already in flight — never REPLACE the in-flight exception, which erases the real error.

## Workflow gap
- **Bug observed:** All 4 #1947 marker cells failed 3 rounds (21:55Z–23:56Z) as "GPU-drain timeout": round 1 raised the drain timeout to 900s — failed again; round 2 ran "serially" but still booted engines wide — failed again. Only then did digging reveal the inner error was an EADDRINUSE vLLM rendezvous port race that the close-gate raise in `finally` had been REPLACING; the real fix (reorder teardown to free the base model before the drain wait + an unmasking guard so inner errors surface) landed 23:21Z and round 3 finished 4/4. Two wasted relaunch rounds spent fixing the masking symptom. `unverified hypothesis — verify at plan time:` mechanism read from the orchestrator's own root-cause note, script source not opened at mining time (miner-inferred, not probed).
- **Why it is a workflow gap:** Exception-masking-in-`finally` is a recurring silent-failure SHAPE (a teardown raise erases the diagnosis every debugging round depends on), and neither the gotchas teardown entries nor the code-reviewer silent-failure checklist names it.
- **Confidence:** medium
- verified-at-filing: `grep -cw 'finally' .claude/rules/gotchas.md` → 0 (no `finally` entry of any kind — absence confirmed); `grep -n -i 'finally\|exception' .claude/agents/code-reviewer.md` → 1 relevant hit (line 1468, the `except Exception` swallow checklist item — swallow ≠ replace; no finally-masking line); `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md .claude/agents/code-reviewer.md` → 8+1 commits (gotchas: teardown-adjacent entries 7bd42ad23f etc.), none adding a finally-masking clause (2026-08-02).

## Proposed change (refine in planning)
- `.claude/rules/gotchas.md` (teardown/vLLM section): new entry — "A `finally`-block raise REPLACES the in-flight exception: a close-gate/drain-wait assert that fires during teardown erases the real error, so every retry round debugs the teardown symptom (#1947: 2 relaunch rounds fixed 'GPU-drain timeout' while the true EADDRINUSE port race stayed masked). Recipe: inside `finally`, detect an in-flight exception (`sys.exc_info()[0] is not None`); if one is live, log the teardown failure and let the original propagate (or `raise ... from exc` to chain) — never bare-raise. Diagnostic signature: every failure in a fan-out reports the SAME teardown-stage error across rounds while fixes to that stage change nothing."
- `.claude/agents/code-reviewer.md` (silent-failure checklist, near the `except Exception` swallow item): add "a `finally` (or teardown/close-gate path) that can raise must chain or suppress-and-log when an exception is in flight — a replacing raise is a silent-failure defect."

## Scope / surfaces
- Primary target: `.claude/rules/gotchas.md, .claude/agents/code-reviewer.md`
- Mirror the reviewer line into `codex-code-reviewer`'s composed rubric only if that rubric inlines the checklist section touched (grep at implement time). The #1947 experiment-script fix itself already landed with the task — no experiment code in scope.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 098a3f039f06
- workflow_fix_target: .claude/rules/gotchas.md, .claude/agents/code-reviewer.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C14.
