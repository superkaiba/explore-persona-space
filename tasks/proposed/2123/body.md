---
title: 'daily-fix: four verifier gaps (lint + verify_plan)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a1713aa1d27e
- daily-auto-filed
created_at: '2026-08-06T07:05:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): persisted plan v11.md mutated
  in place; unpinned agent files inherit session model; duplicate GPU-hour token silently
  first-picked; missing planned_wall_h silently disarms phase-ETA tripwire'
workflow: v1
---
# daily-fix: four verifier gaps (plan-version immutability lint; agent model-pin lint; verify_plan duplicate GPU token; verify_plan planned_wall_h WARN)

## Workflow gap

Four independent 2026-08-05/06 incidents each trace to a check no verifier runs today:

1. **Persisted plan versions are mutable.** #2061's fact-correction round mutated
   `tasks/planning/2061/plans/v11.md` in place (+41/−27 vs its persist commit) instead of
   drafting v12 — self-described "provenance corruption I've been flagging all session";
   recovery tripped a repo-root guard block before v12 was persisted properly (session
   5c878aa4, 02:00–02:40Z 08-06). Nothing lints that a persisted `plans/v*.md` stays
   byte-stable after its persist commit.
2. **Agent files without a `model:` pin silently inherit the session model.** On 08-05
   Thomas found subagents running Opus after `/model` ("I thought subagents were supposed
   to always be fable now"); 4 agent files had no pin. Fixed in-session
   (env.CLAUDE_CODE_SUBAGENT_MODEL + 4 pins, commit 8c1f0235ba) — but nothing prevents the
   next new agent file from shipping unpinned.
3. **verify_plan silently picks the FIRST bold GPU-hour token.** #2061's plan carried the
   canonical `**Estimated GPU-hours (total): N**` token twice with conflicting values (70
   vs 80 — a preserved v5 record kept the old token); the gate parsed the first and
   PASSed. Self-caught pre-approval (session 5c878aa4, 22:17Z).
4. **A plan §9 without a parseable `planned_wall_h` silently disarms the poller's
   phase-ETA tripwire.** #2091's every poll chain logged "no parseable §9 planned_wall_h
   … phase-ETA tripwire disabled (fail-safe)" — and that run then had 4 rung-jobs die,
   detected only by status polling (session b765cdcd, 3 probed poll-tick firings).

verified-at-filing (2026-08-06T07:2xZ): `grep -n 'plans/v' scripts/workflow_lint.py | head -3`
→ no immutability check; `grep -rn 'model:' scripts/workflow_lint.py | grep -ci agent` →
0 (no agent model-pin check); `grep -n 'Estimated GPU-hours' scripts/verify_plan.py | head -3`
→ parser present, no duplicate-token FAIL; `grep -cn 'planned_wall_h' scripts/verify_plan.py`
→ 0. Incidents are the miners' probed marker/tool_result readbacks.

## Proposed change

- `scripts/workflow_lint.py`: (a) check that tracked `tasks/**/plans/v*.md` files are
  unmodified after their persist commit (amendments require a new version file); (b) check
  every `.claude/agents/*.md` carries a `model:` frontmatter pin (allowlist deliberate
  inherit cases explicitly).
- `scripts/verify_plan.py`: (c) FAIL when the bold GPU-hours token appears more than once
  with differing N; (d) WARN when a `kind: experiment` plan's §9 lacks a
  `planned_wall_h`-parseable token (so the phase-ETA tripwire never silently disarms).

## Provenance

- fingerprint: a1713aa1d27e

- workflow_fix_target: scripts/workflow_lint.py, scripts/verify_plan.py
- origin: /daily 2026-08-05 problem sweep — miner 5 P9/P4/P11, miner 6 P24.
