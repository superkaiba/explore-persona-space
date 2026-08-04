---
title: 'workflow-fix: Step 6d.2 tick parse must preserve advisory decision fields'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e370c38f0f0a
created_at: '2026-07-29T22:24:09Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from #1768 orchestrator: a status-only compact
  tick parse dropped gpu_idle_advisory/escalation_posted so the mandated handling
  never fired through ~15h idle 8xH100 (fp e370c38f0f0a)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a candidate raised on task #1768 (emitting agent: orchestrator, post-incident forensics on the 2026-07-29 ~16h download wedge).

## Goal

Amend the `/issue` Step 6d.2 poll-loop recipe so any compacted/filtered tick parse MUST print the full decision field set (status, next_interval, gpu_idle_advisory_posted, gpu_idle_escalation_posted, eta/compute-deviation fields, stall_reason) — a status-only tick parse is banned; provide the canonical parse one-liner.

## Workflow gap

- **Bug observed:** the #1768 orchestrator's Step 6d.2 bg-Bash tick used a compacted parse (`... | tail -1 | uv run python -c "print('NOW:', d['status'], ...)"` with a status-keyed exit code) that dropped `gpu_idle_advisory_posted` / `gpu_idle_escalation_posted`. The poller posted BOTH markers (`[gpu-idle-advisory]` 07:04:13Z, `[gpu-idle-escalation]` 07:32:54Z) but the booleans never entered orchestrator context, so the skill's mandated advisory/escalation handling never fired — ~15h of idle 8×H100 (~$375) was heartbeated as "ticks healthy".
- **Why it is a workflow gap:** SKILL.md Step 6d.2 mandates ACTING on the advisory fields ("If the JSON also has gpu_idle_advisory_posted == true, act per ...", L4535-4540) and details the handling (L4577-4619), but never constrains the PARSE: the recipe's own "Read the JSON line from stdout" guidance invites compaction, and nothing bans a status-only parse that structurally discards the very fields the same section branches on. The handling text is unreachable when the parse drops the trigger.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'gpu_idle_advisory_posted\|gpu_idle_escalation_posted' .claude/skills/issue/SKILL.md` → 5 hits, L4536/L4539/L4578/L4603/L4610 (2026-07-29). Presence hits READ + context judged: they implement the field SEMANTICS and the HANDLING, not a parse-preservation mandate — no line constrains what a compacted tick parse must print, so the proposed change is distinct from the landed text (clause (c) satisfied).

## Proposed change (candidate diff sketch — refine in planning)

```
+ .claude/skills/issue/SKILL.md, Step 6d.2 (near the "Read the JSON line" guidance):
+ **Tick-parse field-preservation (REQUIRED).** Any compacted/filtered parse of
+ the tick JSON MUST print, at minimum: status, next_interval,
+ gpu_idle_advisory_posted, gpu_idle_escalation_posted, any eta/compute-
+ deviation flags, and stall_reason (non-running ticks). A status-only parse is
+ BANNED — it structurally discards the decision fields the handling sections
+ below branch on (#1768, 2026-07-29: a status-only compact parse dropped a
+ posted [gpu-idle-escalation] for ~15h of idle 8xH100). Canonical one-liner:
+   ... | uv run python -c "import json,sys; d=json.loads(sys.stdin.readlines()[-1]);
+   print('NOW:', d['status'], d.get('current_phase'), 'adv=', d.get('gpu_idle_advisory_posted'),
+   'esc=', d.get('gpu_idle_escalation_posted'), 'stall=', d.get('stall_reason')); ..."
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for sibling compact-parse recipes before editing (`grep -rln "tail -1.*json.loads\|d\['status'\]" .claude/ CLAUDE.md scripts/`) and update every recipe hit; list them in the plan. Also check `.claude/skills/issue-tick/SKILL.md` for the same field-dropping shape (its § Digest-only contract governs a DIFFERENT surface — task-state reads — but a poll-tick example there would inherit this mandate).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Does NOT change poller output or any marker schema — SKILL.md prose + canonical one-liner only (non-architectural).
- `scripts/workflow_lint.py --check-asks` passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: e370c38f0f0a

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: the #1768 orchestrator compact tick parse printed status/phase only, dropping gpu_idle_advisory_posted/gpu_idle_escalation_posted, so the mandated handling never fired through ~15h of idle 8xH100
why_workflow_gap: Step 6d.2 mandates acting on the advisory fields but never constrains the parse — the "read the JSON line" guidance invites compaction and nothing bans a status-only parse that discards the trigger fields
proposed_change: mandate that any compacted Step 6d.2 tick parse print the full decision field set (status, next_interval, gpu_idle advisory/escalation flags, stall_reason) — status-only parses banned; add a canonical parse one-liner
diff_sketch: |
  + Step 6d.2 "Tick-parse field-preservation (REQUIRED)" paragraph + canonical
  + parse one-liner printing status/next_interval/adv/esc/stall_reason
confidence: high
related_task: #1768
<!-- /workflow-fix-candidate -->
