---
title: 'daily-fix: route-3 needs-human tag dropped at filing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4a38962ef4b8
- daily-auto-filed
created_at: '2026-07-18T06:46:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): route-3 daily-held filings
  end up WITHOUT the needs-human tag on the created task (#1140, #1472 and all open
  daily-held tasks carry tags: [daily-held] only), so the PM Needs-you enumeration
  and watcher auto-dispatch exclusion never key on them — root cause behind the 9-day
  undecided #1140.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-17 parked-candidate sweep (Step C) from a prose-followup candidate parked on task #1483 (emitting agent: Alternatives critic, plan #1483 Phase 2; parked under the recursion guard).

## Goal

Diagnose and fix why route-3 `daily-held` filings end up WITHOUT the `needs-human` tag on the created task — the tag the PM "Needs you" enumeration and the watcher auto-dispatch exclusion both key on — and reconcile the existing open daily-held tasks.

## Workflow gap

- **Bug observed:** `scripts/daily_drive_filings.py` (`_filer_cmd`, ~L609) passes `--tag needs-human` and `scripts/file_infra_task.py` (~L194) forwards each `--tag`, yet BOTH real route-3 incident bodies (#1140, #1472) carry `tags: [daily-held]` ONLY. `.claude/skills/daily/SKILL.md` keys the PM "Needs you" enumeration AND the watcher auto-dispatch exclusion on `needs-human`, so held decisions may never surface in the PM block — the root cause behind the 9-day-undecided #1140.
- **Why it is a workflow gap:** route 3 exists precisely so held judgment calls are re-surfaced until Thomas acts; a dropped surfacing tag silently reverts route 3 to the pre-#706 dead-end-note behavior.
- **Confidence (emitter):** medium (mechanism unconfirmed — the tag is passed and forwarded, yet absent on the artifact; the drop site is somewhere in the `task.py new` tag-application path or a later overwrite).
- verified-at-filing: `task.py view 1140/1472 --json` at compose time (2026-07-18 UTC) → both show `tags: ['daily-held']` with no `needs-human`; a registry scan of ALL open daily-held tasks (#1038, #1067, #1141, #1331, #1472) shows the same single-tag shape. `grep -n "needs-human" scripts/daily_drive_filings.py` → present (L99 boilerplate-token set; `_filer_cmd` emits the flag); `grep -n '"--tag"' scripts/file_infra_task.py` → forwarding loop present at ~L194. The pass/forward sites exist; the artifact lacks the tag — the drop is downstream (bind is on the artifact evidence, not a code-line absence).

## Proposed change (candidate diff sketch — refine in planning)

Trace one filing end-to-end (`file_infra_task.py` → `task.py new --tag ...` → frontmatter write → REGISTRY snapshot); fix the drop site; add a pinning test that a `--no-dispatch --tag daily-held --tag needs-human` filing round-trips both tags through `view --json`; and add-tag `needs-human` to the existing open daily-held tasks (#1038, #1067, #1141, #1331, #1472) so the PM block picks them up.

## Scope / surfaces

- Primary target: `scripts/file_infra_task.py`, `scripts/task.py` (tag application), `.claude/skills/daily/SKILL.md`
- Grep before editing: `grep -rn "needs-human" scripts/ .claude/skills/daily/ .claude/agents/research-pm.md` and list every consumer of the tag in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; pinning test green.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 4a38962ef4b8

- workflow_fix_target: scripts/file_infra_task.py

source candidate (verbatim, prose park on #1483, 2026-07-18T03:52:00Z): "the needs-human tag drop / PM-surfacing anomaly — _filer_cmd (scripts/daily_drive_filings.py L609) passes --tag needs-human and file_infra_task.py L194 forwards each --tag, yet BOTH real route-3 incident bodies (#1140, #1472) carry tags: [daily-held] ONLY; .claude/skills/daily/SKILL.md L471 keys the PM 'Needs you' enumeration AND the watcher auto-dispatch exclusion on needs-human, so held decisions may never surface to the PM block — the root cause behind the 9-day undecided #1140. target_file: scripts/file_infra_task.py, scripts/task.py (tag application), .claude/skills/daily/SKILL.md."
