---
title: 'workflow-fix: /daily route-3 filings dedup against open daily-held tasks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f8fc681f6baf
created_at: '2026-07-17T22:24:31Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1472: /daily route-3 has no dedup
  against open daily-held tasks (SKILL.md L441 ''route 3 has no such backstop''; daily_drive_filings.py
  L16 ''skips fp-dedup (#1228)''); live incident #1140 (2026-07-08) vs #1472 (2026-07-17)
  — the same Codex-outage decision filed twice 9 days apart'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1472 (emitting agent: /issue orchestrator).

## Goal

Add a same-subject dedup scan to /daily route-3 filings: before filing a
needs-human task, scan open daily-held tasks for informative-token overlap and
log "already tracked in #M" instead of filing a duplicate.

## Workflow gap

- **Bug observed:** /daily route-3 filed #1472 (2026-07-17) duplicating open
  #1140 (2026-07-08) — the same Codex-outage spend decision tracked twice;
  route 3 has no dedup backstop by its own documentation.
- **Why it is a workflow gap:** route-2 filings carry fp-dedup
  (`wf-fix-fp:<fp>` tag + `is_open_workflow_fix_task`), but route-3 filings
  deliberately skip it, and a LONG-LIVED held condition (a 3-week vendor
  outage) recurs in the nightly problem sweep and re-files the same decision
  every time it resurfaces. The PM `Needs you` block then carries N copies of
  one decision.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rliE "codex.*(quota|outage|single.claude)" tasks/proposed/*/body.md` → 2 hits (`tasks/proposed/1140/body.md`, `tasks/proposed/1472/body.md` — the duplicate pair) (2026-07-17). Per-target: `grep -n "route 3 has no such backstop" .claude/skills/daily/SKILL.md` → 1 hit (L441; context READ — it documents the ABSENCE of a route-3 dedup, it does not implement one, so the gap is confirmed un-landed); `grep -n -iE "dedup" scripts/daily_drive_filings.py` → route-3 arm at L16 reads "skips fp-dedup (#1228)" (context READ — the skip is deliberate for FINGERPRINT dedup; no same-subject scan exists either).

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/daily_drive_filings.py (route-3 arm) + .claude/skills/daily/SKILL.md (route-3 prose):
+ Before filing a route-3 item, enumerate OPEN tasks tagged `daily-held`
+ (proposed / on_hold / blocked) and compare informative title+body tokens
+ (the #1446 recently-closed-sibling advisory's token-overlap approach,
+ applied to OPEN daily-held tasks).
+ On a >=2-informative-token overlap match: do NOT file; record outcome
+ "already-tracked" with the existing task id in filed.jsonl + the /daily
+ report, and (optionally) post a one-line freshness note on #M.
+ Fail-open: overlap scan errors -> file as today (never lose a held item).
```

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py`, `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rn "route 3" .claude/skills/daily/SKILL.md scripts/daily_drive_filings.py`)
  and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- Dedup must be FAIL-OPEN (a scan failure files as today — a held item is
  never silently dropped; nothing-silently-dropped is /daily's core contract).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard,
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: scripts/daily_drive_filings.py, .claude/skills/daily/SKILL.md
- fingerprint: f8fc681f6baf

Surfaced prose (verbatim, from the #1472 session): "/daily route-3 has no
dedup against open daily-held tasks — SKILL.md L441 documents 'route 3 has no
such backstop' and daily_drive_filings.py L16 'skips fp-dedup (#1228)'; the
live incident is #1140 (2026-07-08) vs #1472 (2026-07-17), the same
Codex-outage decision filed twice 9 days apart."
