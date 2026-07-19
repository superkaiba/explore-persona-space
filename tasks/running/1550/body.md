---
title: 'workflow-fix: verify_plan WARN when plan header version label mismatches persisted
  filename'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6572ed6a870c
created_at: '2026-07-19T09:17:25Z'
has_clean_result: false
origin_prompt: 'Prose follow-ups from #1482 Phase-2 panel (Methodology critic + consistency-checker):
  stale ''# Plan v4'' header rode persists v5-v7; add a verify_plan.py header-version-vs-filename
  WARN check.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from prose follow-ups raised on task #1482 (emitting agents: critic (Methodology lens) + consistency-checker, both independently, 2026-07-19).

## Goal

Add a verify_plan.py check (WARN-severity) that a plan's header self-declared version label ("# Plan v<K> ..." when present) matches the persisted plans/v{K}.md filename version, so a stale self-declared version can never ride multiple persists unnoticed.

## Workflow gap

- **Bug observed:** task #1482's amendment plan carried the header "# Plan v4 (amendment)" through persists v5, v6, and v7 (each new-plan-version call re-persisted the stale label); two independent Phase-2 reviewers flagged the version-record ambiguity.
- **Why it is a workflow gap:** new-plan-version rotates filenames mechanically but nothing lints the self-declared header label against the landed filename; every downstream consumer that quotes "plan v4" then names the wrong revision.
- **Confidence (emitter):** medium (Methodology critic tagged it "mechanizable: yes"; consistency-checker recommended the retitle)
- verified-at-filing: `grep -niE 'header.*version|title.*(version|v\{|filename)|plan v[0-9]' scripts/verify_plan.py` → 5 hits, ALL incident-citation prose inside unrelated checks (no header-version-vs-filename check exists — absence claim, 0 functional hits in-target); `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` → 5 commits, none touching header/version labels (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

+ def check_header_version_matches_filename(...):  # cNN
+     m = re.match(r"#\s*Plan\s+v(\d+)\b", first_heading)
+     if m and source_is_issue_mode and int(m.group(1)) != persisted_version:
+         WARN (never FAIL): header self-declares v<X> but persisted file is v<K>;
+         escape: a version-neutral title ("# Plan (amendment) — ...") never triggers.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Also: `tests/test_verify_plan.py` (pin the new check + the version-neutral escape)
- Grep the workflow surface for the pattern before editing (`grep -rln 'Plan v' .claude/skills/adversarial-planner/` — the skill prose may want one line naming the version-neutral-title convention).

## Constraints / invariants

- WARN only, never FAIL (a version-neutral title is the sanctioned fix and must pass silently).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under the standard pipeline; the spawned session carries the recursion guard via the Provenance line below.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 6572ed6a870c

Surfaced prose (verbatim): Methodology critic on #1482 plan v6 — "Cosmetic: v6 file still titled 'Plan v4' ... mechanizable: yes (a verify_plan.py check could assert the plan header version matches the filename); one-off here, not filing." Consistency-checker — "One cosmetic fix for the revise round: plans/v6.md line 1 still titles itself 'Plan v4 (amendment)' — stale version label (plan.md → v6.md); retitle to v6 to keep the plan-version record unambiguous."
