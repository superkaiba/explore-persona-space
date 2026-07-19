---
title: 'workflow-fix: check 31 FAILs prose-named unembedded per-unit companions sans
  exemption'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5c3a1019c32d
created_at: '2026-07-18T13:22:42Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1426 r1 surfaced prose: tighten verify_task_body
  check 31''s prose-naming escape to require an explicit deliberately-linked-not-embedded
  exemption phrase (mechanizable: yes)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose workflow-fix suggestion raised on task #1426 (emitting agent: clean-result-critic, round-1 verdict, `mechanizable: yes` sketch).

## Goal

Tighten `verify_task_body.py` check 31 (`check_orphaned_per_unit_figures`) so a Results section that NAMES a committed per-unit companion figure in prose without EMBEDDING it requires an explicit deliberately-linked-not-embedded exemption phrase (else FAIL, or at minimum a distinct WARN class the critic can key on).

## Workflow gap

- **Bug observed:** two #1426 results named their committed per-unit companion figures in prose ("the per-context views behind these aggregates are `mlc_percontext_delta_scatter.png` and `pma_percontext_read1_scatter.png`, committed at the same pin") instead of embedding them; check 31 did not flag it (the stem-in-prose mention satisfied its pattern) and the gap surfaced only at clean-result-critic Lens 11 as a substantive REVISE round.
- **Why it is a workflow gap:** check 31's stem-in-prose blind spot is DOCUMENTED as a caveat in verify_task_body.py (~line 775-777: "the link itself puts the stem in the prose") but not enforced — the mechanical verifier passes a body shape the Lens-11 adversarial gate then rejects, costing a full critic round for a mechanizable check.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rn "check 31\|per-unit companion" scripts/verify_task_body.py` → 5+ hits in 1 file incl. `check_orphaned_per_unit_figures` (line 598 doc row, WARN severity) and the documented stem-in-prose caveat at line 777 (2026-07-18); context read — the caveat is documented-unfixed, not landed; `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → no landed fix touching check 31 this week.

## Proposed change (candidate diff sketch — refine in planning)

In `check_orphaned_per_unit_figures` (check 31): when a per-unit companion basename pattern appears in a `### <result>`'s PROSE but the figure is not embedded in that section, require an explicit exemption phrase (e.g. "deliberately linked, not embedded" / "not embedded: <reason>") in the same section; absent the phrase, emit a FAIL (or a new distinct WARN class `companion-named-not-embedded`) instead of passing on the prose mention. Keep the existing orphaned-figure WARN behavior for figures never mentioned at all.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'stem-in-prose\|check_orphaned_per_unit_figures' scripts/ .claude/`) and update every hit; keep `.claude/skills/clean-results/SPEC.md` + `clean-result-critic` Lens 11 text consistent; update `tests/test_verify_task_body.py` pins.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; grandfathered v3/v2 bodies are never newly hard-FAILed by a v4 rule (forward-only).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 5c3a1019c32d

Surfaced prose (clean-result-critic, #1426 round 1): "my fix list tags a `mechanizable: yes` sketch (tighten check 31's prose-naming escape to require an exemption phrase) — I did not emit a workflow-fix candidate block because this session runs under AUTO_REVIEW_DISABLED=1; route it as you see fit."
