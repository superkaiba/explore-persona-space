---
title: 'daily-fix: rsync_covered checks RSYNC_EXCLUDE_PATTERNS'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8022dbeae808
- daily-auto-filed
created_at: '2026-07-31T06:53:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): a committed citation nested
  under an excluded dir name inside an include tree (tests/fixtures/eval_results/...)
  false-PASSes the rsync-lane coverage check; rsync_covered never consults backends.slurm.RSYNC_EXCLUDE_PATTERNS.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 Step C (parked workflow-fix-candidate routing) from a prose follow-up parked on task #1835 (emitting agent: #1835's code-reviewer round 1, Minor 1 — a deliberate extension beyond the approved plan's predicate, not a defect in the shipped fix; parked 2026-07-30T16:35:12Z). #1835's own fix (SLURM rsync-lane stranding, PR #1596, squash dcf37f97465c) merged without this extension by design.

## Goal

Extend `rsync_covered()` in `scripts/verify_carryover_inputs.py` to also match path segments against `backends.slurm.RSYNC_EXCLUDE_PATTERNS` and downgrade on a hit, closing the nested-excluded-dir false-PASS residual.

## Workflow gap

- **Bug observed:** a committed citation nested under an excluded dir name INSIDE an include tree (e.g. `tests/fixtures/eval_results/...`) currently false-PASSes the rsync-lane coverage check — `rsync_covered()` checks include-tree membership but not per-segment matches against `RSYNC_EXCLUDE_PATTERNS`.
- **Why it is a workflow gap:** the coverage check exists to catch carryover inputs the SLURM lane's rsync would strand; an exclude-pattern hit nested inside an include tree is exactly such a stranding, so a false PASS defeats the check's purpose.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'def rsync_covered\|RSYNC_EXCLUDE_PATTERNS' scripts/verify_carryover_inputs.py src/explore_persona_space/backends/slurm.py` → `rsync_covered` defined at verify_carryover_inputs.py:372 with 0 `RSYNC_EXCLUDE_PATTERNS` hits in that file (absence confirmed); `RSYNC_EXCLUDE_PATTERNS` defined at slurm.py:789 and exported at :3433 (2026-07-31 filing time). Landed-fix check: `git log --oneline --since='7 days ago' -- scripts/verify_carryover_inputs.py` at plan time should confirm no interim extension.

## Proposed change (candidate diff sketch — refine in planning)

Extend `rsync_covered()` with an fnmatch pass of each path segment against `backends.slurm.RSYNC_EXCLUDE_PATTERNS`; a hit downgrades the covered verdict. ~5-line fnmatch extension + 1 test.

## Scope / surfaces

- Primary target: `scripts/verify_carryover_inputs.py`
- Grep the workflow surface for `rsync_covered` before editing and update every caller-visible assumption; list hits in the plan.

## Constraints / invariants

- Workflow-surface only; the check must stay read-only (verify, never mutate).
- ruff on touched files passes; existing verify_carryover_inputs tests stay green; add the 1 pin test.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 8022dbeae808

- workflow_fix_target: scripts/verify_carryover_inputs.py
- origin: parked prose follow-up on #1835 events.jsonl, ts 2026-07-30T16:35:12Z (routed by /daily 2026-07-30 Step C; fingerprint computed by the filing driver from the synthesized fields above)
