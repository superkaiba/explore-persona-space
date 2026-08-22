---
title: 'workflow-fix: ruff-clean alias for the phase-done waiver token'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cd024fd72054
created_at: '2026-08-05T17:44:18Z'
has_clean_result: false
origin_prompt: 'Formal workflow-fix-candidate block from epm:experiment-implementation
  v11 on #2054: PHASE_DONE_WAIVER_RE forces a # noqa: form ruff flags as invalid;
  accept a ruff-clean # workflow-lint: alias alongside the legacy form'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a formal
`<!-- workflow-fix-candidate v1 -->` block raised on task #2054 (emitting
agent: experiment-implementer, round 11, marker `epm:experiment-implementation`
v11).

## Goal

Accept a ruff-clean alias (e.g. `# workflow-lint: phase-done-reserved`) in
`PHASE_DONE_WAIVER_RE` alongside the legacy `# noqa:` form, and document the
alias as preferred for new waivers.

## Workflow gap

- **Bug observed:** the phase-done waiver grammar
  (`PHASE_DONE_WAIVER_RE = r"#\s*noqa:\s*phase-done-reserved\b"`) forces a
  `# noqa:` form ruff parses as an invalid noqa directive, so every
  legitimately-waived phase script emits permanent ruff
  "Invalid `# noqa` directive" warnings, and reviewers flag load-bearing
  waivers as fixable cosmetics (a #2054 r10 code-review Minor asked to "fix"
  them; the r11 probe conversion made `--check-phase-done-reserved` FAIL
  naming exactly those lines and had to be reverted).
- **Why it is a workflow gap:** workflow_lint chose a waiver token inside
  ruff's `# noqa` namespace, guaranteeing a cosmetic collision between two
  workflow-surface linters on every waived file — a standing
  reviewer-confusion generator.
- **Confidence (emitter):** high (probe evidence: the r11 conversion attempt
  broke the phase-done lint and was reverted; ruff warnings reproduced).
- verified-at-filing:
  `grep -rn 'phase-done-reserved' scripts/ tests/ .claude/rules/ .claude/skills/ .claude/agents/ .claude/hooks/`
  → 45 hits across 17 files at the main checkout (2026-08-05): the check +
  regex in `scripts/workflow_lint.py` (13 hits), its pin tests
  `tests/test_workflow_lint_phase_done_check.py` (10 hits), the documented
  convention in `.claude/rules/pod-side-reporting.md` (2 hits), and 14
  waiver-token call sites across live phase scripts (issue1092_gpu_phase.py,
  issue1481_dispatch.sh, issue1739_leg2.sh, issue1739_nlmap_dispatch.sh,
  issue1947_result3_theory_battery.py, issue1979_gpu.py, seven
  issue2054_*.py files, poll_pipeline.py) — every one carries the
  ruff-colliding form; none uses an alias (none exists yet). Landed-fix
  history: `git log --oneline --since='7 days ago' -- scripts/workflow_lint.py`
  → 5 commits, none touching the waiver grammar.

## Proposed change (candidate diff sketch — refine in planning)

```
- PHASE_DONE_WAIVER_RE = re.compile(r"#\s*noqa:\s*phase-done-reserved\b")
+ PHASE_DONE_WAIVER_RE = re.compile(
+     r"#\s*(?:noqa:|workflow-lint:)\s*phase-done-reserved\b"
+ )
  # + docstring note: prefer the `# workflow-lint:` alias (ruff-clean);
  #   legacy `# noqa:` form still honored
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Secondary: `tests/test_workflow_lint_phase_done_check.py` (pin the alias:
  alias-waived line passes, unwaived reserved emission still FAILs),
  `.claude/rules/pod-side-reporting.md` (document the preferred alias).
- Grep the workflow surface for the pattern before editing
  (`grep -rn 'phase-done-reserved' scripts/ tests/ .claude/rules/ .claude/skills/ .claude/agents/ .claude/hooks/`);
  migrating the 14 existing call sites to the alias is OPTIONAL (legacy form
  stays honored — backwards compatible by design); if migrated, keep the
  edits comment-only.

## Constraints / invariants

- Workflow-surface only; the legacy `# noqa:` form MUST remain accepted (14
  live call sites; a breaking grammar change would red the phase-done lint
  fleet-wide).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; the no-flags default run stays green on the landing tree.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard,
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: cd024fd72054

<!-- workflow-fix-candidate v1 -->
target_file: scripts/workflow_lint.py
bug_observed: The phase-done waiver grammar (`PHASE_DONE_WAIVER_RE = r"#\s*noqa:\s*phase-done-reserved\b"`) forces a `# noqa:` form ruff parses as an invalid noqa directive, so every legitimately-waived phase script emits permanent ruff "Invalid `# noqa` directive" warnings (12 sites on #2054 alone), and a code-review Minor asked to "fix" waivers that are load-bearing lint suppressions.
why_workflow_gap: workflow_lint chose a waiver token inside ruff's `# noqa` namespace, guaranteeing a cosmetic collision between two workflow-surface linters on every waived file.
proposed_change: Accept a ruff-clean alias (e.g. `# workflow-lint: phase-done-reserved`) in PHASE_DONE_WAIVER_RE alongside the legacy form, and document the alias as preferred for new waivers.
diff_sketch: |
  - PHASE_DONE_WAIVER_RE = re.compile(r"#\s*noqa:\s*phase-done-reserved\b")
  + PHASE_DONE_WAIVER_RE = re.compile(
  +     r"#\s*(?:noqa:|workflow-lint:)\s*phase-done-reserved\b"
  + )
  # + docstring note: prefer the `# workflow-lint:` alias (ruff-clean); legacy `# noqa:` form still honored
confidence: high
related_task: #2054
<!-- /workflow-fix-candidate -->
