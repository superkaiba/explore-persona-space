---
title: 'daily-fix: fix red main: test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes
  — Fix the `.splitlines()` JSONL read at `s'
kind: infra
tags:
- urgent-main-red
created_at: '2026-08-01T00:25:53Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue1689_user_slot_capture.py\n\
  bug_observed: The no-flags `workflow_lint.py` run FAILs on the live origin/main\
  \ tree — `--check-jsonl-splitlines` flags `scripts/issue1689_user_slot_capture.py:752`\
  \ (generic `read_text().splitlines()` in a `*.jsonl`-globbing module; file untouched\
  \ by #1866's diff, last commit `7c47ac5e9a` already on origin/main), so every session's\
  \ Step 9c gate must re-classify the red on `tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes`\
  \ / the no-flags bundle.\nwhy_workflow_gap: A live-red no-flags lint on main breaks\
  \ the fleet-wide Step 9c oracle for every intervening round (the #1643/#1681 main-red\
  \ class); verified at implement time via `uv run python scripts/workflow_lint.py`\
  \ -> `workflow_lint: FAIL (1 error(s))` naming exactly that site (bounded single-node\
  \ pytest probe timed out at 120s, so test-level rc is unconfirmed — the lint-level\
  \ evidence is direct).\nproposed_change: Fix the `.splitlines()` JSONL read at `scripts/issue1689_user_slot_capture.py:752`\
  \ (text-mode iteration or `split(\"\\n\")` + `if line.strip()`), or waive a genuinely-safe\
  \ site with `# JSONL_SPLITLINES_EXEMPT: <reason>`.\ndiff_sketch: |\n  - rows = path.read_text(encoding=\"\
  utf-8\").splitlines()\n  + rows = [ln for ln in path.read_text(encoding=\"utf-8\"\
  ).split(\"\\n\") if ln.strip()]\n  (or, if provably ASCII-only content: append \
  \ # JSONL_SPLITLINES_EXEMPT: <reason >=10 chars>)\nurgency: main-red\nfailing_test:\
  \ tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes\nwf_fix: false\n\
  confidence: high\nrelated_task: #1866\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1866. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes` red on origin/main: Fix the `.splitlines()` JSONL read at `scripts/issue1689_user_slot_capture.py:752` (text-mode iteration or `split("\n")` + `if line.strip()`), or waive a genuinely-safe site with `# JSONL_SPLITLINES_EXEMPT: <reason>`.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** The no-flags `workflow_lint.py` run FAILs on the live origin/main tree — `--check-jsonl-splitlines` flags `scripts/issue1689_user_slot_capture.py:752` (generic `read_text().splitlines()` in a `*.jsonl`-globbing module; file untouched by #1866's diff, last commit `7c47ac5e9a` already on origin/main), so every session's Step 9c gate must re-classify the red on `tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes` / the no-flags bundle.
- **Failing node (router-verified):** `tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes -q` -> rc=1 at main @ 3c21f5c278 (2026-08-01T00:25:50Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/issue1689_user_slot_capture.py`
- Failing node: `tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: ea2e993eb348
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1689_user_slot_capture.py
bug_observed: The no-flags `workflow_lint.py` run FAILs on the live origin/main tree — `--check-jsonl-splitlines` flags `scripts/issue1689_user_slot_capture.py:752` (generic `read_text().splitlines()` in a `*.jsonl`-globbing module; file untouched by #1866's diff, last commit `7c47ac5e9a` already on origin/main), so every session's Step 9c gate must re-classify the red on `tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes` / the no-flags bundle.
why_workflow_gap: A live-red no-flags lint on main breaks the fleet-wide Step 9c oracle for every intervening round (the #1643/#1681 main-red class); verified at implement time via `uv run python scripts/workflow_lint.py` -> `workflow_lint: FAIL (1 error(s))` naming exactly that site (bounded single-node pytest probe timed out at 120s, so test-level rc is unconfirmed — the lint-level evidence is direct).
proposed_change: Fix the `.splitlines()` JSONL read at `scripts/issue1689_user_slot_capture.py:752` (text-mode iteration or `split("\n")` + `if line.strip()`), or waive a genuinely-safe site with `# JSONL_SPLITLINES_EXEMPT: <reason>`.
diff_sketch: |
  - rows = path.read_text(encoding="utf-8").splitlines()
  + rows = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]
  (or, if provably ASCII-only content: append  # JSONL_SPLITLINES_EXEMPT: <reason >=10 chars>)
urgency: main-red
failing_test: tests/test_workflow_lint_jsonl_splitlines.py::test_live_tree_passes
wf_fix: false
confidence: high
related_task: #1866
<!-- /workflow-fix-candidate -->
