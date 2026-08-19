---
title: 'Optional: widen check_no_repo_root_syspath to src/ (0 hits today; needs own
  justification)'
kind: infra
tags: []
created_at: '2026-08-19T03:41:49Z'
has_clean_result: false
parent_id: 2183
origin_prompt: '#2183 plan v2 §8 item 2 / task body criterion 4: src/ widening optional,
  justified separately if taken.'
workflow: v1
---
# Optional: widen `check_no_repo_root_syspath` to `src/`

## Goal

Extend `workflow_lint.py::check_no_repo_root_syspath` (tests/ + scripts/ as of #2183) to also scan `src/`, with its own justification and test coverage.

## Why (and why optional)

`src/` had 0 hits at both the #2181 enumeration and the #2183 widening — there is no live offender and no incident. #2183's task body (criterion 4) made this widening explicitly OPTIONAL and requiring separate justification; the #2183 plan (v2 §4d/§8 item 2) deliberately did not take it. Filed `on_hold` so the option is durable without entering the active proposed queue — revive via `task.py set-status <N> proposed` if a src/ offender ever appears or the justification is written.

## Acceptance criteria (if taken)

1. Scan scope gains `src/`; CheckScope + docstring + argparse help updated.
2. A src/-scope firing case + green corrective-form case in `tests/test_workflow_lint_no_repo_root_syspath.py`.
3. Live tree green under the widened scope; no-flags bundle no new failures.

## Provenance

Deferred from #2183 (plan v2 §8 item 2; task body acceptance criterion 4).
