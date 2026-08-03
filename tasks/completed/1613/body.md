---
title: 'workflow-fix: WARN on zero-resolving --map-files input in select_step9c_tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2988dcca64b1
created_at: '2026-07-23T04:27:54Z'
has_clean_result: false
origin_prompt: 'code-reviewer round-1 prose follow-up on #1610: --map-files handed
  a source file silently prints zero mapping pairs (each content line a nonexistent
  path) — WARN/error when the FILE argument''s content lines resolve to zero existing
  repo paths. Target: scripts/select_step9c_tests.py. Confidence: medium.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1610 (emitting agent: code-reviewer, round-1 verdict prose
follow-up).

## Goal

Make `scripts/select_step9c_tests.py --map-files` fail loud (WARN, or exit 2 in
the source-file-argument case) when the FILE argument's content lines resolve to
zero existing repo paths — closing the silent zero-map false-negative when a
caller hands a source file directly instead of a path-list file.

## Workflow gap

- **Bug observed:** `--map-files` handed a source file (e.g. a `.py`) instead of
  a path-list file silently prints zero mapping pairs and exits 0 — every
  content line is treated as a nonexistent repo path, and no WARN fires (the
  #1573 `no mapped tests for code file` floor iterates
  `literal_path_targets`-eligible EXISTING files only), so a caller verifying
  test-mapping coverage gets a false negative indistinguishable from "no mapped
  tests". Surfaced when task #1610's plan §7 wrote the verify command in exactly
  this malformed shape and it "passed" vacuously.
- **Why it is a workflow gap:** `select_step9c_tests.py` is the single mapping
  source for the Step-9c gate and the Step-10d merge-gate mapped-test leg
  (#1147); a silent zero-map on an operator argument-shape error defeats the
  fail-loud floor the mapping mode exists to provide.
- **Confidence (emitter):** medium
- verified-at-filing: semantic probe `uv run python
  scripts/select_step9c_tests.py --map-files
  src/explore_persona_space/analysis/representation_shift.py` → 0 stdout pairs,
  stderr carries ONLY the provenance breadcrumb, exit 0 (2026-07-23); per-target
  guard grep `grep -n 'map_files\|map.files' scripts/select_step9c_tests.py` →
  arg parse at :1195, map-files branch at :1236-1302, no zero-resolution guard
  in the branch; the documented adjacent gap (docstring ~:105-123) covers
  out-of-prefix `.sh`/`.py` paths INSIDE a valid list, not a source-file
  argument; landed-fix history `git log --oneline --since='7 days ago' --
  scripts/select_step9c_tests.py` → 5 commits (#1589/#1573-family et al.), none
  adding this guard.

## Proposed change (candidate diff sketch — refine in planning)

```
  # in main(), --map-files branch, after reading `raw` lines:
+ listed = [ln.strip() for ln in raw.splitlines() if ln.strip()]
+ resolved = [p for p in listed if (work_root / p).exists()]
+ if listed and not resolved:
+     print("select_step9c_tests: WARN — --map-files input resolved to ZERO "
+           f"existing repo paths ({len(listed)} lines); did you pass a source "
+           "file instead of a path-LIST file?", file=sys.stderr)
+     # exit 2 when the argument itself is an existing .py/.sh source file
```

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'map-files' .claude/ CLAUDE.md scripts/`) and update every hit if
  the contract wording changes; list them in the plan. Pin the new behavior with
  a test in `tests/test_select_step9c_tests.py` (the existing drift-pin home).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  existing `--map-files` consumers (Step 10d gate blocks, inline lint gate) must
  keep working on VALID path-list inputs (zero-pair output on a valid list whose
  paths simply map to nothing stays exit 0 — only the zero-RESOLUTION shape
  warns/errors).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: 2988dcca64b1

Verbatim surfaced prose (code-reviewer round 1, task #1610):
> Follow-up (orchestrator should consider): `scripts/select_step9c_tests.py
> --map-files` silently prints zero mapping pairs when handed a source file
> directly instead of a path-list file (each source line treated as a
> nonexistent path) — a silent false-negative verification shape. Proposed
> change: WARN (or error) when the FILE argument's content lines resolve to
> zero existing repo paths, especially when the argument itself is an existing
> `.py`/`.sh` file. Target: `scripts/select_step9c_tests.py`. Confidence:
> medium.
