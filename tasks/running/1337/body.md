---
title: 'workflow-fix: step9c scratch oracle for file-anchored scan tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4d6cb4e2252b
created_at: '2026-07-15T08:46:57Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1318: R-F class rule in step9c_baseline.py
  refuses the scratch pristine oracle for all GLOB_SCAN_TESTS nodes; #1318''s diff-linked
  __file__-anchored scan-test failure forced a hand-run oracle + recorded override'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1318 (emitting agent: /issue orchestrator).

## Goal

Allow the step9c scratch pristine oracle for __file__-anchored scan-set tests (per-test anchoring allowlist) instead of refusing all GLOB_SCAN_TESTS nodes.

## Workflow gap

- **Bug observed:** `step9c_baseline.py compare --run-pristine` exited 2 (MF-4c indeterminate) on #1318's gate because the R-F eligibility rule class-refuses the scratch-worktree oracle for EVERY `GLOB_SCAN_TESTS` node, although the failing node (`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`) anchors its scan via `Path(__file__).resolve().parents[1]` (tests/test_shared_vm_thread_caps.py:871) and a manual scratch run resolved it trustworthily (identical offender set; positive control: the root's untracked strays absent from the scratch scan output).
- **Why it is a workflow gap:** R-F's rationale ("repo_root()-anchored live-tree scanners read the MAIN root from any cwd") is correct only for repo_root()-anchored scan tests; applying it as a class rule makes any code dirt on the busy shared root (30+ concurrent sessions parking untracked scripts/*.py) an unavoidable exit-2 for every diff-linked scan-set failure — the #1318 gate had to resolve it with a hand-run oracle + recorded override, exactly the improvisation the mechanical compare exists to prevent.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rln "GLOB_SCAN_TESTS" scripts/step9c_baseline.py scripts/select_step9c_tests.py .claude/skills/issue/SKILL.md` -> 3 files (target scripts/step9c_baseline.py: R-F rule at 2 sites — the `test_file not in ctx.sel.GLOB_SCAN_TESTS` eligibility conjunct in `_resolve_pristine_bucket` + its docstring; select_step9c_tests.py owns the GLOB_SCAN_TESTS map itself, not the refusal; SKILL.md documents the behavior) (2026-07-15). Live repro: /tmp/step9c-compare-issue-1318.json (rc=2, MF-4c, scan_set=True) vs /tmp/issue-1318-manual-oracle.log (identical 9-offender set at clean origin/main 9880b7f28b).

## Proposed change (candidate diff sketch — refine in planning)

- In scripts/step9c_baseline.py, add a module constant, e.g.
+ FILE_ANCHORED_SCAN_TESTS: frozenset[str] = frozenset({
+     "tests/test_shared_vm_thread_caps.py",  # Path(__file__).parents[1] anchor, verified :871
+ })
- and relax the R-F conjunct in _resolve_pristine_bucket:
-     and test_file not in ctx.sel.GLOB_SCAN_TESTS  # R-F
+     and (test_file not in ctx.sel.GLOB_SCAN_TESTS
+          or test_file in FILE_ANCHORED_SCAN_TESTS)  # R-F': __file__-anchored scanners scan their own tree
- (alternative: a mechanical anchoring probe — grep the test file for `Path(__file__)`-derived root vs `repo_root()` — instead of a hand-list; planner decides). Update the docstring + the SKILL.md step-1d exit-2 reason list; extend tests/test_step9c_baseline*.py with a dirty-root + file-anchored-scan-test fixture asserting scratch resolution instead of exit 2.

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'GLOB_SCAN_TESTS' .claude/ CLAUDE.md scripts/`) and update every hit that documents the R-F class rule; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- Fail-closed direction preserved: a scan test NOT provably file-anchored keeps the R-F refusal verbatim; the relaxation is opt-in per verified test.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/step9c_baseline.py
- fingerprint: 4d6cb4e2252b

<!-- workflow-fix-candidate v1 -->
target_file: scripts/step9c_baseline.py
bug_observed: compare --run-pristine exited 2 (MF-4c) on 1318 gate because the R-F rule class-refuses scratch resolution for every scan-set node although the failing test anchors its scan via __file__ and a scratch run resolves it trustworthily
why_workflow_gap: R-F's repo_root()-anchoring rationale does not apply to __file__-anchored scan tests, so shared-root code dirt forces exit-2 indeterminates the scratch oracle could safely resolve
proposed_change: allow the step9c scratch pristine oracle for __file__-anchored scan-set tests (per-test anchoring allowlist) instead of refusing all GLOB_SCAN_TESTS nodes
diff_sketch: |
  + FILE_ANCHORED_SCAN_TESTS = frozenset({"tests/test_shared_vm_thread_caps.py"})
  -     and test_file not in ctx.sel.GLOB_SCAN_TESTS  # R-F
  +     and (test_file not in ctx.sel.GLOB_SCAN_TESTS
  +          or test_file in FILE_ANCHORED_SCAN_TESTS)  # R-F'
confidence: high
related_task: #1318
<!-- /workflow-fix-candidate -->
