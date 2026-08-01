---
title: 'daily-fix: fix red main: test_shared_vm_thread_caps::test_no_new_torch_before_dotenv_vm_entrypoints
  — load_dotenv before heavy imports in issue1773_register_steer_stats.py'
kind: infra
tags:
- urgent-main-red
created_at: '2026-08-01T01:57:23Z'
has_clean_result: false
origin_prompt: 'Surfaced by #1953 implementer gate-scope union: test_no_new_torch_before_dotenv_vm_entrypoints
  red on origin/main; offender scripts/issue1773_register_steer_stats.py (module-top
  matplotlib/numpy/scipy, no load_dotenv; landed 04e111a7ad, task #1773). urgency:
  main-red; wf_fix: false.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the #1953 /issue session from a main-red concern surfaced during
its implementer's gate-scope union (#1288 duty). The named test is red on
origin/main NOW — every intervening session's Step 9c gate re-classifies it
until this fix lands. Root-caused as unrelated to #1953's diff (comment-only,
different file); the offender landed from task #1773.

## Goal

fix `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` red on origin/main: call `explore_persona_space.orchestrate.env.load_dotenv()` BEFORE the module-top heavy imports in `scripts/issue1773_register_steer_stats.py` (matplotlib at line 20, numpy, scipy — no `load_dotenv` anywhere in the module), or use the test's sanctioned grandfather/waiver mechanism if one exists, so the #847 shared-VM thread caps bind in-process.

## Workflow gap

- **Bug observed:** `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` fails on origin/main — the AST import-order scan flags `scripts/issue1773_register_steer_stats.py` ("module-top heavy import at line 20, first load_dotenv( at line None"). Offender landed in `04e111a7ad` (task #1773, 2026-07-31).
- **Why it matters:** a live-red workflow-invariant test on main breaks the fleet-wide Step 9c oracle for every intervening round (the #1643/#1681 main-red class).
- **Failing node (filer-verified):** `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
- **Confidence (filer):** high
- verified-at-filing: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q` → FAILED at repo-root main (2026-08-01T01:57Z), assertion naming exactly this offender; plus `grep -n load_dotenv scripts/issue1773_register_steer_stats.py` → 0 hits, and `sed -n 1,30p` confirms module-top matplotlib/numpy/scipy imports.

## Proposed change (candidate diff sketch — refine in planning)

```diff
 from pathlib import Path

+from explore_persona_space.orchestrate.env import load_dotenv
+
+load_dotenv()
+
 import matplotlib
```

(or the test's grandfather entry, if inspection shows this stats script is
deliberately VM-exempt — the planner decides with the test file open; note
the sibling `# noqa: E402` imports already anticipate post-statement imports)

## Scope / surfaces

- Primary target: `scripts/issue1773_register_steer_stats.py`
- Failing node: `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` — experiment analysis script);
  scope stays on the named target, never `tasks/` state. Task #1773 may have
  a live session/worktree — the fix lands via this task's own branch; do not
  edit #1773's worktree.

## Provenance

- fingerprint: 04ed63db36bd
- routed-by: /issue 1953 session (surfaced-prose auto-route per .claude/rules/workflow-fix-on-bug.md; emitter: #1953 implementer gate-scope union)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1773_register_steer_stats.py
bug_observed: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints is red on origin/main — scripts/issue1773_register_steer_stats.py has module-top heavy imports (matplotlib line 20, numpy, scipy) with no load_dotenv() call (landed 04e111a7ad, task #1773)
why_workflow_gap: A live-red workflow-invariant test on main breaks the fleet-wide Step 9c oracle for every intervening round (the #1643/#1681 main-red class).
proposed_change: call explore_persona_space.orchestrate.env.load_dotenv() before the heavy imports in scripts/issue1773_register_steer_stats.py (or the test's sanctioned grandfather entry) so the #847 shared-VM thread caps bind in-process
diff_sketch: |
  +from explore_persona_space.orchestrate.env import load_dotenv
  +load_dotenv()
  (inserted before `import matplotlib`)
urgency: main-red
failing_test: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
wf_fix: false
confidence: high
related_task: #1773
<!-- /workflow-fix-candidate -->
