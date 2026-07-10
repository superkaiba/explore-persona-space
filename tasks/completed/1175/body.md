---
title: 'workflow-fix: lint unguarded deferred scripts.* imports'
kind: infra
tags:
- wf-fix
- wf-fix-fp:df7ced7d3930
- daily-auto-filed
created_at: '2026-07-09T06:58:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): A deferred (function-body)
  `import scripts.*` / `from scripts...` under src/explore_persona_space/experiments/**
  crashes pod-side where repo root is not on sys.path (the #853 incident class); no
  mechanical check flags the pattern (the shipped #853 fix was a gotchas doc entry
  only).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #853 (recursion-guarded workflow-fix session).

## Goal

Mechanically catch the script-mode scripts.* import trap the #853 gotchas entry can only describe.

## Workflow gap

- **Bug observed:** A deferred (function-body) `import scripts.*` / `from scripts...` under src/explore_persona_space/experiments/** crashes pod-side where repo root is not on sys.path (the #853 incident class); no mechanical check flags the pattern (the shipped #853 fix was a gotchas doc entry only).
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/workflow_lint.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
def check_deferred_scripts_imports():
    for py in glob("src/explore_persona_space/experiments/**/*.py"):
        tree = ast.parse(...)
        for imp in function_body_imports(tree, module_prefix="scripts"):
            if not guarded_by_syspath_insert_or_try(imp):
                fail(f"{py}:{imp.lineno}: deferred scripts.* import crashes pod-side ...")
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #853 at 2026-07-02T09:40:58Z

Verbatim parked note:

> parked: EPM_WORKFLOW_FIX_SESSION — running under the workflow-fix recursion guard (workflow_fix_target Provenance), so these THREE candidates surfaced by the critic ensemble are LOGGED, not auto-routed (see .claude/rules/workflow-fix-on-bug.md § Recursion guard; plan v3 §13 has full detail):

1. target_file: scripts/workflow_lint.py — mechanical check flagging unguarded deferred scripts.* imports under src/explore_persona_space/experiments/** (needs AST/enclosing-scope reasoning to avoid false positives). confidence: medium.
2. target_file: src/explore_persona_space/backends/gcp.py, src/explore_persona_space/backends/slurm.py, scripts/bootstrap_pod.sh — launcher-level PYTHONPATH=$WORKLOAD_ROOT export so repo root is on sys.path for every managed-lane driver (masks, not removes, the trap for hand launches). confidence: medium.
3. target_file: src/explore_persona_space/orchestrate/env.py — canonical shared ensure_repo_root_on_syspath() helper replacing per-driver hand-rolled parents[N] copies. confidence: medium (src/** is outside the standard workflow-fix surface; needs scoping).

related_task: #853 (raised by critic ensemble round 1, 2026-07-02).
