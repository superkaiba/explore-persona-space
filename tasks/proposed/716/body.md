---
title: 'workflow-fix: task.py set-status nested-dir git mv on destination collision'
kind: infra
tags:
- wf-fix
- wf-fix-fp:80df3c2b1fa2
created_at: '2026-06-28T12:04:59Z'
has_clean_result: false
origin_prompt: '<!-- workflow-fix-candidate v1 surfaced from task #681''s Step 10
  incident (2026-06-28) — see epm:done v1 marker on #681 -->'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #681 during its Step 10 set-status transition.

## Goal

Add a destination-dir-collision guard in `set_status()` (the
`task_workflow.py` library function) so the function refuses with a clear
error (or recovers safely) when `tasks/<new-status>/<N>/` already exists as
an orphan directory. The current implementation lets `git mv` nest as
`tasks/<new-status>/<N>/<N>/` and crashes on the subsequent body.md read.

## Workflow gap

- **Bug observed:** `task.py set-status` nested-dir `git mv`: when the
  destination `tasks/<new-status>/<N>/` already exists (even just as an
  empty orphan directory from a prior task numbering or a concurrent
  session's artifact directory), `git mv tasks/<old>/<N> tasks/<new>/<N>`
  treats the destination as a container and nests as
  `tasks/<new>/<N>/<N>/`. The script then crashes when it tries to read
  `tasks/<new>/<N>/body.md` (which is now at `<N>/<N>/body.md`),
  leaving the workflow in a half-moved state. Recovery requires manual
  git surgery (flatten the nested dir + update REGISTRY.json by explicit
  path + re-commit). Concrete incident: #681's Step 10 transition from
  `reviewing` to `completed` on 2026-06-28 (~10 min of manual recovery;
  caught `tasks/completed/681/artifacts/` orphan dir from a prior #681
  task numbering as the trigger).
- **Why it is a workflow gap:** `task.py set-status` is the canonical
  mutator of task lifecycle state; a destination-dir-collision case it
  doesn't detect breaks the atomic-transition contract every other
  workflow surface depends on. The orphan condition is common (different
  task numberings, leftover untracked dirs from prior sessions, concurrent
  filesystem activity), so this should be defended against, not assumed
  away.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```python
# src/explore_persona_space/task_workflow.py, around set_status()
dest_parent = tasks_dir() / new_status
dest = dest_parent / str(N)
if dest.exists():
    if any(dest.iterdir()):
        raise SetStatusError(
            f"task #{N}: destination {dest} already exists and is non-empty "
            f"(orphan from prior task numbering, or concurrent session "
            f"artifact?). Refusing nested-dir git mv. "
            f"Inspect + clean: `ls -la {dest}` then re-run."
        )
    # empty dest: rmdir and proceed (safe — git mv won't nest)
    dest.rmdir()
git("mv", str(src), str(dest))
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (the
  `set_status()` library function — `scripts/task.py` is the CLI wrapper
  that calls into it).
- Add a regression test in `tests/test_task_workflow*.py` that creates an
  orphan `tasks/<status>/<N>/` dir (one empty, one non-empty) and asserts
  the right behavior (empty → recover by rmdir + proceed; non-empty →
  raise SetStatusError).
- Grep the workflow surface for any other `git mv` callers that might
  share this pattern.

## Constraints / invariants

- Workflow-surface only — `src/explore_persona_space/task_workflow.py` IS
  workflow surface (CLAUDE.md `.claude/rules/workflow-fix-on-bug.md`
  scope list).
- The atomic-transition contract (set-status is `git mv` + REGISTRY
  update + epm:status-changed marker in ONE commit, held under `flock`)
  must be preserved.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; `tests/test_task_workflow*.py` passes.
- The fix is mechanical + small (one extension to set_status + a
  regression test); it is NOT architectural per the workflow-fix-on-bug
  rule (does not rename a status enum, does not change task.py CLI
  contract, does not relocate an agent or skill). Auto-approve under
  the 0-GPU-h cap.

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 80df3c2b1fa2

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py
bug_observed: task.py set-status nested-dir git mv when destination tasks/<new-status>/<N>/ already exists (orphan dir from prior task numbering or concurrent session); git mv treats it as a container and nests as tasks/<new-status>/<N>/<N>/, then task.py crashes reading body.md at the un-nested path. Concrete incident: #681's Step 10 transition reviewing → completed on 2026-06-28, ~10 min of manual recovery.
why_workflow_gap: task.py set-status is the canonical mutator of task lifecycle state; an unguarded destination-dir-collision case leaves the workflow in a half-moved state requiring manual git surgery.
proposed_change: add a destination-dir-collision guard in set_status() (task_workflow.py library function): before git mv, check if dest exists; if empty, rmdir + proceed; if non-empty, raise SetStatusError with a clear message naming the orphan + the inspection command.
diff_sketch: |
  dest = tasks_dir() / new_status / str(N)
  if dest.exists():
      if any(dest.iterdir()):
          raise SetStatusError(f"task #{N}: destination {dest} already exists and is non-empty (orphan?). Refusing nested-dir git mv. Inspect: ls -la {dest}")
      dest.rmdir()
  git("mv", str(src), str(dest))
confidence: high
related_task: #681
<!-- /workflow-fix-candidate -->
