---
title: 'Step 9c gate red on main: setsid test asserts PID-1 reparent, but user systemd
  is a child subreaper'
kind: infra
tags:
- wf-fix
- setsid-reparent-assert
created_at: '2026-08-08T03:56:32Z'
has_clean_result: false
origin_prompt: '#2189 Step 9c gate adjudication: 155 files, 6407 passed, 1 failed
  — test_setsid_child_survives_group_kill asserts _ppid == 1 but the orphan reparents
  to user systemd (pid 2887, PR_SET_CHILD_SUBREAPER). Reproduced on unmodified main
  at repo root and in isolation.'
workflow: v1
---
# Goal

`tests/test_workflow_setsid_detach_convention.py::test_setsid_child_survives_group_kill`
asserts an orphaned `setsid` child reparents to **PID 1**. On this VM it
reparents to the user-level **`systemd`** instance (a `PR_SET_CHILD_SUBREAPER`
process), so the test fails on `main` for every session — making the `/issue`
Step 9c gate red fleet-wide for reasons unrelated to any round's payload.

Fix the assertion so it accepts a legitimate subreaper as the reparent target
(or skip when a subreaper is present), without weakening what the test actually
proves: that the detached child survives the group kill and escapes a
ppid-tree walk from the dead session.

## Evidence

Failing assertion, `tests/test_workflow_setsid_detach_convention.py:124`:

```python
# The wrapper exited, so the setsid child reparented to PID 1 — a ppid-tree walk
# from the dead session cannot reach it (the second kill vector, beyond killpg).
assert _ppid(detached_pid) == "1"
```

Observed: `_ppid(detached_pid)` returns **2887**, and `ps -o pid=,comm= -p 2887`
is **`systemd`** — the per-user systemd instance, which sets itself as a child
subreaper. Orphans in its session tree reparent to it rather than to PID 1.
This is documented, intended Linux behaviour (`prctl(PR_SET_CHILD_SUBREAPER)`),
not a fault in the detach convention the test is defending.

Reproduced three ways, all on unmodified `main`:

- Repo root (`/home/thomasjiralerspong/explore-persona-space`, branch `main`,
  test file byte-identical to `origin/main`): FAILS.
- Inside the `issue-2189` worktree: FAILS.
- In isolation (single-test invocation, 0.48 s, no load): FAILS — so it is
  **not** load-flaky.

Scope: 1 failed, **4 passed** in that file — only this one case is affected.

The other four cases pass, which is the useful signal: the convention's
*primary* protection (the child survives `killpg`) is verified fine. It is
specifically the PID-1 reparent claim — the "second kill vector" check — that
encodes a false environment assumption.

## Why this matters beyond cosmetics

The Step 9c gate selects this file whenever a round touches the workflow
surface. During #2189 the gate ran 155 mapped test files: **6,407 passed, 1
failed, 14 skipped** — and the single failure was this one, forcing a manual
pre-existing-vs-introduced adjudication (run the same test at `main`, inspect
PID 2887) before the round could proceed. Every future workflow-surface round
pays that same tax, and a permanently-red gate is exactly the condition under
which a *real* failure gets waved through as "probably the known one".

The test landed 2026-07-30 (`ba8359381c`, filed under #1689).

## Proposed fix (for the implementer to settle)

Preferred: assert the detached child is no longer reachable from the dead
session's ppid chain, rather than hardcoding `1`. Concretely — accept
`_ppid(detached_pid)` being either `1` **or** a pid that is a child-subreaper /
not an ancestor of the wrapper; the invariant that matters is "a ppid-tree walk
from the dead session cannot reach it", which the current literal only
approximates on non-subreaper hosts.

Alternatives, weaker:

- Detect a subreaper in the ancestry and `pytest.skip` that single assertion —
  keeps the check honest on hosts where it is meaningful, but silently drops
  coverage on the VM where the convention is actually used.
- Assert `_ppid(detached_pid) != _ppid_of(wrapper)` and that the pid is alive —
  simpler, but proves less.

Do NOT delete the case or loosen it to a bare liveness check: the reparent
escape is a genuine second kill vector the detach convention relies on.

## Provenance

Surfaced by the #2189 orchestrator while adjudicating the Step 9c gate result
(155 files, 6,407 passed, this 1 pre-existing failure). Not caused by #2189 —
that round touched only `.claude/rules/*.md` prose plus
`scripts/consolidate_lessons.py` and its test, with zero setsid/process-group
code changes.
