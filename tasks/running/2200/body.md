---
title: SKILL.md prose asserts setsid orphans reparent to PID 1; user systemd is a
  child subreaper
kind: infra
tags:
- wf-fix
- setsid-reparent-prose
created_at: '2026-08-08T04:46:53Z'
has_clean_result: false
origin_prompt: '#2199 round-1 code-reviewer CONCERN: .claude/skills/issue/SKILL.md:6950
  still says a setsid phase ''reparents to PID 1 when the launching shell exits'';
  false on this VM (adopted by systemd --user pid 2887, PR_SET_CHILD_SUBREAPER). Prose
  twin of the assertion bug #2199 fixed; out of #2199 scope by its plan must-ask clause.'
workflow: v1
---
# Goal

`.claude/skills/issue/SKILL.md:6950` asserts, as fact, that a `setsid`-detached
phase *"reparents to PID 1 when the launching shell exits"*. On this VM it does
not: the orphan is adopted by the user-level `systemd` instance (pid 2887), a
`PR_SET_CHILD_SUBREAPER` process that sits in the ancestry of every Happy/Claude
session. Correct the prose so the live workflow surface stops teaching future
agents a false invariant, without weakening the detach convention it documents.

## Workflow gap

This is the PROSE twin of the assertion bug fixed in #2199. #2199 repaired the
test (`tests/test_workflow_setsid_detach_convention.py`, which asserted
`_ppid(detached_pid) == "1"`); the SKILL.md sentence that the test was pinning
the spirit of was left untouched, deliberately and on the record — #2199's plan
must-ask clause forbade editing SKILL.md because four cases in the very file
being edited assert literal occurrence COUNTS in that doc, so a prose edit there
carried pin-count blast radius during that round.

Surfaced by #2199's round-1 `code-reviewer` as a non-blocking CONCERN, confirmed
by grep as **the only live-surface occurrence**: the other repo hits are
immutable #884 task plans (`tasks/completed/884/plans/v{1,2,3}.md`) and #2199's
own body, which correctly describes the bug.

Why it matters rather than being a pedantic nit: the surrounding paragraph is the
canonical explanation of WHY the detach convention is safe — it tells the reader
that group kills miss a `setsid` phase and that a ppid-tree walk misses it too.
The second half of that claim is the one stated via PID 1. An agent reasoning
from this text will conclude the escape depends on init-adoption specifically,
and may then "helpfully" write a new assertion or watcher predicate keyed on
`ppid == 1` — which is exactly the failure #2199 spent a full round repairing,
and which made the Step 9c gate red on `main`.

## What is actually true

Linux `find_new_reaper` (`kernel/exit.c`): when a parent dies, the kernel walks
the dying parent's **ancestry** for the nearest process with
`PR_SET_CHILD_SUBREAPER` set in the same PID namespace, and falls back to the
pid-namespace init. So the new parent is always an ancestor of the launcher, or
PID 1 — never the launcher, never a descendant, never inside the launcher's
session.

**The convention's actual guarantee is unchanged and remains correct:** a
ppid-tree walk down from the dead session cannot reach the phase, because the
adoptive parent is strictly above the launcher. Only the *mechanism sentence*
is wrong, by over-specifying PID 1 as the adoptive parent. Verified on this VM:
`ps -p 2887 -o pid=,ppid=,comm=` → `2887 1 systemd`, and 2887 is a genuine
ancestor of the session's process tree.

## Proposed change

Reword `SKILL.md:6950` so the parenthetical states the guarantee rather than the
over-specified mechanism — e.g. group kills miss it, and once the launching shell
exits it is adopted by a process ABOVE the dead session (the nearest
child-subreaper ancestor, else the pid-namespace init), so a ppid-tree walk down
from that session cannot reach it either.

Constraints:

- **Do not touch any pinned literal.** `tests/test_workflow_setsid_detach_convention.py`
  asserts exact counts for `sudo -n choom -n -600 -p $$` (== 5) and
  `[step10d] lint-gate earlyoom protection choom=$LINT_GATE_CHOOM` (== 2), plus
  presence of `setsid nohup`, `Detached VM-side long compute phases`,
  `pid=<PHASE_PID>`, `log=<abs log`, `ps -p <pid> -o args=`, `never relaunch`.
  Line 6950's prose is NOT among the pinned literals, so the edit is safe — but
  re-run that test file to confirm rather than assuming.
- Leave the immutable #884 task plans alone; historical plans are evidence, not
  live surface.
- Consider whether the corrected sentence deserves its own durability pin. A
  presence-pin on a distinctive phrase would stop a future edit from silently
  reintroducing the PID-1 claim — the #884/#1045/#1134 lineage is the precedent
  for pinning protection prose. Decide explicitly either way rather than by
  omission.

## Acceptance

- `SKILL.md` no longer asserts PID-1 adoption as the mechanism, and the
  ppid-walk guarantee it documents is still stated (not dropped).
- `uv run pytest tests/test_workflow_setsid_detach_convention.py -q` stays green
  (6 passed post-#2199).
- `uv run python scripts/workflow_lint.py` clean.
- A repo grep for live-surface `reparents to PID 1` / `reparent.*to PID 1`
  returns only immutable task-history hits.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- Surfaced by the #2199 round-1 `code-reviewer` as a non-blocking CONCERN
  (`.claude/skills/issue/SKILL.md:6950` — "pre-existing prose ... out of this
  diff's scope per the plan's must-ask clause, candidate prose follow-up").
- Sibling: #2199 (the test-side fix, landed). Duplicate-of-#2199: #2112.
