---
title: Autonomous plan-gate cap default is 24 in task.py but 100 in every doc and
  park message
kind: infra
tags: []
created_at: '2026-08-07T07:05:38Z'
has_clean_result: false
origin_prompt: 'Surfaced while verifying that #2163''s autonomous session cannot park
  at plan_pending (user: ''make sure it doesn''t wait at plan pending''). Repo-wide
  grep for EPM_PLAN_AUTOAPPROVE_GPU_HOURS found scripts/task.py:417,420 defaulting
  the code-enforced gate cap to 24 while SKILL.md, workflow.yaml, issue-tick, spawn_session.py
  and autonomous_session_watch.py all use/report 100 — so an env-less autonomous park
  decides on 24 and reports ''over 100 GPU-h cap''.'
workflow: v1
---
# Autonomous plan-gate cap default is 24 in code but 100 everywhere it is documented and reported

## Problem

`_resolve_autonomous_plan_gate` in `scripts/task.py` is the code-enforced Step 2c
autonomous plan-approval gate. It resolves the cap as:

```python
cap_raw = os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "24")   # scripts/task.py:417
try:
    cap = float(cap_raw)
except (TypeError, ValueError):
    cap = 24.0                                                      # scripts/task.py:420
```

Every other surface that documents, defaults, or REPORTS this cap uses **100**:

| surface | line | value |
|---|---|---|
| `scripts/task.py` `_resolve_autonomous_plan_gate` | 417, 420 | **24** |
| `scripts/task.py` `set-status` help text | 1457 | **24** |
| `scripts/spawn_session.py` (watcher-respawn cap read) | 2735 | 100 |
| `scripts/spawn_session.py` (`--auto-approve-gpu-hours` help) | 3480 | 100 |
| `scripts/autonomous_session_watch.py` (park PushNotification text) | 31040 | 100 |
| `.claude/skills/issue/SKILL.md` (Step 2c park branch) | ~1856 | 100 |
| `.claude/skills/issue-tick/SKILL.md` | 353 | 100 |
| `.claude/plans/issue-586.md` (campaign per-child cap) | 85 | 100 |
| `.claude/agents/follow-up-proposer.md` | 401 | (cites the cap, no number) |

## Why it matters

The divergence is invisible in the common path: `spawn_session.py:2391` injects
`EPM_PLAN_AUTOAPPROVE_GPU_HOURS` into every `--auto` session's env (and
`:2588` does the same per-child for campaigns), so the 24 default is never
reached there. It bites wherever the env var is absent while
`EPM_AUTONOMOUS_SESSION` is set — a hand-driven autonomous invocation, a
dispatch path that does not go through `spawn_session`, or any future caller of
`set-status --auto-approve-if-autonomous`.

**The user-visible symptom is worse than a silent threshold change.** The park
DECISION is made in `task.py` against 24, but the park MESSAGE is composed at
`autonomous_session_watch.py:31040` and `issue-tick/SKILL.md:353` with
`os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "100")`. In an env-less
context a 30 GPU-h plan therefore parks and then reports
`over 100 GPU-h cap` — naming a threshold the plan did not cross. That is an
actively misleading notification, not merely a conservative default.

## Fix direction — genuinely open, do not presume

Two defensible resolutions; the planner should pick one with a reason, not
default to the larger number because it is more common:

1. **Align code to 100.** Matches every doc, the notification text, and the
   campaign per-child cap. Argument: the docs are the contract that agents and
   the user actually read, and the injected value is already 100 in practice.
2. **Align docs to 24.** Argument: 24 is the more conservative spend default for
   a path that reaches the gate without an explicit cap having been set, and a
   lower fail-safe for an unconfigured caller is the safer direction for a gate
   whose whole purpose is spend control.

Whichever is chosen, the REAL defect is that the deciding default and the
reporting default are read independently at different call sites. The fix must
make them impossible to diverge again — resolve the cap once (a single helper /
module constant that both the decision and every message read) rather than
repeating `os.environ.get(..., "<literal>")` at six call sites. A test pinning
decision-cap == reported-cap belongs with it.

## Scope

- `scripts/task.py` (`_resolve_autonomous_plan_gate`, the `set-status` help text
  at 1457).
- `scripts/autonomous_session_watch.py:31040`, `.claude/skills/issue-tick/SKILL.md:353`,
  `.claude/skills/issue/SKILL.md` Step 2c park branch — message composition.
- `scripts/spawn_session.py:2735, 3480` — reconcile with whichever default wins.
- A regression test asserting the gate's cap and the park message's cap resolve
  to the same value with the env unset.

Out of scope: the fail-safe-on-blank-estimate behavior itself, which is correct
and should not change.

## Provenance

Found while verifying that task #2163's autonomous session could not park at
`plan_pending` (user directive: "make sure it doesn't wait at plan pending").
#2163 is NOT affected — its session has the env injected at 100.0 and its
estimate is 4 GPU-h, under both candidate defaults. The inconsistency was
surfaced by a repo-wide grep for `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` during that
check, not by an observed failure, so no incident is attached.
