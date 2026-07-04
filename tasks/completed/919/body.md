---
title: 'workflow-fix: route user ''pause <N>'' to on_hold (durable, watcher-safe park)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:be4ec04c96eb
created_at: '2026-07-03T08:38:45Z'
has_clean_result: false
parent_id: 816
origin_prompt: 'workflow-fix candidate raised on #816: user ''pause 816'' was recorded
  as a prose-only epm:progress marker with status left at running; the watcher orphan-respawn
  pass (blind to prose holds) respawned an autonomous session ~2h later against the
  explicit user hold. No workflow surface routes ''pause <N>'' to the existing durable
  mechanism (status on_hold, already in the watcher PARK set).'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #816 (emitting agent: /issue orchestrator, watcher-respawned
session honoring a user pause).

## Goal

Add a "user pause" routing rule to the workflow surface: on a user "pause <N>"
directive, the driving session parks the task at `on_hold` (already in the
watcher's PARK set — durable, watcher-safe) with a resume-recipe note, instead
of leaving status at an ACTIVE value with a prose-only hold marker the watcher
cannot parse.

## Workflow gap

- **Bug observed:** User said "pause 816" (2026-07-02). The driving session
  recorded the hold as a prose `epm:progress` marker (v72: "DO NOT resume,
  respawn, or auto-dispatch ... watcher: do not crash-recover") and left the
  task at `status=running`, then stopped. ~2h later the watcher's
  orphan-respawn pass — which keys on "active status + no live registered
  session + no fresh markers" and cannot parse prose holds — respawned an
  autonomous `/issue 816 --auto` session AGAINST the explicit user hold
  (consuming respawn attempt 1/2 for the day and a fresh session's tokens).
  The respawned session had to detect the pause note itself and park the task
  at `on_hold` by hand.
- **Why it is a workflow gap:** No workflow surface prescribes how a session
  should implement a user "pause <N>". The correct durable mechanism already
  exists — `on_hold` is in the watcher's PARK set (`autonomous_session_watch.py`
  line ~478) and the status enum documents it as "explicitly set aside ...
  excluded from auto-dispatch" — but nothing routes the "pause" verb to it, so
  sessions improvise prose markers that every automated pass is blind to.
  (#903's `paused-takeover` sentinel does NOT cover this: it is a short-TTL
  (default 6h, FAIL OPEN) session-takeover shield, not an indefinite user
  hold — even a sentinel-writing pause would be respawned after 6h.)
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ CLAUDE.md (§ Routing experiment intent, next to the abort affordance) and/or
+ .claude/skills/issue/SKILL.md (§ abort affordance):
+   **User pause** ("pause <N>", "put <N> on hold"): implement the hold
+   DURABLY, not as prose —
+     1. uv run python scripts/task.py set-status <N> on_hold \
+          --note "USER PAUSE <verbatim>; resume: set-status <N> running +
+                  spawn_session.py spawn-issue --issue <N> --auto"
+        (on_hold is in the watcher PARK set: no orphan-respawn, no
+         auto-dispatch; markers re-enter the /issue loop at the same round
+         on resume). If a same-issue follow-up round holds
+         `followups_running`, note the status-hold interaction explicitly.
+     2. Stop/park the driving session; stop any pod per Step 8-bis.
+   **Resume** (user greenlight only): set-status <N> running, then
+   spawn-issue --issue <N> --auto.
+ .claude/rules/background-automation.md: one line noting on_hold is the
+   user-pause mechanism (PARK set) vs the #903 paused-takeover sentinel
+   (short-TTL takeover shield) — the two are not interchangeable.
```

## Scope / surfaces

- Primary target: `CLAUDE.md`, `.claude/skills/issue/SKILL.md`, `.claude/rules/background-automation.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'pause' .claude/ CLAUDE.md scripts/spawn_session.py`) and update
  every relevant hit; list them in the plan. Do NOT touch the #903
  paused-takeover sentinel logic (different mechanism, working as designed).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; if `CLAUDE.md` changes it
  stays consistent with the rule files.
- The rule must be documentation/routing only in v1 — no new marker kinds, no
  status-enum change (on_hold already exists), no watcher code change needed.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: CLAUDE.md, .claude/skills/issue/SKILL.md, .claude/rules/background-automation.md
- fingerprint: be4ec04c96eb

Origin: task #816 user pause (2026-07-02 "pause 816", epm:progress v72) vs
watcher orphan-respawn (epm:progress v73, 2026-07-03T08:33:29Z, attempt 1/2).
The respawned session detected the conflict and parked #816 at on_hold
manually (epm:status-changed running -> on_hold, 2026-07-03).
