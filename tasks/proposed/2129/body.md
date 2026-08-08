---
title: 'daily-fix: normalize claude-fable-5[1m] in settings.json'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e45811eb0a29
- daily-auto-filed
created_at: '2026-08-06T07:08:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): /model persisted claude-fable-5[1m]
  (the #545 fleet-outage id) into settings.json; caught only by luck'
workflow: v1
---
# daily-fix: guard against the /model command re-planting `claude-fable-5[1m]` in settings.json (the #545 fleet-outage id)

## Workflow gap

The built-in `/model` command persisted `"model": "claude-fable-5[1m]"` into
`~/.claude/settings.json` on 2026-08-05 (~18:12Z) — the suffixed id that does not exist
for Fable (1M is native; no `[1m]` routing variant) and whose presence caused the #545
fleet-wide subagent outage (~72 h, commit d07424178). It was caught only because Thomas
happened to ask for a settings change minutes later ("Set the default model for when I
write 'claude' on this VM to Fable"); the session normalized it to `claude-fable-5` and
verified with jq. Any future `/model` flip to Fable can silently re-plant the outage-class
id, and nothing watches for it — `scripts/workflow_lint.py`'s AGENT_MODEL_ALLOWLIST covers
EPS project agent files only, not the global settings file.

verified-at-filing: the settings.json read showing `[1m]` and the jq-verified fix are
probed rows in session bee0e15d (rows 9/35/49/51). Compose-time state:
`grep -c 'fable-5\[1m\]' ~/.claude/settings.json` → 0 (currently clean). Dedup: no open
task title matches ([1m]/model-id scan of list-by-status at compose time).

## Proposed change

Add a cheap normalizer/alarm with no behavior change when the file is clean — planner
picks the surface: (a) a watcher arm in `scripts/autonomous_session_watch.py` that checks
`~/.claude/settings.json` (and `~/.claude/settings.local.json`) each pass for
`claude-fable-5[1m]` / any `[1m]`-suffixed Fable/Mythos id, normalizes to
`claude-fable-5`, and posts one deduped alert row; or (b) a SessionStart hook doing the
same read-only check + loud warning. The watcher arm is preferred (fleet-wide, no
per-session cost). Rationale for auto-normalize over alert-only: the broken id kills
every subagent fleet-wide within hours (#545), and the normalization is a byte-exact,
single-known-bad-string rewrite.

## Provenance

- fingerprint: e45811eb0a29

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: /daily 2026-08-05 problem sweep — miner 1 P18 (probed in-session).
