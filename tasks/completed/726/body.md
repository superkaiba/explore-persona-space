---
title: 'daily-fix: spawn_session verifies Happy injection patch before relying on
  it'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4e4574ecae6d
- daily-auto-filed
created_at: '2026-06-29T06:38:54Z'
has_clean_result: false
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-06-28 problem sweep (route 2 — behavior change, files for independent review). Surfaced during the in-session deep-check of the idle-session pile (the `#720` investigation, session `3cd355aa`).

## Goal

Make `spawn_session.py` verify (and re-apply / warn loudly about) the Happy daemon injection patch before relying on `HAPPY_INITIAL_PROMPT` injection — so a silently-reverted patch fails loud instead of producing idle sessions.

## Workflow gap

- **Bug observed:** The Happy initial-prompt injection path is a `sudo` monkey-patch of a vendored Happy file (`/usr/lib/node_modules/happy/dist/index-q9G4ktSK.mjs`, sentinel v4) applied by `scripts/patch_happy_daemon.py`. It silently reverts on every `npm update happy`; after a revert, `spawn_session.py spawn-issue --auto`'s prompt injection + `claudeArgs` forwarding silently no-op, so a spawned session never fires its `/issue N` skill and sits idle (the exact "spawned but never ran" failure mode behind today's idle-session pile — e.g. `#685` "silently sat idle for 10 hours, never ran /issue 685").
- **Why it is a workflow gap:** `scripts/spawn_session.py` + `scripts/patch_happy_daemon.py` are workflow-helper surface. `patch_happy_daemon.py` already has a `check` mode that anticipates "the daemon has probably been upgraded," but nothing in the spawn path RUNS it — so the revert is invisible until someone notices idle sessions. Dormant today (the patch is currently applied) but guaranteed to bite on the next Happy upgrade.
- **Confidence (emitter):** medium (concrete + reproducible failure mode; the immediate idle-session pile was a different root cause — `#720` — so this is the latent residual, not tonight's fire)

## Proposed change (refine in planning)

- Before `spawn_session.py` relies on injection, run `patch_happy_daemon.py --check`; on a detected revert, re-apply (if non-interactive/safe) or fail loud with a clear message naming the re-apply command — never silently spawn a session that won't fire its skill.
- Optionally add the same `--check` to a cron / the autonomous-session watcher so a revert is surfaced proactively.

## Scope / surfaces

- Primary: `scripts/spawn_session.py`, `scripts/patch_happy_daemon.py`
- Add a test asserting the spawn path detects a reverted/absent patch sentinel and fails loud rather than no-op-spawning.

## Constraints / invariants

- Workflow-surface only. Re-applying a `sudo` patch is environment-sensitive — if a non-interactive re-apply isn't safe, prefer fail-loud + surface over a silent spawn.
- `scripts/workflow_lint.py --check-references` + ruff on touched files pass.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/spawn_session.py, scripts/patch_happy_daemon.py
- fingerprint: 4e4574ecae6d
