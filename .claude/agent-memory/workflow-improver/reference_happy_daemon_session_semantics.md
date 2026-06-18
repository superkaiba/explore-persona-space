---
name: happy-daemon-session-semantics
description: Verified Happy daemon/wrapper lifecycle facts (from /usr/lib/node_modules/happy/dist source) needed for watcher session-reaping work
metadata:
  type: reference
---

Verified 2026-06-11 from `/usr/lib/node_modules/happy/dist/index-q9G4ktSK.mjs` (the daemon + wrapper bundle):

- **Live wrapper revives its inner Claude IN PLACE.** `claudeRemoteLauncher` loops `claudeRemote`, and `claudeRemote` runs `await opts.nextMessage()` BEFORE spawning the Claude SDK subprocess via `query()`. So a wrapper with NO Claude descendant can be a healthy idle session (post-/clear, post-abort) that the next phone message revives. A no-Claude /proc snapshot is necessary but NOT sufficient evidence of a zombie — any reaper keyed on it needs a grace window (the zombie-wrapper pass uses ≥2 checks + ≥2h).
- **Daemon `resumeSession` always spawns a FRESH wrapper** (`buildResumeLaunch` → `spawnTrackedHappyProcess` with `HAPPY_RECONNECT_*` env); it never revives an existing wrapper's process.
- **`stopSession` deletes the tracking entry BEFORE the child exits**, so `onChildExited` cannot preserve it in `sessionIdToFinishedSession` — a daemon-stopped session is NOT daemon-resumable afterward. The recovery story for reaped sessions is a fresh `spawn_session.py spawn-issue` / `spawn-pm` (same contract as the session-reconcile stop).
- **Claude process cmdline markers on this VM (both must be matched):** native installer `~/.local/share/claude/versions/<v>` and Happy-bundled `@anthropic-ai/claude-agent-sdk-linux-x64/claude`. The wrapper's own cmdline (`node .../happy/dist/index.mjs claude ...`) matches neither.
- **PM session has no intrinsic identification** (no distinguishing field in `~/.happy/sessions.json` metadata; cwd = repo root like other chat sessions). Explicit registration (`pm-session.json` via `spawn_session.py register-pm` / `spawn-pm` / the `/pm` bootstrap, added 2026-06-11) is the only reliable marker.

**How to apply:** any watcher / session-reaper change that reasons about Happy session liveness, resumability, or stopping must respect these; re-verify against the dist bundle if Happy is upgraded ([[watcher-forensics-surfaces]]).
