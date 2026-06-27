---
name: watcher-forensics-surfaces
description: Where to look when investigating a missed crash-recovery / watcher incident (logs, state dir, Happy sessions.json), plus the cwd=worktree spawn fact
metadata:
  type: reference
---

Investigating "the session watcher missed task #N" incidents (like 2026-06-10 #472/#518):

- **Watcher logs:** `logs/autonomous_session_watch/YYYY-MM-DD.log` (main checkout, one file/day, PT timestamps). Each ~10-min run logs per-issue lines for the respawn / pod-safety / stalled / orphan passes. `awk '/^=== .* start/{ts=$2} /issue #N:/{print ts, $0}'` reconstructs a per-run timeline.
- **State dir:** `~/.eps-autonomous/` — `issue-<N>.json` (autonomous registry; REWRITTEN on every `spawn-issue --auto`, so post-recovery mtimes destroy pre-incident evidence), `manual-issue-<N>.json`, `stalled-<N>.json`, `orphan-<N>.json`, `pod-safety-<N>.json`, `issue-progress/<N>.json` (self-report; its frozen `ts` dates a session death precisely), `issue-tick-last-status/<N>.json`.
- **Session generations:** `~/.happy/sessions.json` → `sessions.{sid}.metadata.path` + `savedAt` (ms epoch). Filter by `issue-<N>` in path to see every driver generation.
- **Key fact:** `spawn-issue --auto` registers sessions with **cwd = the issue worktree** when it exists (NOT repo root) — so superseded driver generations all sit in the worktree, and any cwd-based liveness heuristic matches zombies. The registered `happy_session_id` (+ manual entry id) is the only precise driver signal.
- The self-report (`issue-progress/<N>.json`) refresh cadence ≈ the /issue poll loop — ADAPTIVE since 2026-06-12 (§7): sleep 540s normally, up to 1800s on quiet healthy `running` stretches (poller-emitted `next_interval`), so a self-report up to ~30 min stale can be a healthy quiet run; a fresh self-report does NOT prove the REGISTERED driver is alive (a zombie generation can tick it — #518).
