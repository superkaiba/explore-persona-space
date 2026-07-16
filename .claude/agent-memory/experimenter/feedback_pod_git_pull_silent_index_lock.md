---
name: pod git pull silent on stale .git/index.lock
description: A `git pull --ff-only` on a pod whose .git/index.lock survives a crashed mid-git workload prints "Updating ... " and exits 0, but HEAD does NOT advance. Always verify `git rev-parse HEAD` against the brief's `commit=` after a sync — never trust pull output.
type: feedback
---

When a pod's prior workload crashes (SIGKILL, OOM, abrupt shutdown) while a `git` operation is in flight (e.g. the workload itself shelled out to git, or a stage hook ran), `.git/index.lock` survives as a 0-byte file. The NEXT `git pull --ff-only origin <branch>` on the pod:

- Prints `Updating <old_sha>..<new_sha>` and the usual file list.
- Exits with code **0**.
- But HEAD on disk does NOT advance — the index-locked working-tree update step is silently skipped (no stderr, no return-code change).

**Why:** The fetch + ref update phases of `git pull` don't take the index lock; the index/working-tree mutation phase does. Holding the lock blocks the mutation but the surrounding pull command silently considers itself successful (this is a known git rough edge in `pull`'s composition of fetch + merge — the ff-merge fast-path under index lock doesn't error out in older git versions widely deployed on RunPod images).

**Why:** A pod cycle that pulls clean code, runs `git rev-parse HEAD`, gets the OLD sha back, and proceeds to launch is a silent-stale-code disaster — every fix in the latest commit is invisible to the run, and the orchestrator's launch marker (`commit=...`) lies. The round-6 #653 relaunch (2026-06-26) hit this: a crashed mid-Jun-25 dispatcher left an index.lock; the relaunch's `git pull` returned exit 0 with the "Updating" line, but HEAD stayed at `a7c67b23ae` instead of advancing to `b4e40869f5`.

**How to apply:** ALWAYS, on EVERY pod-side git sync:

1. Run `git pull --ff-only origin <branch>` as usual.
2. Then run `git rev-parse HEAD` and verify the sha matches the brief's `commit=` field. Never trust the pull's stdout — verify the on-disk HEAD instead.
3. On mismatch: `ls -la .git/index.lock`. If present AND the mtime is old AND `pgrep -af git` shows no live git proc, remove the lock and re-pull.
4. The re-pull (after lock removal) can take long enough that the SSH MCP ~30s client cap times out before git returns — but git completes server-side regardless. Re-probe HEAD in a FRESH SSH call rather than treating the timeout as a failure.
5. If `pgrep -af git` shows a live git proc, do NOT remove the lock — wait. A live git is doing legitimate work.

**Worked recipe for the experimenter pre-launch sync** (now standing instruction
in `experimenter.md` § "Before Running" **item 2** — the canonical sync step;
implemented commit `084211a364`):

```
ssh_execute pod-X 'cd /workspace/explore-persona-space && \
  git fetch origin <branch> && \
  git checkout <branch> && \
  git pull --ff-only origin <branch>; \
  echo "HEAD=$(git rev-parse HEAD)"'
# expect HEAD == brief's `commit=` field if present; else HEAD ==
# `origin/<branch>` (the ref the fetch just advanced) — the realistic
# fallback when the brief carries no explicit `commit=`.

# On mismatch:
ssh_execute pod-X 'cd /workspace/explore-persona-space && \
  ls -la .git/index.lock; pgrep -af git'
# Verify lock is stale (mtime old, no live proc), then:
ssh_execute pod-X 'cd /workspace/explore-persona-space && \
  rm -f .git/index.lock && git pull --ff-only origin <branch>'
# Re-probe HEAD in a fresh call (the pull may have outlived the SSH cap):
ssh_execute pod-X 'cd /workspace/explore-persona-space && git rev-parse HEAD'
```

**Write-side variant (#1336, 2026-07-15):** the same stale 0-byte lock also
kills the WORKLOAD's own pod-side `git commit` (dispatch result-commit phases
per `.claude/rules/pod-side-reporting.md`) — loudly this time (`Unable to
create ... index.lock: File exists`), crashing the dispatch mid-upload so its
results sentinel / `[phase=done]` never fire. So the pre-launch lock check is
not only a sync-mismatch recovery: before launching ANY workload whose tail
commits pod-side, probe `ls .git/index.lock` and clear a confirmed-stale lock
(0 bytes + old mtime + `pgrep -a git` empty) even when HEAD already matches.
On #1336 the lock predated the launch (14:55 vs 14:56); the run halted at G1
as designed, then died at the upload commit; recovery = rm lock + relaunch the
resumable upload tail (idempotent bulk uploads + commit-if-diff + verified
push completed clean in seconds).
