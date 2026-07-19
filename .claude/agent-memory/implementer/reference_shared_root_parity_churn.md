---
name: Shared-root before/after parity runs fight per-minute tree churn + rewritten local main
description: Recipe for byte-parity comparisons at the shared repo root — same-instant pairs, tasks/-only commit check, and the vanished-FF-target rebase recovery
type: reference
---

Two traps hit while running before/after findings-parity on the shared repo root (#1163, 2026-07-09):

1. **The root tree churns per minute.** Concurrent sessions write `.claude/plans/issue-<N>.md`, task markers, and agent-memory files continuously. A scan-set dump taken minutes apart differs by ±1-2 files that are pure churn, not behavior. Recipe: (a) run the slow leg first, the fast leg IMMEDIATELY after (same-instant pair); (b) record `git rev-parse HEAD` + `git status --porcelain | sha256sum` around the pair; (c) on a HEAD move, check `git diff --name-only <a>..<b> | grep -v '^tasks/'` — an all-`tasks/` delta cannot affect lint findings (no check scans `tasks/`), so the pair stands.

2. **A worktree FF'd to the LOCAL main tip can strand on a vanished commit.** The shared root's `pull --rebase=merges` rewrites unpushed local-main commits, so the sha your `merge --ff-only main` landed on may later not exist on main at all — a later `git merge main` then hits bogus `tasks/` rename/rename conflicts (merge-base regressed far behind). Recovery: `git merge --abort`, then `git rebase --onto main <old-ff-target>` — replays exactly your commits, no tasks/ conflicts. Prevention: FF onto freshly-fetched `origin/main` (fetch + `git -C "$WT" merge --ff-only origin/main`), never local `main` — the local-main FF is itself the #1530 contamination vector (it imports unpushed root task-state commits into the branch).

**How to apply:** any before/after behavior-parity or scan-set comparison against the live repo root; any worktree branch cut while concurrent sessions commit to the root.
