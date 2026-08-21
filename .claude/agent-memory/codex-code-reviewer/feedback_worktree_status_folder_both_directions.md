---
name: worktree task-folder status can be stale in EITHER direction
description: Brief/spec plan paths cite a status folder; the worktree may carry the task under a DIFFERENT status dir (frozen-at-cut OLD status, or rebased-post-move CURRENT status). Probe `ls <worktree>/tasks/*/<N>` before falling back to plan inlining.
metadata:
  type: feedback
---

Step 2-pre-b's binary logic (path exists → diff-check → else inline the
canonical plan) misses a third case, hit on #2324 Leg B r1 (2026-08-16): the
brief cited `tasks/approved/2324/` but the worktree carried the folder at
`tasks/running/2324/` — the branch tree POSTDATES the status move (squash-
merge/rebase after `set-status running`), so the worktree had the CURRENT-
status folder, the opposite direction of the usual #489/#550 frozen-at-cut
class. Blindly inlining would have bloated the prompt with a ~30 KB plan that
was path-referenceable all along.

**Why:** `git mv` status transitions land on main; whether the worktree sees
the old or new path depends on when the branch last absorbed main — both
directions occur in practice.

**How to apply:** when `test -f <worktree>/<plan_marker_path>` fails, run
`ls -d <worktree>/tasks/*/<N>` BEFORE falling back to inlining. If the folder
exists under another status dir, use that path — the content-identity diff
against `task.py find <N>` output still binds unchanged (on #2324 all three
grounding docs — plans/v4.md, body.md, concerns.jsonl — were byte-identical).
Tell Codex the corrected path explicitly and note the brief's path does not
exist, or it may report a spurious unreachable-plan lens. Related:
[[concurrent-followups-wrong-plan-symlink]].

**Fourth case — brief CLAIMS stale, probe shows identical (#2261 r1,
2026-08-21):** a brief may order "INLINE — the worktree copy is frozen at
base and would be stale" when the branch was in fact cut AFTER the plan
version landed on main, so the worktree copy diffs byte-identical. Run the
identity diff anyway; when identical, still FOLLOW the brief's INLINE
instruction (strictly more robust, and the brief is the extraction
contract) but write ACCURATE envelope wording — say the worktree copy was
verified identical, never assert staleness the probe refuted (a false
staleness attestation invites Codex to distrust a perfectly good
cross-check path) — and flag the divergence in the composer's return text.
