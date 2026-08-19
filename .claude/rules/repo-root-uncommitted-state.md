# Repo-root uncommitted state — the pre-commit stash race (#2015)

**Load this rule whenever you leave tracked files modified/deleted-but-
uncommitted at the shared repo root, see root files revert/resurrect within
seconds, or need to verify a commit landed under concurrency.** Mechanism
verified against installed pre-commit 4.6.0
(`~/.local/share/uv/tools/pre-commit/lib/python3.11/site-packages/pre_commit/`);
deterministic reproduction: `scripts/repro_precommit_stash_race.sh` (run it
after any pre-commit upgrade — it is the upgrade probe; committed evidence:
task #2015 `artifacts/repro_output.txt`).

## The mechanism (file:line, pre-commit 4.6.0)

- `pre_commit/commands/run.py:344` — `stash = not args.all_files and not
  args.files`; the generated `.git/hooks/pre-commit` passes neither flag, so
  the stash cycle is armed on EVERY fleet commit.
- `staged_files_only.py:53-56` — `git write-tree` + `git diff-index
  --binary --exit-code <tree> --` captures the **repo-wide** unstaged
  tracked diff — every session's, not just the committer's — into
  `~/.cache/pre-commit/patch<epoch>-<pid>`.
- `staged_files_only.py:57-61` — retcode 0 (no unstaged tracked diff) ⇒
  `yield` with **no stash, no checkout, no race**. A CLEAN tracked tree
  disarms the whole cycle — the self-eliminating property the guidance +
  watcher pass rely on.
- `staged_files_only.py:23,81` — `git -c submodule.recurse=0 checkout -- .`
  runs BEFORE the hooks: the repo-wide transient reversion (seconds to
  minutes, the hook-suite wall).
- `staged_files_only.py:85-96` — the `finally:` restore re-applies the
  patch; on `CalledProcessError` it logs `Rolling back fixes`, runs the
  checkout **a second time**, and re-applies only its own STALE snapshot —
  the permanent-loss path for any write that landed inside the window. In
  THIS repo all mutating hooks are `stages: [manual]`, so an apply conflict
  can only come from a concurrent writer: that branch is pure loss here.

## The three race shapes (+ measured commit-rc nuance)

1. **Transient reversion** — every unstaged tracked edit reverts to index
   content for the hook window, restores after.
2. **Permanent loss** — a write landing inside another commit's window that
   conflicts with the stashed hunk is DESTROYED by the double-checkout
   rollback. Measured (repro S2): the concurrent commit itself exits rc=1 —
   pre-commit's post-hook file-hash check reads the mid-window write as a
   hook auto-fix ("files were modified by this hook") — but the loss
   happens regardless.
3. **Deletion resurrection** — an unstaged `rm` of a tracked file is
   resurrected (HEAD content) for the window, deleted again after.

Clean-tree residual (repro S4): with a clean tracked tree NO patch file is
created and a mid-window write SURVIVES verbatim; the concurrent COMMIT can
still fail rc=1 via the same post-hook check — retryable, never data loss.

## Risk surface

| Working-tree state | Exposed? |
|---|---|
| Tracked, modified, unstaged | YES — reverted every window; lost on restore-conflict |
| Tracked, deleted, unstaged | YES — transiently resurrected |
| Untracked NEW file | NO — invisible to `diff-index` and `checkout -- .` (repro canary) |
| Staged-but-uncommitted | Does NOT arm the stash, but exposed to the #1894 bare-commit sweep (CLAUDE.md § Concurrent repo-root committers) |

## The discipline (why "write→add→commit in one short window")

Never leave tracked files modified/deleted-but-uncommitted at the root
beyond the current turn: generate artifacts off-root (worktree /
`/mnt/eps-data` staging) and copy in immediately before the add; stage
deletions the moment you make them. **Agent-memory writes
(`.claude/agent-memory/**`) are the fleet's recurring armer class** — 8 of
the 14 files in the #2015 standing diff, written continuously across ~23
sessions with no natural committer, so the tree RE-ARMS after any one-time
cleanup: the session whose agent wrote a memory file commits that file by
explicit path in the SAME turn. Without that, the steady state is
alert-driven cleanup and the watcher pass becomes a recurring-alert channel
sessions learn to ignore.

## Landing verification under concurrency

A FAILED commit (rc≠0, "modified files") can coincide with a SUCCEEDING
push of ANOTHER session's SHA — never verify by the push line. Verify by
blob read at the specific SHA:

```bash
git fetch origin main
git show <sha>:<path> | head        # the content you meant to land
git log -1 --format=%H origin/main -- <path>   # which commit last touched it
```

## Diagnostic tell (this race vs a live writer)

The tell is a CONJUNCTION: content converges back to HEAD within seconds
**AND** `~/.cache/pre-commit/patch<epoch>-<pid>` files bracket the window.
A live writer instead leaves a monotonically growing file set + a live pid.
**Negative arm:** convergence-to-HEAD *without* bracketing patch files is
NOT this mechanism — it points at git's own `--autostash` (the
`sync_repo_root.py` rejected-push recovery, #2182), which writes git stash
entries rather than patch files and fires per-recovery, not per-commit.
Reviewer-side: a reviewer polling `git status` on a parked task can see
cells flip deleted→modified→clean purely from this mechanism — do not
diagnose a live writer from that signature alone.

**Double-apply-failure residue:** when the restore's FIRST `git apply`
raises AND the post-rollback re-apply (`staged_files_only.py:96`) ALSO
raises, the exception propagates with the patch never applied — the tree
stays reverted (edits gone, deletions resurrected) and the `patch*` file on
disk is the sole rescue surface. The #1806 `stash_rescue_audit_pass`
(watcher pass 34) is the standing recovery channel for exactly that
residue.

## Watcher pass 36 (`root_unstaged_audit_pass`) contract

ESCALATE-ONLY, hourly (`EPM_ROOT_UNSTAGED_INTERVAL_HOURS`): one read-only
`git status --porcelain=v1 -z -uno --ignore-submodules`, keeps worktree
column `M`/`D`, and alerts when an entry is present in TWO collections
≥ `EPM_ROOT_UNSTAGED_MIN_AGE_MINUTES` (30 min) apart — a standing armer,
never a genuine in-flight write→commit window. ONE deduped push per
fingerprint episode (re-alert `EPM_ROOT_UNSTAGED_REALERT_HOURS`, 24 h);
sidecar `.claude/cache/root-unstaged-audit-events.jsonl`. NEVER mutates
git/filesystem state, never auto-commits or auto-restores, posts no task
markers; kill switch `EPM_DISABLE_ROOT_UNSTAGED_AUDIT=1`;
`--root-unstaged-audit-only --dry-run` is the zero-write live smoke.

## Rejected levers (one line each; full table: #2015 plan §4 D7)

Suppressing the stash (`--files` semantics) is fail-OPEN for the
worktree-reading secret/lint hooks (only gitleaks is index-native) — the
task's fail-closed constraint forbids it; flock commit-serialization cannot
stop a plain WRITE landing in another session's hook window (the dominant
loss shape); patching vendored pre-commit is upgrade-fragile and forbidden;
an auto-commit/auto-restore janitor can land mid-write inconsistent pairs
or destroy sibling work — surfacing stays escalate-only.

## Files of record

Task #2015 (plan + repro evidence); #1768 (the five red-handed reversions);
#1806 (stash-rescue audit, pass 34); #1751/#1870 (Step 10d KEPT-stash
surfacing); #2182 (`sync_repo_root.py` autostash sibling); #897
(`guard_repo_root_branch.sh` — structurally cannot see pre-commit's
subprocess checkout); CLAUDE.md § Concurrent repo-root committers (the
always-on summary); `.claude/skills/issue/SKILL.md` § 9a-ter
"Uncommitted-exposure window" (the inline-round copy).
