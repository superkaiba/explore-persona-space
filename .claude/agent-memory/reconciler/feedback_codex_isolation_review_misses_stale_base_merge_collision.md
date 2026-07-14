---
name: Codex reviews diff in isolation against merge-base, misses stale-base merge collision + concurrent-main regression
description: code-review reconcile — when the branch was authored on a stale merge-base, Codex's isolation review (vs merge-base, no `git merge` attempt) misses the collision with concurrent main; run the merge yourself + grep merge-target for dropped guards/new constants
type: feedback
---

When the Claude code-reviewer FAILs a code change for a **stale-base merge
collision** and Codex returns CONCERNS/PASS, the disagreement usually
resolves to Claude — Codex reviewed the diff in ISOLATION against the
merge-base, never tested it against the concurrent merge TARGET.

**Why:** Codex's code-review is diff-vs-merge-base by construction. It does
NOT run `git merge --no-commit origin/main`, so it cannot see (a) a content
conflict in a file both the branch and concurrent main touched, (b) a
load-bearing guard the branch's rewrite DROPS that a sibling PR added to
main since the branch forked, or (c) a new constant/import the merged tree
will require that the stale branch lacks (→ `ImportError` on the package).
A green LOCAL test suite on the branch worktree is meaningless here — the
worktree predates the concurrent change, so it has no test exercising the
new code path.

**How to apply (mechanical, do this yourself in Step 2):**
1. `git merge-base HEAD origin/main` + `git log origin/main --oneline | grep <recent-issue#>` — is the merge-base stale relative to a sibling PR touching the SAME file?
2. `git -C <worktree> merge --no-commit --no-ff origin/main` (stash first; the `-C <worktree>` form is required — a bare `git merge` is hook-blocked, #1128) → `git diff --name-only --diff-filter=U`; `git -C <worktree> merge --abort`. A real conflict in a production-path file is Real-blocking.
3. `grep -c '<dropped-guard-expr>' <file>` on branch vs `git show origin/main:<file> | grep -c '<dropped-guard-expr>'` — branch=0/main>0 means the branch's rewrite silently dropped a guard concurrent main added.
4. `grep -c '<new-constant>' <file>` branch vs main — branch=0/main>0 + the constant is imported in `__init__.py`/`__all__` ⇒ merged-tree import break.

If all four confirm, side with Claude — FAIL, severity preserved. Persist a
BLOCKER concern (deferred-production-path duty): the merged GCP/router path
either crashes on import or silently mis-routes.

**Confirmed instance:** #680 r1. Branch `_gcp_ladder_specs` (length-aware
ladder) authored on merge-base `d0675aee71`, which predates #677
(`a5cda2f8f2` on origin/main). #677 added the `cpu-bigmem` intent + the
load-bearing `if base.gpu_count == 0: return [...]` CPU short-circuit
(without it `_is_short_job` floors gpu_count→1 and appends a spot GPU rung
to a CPU-only analysis job) + the `ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD`
constant (defined, used in a terminal route, exported in `__all__`). Branch
had 0 hits for both; `git merge` conflicted in router.py. Codex explicitly
"did NOT attempt git merge" → CONCERNS (stale comments only). FAIL.
