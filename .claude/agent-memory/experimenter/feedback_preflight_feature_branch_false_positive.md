---
name: preflight-feature-branch-false-positive
description: "FIXED at source by #554 (2026-06-12): preflight is now branch-aware — behind-origin/main on an issue-<N> branch is a WARNING, not an ERROR. The tolerance/pre-clear override applies ONLY to pods still running pre-#554 code. On current code, behind-origin/issue-<N> and git-fetch-failed ERRORs are REAL."
metadata:
  type: feedback
---

**FIXED at source by #554 (2026-06-12, commit `25f227273`).**
`orchestrate.preflight` is now branch-aware: a feature branch is compared
against its OWN `origin/<branch>` ref; `Local is N commit(s) behind
origin/main` on an `issue-<N>` checkout is an informational WARNING, not
an ERROR; bare (non-`--json`) preflight fails loud (summary on stdout,
per-error stderr lines) instead of dying silently. On post-#554 code the
override below is NOT needed — and a `Local is N commit(s) behind
origin/issue-<N>` or `git fetch origin failed` ERROR is REAL (missing
reviewed commits / broken fetch): never tolerate it, never `epm:failure`-
skip it as "the known false positive."

**Legacy override (pods still running pre-#554 code only).** Pre-fix
`orchestrate.preflight` compared HEAD to origin/main with no
branch-awareness, so every `issue-<N>` / `task-<N>-...` launch reported
`ok=false, "Local is N commit(s) behind origin/main"`. When that is the
ONLY error and HEAD is the implementer's commit, treat as PASS and launch
— verify the other indicators (`gpu_info`, `disk_free_gb`, `env_synced`)
are healthy. Do NOT `git checkout main && git pull` (destroys the branch)
and do NOT post `epm:failure` for this alone.

**Why:** burned at #383 (2026-05-24, "977 commits behind" on a healthy
branch) and #550 (2026-06-10).

**#550 variants (both closed by #554's fail-loud bare mode + branch-aware
check; recipes kept for pre-#554 pods):**
1. **Silent launcher death:** pipeline launchers invoking BARE preflight
   (no `--json`) under `set -e` died with ZERO output — non-JSON failure
   went through `logger.info()` with no console handler, then
   `sys.exit(1)`. Symptom: log frozen at the `[phase=preflight] starting`
   banner, 0-byte phase log, PID dead in seconds — looks like
   SSH-disconnect reaping. Discriminate by re-running the preflight
   command synchronously (rc=1 + no output = this bug). The pre-clear
   protocol lives in `experimenter.md` ("Pre-clear the false positive for
   launchers that re-run preflight"), now scoped LEGACY.
2. **Sanctioned clear-the-gate recipe (no force-push, tree unchanged):**
   on the VM worktree, `git -C <worktree> merge <pod's origin/main tip SHA>`
   (the `-C <worktree>` form is required — a bare `git merge` is
   hook-blocked, #1128) then
   immediately `git -C <worktree> revert -m 1 <merge> --no-edit` (the
   `-C <worktree>` form is required here too — a bare `git revert` is
   hook-blocked, #1234); assert
   `git diff <reviewed-sha> HEAD | wc -c` == 0; push; `git pull --ff-only`
   on the pod. Behind-count becomes 0 while the tree stays identical to
   the reviewed commit. CAUTION: never keep the merge content — at #550 it
   auto-merged sft.py/callbacks.py from main; the revert is what protects
   the reviewed tree.
