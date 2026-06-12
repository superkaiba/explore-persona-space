---
name: preflight-feature-branch-false-positive
description: orchestrate.preflight reports ok=false "Local is N commit(s) behind origin/main" on ANY feature branch (issue-<N>) — a false positive when that's the only error. Worse, launchers running BARE preflight (no --json) under set -e die SILENTLY on it.
metadata:
  type: feedback
---

`orchestrate.preflight` compares HEAD to origin/main with no branch-awareness, so every `issue-<N>` / `task-<N>-...` launch reports `ok=false, "Local is N commit(s) behind origin/main"`. When that is the ONLY error and HEAD is the implementer's commit, treat as PASS and launch — verify the other indicators (`gpu_info`, `disk_free_gb`, `env_synced`) are healthy. Do NOT `git checkout main && git pull` (destroys the branch) and do NOT post `epm:failure` for this alone.

**Why:** burned at #383 (2026-05-24, "977 commits behind" on a healthy branch) and #550 (2026-06-10).

**#550 variants:**
1. **Silent launcher death:** pipeline launchers invoking BARE preflight (no `--json`) under `set -e` die with ZERO output — non-JSON failure goes through `logger.info()` with no console handler, then `sys.exit(1)`. Symptom: log frozen at the `[phase=preflight] starting` banner, 0-byte phase log, PID dead in seconds — looks like SSH-disconnect reaping. Discriminate by re-running the preflight command synchronously (rc=1 + no output = this bug). The pre-clear protocol now lives in `experimenter.md` ("Pre-clear the false positive for launchers that re-run preflight").
2. **Sanctioned clear-the-gate recipe (no force-push, tree unchanged):** on the VM worktree, `git merge <pod's origin/main tip SHA>` then immediately `git revert -m 1 <merge> --no-edit`; assert `git diff <reviewed-sha> HEAD | wc -c` == 0; push; `git pull --ff-only` on the pod. Behind-count becomes 0 while the tree stays identical to the reviewed commit. CAUTION: never keep the merge content — at #550 it auto-merged sft.py/callbacks.py from main; the revert is what protects the reviewed tree.

Source fix (still TODO as of 2026-06-12): preflight should gate the behind-main check on `git rev-parse --abbrev-ref HEAD == main`.
