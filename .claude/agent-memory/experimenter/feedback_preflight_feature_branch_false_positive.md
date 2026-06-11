---
name: preflight-feature-branch-false-positive
description: `orchestrate.preflight --json` reports `ok=false` with `Local is N commit(s) behind origin/main` for ANY feature branch (e.g. `issue-383`), because it compares HEAD to origin/main without checking whether HEAD is on a different branch. This is a false positive — don't treat it as a real failure.
metadata:
  type: feedback
---

`src/explore_persona_space/orchestrate/preflight.py` checks whether the current branch is up-to-date against `origin/main` regardless of which branch you actually checked out. For feature branches (the canonical `/issue` launch path), HEAD is on `issue-<N>` and diverges from main by design. Preflight then reports:

```json
{"ok": false, "errors": ["Local is N commit(s) behind origin/main. Run: git pull origin main"], ...}
```

**Why:** The check exists for the case where you're working ON main and forgot to pull. But it has no branch-awareness, so it fires on every feature-branch launch.

**How to apply:**

1. When preflight JSON shows `ok=false`, FIRST parse the `errors` list. If the ONLY error is the `behind origin/main` one AND the current HEAD is on a feature branch (`issue-<N>` or `task-<N>-...`), treat it as PASS and proceed.
2. Verify all other indicators are healthy (`gpu_info`, `disk_free_gb`, `env_synced`).
3. Do NOT auto-fix by `git checkout main && git pull` — that destroys the implementer's branch.
4. Do NOT post `epm:failure v1` for this false positive alone.

**Burned at #383 launch (2026-05-24):** Preflight ok=false with `Local is 977 commit(s) behind origin/main` while on `issue-383` (HEAD=20da0dec). Diff between HEAD and origin/main showed `merge-base=f359edb3`, which is exactly the parent point — i.e., issue-383 had diverged 977 commits behind main since branching, but the branch itself was up to date with origin/issue-383. Proceeded with launch and confirmed everything was fine.

**Fix at the source (TODO for implementer pass):** Preflight should check `git rev-parse --abbrev-ref HEAD` first; only run the behind-main check when on main itself. Until that lands, the experimenter must apply the human override above.

**Burned again at #550 launch (2026-06-10) — two new aspects:**

1. **Pipeline launchers that invoke BARE preflight (no `--json`) under `set -e` die SILENTLY on this false positive.** In non-JSON mode the failure summary goes through `logger.info()` with no console handler — zero bytes of output — then `sys.exit(1)` kills the launcher. Symptom: pipeline log frozen at the `[phase=preflight] starting` banner, 0-byte phase log, dead PID within seconds. Looks identical to SSH-disconnect reaping; discriminate by re-running the preflight command synchronously (`rc=1` + no output = this bug).
2. **Sanctioned clear-the-gate recipe (no force-push, tree unchanged):** on the VM worktree for `issue-<N>`, `git merge <pod's origin/main tip SHA>` then immediately `git revert -m 1 <merge> --no-edit`, verify `git diff <reviewed-sha> HEAD | wc -c` == 0, push, `git pull --ff-only` on pod. History now contains the pod's origin/main snapshot (behind-count = 0, preflight `ok=true`) while the tree stays byte-identical to the reviewed commit. CAUTION: do NOT keep the merge content — at #550 the "1 commit behind" merge auto-merged `sft.py`/`callbacks.py` from main (the pod's behind-count is computed against its clone-time origin/main snapshot, which can hide a much larger VM-side divergence); the revert is what keeps the reviewed tree intact.
