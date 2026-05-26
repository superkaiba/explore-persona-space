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
