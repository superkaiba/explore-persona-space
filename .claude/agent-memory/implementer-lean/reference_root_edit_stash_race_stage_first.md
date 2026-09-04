---
name: root-edit-stash-race-stage-first
description: Repo-root script edits — stage immediately or concurrent commits' pre-commit stash windows revert them (renders run stale bytes, inline lint gate reads INCONCLUSIVE)
metadata:
  type: reference
---

Editing tracked `scripts/*.py` at the shared repo root (figure-regeneration
rounds): every concurrent fleet commit's pre-commit stash cycle reverts
UNSTAGED tracked edits repo-wide for its hook window (#2015). Observed
2026-09-03 (paper-figure flattening round), three distinct bites in one hour:

1. A render launched after a passing `grep -q` on the edit still EXECUTED
   pre-edit bytes (file reverted mid-window) — outputs came back with the old
   geometry at rc=0. Detect by checking the sidecar record
   (`exported_size_inches`, `text` list) against the intended change, never
   the exit code.
2. `inline_lint_gate.py` returned `INCONCLUSIVE (edited during gate)` for
   every payload file — the stash cycle flipped their hashes mid-gate.
3. What looked like the F401 PostToolUse hook re-stripping a fresh import was
   actually the race restoring an older file state.

Fix that worked: `git add` the edited scripts IMMEDIATELY (worktree == index
⇒ `checkout -- .` is a no-op for them and they leave the unstaged diff), then
re-run the gate (PASS), then `git commit -F <msgfile> -- <paths>`. After the
commit, renders are immune (revert target == HEAD == your edits).

Also from the same round:
- The root-commit guard demands inline-lint-gate certification for ANY
  repo-root `scripts/` commit — budget ~5-10 min per gate run (load-wait up
  to 300 s at fleet load > 20); payload path must be round-unique
  (`/tmp/issue-<N>-<slug>-inline-payload.txt`).
- Bare `/tmp/<name>.txt` commit-message files can vanish mid-round (tmp
  sweeps); recreate just before `git commit -F`, and expect rc=128
  `could not read log file` if stale.
- `legend_kicker` uppercases its text: presence checks against the sidecar
  `text` list must be case-insensitive (`'QWEN' in t.upper()`).
