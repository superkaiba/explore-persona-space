---
name: claude-misses-tracked-artifact-lifecycle-across-phases
description: "#2378 r5: Claude PASSed after verifying a new tracked artifact's harvest inclusion at the PRODUCING phase only; Codex's real Critical was the LATER phases rewriting the now-tracked file (volatile run_metadata) while their scoped git_harvest omitted it — plain rebase (no autostash on pod clones) then refuses deterministically"
metadata:
  type: feedback
---

When a round INTRODUCES a persisted artifact that a phase COMMITS to the issue
branch (e.g. `ensure_model_venv`'s `model_venv_pins.json`, committed by P1's
`git_harvest`), adjudicate the artifact's WHOLE cross-phase lifecycle, not just
the producing phase's harvest inclusion:

1. Grep every later call site of the WRITER (each phase-entry re-ensure) — an
   unconditional `atomic_write_json` embedding `run_metadata()` (timestamp +
   argv + git-dirty flag) rewrites DIFFERENT bytes every invocation, dirtying
   the tracked file on every later pod/phase.
2. Diff each later phase's `git_harvest` path list against the artifact path —
   a scoped harvest that omits it commits fine (pathspec commit) but then runs
   `git rebase origin/<branch>`.
3. Pod clones have NO `rebase.autoStash` (bootstrap `git init` configures only
   credential helper/promisor/sparse-checkout; the VM repo-root autostash
   config does not travel). LIVE-PROBED (git 2.34.1, #2378 r5): the rebase
   refuses `cannot rebase: You have unstaged changes` in ALL states —
   up-to-date, ahead-by-1 after the harvest's own commit, and behind. There is
   NO self-healing ordering; the crash lands at the harvest, AFTER the phase's
   GPU work, and can even shadow a designed partial-result stop sequenced
   after the harvest.

**Why:** Claude's Step 0.65 line read "the new artifact was added to the p1
git_harvest list in the same change" — true, and exactly the trap: the
producing phase is self-consistent (writes then commits), so per-phase review
reads clean while the round's own diff creates the cross-phase contract
violation. Codex re-derived the P1→P2→P4 pod-A/pod-B sequence and caught it.

**How to apply:** on any pins-record / ledger-file / status-file written by a
shared entry gate AND committed by one phase's harvest, run the 3-step trace
above yourself. Reproduce the git behavior in a scratch repo (literal-path
`git -C`, no `cd` — the repo-root guards fail-closed on unprovable cd targets)
rather than reasoning about whether rebase tolerates dirt. Fix shapes:
content-stable write (skip when non-metadata payload unchanged — also covers
`git pull --rebase` consumers) > add the path to every post-ensure harvest >
`--autostash`. Related: [[split-review-misses-cross-commit-plan-contracts]],
[[claude-misses-same-file-siblings]].
