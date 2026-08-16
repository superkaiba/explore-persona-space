---
name: whole-round-unsplit-compose
description: "#2074 split rounds: Codex gets the WHOLE-ROUND brief — base is the round-parent SHA (never origin/main), strip the Step-0 split-review paragraph (its literal trigger token must not enter the prompt), and over-300KB rounds get per-file reads with committed data artifacts digest-only"
metadata:
  type: feedback
---

When the /issue Step 5 round is split-reviewed on the Claude side (#2074
per-commit sub-reviews), the Codex twin's brief is a WHOLE-ROUND UNSPLIT
review — the deliberate catching arm for cross-commit interaction bugs.
Compose deltas vs an ordinary round (first hit: #2330 r1, 2026-08-16):

1. **Base is the brief's `round_parent` SHA, not origin/main.** Verify
   `git -C <wt> merge-base <parent> HEAD == <parent>` at compose time, then
   pin `git diff <parent>..HEAD` in the prompt and BAN main/origin-main
   body diffs (main-side drift pollutes them — the brief usually says so).
   Tell Codex to record `sha-range <parent>..HEAD` in Diff acquisition.
2. **Strip the copied Step 0 "Split-review sub-scope briefs (#2074)"
   paragraph.** Copying it verbatim puts the literal trigger token
   `SPLIT-REVIEW SUB-...` INTO the prompt, arming split-mode behavior
   (write-to-file, skip contract gates) the whole-round review must not
   take. Validate post-compose that the token is absent.
3. **Over-300KB rounds: per-file read strategy in the prompt.** Measure
   `git diff <parent>..HEAD | wc -c` and per-file sizes at compose time;
   scripts get read-every-line per-file diffs, committed DATA artifacts
   (large JSONs) get structural-digest-only instructions (head -c, grep -c
   keys, wc -l) against plan + consumer assumptions.
4. **Leak-validation gotcha:** the adaptation note "the `git stash push`
   alternative is OMITTED" itself re-introduces the literal your own
   validation greps for — word the note without the literal.
5. **Tell Codex to prioritize cross-commit checks** the split reviews
   structurally cannot see (constant defined in one commit / consumed at a
   different grain in another; waivers detached by later refactors;
   committed-artifact grain vs consumer assumptions).

**Why:** the whole-round view is the ONLY reviewer seeing commit
interactions; a mis-based diff (origin/main) or a leaked split-token
defeats exactly that purpose.

**How to apply:** any brief carrying `round_parent=` + `round_commits=` +
"whole-round UNSPLIT review" context. Related:
[[revision-round-compose-recipe]], [[worktree-task-folder-status-can-be-stale-in-EITHER-direction]].
