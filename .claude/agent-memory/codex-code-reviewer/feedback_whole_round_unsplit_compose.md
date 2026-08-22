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
   **HEAD-side variant (#2184 r1):** when out-of-scope commits sit ON TOP of
   the feature commits (spec-freshness syncs from origin/main after the
   round's last feature commit), HEAD itself is out of scope — pin
   `git diff <parent>..<last-feature-sha>` and ban `..HEAD` / `...HEAD`
   BODY forms entirely; verify `merge-base(origin/main, HEAD) == <parent>`
   still holds and name the excluded sync SHAs in the compose-time facts so
   Codex never flags their spec churn.
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
6b. **Pre-split single-marker variant (#2379 r1):** a head reading
   "round 1 (pre-split build, units 1-4 of 4)" with a body stating "covers
   the WHOLE round" is NOT the item-6 thin-final-unit shape — probe
   events.jsonl for `note.startswith('[unit ')` rows (0 hits ⇒ skip the
   progress-notes envelope) and tell Codex the marker IS the full-round
   report so "units 1-4 of 4" is never misread as partial coverage.
6. **Multi-unit rounds: only the FINAL unit posts `epm:results`** (#2168 r1:
   note head "unit 3 of 3 (FINAL)"; units 1-2 posted `[unit k/N]`
   `epm:progress` notes). Two duties, applicable to ANY round whose fetched
   marker head matches `unit \d+ of \d+`: (a) inline the earlier units'
   progress notes in a supplementary `---BEGIN/END UNIT PROGRESS NOTES---`
   envelope (filter events.jsonl on `note.startswith('[unit ')`); (b) tell
   Codex the Step 0.5 gate scores the inlined `epm:results` body and that
   thin early-unit coverage is at most a present-but-imperfect CONCERNS —
   otherwise an adversarial twin reads "unit K of K" as "the report does not
   cover the round" and false-FAILs `marker-shape` (the #489 class in a new
   costume).

**Why:** the whole-round view is the ONLY reviewer seeing commit
interactions; a mis-based diff (origin/main) or a leaked split-token
defeats exactly that purpose.

**How to apply:** any brief carrying `round_parent=` + `round_commits=` +
"whole-round UNSPLIT review" context. Related:
[[revision-round-compose-recipe]], [[worktree-task-folder-status-can-be-stale-in-EITHER-direction]].
