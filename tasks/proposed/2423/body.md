---
title: 'Step 5a sibling sync: the #2208 satisfiability probe never runs on a SCRIPT-only
  sync, so a synced sibling script''s branch-era-unsatisfiable src/ import reds the
  gate'
kind: infra
tags:
- wf-fix
created_at: '2026-08-20T12:30:46Z'
has_clean_result: false
parent_id: 2212
origin_prompt: 'Surfaced during /issue 2212 Step 10d: pre-push lint gate returned
  verdict=crash from a TG collection ImportError in tests/test_issue2094_figures.py,
  caused by the sibling-file sync syncing scripts/issue2094_figures.py forward without
  its src/ import closure; the #2208 probe never ran because it keys on synced TEST
  files and no 2094 test was synced.'
workflow: v1
---
# Step 5a sibling sync: the #2208 satisfiability probe never runs on a SCRIPT-only sync, so a synced sibling script's branch-era-unsatisfiable `src/` import reds the gate

## Goal

Close the script-only hole in the #2208 import-satisfiability probe. The probe
that guards Step 5a's sibling-issue file sync is keyed on synced **TEST** files
only:

```bash
for f in "${SIBLING_SYNCED[@]}"; do
  case "$f" in
    tests/test_issue*_*.py)
      if ! (cd "$WT" && timeout --kill-after=15s 180s uv run pytest --collect-only -q "$f" >/dev/null 2>&1); then
```

(`.claude/skills/issue/steps/09-step-5.md`, the block around L365-L390.)

The enumeration that FEEDS `SIBLING_SYNCED`, however, globs scripts as well as
tests, and lists only files that DIFFER from `origin/main`:

```bash
done < <(git -C "$WT" -c core.quotePath=false diff --name-only origin/main \
  -- ':(glob)scripts/issue[0-9]*_*.py' ':(glob)scripts/issue[0-9]*_*.sh' \
     ':(glob)tests/test_issue[0-9]*_*.py')
```

So whenever a sibling issue's SCRIPT has drifted from the branch but its
covering TEST is already identical to `origin/main`, the script is synced and
**no test enters `SIBLING_SYNCED` for that issue number — so no probe runs at
all**. The synced script can then carry a module-level `src/` import added to
`main` after this branch's fork point, and the covering (unsynced) test dies at
COLLECTION, exactly the #2206/#2208 symptom the probe exists to prevent.

## Measured incident (task #2212, 2026-08-20)

Step 10d's pre-push lint gate returned verdict `crash` — fail-closed, merge
blocked — on a branch whose own five deliverable files were clean (the lint
legs' baseline and gated normalized outputs were byte-identical at 380 bytes /
3 lines each; `lint-new` and `lint-owndiff` both empty).

The crash came from the TG mapped gated leg:

```
ERROR collecting tests/test_issue2094_figures.py
  tests/test_issue2094_figures.py:27: in <module>
      import issue2094_figures as F
  scripts/issue2094_figures.py:65: in <module>
      from explore_persona_space.analysis.paper_plots import (
E ImportError: cannot import name 'figsize_iclr_full' from
  'explore_persona_space.analysis.paper_plots'
Interrupted: 1 error during collection
```

Evidence that this is the script-only shape, not #2208's already-fixed
test-sync shape:

| Fact | Probe |
|---|---|
| `tests/test_issue2094_figures.py` is **NOT** in the branch own-diff (0 paths) | `git diff --name-only $MB...HEAD -- tests/test_issue2094_figures.py` |
| It was already identical to `origin/main` at the merge-base — so it never entered the sync enumeration | `git diff --quiet $MB origin/main -- tests/test_issue2094_figures.py` → clean |
| `scripts/issue2094_figures.py` **IS** in the branch own-diff — it WAS synced | `git diff --name-only $MB...HEAD -- scripts/issue2094_figures.py` → 1 path |
| The sync that introduced it is the sibling-file arm | `git log $MB..HEAD -- scripts/issue2094_figures.py` → `e10ad8b8b4 ... (spec-freshness; sibling-issue files)` |
| main's script carries the module-level import at L66; the merge-base script had **zero** references | `git show origin/main:scripts/issue2094_figures.py` vs `git show $MB:...` |
| The branch never authored `paper_plots.py` — it sits at merge-base vintage, which lacks `figsize_iclr_full` | 0 own-commits touching it; `grep -c 'def figsize_iclr_full'` → main 1, branch 0, merge-base 0 |

Cost: one full ~45-minute gate run consumed for a `crash` verdict attributable
to sync machinery rather than to the branch payload, plus the diagnosis round.
The gate's own single-re-run budget was then spent on the remediation.

## Why this is NOT already covered

- **#2208** (completed) added the probe for the case where the synced sibling
  **TEST** is main-NEW with unsatisfiable `src/` imports. Here the test was not
  synced at all, so its `case` arm never matched.
- **#2412** (running) reports the probe is `--collect-only` and therefore misses
  runtime API skew and function-body imports. This incident is the opposite
  direction: the break IS collect-time, so `--collect-only` would have caught
  it — the probe simply was never invoked. Fixing #2412 does not fix this.
- The pair-atomic revert (#1824/#1860) triggers only from a probe FAILURE, so it
  never fires either.

Step 10d's own prose already names the mechanism as a known hazard — "syncing
arbitrary individual test files without their import closure — conftest, tests/
helpers — risks hybrid trees" — but the mitigation was scoped to synced tests.

## Candidate fixes (for the planner to adjudicate — not pre-decided)

1. **Key the probe on the ISSUE NUMBER, not the synced file's kind.** For every
   distinct `<M>` appearing in `SIBLING_SYNCED`, probe every existing
   `tests/test_issue<M>_*.py` in the worktree, whether or not that test was
   itself synced. Smallest change; reuses the existing pair-atomic revert.
   Cost: one `--collect-only` per covered issue rather than per synced test.
2. **Sync the `src/` import closure alongside a synced sibling script.** More
   faithful to what `main` actually looks like, but widens the sync's blast
   radius into `src/` and needs its own lost-update reasoning — a `src/` module
   is not workflow surface, and blindly checking it out could clobber a genuine
   branch edit (the family-atomic dirty check would have to extend to it).
3. **Revert a synced sibling script whose module-level imports are
   unsatisfiable**, probed directly (`python -c 'import <module>'` under the
   worktree venv) rather than via its covering test. Catches the script even
   when no covering test exists.

Option 1 is the least invasive and closes the measured incident; option 3
generalizes to script/no-test cases. Deciding between them (and whether to do
both) is the plan's job.

## Acceptance criteria

1. A worktree in which only a sibling SCRIPT has drifted (its covering test
   identical to `origin/main`) and whose synced script carries a
   branch-era-unsatisfiable module-level `src/` import either (a) does not
   commit that sync, or (b) commits it together with a closure that makes
   collection succeed — verified by a test reproducing the #2212 shape above.
2. The existing #2208 synced-test arm and the pair-atomic revert keep their
   current behavior (no regression in `tests/` pins covering them).
3. The fail-safe direction is preserved: a probe failure or timeout results in
   status-quo staleness, never a committed hybrid tree.

## Provenance

Surfaced during `/issue 2212` Step 10d, 2026-08-20, from a real `crash` verdict
on the pre-push workflow-lint gate. Remediated in-branch for #2212 by syncing
the import closure (`src/explore_persona_space/analysis/paper_plots.py`, made
byte-identical to `origin/main`; commit `33f4fb2236`) — that unblocked #2212 but
does nothing for the next branch to hit this.
