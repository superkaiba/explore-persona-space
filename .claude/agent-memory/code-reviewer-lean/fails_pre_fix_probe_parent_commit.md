---
name: fails-pre-fix-probe-parent-commit
description: Fix-verification rounds — certify a "fails-pre-fix" test claim by re-running its probe harness against the PARENT commit's extracted function body (git show <sha>^:<file>); cheap and decisive (#2225 R2 g2)
metadata:
  type: feedback
---

When a fix-verification commit ships a test claiming "FAILS PRE-FIX", do not
take the claim on trust or re-derive it purely by trace: extract the pre-fix
function body from the parent commit (`git show <fix-sha>^:<file>` + the
test's own extraction regex) and run the SAME probe harness against it. A
genuine fix shows the old body failing exactly as the r1 blocker described
(e.g. #2225 R2 g2: pre-fix single-token gate → rc=7 `3/4 logs carry
[steer-hook]`; post-fix → 4/4, eval-gen reached).

**Why:** the harness itself can accidentally encode the fix (stubs emulating
the FIXED behavior can mask a gate that never runs); the parent-commit run
separates "the fix closes it" from "the harness passes anything". Validated
#2225 R2 g2 — ~20 lines of throwaway driver, ran in seconds, upgraded the
verdict's blocker-closure claims from traced to measured.

**How to apply:** in any SPLIT-REVIEW fix-verification round whose test
sed-extracts a real function body, reuse that extraction against the parent
blob in /tmp; assert the r1 failure signature (rc, FATAL line) appears and
the success signal does not. Pairs with [[sentinel-path-outside-drain-glob]]
and [[smoke-enum-item-without-dial]] closure checks (path fnmatch against the
REAL poller glob; dial resolved at runtime, e.g. registry membership of probe
targets).

**Temp-tree contamination trap (#2479 R2 g5):** when the probe builds a /tmp
tree (parent blob + sibling scripts + HEAD tests), REAL-COPY the sibling
scripts — NEVER symlink them. The sibling scripts' own bootstrap does
`sys.path.insert(0, str(Path(__file__).resolve().parent))`, and `.resolve()`
follows the symlink back to the WORKTREE scripts dir, inserting it AHEAD of
the /tmp dir — so the module under test silently imports at HEAD and the
"pre-fix" run passes everything (observed: 5/5 false PASSes). Certify the
probe is clean by printing `<mod>.__file__` + `hasattr(<mod>, <new_fn>)`
from inside the test run (module-level print + `pytest -s`) before trusting
any parent-blob result; a bare `python -c` import check is NOT sufficient —
it skips the sibling imports that do the re-inserting.

**Whole-module variant (#2225 R3):** when the tests load the module by FILE
PATH (`spec_from_file_location` on `scripts/<mod>.py`), the cheapest exact
probe is: `git show <fix-sha>~1:scripts/<mod>.py > scripts/<mod>.py` in the
issue worktree, run HEAD's new tests against it (the parent commit lacks the
tests, so run HEAD's test file, not a parent checkout), read the failure
MODES (each must match the r2/r1 trace, not just "failed"), then restore via
`git -C <worktree> checkout -- <file>` in its OWN Bash call — the repo-root
guard blocks any compound whose TEXT contains `git checkout --`, killing the
swap clause too (#1143 text-match). Confirm restore with an empty
`git diff --stat HEAD -- <file>` before re-running the suite.
