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

**Zero-mutation dual-load variant (#2321 R2 g4):** when the target is a
standalone script whose deps resolve from the current venv, skip the
worktree swap entirely: `git show <round-parent>:scripts/<mod>.py >
/tmp/<mod>_parent.py`, then in ONE python process importlib-load BOTH
parent and HEAD as differently-named modules and run the same adversarial
scenario against each (e.g. C3: build a real pack, tamper one index-part
offset, expect parent ACCEPTS / HEAD raises AbortPrefix). Two-sided in one
run — hole-present-pre-fix AND fix-engaged-at-HEAD — with no working-tree
mutation and no restore step. Also the cheapest way to independently settle
a reviewer-vs-reviewer disagreement about what the pre-fix code did (the
#2321 C3 adjudication: probe, don't re-read).

**Pick a tamper the parent's AGGREGATE checks don't already catch (#2321 R2
g2):** a naive "add a ghost entry" tamper was caught by the PARENT's
`n_members != manifest` count check — the probe initially "failed pre-fix"
for the wrong reason. The true pre-fix hole needed a COUNT-PRESERVING
mutation (rename one index key to a ghost path; bump one offset). When
certifying tamper tests, prefer count-preserving/aggregate-invariant
mutations, and read WHICH check fired on the parent, not just that one did.
