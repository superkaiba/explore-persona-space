---
title: 'workflow-fix: test_backend_poll module-mode test fails only in large collections,
  always classified NEW by the Step 9c oracle'
kind: infra
tags:
- wf-fix
created_at: '2026-08-07T13:33:58Z'
has_clean_result: false
origin_prompt: 'Surfaced in #2164 Step 9c: compare --run-pristine classified test_ensure_scripts_dir_bootstrap_resolves_runpod_api_in_module_mode
  as NEW because the oracle runs candidate files individually and the failure is collection-scale.
  Disproven by running the identical 220-file selection on a pristine origin/main
  scratch worktree (same 2 failures both sides).'
workflow: v1
---
## Overview / Motivation

`tests/test_backend_poll.py::test_ensure_scripts_dir_bootstrap_resolves_runpod_api_in_module_mode`
passes standalone and **fails inside large pytest collections** (~189+ files).
It therefore reds essentially every `/issue` Step 9c gate run fleet-wide, and
each session must re-classify it by hand.

Worse, it is **invisible to the gate's own oracle**.
`step9c_baseline.py compare --run-pristine` runs each candidate file
individually, so the test always passes there and is always classified **NEW** —
i.e. blamed on whatever branch is being gated. Clearing it requires a manual
full-collection repro on pristine main (~43 min) every single time.

## Evidence (#2164 Step 9c, 2026-08-07)

Identical 220-file selection, same invocation, two trees:

```
branch issue-2164 (cbca2dd08f):  2 failed, 10567 passed, 14 skipped  (40m53s)
pristine main    (3c643ba7e2):   2 failed, 10561 passed, 14 skipped  (43m25s)
```

Same test fails in both ⇒ pre-existing, not diff-caused. Independently reached
earlier by the #2164 round-1 code-reviewer, which reverted the test file to
`origin/main`'s version and still saw the failure under full collection.

## Goal

Make the test's outcome independent of collection size, so it stops producing a
false NEW on every gated branch.

## Diagnosis (from the #2164 round-1 review, needs confirming)

Some module's **collection-time `sys.path` insert** makes `runpod_api`
importable from outside the scrubbed `scripts/` directory, which defeats what
the test is asserting. The test checks that
`ensure_scripts_dir_bootstrap` resolves `runpod_api` in module mode; when an
unrelated module has already polluted `sys.path`, the resolution the test wants
to prove is doing the work happens for the wrong reason.

Confirm the mechanism before fixing — identify the specific module whose import
side effect leaks, rather than patching the symptom.

## Proposed change

Preferred: harden the test to scrub by **module name** (drop any already-imported
`runpod_api` from `sys.modules` and remove the offending `sys.path` entries)
so it asserts the same invariant regardless of what ran before it.

Better if cheap: fix the **leaking module** so it does not mutate `sys.path` at
import/collection time. A collection-order-dependent `sys.path` mutation is a
latent hazard well beyond this one test.

Do not fix by marking the test `xfail` or excluding it from the gate selection —
that removes the signal instead of the flakiness.

## Acceptance criteria

- The test passes both standalone and inside the full ~220-file Step 9c
  selection.
- The specific leaking module (or the scrub that neutralizes it) is named in the
  fix, not just "cleaned up sys.path".
- A regression guard that would catch reintroduction — e.g. asserting the
  relevant `sys.path` / `sys.modules` state at test entry — so the next
  collection-order change does not silently restore the bug.
- No `xfail`, no skip, no removal from the selector.

## Related

Sibling of the gate-blindness note recorded on **#2166**: a fresh ledger
reported `failing_tests: 1` while 5 other tests were red on main, so the
ledger's universe does not cover every pin family either. Different mechanism
(ledger scope vs per-file oracle), same consequence — the gate cannot see a
main-red test and misattributes it to the branch under review.
