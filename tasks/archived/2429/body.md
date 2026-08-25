---
title: 'workflow-fix: issue worktrees run an unpinned Python (19/25 at 3.12.13 vs
  root 3.11.15) — Step 9c verdicts flip on interpreter'
kind: infra
tags:
- wf-fix
- from-2214
created_at: '2026-08-20T21:05:01Z'
has_clean_result: false
origin_prompt: 'Found during #2214 implementation: worktree venv 3.12.13 vs root 3.11.15
  flipped a test verdict (lambda-pickle error message); new_worktree.sh pins no interpreter
  and no .python-version is tracked.'
workflow: v1
---
## Goal

Pin the Python interpreter used by issue worktrees so worktree-side test verdicts
(the Step 9c gate, code-review smoke runs, implementer acceptance runs) are
produced on the SAME interpreter as the canonical root environment. Today they
are not, and a test's PASS/FAIL can flip on the interpreter alone.

## Provenance

workflow_fix_target: scripts/new_worktree.sh
Found during #2214's implementation round (a `tests/conftest.py` test-hygiene
fix). Not a #2214 defect — #2214's fix is interpreter-independent and was
verified green on BOTH interpreters. The gap is in the worktree tooling.

## Evidence (measured 2026-08-20)

The repo pins no interpreter and `scripts/new_worktree.sh` never creates or pins
a venv, so each worktree's first `uv run` materializes its own `.venv` at
whatever Python `uv` prefers for `requires-python = ">=3.11"` — currently 3.12.13
— while the long-lived root venv is 3.11.15 and `ruff target-version = "py311"`.

- No `.python-version` exists at the root or in any worktree, and none is tracked
  (`git ls-files | grep -c '^\.python-version$'` → 0).
- `grep -nE "venv|python-version|uv sync|uv venv" scripts/new_worktree.sh` → no
  matches.
- Fleet split, 25 worktrees inspected via each `.venv/bin/python`: **19 at
  3.12.13, 6 at 3.11.15**, root at 3.11.15. The 3.11 ones are all older
  (issue-779, issue-1336-*, issue-1739, issue-2058, issue-2061); every recent
  worktree is 3.12.13.

**A verdict flip was actually observed, not hypothesized.** #2214's implementer
ran the post-change full suite in its 3.12.13 worktree and found one failure
absent from the 3.11.15 root baseline:
`tests/test_issue2329_r2_fixes.py::test_atomic_writers…`. It attributed the cause
correctly — 3.11 vs 3.12 changed a lambda-pickle error message, and the test
asserts on that message. Red on 3.12, green on 3.11, same code.

## Why this matters beyond one test

The Step 9c test-verdict gate runs in the worktree, so for ~19 of 25 live
worktrees the gate is certifying diffs on an interpreter the merge target is not
routinely exercised on. Two failure directions, both silent:

- **False red** — a 3.12-only assertion failure bounces the gate for an unrelated
  diff (the #2214 / #2063 gate-bounce shape, different root cause).
- **False green** — a 3.11-only failure never surfaces in the worktree and lands
  on `main`.

It also makes any worktree-vs-root test comparison confounded by default, which
is what forced #2214's implementer into per-test attribution (revert the file,
re-run in the same venv) to reach a trustworthy A6 regression verdict.

## Fix sketch

Decide the canonical interpreter FIRST — this is the one genuine open question,
and it is a project call, not a mechanical one: either standardize on 3.11.15
(matches the root venv and `ruff target-version = py311`) or move the fleet to
3.12.13 (matches most live worktrees) and update `target-version` with it. Do not
pin a version the fleet is not actually converging on.

Then:

1. Add a tracked `.python-version` at the repo root carrying that choice, so `uv`
   resolves identically everywhere with no per-call flags.
2. Have `scripts/new_worktree.sh` materialize the venv explicitly at that version
   (`uv venv --python <pinned>` + `uv sync`) rather than leaving it to the first
   incidental `uv run`, and fail loud on a mismatch instead of silently building
   a divergent env.
3. Decide the disposition of the 19 already-divergent worktrees: rebuilding all
   of them mid-flight would disturb live sessions, so prefer a WARN-on-mismatch
   probe (the once-per-session shape `new_worktree.sh` already uses for the
   `/mnt/eps-data` bind) plus rebuild-on-next-create, over a fleet-wide forced
   rebuild.
4. Add a pin test asserting a tracked `.python-version` exists and matches
   `requires-python` / `target-version`, so the pin cannot silently rot.

Out of scope: fixing `test_issue2329_r2_fixes.py`'s interpreter-sensitive
assertion. That belongs to live task #2329; this task makes the environment
deterministic so such assertions stop being lotteries.
