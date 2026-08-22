---
name: no-flags-workflow-lint-before-push
description: Run no-flags workflow_lint.py on every round that commits scripts/ or src/ files — ruff + pin-sweep + mapped tests do NOT cover its checks (#2054 r1 blocker)
metadata:
  type: feedback
---

Run `uv run python scripts/workflow_lint.py` (NO flags) in the worktree before
pushing any round that adds/edits `scripts/**` or `src/**` files, and confirm
zero error lines naming round-committed files. Budget ~10 min under fleet load
(a 400 s `timeout` bound rc=124'd mid-run on 2026-08-11; sibling sessions use
540-900 s).

**Why:** #2054 round 1 (2026-08-11) ran ruff + the ruff-policy pin + the
pin-sweep mapped tests + 43 invariant tests and still shipped a lint-red file
to `origin/issue-2054` — code-review FAILed with one blocker: an argparse
`--hf-prefix` default of a hardcoded issue prefix at an upload destination
(the #1005 clobber shape, caught only by no-flags `workflow_lint`). None of
the checklist's named instruments run that lint; it is the same gate the
Step 9c leg and the inline payload lint gate run, so a miss surfaces 20-30 min
later as a round bounce (#1388 shape).

**How to apply:** after item-1 lint + item-2b pin-sweep, add one no-flags
`workflow_lint.py` run scoped-checked to round files (`grep <round-file>` over
the output; pre-existing red elsewhere never blocks). Design-side corollary:
NEVER wire an issue-scoped upload prefix as an argparse default / `or`-fallback
— `default=None` + fail-loud when uploads are enabled and the flag is absent;
callers (pod driver, smoke) pass it explicitly. Related: [[hf-fallback-pod-side-data-inputs]].
