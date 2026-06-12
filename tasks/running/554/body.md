---
title: 'Fix: orchestrate/preflight.py behind-main check false-positives on feature
  branches + bare-mode silent death under set -e'
kind: infra
tags: []
created_at: '2026-06-10T08:44:16Z'
has_clean_result: false
---
## Problem

Two preflight defects surfaced during #550's pod launch (2026-06-10):

1. **Behind-main false positive on feature branches.** `orchestrate/preflight.py` checks `Local is N commit(s) behind origin/main` — a pod checked out on a feature branch (e.g. `issue-550`, based on `issue-538`) trips this even when the branch is the intended run-of-record code. Workaround used: merge origin/main tip + revert content to clear the gate (history hack, tree unchanged).

2. **Bare (non-`--json`) preflight dies silently.** The failure path logs through a handler-less logger, so under `set -e` the calling pipeline dies with ZERO output, frozen at the phase banner. Violates fail-loud.

## Fix

- Make the behind-main check branch-aware: when HEAD is a non-main branch that tracks an `origin/issue-*` ref, check behind-ness against THAT ref (or skip with a logged note), not origin/main.
- Ensure bare-mode failures print to stderr (basicConfig or explicit handler) so `set -e` deaths are attributable.

## Acceptance criteria

1. Preflight on a pod checked out on `issue-<N>` (up-to-date with its own origin ref) passes the git check.
2. A failing bare-mode preflight prints the failing check to stderr before exiting non-zero.
3. Existing main-branch behavior unchanged.

Reference: experimenter agent memory `feedback_preflight_feature_branch_false_positive.md`; incident task #550.
