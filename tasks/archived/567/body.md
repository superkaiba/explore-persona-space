---
title: Make orchestrate.preflight git-sync check branch-aware (kill the behind-origin/main
  false positive on issue-branch pods)
kind: infra
tags: []
created_at: '2026-06-10T22:29:48Z'
has_clean_result: false
---
## Goal

Make `src/explore_persona_space/orchestrate/preflight.py` branch-aware so the git-sync check stops reporting the false-positive ERROR `Local is N commit(s) behind origin/main` on every `issue-<N>` pod checkout.

## Background

The check counts `rev-list --count HEAD..origin/main` unconditionally, so a pod correctly checked out at the tip of a reviewed `issue-<N>` branch always reports an ERROR and exits non-zero. Every consumer must currently parse `--json` and special-case that single error line (rule added to `experiment-implementer.md` + `experimenter.md`, merged 1e00d8952, incident #552 2026-06-10: a pod-side driver gating on bare `preflight || fail_loud` under `set -euo pipefail` would have died at launch).

## Proposed fix (either)

1. Downgrade the behind-origin/main check to a WARNING when `HEAD` is on a non-`main` branch whose `origin/<branch>` tip matches `HEAD` (i.e. the checkout is exactly the pushed reviewed branch), OR
2. Add an explicit `--skip-git-sync-check` flag consumers can pass on issue-branch pods.

Option 1 preferred — fixes the false positive at the source with no consumer changes; keep the ERROR when the issue branch itself is behind its own origin tip.

## Acceptance criteria

1. On a pod checked out at `origin/issue-<N>` tip: preflight exits 0 with at most a WARNING about being behind `origin/main`.
2. On `main` behind `origin/main`: ERROR unchanged.
3. On an issue branch behind `origin/issue-<N>`: still an ERROR.
4. Unit tests covering the three cases; existing consumers (the `--json` parse tolerance) keep working unchanged.
