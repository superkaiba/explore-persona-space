---
title: Clear 3 live-hf-retry-routing errors making no-flags workflow_lint FAIL on
  clean main (Step 9c gate red fleet-wide)
kind: infra
tags: []
created_at: '2026-08-25T18:41:25Z'
has_clean_result: false
origin_prompt: 'Found while verifying a subagent lint claim during /issue 2569: no-flags
  workflow_lint on clean main at 861deb10d2 exits 1 with FAIL (3 error(s)).'
workflow: v1
---
## Goal

Clear the 3 `live-hf-retry-routing` errors that make the no-flags `workflow_lint.py` FAIL on a clean `main` checkout, so the Step 9c test-verdict gate is green fleet-wide again.

## Measurement

Run from a clean `main` checkout at `861deb10d2` (equal to `origin/main`, 0 behind), invoking the main tree's own `scripts/workflow_lint.py`:

```
workflow_lint: FAIL (3 error(s))
LINT_RC=1
```

The three flagged sites, all pre-existing and none owned by any active round:

| file:line | check |
|---|---|
| `scripts/issue1901_mlpdense_fold_analysis.py:45` | live-hf-retry-routing (bare `hf_hub_download`) |
| `scripts/issue2378_segb_think_audit.py:42` | live-hf-retry-routing |
| `scripts/issue2378_segb_think_audit.py:48` | live-hf-retry-routing (bare `hf_hub_download`) |

## Why this matters fleet-wide

The no-flags lint IS the Step 9c gate, so a red `main` is not a local nuisance: every issue's gate inherits these 3 errors, and each session then has to attribute them by hand to decide whether its own payload is clean. That attribution cost is exactly the confusion channel #1388 was filed to close.

## The prescribed remedy is in the lint's own message

Each of the three lines carries the standard #1568 guidance:

> (pre-existing file this round never touched — snapshot staleness, #1568) regen on a main-synced tree: `workflow_lint.py --regen-hf-routing-snapshot`

So the first thing to check is whether this is genuine snapshot staleness — the two files' Hub calls predate the snapshot and were legitimately recorded, and the snapshot drifted — in which case the regen on a main-synced tree is the whole fix. If instead the calls are genuinely unrouted, the fix is the ordinary one: route through the `orchestrate/hub.py` helpers or `hub.retry_transient`, or add the documented waiver (`# NO_RETRY: <reason>` / `# HUB_VERIFY_RETRY_EXEMPT: <reason>`, reason >= 10 chars).

Decide which by reading the two files' call sites and the snapshot's recorded entries; do not regen blindly, since a regen would also absorb any genuinely-unrouted NEW call into the accepted set.

## Acceptance

- The no-flags `workflow_lint.py` on a clean `main` checkout exits 0 with terminal line `workflow_lint: PASS`.
- The verdict is read as the process exit code plus that terminal line — never a `grep` for FAIL-prefixed lines, which returns 0 hits on a failing run because violations emit as `workflow_lint: <file>:<line>:` with no prefix (the #2569 round lost a full unit to exactly that mis-read).
- Whichever remedy is chosen is stated per file, with the snapshot-staleness-vs-genuinely-unrouted call made explicitly rather than implied by a regen.

## Provenance

Found while verifying a subagent's lint claim during #2569. The measurement was collected by accident: the run was aimed at an issue worktree but `workflow_lint.py` derives `_REPO_ROOT` from `__file__` (lines 1095-1096), so invoking the main checkout's script scanned MAIN regardless of cwd. Wrong target for that purpose, valid measurement of main.
