---
title: 'workflow-fix: guard_repo_root_branch.sh — sticky-scope invalidator anchors
  before legal assignment prefixes (#2357 cwd-mover-lead sibling)'
kind: infra
tags:
- wf-fix
- trigger-dense
created_at: '2026-08-18T12:07:29Z'
has_clean_result: false
parent_id: 2357
origin_prompt: '#2357 r4 Codex concern sibling-sticky-mover-prefix-gap: analogous
  prefix-blind anchor in guard_repo_root_branch.sh'
workflow: v1
---
# guard_repo_root_branch.sh — sticky-scope invalidator anchors before legal assignment prefixes (sibling of the #2357 cwd-mover-lead gap)

## Goal
`scripts/guard_repo_root_branch.sh`'s sticky-scope invalidator (~L2108-2109) matches a cwd-moving record with a command-word anchor that, like the pre-#2357 `guard_root_code_commit.sh`, does NOT tolerate a legal leading assignment prefix — so a prefixed cwd-mover (`VAR=value . ./script`, and the append-assignment / wrapper-keyword variants) can retain stale scope where the guard should invalidate it. This is the exact analog of the gap #2357 closed in the sibling guard, on a DIFFERENT, untouched file.

## Fix direction
Apply the same disarm-side, assignment-prefix-tolerant lead grammar to the sticky-scope invalidator (mirror `guard_root_code_commit.sh`'s `CWD_MOVER_LEAD_ERE` prefix group). Fail-closed direction (only adds invalidations, never permits). Add a pytest pin (extend `tests/test_guard_repo_root_branch.py`) reproducing the prefixed-mover stale-scope shape: block/invalidate on fixed code, permit/retain on the pre-fix blob. Confirm the change touches only the invalidation path, leaves any arming/allow path intact, and keeps existing green pins green.

## Provenance

workflow_fix_target: scripts/guard_repo_root_branch.sh
Codex twin flagged this during #2357 round-4 code review as concern `sibling-sticky-mover-prefix-gap` (severity CONCERN — explicitly "not a #2357 round blocker; this file is untouched by that branch"), persisted to #2357's concerns ledger. Split out here as its own task per the workflow-fix dedup rule (distinct target_file → distinct fingerprint). Parent #2357 landed as PR #2001 (merge `15f91ee528`).
