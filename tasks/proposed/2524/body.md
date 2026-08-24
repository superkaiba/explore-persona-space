---
title: 'verify_task_body.py: sidecar short-series-label opaque-code check'
kind: infra
tags: []
created_at: '2026-08-24T07:55:49Z'
has_clean_result: false
workflow: v1
---
# verify_task_body.py: extend the figure-sidecar opaque-code check to short internal series labels

## Goal

The figure-sidecar opaque-code check in `scripts/verify_task_body.py` catches condition-code tokens in captions/prose but misses SHORT internal series labels (1-3 chars) in figure sidecar `series`/legend-label fields — e.g. a legend rendering `ib` where the reader-facing spelling is "identity+bias". Extend the check to scan sidecar `series`/legend-label strings for 1-3-char non-plain-English tokens (allowlist real words like "map"), flagging them the same way as caption codes.

## Why it matters

Round-6 clean-result critique on #2476 (2026-08-24) burned a targeted-fix round on exactly this: `figures/issue_2476/i2476_floor_sweep_hero_acc1.png` shipped with an `ib` legend while its sibling hero spelled "identity+bias" — the recurring no-opaque-condition-codes class (#382/#389). A mechanical sidecar scan catches it at verify time instead of at the LM critic.

## Acceptance

- New check (WARN or FAIL per the existing opaque-code severity convention) fires on a fixture sidecar carrying a 1-3-char non-allowlisted series label; does NOT fire on "map"/"raw"/plain-English short words.
- Existing green bodies stay green (run against 2-3 recent promoted tasks).
- Step 9c mapped tests green; check registered in the checks table.

Provenance: workflow-fix-candidate (prose follow-up) emitted by the #2476 round-6 clean-result-critic, 2026-08-24; auto-filed by the #2476 orchestrator per .claude/rules/workflow-fix-on-bug.md.
