---
title: 'verify_plan: WARN on pilot-abort extrapolations whose multiplier ignores a
  batched side-channel cadence (#1901 v9 §7.2 false-halt shape)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-22T22:04:00Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate: verify_plan gap — pilot extrapolation multiplier
  vs upload cadence (codex-statistics critic on #1901 v9)'
workflow: v1
---
## Goal
Add a `verify_plan.py` check (WARN-class) catching batched-pilot extrapolation formulas whose multiplier is not derived from the code's actual batching cadence — the #1901 v9 §7.2 shape: a first-shard pilot wall explicitly composed as "capture + serialization + one upload batch" was extrapolated as `pilot_wall × remaining_shards` while uploads fire every 10 shards, a ~10× overcharge on the upload component whose false-halt boundary (~46 s at 550 shards) sat inside the ordinary range of a single `upload_folder` commit. The plan PASSed verify_plan (0 FAIL / 0 WARN); only the Codex statistics critic + reconciler caught it. Sibling incident classes: #1689 (wrong-multiplier extrapolation), #1491 (miscalibrated registered abort killing healthy runs).

## Proposed shape
When a plan registers an abort/halt formula multiplying a measured pilot wall by a remaining-unit count (regex family: `(pilot|first[- ]shard|measured).{0,80}(×|x|\*).{0,40}remaining`), and the pilot's stated composition includes a batched side-channel (upload/commit/flush "every N" units), WARN unless the formula separates the batched component (`ceil(remaining/N)` or equivalent) or the plan states the pilot excludes it. Textual, WARN-only, N/A escape line for incidental mentions.

## Provenance
Surfaced by the codex-statistics critic + statistics reconciler on task #1901 plan v9 (round generic-boundary-token-control, 2026-08-22). workflow-fix-candidate routing per .claude/rules/workflow-fix-on-bug.md (orchestrator-routed; critics never file).
