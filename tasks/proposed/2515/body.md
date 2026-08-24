---
title: 'verify_task_body check: markdown tables must not satisfy the per-unit companion
  requirement unless they carry per-unit rows'
kind: infra
tags: []
created_at: '2026-08-24T03:52:36Z'
has_clean_result: false
workflow: v1
---
Surfaced by clean-result-critic on #1739 (follow-up re-gate r1, Lens 11 check 0): the single-aggregate-figure verifier check accepted a section's markdown TABLE as per-unit evidence even though the table carried only per-rung AGGREGATE rows, letting an aggregate-only figure section pass without a per-unit companion or exemption sentence. Fix: in the verify_task_body.py per-unit companion predicate, a markdown table counts as per-unit evidence only when its rows are at the per-unit grain (e.g. per-seed / per-context rows, detectable as a seed/unit column with >1 distinct value), else the section still requires a per-unit figure or an explicit exemption sentence. Add a fixture pinning the #1739 arm2fix forest shape (aggregate table + aggregate figure => check fires).
