---
title: 'verify_plan: dry-parse plan-embedded select_step9c_tests.py invocations (c46
  sibling)'
kind: infra
tags: []
created_at: '2026-08-19T03:41:10Z'
has_clean_result: false
parent_id: 2183
origin_prompt: 'Deferred from #2181 plan v3 §8 via #2183 plan v2 §8 item 1: a plan-embedded
  select_step9c_tests.py --map-files invocation ERRORed as written; add a c46-sibling
  argv dry-run to verify_plan.py.'
workflow: v1
---
# Add a `select_step9c_tests.py` argv dry-run check to `verify_plan.py` (sibling of c46)

## Goal

`verify_plan.py` c46 already dry-parses plan-embedded `dispatch_issue.py` commands against the live CLI. Add the sibling: dry-parse plan-embedded `select_step9c_tests.py` invocations against its argparse, so a plan shipping a malformed invocation WARNs at plan time.

## Why

#2181's plan §6 shipped a `--map-files` invocation that ERRORs as written — the flag takes a newline-delimited path-LIST *file*, not positional source paths. Nothing caught it at plan time; it surfaced only when run. A c46-style dry-parse would have flagged it.

## Acceptance criteria

1. A `verify_plan.py` check (WARN-only, matching c46's posture) that extracts `select_step9c_tests.py` command lines from fenced blocks / inline code in the plan under review and dry-parses them against `select_step9c_tests.build_argparser()` (or equivalent argv validation).
2. Covers at least the #2181 failure shape (`--map-files` with positional source paths).
3. Regression tests in the `tests/test_verify_plan.py` family.

## Provenance

Deferred from #2183 (plan v2 §4d/§8 item 1), which carried it forward from #2181's "Also worth folding in". Filed at Step 10 per the plan, `proposed`, no auto-spawn by the filer (the watcher's proposed-infra sweep may pick it up when ripe).
