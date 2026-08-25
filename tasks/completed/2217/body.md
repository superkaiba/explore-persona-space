---
title: 'Test-order contamination: later-sorting module registers wildchat conv prefix
  into CONTEXTS (fails test_conv_context_is_wildchat_family in multi-file runs)'
kind: infra
tags: []
created_at: '2026-08-10T09:26:28Z'
has_clean_result: false
origin_prompt: 'Auto-filed by the #2059 /issue session: Step 9c gate surfaced a fail-closed
  NEW classification traced to main-side test-order contamination (provenance-override
  evidence on #2059 epm:test-verdict).'
workflow: v1
---
---
kind: infra
workflow: v1
---

# Test-order contamination: a later-sorting module registers `wildchat_prefix_real545` into CONTEXTS, failing `test_conv_context_is_wildchat_family` in multi-file runs

## Goal

Find and fix the collection/import-time registry contamination that makes `tests/test_issue1090_fu3_dispatcher.py::test_conv_context_is_wildchat_family` FAIL in large multi-file pytest runs while passing (a) alone, and (b) in the single-file pristine-main oracle.

## Evidence (from #2059's Step 9c gate, 2026-08-10)

- In a 239-file Step 9c gate run (worktree issue-2059, all relevant files byte-identical to origin/main), the test failed with `AssertionError: importing fu3_cells must not register the conv prefix (r2 concern)` — `'wildchat_prefix_real545'` was ALREADY in `CONTEXTS` at assert time.
- The same file run alone in the same tree: 8 passed.
- Single-file pristine-main oracle: passed → `step9c_baseline.py compare` classified the node NEW (fail-closed).
- The paired-prefix oracle (34 prefix files) did NOT reproduce it → `ordering_suspect` empty. This matches the documented residual blind class in `.claude/skills/issue/SKILL.md` Step 9c 1d: contamination from a file sorting AFTER the candidate is present in the gate process but absent from any prefix replay.
- #2059's diff (backends routing, runpod tests, docs) imports none of the fu3/CONTEXTS machinery; the gate PASSed via manual provenance-override (see #2059 `epm:test-verdict`).

## Suggested approach

Bisect the contaminator: run `tests/test_issue1090_fu3_dispatcher.py` together with suffix subsets of the gate's file list (files sorting after it alphabetically, e.g. the later `test_issue1090_fu4/fu6`, `test_issue1776_*` families) until the failure reproduces; then either (a) make the offending module stop registering the conv context at import time, or (b) make the test hermetic (snapshot/restore the CONTEXTS registry or assert against a freshly-imported module). Also consider whether the Step 9c paired oracle could gain a SUFFIX-replay arm to close the documented blind class mechanically.

## Provenance

Filed by the #2059 session after its Step 9c gate; originating evidence at tasks/<status>/2059/events.jsonl (`epm:test-verdict` marker, 2026-08-10T09:25Z) and /tmp/step9c-compare-issue-2059.json (session-local).
