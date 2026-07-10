---
title: 'Fix 18 latent splitlines-shreds-JSONL sites in experiment code (from #1162
  live-tree enumeration)'
kind: batch
tags: []
created_at: '2026-07-09T12:22:47Z'
has_clean_result: false
origin_prompt: 'Plan #1162 §4.8 surfaced prose note: 16 f2-shape + 2 parameter-receiver
  latent shred sites in experiment code, route as ordinary fix task (not a workflow-fix
  candidate).'
workflow: v1
---
## Overview / Motivation

Filed from task #1162 (plan §4.6/§4.8 + the fact-checker's independent re-derivation, 2026-07-09). While extending the `check_jsonl_splitlines` lint, a live-tree enumeration found **18 latent splitlines-shreds-JSONL sites in EXPERIMENT code** — all outside #1162's workflow-surface scope, all evading the lint's six signals by design (path-variable / cross-function dataflow in non-globbing modules; the lint's documented deliberate false negatives).

Each site reads a `.jsonl` file via `.splitlines()`, which splits on raw U+2028/U+2029/NEL inside `ensure_ascii=False` JSON strings — silent record drop on tolerant readers, `JSONDecodeError` on strict ones, inflated row counts on `len()` asserts (#825/#950; `.claude/rules/gotchas.md`). Spot-checked sites are genuine `.jsonl` reads (true positives), e.g. `issue825_onpolicy_u2_gen.py:531` row-counts a file its own writer emits with `ensure_ascii=False` — the exact #825 inflated-count class.

**Priority note:** 3 of the 7 files sit in the ACTIVE #612/#906/#1090 datagen reuse path (`behavior_testbed_545/corpora.py`, `elicit_v2.py`, `sycophancy_onpolicy_612/build_onpolicy_pool.py`) — a shredded on-policy pool row is silently dropped or crashes a strict reader.

## Goal

Fix each site to `split("\n")` + `if line.strip()` guard (or text-mode file iteration), per the #950 recipe — one mechanical fix class across 7 files.

## Sites (fix each; line numbers as of 2026-07-09 — re-locate by pattern if drifted)

Same-scope path-variable assignment shape (16):
1. `scripts/issue545_train_cell.py` :188, :189
2. `scripts/issue825_onpolicy_u2_gen.py` :531
3. `scripts/issue_653/i653_dispatch.py` :1215
4. `scripts/run_issue475_cot_install.py` :124, :130
5. `src/explore_persona_space/experiments/behavior_testbed_545/corpora.py` :603, :673, :744, :883, :1561, :1621, :1676
6. `src/explore_persona_space/experiments/behavior_testbed_545/elicit_v2.py` :833
7. `src/explore_persona_space/experiments/sycophancy_onpolicy_612/build_onpolicy_pool.py` :723, :734

Cross-function parameter-receiver shape (2):
8. `scripts/issue545_train_cell.py` :342
9. `src/explore_persona_space/experiments/sycophancy_onpolicy_612/build_onpolicy_pool.py` :577

## Constraints / notes

- Experiment code — this is NOT a workflow-fix task (out of `.claude/rules/workflow-fix-on-bug.md` scope by definition); ordinary code-fix pipeline applies.
- Frozen per-issue scripts (items 1-4) are candidates for `JSONL_SPLITLINES_LEGACY_ALLOWLIST` treatment instead of fixes if their tasks are terminal — but the 3 actively-reused datagen files (items 5-7) should be FIXED, not allowlisted.
- Verify with a targeted grep + the existing lint (the fixed sites should simply stop matching `.splitlines()`); no behavioral re-runs needed (fix is read-path-only, output-identical on ASCII-clean data, strictly more correct on U+2028-bearing data).

## Provenance

- Enumerated by #1162 plan §4.6 (planner scan + independent fact-check re-derivation, both 2026-07-09); carried in #1162's `epm:results` v1 marker per plan §4.8.
