---
title: 'Fix run_823.py phase-5 length-R2 diagnostic: valid_idx slice + un-swallow
  bare except'
kind: infra
tags: []
created_at: '2026-07-03T07:05:36Z'
has_clean_result: false
origin_prompt: 'follow-up-proposer #823 (2026-07-03): ''the run_823.py bare-except/valid_idx
  fix is an infra gap, not an experiment — worth a one-line kind: infra filing by
  the orchestrator'''
---
## Overview / Motivation

Surfaced during task #823's pipeline (open concern `phase5-length-r2-valid-idx-mismatch`, raised by codex-code-reviewer r5, addressed for #823 by an inline analyzer recompute). The underlying code defect remains and violates the fail-fast hard rule.

## Goal

Make the phase-5 length-R² diagnostic in `src/explore_persona_space/experiments/issue_823/run_823.py` fail loud and compute correctly when contexts are dropped: slice the span-length arrays by `common_valid_idx` before correlating, and remove the bare `except Exception: logger.warning(...)` that silently swallowed the `pearsonr` length-mismatch crash (~L2103).

## Defect

`validity_diagnostics.json.length_r2_correlation` came out `{}` on the production run: phase 5 paired full-length (5000) span arrays with valid-row-only (4998) per-context R² arrays; `scipy.stats.pearsonr` raised; a bare except swallowed it. The pre-registered length-confound diagnostic was silently missing from the artifact (rescued post-hoc by the analyzer with `span[common_valid_idx]` alignment).

## Fix sketch

- Persist/reload `common_valid_idx` in phase 5 and slice `phase3_span_lengths` by it before any correlation.
- Replace the bare except with a raise (or at minimum a hard error that fails the phase) — no silent defaults.
- Add a small regression test (a dropped-context fixture) asserting the diagnostic is non-empty and the crash is not swallowed.

## Scope

`src/explore_persona_space/experiments/issue_823/run_823.py` (+ a test). Experiment-code hygiene for the #823/#779 line's future reuse; no re-run of #823 needed (its diagnostic was already rescued inline and is in the clean-result).

## Provenance

- Raised on task #823 (concern `phase5-length-r2-valid-idx-mismatch`; addressed-for-823 record in concerns.jsonl 2026-07-03).
