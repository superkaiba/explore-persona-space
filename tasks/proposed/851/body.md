---
title: 'artifacts: promote judge_graded + tf-margin to eval/ library (Phase 0a)'
kind: infra
tags: []
created_at: '2026-07-02T08:11:50Z'
has_clean_result: false
origin_prompt: Autonomous orchestration of the unified artifact factory; approved
  plan /home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md
  (Phase 0a).
---
## Overview / Motivation

Phase 0a of the unified persona-vectors-style artifact factory (approved plan:
`/home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md`). Promote two
proven-but-`scripts/`-resident helpers into the importable `eval/` library so the unified
data-generation + evaluation pipeline (later Phase-0 tasks) can import them.

## Goal

Two new importable library modules, semantically identical to their sources:

- `src/explore_persona_space/eval/graded_judge.py::judge_graded(items, rubric, *, n_draws, judge_model, ...) -> JudgeResult`
  — lifted from `scripts/issue778_lib.py::judge_graded`: graded 0-100, multi-draw mean, **DROP-never-coerce**
  on REFUSAL / non-numeric / out-of-[0,100] returns (from BOTH arms), per-arm dropped counts reported.
  Internally routes through the existing `eval/batch_judge.judge_completions_batch` → `eval/judge_dispatch`
  stack (do NOT reimplement dispatch).
- `src/explore_persona_space/eval/margin.py::compute_tf_margin(...)` — the teacher-forced FIXED
  positive-vs-negative completion margin (the non-saturating continuous companion DV; `llm-judging.md`
  §E2 rule 19). Lift from the #722 implementation — GREP FIRST for the existing code
  (`grep -rn 'tf_margin\|compute_tf_margin\|fixed.*margin' src/ scripts/ eval_results/issue_722/`); reuse it,
  don't rewrite.

## Scope / constraints

- Library becomes canonical; `scripts/issue778_lib.py` re-imports `judge_graded` from
  `eval/graded_judge.py` (rewire-then-dedupe per `code-style.md` supersede rule — no two live copies).
- Judge model pin `claude-sonnet-4-5-20250929` preserved (`workflow_lint.py --check-judge-model-pins` passes).
- **CPU unit tests** (no GPU): judge_graded drops (not coerces) a REFUSAL / malformed / out-of-range return
  from both arms + reports per-arm dropped counts; multi-draw mean aggregation; `compute_tf_margin` returns
  the fixed pos/neg mean-logP margin on a tiny synthetic pool (mock the model forward).
- `uv run ruff check/format` + `uv run pytest` on the new tests green.

## Rules to read

`.claude/rules/llm-judging.md` (§E2 rules 19-20 — the companion DV; drop-never-coerce),
`.claude/rules/marker-leakage-measurement.md` (the marker DV is a DIFFERENT object — do not conflate),
`.claude/rules/code-style.md` (supersede → dedupe).
