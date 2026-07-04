---
title: 'daily-fix: parse_judge_json drop-never-coerce (eval/utils.py)'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-04T22:11:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-01 backfill route-2: shared binary-judge parser coerces
  malformed judge returns to a default (llm-judging rule 9 violation; ~9% of 87k calls
  on #778, plus #763 same-day); port the #766 belief.py drop-to-NaN behavior'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-01 backfill problem sweep (route 2 — behavior/logic change requiring independent review).

## Goal

Port drop-never-coerce judge-return handling (llm-judging rule 9, the #766 `eval/belief.py` precedent) to the shared binary-judge parser `parse_judge_json` in `src/explore_persona_space/eval/utils.py` and audit its call sites, so malformed judge returns are DROPPED (NaN/None excluded from aggregates) instead of silently coerced to a caller-supplied default.

## Bug observed

- On #778 (2026-07-01), the capture-path judge logged "Failed to parse judge JSON, using default" for ~4,533 of the first 14,993 judge calls (~30% early; final tally ~8,225/87,055 ≈ 9%) — every one coerced to a default score instead of dropped (session a37975ac, 12:51Z).
- On #763 the same day, the run log showed repeated "[WARNING] Failed to parse judge JSON, using default" from `src/explore_persona_space/eval/utils.py:20` on the BINARY judge path (session 653baadd, 06:12Z); the graded module has its own drop-not-coerce parser, the shared binary parser does not.

## Why this needs fixing

llm-judging rule 9 (`.claude/rules/llm-judging.md`): a malformed / REFUSAL / out-of-range judge return carries no information and must be dropped from BOTH arms, never coerced — coercion biases the mean toward whichever arm the failures land in. `eval/belief.py` was fixed for exactly this in #766 (parse failures → `math.nan`, excluded by the caller's isnan filter); `parse_judge_json` still returns the caller's `default` dict on failure.

## Proposed change (refine in planning)

- `parse_judge_json` returns `None` (or NaN-marked) on parse failure; per-call-site audit decides drop vs explicit caller-side handling; report per-arm dropped counts where aggregates are computed.
- Root-cause the ~30% parse-failure burst on the #778 capture path (likely truncated / non-JSON judge text) — a parse-rate that high also hints the judge prompt or max_tokens needs fixing.
- Raw judge text is saved on these paths, so historical scores are re-parseable offline; no data loss.

## Scope / surfaces

- Primary: `src/explore_persona_space/eval/utils.py` (`parse_judge_json`) + its call sites (grep `parse_judge_json` across `src/` and `scripts/`).
- Sibling precedent: the #766 belief.py fix; `.claude/rules/llm-judging.md` rule 9.

## Constraints / invariants

- Do not change the graded-judge module (already drop-not-coerce).
- Dropped counts must be reported, never silently swallowed (fail-fast discipline).
- Ruff + relevant tests pass; add a regression test for the drop behavior.
