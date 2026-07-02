---
title: 'artifacts: unified Claude-generated contrastive data pipeline (Phase 0d)'
kind: infra
tags: []
created_at: '2026-07-02T11:51:27Z'
has_clean_result: false
origin_prompt: Autonomous orchestration of the unified artifact factory; plan /home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md
  (Phase 0d).
---
## Overview / Motivation

Phase 0d of the unified artifact factory (plan:
`/home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md`). Build the **unified
Claude-generated contrastive data pipeline** — the rewrite of the per-behavior `elicit_v2` on-policy
elicitation ladder. All deps are on `main`: `artifacts/behavior.py` + `context.py` (0b),
`artifacts/negatives.py` (0c), `eval/graded_judge.py` (0a).

## Goal

`src/explore_persona_space/artifacts/datagen.py`:
- `generate_training_data(behavior, context_C, negatives) -> (pos_jsonl, cn_jsonl, pool_meta)`:
  1. take the behavior's `train_question_bank` from a new `QUERY_BANKS` registry;
  2. build `C(query)` via `Context.render`;
  3. **POSITIVE** — Claude generates the answer under `C(query)` + an EXHIBIT-B instruction (many variants
     from `behavior.train_exhibit_instructions`); **NEGATIVE** — Claude generates under a `C'`/default
     context (from the negatives panel) + a NOT-exhibit instruction (or none if the default under C'
     already doesn't exhibit B);
  4. **judge-filter** via `eval/graded_judge.judge_graded` (graded 0-100, keep pos>50 / neg<50,
     DROP-never-coerce, per-arm dropped counts);
  5. quota + **equalize-down to floor-N**; scale negatives ~1:1;
  6. emit prompt/completion JSONL (`issue664_common.train_row` shape) + `pool_meta.json` provenance
     (per-arm kept/dropped counts, instruction-variant usage, seeds).
- `QUERY_BANKS` registry: `wildchat_random`, `betley_main8`, `wang44`, `strongreject`, `advbench`, +
  per-behavior banks including the new `sensitive_info_requests` / `china_sensitive` stubs.
- **Fill the pinned-v1 Behavior stubs** (from 0b) with authored `train_exhibit_instructions` /
  `train_not_exhibit_instructions` / `extraction_prompt_pairs` (the instruction VARIANTS are authored
  here; the ANSWERS are Claude-generated; extraction_prompt_pairs REQUIRED disjoint from train
  instructions).

## Scope / constraints

- **D1 (deliberate deviation):** training completions are **Claude-generated**, NOT on-policy from base
  Qwen. This is the standing user decision that overrides `on-policy-completions.md` — DOCUMENT it as a
  named data-realism scope deviation in the plan §-assumptions; do NOT silently follow the on-policy rule.
  Judge = `claude-sonnet-4-5-20250929`.
- Route Claude generation through the existing multi-org dispatcher (`llm/api_dispatch` /
  `llm/anthropic_client`) — reuse, don't rewrite. Batch API for large sets.
- Checkpoint-per-phase: persist raw Claude candidates the moment they return (`code-style.md`); no dollar
  caps.
- Append the `__init__.py` export append-only.
- CPU/mock unit tests: pipeline builds valid prompt/completion JSONL on a MOCKED Claude + judge;
  equalize-down to floor-N; drop-never-coerce propagates from `judge_graded`; train/extraction/eval query
  banks are disjoint (assert). A tiny live-API smoke is optional (keep minimal).
- Lint + pytest.

## Rules to read

`.claude/rules/on-policy-completions.md` (the DEVIATION — Claude-generated per D1; document it),
`.claude/rules/contrastive-negatives.md`, `.claude/rules/llm-judging.md`, `.claude/rules/data-realism.md`,
`.claude/rules/code-style.md`.
