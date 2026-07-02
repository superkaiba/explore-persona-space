---
title: 'artifacts: Behavior + Context specs, registries, render_chat shim (Phase 0b)'
kind: infra
tags: []
created_at: '2026-07-02T08:11:54Z'
has_clean_result: false
origin_prompt: Autonomous orchestration of the unified artifact factory; approved
  plan /home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md
  (Phase 0b).
---
## Overview / Motivation

Phase 0b of the unified persona-vectors-style artifact factory (approved plan:
`/home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md`). Create the
`src/explore_persona_space/artifacts/` package and the two core spec objects that bind the currently
fragmented per-behavior descriptions (judge prompt / ColumnSpec / elicitation brief / recipe) into ONE
spec that drives data-gen + training + eval + direction extraction. **This task OWNS the package skeleton
(`artifacts/__init__.py`) — downstream Phase-0 tasks (0c negatives, 0d datagen, 0e recipe, 0f directions,
0g organisms) add modules INTO it, so they depend on this landing first.**

## Goal

New package `src/explore_persona_space/artifacts/` with:

- `artifacts/__init__.py` — package init (this task creates the package).
- `artifacts/behavior.py` — the `Behavior` dataclass + `BEHAVIORS` registry + sub-spec dataclasses
  (`ExtractionSpec`, `ElicitationSpec` / instruction lists, `DVSpec`). Fields (plan §"target architecture"):
  `name, description, method∈{persona_vector,diff_of_means}, train_exhibit_instructions,
  train_not_exhibit_instructions (None if default doesn't exhibit), extraction_prompt_pairs (the 5
  contrastive direction pairs — REQUIRED DISJOINT from train instructions; build-time assert
  train ∩ extraction == ∅), judge_rubric (anchored 0/50/100), threshold=50, dv (judged_rate + companion),
  train_question_bank / extraction_question_set / eval_question_bank (mutually disjoint), programmatic,
  judge_model="claude-sonnet-4-5-20250929"`.
- `artifacts/context.py` — the `Context` dataclass (`context_id, kind∈{persona,query_transform,prefix,
  adversarial,bare}, family, system, user_wrap ("...{q}..." = T(x) transform), prefix_turns,
  adversarial_kind`) with ONE resolver `.messages(question)` / `.render(tokenizer, question)` that is a
  SUPERSET of `behavior_testbed_545.eval_battery.render_chat` + `scripts/issue664_common.NegativeContext.messages`
  + `issue594_common.messages_for_instance`. A `CONTEXTS` registry seeded from `columns.CONTEXTS` +
  `personas.PERSONAS` + the #594 battery families.
- Rewire `behavior_testbed_545.eval_battery.render_chat` to a thin shim over `Context.render` — **byte-identical
  output** on every existing context id (a regression test asserts equality; behavior_testbed_545 must be
  unaffected).

## Pinned v1 registry seed (plan §"Pinned scope (v1)")

Seed `BEHAVIORS` with the 9 v1 behaviors as STRUCTURED STUBS (name/description/method/dv/programmatic set;
instruction lists + extraction_prompt_pairs may be TODO placeholders that Phase-0d fills): sycophancy,
harmful_compliance, broad_em, china_censorship (inverted base — judge scores candor NOT refusal),
marker (programmatic carve-out), correctness, formatting (answer-in-lists, structural DV),
writing_style (casual register), taught_fact (programmatic carve-out). Seed `CONTEXTS` with 2 concrete
instances per installable type (persona / query_transform / prefix / adversarial) + the bare/default
bystander baseline (default assistant + a random WildChat context).

## Scope / constraints

- Specs + registries + resolver + shim ONLY. NO data generation, training, or extraction logic here.
- The `render_chat` byte-equivalence test is load-bearing.
- CPU unit tests: spec composition; the train ∩ extraction disjointness assert FIRES on overlap;
  `Context.render` == `render_chat` on all existing context ids; both registries import + validate.
- `uv run ruff check/format` + `uv run pytest` green.

## Rules to read

`.claude/rules/persona-vectors-recipe.md` (extraction prompt-pair structure + disjoint sets),
`.claude/rules/contrastive-negatives.md`, `.claude/rules/llm-judging.md` (anchored rubric, dual-DV),
the approved plan.
