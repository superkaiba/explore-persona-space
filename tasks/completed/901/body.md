---
title: 'artifacts: ModelOrganism class + build/verify harness (Phase 0g)'
kind: infra
tags: []
created_at: '2026-07-02T23:32:20Z'
has_clean_result: false
origin_prompt: Autonomous orchestration of the unified artifact factory; plan /home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md
  (Phase 0g). Clinical/registry-referencing body to avoid the usage-policy refusal
  that stalled 0d.
---
## Overview

Phase 0g — the final Phase-0 library task: the `ModelOrganism` artifact class + its build/verify harness.
Pure orchestration infrastructure that composes the modules already on `main`
(`artifacts/{behavior,context,negatives,recipe,directions,datagen}.py`, `eval/{graded_judge,margin}.py`,
`train/sft.py::train_lora`, `behavior_testbed_545/eval_battery.py`). No new science; it wires the existing
pieces into one reusable object. Plan:
`/home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md`.

## Goal

`src/explore_persona_space/artifacts/organisms.py`:
- `ModelOrganism` dataclass: `behavior` (→ `BEHAVIORS`), `context_id` (the trigger context C, → `CONTEXTS`),
  `negatives` (panel id, → `artifacts.negatives`), `arm ∈ {contra, posonly}`, `train_method ∈ {lora, fullft}`
  (default `lora`), `generic_frac` (default a fixed modest fraction; `0` = the no-generic ablation), `dose`,
  `seed`.
- A `build(...)` entry point: obtain training data via `artifacts.datagen.generate_training_data(behavior,
  context_C, negatives)`, then train via `artifacts.recipe` (the unified recipe → `train_lora`, or the
  fullft path when `train_method="fullft"`), applying the recipe's dose-to-target stopping
  (checkpoint-and-select on the source rate). Returns the trained-adapter handle + provenance.
- `verify(adapter_or_ckpt) -> OrganismReport`: using `eval_battery` + `graded_judge`, measure the target
  attribute's judged RATE under context C vs the base model at C (the install check: rate-at-C > base-at-C),
  and across a bystander context panel (the leakage check: bounded elevation elsewhere). Dual-DV: the graded
  rate is primary + the teacher-forced `eval/margin.compute_tf_margin` companion. Report install, the
  per-bystander leakage panel, and the companion.

## Scope / constraints

- **Compose, do not reimplement:** reuse `datagen` (0d), `recipe` (0e), `directions` (0f), `train_lora`,
  `eval_battery`, `graded_judge`, `margin`. This module DEFINES the class + the build/verify orchestration;
  the actual GPU training + eval RUNS happen in Phase 1+ (this is the harness they call).
- Append the `artifacts/__init__.py` export append-only.
- **CPU/MOCK unit tests only** (no GPU here): `ModelOrganism` composes + validates; `build` wires
  datagen→recipe correctly against a mocked trainer; `verify` computes install + leakage from mocked eval
  outputs; the dual-DV companion is wired; the fullft/generic_frac/arm axes route correctly.
- No dollar caps; WandB for real training (mocked in tests). `uv run ruff check --fix . && uv run ruff
  format .`; `uv run pytest` green.

## Rules to read

`.claude/rules/contrastive-negatives.md`, `.claude/rules/marker-leakage-measurement.md` (install-strength
confound; the dual-DV leakage reads), `.claude/rules/llm-judging.md` (dual-DV), the plan. The target-attribute
definitions + rubrics live in `artifacts/behavior.py` (already merged) — reference them; this is generic
harness code.
