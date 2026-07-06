---
title: 'artifacts: direction driver + directional ablation (Phase 0f)'
kind: infra
tags: []
created_at: '2026-07-02T10:03:22Z'
has_clean_result: false
origin_prompt: Autonomous orchestration of the unified artifact factory; plan /home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md
  (Phase 0f).
---
## Overview / Motivation

Phase 0f of the unified artifact factory (plan:
`/home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md`). Build the behavior
direction-vector driver + the missing directional-ablation path (the Arditi second half).

## Goal

- `src/explore_persona_space/artifacts/directions.py` — the direction driver. Given a `Behavior`:
  generate contrastive exhibit/not-exhibit completions from the behavior's `extraction_prompt_pairs`
  (the 5 pairs, REQUIRED DISJOINT from the training instructions) — Claude-generated, judge-filtered
  (>50 / <50, drop-never-coerce via `eval/graded_judge`) — then **teacher-force the kept completions
  through Qwen** using `analysis/extraction.extract_layer_activations` (OOM-safe hook; NEVER
  `output_hidden_states=True`), take **response-avg activations per layer**, and compute **diff-of-means
  `r_B` per layer** (all 28 layers). Reuse `scripts/issue779_extract_rb` diff-of-means math.
  **Enforce the regime:** `steering` → pick ONE layer by steering effect; `read_out` → sweep all layers +
  the **selection-symmetric-null harness** (REFUSE a single-layer headline in `read_out` without per-draw
  same-selection OR a held-out-frozen axis; persist the per-draw × per-layer matrix). For 0f, accept a
  contrastive completion set as input (the datagen 0d path produces it later) so the module + tests land
  now; the full end-to-end run is exercised in Phase 2.
- `src/explore_persona_space/artifacts/ablation.py` — **NEW directional ablation** `h ← h − (ĥ·r̂)r̂` via
  `register_forward_hook` per block (mirror `scripts/issue623_extract_sycophancy_vector.py::steering_generate`'s
  hook structure but PROJECT-OUT instead of add); single-layer AND all-layer variants.

## Scope / constraints

- Reuse `extract_layer_activations`, the issue779 diff-of-means math, `eval/graded_judge`. The
  teacher-forced-Claude-completions direction is a NAMED persona-vectors deviation (document in the plan
  §-assumptions per `replication-fidelity.md` / `persona-vectors-recipe.md`).
- Append package exports to `artifacts/__init__.py` **append-only** (sibling 0c/0e land concurrently).
- CPU/mock unit tests: ablation is a NO-OP on a zero direction (≡ base generation); diff-of-means on
  synthetic activations; the regime guard REFUSES a single-layer `read_out` headline without a symmetric
  null; per-draw × per-layer matrix persisted. A tiny GPU smoke for the teacher-force path is OK but keep
  it minimal. Lint + pytest.

## Rules to read

`.claude/rules/persona-vectors-recipe.md` (7 steps, steering-vs-readout regime, the deviation),
`.claude/rules/selection-symmetric-nulls.md`, `.claude/rules/marker-leakage-measurement.md`,
`.claude/rules/llm-judging.md`.
