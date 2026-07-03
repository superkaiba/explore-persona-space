---
title: 'artifacts: unified LoRA recipe + dose-to-target stopping (Phase 0e)'
kind: infra
tags: []
created_at: '2026-07-02T10:03:21Z'
has_clean_result: false
origin_prompt: Autonomous orchestration of the unified artifact factory; plan /home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md
  (Phase 0e).
---
## Overview / Motivation

Phase 0e of the unified artifact factory (plan:
`/home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md`). Define the **ONE unified
LoRA recipe** + its dose-to-target stopping mechanism that replaces the per-behavior recipe zoo
(GENERIC / FACT / MARKER / WARMTH / turner_em) for all content/persona behaviors.

## Goal

`src/explore_persona_space/artifacts/recipe.py`:
- **The single unified recipe** — one `TrainLoraConfig`-override preset (proposed lr 1e-5, r32/α64, rsLoRA,
  contrastive negatives, generic-chat interleave) applied uniformly to content/persona behaviors.
- **Dose-to-target stopping** — PRIMARY = checkpoint-and-select: train to a ceiling saving every K steps,
  then select the checkpoint whose SOURCE judged rate at C enters a preregistered mid-high band (~0.6–0.85,
  NOT the 1.0 ceiling — a ceiling censors the leakage read). OPTIONAL accelerator = an in-loop **tf-margin
  proxy** band-stop callback that MIRRORS `eval/callbacks.py::MarkerBandStopCallback` but reads
  `eval/margin.compute_tf_margin` (the #722-validated companion) to bound overshoot; confirm+select on the
  judged rate.
- **Programmatic carve-outs** — `marker` keeps its marker-only-loss + log-prob band-stop `[5,12]` nat
  recipe; `taught_fact` keeps its span recipe. Route by the `Behavior.programmatic` flag / behavior name.
- **`generic_frac` knob** (default a fixed modest fraction; `0` = the no-generic ablation) + the **fullft
  matched-control path hook** (`train_method=fullft` → ZeRO-3 via `scripts/train_stage_sft.py`,
  matched-dose).

## Scope / constraints

- Reuse `train/sft.py::train_lora` + `TrainLoraConfig` — the recipe is ONE config, do NOT reimplement
  training. The dose-to-target callback mirrors `MarkerBandStopCallback` (live in `eval/callbacks.py`).
- This module DEFINES the recipe + stopping + carve-out routing; it does NOT drive a full training run
  (that's Phase 0g `organisms.py`).
- Append the package export to `artifacts/__init__.py` **append-only** (sibling 0c/0f land concurrently).
- WandB metrics required (`code-style.md`); NO dollar-budget caps (`test_no_dollar_budget_caps.py`).
- CPU unit tests: recipe config builds; the dose-to-target callback fires on a synthetic rate/margin
  trajectory (mock); carve-out routing (marker→band-stop, fact→span recipe, content→unified). Lint + pytest.

## Rules to read

`.claude/rules/marker-training-recipe.md`, `.claude/rules/marker-leakage-measurement.md`,
`.claude/rules/llm-judging.md` (dose/DV), `.claude/rules/on-policy-completions.md`,
`.claude/rules/contrastive-negatives.md`, `.claude/rules/code-style.md`.
