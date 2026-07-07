---
title: 'artifacts factory Phase 1: validate unified pipeline + calibrate cost on 4
  pilot classes'
kind: batch
tags: []
created_at: '2026-07-03T02:21:53Z'
has_clean_result: false
origin_prompt: Autonomous orchestration of the unified artifact factory; plan /home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md
  (Phase 1 pilot / calibration). Clinical body — generates registry-defined attribute-class
  data via the merged artifacts library.
---
## Overview

Phase 1 of the reusable-artifact factory (plan:
`/home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md`). Exercise the now-merged
`artifacts/` library end-to-end on a small **validation set of 4 representative attribute classes** and
**calibrate the real per-unit cost** (GPU-hours + generator/judge API-call counts) before the full-scale
fleet. This is a pipeline-validation + calibration run — not a research finding.

## Goal

For each of the 4 pilot classes defined in the `BEHAVIORS` registry (`artifacts/behavior.py`) —
`sycophancy`, `harmful_compliance`, `china_censorship`, `marker` — chosen to span the type-space
(persona-trait / diff-of-means / inverted-base / programmatic):

1. **Build one model organism** via `artifacts.organisms` — the `primary` arm (contrastive negatives ON +
   generic interleave ON), one source context C, one seed, dose-to-target on the source rate
   (`artifacts.recipe`). Marker uses its log-prob band-stop carve-out.
2. **Verify** via `organisms.verify`: the attribute's judged rate under C vs the base model at C (the
   install check), plus the bystander-context panel (the leakage check), dual-DV (graded rate primary +
   the `eval/margin` teacher-forced companion).
3. For the 3 non-programmatic classes, run `artifacts.directions` (`r_B` extraction) and **confirm it
   reproduces the saved reference direction** where one exists (cosine ≥ threshold vs the on-disk
   `sycophancy` / `refusal` r_B from `data/issue_779|778|661`) — a reuse-sanity gate.
4. **Report the realized per-unit cost**: GPU-hours per organism + per direction, generator + judge
   API-call counts per class, install + leakage numbers, and any API refusals encountered. This
   calibrates the Phase-2/3 estimates.

## Compute routing (per CLAUDE.md #747 + the plan's routing rule)

- Organism TRAINING + direction teacher-forcing + vLLM eval-generation → GPU lane (`lora-7b` / `eval`)
  via the backend router (`dispatch_issue.py`).
- Data generation + judging → Claude API / Batch API only (no pod; API-bound, `claude-sonnet-4-5-20250929`).
- Aggregation / any bootstrap-null / plotting → CPU: a `cpu-small`/`cpu-mid` pod if non-trivial, else the VM
  for trivial sub-minute ops (the #747 carve-out).

## Scope / constraints

- Data provenance = generator-model (Claude) then judge-filtered (the standing D1 decision — carry as a
  data-realism scope note; the class definitions + rubrics live in `artifacts/behavior.py`).
- Defaults for this pilot (the two open fleet-scale choices do NOT bind here): the default assistant is a
  bystander in the leakage panel; one dose (the target band). These firm up before Phase 3.
- Reuse the merged library — do NOT reimplement datagen/recipe/organisms/directions.
- Per-cell upload of artifacts (adapters → HF model repo, raw generations + judge outputs → HF data repo)
  the moment each cell completes; never a terminal batch.
- No dollar-budget caps; WandB live training metrics; checkpoint-per-phase.

## Success / kill

- SUCCESS: each of the 4 organisms installs (rate at C > base at C) with a bounded bystander-leakage read
  and a fired dose-to-target stop; each direction extracts; the reproduction gate passes; the calibration
  report is written.
- A class that cannot reach its install target under the recipe is REPORTED as a calibration finding (its
  yield / dose behavior), not silently dropped.

## Rules to read

`.claude/rules/contrastive-negatives.md`, `.claude/rules/marker-training-recipe.md`,
`.claude/rules/marker-leakage-measurement.md`, `.claude/rules/llm-judging.md`,
`.claude/rules/on-policy-completions.md` (D1 deviation), `.claude/rules/persona-vectors-recipe.md`
(direction reproduction), `.claude/rules/selection-symmetric-nulls.md`, the plan. Author all wording
clinically/functionally — the class definitions live in the registry; reference them.
