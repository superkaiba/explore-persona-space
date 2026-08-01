---
title: 'Single-visit organism fleet: no training row repeated; theory assumptions
  tested on exactly the trained rows'
kind: experiment
tags: []
created_at: '2026-07-31T22:12:00Z'
has_clean_result: false
parent_id: 1768
origin_prompt: how long would it take to retrain so we never have to repeat data?
  (and then compare the assumptions ONLY on the data that was trained on) -> run it
  and tell it to paralelize/use batch api as much as possible
workflow: v1
goal: 'Retrain the content-behavior organism fleet under a single-visit data regime
  (every training row consumed exactly once — ~1,200 unique rows per cell covering
  the full 75-step schedule; pilot cell gates the fleet on reaching the 0.60-0.85
  install band), then run the theory-assumption battery (write vs delta/r_B, write
  rank, base-geometry gate, map change) evaluated on exactly the trained rows (TF
  base+trained on the gradient-producing rows, delta over the same ~300-row set) alongside
  the standard panel/corpus surfaces, to decide whether the #1768 assumption failures
  survive without data repetition and on-distribution.'
relates_to:
- identity-contextual-vs-base
- leak-predictor
---
# Single-visit (no-repeat) organism fleet: retrain with enough unique data that no training row is ever repeated, then test the theory assumptions on exactly the trained rows

## Provenance

- origin_prompt (verbatim): "how long would it take to retrain so we never have to repeat data? (and then compare the assumptions ONLY on the data that was trained on)" → "run it and tell it to paralelize/use batch api as much as possible"
- Parent line: #1481 (organism fleet, the current 80/60-row × 15-epoch recipe), #1768 (the assumption battery this feeds). Chat context 2026-07-31: current verdict checkpoints are effectively 1–8 epochs (median 3–4) over ~20 behavior-expressing exemplars; δ is a 20-row mean (split-half ≈ 0.55).

## Goal

Retrain the content-behavior organism fleet under a single-visit data regime (every training row consumed exactly once — ~1,200 unique rows per cell covering the full 75-step schedule; pilot cell gates the fleet on reaching the 0.60-0.85 install band), then run the theory-assumption battery (write vs delta/r_B, write rank, base-geometry gate, map change) evaluated on exactly the trained rows (TF base+trained on the gradient-producing rows, delta over the same ~300-row set) alongside the standard panel/corpus surfaces, to decide whether the #1768 assumption failures survive without data repetition and on-distribution.

## Design sketch (planner refines)

- **No-repeat = expand data, not cut steps.** In-band install currently needs 80–640 row-visits (batch 16, steps 5–40); a true single pass over 80 rows never installs. Target ~1,200 unique rows per (behavior × context) cell to cover the full 75-step schedule with margin: ~300 judge-accepted on-policy positives (elicitation ladder, standardized persona-vectors-shape behavior definitions), contrastive negatives scaled to preserve the ~1:1 positives-to-total-negatives recipe (5-member factory panel, on-policy, incl. default assistant), fresh generic rows. Mix builder/sampler asserts each row is consumed exactly once (hard assert, fail loud).
- **PILOT GATE before fleet fan-out:** one cell end-to-end (datagen → train → per-rung judged install) must reach the 0.60–0.85 band under single-visit data; fleet is conditional on the pilot.
- **Disambiguation arm (planner decides scope):** unique-rows-at-matched-visit-count (e.g. 640 unique × 1 visit vs the parent's 80 × 8 visits) to separate "no repetition" from "more diverse data" — otherwise the two change together (acknowledged inherent confound of the main arm).
- **Fleet scope:** 3 content behaviors × 4 training contexts × 2 regimes × 2 seeds = 48 cells (+ marker cells with programmatically generated single-visit data, trivial datagen). Same recipe otherwise (LoRA rank 32/α64 rsLoRA, lr per parent cell, checkpoint every 5 steps, in-band verdict selection) — the manipulated variable is the data-visit regime.
- **Assumption battery, trained-rows-only:** TF base + trained checkpoints on the exact trained rows (each seen once in training); δ over the SAME rows (now a ~300-row mean, vs 20 today); horse race (disjoint-baseline discipline), write rank, gate — all on-distribution. Plus the standard 20-question panel + a corpus read for cross-surface comparison (reuse #1768 rigs: issue1768_directions.py conventions, ckpt-dynamics machinery for per-rung curves).

## Constraints (user directive — binding)

- **Parallelize aggressively at every phase:** vLLM fan-out for datagen generation; all cell trainings sharded across every provisioned GPU (width-aware wide provisioning encouraged — wall-clock is the scarce resource); per-rung eval generation batched via vLLM; no serial per-cell loops anywhere.
- **Anthropic Batch API for EVERY judge set** (datagen judge-filtering, per-rung install evals, any battery judging) — never sync loops for large sets; judge = claude-sonnet-4-5-20250929; graded 0–100 multi-draw where the DV is a ranking target; drop-never-coerce; rubric-keyed caches.
- Reuse the #1481 factory machinery (mix builders, ladder trainer, install-eval rig) and the #1090-rebuild standardized behavior definitions; new code only for the single-visit sampler + trained-rows battery glue.
- Est ~30–50 GPU-h + datagen/eval judge spend (Batch API); wall ~2–4 days.
