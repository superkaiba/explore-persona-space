---
name: Carry-over staging recipe omits R_eval.json (only R_train)
description: Round-2 carry-over staging recipe for #504 only staged R_train.json but i504_eval_trajectory.py + i504_run_cell.py both default --r-eval-path to data/issue_472/on_policy_R/R_eval.json; full sweep crashes at first cell's eval phase
type: feedback
---

The #504 carry-over staging recipe (from #472 artifacts on HF Hub) lists 5 files: persona_bank.json + 3 centroids (L10/L15/L20) + R_train.json. It OMITS `R_eval.json` even though `i504_run_cell.py:86` and `i504_eval_trajectory.py:53` BOTH hard-default `--r-eval-path = data/issue_472/on_policy_R/R_eval.json`. The dispatcher reads from local disk; the smoke phase doesn't touch eval (so the gap goes undetected through Phase 0.5 + Phase 0); the FULL sweep crashes in the eval trajectory loop after training a cell. Cost: 1 full round of pod time.

**Why:** Hand-curated staging recipes drift from dispatcher argparse defaults. The post-#468 workflow-fix on `experimenter.md § Before Running item 4` extends this gate to introspect argparse defaults + auto-stat-check + auto-stage — this is the canonical defense.

**How to apply:** On any carry-over launch where the dispatcher reads inputs from a prior issue's data dir, do NOT trust the brief's staging recipe verbatim. Run `--help` AND grep `add_argument.*Path\(.*data/issue_<M>` across the dispatcher + every script it shells out to (`i<N>_run_cell.py`, `i<N>_eval_trajectory.py`, phase scripts). For each LOCAL filesystem default whose path includes a prior-issue data dir, stat-check on the pod. If missing AND an HF mirror exists at `superkaiba1/explore-persona-space-data`, stage it. If missing AND no HF mirror, post `epm:failure v1 infra reason: dispatcher-default-path-no-hf-mirror`.

Burned at #504 v10 launch (2026-06-06) — round-9 ran smoke clean (3 cells × 1 seed Phase 0) but would have crashed at first sweep cell's eval trajectory. v10 staged R_eval.json (5.4 MB) from `issue472_neg_geometry/on_policy_R/R_eval.json` on dataset repo, verified all dispatcher defaults, launched.
