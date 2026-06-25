---
title: 'Phase 2 — fine-tune fleet + trained store + ground-truth leakage (leakage
  program #660)'
kind: experiment
tags: []
created_at: '2026-06-25T07:26:49Z'
has_clean_result: false
parent_id: 660
goal: 'Phase 2 — fine-tune fleet: train source×behavior×arm adapters (positive-only
  + contrastive arms per B5; on-policy completions; marker band-stop), build the trained
  store (t_CB, v+(C''), r+_B''), measure ground-truth leakage, using Phase 1''s locked
  layer + the C-primary r_B recipe.'
---
## Goal

Phase 2 — fine-tune fleet: train source×behavior×arm adapters (positive-only + contrastive arms per B5; on-policy completions; marker band-stop), build the trained store (t_CB, v+(C'), r+_B'), measure ground-truth leakage, using Phase 1's locked layer + the C-primary r_B recipe.

## Design
Designed by /adversarial-planner at dispatch from docs/theory_assumption_test_plan.md (§3 Phase 2 + §4) AND Phase 1 (#658) clean-result (locked layer + C-primary r_B recipe). READ docs/leakage_theory_paper.tex first. Largest-parallelism phase (all cells concurrent, §10e). Autocontinue.

## Reuse — REUSE-FIRST (planner: run the artifact-reuse fitness check (a)-(g) at dispatch)

**PRIMARY reuse target: #537 context-generalization testbed.** 84 contrastive LoRA adapters
(5 behaviors {marker, fact, refusal, sycophancy, EM} × 16 train-contexts
{persona / WildChat / ICL / rephrasing / format / default / inoculation}, seed 42) + 32
seed-1042 (marker + fact). Qwen-2.5-7B-Instruct, marker ` ※` band-stopped 5–12 nat,
contrastive-everywhere. On HF: `adapters/i537_*` (Hub-verified); the G[behavior, train→eval]
ground-truth grid + raw completions + activation clouds at `issue537_context_generalization/`
(data repo). HIGH confidence, seed-1042 reproduced (marker r=0.999).

REUSE the matching behavior×context cells (saves the bulk of the contrastive-arm fleet GPU);
**TRAIN ONLY THE GAPS:**
- **B5 positive-only arm** — #537 is contrastive-ONLY; the positive-only identification arm is
  absent → train fresh.
- **Contexts** — #537's 16 train-contexts are a SUBSET of the program's 50-context #594 battery →
  reuse the overlap, train the remainder.
- **Behaviors** — #537's 5 vs the #545 column registry → reuse the overlap, train any additional
  behaviors the design requires.
- **Re-eval on-policy** — #537's marker eval was teacher-forced (Δ log P at a fixed slot); the
  program reads the on-policy activation gate ĝ^real. REUSE the adapters but RE-EXTRACT
  v⁺(C') / ĝ^real on-policy under them (the stored teacher-forced eval is NOT reused).

Fitness check per `.claude/rules/artifact-reuse.md` (a)–(g): recipe match on each adapter's
`adapter_config.json`; marker token = ` ※` (id 83399, NOT `[ZLT]` — #537's final retrain used ※,
verify per-cell); required cells present; content / sha-pin; rsLoRA application-scaling parity.
Any check that fails for a cell → retrain THAT cell, and say why.

**SECONDARY candidates:** #474 (16 epoch-1 non-saturated marker adapters, the canonical marker
spine reused by #532/#545/#601/#650); #545 (behavior testbed / column registry). #537 is the
primary — the only explicit behavior×context adapter fleet.
