---
title: '[Program] Leakage-predictor theory: empirically test all 10 assumptions (A3.1–A3.10)
  on Qwen2.5-7B (4-phase)'
kind: survey
tags:
- program-tracker
created_at: '2026-06-24T21:33:57Z'
has_clean_result: false
origin_prompt: Design a comprehensive experimental plan to test all the assumptions
  [of the leakage-predictor theory].
---
## Overview

Program to empirically test **all assumptions (A3.1–A3.10)** of the leakage-predictor
theory on **Qwen2.5-7B-Instruct only**.

- Design: `docs/theory_assumption_test_plan.md`
- Full theory (pinned snapshot): `docs/leakage_theory_paper.tex` (canonical: Overleaf `6a2df2d2`)
- Live orchestration status: `.claude/cache/theory-program-status.md`

This task is the **dashboard home + lineage parent** for the program. It is
tracking-only (`kind: survey`) — it is NOT itself executed. Each phase runs as its
own `kind: experiment` task via a dedicated `/issue <N> --auto` session.

## Phases (sequenced `/issue --auto`, strict dependency chain)

| Phase | Scope | Task |
|---|---|---|
| 0+1 | Base-model chain A3.2 / A3.3 / A3.4-5 + extraction-recipe lock + single-context edge case (§1.10, C=δ_x) + within-condition coherence (§1.2) + (G1) genre-generalization: Betley vs UltraChat generic queries (follow-up, runs unconditionally / in parallel — does NOT wait on the Betley verdict) | **#658** [RUNNING] |
| 2 | Fine-tune fleet + trained store (t_{C,B}) + ground-truth leakage | not filed |
| 3 | A3.6–A3.10 + joint factorization (CPU) | not filed |
| 4 | End-to-end predictor L̂ + cosine variant + baselines | not filed |

Phases are a dependency chain: Phase 1 locks the extraction recipe → Phase 2 needs it →
Phase 2's trained store feeds Phase 3/4. They cannot run in parallel.

## Orchestration

- **Meta-loop** (ScheduleWakeup, the PM-style session): advances phases on a clean PASS;
  owns the go/no-go gate + files the next phase. Foundation failure HALTS + surfaces.
- **Per-phase**: each phase = its own autonomous `/issue <N> --auto` session running the
  full subagent pipeline (planner → adversarial-critic → experiment-implementer →
  code-reviewer → experimenter → analyzer → interp/clean-result critics →
  awaiting_promotion). Crash-recovered by the autonomous-session-watcher cron.
- **Promotion stays user-only** — the loop reads clean-results, never promotes.

Children filed with `--parent <this task>`. Loop updates this body as phases land.
