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

## Phases (all filed; Phase 2 launched early in parallel — see Orchestration)

| Phase | Scope | Task | Status (2026-06-26) |
|---|---|---|---|
| 0+1 | Base-model chain A3.2 / A3.3 / A3.4-5 + extraction-recipe lock + single-context edge case (§1.10, C=δ_x) + within-condition coherence (§1.2) + (G1) genre-generalization (Betley vs UltraChat generic) | **#658** | `followups_running` (parent #660 via marker) |
| 2 | Fine-tune fleet + trained store (t_{C,B}) + ground-truth leakage | **#664** | **RUNNING — `p2_extract_eval`** (fleet trained; extracting + evaluating). Spawned EARLY (parallel override 2026-06-25), ≤600 GPU-h |
| 3 | A3.6–A3.10 + joint factorization (CPU) | **#665** | `proposed` (daemon spawns after #664 hits awaiting_promotion) |
| 4 | End-to-end predictor L̂ + cosine variant + baselines | **#666** | `proposed` (daemon spawns after #664, overlapped with #665) |

**Dependency reality (revised by user directives):** the default gated chain was
overridden. Phase 2 (#664) launched in parallel with #658 BEFORE the foundation
go/no-go and the recipe-freeze (training is recipe-independent — the fleet captures
all 28 layers and applies the frozen layer/`r_B` recipe at analysis). **#658 is NOT
a gate** (user, 2026-06-26): its foundation result is INFORMATIONAL (read for
write-up caveats), it does not block or go/no-go-gate Phases 2-4. The one real
dependency that DOES gate is **Phase 2 → Phase 3/4**: #665/#666 read #664's trained
store (`v⁺`, `c⁺`, `t`), so they spawn only after #664 reaches `awaiting_promotion`.

## Behavior battery — broadly-applying style behaviors added (rounds 10/11)

Three style behaviors that express across all prompts — **`concise`, `verbose`,
`casual_register`** — were added to the design (`docs/theory_assumption_test_plan.md`,
revision logs round 10 + round 11):

- **Eval-only columns `B'`** (round 10): new leakage DVs on the generic-query
  (UltraChat) battery; verbosity is one bipolar axis (`r_verbosity = D_verbose −
  D_concise`, length = judge-free continuous DV). 11 → 14 top-level eval columns.
- **Trained row families** (round 11): **B11 `verbosity`** (concise + verbose, both
  poles trained — antisymmetry NOT assumed) + **B12 `casual_register`**; lean grid
  ~40 GPU-h.

**Injection path (#664 fleet is already trained):** because Phase 2 (#664) is past
training and into `p2_extract_eval`, these behaviors do NOT ride the original fleet.
They fold onto #664 as a **same-issue follow-up round** (same question — does
trained/eval behavior leakage stay predictable, on more behaviors): the B11/B12
adapters train as a fresh parallel batch + fold into #664's store, and the new `B'`
columns re-score on the existing + new fleet. Pending: file the `epm:followup-scope`
on #664 (#664 already carries one user follow-up, `em-provenance-robustness`).

## Orchestration

- **Driver is a BASH DAEMON, not a Claude/ScheduleWakeup session:**
  `scripts/run_program_orchestrator.sh` in tmux `eps-program` (started 2026-06-25).
  Polls every 120s; logs to `.claude/cache/program_orchestrator.log` (NOT to #660
  events.jsonl — a quiet #660 feed does NOT mean the loop is dead; check the log +
  `pgrep -af run_program_orchestrator.sh` + `tmux ls`). Stop:
  `touch .claude/cache/program_orchestrator.STOP`.
  Flow: spawn each phase via `/issue --auto`; advance when a phase reaches
  `awaiting_promotion`/`completed` (critic-gated PASS), NOT on promotion.
  #665/#666 already EXIST (proposed) — the daemon spawns, it does not create.
- **Not crash-recovered:** the per-phase `/issue --auto` sessions ARE respawned by
  the autonomous-session watcher; this single bash process is NOT. If it dies, phase
  ADVANCEMENT stops (active phase keeps running + parks). Recovery:
  `tmux new -d -s eps-program 'bash scripts/run_program_orchestrator.sh'`
  (idempotent — re-checks statuses, won't double-spawn).
- **Promotion stays user-only** — the loop reads clean-results, never promotes.

Children filed with `--parent <this task>`. Caveat: #664's frontmatter `parent`
field currently reads unset — lineage rides on the title + the status cache; harmless
to the run.
