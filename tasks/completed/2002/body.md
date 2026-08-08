---
title: 'daily-fix: smoke contract: resume/salvage-leg matrix + real-'
kind: infra
tags:
- wf-fix
- wf-fix-fp:39fb321ab922
- daily-auto-filed
created_at: '2026-08-02T07:11:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): #1947/#1979: 11+ distinct
  production defects in 3-round-PASS scripts, concentrated in resume/topup/salvage
  re-entry branches the smoke contract never exercises (recorded-verdict re-entry
  crash; salvage overwrote judge_raw_pos.json; missing mkdir + reused fit-core registry
  seam = 2 lost rounds); the r4 smoke-first mechanism caught defect #4 in 13 min once
  added.'
workflow: v1
---
# daily-fix: smoke contract: resume/salvage-leg matrix + real-unit run

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C7 (miners 4, 6, 8; sessions 8fc069db + 75f66748 (#1947), 24f7b592 (#1979)).

## Goal
Extend the experiment-implementer smoke-architecture contract with (a) a RESUME-MATRIX requirement — resume-from-partial-state / topup / salvage legs of multi-stage datagen and dispatcher scripts are exercised pre-launch, and (b) one REAL corpus/fit unit run end-to-end into the PRODUCTION out-root before the full launch.

## Workflow gap
- **Bug observed:** fully reviewed (3-round PASS) scripts crashed one-defect-per-launch in production, defects concentrated in unexercised resume/salvage branches: #1947 had 5 distinct P0/datagen crashes over ~5h (5 `epm:failure` firings 13:21Z–17:10Z) — crash #4: a recorded survivable yield-floor verdict became a hard crash on re-entry; crash #5: salvage overwrote `judge_raw_pos.json` before its own guard refused; #1947 P4/P5 lost 2 more rounds to a missing mkdir + the reused #1768 fit-core registry-lookup seam (both would have surfaced on one real unit into the production out-root). #1979 had 6+ crash rounds (EBADF → C2; a fail-fast persona-label seam draining the whole 8-GPU job — anticipated by a reviewer r1 MINOR "not applied pre-launch"; KeyError 'logp_marker' guessed key; 1.99<2.0 threshold; first-marker-occurrence mis-gating ICL arms). The r4 smoke-first mechanism demonstrably cut blast radius once added (caught defect #4 in 13 min).
- **Why it is a workflow gap:** the smoke-architecture contract mandates smoke=sweep unification, per-phase end-to-end smokes, cross-phase data-contract smokes, and real-corpus streaming smokes — but never requires exercising the RESUME/salvage re-entry legs or running one real unit into the production out-root, exactly where the day's 11+ defects sat.
- **Confidence:** medium-high (incident mechanisms miner-inferred from session forensics; the contract gap re-verified by grep).
- verified-at-filing: `grep -n 'smoke\|resume\|salvage\|out-root' .claude/agents/experiment-implementer.md` → extensive smoke contract (:116-296 unification + four additional requirements: cross-phase data-contract, real-trainer-path, subprocess-dispatcher, real-corpus ingestion; :517 per-phase end-to-end smokes; :452-474 smoke-fenced import/branch probes) but ZERO hits for salvage, ZERO for out-root, and the only resume hits (:393, :411) concern chunk persistence / poisoned-context refusal — no requirement to exercise resume/topup/salvage re-entry pre-launch. `ls .claude/agent-memory/experiment-implementer/` → per-agent memories already persisted in-session (feedback_resume_predicate_recorded_terminal_verdicts.md, feedback_salvage_inputs_pin_identity.md, feedback_reused_fit_core_registry_lookup_seam.md) — memories are guidance, not the gated contract; the code-reviewer's `smoke-run-missing` blocker cannot fire on a leg the contract never names. `git log --oneline --since='7 days ago' -- .claude/agents/experiment-implementer.md` → 1 commit (9f5df33255, real-corpus smoke-slice probes) — adjacent, does not cover resume/salvage (2026-08-02).

## Proposed change (refine in planning)
1. Add a fifth smoke-contract requirement, **Resume-matrix smoke**: for any script with resume / topup / salvage / re-entry branches (a resume predicate, a `--from-phase` flag, a salvage leg, recorded-verdict re-reads), the pre-launch smoke exercises EACH such leg at least once against a synthesized partial state (run the smoke, interrupt/seed partial outputs, re-enter) — a leg that cannot be brought to the smoke floor is declared in the `epm:smoke-architecture-check` marker with why, mirroring the existing REAL/FALLBACK/N-A arm-status vocabulary.
2. Add **one real production unit into the production out-root**: before the full launch, exactly one real corpus/fit unit runs end-to-end writing to the PRODUCTION out-root (not a /tmp twin) — catching mkdir, registry-lookup, and path-predicate seams (#1947 P4/P5); compose with the existing PASS_CANARY escape for expensive cells.
3. Fold in the persisted agent memories as the incident citations; note (process, no rule change): reviewer MINORs naming single-point-of-failure dispatcher behavior deserve pre-launch application.

## Scope / surfaces
- Primary target: `.claude/agents/experiment-implementer.md`
- Coordinate wording with `.claude/agents/code-reviewer.md`'s `smoke-run-missing` blocker definition and the `epm:smoke-architecture-check` marker fields (grep both before editing); keep the marker schema backward-compatible.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Do not inflate smoke cost unboundedly — the resume-matrix requirement is per-BRANCH-CLASS, with the same declared-escape shape the contract already uses for un-smokeable slices.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 39fb321ab922
- workflow_fix_target: .claude/agents/experiment-implementer.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C7.
