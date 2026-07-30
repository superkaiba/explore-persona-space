---
title: 'workflow-fix: add capture-7b to the fellows/SLURM lane intent tables'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a1d230223866
created_at: '2026-07-30T20:07:02Z'
has_clean_result: false
origin_prompt: 'orchestrator observation (PM chat 2026-07-30): capture-7b dispatches
  ValueError off the fellows lane and burn GCP; 3 marker occurrences since 2026-07-23'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation during a PM-chat compute-mix audit (2026-07-30). No parent task (chat-mode; candidate logged to `.claude/cache/workflow-fix-events.jsonl`).

## Goal

Make `capture-7b` dispatches eligible for the free fellows/SLURM lane instead of silently skipping it: add `capture-7b` to the SLURM intent tables (`_DEFAULT_GPUS_FOR_INTENT`: 1, `_DEFAULT_TIME_BUDGETS_HOURS`: ~4-6h, plus `stages_for_spec` if intent-keyed) — or, alternatively, a #940-style intent translation (`capture-7b` → `eval`) at SLURM candidate/prepare time, mirroring `RUNPOD_INTENT_FOR_GCP_INTENT`.

## Workflow gap

- **Bug observed:** every `capture-7b` dispatch on the auto lane raises `ValueError: no default GPU count for intent 'capture-7b'` (or the `_DEFAULT_TIME_BUDGETS_HOURS` twin) inside the fellows lane's prepare, and falls through to paid GCP. 3 marker-recorded occurrences since the fellows lane went live: `epm:backend-selected` at 2026-07-23T17:24Z, 2026-07-29T03:37Z, 2026-07-29T04:21Z.
- **Why it is a workflow gap:** `capture-7b` is a first-class single-GPU 7B GPU intent on GCP (`a2-ultragpu-1g`, #752) and the RunPod terminal rung already translates it (`RUNPOD_INTENT_FOR_GCP_INTENT["capture-7b"] = "eval"`, router.py:427, #940) — but the SLURM lanes have no row and no translation, so the standing "fellows first, free lane preferred" policy (#1609, user directive 2026-07-22) is silently bypassed for an entire intent class. The deliberate-exclusion comments in slurm.py (#1464 / #747) cover CPU-only intents ONLY; `capture-7b` is a GPU intent, so its absence is a gap, not a design decision.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "capture-7b" src/explore_persona_space/backends/slurm.py` → 0 hits (absence claim — the missing-row evidence); `grep -rn "capture-7b" src/explore_persona_space/backends/ scripts/` → 15 hits in 5 files (router.py 4, gcp.py 8, issue-scripts 3 — the intent is real, GCP-mapped, and RunPod-translated) (2026-07-30). Landed-fix history check: `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/slurm.py` shows no capture-7b-related commit; the 2026-07-29 marker occurrences postdate the #1609/#1393 fellows merges.

## Proposed change (candidate diff sketch — refine in planning)

```
# src/explore_persona_space/backends/slurm.py
 _DEFAULT_TIME_BUDGETS_HOURS: dict[str, float] = {
     ...
+    "capture-7b": 4.0,  # #752 activation-capture path; sized like eval
 }
 _DEFAULT_GPUS_FOR_INTENT: dict[str, int] = {
     ...
+    "capture-7b": 1,  # single-GPU 7B capture, matches GCP a2-ultragpu-1g row
 }
# + stages_for_spec branch if it raises on unknown intents
# ALT design (planner's call): SLURM-side intent translation mirroring
# router._translated_runpod_intent (#940), so future GCP-only intents
# fail over uniformly instead of needing per-table rows.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py` (the ValueError construction sites: `time_budget_hours`, `default_gpus_for_intent`, `stages_for_spec`)
- Secondary (ALT design): `src/explore_persona_space/backends/router.py` (SLURM candidate assembly / a translation helper)
- Grep the workflow surface for the pattern before editing (`grep -rn 'capture-7b' src/explore_persona_space/backends/ scripts/ tests/`) and update every hit needed; list them in the plan. Tests: `tests/test_slurm_backend_render.py`, `tests/test_router.py` (byte-identical-render snapshots must be respected for non-capture intents).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Non-capture-intent sbatch renders stay byte-identical (the #1609 snapshot-test contract).
- CPU-only intents stay excluded (the #1464/#747 deliberate exclusion is untouched).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py
- fingerprint: a1d230223866

Orchestrator observation (verbatim marker evidence): `epm:backend-selected` attempts with `{"cluster": "fellows", "detail": "ValueError: no default GPU count for intent 'capture-7b'. Supported intents: ['debug', 'eval', 'ft-70b', 'ft-7b', 'inf-70b', 'lora', 'lora-7b']. Pass an explicit ``gpus=`` in the RunSpec or add the intent to ``_DEFAULT_GPUS_FOR_INTENT``..."` (2026-07-23T17:24Z) and the `_DEFAULT_TIME_BUDGETS_HOURS` twin (2026-07-29T03:37Z, 2026-07-29T04:21Z).
