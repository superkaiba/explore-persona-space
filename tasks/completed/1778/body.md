---
title: verify_artifacts_exist crashes on planned-output hf:// globs + wrong repo_type
  (Step 6a.5 gate)
kind: infra
tags: []
created_at: '2026-07-29T00:11:55Z'
has_clean_result: false
origin_prompt: '/issue 1482 early-layer-arm Step 6a.5: verify_artifacts_exist RepositoryNotFoundError
  crash on pooled_l3_*.npz planned-output row'
workflow: v1
---
## Overview / Motivation

Auto-filed from /issue 1482 (early-layer-arm round, Step 6a.5): `orchestrate.hub.verify_artifacts_exist` CRASHED (RepositoryNotFoundError traceback) instead of returning a missing-list when the plan under scan cites (a) an hf:// path that is the plan's own PLANNED OUTPUT (a glob under a prefix that does not exist yet), and (b) a dataset repo id probed as repo_type=models (404 on api/models/superkaiba1/explore-persona-space-data).

## Goal

verify_artifacts_exist returns (ok, missing) fail-soft on unresolvable hf:// citations — classifying planned-output/glob paths and wrong-repo-type probes as reportable rows (or skipping own-issue planned outputs like verify_carryover_inputs.py does) — never an uncaught traceback that aborts the Step 6a.5 gate.

## Workflow gap

- **Bug observed:** `uv run python -c "from explore_persona_space.orchestrate.hub import verify_artifacts_exist; verify_artifacts_exist(plan_path='tasks/followups_running/1482/plans/plan.md')"` raised RepositoryNotFoundError (2026-07-29) on `issue1482_error_analysis/analysis_tensors/early_layer/pooled_l3_*.npz` — a §6.5 planned-output row.
- **Why it is a gap:** the Step 6a.5 launch gate cannot distinguish "carry-over missing" from "helper crashed"; a crash forces a manual judgment call at every launch whose plan lists hf:// planned outputs.
- **Confidence (emitter):** high
- verified-at-filing: reproduced live this session on plan v16 (traceback above); `grep -n 'def verify_artifacts_exist' src/explore_persona_space/orchestrate/hub.py` → :2483 (1 hit, construction site) (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

In `_hf_artifact_exists` / `verify_artifacts_exist`: catch RepositoryNotFoundError + glob-bearing paths; skip own-issue planned-output prefixes (mirror verify_carryover_inputs.py planned-output classification); try repo_type in (model, dataset) before concluding missing; return the row in `missing` instead of raising.

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/hub.py` (verify_artifacts_exist, :2483; _hf_artifact_exists, :2417)
- Secondary: a pin test in tests/

## Constraints / invariants

- Fail-soft on classification, fail-loud on genuine transport errors; no behavior change for resolvable citations.
