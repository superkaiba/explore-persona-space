---
title: 'workflow-fix: forward HF_HUB_DISABLE_XET in lane allowlists'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d78aae0cb3d8
- daily-auto-filed
created_at: '2026-07-09T07:00:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The fleet-wide xet-disable
  override channel forwards only HF_XET_DISABLE — consumed by neither huggingface_hub
  0.36.2 (which reads HF_HUB_DISABLE_XET; live-tested) nor the hf_xet Rust binary
  — so a launch-time xet disable is a no-op locally and cannot reach GCP/SLURM workers
  at all.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1049 (recursion-guarded workflow-fix session).

## Goal

Add HF_HUB_DISABLE_XET to both STARTUP_PASSTHROUGH_ENV_KEYS allowlists (keep HF_XET_DISABLE as a legacy alias), correct the HF_XET_DISABLE=1 prescriptions in bootstrap_pod.sh comments (l.345/366) and gotchas.md (and orchestrate/env.py ride-along), and add a test pinning that both allowlists contain the real var.

## Workflow gap

- **Bug observed:** The fleet-wide xet-disable override channel forwards only HF_XET_DISABLE — consumed by neither huggingface_hub 0.36.2 (which reads HF_HUB_DISABLE_XET; live-tested) nor the hf_xet Rust binary — so a launch-time xet disable is a no-op locally and cannot reach GCP/SLURM workers at all.
- **Why it is a workflow gap:** The dispatch layer's documented escape hatch for xet CDN failures (#515/#825/#931) is wired to a dead env var, so every past and future 'disable xet' mitigation silently never engages.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

# backends/gcp.py STARTUP_PASSTHROUGH_ENV_KEYS (~l.1034), backends/slurm.py (~l.804):
+    "HF_HUB_DISABLE_XET",
     "HF_XET_DISABLE",   # legacy no-op alias; kept so old launch commands don't error
# bootstrap_pod.sh l.345/366 + gotchas.md: HF_XET_DISABLE=1 -> HF_HUB_DISABLE_XET=1
# tests/test_backend_*: assert "HF_HUB_DISABLE_XET" in STARTUP_PASSTHROUGH_ENV_KEYS (both lanes)

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py, src/explore_persona_space/backends/slurm.py, scripts/bootstrap_pod.sh, .claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py, src/explore_persona_space/backends/slurm.py, scripts/bootstrap_pod.sh, .claude/rules/gotchas.md
- origin: parked candidate on task #1049 at 2026-07-05T13:41:35Z

Verbatim parked note:

parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). NOT auto-routed; surfaced for the next orchestrator/human pass.

source: candidate-block (planner return, plan v1)
routed: parked: EPM_WORKFLOW_FIX_SESSION

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py, src/explore_persona_space/backends/slurm.py, scripts/bootstrap_pod.sh, .claude/rules/gotchas.md
bug_observed: The fleet-wide xet-disable override channel forwards only `HF_XET_DISABLE` — a variable consumed by neither huggingface_hub 0.36.2 (which reads `HF_HUB_DISABLE_XET`; live-tested) nor the hf_xet Rust binary (strings-checked) — so a launch-time xet disable is a no-op locally and cannot reach GCP/SLURM workers at all (STARTUP_PASSTHROUGH_ENV_KEYS at gcp.py:967 / slurm.py:804 omit HF_HUB_DISABLE_XET).
why_workflow_gap: The dispatch layer's documented escape hatch for xet CDN failures (#515/#825/#931) is wired to a dead env var, so every past and future "disable xet" mitigation silently never engages.
proposed_change: Add HF_HUB_DISABLE_XET to both passthrough allowlists (keep HF_XET_DISABLE for back-compat or drop it), correct the HF_XET_DISABLE=1 prescriptions in bootstrap_pod.sh comments (lines 345/366) and gotchas.md:187 (and orchestrate/env.py:315/374, out-of-scope src, ride-along), and add a test pinning the allowlist contains the real var.
diff_sketch: |
  # backends/gcp.py STARTUP_PASSTHROUGH_ENV_KEYS (~l.967), backends/slurm.py (~l.804):
  +    "HF_HUB_DISABLE_XET",
       "HF_XET_DISABLE",   # legacy no-op alias; kept so old launch commands don't error
  # bootstrap_pod.sh l.345/366 + gotchas.md l.187: HF_XET_DISABLE=1 -> HF_HUB_DISABLE_XET=1
  # tests/test_backend_*: assert "HF_HUB_DISABLE_XET" in STARTUP_PASSTHROUGH_ENV_KEYS (both lanes)
confidence: high
related_task: #1049
<!-- /workflow-fix-candidate -->

