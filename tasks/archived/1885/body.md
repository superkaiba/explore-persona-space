---
title: 'workflow-fix: lane-aware carry-over gate — committed inputs unshippable on
  SLURM rsync lanes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ad635676e4c3
created_at: '2026-07-30T15:34:52Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1689 (2026-07-30, fellows job 15724
  wp_fence crash): verify_carryover_inputs passes git-reachable committed eval_results
  inputs that the SLURM lane''s rsync include/exclude set never ships to the node;
  add a lane-aware slurm-lane-unshipped class + a planner section-9 lane-constraint
  line'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1689 (emitting agent: issue-orchestrator).

## Goal

Make the Step 6a.5 carry-over gate and planner section-9 lane constraints SLURM-rsync-aware: a plan-cited committed eval_results (or other RSYNC_EXCLUDE-dir) input consumed by the workload FAILs or WARNs when the dispatch lane can be SLURM, directing upload-first HF staging.

## Workflow gap

- **Bug observed:** A committed eval_results parent input consumed by a fellows-lane workload passed verify_carryover_inputs (git-tree-reachable on the pushed ref) and every plan review, then crashed the run FileNotFoundError on the node - the SLURM lane rsyncs only RSYNC_INCLUDE_PATHS and excludes eval_results wholesale.
- **Why it is a workflow gap:** The #734 artifact-reuse check (h)(iii) names target-backend fetchability, and verify_carryover_inputs.py exists to catch unreachable plan-cited local inputs — but its predicate is git-reachability on the pushed ref, which is TRUE for a committed file the SLURM lane still never ships (`backends/slurm.py` `RSYNC_INCLUDE_PATHS` = pyproject/uv.lock/src/scripts/configs/external/tests/data-sft; `RSYNC_EXCLUDE_PATTERNS` includes `eval_results/`). The GCE lane git-clones the branch (full tree) so the same plan is lane-dependent-correct — exactly the silent class the gate should classify. Incident #1689 (2026-07-30): fellows job 15724 died at the fence phase on `eval_results/issue_1689/analyzer/dvf_unit_digest.csv`; the leg consumed FOUR such inputs (the paired-digest ambient-root defaults would have crashed next); ~1.3 GPU-h + a full launch cycle burned; plan v10 had been critic-APPROVEd 0 Must-Fix with the digest's CONTENT verified but not its lane fetchability.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "slurm\|SLURM\|rsync\|lane" scripts/verify_carryover_inputs.py` → 6 hits, none implementing a lane-aware classification (the hits are docstring references to lane materialization, not an rsync-exclude predicate); `RSYNC_INCLUDE_PATHS`/`RSYNC_EXCLUDE_PATTERNS` confirmed in `src/explore_persona_space/backends/slurm.py` (`eval_results/` excluded) (2026-07-30)

## Proposed change (candidate diff sketch — refine in planning)

1. `scripts/verify_carryover_inputs.py`: add a lane-fetchability classification — when the task's `backend:` frontmatter is absent/`auto`/a SLURM lane (fellows/nibi/fir/mila reachable), a plan-cited repo-local input whose path matches a SLURM `RSYNC_EXCLUDE_PATTERNS` dir (import the tuples from `backends.slurm` — single source) and is NOT under an `RSYNC_INCLUDE_PATHS` root emits a new failure class `slurm-lane-unshipped` (recoverable: remediation = upload-first to the HF data repo + leg-side staging, or pin `backend: gcp`). Keep the existing git-tree checks unchanged.
2. `.claude/agents/planner.md` §9: add the lane-constraint line next to the existing /workspace-sentinel constraint — a workload consuming committed `eval_results/`/`figures/`/`docs/` files must either pin a git-clone lane (gcp/runpod) or declare HF staging for those inputs.

## Scope / surfaces

- Primary target: `scripts/verify_carryover_inputs.py,.claude/agents/planner.md`
- Grep the workflow surface before editing (`grep -rln 'RSYNC_EXCLUDE_PATTERNS' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every consumer that documents the 6a.5 gate's coverage (SKILL.md § Step 6a.5 second stanza names the residual list — update it to name this closed class).

## Constraints / invariants

- Workflow-surface only. `tests/test_verify_carryover_inputs*.py` (if present) extended with the new class; a WARN-vs-FAIL decision for the `auto` lane (which MAY resolve to GCP) belongs to the planner session.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_carryover_inputs.py,.claude/agents/planner.md
- fingerprint: ad635676e4c3

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_carryover_inputs.py,.claude/agents/planner.md
bug_observed: A committed eval_results parent input consumed by a fellows-lane workload passed verify_carryover_inputs (git-tree-reachable on the pushed ref) and every plan review, then crashed the run FileNotFoundError on the node - the SLURM lane rsyncs only RSYNC_INCLUDE_PATHS and excludes eval_results wholesale
why_workflow_gap: The gate's predicate is git-reachability, which is lane-independent; SLURM lanes ship a restricted include set, so a committed input can be reachable-on-ref yet unshippable — the #734 (h)(iii) target-backend fetchability check has no mechanical enforcement for this class (incident #1689 job 15724, 4 such inputs, ~1.3 GPU-h burned)
proposed_change: Lane-aware classification in verify_carryover_inputs.py (new `slurm-lane-unshipped` class keyed on backends.slurm RSYNC tuples when the lane can be SLURM) + a planner.md section-9 lane-constraint line mirroring the /workspace-sentinel precedent
diff_sketch: |
  + from explore_persona_space.backends.slurm import RSYNC_INCLUDE_PATHS, RSYNC_EXCLUDE_PATTERNS
  + # after the git-tree check passes, classify lane fetchability:
  + if lane_may_be_slurm(frontmatter_backend) and _matches_exclude(path) and not _under_include_root(path):
  +     failures.append(("slurm-lane-unshipped", path, "upload-first to HF + leg staging, or pin backend: gcp"))
confidence: high
related_task: #1689
<!-- /workflow-fix-candidate -->
