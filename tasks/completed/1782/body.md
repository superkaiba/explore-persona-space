---
title: 'workflow-fix: direction-agnostic cross-phase reads[] in off_pod_phases (pod
  reads VM outputs)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:19ce66a6b9cd
created_at: '2026-07-29T01:49:42Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed during #1773 passB crash triage: off_pod_phases
  spec direction gap (pod/GCE phase reading VM-produced inputs undeclared)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1773 (emitting agent: orchestrator, own observation during crash triage).

## Goal

Generalize the planner `off_pod_phases` declaration to a direction-agnostic cross-phase-reads rule: every dispatched phase (pod/GCE/SLURM AND off-pod alike) that reads another phase's outputs declares `reads[]` with a fetchable permanent source + the staging step.

## Workflow gap

- **Bug observed:** #1773 Pass B (GCE) crashed `FileNotFoundError` loading Pass A's VM-produced selection outputs; the `off_pod_phases` spec only requires OFF-POD consumers to declare reads of pod outputs, so the inverse pod-reads-VM seam passed plan review and c39 by construction.
- **Why it is a workflow gap:** the §9 block (planner-section-reference.md L624, #1535) is direction-specific — "REQUIRED only when the plan has a pod/backend dispatch AND >=1 SUBSEQUENT off-pod phase"; a GPU-lane phase consuming VM-produced inputs (a selection manifest under gitignored `data/`) has NO declaration requirement, and the git-clone lanes stage nothing outside the pushed branch (#734/#1434 class). The #1469 carry-over gate glob-skips such paths (`reason=glob-or-template`), so no mechanical gate covers the seam either. Cost: one full 4xA100 GCE provision+boot cycle burned on #1773 (att-20260729-010419).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "off_pod_phases" .claude/rules/planner-section-reference.md` → 3 hits (L624 section head, L643/L693 fenced examples); section read in full at compose time — the REQUIRED-only-when clause and every rule bullet scope reads-enumeration to off-pod consumers of pod outputs; a repo-root grep for any pod-side-reads requirement (`grep -rn "pod_phase_reads\|pod-side reads" .claude/rules/ .claude/agents/planner.md`) → 0 hits (absence claim; the 0-hit in-target result is the evidence) (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
In planner-section-reference.md §9 (off_pod_phases, #1535):
+ Rename the concept in prose to cross-phase reads (block name stays
+ off_pod_phases for back-compat): the block is REQUIRED whenever ANY
+ dispatched phase reads ANOTHER phase's outputs — including a pod/GCE/SLURM
+ phase reading VM-produced inputs (the #1773 inverse seam: git-clone lanes
+ stage only the pushed branch; data/ is gitignored).
+ Add a rules bullet: a read produced VM-side and consumed by a git-clone
+ lane MUST name its HF upload step (producer-final upload_folder) AND the
+ consumer launcher's staging step (scoped list_repo_tree + per-file
+ download), with the [stage] log line named as the crash-fix fix-engaged
+ signal.
```

## Scope / surfaces

- Primary target: `.claude/rules/planner-section-reference.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'off_pod_phases' .claude/ CLAUDE.md scripts/`) and update every MAIN-CHECKOUT hit consistently (worktree copies are stale mirrors — never edit those); consider whether `scripts/verify_plan.py` c39's WARN text should name the direction-agnostic rule (presence-check semantics unchanged) and whether `.claude/agents/upload-verifier.md` Step 2.8 wording needs the same generalization.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/planner-section-reference.md
- fingerprint: 19ce66a6b9cd

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/planner-section-reference.md
bug_observed: #1773 Pass B (GCE) crashed FileNotFoundError loading Pass A VM-produced selection outputs; the off_pod_phases spec only requires OFF-POD consumers to declare reads of pod outputs, so the inverse pod-reads-VM seam passed plan review and c39 by construction
why_workflow_gap: the §9 block is direction-specific; git-clone lanes stage nothing outside the pushed branch, and the #1469 carry-over gate glob-skips phase-output globs — no surface requires a pod-side phase to declare its cross-phase reads
proposed_change: generalize the planner off_pod_phases declaration to a direction-agnostic cross-phase-reads rule: every dispatched phase (pod/GCE/SLURM and off-pod alike) that reads another phase outputs declares reads[] with a fetchable permanent source + the staging step
diff_sketch: |
  + REQUIRED whenever ANY dispatched phase reads ANOTHER phase's outputs
  + (incl. pod/GCE/SLURM phases reading VM-produced inputs — the #1773 seam)
  + new rules bullet: VM-produced -> git-clone-lane reads name the producer
  + upload step AND the consumer launcher staging step + fix-engaged log line
confidence: high
related_task: #1773
<!-- /workflow-fix-candidate -->
