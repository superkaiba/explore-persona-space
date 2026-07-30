---
title: 'workflow-fix: planner §10 — text/JSON outputs on ephemeral lanes need an HF
  dest, git-only banned'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e7325022a978
created_at: '2026-07-29T00:49:57Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1738 r4 implementer (summary JSONs lost
  with DELETE-on-exit boot disk)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1738 (emitting agent: experiment-implementer, round 4).

## Goal

planner §10 requires every text/JSON output of a stage running on an ephemeral lane (GCE DELETE-on-exit, terminate-on-verify pod) to name an HF (non-LFS) destination — git-only is valid only for VM-produced artifacts or when an explicit pre-teardown harvest phase is named; critic Methodology lens REVISEs violations.

## Workflow gap

- **Bug observed:** #1738's approved plan §10 declared "JSON summaries → git issue branch" for artifacts produced on the DELETE-on-exit GCE lane with no harvest phase; the instance was reaped minutes after a clean exit and both summary JSONs (multiturn_100k_fits.json, mapping_baselines.json) were lost; a CPU rebuild round (28 min) recovered the holdout side from retained tensors.
- **Why it is a workflow gap:** CLAUDE.md persist-by-default says text/JSON uploads ALWAYS, but planner.md's per-stage output-destination slot accepts a git-only destination for text/JSON produced on an EPHEMERAL lane, and no critic lens cross-checks destination-vs-lane durability.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'destination' .claude/agents/planner.md` → per-stage output-destination slot exists with no lane-durability constraint; `grep -rn 'git issue branch' .claude/agents/planner.md .claude/rules/upload-policy.md` → no ephemeral-lane qualifier at either site (2026-07-28); per-target: .claude/agents/planner.md present, constraint absent (absence-of-guard claim).

## Proposed change (candidate diff sketch — refine in planning)

+ §10 per-stage destinations: for any stage whose §9 lane is ephemeral
+ (gcp DELETE-on-exit / runpod terminate-on-verify), each text/JSON output
+ row MUST carry an HF dest (data repo, non-LFS) in addition to any git
+ dest; "git issue branch" alone is only legal for VM-resident stages or
+ with a named harvest phase that runs BEFORE teardown.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md` (§10); secondary: `.claude/rules/critic-lens-reference.md` Methodology lens (destination-vs-lane cross-check) — planner decides final scope.
- Grep the workflow surface for the pattern before editing (`grep -rln 'git issue branch' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: e7325022a978

(verbatim candidate block from the #1738 round-4 implementer report; see epm:experiment-implementation v4 on #1738)
