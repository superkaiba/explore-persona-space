---
title: 'workflow-fix: fact-check HF reuse existence at the pinned revision'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5312ed1d9137
created_at: '2026-07-15T13:17:58Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1345 Methodology critic: fact-checker
  verified HF shards at default branch, not the plan''s pin; 2/4 stems absent at pin'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1345 (emitting agent: critic (Methodology lens)).

## Goal

Require the Phase-1.5 fact-checker (and add a verify_plan.py WARN check) to verify HF reuse-artifact existence AT the plan's pinned revision — per named stem/path — whenever a plan reuse row carries an explicit revision.

## Workflow gap

- **Bug observed:** The Phase-1.5 fact-checker verified the #1345 reuse shards' HF existence at the default branch while the plan consumes them at a pinned revision; two of four stems (naturalistic S-track) return 0 files at the pin, and the plan recorded the reuse as "confirmed by fact-check" (§12, High confidence). Caught only by the Phase-2 Methodology critic + consistency-checker running revision-scoped probes.
- **Why it is a workflow gap:** Neither the fact-checker instructions (.claude/skills/adversarial-planner/SKILL.md Phase 1.5) nor scripts/verify_plan.py check HF artifact existence AT THE PLAN'S PINNED REVISION — existence at main does not imply existence at the pin, so every revision-pinned reuse plan can ship a staging crash past fact-check.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "list_repo_tree" scripts/verify_plan.py` → 0 hits (no revision-scoped existence check exists in the verifier); `grep -n "list_repo_tree\|pinned revision" .claude/skills/adversarial-planner/SKILL.md` → 0 revision-scoped probe instructions in the fact-checker template (2026-07-15). Per-target: scripts/verify_plan.py 0 hits (absence-of-guard claim — the 0-hit result IS the evidence); .claude/skills/adversarial-planner/SKILL.md 0 hits (same).

## Proposed change (candidate diff sketch — refine in planning)

+ Fact-checker HF-existence step: when a plan's §10/§12 reuse row names a
+ pinned revision, run list_repo_tree(repo, path_in_repo=<prefix>,
+ revision=<pin>) and assert >=1 file per named stem/pattern; a probe at
+ the default branch does NOT satisfy the check.
+ verify_plan.py cNN_pinned_revision_reuse (WARN): reuse row with a
+ 40-hex revision + stem pattern but no revision-scoped verify command.

## Scope / surfaces

- Primary target: `.claude/skills/adversarial-planner/SKILL.md, scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'list_repo_tree\|pinned revision' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/adversarial-planner/SKILL.md, scripts/verify_plan.py
- fingerprint: 5312ed1d9137

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/adversarial-planner/SKILL.md, scripts/verify_plan.py
bug_observed: The Phase-1.5 fact-checker verified the #1345 reuse shards' HF existence at the default branch while the plan consumes them at a pinned revision; two of four stems (naturalistic S-track) return 0 files at the pin, and the plan recorded the reuse as "confirmed by fact-check" (§12, High confidence).
why_workflow_gap: Neither the fact-checker instructions nor verify_plan check HF artifact existence AT THE PLAN'S PINNED REVISION — existence at main does not imply existence at the pin, so every revision-pinned reuse plan can ship a staging crash past fact-check.
proposed_change: Require the fact-checker (and add a verify_plan WARN check) to run the scoped list_repo_tree/file_exists probe at the revision the plan pins, per named stem/path, whenever a §10 reuse row carries an explicit revision.
diff_sketch: |
  + Fact-checker HF-existence step: revision-scoped list_repo_tree per stem
  + verify_plan.py cNN_pinned_revision_reuse (WARN)
confidence: high
related_task: #1345
<!-- /workflow-fix-candidate -->
