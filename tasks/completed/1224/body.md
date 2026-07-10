---
title: reconcile v2 SKILL 7c/7f figure-commit vs SHA-pin ordering
kind: infra
tags:
- wf-fix
- wf-fix-fp:5d4cbd366b36
- daily-auto-filed
created_at: '2026-07-10T06:53:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): 7c splices SHA-pinned image
  URLs at assembly while 7f commits held plotter figures only after generation-mode
  verify PASS; report-verifier pin bullet unsatisfiable at 7e'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1168.

## Goal
Reconcile issue-v2 SKILL.md Steps 7c vs 7f figure-commit ordering (commit-then-assemble, or placeholder URLs rewritten at 7f plus a generation-mode-aware pin bullet); once fixed, mechanize a verify_report.py blob-identity check (extract <sha>/<path> from body image URLs; assert git hash-object == git rev-parse when a local copy is consulted).

## Workflow gap
- **Bug observed:** issue-v2 SKILL.md 7c splices SHA-pinned image URLs at report assembly while 7f commits the held plotter figures only AFTER generation-mode verify PASS (plotter.md gives a third variant); under the 7f-late reading, report-verifier check (b)'s pin-required bullet is unsatisfiable at 7e (verified live on main 2026-07-09: 7c at SKILL.md ~391, 7f at ~431).
- **Why it is a workflow gap:** The v2 report pipeline's figure-pinning contract is self-contradictory across its own steps, so every v2 run must improvise an ordering.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
(none — synthesized from prose follow-ups.) Secondary items to assess in planning: (2) .claude/agents/codex-clean-result-critic.md:295 paper-mode branch points Codex at working-tree figures/issue_<N>/ PNGs; verify #1056's pin-first EXCEPTION carve-out covers the paper-mode prompt body and add it if uncovered. (3) lower priority: consolidate the ~8 per-file copies of the pin-first rule into one shared .claude/rules/ file with pointers.

## Scope / surfaces
- Primary target: `.claude/skills/issue-v2/SKILL.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/skills/issue-v2/SKILL.md
- fingerprint: 206ad94ba05c

Parked prose follow-ups on #1168, 2026-07-09T13:10:35Z (planner prose concerns 1-2 + alternatives critic, confirmed by methodology critic): 7c/7f figure-commit ordering contradiction; codex-clean-result-critic paper-mode pin-first carve-out coverage; pin-first rule consolidation.
