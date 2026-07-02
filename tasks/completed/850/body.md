---
title: 'workflow-fix: analyzer.md context-budget protocol (spawn thrash)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c42fbfa120d6
created_at: '2026-07-02T08:01:24Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 763: analyzer spawns die of autocompact
  thrash — spec-side context budget needed'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #763 (emitting agent: orchestrator, /issue 763 autonomous session).

## Goal

Give the analyzer agent a context-budget protocol so its spawns stop dying of autocompact thrash: shrink/split the 112KB `.claude/agents/analyzer.md` spec (or add an explicit context-budget section + staged-input contract) so the fixed prefix + required working set (SPEC.md, results JSONs, plan, draft body) fits a subagent window.

## Workflow gap

- **Bug observed:** THREE consecutive analyzer spawns on task #763 died with "Autocompact is thrashing: the context refilled to the limit within 3 turns of the previous compact, 3 times in a row" at 8-9 tool calls each — including a spawn whose brief pre-staged ALL inputs to ~70KB and forbade every known large read. Failure is brief-independent.
- **Why it is a workflow gap:** the analyzer's FIXED prefix is structural: analyzer.md spec 112,229 bytes + auto-loaded skills (paper-plots 18.6KB, independent-reviewer 3.1KB) + agent-memory MEMORY.md 9.2KB + the project CLAUDE.md tree (130.8KB + inlined rules) — before the agent reads SPEC.md (86.9KB) or any result file. No brief can fix this; the spec itself must budget context.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/agents/analyzer.md:
+ ## Context budget (HARD)
+ - Never run `task.py view <N> --json` unfiltered (389KB on marker-heavy tasks); use `latest-marker` + jq-filtered slices.
+ - Read SPEC.md by section slice, never whole (86.9KB).
+ - Results JSONs >1MB: extract via python -> /tmp digest, never Read.
- (move the ~60KB of worked examples / exemplar blocks out of analyzer.md into an on-demand reference file, or split the spec)
```
Sibling precedent: #835 (planner.md context-budget protection) — same failure class, different agent file.

## Scope / surfaces

- Primary target: `.claude/agents/analyzer.md`
- Grep the workflow surface for other >80KB agent specs before editing (`wc -c .claude/agents/*.md`) and consider the same protocol for each; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/analyzer.md
- fingerprint: c42fbfa120d6

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/analyzer.md
bug_observed: Three consecutive analyzer spawns on #763 died of autocompact thrash at 8-9 tool calls each, including one with a fully pre-staged ~70KB brief — the agent's fixed prefix (112KB spec + skills + memory + CLAUDE.md tree) leaves no working room.
why_workflow_gap: The overflow is caused by the agent spec's own size + auto-loads, not by any brief; only a spec-side context-budget protocol / spec split can fix it.
proposed_change: Add a hard context-budget section to analyzer.md (forbid unfiltered view --json, whole-SPEC.md reads, >1MB JSON reads) and move exemplar bulk to an on-demand reference file.
diff_sketch: |
  + ## Context budget (HARD): latest-marker not view --json; SPEC.md by slice; >1MB JSONs via python->/tmp digest
  - (split ~60KB of exemplars out of analyzer.md)
confidence: high
related_task: #763
<!-- /workflow-fix-candidate -->
