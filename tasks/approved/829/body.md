---
title: 'workflow-fix: trim heavyweight agent specs + size lint (subagent context headroom)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-02T02:03:08Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed workflow gap on #825: 5 subagent deaths from
  context exhaustion; heavyweight agent specs (planner 95KB, experiment-implementer
  72KB) + 130KB CLAUDE.md leave no working headroom'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #825 (emitting agent: orchestrator, /issue 825 session, 2026-07-02).

## Goal

Restore working context headroom for heavyweight subagent spawns by trimming the largest agent specs (move recipe detail to on-demand rules) and adding a size lint.

## Workflow gap

- **Bug observed:** five subagent deaths in one /issue 825 session: 3× `planner` spawns died to autocompact thrash (context refilled to limit within 3 turns, 3× in a row; total_tokens reported 0), 1× `experiment-implementer` died with "Prompt is too long" after 5 tool uses, 1× `codex-critic` composer died to the same thrash. Lighter-spec agents (critic 65KB — borderline, general-purpose ~0KB, implementer 16KB) completed fine at 150-220K total tokens.
- **Why it is a workflow gap:** agent base context stacks: agent spec (planner.md 95,615 B; experiment-implementer.md 71,578 B; critic.md 65,514 B) + always-on agent MEMORY.md + project CLAUDE.md (130,424 B) + its rule imports + user-global CLAUDE.md/SOUL/USER + MCP tool schemas. The heavyweight specs leave so little headroom that a handful of ordinary tool results (an arXiv search, a windowed file read) trips permanent autocompact thrash — the agent cannot complete ANY bounded research task. The orchestrator had to compose the plan itself (permitted by the refusal-recovery recipe but the wrong default) and substitute `implementer` for `experiment-implementer`.
- **Confidence (emitter):** high (reproduced 5×, with a clean size gradient across agent types).

## Proposed change (candidate diff sketch — refine in planning)

- Audit `.claude/agents/*.md` by size; for planner.md / experiment-implementer.md / critic.md move per-scenario recipe blocks (marker-training walkthroughs, incident post-mortems, worked examples) into `.claude/rules/*.md` with `paths:` frontmatter or short pointers, targeting ≤35 KB per spec.
- Add `workflow_lint.py --check-agent-spec-size` (WARN >40 KB, FAIL >70 KB) so regrowth is caught at commit time.
- Consider trimming always-loaded agent MEMORY.md indexes to the 150-char-per-line contract.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`, `.claude/agents/experiment-implementer.md`, `.claude/agents/critic.md`, `scripts/workflow_lint.py`
- Grep the workflow surface before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes on touched files.
- No behavioral contract changes to the agents — content relocation + linting only (anything more is architectural and parks at plan_pending).
- This session runs under the workflow-fix recursion guard once spawned.

## Provenance

- workflow_fix_target: .claude/agents/planner.md, .claude/agents/experiment-implementer.md, .claude/agents/critic.md, scripts/workflow_lint.py
- fingerprint: (computed at filing)

Orchestrator observation (verbatim summary): "planner spawns for #825 died 3× to autocompact thrash before any plan text was produced; experiment-implementer died 'Prompt is too long' after 5 tool uses; the codex-critic statistics composer thrashed. Agent-spec size is the controlling variable: 95KB/72KB specs died, 16KB/0KB specs succeeded on the same task in the same session."
