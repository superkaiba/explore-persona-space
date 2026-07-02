---
title: 'workflow-fix: fable-5+xhigh agent pins autocompact-thrash subagent spawns'
kind: infra
tags:
- wf-fix
- wf-fix-fp:24d6590ee474
created_at: '2026-07-02T01:01:15Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed workflow gap on task #823 (2026-07-01): planner
  subagent spawns on the agent-default pin (model: claude-fable-5, effort: xhigh)
  crash with autocompact thrash within ~10 tool uses despite small tool results; identical
  brief succeeds with a model=sonnet override. All 28 .claude/agents/*.md files carry
  the same pin.'
---
# workflow-fix: agent model/effort pin (claude-fable-5 + xhigh) autocompact-thrashes subagent spawns

## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #823 (emitting agent: orchestrator, /issue 823 session).

## Goal

Diagnose the claude-fable-5 + effort:xhigh subagent context budget and adjust the
`.claude/agents/*.md` model/effort pins (or spec sizes) so agent spawns do not
autocompact-thrash.

## Workflow gap

- **Bug observed:** Two consecutive `planner` subagent spawns (2026-07-01, task
  #823, /adversarial-planner Phase 1) crashed with "Autocompact is thrashing:
  the context refilled to the limit within 3 turns of the previous compact, 3
  times in a row" — after only 9–11 tool uses, with every tool result ≤17 KB.
  Transcript inspection shows compaction firing every 2–3 turns from the FIRST
  turns, i.e. the agent's BASE context at spawn (system prompt + CLAUDE.md tree
  + the 95 KB planner.md spec + agent memory + xhigh thinking budget) is already
  at/near the compact threshold. The IDENTICAL brief re-run with an Agent-tool
  `model: sonnet` override completed cleanly (141K tokens, 30 tool uses, 17.5 min)
  — the differential implicates the fable-5 + xhigh pin combo, not the brief.
- **Why it is a workflow gap:** every one of the 28 files in `.claude/agents/*.md`
  pins `model: claude-fable-5` + `effort: xhigh`; the largest specs (analyzer.md
  126 KB, clean-result-critic.md 123 KB, planner.md 96 KB) are the most exposed.
  If the fable-5 subagent context window/thinking-budget configuration leaves
  ~<40 KB of usable headroom, EVERY pipeline stage (planner, analyzer, critics,
  implementer) can thrash and burn its spawn, stalling /issue sessions fleet-wide.
- **Confidence (emitter):** medium (differential evidence from one brief; other
  fleet sessions may or may not be hitting the same failure — check recent
  subagent transcripts for the thrash signature during diagnosis).

## Proposed change (candidate diff sketch — refine in planning)

```
# Option A (per-agent effort downgrade for the largest specs):
- effort: xhigh
+ effort: high        # analyzer.md, clean-result-critic.md, planner.md first
# Option B (model re-pin if the harness exposes the long-context variant):
- model: claude-fable-5
+ model: claude-fable-5[1m]
# Option C (spec-size reduction: move on-demand sections of the >90 KB agent
# specs into .claude/rules/ referenced files)
```

Diagnose FIRST (measure a trivial fable-5 planner spawn's base token usage vs
the compact threshold; check whether the harness honors `[1m]` in agent
frontmatter), then pick the minimal fix. Verify by re-spawning a planner-shaped
agent on the fixed pin with a real brief and confirming no compact within 20
tool uses.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`
- Pattern hits ALL of `.claude/agents/*.md` (28 files, every one pins
  `model: claude-fable-5` + `effort: xhigh` — verified by grep 2026-07-01);
  update every file the chosen fix applies to, and list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).
- NOTE for the implementing session: your own subagent spawns may hit this very
  bug — use Agent-tool `model: sonnet` overrides for your planner/critic/
  implementer spawns if they thrash (the #823 workaround).

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: 24d6590ee474

Surfaced prose (orchestrator observation, /issue 823 session, 2026-07-01):
two planner spawns on the agent-default model (claude-fable-5, effort xhigh)
crashed with autocompact thrash (context refilled to limit within 3 turns;
tool results were all <=17 KB, so base context + thinking budget is the
pressure, not reads); identical brief succeeded with model=sonnet override.
