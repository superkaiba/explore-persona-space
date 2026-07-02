---
title: 'workflow-fix: planner.md context-budget protocol (autocompact thrash in MCP-heavy
  sessions)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:70436dbc114c
created_at: '2026-07-02T06:07:12Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by /issue 833 orchestrator: 3 consecutive
  planner-agent spawns autocompact-thrashed (total_tokens 0) while lean general-purpose
  agents completed the same briefs; see candidate block in body'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #833 (emitting agent: orchestrator, /issue 833 session).

## Goal

Slim `.claude/agents/planner.md` and add an explicit context-budget read protocol so planner spawns stop autocompact-thrashing in MCP-heavy sessions.

## Workflow gap

- **Bug observed:** Three consecutive `Agent(subagent_type="planner")` spawns for #833 died with "Autocompact is thrashing: the context refilled to the limit within 3 turns of the previous compact, 3 times in a row" and `total_tokens: 0`, within 8–12 tool uses each — including one attempt whose prompt carried strict read-budget guardrails and a pre-digested context file. A trivial `Explore` probe and two `general-purpose` agents running the SAME planner/fact-checker briefs (197K / 211K tokens) completed cleanly in the same session.
- **Why it is a workflow gap:** `.claude/agents/planner.md` is 96 KB (~24K tokens) with `effort: xhigh` and a mandatory heavy read protocol ("Before Planning" steps that pull in large plans/rules). Combined with the project CLAUDE.md import tree and the MCP tool surface in a user-level-MCP-heavy session (todoist + google-workspace + runpod + arxiv + hf-skills were connected here), the baseline leaves too little working context — the workflow's canonical Phase-1 agent becomes structurally unusable, and every /adversarial-planner invocation in such a session silently degrades or fails.
- **Confidence (emitter):** medium (root-cause attribution to spec size + baseline overhead is inferred from the Explore/general-purpose contrast, not directly measured).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/agents/planner.md
+ ## Context budget (READ FIRST)
+ - Your spec + the project rule tree already consume a large fraction of your
+   context. Never `task.py view <N>` bare (event logs are MBs); read plans/bodies
+   via jq '.body' / Read with limit, in <=300-line chunks, only the sections needed.
+ - Grep-first on large scripts; never read >40 KB files end-to-end.
+ - Prefer the orchestrator-supplied digest file when the brief names one.
- (move the longest reference material — worked examples, extended §-templates —
-  out of planner.md into an on-demand references/ file the planner reads only
-  when the matching section is being written)
```

Also evaluate the same treatment for `.claude/agents/critic.md` (65 KB + 21 KB always-loaded memory) — same failure family, not yet observed to thrash.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`
- Secondary (same pattern, planner's discretion): `.claude/agents/critic.md`, `.claude/agent-memory/critic/MEMORY.md` (20 KB always-loaded — consider pruning per the memory-consolidation rules)
- Grep the workflow surface for other >60 KB agent specs before editing: `find .claude/agents -name '*.md' -size +60k`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- No behavioral contract change to the planner's output sections (§0–§13 requirements stay identical); this is a context-budget / spec-size fix only.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: 70436dbc114c

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/planner.md
bug_observed: three consecutive Agent(subagent_type=planner) spawns for #833 died with autocompact thrashing and total_tokens 0 within 8-12 tool uses while lean general-purpose agents completed the same brief
why_workflow_gap: planner.md is 96KB with effort xhigh and a heavy mandatory read protocol; combined with the project CLAUDE.md import tree + MCP schemas the subagent baseline leaves too little working context in MCP-heavy sessions
proposed_change: slim planner.md and add an explicit context-budget read protocol so planner spawns stop autocompact-thrashing in MCP-heavy sessions
diff_sketch: |
  + ## Context budget (READ FIRST) — never bare task.py view; jq '.body' + chunked
  +   Reads; grep-first on large scripts; prefer orchestrator digest files
  - move extended reference material out of planner.md into on-demand references/
confidence: medium
related_task: #833
<!-- /workflow-fix-candidate -->
