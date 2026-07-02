---
title: 'workflow-fix: restricted tools allowlists on agents (MCP schema bloat)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6a2c336559e8
created_at: '2026-07-02T06:50:20Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 778: All-tools subagents inherit
  ~120K tokens of MCP schemas and autocompact-thrash to death; give each agent an
  explicit minimal tools allowlist'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #778 (emitting agent: orchestrator, /issue 778 session).

## Goal

Replace `tools: All tools` with explicit restricted tool allowlists on the
heavy workflow agents so subagents stop inheriting the full MCP tool-schema
payload at spawn.

## Workflow gap

- **Bug observed:** two consecutive `experiment-implementer` spawns on #778
  died in autocompact thrash within 4-6 tool calls. Transcript usage rows
  show the FIRST assistant turn at ~49K cache-read + ~120K cache-creation
  ≈ **168K static tokens before any work** — the un-deferred MCP tool
  schemas (todoist ~100 tools, google-workspace ~90, runpod, playwright,
  arxiv, ssh, huggingface-skills, context7) are inlined for every
  "All tools" agent, while the parent session gets them DEFERRED via
  ToolSearch. Compaction cannot shrink the static block, so the agent
  compacts after nearly every tool result ("context refilled within 3
  turns, 3 times in a row") and dies.
- **Why it is a workflow gap:** agent capability declarations live in
  `.claude/agents/*.md` frontmatter. `All tools` was chosen for
  convenience; none of the coding/analysis agents need todoist /
  google-workspace / playwright schemas. As user-level MCP servers
  accumulate, every "All tools" agent silently loses working context until
  it cannot function — a fleet-wide failure mode that surfaced today.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/agents/{experiment-implementer,implementer,analyzer,code-reviewer,
                planner,critic,experimenter,follow-up-proposer,
                living-docs-updater,...}.md
- tools: All tools   (or the "Tools: All tools" description line)
+ tools: Bash, Read, Write, Edit, Grep, Glob, TodoWrite
+   (+ mcp__ssh__ssh_execute etc. ONLY where the role needs pod SSH:
+    experimenter, upload-verifier, uploader;
+    + WebFetch/WebSearch + arxiv MCP only where the role needs literature:
+    planner, critic, related-work-finder, clarifier flows)
Audit EVERY agent file: keep the minimal set each role actually uses
(grep its spec for tool mentions). Verify restricted agents do NOT get MCP
schemas inlined (spawn probe + usage-row check).
```

## Scope / surfaces

- Primary target: `.claude/agents/*.md` (frontmatter tools lines)
- Verification: spawn a probe agent before/after and compare first-turn
  `cache_creation_input_tokens` in the transcript.

## Constraints / invariants

- Do not remove a tool an agent's spec text instructs it to use (grep each
  spec for `mcp__`, `WebFetch`, `WebSearch`, `Agent(`, `NotebookEdit`).
- Workflow-surface only; `workflow_lint` passes.

## Provenance

- workflow_fix_target: .claude/agents/*.md
- fingerprint: 6a2c336559e8

Surfaced prose (verbatim): two #778 experiment-implementer spawns thrashed
at ~168K static tokens from inlined MCP schemas; orchestrator implemented
the round's code change itself + Codex-only review as the emergency path.
