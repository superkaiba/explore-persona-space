---
title: 'workflow-fix: restrict pipeline agents'' tools: frontmatter (MCP-schema context
  exhaustion)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:646ebbad3bda
created_at: '2026-07-03T02:13:17Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: .claude/agents/analyzer.md\
  \ (+ .claude/agents/*.md audit)\nbug_observed: Two analyzer spawns autocompact-thrashed\
  \ and died in an MCP-heavy session; a probe showed an All-tools agent burns ~138K\
  \ base tokens on MCP schemas before any work\nwhy_workflow_gap: analyzer.md and\
  \ most pipeline agents declare no tools: frontmatter, so subagent spawns load every\
  \ connected MCP server's schemas (~300+ tools, ~138K tokens base) and thrash in\
  \ MCP-heavy sessions\nproposed_change: Add explicit minimal tools: frontmatter to\
  \ pipeline agents that default to All tools so spawns do not load every MCP tool\
  \ schema\ndiff_sketch: |\n  + tools: Bash, Read, Write, Edit, Grep, Glob   (analyzer.md;\
  \ per-agent lists vary — keep mcp__ssh for experimenter/uploader/upload-verifier,\
  \ mcp__arxiv for related-work-finder)\nconfidence: high\nrelated_task: #823\n<!--\
  \ /workflow-fix-candidate -->"
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #823 (emitting agent: /issue orchestrator, autonomous session).

## Goal

Add explicit minimal `tools:` frontmatter to pipeline agents that default to All tools so spawns do not load every MCP tool schema.

## Workflow gap

- **Bug observed:** Two analyzer spawns autocompact-thrashed and died in an MCP-heavy session; a probe showed an All-tools agent burns ~138K base tokens on MCP schemas before any work.
- **Why it is a workflow gap:** `.claude/agents/analyzer.md` (and most pipeline agents) declare no `tools:` frontmatter, so the Agent tool loads EVERY connected MCP server's schemas (todoist ~130 + google-workspace ~100 + runpod + ssh + arxiv + playwright + huggingface ≈ 300+ tools) into the subagent's fixed base context. In a session with many MCP servers connected (e.g. a watcher respawn inheriting the full user-level mcp.json) the base alone is ~138K tokens (measured 2026-07-03, general-purpose probe, 1 tool call = 138,207 total tokens), leaving no working headroom → autocompact thrash → the agent dies. Restricted-tools agents (upload-verifier) ran fine in the same session. #823 lost two analyzer rounds to this and had to fall back to a role-overridden `uploader` spawn.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

For each pipeline agent that needs only core tools, add a `tools:` line to its frontmatter, e.g. for analyzer.md:

```
+ tools: Bash, Read, Write, Edit, Grep, Glob
```

Keep MCP tools only where the agent genuinely uses them (upload-verifier/uploader: + mcp__ssh__ssh_execute; related-work-finder: + mcp__arxiv__*). Audit every `.claude/agents/*.md` with no (or an unbounded) `tools:` declaration; the priority set is the agents the /issue pipeline spawns in-session: analyzer, experiment-implementer, implementer, experimenter, planner, critic, code-reviewer, consistency-checker, reconciler, follow-up-proposer, and the four codex-* prompt-composer wrappers (these need only Bash, Write, Edit, Read).

## Scope / surfaces

- Primary target: `.claude/agents/analyzer.md` (+ glob `.claude/agents/*.md`)
- Grep the workflow surface before editing: `for f in .claude/agents/*.md; do grep -L "^tools:" "$f"; done` (line-anchored, whole frontmatter — the head-12 heuristic used at filing time over-matches) and update every hit that is a live pipeline agent; list them in the plan. Cross-check each agent's actual tool usage before restricting (e.g. experimenter uses mcp__ssh__*).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Do NOT restrict an agent below what its spec's procedures actually invoke (search each spec body for `mcp__`, `WebFetch`, `WebSearch`, `Agent`, `Skill` usage first).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/analyzer.md
- fingerprint: 646ebbad3bda

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/analyzer.md (+ .claude/agents/*.md audit)
bug_observed: Two analyzer spawns autocompact-thrashed and died in an MCP-heavy session; a probe showed an All-tools agent burns ~138K base tokens on MCP schemas before any work
why_workflow_gap: analyzer.md and most pipeline agents declare no tools: frontmatter, so subagent spawns load every connected MCP server's schemas (~300+ tools, ~138K tokens base) and thrash in MCP-heavy sessions
proposed_change: Add explicit minimal tools: frontmatter to pipeline agents that default to All tools so spawns do not load every MCP tool schema
diff_sketch: |
  + tools: Bash, Read, Write, Edit, Grep, Glob   (analyzer.md; per-agent lists vary — keep mcp__ssh for experimenter/uploader/upload-verifier, mcp__arxiv for related-work-finder)
confidence: high
related_task: #823
<!-- /workflow-fix-candidate -->
