---
name: no-agent-tool-in-spawn
description: workflow-improver spawns have no Agent tool — the §6 code-reviewer pairing for substantive changes cannot be executed; report it as skipped with reason
metadata:
  type: reference
---

Observed 2026-06-09 (#530 all-three-spaces candidate): the workflow-improver
spawn's toolset is Bash/Edit/Read/Skill/ToolSearch/Write plus deferred MCP
tools — there is NO `Agent` tool, and ToolSearch finds none. The spec's §6
"pair with code-reviewer for substantive/architectural changes" is therefore
unexecutable from inside the agent.

**How to apply:** don't burn a ToolSearch round-trip looking for `Agent`
beyond one check; for substantive changes, self-verify thoroughly, commit per
§6.5, and report `code-reviewer: skipped — Agent tool unavailable in this
spawn` so the orchestrator can route a review at merge time if it wants.
Never fabricate a review round.
