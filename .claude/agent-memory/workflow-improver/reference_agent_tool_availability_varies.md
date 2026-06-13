---
name: agent-tool-availability-varies
description: Agent-tool availability varies per workflow-improver spawn — check once; when absent, self-review + tests and report code-reviewer as skipped with reason
metadata:
  type: reference
---

Some workflow-improver spawns expose the `Agent` tool (e.g. the 2026-06-12
memory-prune run used it for parallel subagents); others expose only
Bash/Edit/Read/Write/Skill/ToolSearch + MCP (observed 2026-06-09 on #530 and
#531). The spec's §6 "pair with code-reviewer for substantive changes" is only
executable when `Agent` is present.

**How to apply:** check the tool list (at most one ToolSearch probe) before
claiming you'll spawn a reviewer. If `Agent` is absent: do a deliberate
self-review (enumerate edge cases + call sites), pin behavior with unit tests
where possible, and report `code-reviewer: skipped — Agent tool unavailable in
this spawn`. Never fabricate a review round, never stall, and never downgrade
the classification to "surgical" to dodge the requirement.
