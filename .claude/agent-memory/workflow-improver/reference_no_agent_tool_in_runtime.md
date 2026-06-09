---
name: no-agent-tool-in-runtime
description: workflow-improver runs may have no Agent tool — code-reviewer pairing for substantive changes can be impossible; self-review + tests, report reviewer as skipped with reason
metadata:
  type: reference
---

The workflow-improver spec (§6) says substantive/architectural changes pair with
`code-reviewer` via `Agent(subagent_type="code-reviewer", ...)`. But at least some
spawns (observed 2026-06-09, #531 pod_lifecycle fix) expose NO `Agent` tool — the
available + deferred tool lists contain Bash/Edit/Read/Write/Skill/ToolSearch and
MCP tools only.

**How to apply:** verify the deferred-tool list before claiming you'll spawn a
reviewer. If `Agent` is absent: do a deliberate self-review (enumerate edge cases
+ call sites), pin behavior with unit tests, and report
`code-reviewer: skipped — Agent tool unavailable in this runtime` so the
orchestrator can run its own review at merge time. Do not stall or downgrade the
classification to "surgical" to dodge the requirement.
