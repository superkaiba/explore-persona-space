---
title: '[Investigate] Add a post-provision MCP-reload nudge to pod.py provision'
kind: infra
tags: []
created_at: '2026-05-06T08:12:34.000Z'
has_clean_result: false
sagan_id: 127da845-234a-41fa-8d87-529b7c5e1c93
sagan_number: 287
priority: normal
legacy_why_unset: true
---
**Parent:** #275 (audit item 1)

## Background

Per the audit posted on #275 (item 1), there is a gap between
`pod.py provision` writing the new pod to `~/.claude/mcp.json` and
Claude Code's MCP runtime picking up the change. The user must
manually `/mcp` to reload before SSH MCP tools resolve for the
freshly-provisioned pod. The lifecycle code IS correct — `pod_config.update_mcp_config` writes the right file at `pod_config.py:275-340` — the gap is in the runtime reload flow.

In production this is a non-issue because the `/issue` skill exits
after Step 6b and the experimenter starts in a NEW Claude Code session
where MCP loads fresh. The gap only bites when an agent provisions a
pod and then tries to immediately SSH into it without an intervening
session boundary.

## Two viable fixes

1. **Cosmetic.** Emit a `→ Run /mcp in Claude Code to reload SSH MCP
   tools for this pod` line at the end of `pod.py provision` output.
   Zero risk.
2. **Functional.** Add a one-shot `pod.py mcp-reload` command that
   invokes Claude Code's reload via the `claude` CLI if available.
   Higher risk — depends on the `claude` CLI's reload semantics.

## Out of scope

Pre-loading SSH MCP tools at provision-time inside the same Claude
Code session — that's a Claude Code runtime feature, not a
project-side bug.
