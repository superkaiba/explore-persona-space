---
title: cron_uv_cache_prune can prune the uvx env under long-running MCP servers (arXiv
  MCP wedged fleet-wide with [Errno 2])
kind: infra
tags:
- workflow-fix
- mcp
created_at: '2026-08-25T22:28:06Z'
has_clean_result: false
origin_prompt: 'related-work-finder #2564 report: arXiv MCP wedged, [Errno 2] on all
  tools; plausible uvx-prune cause; fleet-visible'
workflow: v1
---
## Goal
Stop the daily uv cache prune cron from wedging long-running uvx-launched MCP servers, and restore the arXiv MCP.

## Symptom (2026-08-25, session a33e25e6 / task #2564 Step 10c-bis)
All arXiv MCP tool calls (search_papers, get_abstract) fail with `[Errno 2] No such file or directory`. Probes: the configured storage path ~/explore-persona-space/.arxiv-papers exists and is populated; disk has 40G headroom. Diagnosis: server-internal — consistent with the server's uvx environment having been pruned out from under the running process by the daily `scripts/cron_uv_cache_prune.sh` run (uv cache prune keeps in-use entries by its own accounting, but a long-running uvx server's env can still lose files it lazily loads). Fleet-visible: every session using the arXiv MCP hits the same error. Evidence artifact: tasks/awaiting_promotion/2564/artifacts/related-work-proposal.md.

## Asks
1. Reproduce/confirm the mechanism (uvx env path of the arxiv MCP server vs uv cache prune's reap set).
2. Fix: exclude live uvx server envs from the prune (or pin the arxiv MCP to a persistent venv), and document the restart recipe (/mcp in the driving session).
3. Immediate remediation: restart the arXiv MCP; re-run the deferred #2564 related-work positioning pass afterward (recorded in #2564 events as NOT-RUN).

## Provenance
Surfaced in prose by the related-work-finder on #2564 (Step 10c-bis); routed by the orchestrator per .claude/rules/workflow-fix-on-bug.md (surfaced-prose follow-ups).
