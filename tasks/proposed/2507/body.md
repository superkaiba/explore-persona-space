---
title: 'Fix pre-existing no-flags lint red: bare list_repo_tree + hf_hub_download
  in scripts/issue2378_segb_think_audit.py'
kind: infra
tags: []
created_at: '2026-08-23T22:00:35Z'
has_clean_result: false
origin_prompt: workflow-fix-candidate from issue-2474 Step 10d merge agent, 2026-08-23
workflow: v1
---
## Goal

Clear the trunk no-flags workflow-lint red: scripts/issue2378_segb_think_audit.py carries a bare `list_repo_tree(` Hub verify call and a bare `hf_hub_download` (live-hf-retry-routing check). Route both through explore_persona_space.orchestrate.hub helpers / hub.retry_transient, or add the documented waivers.

## Context

Detected by the issue-2474 Step 10d lint gate (both legs red, subtracted as pre-existing). Failing check family: workflow_lint --check-live-hf-retry-routing + the hub-verify check; the no-flags run IS the fleet Step 9c gate, so this red is subtracted noise for every gate run until fixed.
