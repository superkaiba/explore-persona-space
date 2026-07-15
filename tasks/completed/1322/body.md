---
title: 'daily-fix: ownership check before resume; no double-bg'
kind: infra
tags:
- wf-fix
- wf-fix-fp:61acc4a8fc6b
- daily-auto-filed
created_at: '2026-07-15T06:51:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): orchestrator twice ran
  duplicate/mid-edit runs of jobs owned by live subagents (false Monitor DONE on empty
  results dir -> duplicate resume runners; launching a script the owning agent was
  mid-edit -> NameError double-writer), and separately double-backgrounded (nohup
  & inside a bg Bash) producing a false completion signal and a premature read'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (session 09f28ede, #825): two same-shape double-writer incidents (22:47Z map-alignment duplicate "resume" runners racing a healthy owned run; 06:15Z launching issue825_turn_depth_map.py while the owning agent was mid-edit -> NameError), plus a double-backgrounding false-completion (05:24Z premature FileNotFoundError read).

## Goal

add to CLAUDE.md § Orchestrator vs subagent re-invocation: (a) ownership check before resume/launch — pgrep the script + ping the owning agent before resuming or launching a run on an artifact path a live subagent owns; (b) never nest nohup ... & inside a run_in_background Bash — the wrapper's completion signal is false; let the bg Bash be the process

## Workflow gap

- **Bug observed:** orchestrator twice ran duplicate/mid-edit runs of jobs owned by live subagents (false Monitor DONE on empty results dir -> duplicate resume runners; launching a script the owning agent was mid-edit -> NameError double-writer), and separately double-backgrounded (nohup & inside a bg Bash) producing a false completion signal and a premature read
- **Why it is a workflow gap:** CLAUDE.md § Orchestrator vs subagent re-invocation governs bg-work mechanics but has no ownership-check or no-double-backgrounding line; the pattern recurred twice in one session.
- **Confidence:** medium-high (3 incidents, one session)
- verified-at-filing: `grep -n "double-background\|ownership check\|pgrep" CLAUDE.md` -> 0 relevant hits (the only pgrep hit is the codex_task reattach recipe at :137; absence-of-guard claim) (2026-07-15).

## Proposed change

Two bullets in § Orchestrator vs subagent re-invocation (or the subagent-default section), each one line, citing the 2026-07-14 incidents.

## Constraints

- Workflow-surface only; keep the always-on token budget in mind (2 lines max); recursion guard applies.

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: 61acc4a8fc6b
