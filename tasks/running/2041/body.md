---
title: 'daily-fix: fan-out completion contract in spawn briefs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fcc42ef9fba9
- daily-auto-filed
created_at: '2026-08-03T07:02:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): (a) Three fold/fan-out
  subagents in one session idled mid-delivery with work staged-but-uncommitted (fold1739,
  judgerel1739, ladderfold1739 -- the last idled 3x; the orchestrator committed for
  it and fixed a broken doc anchor it left; it had also staged an uncertified code
  payload that tripped the root-commit guard) (session 55419495). (b) Four lit-dive
  scout reports lived only in /tmp for ~11h wi'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miners 3/8, sessions 55419495/c905b084/3a60e6ee, tasks #1739/#1902).

## Goal

Fan-out work products land durably (committed or consolidated) in the same turn they are produced, and delegated turns never end awaiting background work.

## Workflow gap

- **Bug observed:** (a) Three fold/fan-out subagents in one session idled mid-delivery with work staged-but-uncommitted (fold1739, judgerel1739, ladderfold1739 -- the last idled 3x; the orchestrator committed for it and fixed a broken doc anchor it left; it had also staged an uncertified code payload that tripped the root-commit guard) (session 55419495). (b) Four lit-dive scout reports lived only in /tmp for ~11h with the synthesis chat-only, until Thomas asked 'did you save all this to a report' (session c905b084). (c) A delegated Step 10d subagent ended its one turn blocked on a background gate, orphaning its own monitor; the orchestrator had to detect and re-drive with explicit finish-synchronously instructions (session 3a60e6ee).
- **Why it is a workflow gap:** The same-turn completion contract binds the 9a-ter inline path and teammate REPORTS, but fold/fan-out spawn briefs do not restate it, so subagents idle with uncommitted work; and no clause makes join-time consolidation the default for multi-agent research output.
- **Confidence (emitter):** medium (incidents probed by miners with verbatim rows)
- verified-at-filing: `grep -c -iE 'same-turn commit' CLAUDE.md .claude/skills/issue/SKILL.md` -> present for the 9a-ter inline rounds; 0 hits binding SPAWN BRIEFS for fold/fan-out agents; join-time consolidation default 0 hits (item 7 covers the orchestrator's own chat analyses only).

## Proposed change (refine in planning)

(i) fan-out/fold spawn briefs RESTATE the same-turn commit+push completion contract AND the inline-payload-lint-gate duty for any script the round writes; (ii) the orchestrator consolidates fan-out reports into a durable home (repo doc / vault) at JOIN time by default -- offer-to-save is the banned shape; (iii) a brief delegating a gate-wait mandates synchronous waiting (bounded foreground or Monitor until-loop inside the turn).

## Scope / surfaces

- Primary target: `CLAUDE.md, .claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- sha-verify (filing-time, #1467): `c905b084` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `3a60e6ee` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: fcc42ef9fba9

- workflow_fix_target: CLAUDE.md, .claude/skills/issue/SKILL.md

