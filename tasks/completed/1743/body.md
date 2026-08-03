---
title: 'daily-fix: code-reviewer verdict post via --file + read-back'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5d54a2914d13
- daily-auto-filed
created_at: '2026-07-28T06:59:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): code-review round 1 for
  #1723 returned ''verdict marker was posted successfully'' but neither the epm:code-review
  marker nor the durable /tmp file existed; orchestrator burned ~9 min + a duplicate
  reviewer spawn (Step 5b durable-verdict-first re-spawn)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 4d10421c (#1723), 2026-07-27T14:52Z.

## Goal

Make the code-reviewer's verdict post durable and self-verified: `--file` channel + a mandatory read-back before the agent returns.

## Workflow gap

- **Bug observed:** the reviewer's return claimed its `epm:code-review` marker was posted; no marker existed on events.jsonl and the durable verdict file `/tmp/code-review-1723.md` did not exist either. The orchestrator burned 3 probe calls + a full duplicate reviewer round (~9 min).
- **Why it is a workflow gap:** the agent spec's canonical post recipe is `post-marker ... --note "$(cat /tmp/code-review-<N>.md)"` — the argv-prose channel CLAUDE.md #1722 already deprecates for git-verb-bearing bodies (a review body quotes diff text/git verbs, a plausible silent-block cause) — and no read-back-verify duty exists, so 'posted' can be claimed unverified.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'note \"\$(cat' .claude/agents/code-reviewer.md` -> L37 (`--version <revision_round> --note "$(cat /tmp/code-review-<N>.md)"`), run at compose time 2026-07-28T06:5xZ; no `latest-marker` read-back duty near it (sed 33-60 inspected).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/agents/code-reviewer.md` (~L33-40): (a) change the post recipe to `post-marker <N> epm:code-review --file /tmp/code-review-<N>.md`; (b) add a final duty: read back `task.py latest-marker <N> --prefix epm:code-review` and confirm the just-posted version BEFORE returning — never claim 'posted' unverified.

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- Check the codex twin composer + Step 5b SKILL.md text for the same recipe string (grep before editing).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 5d54a2914d13

- workflow_fix_target: .claude/agents/code-reviewer.md
