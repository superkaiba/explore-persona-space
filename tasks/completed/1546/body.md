---
title: 'workflow-fix: digest crash-tails on orchestrator poll turns'
kind: infra
tags:
- wf-fix
- wf-fix-fp:75a693142385
- daily-auto-filed
created_at: '2026-07-19T07:08:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): 7 refusal kills wedged
  the #1481 orchestrator paging raw crash-forensics tails on its own poll turns (session
  replaced); guard-hook briefs carrying the hook''s BLOCKED text drew 3 more kills.
  First-pass brief neutralization does not cover orchestrator poll turns (c3-P1 +
  c5-P10).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c3-P1 + c5-P10). Route-2 filing.

## Goal

Extend the trigger-dense-review discipline so (a) orchestrator poll/forensics
turns ingest crash-log tails + lane state as STRUCTURAL DIGESTS (counts +
file references, not raw text) and (b) guard-hook task briefs pass the hook's
own BLOCKED text by file reference — closing the two refusal-kill classes
that first-pass brief neutralization does not cover.

## Workflow gap

- **Bug observed:** 7 refusal kills 15:04-17:52 wedged the #1481 orchestrator
  as it paged raw pod crash-forensics tails on its OWN poll turns (session
  replaced); and guard-hook task briefs carrying the hook's own BLOCKED text
  drew 3 more refusal kills (c5-P10).
- **Why it is a workflow gap:** `trigger-dense-review.md` § First-pass briefs
  (#1503) and § Revision-round briefs (#1413) neutralize SUBAGENT BRIEF
  composition, but neither covers the ORCHESTRATOR's own poll/forensics turns
  (where it reads raw crash tails directly), nor explicitly the case of a
  brief whose TARGET is a guard hook carrying that hook's BLOCKED text.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n -i 'poll turn\|forensic\|crash-log tail' .claude/rules/trigger-dense-review.md` → 0 hits; the § First-pass briefs section scopes its read/vocabulary discipline to SUBAGENT briefs, with no orchestrator-poll-turn digest duty and no explicit hook-BLOCKED-text-by-reference clause (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# trigger-dense-review.md: add an § Orchestrator poll/forensics turns clause —
+ When the ORCHESTRATOR itself polls a run whose forensics are trigger-dense
+ (crash-log tails, pod stderr, guard-BLOCKED output), it ingests them as
+ STRUCTURAL DIGESTS: counts, grep -c of error/traceback/killed/OOM, sha +
+ file references, a bounded windowed excerpt — NEVER the raw multi-KB tail
+ into its own context (the #1481 orchestrator wedge: 7 kills paging raw
+ crash tails on poll turns).
# And in § First-pass briefs: a brief whose target is a guard hook passes the
# hook's BLOCKED text by file reference, never inlined (c5-P10: 3 kills).
# issue-tick SKILL.md: confirm the tick's crash-tail read stays digest-only
# (grep -c / tail-bounded), consistent with this clause.
```

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md, .claude/skills/issue-tick/SKILL.md`
- Add the orchestrator-poll-turn clause + the hook-BLOCKED-text-by-reference
  clause; verify the issue-tick crash-tail read is already digest-only (it is
  documented as vocabulary-thin — confirm it does not page raw tails).

## Constraints / invariants

- Workflow-surface only. Neutral gate vocabulary throughout (CLAUDE.md §
  Spurious usage-policy refusals rung (e)).
- `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/trigger-dense-review.md, .claude/skills/issue-tick/SKILL.md
- fingerprint: b74a21c183c1

Surfaced problem (c3-P1 + c5-P10): the #1481 orchestrator wedged on 7 refusal
kills paging raw crash tails on poll turns; guard-hook briefs carrying BLOCKED
text drew 3 more kills.
