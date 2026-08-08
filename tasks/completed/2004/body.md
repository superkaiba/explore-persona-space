---
title: 'daily-fix: teammate report channel + deliverable content pro'
kind: infra
tags:
- wf-fix
- wf-fix-fp:335455bfe33d
- daily-auto-filed
created_at: '2026-08-02T07:12:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): Named-teammate final text
  did not route to the lead (4-scout fan-out: 2 re-brief rounds, 8 extra SendMessages
  before file-based delivery worked); 2 ''summary is required when message is a string''
  SendMessage tool errors in 2 /issue sessions; a subagent idled with a present-committed-serving
  dashboard whose table had ZERO data rows — durable-state probes read done, only
  a content probe caught it.'
workflow: v1
---
# daily-fix: teammate report channel + deliverable content probes

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entries C9+C17 (miners 5, 2, 6, 7; sessions c905b084, 438bf106, 0ac15c23, 7a758c23, cae57d2b).

## Goal
Extend the CLAUDE.md teammate-coordination bullet: (1) named-teammate fan-out briefs declare a file+SendMessage delivery channel up front (final text does not route to the lead); (2) SendMessage with a string `message` REQUIRES `summary`; (3) the durable-state probe list gains a content probe — a present-and-serving artifact is not a delivered deliverable until non-empty payload is confirmed.

## Workflow gap
- **Bug observed:** (C9) In a 4-scout lit-dive fan-out (c905b084, 04:38–04:47Z), all four named teammates idled post-spawn; s1s2-scout idled AGAIN with no report received — the lead concluded "its final text doesn't route back automatically" and re-briefed all 4 to file-based delivery (`/tmp/litdive-*.md` + SendMessage path): 2 re-brief rounds, 8 extra SendMessages. Plus 2 `summary is required when message is a string` tool errors in 2 /issue sessions (7a758c23 16:23:06Z, cae57d2b 01:30:04Z), one wasted turn each; plus 2 TaskOutput-keyed-on-NAME recurrences (438bf106 "research-docs", 0ac15c23 "feature-dashboard" — already documented, context only). (C17) The `feature-dashboard` subagent (0ac15c23, 03:28–03:31Z) idled with its deliverable present, committed, and serving 200 — but a content probe found ZERO data rows (82 KB HTML shell); only content inspection caught it. Sibling: `sae-dense-bridge-1482` looked dead on an empty output dir 40 min into an 82-min loop (write-only-at-end design; ownership deferral per #825 was correct, run completed at 0371b0066c).
- **Why it is a workflow gap:** The teammate-coordination bullet's FINAL-TEXT clause and durable-state probe list are silent on named-teammate final-text routing, the SendMessage `summary` requirement, and content-level deliverable verification — so leads predictably strand reports and accept hollow deliverables.
- **Confidence:** medium (C9 channel behavior), high (summary param + content probe — probed firings / probed content inspection)
- verified-at-filing: `grep -n 'Teammate coordination' CLAUDE.md` → 1 hit (line 102, the bullet — edit anchor confirmed); `grep -n 'content probe\|non-empty payload' CLAUDE.md` → 0 hits (proposed C17 clause absent); `grep -n 'summary' CLAUDE.md | grep -ci SendMessage` → 0 (proposed summary-param clause absent); `git log --oneline --since='7 days ago' -- CLAUDE.md` → 8 commits, nearest-neighbor 517a4aa90d (file-scoped teammate probe, #1697) does not cover any of the three clauses (2026-08-02).

## Proposed change (refine in planning)
In the CLAUDE.md teammate-coordination bullet:
- Clause (d), FINAL-TEXT branch: add "For NAMED teammates in a fan-out team, final text does not route to the lead session — fan-out briefs declare the delivery channel UP FRONT: write the report to a file + SendMessage the path. `unverified hypothesis — verify at plan time:` the non-routing is inferred from one session's observed failure + successful channel switch (miner-inferred, not probed against the harness)."
- Clause (c)/(d): add "SendMessage with a string `message` REQUIRES the `summary` parameter (2 tool-error firings, 2026-08-01/02)."
- Clause (b) durable-state probe list: append "and a CONTENT probe on the expected output files — a present-and-serving artifact is not a delivered deliverable until the probe confirms non-empty payload (0ac15c23: an empty-table dashboard read as done on presence+HTTP-200 alone)."

## Scope / surfaces
- Primary target: `CLAUDE.md` (the § "Orchestrator vs subagent re-invocation" teammate-coordination bullet, line ~102).
- Keep the additions inside the existing bullet's clause structure ((b)/(c)/(d)); do not fork a new bullet. Grep `.claude/skills/` for fan-out brief templates that restate clause (d) and mirror the summary-param line where found.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 335455bfe33d
- workflow_fix_target: CLAUDE.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entries C9+C17.
