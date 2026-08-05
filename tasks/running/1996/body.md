---
title: 'workflow-fix: Step 6d.2 tick-source comments omit quiet-wait'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6438c76ac16a
- daily-auto-filed
created_at: '2026-08-02T07:07:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): After #1924 quiet-wait
  branch landed, three Step 6d.2 comments still describe tick JSON as arriving only
  via bg-Bash output/exit (the result definition ~L4707, the harness-re-invoke line
  ~L4755, the LAST-line-of-output line ~L4760), though a quiet-cycle tick now arrives
  via the Monitor notification.'
workflow: v1
---
# workflow-fix: Step 6d.2 tick-source comments omit the quiet-wait Monitor path

## Overview / Motivation

Auto-filed by the /daily 2026-08-01 Step C parked-candidate sweep from a workflow-fix candidate parked on task #1924 (emitting agent: code-reviewer round 1, recursion-guarded; formal candidate block, fingerprint 6438c76ac16a). Routed as a behavior-adjacent SKILL.md instruction change (the tick-source description drives how the orchestrator parses ticks), not a route-1 prose fix.

## Goal

Reword the Step 6d.2 tick-source comments in `.claude/skills/issue/SKILL.md` so all three sites name BOTH tick sources — bg-Bash exit output AND the #1924 quiet-wait Monitor notification — and optionally extend the #1818 pin test with the both-sources phrase.

## Workflow gap

- **Bug observed:** after the #1924 quiet-wait branch landed, three Step 6d.2 comments still describe tick JSON as arriving only via bg-Bash output/exit (the result-definition comment, the harness-re-invoke line, the LAST-line-of-output line — cited by the emitter at ~L4707/~L4755/~L4760), though a quiet-cycle tick now arrives via the Monitor notification.
- **Why it is a workflow gap:** the skill tick-source description is source-incomplete; a byte-precise reader could conclude `result` is only ever set from bg-Bash ticks and skip parsing Monitor-delivered ticks into it.
- **Confidence (emitter):** low
- verified-at-filing: `grep -n 'bg-Bash output\|LAST line of output\|harness re-invoke\|re-invokes' .claude/skills/issue/SKILL.md` → bg-Bash-only phrasing present at L4706 ("bg-Bash output (the same `result` the status branch below reads)"), L4716 ("Harness re-invokes orchestrator on bg-Bash exit"), L4721 ("line of the bg-Bash output — parse per § Tick-parse") — the emitter's ~L4707/4755/4760 line numbers have drifted but the three sites exist (2026-08-02 UTC). Landed-fix check: `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 4+ commits (#1897/#1883/#1882/#1876), none rewording the tick-source comments. Note: #1924's own merge may land text near these sites — the planner should re-locate the three sites against current main before editing.

## Proposed change (candidate diff sketch — refine in planning)

```diff
- # bg-Bash output (the same `result` the status branch below reads);
+ # output — bg-Bash exit OR quiet-wait Monitor notification (#1924) — (the same `result` the status branch below reads);
```
(analogous rewording at the harness-re-invoke and LAST-line-of-output sites; optionally extend the #1818 pin test with the both-sources phrase.)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Optional: the #1818 pin test (locate via `grep -rn '1818' tests/test_issue_skill_*`).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 6438c76ac16a
- origin: parked candidate on task #1924, ts 2026-08-02T06:07:27Z, routed by /daily 2026-08-01 Step C.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: After #1924 quiet-wait branch landed, three Step 6d.2 comments still describe tick JSON as arriving only via bg-Bash output/exit (the result definition ~L4707, the harness-re-invoke line ~L4755, the LAST-line-of-output line ~L4760), though a quiet-cycle tick now arrives via the Monitor notification.
why_workflow_gap: The skill tick-source description is now source-incomplete; a byte-precise reader could conclude result is only ever set from bg-Bash ticks and skip parsing Monitor-delivered ticks into it.
proposed_change: Reword the three sites to name both tick sources (bg-Bash exit or quiet-wait Monitor notification); optionally extend the #1818 pin test with the both-sources phrase.
confidence: low
related_task: #1924
<!-- /workflow-fix-candidate -->
