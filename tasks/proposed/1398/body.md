---
title: 'daily-fix: neutral gate vocabulary first-pass briefs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:abb7f766962f
- daily-auto-filed
created_at: '2026-07-16T07:21:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): >=12 spurious Usage-Policy
  refusal kills across ~8 sessions; #1336 lost 3 spawns to kill-criteria vocabulary
  and neutralized only AFTER the kills'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Extend refusal-prevention rung (e): briefs for kill-gate / RLVR / guard tasks use neutral vocabulary ("halt gate", "stop criterion") FIRST-PASS, not only on post-kill retries.

## Workflow gap

- **Bug observed:** ≥12 spurious Usage-Policy refusal kills across ~8 sessions on 07-15; the #1336 session lost 3 spawns to "G1-kill"/"kill criteria" vocabulary and only neutralized ("halt gate") AFTER the kills (25ba019b 09:59/10:23/17:06Z); further kills on #1313, #1332 ×2, #1333 ×3, #1335 ×2, #1345 ×2, #1348, #825 (52 tool calls, ~1.7 h lost).
- **Why it is a workflow gap:** CLAUDE.md rung (e) "prevention beats recovery" currently names only harmful-content banks and real-world corpora as first-pass brief discipline — gate/kill-criteria vocabulary is not covered, so each session rediscovers the neutralization only after burning spawns.
- **Severity:** medium
- verified-at-filing: `grep -n 'prevention beats recovery' CLAUDE.md` → 1 hit (L147, rung (e): "briefs and prompts for harmful-content AND real-world-corpus (LMSYS/WildChat-class) tasks name banks/corpora by filename + row count" — gate/kill vocabulary absent from the rung); `grep -c 'halt gate\|gate vocabulary' CLAUDE.md` → 1 (no "halt gate" guidance; the 1 hit is unrelated); SKILL.md brief-composition guidance covers trigger-dense REVIEW reads (L2307-2343, `.claude/rules/trigger-dense-review.md`) but not first-pass brief vocabulary for gate tasks (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend CLAUDE.md's refusal ladder rung (e) (L147) to add gate/kill-criteria vocabulary to the FIRST-PASS brief discipline: briefs and prompts for kill-gate / RLVR / guard / stop-criteria tasks use neutral phrasing ("halt gate", "stop criterion", "termination predicate") from the first spawn, reserving the loaded terms for the artifacts themselves. Add the same line to `.claude/skills/issue/SKILL.md`'s brief-composition guidance (near the trigger-dense-review pointers at L2307), so subagent brief composers see it at compose time.

## Scope / surfaces

- Primary target: `CLAUDE.md` (rung (e) in "Spurious usage-policy refusals", L147)
- Secondary: `.claude/skills/issue/SKILL.md` (brief composition; anchor L2307-2343)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: abb7f766962f

- workflow_fix_target: CLAUDE.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 25ba019b (#1336) 09:59/10:23/17:06Z (batch 02 P1); also refusal kills on #1313, #1332 ×2, #1333 ×3, #1335 ×2, #1345 ×2, #1348, #825 (52 tool calls, ~1.7 h lost) — batches 02, 04, 05, 06, 09.
