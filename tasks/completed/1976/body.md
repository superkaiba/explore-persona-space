---
title: 'daily-fix: pre-split trigger for fits+figures+smoke units'
kind: infra
tags:
- wf-fix
- wf-fix-fp:393108e48903
- daily-auto-filed
created_at: '2026-08-01T07:10:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): #1902 unit-C implementer
  (fits+figures+smoke in one unit) died at the subagent context ceiling at 114 tool
  calls DESPITE the #1810 3-unit pre-split — the count-keyed heuristic misses composition-heavy
  units.'
workflow: v1
---
# daily-fix: pre-split trigger for fits+figures+smoke units

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M25; miner-3:P1). Source: session 3318f0b2 (#1902) — the unit-C implementer died at the subagent context ceiling ("Prompt is too long") at 114 tool calls / 58 min, on its final report turn, DESPITE the build having been pre-split into 3 units per the SKILL.md #1810 clause. The durable commit had landed; a micro-scoped "Unit C2" respawn completed per the Step 5b recipe. The recovery machinery worked — the residual gap is that a fits+figures+smoke unit passed the pre-split heuristic and was still too big.

## Goal

Tighten the `.claude/skills/issue/SKILL.md` pre-split clause so a unit combining fits + figures + a heavy smoke phase is a NAMED mandatory split trigger (smaller unit cap / mandatory judgment-split), not only covered by the discretionary "MAY pre-split by judgment" sentence.

## Workflow gap

- **Bug observed:** #1902's unit C (fits + figures + smoke in one unit) hit the subagent context ceiling at 114 tool calls even though the dispatch had already applied the #1810 pre-split (3 units, ≤3 deliverables each).
- **Why it is a workflow gap:** The SKILL.md clause "Pre-split multi-deliverable builds at dispatch (#1810…)" (line ~2193) keys the MANDATORY split on deliverable COUNT ("More than 4 code deliverables ⇒ … units of ≤3 deliverables each") and leaves composition-heavy lower-count units to discretion ("a lower-count build with a comparably large projected build volume (very large per-file scope, heavy smoke phases) MAY pre-split by judgment"). A unit whose deliverables individually carry a fit battery + figure generation + a multi-phase smoke run satisfies the count rule while its tool-call volume (~20 calls/deliverable was the clause's own risk basis; #1902's unit ran 114) exceeds the ceiling — the heuristic measures the wrong axis for this composition class.
- **Confidence (emitter):** low (per CONSOLIDATED; one incident, and the discretionary clause arguably already licenses the split — the gap is that it did not FIRE)
- verified-at-filing: `grep -n "pre-split\|deliverable" .claude/skills/issue/SKILL.md` + context read of lines 2193-2240 → the clause exists with the >4-count mandatory trigger, ≤3-deliverable unit shape, and the discretionary heavy-smoke sentence; NO mandatory trigger keyed to fits+figures+smoke composition (presence + absence claims both bound by the context read). `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 5+ commits (rule-23 drop-diagnosis, Step 10d landing confirmation, choom routing, rate/ETA duty, humanize verify) — none touch the pre-split clause; no landed fix (2026-08-01 compose time).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/issue/SKILL.md, "Pre-split multi-deliverable builds at
dispatch" clause:
+ Composition trigger (mandatory, #1902): a planned UNIT that combines a
+ fit/battery deliverable WITH figure-generation AND a smoke phase (or any
+ two of those where the smoke covers >=2 pipeline phases) is split further
+ regardless of deliverable count — fits and figures land in separate units,
+ and the smoke-bearing unit carries at most ONE other deliverable. The
+ existing ~20-tool-calls-per-deliverable sizing basis prices a fit+figures+
+ smoke deliverable at >=2 deliverable-equivalents.
- "MAY pre-split by judgment" (keep, but the named composition above is a
- MUST, not judgment).
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (the #1810 pre-split clause, ~line 2193)
- Secondary: none expected (the Step 5b thrash-respawn recipe stays unchanged as the backstop; Step 9b inherits via the existing brief-contract reference)
- Grep before editing: `grep -n 'Pre-split\|pre-split' .claude/skills/issue/SKILL.md` and update every cross-reference; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Do not lower the >4 COUNT threshold itself without evidence — the change is a composition-keyed trigger ADDED beside it (the planner may deflect with a reasoned no-change report if it judges the discretionary clause sufficient with a sharper wording).
- Marker/round semantics of the pre-split clause (intermediate units commit + breadcrumb, final unit posts the marker, one review round) are UNCHANGED.
- `scripts/workflow_lint.py --check-asks` passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- sha-verify (filing-time, #1467): `3318f0b2` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 393108e48903

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: (driver-computed; tag authoritative)

Origin: CONSOLIDATED M25 (miner-3:P1), /daily 2026-07-31 — "#1902 unit-C implementer died at the subagent context ceiling ('Prompt is too long') despite the pre-split into 3 units" (session 3318f0b2).
