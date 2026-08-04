---
title: 'workflow-fix: on-policy datagen must persist judge-rejected '
kind: infra
tags:
- wf-fix
- wf-fix-fp:10d241a02075
- daily-auto-filed
created_at: '2026-08-04T06:55:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 2): #1689 ships only judge-kept
  rows under raw_completions/gen/; the rejected generations were never persisted,
  so inspecting rejects required a full regeneration. This contradicts persist-by-default
  (model generations / rollout text are NEVER discardable) and destroys the audit
  trail rules 9/23 reason about; on-policy-completions.md never says to keep them
  (0 hits for reject).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-03 (route 2: behavior/logic change → independent review) from the nightly problem sweep (miner3, session f4f0e16a, task #1689).

## Goal

An on-policy datagen pipeline must persist its judge-REJECTED generations (with their verdicts) alongside the kept rows, so re-filtering or inspecting rejects never requires regenerating the wave.

## Workflow gap

- **Bug observed:** #1689 ships only judge-KEPT rows under `raw_completions/gen/`; the rejected generations were never persisted. When the round needed to look at rejected answers, the blocker was absolute: "the judge-dropped rows were never persisted ... the rejects are gone. So this isn't re-filtering existing data — it's a full regeneration" (session f4f0e16a, 2026-08-03T18:38:22Z).
- **Why it is a workflow gap:** it contradicts the standing persist-by-default contract head-on. CLAUDE.md § Upload Policy states text/JSON "uploads ALWAYS, unconditionally" and that "model generations / rollout text are NEVER discardable" — a judge-rejected generation is model-generated rollout text, and it is KB-scale, riding the non-LFS Hub path that stays open even over the storage quota. Dropping it also destroys the audit trail for the judge itself: rule 9's per-arm drop counts and rule 23's truncation-vs-content diagnosis both reason about rejected draws, and neither is checkable after the fact if the drops were never written. `.claude/rules/on-policy-completions.md` — which owns the elicitation ladder and the judge-filter step that CREATES the rejects — never says to keep them.
- **Confidence (emitter):** high (the blocker is quoted from the session; the persist-by-default rule is verbatim in CLAUDE.md).
- verified-at-filing: `grep -cin 'reject' .claude/rules/on-policy-completions.md` → **0** (2026-08-04) — absence claim verified in the rule that owns the judge-filter step. `grep -cin 'reject' .claude/rules/upload-policy.md` → 8, so the upload rule discusses rejection but (per the read) not the on-policy judge-filter reject set specifically — the planner should confirm which of the two rules is the right home, or whether both need a line.
- unverified hypothesis — verify at plan time: how widely the omission spreads. #1689 is the observed instance; the same judge-filter-then-discard shape plausibly exists in every datagen script built from the same recipe, so the planner should grep the datagen sites (`grep -rn 'judge' scripts/*datagen*.py scripts/issue*_gen*.py`) and report the blast radius rather than fixing only #1689's pipeline.

## Proposed change (candidate sketch — refine in planning)

```
in .claude/rules/on-policy-completions.md (§ The recipe, judge-filter step):
+ persist REJECTED generations alongside kept rows — same stage prefix, a
+ sibling path (e.g. raw_completions/<stage>/rejected/), each row carrying its
+ judge verdict + score + the drop reason (content-drop vs transport-loss per
+ llm-judging rules 9/24). Text is KB-scale and rides the unconditional
+ non-LFS upload path; a rejected generation is NEVER a valid
+ discarded_artifacts: entry.
```

Plus the enforcement pointer: the upload-verifier's reconciliation should be able to see the reject set (so a pipeline that writes none is visible rather than assumed-empty).

## Scope / surfaces

- Primary target: `.claude/rules/on-policy-completions.md`; secondary `.claude/rules/upload-policy.md` (whichever the planner establishes as the right home — possibly a line in each).
- Report (do not necessarily fix) the datagen blast radius.

## Constraints / invariants

- Rejected TEXT is never optional; this must not be written as a "when convenient" upload.
- Must not change the kept-row path/layout that existing consumers read.
- The yield-floor accounting (80% floor, equalize-down) is computed on KEPT rows and must not shift.
- Workflow-surface only for the rule change.

## Provenance

- fingerprint: 10d241a02075

- workflow_fix_target: .claude/rules/on-policy-completions.md
