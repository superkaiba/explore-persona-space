---
title: 'daily-fix: capture plans name pooling convention per vector'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e93398a10cbf
- daily-auto-filed
created_at: '2026-08-02T07:14:04Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): Context-vector pooling
  convention drifted across rounds (span-mean in #1768 r1/#1900 vs last-token in #1768
  re-pool); Thomas had to direct ''NEWLINE BEFORE ASSISTANT ANSWERS''; the #1900 leakage
  race ran span-mean pre-challenge — cross-round comparability convention-confounded.
  No planner field or glossary row names pooling position.'
workflow: v1
---
# daily-fix: capture plans name pooling convention per vector

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C22 (miner-6 P2; session 75f66748, #1768/#1900/#1947 line).

## Goal
Require activation-capture / representation-mapping plans to name the pooling convention (span-mean vs prompt-final-token / newline-before-assistant) PER VECTOR as a plan-time field (`.claude/agents/planner.md` §6 Evaluation + the CLAUDE.md mapping-capture default), and add a pooling-ambiguity row to `docs/glossary_context_answer_map.md` § Retired/ambiguous terms.

## Workflow gap
- **Bug observed:** Thomas: "we should be using NEWLINE BEFORE ASSISTANT ANSWERS for the context vector"; assistant concession: "two different rounds use different conventions and I've been talking about both" — #1768 r1 + #1900 ran span-mean, #1768 re-pool used last-token, #1947 captures both. The #1900 leakage race ran span-mean before the challenge, so cross-round comparability is convention-confounded. Partially mitigated by #1947's capture-both design (miner-6), but nothing prevents the next capture plan from leaving the convention unstated.
- **Why it is a workflow gap:** The prefix-vs-context "both arms" rule and the glossary retired-terms check both exist, but neither surface names the POOLING position convention — an unstated per-vector convention is exactly the ambiguity class the glossary check was built for, and no planner field forces it to be declared.
- **Confidence:** medium
- verified-at-filing: `grep -in 'pooling\|span-mean\|last.token\|final.token' .claude/agents/planner.md` → 0 hits; `grep -in 'pooling\|span-mean' CLAUDE.md` → 0 hits; `grep -in 'pooling\|span-mean' docs/glossary_context_answer_map.md` → 0 hits (the § "Retired / ambiguous terms" table at line 80 has rows for "prefix vector"/"prefix map" ambiguity but none for pooling position); `grep -in pooling .claude/rules/planner-section-reference.md` → 0 hits; repo-wide `grep -rn 'span-mean' .claude/ CLAUDE.md` → hits only in `.claude/cache/experiment-*.md` clean-result copies (usage, not rules); `git log --oneline --since='7 days ago' -- .claude/agents/planner.md docs/glossary_context_answer_map.md` → commits present, none pooling-related (2026-08-02 UTC).

## Proposed change (refine in planning)
1. `.claude/agents/planner.md` §6 Evaluation (and/or §4 Design, planner's call — §6 lives at line ~330; note §-content may live in `planner-section-reference.md` per the #1740 relocation): any plan that captures/pools activation vectors names, PER VECTOR, the pooling convention — token span (prefix / context / query / response) AND position statistic (span-mean vs prompt-final-token, e.g. newline-before-assistant-answer) — with a `Source:` like any grounded choice.
2. `docs/glossary_context_answer_map.md` § Retired / ambiguous terms: new row — bare "context vector" without a pooling qualifier is ambiguous post-#1768/#1900 (span-mean vs final-token conventions coexist); use "context vector (span-mean)" / "context vector (final-token, newline-before-assistant)". This extends the existing glossary retired-terms check's enforcement surface (the interim-writeup rule already binds mapping-line writeups to this table).
3. CLAUDE.md: planner decides whether the "Prefix mapping AND context mapping" capture-default bullet gains one clause naming the pooling field, or whether the planner.md + glossary edits suffice (keep CLAUDE.md growth minimal).

## Scope / surfaces
- Primary target: `.claude/agents/planner.md, CLAUDE.md` (+ `docs/glossary_context_answer_map.md` as the session's companion edit; + `.claude/rules/planner-section-reference.md` if §6's body lives there)

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`. (The glossary is a docs/ file: it is the named enforcement surface of the existing retired-terms check and rides this task as the companion edit.)
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: e93398a10cbf
- workflow_fix_target: .claude/agents/planner.md, CLAUDE.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C22.
