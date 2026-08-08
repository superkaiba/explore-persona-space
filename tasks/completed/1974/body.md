---
title: 'daily-fix: pooling-convention row in mapping disclosure'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c9f88143e1b6
- daily-auto-filed
created_at: '2026-08-01T07:08:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): #1768 shipped span-mean
  context pooling as an unexamined inherited convention mismatching the #779 last-token
  comparison line; no plan lens requires naming/matching the pooling convention —
  user caught it, ~15-18 GPU-h re-pool round.'
workflow: v1
---
# daily-fix: pooling-convention row in mapping disclosure

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M10; miner-5:P2). Source: session 75f66748 (#1768; binding convention change also posted to #1947) — #1768 shipped span-mean context pooling as an unexamined inherited convention, mismatching the #779 last-token line it compares against. The user caught it ("why are we fitting span mean …???"); the assistant conceded "it wasn't a deliberate measurement decision — it's an inherited convention… The plan *noticed* the mismatch… but only in one place: the base-fit sanity gate." Cost: a ~15–18 GPU-h re-pool round. Plan critics did not flag the pooling mismatch against the cited comparison line.

## Goal

Add a pooling-convention row to the both-arms representation-mapping disclosure: every mapping plan names the pooling of every context/answer vector (span-mean vs last-token vs response-avg) AND its match to the cited baseline line's convention; a mismatch is a plan-time REVISE, not a sanity-gate footnote.

## Workflow gap

- **Bug observed:** A representation-mapping plan inherited span-mean pooling while its headline comparison target (#779) used last-token pooling; no plan-time lens required the pooling convention to be named or matched, so the mismatch survived the full critic ensemble and cost a re-pool round after user catch.
- **Why it is a workflow gap:** The mapping disclosure surfaces (planner §4 Design both-arms rule; Statistics lens item 15 "Mapping-baselines pair" in `.claude/rules/critic-lens-reference.md` line ~984) require the prefix/context arms and the identity+kNN baseline reads, but say NOTHING about the pooling convention of the vectors being mapped or its parity with the cited baseline line — pooling is a load-bearing measurement choice that silently rides in from reused capture code.
- **Confidence (emitter):** high (absence probed; incident is the session's own concession)
- verified-at-filing: `grep -n "pooling\|span-mean\|span mean" .claude/agents/planner.md .claude/agents/statistics-critic.md .claude/rules/critic-lens-reference.md .claude/rules/planner-section-reference.md` → 0 hits in all four files (absence claim — no pooling-convention row exists anywhere in the mapping-disclosure surface; the 0-hit result is the evidence). Statistics lens item 15 confirmed at `.claude/rules/critic-lens-reference.md:984` (identity+bias / kNN only; context read — no pooling clause). `git log --oneline --since='7 days ago' -- .claude/agents/planner.md` → 5 commits (smoke-slice probes, c39 prose sync, off_pod_phases, §10 durability, §3 relocation) — none pooling-related; no landed fix (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/agents/planner.md §4 Design (the both-arms mapping clause):
+ Pooling-convention row: for every representation mapping, name the pooling
+ of EVERY vector entering the map (context/prefix/answer: span-mean |
+ last-token | response-avg | other) AND state whether it MATCHES the pooling
+ of the cited comparison/baseline line (e.g. the #779 last-token line);
+ a deliberate mismatch carries a one-line justification.

.claude/rules/critic-lens-reference.md, Statistics & Measurement lens
(extend item 15 or add a sibling item):
+ REVISE a mapping plan that does not name its pooling convention per vector,
+ or whose pooling mismatches the cited baseline line without a stated
+ justification — a mismatch is a REVISE, never a sanity-gate footnote
+ (#1768: span-mean inherited vs #779 last-token; ~15-18 GPU-h re-pool).

.claude/agents/statistics-critic.md (v2 twin): mirror the same item.
```

## Scope / surfaces

- Primary target: `.claude/agents/planner.md` (§4 Design)
- Secondary: `.claude/rules/critic-lens-reference.md` (Statistics lens), `.claude/agents/statistics-critic.md` (v2 owner), `.claude/rules/experiment-guidelines.md` guideline 11 (one sentence), `.claude/agents/codex-statistics-critic.md` if it inlines the lens text
- Grep before editing: `grep -rn 'item 15\|Mapping-baselines' .claude/agents/ .claude/rules/` and update every surface that mirrors the lens; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Keep the v1 lens numbering stable (extend item 15 or append; do not renumber).
- `scripts/workflow_lint.py --check-lens-coverage` (if the lens ledger is touched) + `--check-asks` pass.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: c9f88143e1b6

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: (driver-computed; tag authoritative)

Origin: CONSOLIDATED M10 (miner-5:P2), /daily 2026-07-31 — "#1768 shipped span-mean context pooling as an unexamined inherited convention, mismatching the #779 last-token line it compares against" (session 75f66748).
