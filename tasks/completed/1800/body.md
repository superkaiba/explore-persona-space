---
title: 'daily-fix: WARN when GCP workload chain has no persist phase'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c6b74b98ad55
- daily-auto-filed
created_at: '2026-07-29T07:11:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1739''s GCP run finished
  with ZERO artifacts on HF (all 7 expected prefixes MISS) — the --workload-cmd phase
  chain carried no upload/persist phase for a raw-completion-producing run; ~2h of
  improvised recovery uploads raced the grace-poweroff clock'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-A P1.

## Goal

Add a dispatch-time backstop so a GCP workload chain with no upload/persist phase for its produced artifacts cannot launch silently.

## Workflow gap

- **Bug observed:** #1739's GCP run completed its phases and the instance approached grace-poweroff with ZERO artifacts on HF — all 7 expected prefixes were misses; ~2h of improvised recovery uploads followed. (inferred — not probed at mining time: the miner did not re-read the round-2 diff to confirm which phase should have carried the upload wiring; the planner should reconstruct from #1739's events.)
- **Why it is a workflow gap:** the planner-side fix landed as #1779 (verified: commit `f7232aca11`, 'planner §10 + Methodology lens 18 — ephemeral-lane text/JSON dest durability') but nothing checks at DISPATCH time — a plan drift or hand-composed relaunch chain can still launch persist-less.
- **Confidence (emitter):** medium
- verified-at-filing: #1779's merge resolves (`git rev-parse f7232aca11^{commit}` OK, 2026-07-29 UTC) — this filing is the dispatch-time sibling, not a duplicate; `grep -n 'upload' .claude/agents/experimenter.md` → no pre-launch persist-phase check among the hits.

## Proposed change (candidate diff sketch — refine in planning)

Experimenter pre-launch gate item: enumerate the workload chain's phases; if the plan declares git/HF-destined outputs and no phase names an upload/persist step, WARN loudly (or refuse for raw-completion producers). Optionally a mechanical dispatch_issue.py lint.

## Scope / surfaces

- Primary targets: `.claude/agents/experimenter.md` (pre-launch protocol), `scripts/dispatch_issue.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: c6b74b98ad55

- workflow_fix_target: .claude/agents/experimenter.md

