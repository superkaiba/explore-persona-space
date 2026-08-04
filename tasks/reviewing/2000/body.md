---
title: 'daily-fix: sizing duty binds teammate-dispatched box sweeps'
kind: infra
tags:
- wf-fix
- wf-fix-fp:69f11f0393b8
- daily-auto-filed
created_at: '2026-08-02T07:11:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): #1739 all day: wall bases
  estimated not measured (~3x+ overruns, 4 re-scope events; R5 ~9.2h vs 4h budget,
  ~25 GPU-h vs 5-15 est); 5/10 new-arm GCE boxes OOM rc=137 in PILOT with no measured
  RSS basis; 3 independent behaviors serial-chained on one box. The measured-pilot/RSS
  duties bind plan section 9 and the user-chat inline carve-out, not teammate box
  dispatches.'
workflow: v1
---
# daily-fix: sizing duty binds teammate-dispatched box sweeps

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C3 (miners 1, 3, 8; sessions 55419495 / f98a12ed / 20e82ec2, all #1739).

## Goal
Extend the compute-character pre-launch statement duty (measured 1-cell pilot wall basis + measured RSS basis, per-behavior boxes when each behavior's leg is >~2h) to explicitly BIND teammate-dispatched box sweeps in live interactive sessions — today the duty is written for the /issue plan §9 pipeline and the user-chat inline carve-out, and #1739's ad-hoc teammate box dispatches sat in neither binding.

## Workflow gap
- **Bug observed:** #1739's box sweeps missed sizing all day: (a) wall-time bases estimated, not measured — tierbb ~10h on one transfer; R5 syco/hallu ~9.2h vs a 4h budget and ~25 GPU-h vs ~5–15 estimated ("My per-cell estimate was too optimistic"); 4 distinct overrun/re-scope events; corehall's measured pilot projected 8.494h vs plan 2.7h (~3.1× low; rc=7 fence fired correctly). (b) 5 of 10 new-arm GCE boxes OOM-killed rc=137 in the PILOT phase (whitening fits at n=18793 d=3584) — no measured RSS basis in §9. (c) Three independent behaviors serial-chained on one box ("sum instead of max"), split only after ~3h+.
- **Why it is a workflow gap:** the measured-pilot + RSS duties exist but their binding surfaces (plan §9; the CLAUDE.md user-chat-inline compute-character block) do not name teammate/subagent box dispatches issued mid-session, so the duty predictably fails to fire exactly where #1739 spent its GPU-hours.
- **Confidence:** medium (rc=9 RSS guard + host-RSS bounds already landed for the fit cores per miner-3 — eb06ab7e34, review PASS — so the residual is the RULE-binding gap, not the code).
- verified-at-filing: `grep -n 'teammate' .claude/rules/plan-compute-sizing.md` → 0 hits (target file never names teammate dispatch); § Per-cell fit phases exists at :361 ("the per-call basis MUST be a MEASURED 1-cell pilot ...") and the ×2 RSS presumption at :470-475, both plan-§9-scoped; CLAUDE.md's compute-character pre-launch statement + scope-extension addenda live inside the "User-chat inline free analysis" carve-out only (grep 'Compute-character pre-launch statement' CLAUDE.md → 2 hits, both in that carve-out block). Experiment-guidelines guideline 2 (`grep -n 'shardable axis' .claude/rules/experiment-guidelines.md`) covers GPU width, not per-behavior box splitting of multi-hour legs. `git rev-parse --verify eb06ab7e34^{commit}` resolves. `git log --oneline --since='7 days ago' -- .claude/rules/plan-compute-sizing.md CLAUDE.md` → recent commits (096405e94b pilot-at-production-shape; 517a4aa90d four discipline clauses) — none binds teammate dispatch (2026-08-01).

## Proposed change (refine in planning)
1. `.claude/rules/plan-compute-sizing.md`: add a short "Teammate / mid-session box dispatches" clause — ANY multi-hour (>~1h projected) box/leg dispatched by a teammate or orchestrator mid-session (outside a plan §9 row) carries the same pre-launch statement: measured 1-cell pilot wall basis (or cited prior measured figure, same kernel + shape), measured/×2-presumed RSS basis keyed to the largest cell, and fence ≥2× the pilot-extrapolated wall.
2. Same clause (or the CLAUDE.md carve-out's compute-character block): serial-chaining independent behaviors on one box is a REVISE-shape default violation — per-behavior boxes by default when each behavior's leg is >~2h (wall budget is max, not sum; sibling of experiment-guidelines guideline 2's shardable-axis duty).
3. CLAUDE.md: one sentence in the compute-character block extending its binding from "the subagent launches any fit/battery" to "any teammate/box dispatch of such work, mid-session extensions included".

## Scope / surfaces
- Primary target: `.claude/rules/plan-compute-sizing.md, CLAUDE.md`
- Keep CLAUDE.md delta to 1-3 sentences (always-on budget); the mechanics live in plan-compute-sizing.md. Grep `compute-character` across `.claude/` and keep wording consistent.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Do not duplicate the §9 recipes — the new clause POINTS at § Per-cell fit phases / the RSS gate and only widens who is bound.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 69f11f0393b8
- workflow_fix_target: .claude/rules/plan-compute-sizing.md, CLAUDE.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C3.
