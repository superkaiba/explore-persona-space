---
title: 'workflow-fix: verify_task_body judge-health drop-line population reconciliation'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6dfbf4ede182
created_at: '2026-07-30T13:12:37Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1776 fold-verify formal candidate: check_judge_error_denominator
  never parses ''<X> content drops of <Y> draws (<Z>%)'' sentences; reconcile X and
  Y against judge_scores.json per_arm totals — population mismatches are invisible
  to arithmetic-only lints'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a formal workflow-fix candidate raised on task #1776 (emitting agent: clean-result-critic, p3p4 fold-verification round).

## Goal

Extend `scripts/verify_task_body.py::check_judge_error_denominator` to parse "<X> content drops of <Y> draws (<Z>%)" judge-health claims and reconcile BOTH X and Y against the round's committed judge_scores.json per_arm totals — FAIL when the (X, Y) pair matches neither the all-arms totals nor a coherent named-subset split.

## Workflow gap

- **Bug observed:** #1776's folded body quoted "192 content drops of 56,250 draws (0.34%, worst arm 0.9%)" — an ALL-ARMS drop numerator over a STEERED-ONLY draw denominator (consistent reads: 192/67,500 = 0.28% all-arms, or 156/56,250 steered-only; per-stratum worst 1.2%, not 0.9%). `check_judge_error_denominator` (verify_task_body.py L5713) reported "no judge denominator asserted" — it never parsed this sentence form, and the stated Z is arithmetically self-consistent with X/Y, so a pure-arithmetic lint cannot catch the population mismatch. Only the clean-result-critic's manual trace caught it.
- **Why it is a workflow gap:** every judged round quotes a drop-rate health line (llm-judging rule 18), and the population-mismatch defect is only catchable by reconciling X and Y against the round's committed judge_scores.json per_arm totals — a mechanical check the verifier already has the plumbing for (it resolves round eval-JSON dirs for other checks) but no rule for.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'check_judge_error_denominator' scripts/verify_task_body.py` → 3 hits (def L5713 + registration L14574 + banner — the check EXISTS but has no "content drops of ... draws" parse rule; presence-of-anchor + absence-of-rule claim); landed-fix history `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → 3 commits, none touching the judge-denominator check (2026-07-30).

## Proposed change (candidate diff sketch — refine in planning)

```
+ JUDGE_HEALTH_RE = re.compile(r"(\d[\d,]*) content drops of (\d[\d,]*) draws")
+ # in check_judge_error_denominator: for each match, locate judge_scores*.json
+ # under the body-pinned eval_results/issue_<N>/**; X_art = sum(content_drops);
+ # Y_art = sum(content_drops + valid_draws); accept (X, Y) == (X_art, Y_art)
+ # or (X_steered, Y_steered) [baseline arms excluded]; else FAIL naming both
+ # consistent pairings.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py` (check_judge_error_denominator, L5713 family)
- Pins in tests/test_verify_task_body.py; WARN-vs-FAIL severity decided at plan time (a FAIL on new bodies only — never newly hard-FAIL grandfathered bodies).

## Constraints / invariants

- Workflow-surface only. Existing verify_task_body tests stay green; `workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 6dfbf4ede182
