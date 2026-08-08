---
title: 'daily-fix: HF fan-out pre-stage rule + FLEX preemption escal'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2aeddaa47d87
- daily-auto-filed
created_at: '2026-08-02T07:10:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): Three parallel #1739 boxes
  staged the same ~144 GB from HF simultaneously; the last died rc=137 on an HF 429
  storm. Its FLEX_START replacement was preempted rc=137 at transfer unit 30/45 (~2h
  in); a third FLEX attempt launched anyway — the STANDARD switch came only that evening.
  5 attempts to land one OOD leg.'
workflow: v1
---
# daily-fix: HF fan-out pre-stage rule + FLEX preemption escalation

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C10 (miner 1, P1+P2; session 55419495, issue #1739).

## Goal
(a) Add a plan-compute-sizing clause: when fanning N boxes over the same HF inputs, pre-stage once and fan from the staged snapshot (or serialize/jitter concurrent same-prefix pulls). (b) Add a compute-backend-failover clause: a multi-hour serial leg (>~2h, no mid-leg checkpoints) relaunched after a first FLEX_START preemption escalates to STANDARD (on-demand), not another FLEX attempt.

## Workflow gap
- **Bug observed:** Three parallel #1739 OOD boxes (oodw/oodsyc/oodhall) staged the same ~144 GB from HF simultaneously; the last (oodhall) died rc=137 — `unverified hypothesis — verify at plan time:` on an "HF 429 storm" (miner-inferred from the session's own crash-log diagnosis, staging helpers not read). Its replacement oodhall2 was FLEX_START-preempted rc=137 at transfer unit 30/45 (~2h in, 16:37Z); a THIRD FLEX attempt (oodhall3) launched anyway, and the switch to `--provisioning-model STANDARD` came only at 20:37–20:41Z ("no FLEX_START preemption — the fix"). 5 total attempts to land the hallucination OOD leg.
- **Why it is a workflow gap:** Neither rule surface names concurrent same-prefix HF fan-out staging as a sizing hazard, nor prescribes provisioning-model escalation after a first preemption on a long checkpoint-less leg — both decisions were improvised hours late.
- **Confidence:** medium
- verified-at-filing: `grep -n -i 'pre-stage\|prestage\|stagger\|jitter\|429' .claude/rules/plan-compute-sizing.md` → 0 hits (clause (a) absent; the ≥5 GB inline-staging clause at lines 288–289 is the natural anchor — staging-path/filesystem naming only, no concurrency arm); `grep -n -i 'preemption' .claude/rules/compute-backend-failover.md` → 4 hits (lines 501/511: short-job spot-absorb rationale; 1029: "a lone spot preemption — is unchanged" (boot-loop-streak context); 1074: setup-phase preemption streak) — NO relaunch-after-workload-preemption escalation clause; `git log --oneline --since='7 days ago' -- <both files>` → 3+5 commits, none touching fan-out staging or preemption escalation (2026-08-02).

## Proposed change (refine in planning)
- `.claude/rules/plan-compute-sizing.md` (near the ≥5 GB staging clause): a §9 plan (or inline dispatch) that fans N>1 boxes/legs over the SAME multi-GB HF prefix names its staging shape — pre-stage ONCE and fan from the staged snapshot (rsync/instance image/shared read path), or serialize/jitter the pulls; N concurrent same-prefix multi-GB pulls are a rate-limit kill risk (#1739 oodhall, rc=137).
- `.claude/rules/compute-backend-failover.md` (length-aware ladder section): after a FIRST FLEX_START/spot preemption kills a serial leg >~2h with no mid-leg checkpoint resume, the relaunch escalates to the STANDARD (on-demand) rung — do not re-roll FLEX for the same checkpoint-less leg (#1739 oodhall2→oodhall3 burned a third attempt). Distinct from the existing boot-loop streak (setup-phase deaths) — this binds workload-phase preemption on checkpoint-less legs.

## Scope / surfaces
- Primary target: `.claude/rules/plan-compute-sizing.md, .claude/rules/compute-backend-failover.md`
- Rule-text-only is acceptable for v1; if the planner opts to mechanize (b) in `backends/gcp.py`/dispatch, that is in-scope (backends/*.py is workflow surface) but not required. Check whether `verify_plan.py` should WARN on a fan-out plan with no staging-shape statement.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 2aeddaa47d87
- workflow_fix_target: .claude/rules/plan-compute-sizing.md, .claude/rules/compute-backend-failover.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C10.
