---
title: 'workflow-fix: plan-compute-sizing — phase-entry headroom gates must be resume-aware'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e860ee8dfc7b
created_at: '2026-07-24T03:28:40Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson from #1586 fu crash 5 (epm:failure-lesson
  v8): blanket fresh-run phase-entry headroom floor deadlocks a resume; skip/scale
  by pending cells (r8 fix implements the code side)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha_candidate failure-lesson raised on task #1586 (emitting agent: experimenter relaunch-4 report; crash-fix round 8 implements the code fix).

## Goal

Extend plan-compute-sizing.md's assert_out_root_headroom preamble duty with the resume-awareness requirement: phase-entry headroom gates skip or scale need_gb by PENDING cells — never a blanket fresh-run floor on a resume.

## Workflow gap

- **Bug observed:** #1586 fu relaunch 4 died ~0.1 s into p2: the phase-entry blanket 60 GB floor fired against a disk legitimately occupied by the run's own resume-done artifacts, while the sibling wave-level gate had correctly skipped (0 pending / 1 done). Deterministic on every respawn; it also blocked the later phase whose reclaim arms would free the space.
- **Why it is a workflow gap:** the rule mandates the per-phase assert duty but says nothing about resume semantics, so a dispatcher author implementing the duty verbatim ships the resume-deadlock shape (exactly what #1586's fu dispatcher did, plan-approved + twice code-reviewed with the rule on the books).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -i -E 'headroom.{0,60}resume|resume.{0,60}headroom|assert_out_root_headroom' .claude/rules/gotchas.md .claude/rules/plan-compute-sizing.md` → 1 hit (plan-compute-sizing.md:262 — the assert duty EXISTS; context READ: no resume-awareness clause anywhere in the paragraph — the proposed change is additive) + `git log --oneline --since='7 days ago' -- .claude/rules/plan-compute-sizing.md` → 4d50cb2d47 issue-1633: multi-arm min-width + stall-time down-width split guidance (#1633) (#1411);8c9087613b issue-1612: phase-ordering checkpoint high-water (workflow-fix #1612) (#1386);b634b2d037 task #1541: fan-out end-of-run accumulated-footprint §9 disk-sizing rule + item-16 extension + durability pins (#1314);— none touch resume semantics (2026-07-24)

## Proposed change (candidate diff sketch — refine in planning)

+ plan-compute-sizing.md, the assert_out_root_headroom preamble-duty paragraph (~L262): "Phase-entry gates MUST be resume-aware: compute the phase's PENDING set with the same predicates the phase's own resume scan uses — zero pending ⇒ skip the gate (one INFO line); partial ⇒ scale need to the pending subset (per-cell need × n_pending; constants untouched); fresh runs compute byte-identical need (pin it). A blanket fresh-run floor at phase entry deadlocks a resume whose own done artifacts occupy the disk and blocks the very reclaim phase that would free it (incident #1586 fu crash 5; fix pattern: the wave-level pending-aware gate)."
+ Optional sibling: a one-line cross-pointer from the gotchas.md dispatcher family if the planner judges it warranted.

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'assert_out_root_headroom' .claude/ CLAUDE.md scripts/`) and update every duty-prescribing hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `workflow_lint.py` no-flags + `--check-lessons-index` pass; ratchets respected.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/plan-compute-sizing.md
- fingerprint: e860ee8dfc7b

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: issue1586_dispatch p2_train phase-entry (resume path)
lesson: Phase-entry disk-headroom gates (PHASE_HEADROOM_GB via assert_out_root_headroom) must be resume-aware — skip or scale need_gb by PENDING cells, as the sibling wave-level gate already does. A blanket floor at phase entry deadlocks a resume whose own done artifacts legitimately occupy the disk: the gate demands headroom for work that won't run and blocks the later phase (p6 wipe-before-restage) that would free the space.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
supersedes:
<!-- /epm:failure-lesson -->
