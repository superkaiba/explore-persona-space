---
title: 'workflow-fix: gotchas.md entry for registry side effects lost on resume'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1412191f4508
created_at: '2026-07-15T16:46:05Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1315 r6 (experiment-implementer):
  in-process registry registration side effects lost on resumed processes'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1315 (emitting agent: experiment-implementer, crash-fix round 6).

## Goal

Add a gotchas.md entry: register dynamic registry entries (CONTEXTS ids, trait names) at point of use — a resumed dispatcher process fast-forwards earlier phases and loses their in-process registration side effects, crashing consumers only on resume while fresh-out_root smokes pass.

## Workflow gap

- **Bug observed:** #1315 r6: phase_tier2's ModelOrganism construction relied on p0/p4's in-process _context() registration side effect; the resumed production process fast-forwarded those phases, so CONTEXTS lacked 'icl_prefix_impolite' and tier2 crashed (ValueError) while the fresh smoke passed.
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` (the codebase-trap register loading when writing training/eval/orchestration code) has no entry for the resume-loses-in-process-registration class — a resume-skip design plus dynamic registry ids is a recurring dispatcher pattern (#1090-lineage `_context()` side effects), and the class is invisible to fresh-out_root smokes by construction.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix + resumed-process regression tests landed on issue-1315 @ a4f18a320a)
- verified-at-filing: `grep -ciE "registry.*side.effect|side.effect.*registr|point of use.*regist|resume.*registr" .claude/rules/gotchas.md` → 0 hits (absence-of-entry claim — the 0-hit in-target result IS the evidence) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

In .claude/rules/gotchas.md (near the resume/checkpoint-per-phase entries):
+ ## In-process registry side effects are lost on resume (#1315 r6)
+ Resume predicates skip phases by done-file, but in-process state (dynamic
+ CONTEXTS registrations, trait registries, caches) is rebuilt per process:
+ a consumer relying on an earlier phase's registration side effect crashes
+ ONLY on a resumed process, while a fresh-out_root smoke passes via the
+ side effect. Register dynamic contexts/ids at POINT OF USE (immediately
+ before each consumer), never via phase-ordering side effects; audit
+ shared-lineage registries (e.g. issue779_common.TRAITS) for membership
+ asserts a new behavior trips only on production-only branches.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep before editing (`grep -rln 'point of use\|side effect' .claude/rules/`) and keep the entry consistent with the checkpoint-per-phase guidance in `.claude/rules/code-style.md`; list hits in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 1412191f4508

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p5_tier2 (issue1315_dispatch.phase_tier2)
lesson: A dispatcher whose ModelOrganism/CONTEXTS consumers rely on an EARLIER phase's in-process registry-registration side effect crashes on any RESUMED process — resume fast-forward skips the registering phase, so the fresh process's registry lacks the dynamic id ('icl_prefix_impolite') while a fresh-out_root smoke passes via the side effect. Register dynamic contexts at POINT OF USE (a `_context()` call immediately before each organism construction), never rely on phase-ordering side effects; and audit shared-lineage registries (issue779_common.TRAITS) for membership asserts that a new behavior only trips on production-only branches the smoke never reaches.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
