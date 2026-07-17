---
title: 'workflow-fix: gotchas entry — subprocess phase registry + full-panel fresh-child
  smoke'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ac2c5103f1d9
created_at: '2026-07-17T01:24:45Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1090 fu6 crash-fix round
  (subprocess_parent_state_context_registry); see epm:failure-lesson v5 on #1090'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1090 (emitting agent: experiment-implementer, fu6 crash-fix round 3).

## Goal

Add a `.claude/rules/gotchas.md` entry (smoke/production-parity family): subprocess-per-phase dispatchers inherit NO module-level registry state — register dynamic ids idempotently at each phase entry, and smoke the FULL production id panel in a fresh child process (a smoke panel slice that drops panel-member-only ids masks the seam).

## Workflow gap

- **Bug observed:** #1090 fu6 p1c crashed in production with `ValueError: unknown context 'neg_sp_police'` — the held-out panel contexts existed only as `default_panel()` members registered in NO process (`register_fu3_contexts()` only registered the wildchat prefix), and the tiny-real e2e smoke passed because its 3-id `SMOKE_PANEL_IDS` slice dropped exactly the panel-member-only ids. One RunPod 2xH100 cycle burned.
- **Why it is a workflow gap:** gotchas.md's smoke/production-parity family documents the WIDTH member (#1315/#1333) and the GATE-CALIBRATION member (#1345) but not this REGISTRY/PANEL-MEMBERSHIP member; the next subprocess-phase dispatcher author has no rule to load.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -i "subprocess|registry|register" .claude/rules/gotchas.md` → related hits are the CVD-clobber + width/gate parity entries only, 0 hits for module-registry-state-across-subprocess / panel-slice-mask (absence claim — 0-hit in-target IS the evidence); `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` → 5 unrelated entries, no landed fix for this class (2026-07-17)

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md, after the gate-calibration parity entry (~L43):
+ - **Smoke/production parity includes REGISTRY/PANEL MEMBERSHIP — a subprocess-per-phase dispatcher inherits NO module-level registry state, and a smoke panel SLICE that drops panel-member-only ids masks the unregistered-context seam entirely.** RULES: (i) each subprocess phase entry that resolves dynamic ids calls an idempotent registrar unconditionally; (ii) the phase smoke runs in a FRESH child process and resolves the FULL production id set. (Incident #1090 fu6 p1c, 2026-07-17: ValueError unknown context 'neg_sp_police'; fix commit 2a36cccaa7 on issue-1090-fu6 — `_register_capture_contexts()` + fresh-child seam test.)
+ Cross-link the two agent memories (experiment-implementer feedback_subprocess_phase_registry_and_full_panel_smoke.md).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'panel-member\|register_fu3_contexts' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. Check whether `.claude/rules/LESSONS.md`'s gotchas trigger line needs a phrase (subprocess phase dispatchers / dynamic-id registries).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; LESSONS.md index stays consistent (`--check-lessons-index`).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: ac2c5103f1d9

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: fu6 p1c capture-organisms (scripts/issue1090_fu6.py _panel_specs -> issue1090_fu3_worker.ensure_context)
lesson: A subprocess-per-phase dispatcher inherits NO module-level registry state — and "the registrar runs somewhere" is not enough: fu3's register_fu3_contexts() only ever registered the wildchat prefix, while the held-out panel contexts (neg_sp_police/neg_sp_ph4) existed only as default_panel() members registered in NO process. Register every dynamic context id idempotently AT EACH PHASE ENTRY that resolves it, and make the smoke resolve the FULL production panel in a FRESH child process — a smoke panel slice that drops panel-member-only ids masks the seam entirely.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
