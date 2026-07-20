---
title: 'daily-fix: committed scripts, not inline -c, for probe runs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:db29fb729dc8
- daily-auto-filed
created_at: '2026-07-20T06:47:42Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): inline -c one-liner probe
  dispatch SyntaxError''d, burning a GCE create+cancel'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from a transcript-mined problem (session efbcc710 / #1482 @ 11:48-11:51 UTC).

## Goal

Prescribe committed-script (never inline `python -c` quoting) workload commands for ad-hoc probe dispatches in the `/issue` backend-dispatch section.

## Workflow gap

- **Bug observed:** a G1-probe dispatch fired with a placeholder-broken inline staging one-liner that would SyntaxError after phase b0 and trigger a spurious RunPod failover; the just-created GCE instance was cancelled ~2 min after create (`reason=orchestrator-quoting-error`), burning a create + cancel round.
- **Why it is a workflow gap:** inline `-c` one-liners in `--workload-cmd` are un-lintable and quoting-fragile; the session's own recovery (rewrite as committed branch scripts, re-dispatch) is the recipe worth prescribing.
- **Confidence (emitter):** low (one incident, self-corrected; filed per the standing any-confidence directive)
- verified-at-filing: `grep -n "workload-cmd" .claude/skills/issue/SKILL.md | head` → backend-dispatch section present; no existing "committed-script only / no inline -c" prescription found near the dispatch recipe (absence claim — grep `python -c` guidance in the dispatch section returns no prescriptive rule) (2026-07-19).

## Proposed change (candidate diff sketch — refine in planning)

(none — add one paragraph to the backend-dispatch section: ad-hoc probe workloads are committed branch scripts invoked by path; inline `-c` one-liners in workload commands are the named anti-pattern, incident #1482 2026-07-19)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (§ Backend dispatch)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 866d2e62ab60

Mined evidence: "deliberate-stop ... reason=orchestrator-quoting-error — the G1-probe dispatch fired with a placeholder-broken staging one-liner ... instance cancelled via finalize --skip-confirm-artifacts ~2 min after create" (#1482, 2026-07-19).
