---
title: 'daily-fix: PM ownership check before spend emergency'
kind: infra
tags:
- wf-fix
- wf-fix-fp:67ad7b463cf9
- daily-auto-filed
created_at: '2026-07-16T07:21:55Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): PM escalated a false $128/hr
  spend emergency on four 8xH100 team pods; Thomas: ''no leave them. they''re doing
  important work''; non-EPS pods with the pod- prefix are exposed to the EXITED>24h
  auto-terminate'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Before declaring unmapped RUNNING pods a spend leak, the PM checks team-shared ownership signals (creator, recent activity) and frames the surfacing as a question, not an emergency; document that non-EPS pods may legitimately carry the managed `pod-` prefix and audit the stale-pod sweep's exposure for them.

## Workflow gap

- **Bug observed:** the PM escalated a false $128/hr "spend emergency" on four 8×H100 pods (#77961-77964) that were legitimate team work; Thomas corrected: "no leave them. they're doing important work" (194f5813 11:26-11:40Z). Residual real hazard the PM itself flagged: non-EPS pods carrying the managed `pod-` prefix are exposed to the EXITED>24h auto-terminate.
- **Why it is a workflow gap:** research-pm.md has no ownership-triage step for unmapped RUNNING pods — the team-shared RunPod account means "not in pods_ephemeral.json" does not imply "leak" — and no doc anywhere records that non-EPS team pods can collide with the managed `pod-` name prefix the audit sweep keys on.
- **Severity:** medium
- verified-at-filing: `grep -n 'spend\|leak\|emergency\|unmapped' .claude/agents/research-pm.md` → 3 hits, all unrelated (`held: spend` filing-classification lines L620/L635; a credentials/spend gate L229) — no unmapped-pod ownership-triage guidance (absence confirmed); `grep -n 'pod-' scripts/pod_lifecycle.py | grep -i 'managed\|prefix'` → `_MANAGED_PREFIXES` keys the sweep on the `pod-` prefix (presence of the collision surface confirmed per CLAUDE.md § Pods naming) (2026-07-16 UTC).

## Proposed change (refine in planning)

Add to `.claude/agents/research-pm.md` a pod-spend-triage protocol: on spotting unmapped RUNNING pods, check ownership signals FIRST (creator field via the team-scoped API, recent activity, GPU util, name shape) and surface to Thomas as a neutral question ("4 unmapped 8×H100 pods — is this team work?") rather than an emergency with terminate recommendations. Add a note to the pod-audit docs (pod_audit.py docstring or `.claude/rules/background-automation.md`) that non-EPS team pods may legitimately carry the `pod-` prefix, and audit `_MANAGED_PREFIXES`-keyed auto-terminate exposure for them (e.g. require the pods_ephemeral.json/issue mapping, not just the name, before reaping).

## Scope / surfaces

- Primary target: `.claude/agents/research-pm.md`
- Secondary: `scripts/pod_audit.py` / `.claude/rules/background-automation.md` (the `pod-` prefix exposure note + possible mapping-required guard)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 67ad7b463cf9

- workflow_fix_target: .claude/agents/research-pm.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 194f5813 11:26-11:40Z (batch 09 P22).
