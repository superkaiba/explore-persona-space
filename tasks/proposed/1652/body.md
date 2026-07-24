---
title: 'daily-fix: scope RunPod hourly cap to EPS-managed pods'
kind: infra
tags:
- wf-fix
- wf-fix-fp:85c035edf518
- daily-auto-filed
created_at: '2026-07-24T06:47:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): _assert_under_account_hourly_cap
  counts the whole shared team account including ~2855 USD/hr of unmanaged fellows
  pods so EPS provisions are falsely blocked'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Incident on #1586 (session 62e315d1, 2026-07-23): with GCP 8×A100 queue-starved ~5.5h, the RunPod fallback was FALSELY blocked because `_assert_under_account_hourly_cap` counts the WHOLE team account's live burn — the shared Anthropic-fellows fleet's ~$2,855/hr across ~93 unmanaged pods — not EPS-managed spend. The user had to approve raising `RUNPOD_ACCOUNT_HOURLY_CAP` 120→3400 plus a manual `backend: runpod` pin to unblock a run he had already prioritized. Same day the user issued the policy correction in the PM session (b2b4d655): unmanaged team pods (~$2.7k/hr) must be IGNORED by burn audits.

## Goal

Scope the account-hourly-cap guard (and the PM burn-audit read) to EPS-MANAGED pods only (the `pod-<N>` / legacy `epm-issue-<N>` managed prefixes, `pod_lifecycle._is_managed_pod`), so shared-team-account noise can never block an EPS provision.

## Workflow gap

- **Bug observed:** `scripts/pod_lifecycle.py`'s `_assert_under_account_hourly_cap` sums ALL team pods' hourly cost; on a shared team account the unmanaged fellows fleet dominates (~$2,855/hr), so the default cap (`_DEFAULT_ACCOUNT_HOURLY_CAP_USD = 80.0`, line 1616) is permanently exceeded and every EPS provision/resume is refused regardless of EPS's own burn.
- **Why it is a workflow gap:** the cap exists to bound EPS spend; counting foreign pods converts it into a permanent false block that costs user-priority runs hours (and pushes the user to blanket-raise the cap, which defeats it).
- **Confidence:** high (incident + explicit user policy correction, both 2026-07-23)
- verified-at-filing: `grep -rn "HOURLY_CAP\|hourly_cap" scripts/*.py` → all hits in `scripts/pod_lifecycle.py` (lines 1072/1210/1228/1277/1616 — `_assert_under_account_hourly_cap` + default 80.0) (2026-07-24 UTC). No open task found matching hourly/cap/burn/unmanaged in the registry title scan.

## Proposed change (refine in planning)

Filter the cap's pod enumeration to managed pods (`_is_managed_pod`) before summing; report the unmanaged remainder as an FYI line, never a block. Mirror the same managed-only scope in the PM burn-audit surface (`.claude/rules/pm-audit-reference.md`) per the user's 2026-07-23 correction.

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py` (+ `.claude/rules/pm-audit-reference.md` scope note)

## Constraints / invariants

- No dollar-budget caps on science (tests/test_no_dollar_budget_caps.py) — this guard is a provision-time rate sanity check, keep it that way.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 85c035edf518

- workflow_fix_target: scripts/pod_lifecycle.py
