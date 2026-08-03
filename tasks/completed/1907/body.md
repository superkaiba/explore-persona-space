---
title: 'workflow-fix: flag malformed footer Reused bullets escaping the form-keyed
  check 37'
kind: infra
tags:
- wf-fix
- wf-fix-fp:19a580d5bad9
created_at: '2026-07-31T01:29:14Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r1 on #1739 prose follow-up: tighten verify_task_body
  footer-reuse check (Lens 5 FAIL, mechanizable: yes)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by clean-result-critic (round 1 on task #1739, 2026-07-31).

## Goal

Extend verify_task_body's footer reuse-provenance checking so a Reused bullet that LACKS the canonical `- Reused ... from [#M](...)` form (bare issue number + bare rev pin, no link, no repo-qualified path, no fitness rationale) is flagged instead of silently escaping the form-keyed check.

## Workflow gap

- **Bug observed:** task #1739's footer carried `#779 direction bank rev 037fcbb` as a Reused bullet — no `[#M](...)` link form, no repo-qualified pinned path, no fitness rationale — and `verify_task_body.py --issue 1739` returned OVERALL PASS; clean-result-critic Lens 5 caught it manually (FAIL, tagged mechanizable: yes).
- **Why it is a workflow gap:** check 37 (`check_footer_reuse_bullets_pinned`, WARN, v4-only, #1370) verifies a revision pin on bullets matching the canonical `- Reused ... from [#M](...)` shape; unverified hypothesis — verify at plan time: its predicate is keyed on that link form, so a malformed Reused-intent bullet never enters the check's match set (consistent with the #1739 incident: the malformed bullet passed). The mechanical verifier should be the first line for exactly this class (the clean-result-critic's mechanizable:-yes note).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "Reused\|reuse" scripts/verify_task_body.py` → 8+ hits incl. check 35 (`check_cross_issue_reuse_provenance`, L725-728) and check 37 (`check_footer_reuse_bullets_pinned`, L776-779, form-keyed docstring) (2026-07-31). Presence of the checks confirmed; the escape-by-malformation mechanism is the labeled hypothesis above, evidenced by the live #1739 OVERALL-PASS-with-malformed-bullet incident.

## Proposed change (candidate diff sketch — refine in planning)

In `check_footer_reuse_bullets_pinned` (or a sibling check):
+ ALSO scan footer bullets for Reused-INTENT markers (case-insensitive `reused` / `reuse of` / a bare `#<M> ... rev <hex>` pattern in the Repro footer) that do NOT match the canonical `- Reused ... from [#M](...)` form → WARN naming the bullet + the canonical form.
+ For canonical-form bullets, additionally WARN when the bullet carries no repo-qualified artifact path alongside the rev pin (a bare rev is not independently resolvable) and no fitness clause.
(Severity WARN to match check 37's class; escalation to FAIL is a planner decision.)

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'check_footer_reuse_bullets_pinned' scripts/ tests/ .claude/`) and update every hit; list them in the plan (tests pinning check 37 likely need companion cases).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Grandfathered v3/v2 bodies must not newly FAIL (v4-only check class, per check 37's existing scoping).
- This session runs under a workflow-fix Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 19a580d5bad9

Surfaced prose (verbatim, from clean-result-critic round-1 report on #1739): "The `#779 direction bank rev 037fcbb` bullet lacks a fitness rationale, and neither Reused bullet carries a repo-qualified pinned path or the `[#M](...)` link form (rev pins not independently resolvable). Tagged `mechanizable: yes` with a check sketch; surfaced a follow-up to tighten the verifier's footer-reuse check accordingly."
