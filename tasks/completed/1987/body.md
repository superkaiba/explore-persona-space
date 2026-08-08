---
title: 'workflow-fix: pm_inline rule for value-plus-minus-err in body audit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7511c5db4555
created_at: '2026-08-02T06:21:36Z'
has_clean_result: false
origin_prompt: 'clean-result-critic fold-round review on #1768, 2026-08-02 (candidate/prose
  follow-up; verbatim in body Provenance)'
workflow: v1
---
## Overview / Motivation
Auto-filed from a formal workflow-fix candidate emitted by clean-result-critic (single-Claude round) on task #1768 (fold-round review, 2026-08-02).
## Goal
Add a `pm_inline` rule to the clean-result body audit matching `value ± err` / bare `±<num>` in reader-facing prose, reusing interval_inline's caption-blockquote + GFM-table blanking.
## Workflow gap
- **Bug observed:** `median ±0.16 displacement, ±0.06 read-out` sat in #1768's `## Results` prose through a prior clean-result gate and its Codex twin; caught only on a fresh LM read.
- **Why it is a workflow gap:** the audit's own comment (L892) implies `value ± err` coverage but no live rule matches the character — the Lens 7 sub-category has no mechanical backstop at all.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n $'±' scripts/audit_clean_results_body_discipline.py` → 2 hits, BOTH comment lines (L311, L892); live-rule occurrences = 0 (2026-08-02, re-run by filer; matches the emitter's grep).
## Proposed change (candidate diff sketch — refine in planning)
diff_sketch: |
  +    "pm_inline": (
  +        r"\d\s*±\s*\d|±\s*\d*\.?\d+",
  +        "Inline credence interval as `value ± err` in reader-facing prose (banned)",
  +    ),
  reusing interval_inline's caption-blockquote + GFM-table blanking so chart annotations and parameter tables stay legal.
## Scope / surfaces
- Primary target: `scripts/audit_clean_results_body_discipline.py` (+ its test file); grep `.claude/skills/clean-results/SPEC.md` for the Lens 7 wording and keep consistent.
## Constraints / invariants
- Workflow-surface only; existing grandfathered bodies not newly hard-FAILed (WARN-first or forward-only per the audit's conventions — planner's call).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.
## Provenance
- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 7511c5db4555
Verbatim candidate block preserved in the emitting critic's report (task #1768 events context, 2026-08-02).
