---
title: 'workflow-fix: gotchas entry — Batch custom_id charset+length shape (^[a-zA-Z0-9_-]{1,64}$)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:db5742bed8c1
created_at: '2026-07-30T01:33:23Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1776 crash-fix cycle 9: Anthropic
  Batch custom_ids must match ^[a-zA-Z0-9_-]{1,64}$ — caller keys with dots/colons
  (stratum names like evil_a0.5, :: separators) 400 the FIRST batches.create, and
  a routing-only dry run can never catch charset bugs. Alias at the caller seam (bijective,
  collision-asserted, 53-char budget) and validate ids pre-submit in the shared dispatcher,
  dry-run included.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson `gotcha_candidate: yes` block raised on task #1776 (emitting agent: experiment-implementer, crash-fix cycle 9).

## Goal

Add a `.claude/rules/gotchas.md` entry to the Anthropic request-SHAPE family: Batch API `custom_id`s must match `^[a-zA-Z0-9_-]{1,64}$` — caller keys with dots/colons/slashes 400 the FIRST `batches.create`, and a routing-only dry run structurally cannot catch charset bugs; alias at the caller seam (bijective, collision-asserted) + validate composed ids pre-submit in the shared dispatcher, dry-run included.

## Workflow gap

- **Bug observed:** #1776's p6 off-pod judge died on its first `batches.create`: `400 — requests.0.custom_id: String should match pattern '^[a-zA-Z0-9_-]{1,64}$'`. Persona keys `"<stratum>::<context_id>"` (stratum names carry dots, e.g. `evil_a0.5`) rode the custom_id verbatim via the dispatcher scheme `f"{persona}__{global_idx:05d}__{ri:02d}"`; the routing-only `--dry-run` made zero API calls so the violation surfaced only at live submit.
- **Why it is a workflow gap:** gotchas.md's Anthropic request-SHAPE family (the Batch-API empty-system-block entry + the sync-path system-role lift at ~L335 — "request-SHAPE bugs at never-live-executed construction seams, invisible to every offline smoke") has NO member documenting the custom_id charset+length constraint, though the class is recurrent: #1415 hit the LENGTH half (64-cap overflow), #1776 the CHARSET half, at the same builder seam every Batch-judge caller touches. The id-shape member completes the family.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c -i 'custom_id' .claude/rules/gotchas.md` → 0 hits (absence claim; the family anchor "Anthropic request-builder seam" resolves at gotchas.md L335); landed-fix history check `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` → 8 commits, none touching custom_id (2026-07-30). The CODE defense itself landed in #1776's branch (`ded60e3fa8`: `judge_dispatch.validate_batch_custom_ids` wired pre-dry-run-return + at `_run_batch_path` entry) — this filing is the DOC entry so the class is visible at plan/build time repo-wide, not a code re-fix.

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **Anthropic Batch `custom_id`s must match `^[a-zA-Z0-9_-]{1,64}$` — BOTH
+   a charset and a length constraint; caller keys with dots/colons/slashes
+   (stratum names like `evil_a0.5`, `::` separators, hierarchical `/` ids)
+   400 the FIRST `batches.create`, and a routing-only dry run can never
+   catch it.** The id-shape member of the request-SHAPE family (siblings:
+   Batch empty-system-block, sync system-role lift). RULES: (i) alias at
+   the CALLER seam — bijective, collision-ASSERTED over the full realized
+   key set (char substitution alone is not injective: `evil_a0p5` vs
+   sanitized `evil_a0.5`), alias budget <=53 chars against batch_judge's
+   11-char `__NNNNN__NN` suffix (#1415's length half); persist the
+   id_map BEFORE any submit and reverse-map at result join so downstream
+   artifacts keep original keys; (ii) validate EVERY composed custom_id
+   pre-submit in the shared dispatcher (judge_dispatch.
+   validate_batch_custom_ids — wired before the dry-run return AND at
+   _run_batch_path entry), so violations are instant named pre-flight
+   failures at zero API cost. Worked pins: tests/test_issue1776_judge_ids.py
+   (branch issue-1776; fix ded60e3fa8). Long-form twin: .claude/agent-memory/
+   experiment-implementer/feedback_batch_custom_id_53_char_budget.md.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Place adjacent to the Anthropic request-SHAPE family (~L330-335) so the members cluster (system-block / system-role / custom_id shape).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; gotchas.md `paths:` frontmatter untouched unless the trigger set genuinely widens (the "Anthropic request-builder seam" trigger already covers this member).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: db5742bed8c1
