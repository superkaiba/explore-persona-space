---
title: 'workflow-fix: extend pre_reg head-noun list for the #1586 ''registered layer''
  escape'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4a643434c286
created_at: '2026-07-23T09:42:21Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1586 r1 prose follow-up: mechanizable audit regex
  for bare pre-registration framing — \bregistered\s+(layer|verdict|margin|read|lattice|rung|window|band)\b'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised by the clean-result-critic on task #1586 (emitting agent: clean-result-critic, round 1, 2026-07-23).

## Goal

Extend `audit_clean_results_body_discipline.py`'s `pre_reg` head-noun list to catch the #1586 escape class ("the registered layer" ×5), measuring the critic-sketched sibling nouns under the file's established corpus-measurement discipline.

## Workflow gap

- **Bug observed:** five "the registered layer" Lens-7 register violations in #1586's clean-result body passed the audit's `pre_reg` mechanical check; only the LM critic caught them.
- **Why it is a workflow gap:** the pre_reg branch (added #1419, extended #1475) covers a measured head-noun list that lacks `layer` — the exact escape-then-extend pattern both prior fixes followed.
- **Confidence (emitter):** high (probe-verified)
- verified-at-filing: `uv run python -c "re.search(PATTERNS['pre_reg'][0], 'the registered layer')"` → **False** ('the registered verdict' → True; live probe against the current module, 2026-07-23). `grep -c '\blayer' scripts/audit_clean_results_body_discipline.py` → 0. Context read per clause (c): the existing #1419/#1475 check is present but its head-noun list demonstrably excludes this class — a DISTINCT gap, not a landed fix.

## Proposed change (candidate diff sketch — refine in planning)

+ Add `layers?` to the pre_reg head-noun alternation; MEASURE (per the file's corpus discipline: full-pattern old-vs-new match-start diff over all promoted/parked bodies) the critic's sketched siblings `lattice|rungs?|windows?|bands?|reads?|margins?` and include only those with 0 (or explicitly accepted) benign verb-use false positives — `margin`/`read` are the likely FP-risk cases to measure carefully.
+ Update the pattern's provenance comment with the #1586 incident strings ("the registered layer", "reported at the registered layer").

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; the file's measured-corpus discipline (0 unaccounted FPs) is the acceptance bar, matching #1419/#1475.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 4a643434c286

Surfaced prose (verbatim from the clean-result-critic round-1 report): "mechanizable: yes — audit regex \bregistered\s+(layer|verdict|margin|read|lattice|rung|window|band)\b over the four content sections (…the sketch is in the marker for routing)."
