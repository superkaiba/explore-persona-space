---
title: 'workflow-fix: pre_reg audit gains pre-set/declared/specified synonym branch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ab8ac65ac828
created_at: '2026-08-01T05:47:15Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by clean-result-critic on task #1945
  round 1 (verbatim block preserved in the task body ## Provenance section): pre_reg
  audit regex misses the pre-set/pre-declared/pre-specified synonym family adjacent
  to verdict-class nouns; fingerprint ab8ac65ac828'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1945 (emitting agent: clean-result-critic, round 1).

## Goal

Extend the `pre_reg` pattern in `scripts/audit_clean_results_body_discipline.py`
with a `pre-(set|declared|specified)` synonym branch adjacent to a verdict-class
noun, so the mechanical backstop catches the pre-registration-mention synonym
family the LM lens currently has to catch by judgment.

## Workflow gap

- **Bug observed:** The #1945 body carried "The pre-set verdict lattice"
  (Takeaways) and "the pre-declared fallback" (Methodology prose) yet the
  audit's `pre_reg` check passed clean — the regex covers
  "pre-registered"/"registered <noun>" but not the
  "pre-set"/"pre-declared"/"pre-specified" synonym family.
- **Why it is a workflow gap:** The statistical-framing rule
  (clean-result-critic Lens 7) bans pre-registration mentions semantically,
  but the mechanical backstop only matches the "registered" lexeme, so synonym
  escapes recur and depend on LM judgment to catch. The file's own
  `verdict_caps` comment (L286) documents the same family escaping before
  ("Under the pre-set decision rule, SUCCESS was not met", the #763/#970
  residual) — that fix covered only CAPS verdict tokens, not the prose
  modifier itself.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -nE "pre-?\(?(set|declared|specified)" scripts/audit_clean_results_body_discipline.py` → 1 hit in 1 file: a COMMENT at L286 inside the `verdict_caps` check documenting the historical "#763 pre-set decision rule" escape (context read per clause (c): the hit does NOT implement the proposed synonym branch); per-target presence: the `pre_reg` pattern block confirmed at L74–L145 with no `pre-(set|declared|specified)` alternative; landed-fix history `git log --oneline --since='7 days ago' -- scripts/audit_clean_results_body_discipline.py` → `2c437f216e` (#1595 head-noun family) + `a865cf8a91` (#1537 bare registered-noun) — neither adds the synonym family (2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

In audit_clean_results_body_discipline.py "pre_reg" pattern (near line 74):

```
+ add alternative branch:
+   r"\bpre-?(set|declared|specified)\b(?:\s+\w+){0,3}?\s+(verdict|lattice|read|margin|floor|threshold|fallback|hypothesis)"
+ (and the noun-first order: "the (verdict|lattice|...) (was|were) pre-(set|declared)")
  keep existing exemptions (Methodology hyperparameter-table rows, Why-this-test sentences).
```

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'pre_reg' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan. The pin tests
  (`tests/` files exercising the audit) likely need fixture rows for the new
  branch — the #1595/#1537 precedent commits show the expected shape.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- False-positive discipline: the new branch must keep benign uses clean
  (e.g. "preset" as one word, hyperparameter-table rows, Why-this-test
  sentences — reuse the existing exemption plumbing).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: ab8ac65ac828

<!-- workflow-fix-candidate v1 -->
target_file: scripts/audit_clean_results_body_discipline.py
bug_observed: The body carried "The pre-set verdict lattice" (Takeaways) and "the pre-declared fallback" (Methodology prose) yet the audit's pre_reg check passed clean — the regex covers "pre-registered"/"registered <noun>" but not the "pre-set"/"pre-declared"/"pre-specified" synonym family.
why_workflow_gap: The statistical-framing rule (clean-result-critic Lens 7) bans pre-registration mentions semantically, but the mechanical backstop only matches the "registered" lexeme, so synonym escapes recur and depend on LM judgment to catch.
proposed_change: Extend the pre_reg pattern with a pre-(set|declared|specified) branch adjacent to a verdict-class noun (verdict/lattice/read/margin/floor/threshold/fallback/hypothesis), scoped to the same prose sections and with the existing Why-this-test/table exemptions.
diff_sketch: |
  In audit_clean_results_body_discipline.py "pre_reg" pattern (near line 74):
  + add alternative branch:
  +   r"\bpre-?(set|declared|specified)\b(?:\s+\w+){0,3}?\s+(verdict|lattice|read|margin|floor|threshold|fallback|hypothesis)"
  + (and the noun-first order: "the (verdict|lattice|...) (was|were) pre-(set|declared)")
  keep existing exemptions (Methodology hyperparameter-table rows, Why-this-test sentences).
confidence: medium
related_task: #1945
<!-- /workflow-fix-candidate -->
