---
title: 'workflow-fix: decide bit-deterministic vs bit_byte_identical ban family (allowlist
  or widen)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4268862fa320
created_at: '2026-07-23T05:20:45Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #1415 (critique v7): whether
  bit-deterministic/bit-determinism should join or be explicitly allowlisted against
  the byte/bit-identical ban family in audit_clean_results_body_discipline.py'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1415 (emitting agent: clean-result-critic, hooked-unhooked-decomposition
fold round 1 — surfaced as a mechanizable non-blocking follow-up in its verdict marker
`epm:clean-result-critique` v7).

## Goal

Decide whether "bit-deterministic" / "bit-determinism" joins the `bit_byte_identical`
ban family in `scripts/audit_clean_results_body_discipline.py`, or is explicitly
allowlisted (a code comment + test pinning it as legitimate determinism vocabulary),
so future clean-result critics stop re-raising the ambiguity.

## Workflow gap

- **Bug observed:** the #1415 fold body uses "bit-deterministic" / "bit-determinism"
  (describing bit-identical re-forwards at causally-zero layers); the audit passes it
  silently, and the reviewing critic could not tell whether that is an allowlisted
  determinism term or an unwidened escape of the byte/bit-identical ban family.
- **Why it is a workflow gap:** the `bit_byte_identical` category has been widened
  twice recently (issue-1423 -equal family, issue-1447 synonym tail) with no recorded
  decision boundary for "deterministic"; every future critic re-litigates it.
- **Confidence (emitter):** low
- verified-at-filing: functional probe of the category regex
  `(?<!-)\b(?:byte|bit)(?:wise)?[\s-](?:identical|equal|exact)\b` at compose time:
  "bit-deterministic" -> no match, "bit-determinism" -> no match, "bit identical" ->
  MATCH (2026-07-23); `grep -n -i "bit.determin" scripts/audit_clean_results_body_discipline.py`
  -> 0 hits in the target file; landed-fix history
  `git log --oneline --since='7 days ago' -- scripts/audit_clean_results_body_discipline.py`
  -> d1afe0730d + ab53595c54 widened the identical/equal/exact tail only, neither
  covers "deterministic" (not a just-landed duplicate).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; two candidate shapes for the planner:)
- EITHER extend the `bit_byte_identical` alternation with `deterministic|determinism`
  (if the register is the same banned claim-shape),
- OR add an explicit allowlist comment + a pin test asserting "bit-deterministic"
  is deliberately NOT flagged (legitimate determinism vocabulary describing exact
  re-forward reproducibility, as in #1415's jitter-floor evidence).

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'bit_byte_identical' .claude/ CLAUDE.md scripts/ tests/`) and update
  every hit; list them in the plan (expect the audit script + its tests +
  `.claude/rules/clean-result-critic-lens-reference.md` prose).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Grandfathered bodies must not be newly hard-FAILed: if the ban is widened, the
  #1415 body (which legitimately describes bit-identical determinism EVIDENCE, not a
  results claim) must either pass via context-scoping or be exempted — the planner
  weighs this; a widening that FAILs #1415's honest determinism evidence is wrong.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 4268862fa320

Verbatim surfaced prose (clean-result-critic, epm:clean-result-critique v7 on #1415):
"whether 'bit-deterministic'/'bit-determinism' (used in Methodology + plan v11) should
join or be explicitly allowlisted against the byte/bit-identical ban family in
audit_clean_results_body_discipline.py (low confidence — likely legitimate determinism
vocabulary; mechanizable)."
