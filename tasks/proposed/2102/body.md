---
title: 'workflow-fix: workflow.yaml wf-fix prose vs 8 daily-doc pins (main-red)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0ab238f4e806
created_at: '2026-08-05T21:41:09Z'
has_clean_result: false
origin_prompt: 'Step 9c compare urgent_park_required on #2092: 8 pre-existing main-red
  workflow-invariant nodes in tests/test_daily_three_route_classifier_doc.py; YAML-surface
  prose dropped by c20aabc59a slimming'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol (urgent main-red subclass, #1713/#1742) from the Step 9c gate of task #2092 (emitting agent: issue-2092 orchestrator; compare urgent_park_required, 8 nodes).

## Goal

Make `tests/test_daily_three_route_classifier_doc.py` green on pristine main again: EITHER restore the `workflow_fix_on_bug` orchestrator-actions prose in `.claude/workflow.yaml` (the `verified-at-filing:` grep-evidence line + the clause mentions the 8 pins assert on the YAML surface) OR re-pin the 8 tests to the slimmed pointer-style YAML (`enabled: true` + `protocol_doc:` delegation) — whichever the `c20aabc59a` slimming intended.

## Workflow gap

- **Bug observed:** 8 pins in `tests/test_daily_three_route_classifier_doc.py` are red on PRISTINE MAIN (nodes: test_verified_at_filing_line_required, test_context_consistency_clause_present, test_semantic_probe_absence_clause_present, test_sha_verification_duty_present, test_artifact_state_mutation_clause_present, test_marker_existence_clause_present, test_call_hop_target_tracing_clause_present, test_suppression_predicate_clause_present). Representative failing assert (reproduced 2026-08-05): `AssertionError: workflow.yaml orchestrator_actions no longer mention the verified-at-filing: grep-evidence line (#1272)` — the test's YAML-surface leg; the daily-SKILL.md and workflow-fix-on-bug.md legs still pass.
- **Why it is a workflow gap:** commit `c20aabc59a` ("workflow: remove the architectural-greenlight gate", 2026-08-04) slimmed workflow.yaml's `workflow_fix_on_bug` section to `enabled: true` + `protocol_doc: ".claude/rules/workflow-fix-on-bug.md"`, dropping the prose these tri-surface pins grep on the YAML leg. Every Step 9c gate selection since then carries 8 pre-existing-red workflow-invariant nodes that each intervening session must re-classify (the #1713 fleet-wide per-hour cost).
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'verified-at-filing:' .claude/workflow.yaml` → 0 hits (absence-of-prose claim: the 0-hit in-target IS the evidence; the pin requires ≥1); `git rev-parse --verify c20aabc59a^{commit}` resolves (the slimming commit, latest touch of `.claude/skills/daily/SKILL.md`-adjacent workflow surface); `uv run pytest tests/test_daily_three_route_classifier_doc.py -q` → 8 failed, reproduced on a pristine-scratch oracle at main `82a05837f091` (task #2092 Step 9c compare, 2026-08-05, via=pristine-scratch on all 8) (2026-08-05 UTC).

urgency: main-red
failing_test: tests/test_daily_three_route_classifier_doc.py::test_verified_at_filing_line_required
wf_fix: true

## Proposed change (candidate diff sketch — refine in planning)

Two admissible shapes; the planner picks against `c20aabc59a`'s intent:

Option A (restore prose): re-add a short orchestrator-actions block under `workflow_fix_on_bug:` in workflow.yaml carrying the `verified-at-filing:` line + one-line mentions of clauses (a)-(h) (context-consistency, semantic-probe, sha-verification, artifact-state-mutation, marker-existence, call-hop-target-tracing, suppression-predicate) — the pins pass unchanged.

Option B (re-pin): update the 8 tests' YAML-surface legs to assert the slimmed shape (`workflow_fix_on_bug.enabled` + `protocol_doc` pointing at the rule file, which retains ALL clause text) — the single-source-of-truth reading of the slimming.

## Scope / surfaces

- Primary target: `.claude/workflow.yaml`, `tests/test_daily_three_route_classifier_doc.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'verified-at-filing' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- The Step 5a family-atomic sync couples workflow.yaml with `.claude/skills` markers.md and the `tests/test_issue_skill_*.py` prose pins — regenerate tables (`workflow_lint.py --emit-tables`) if workflow.yaml prose changes.

## Provenance

- workflow_fix_target: .claude/workflow.yaml, tests/test_daily_three_route_classifier_doc.py
- fingerprint: 0ab238f4e806

<!-- workflow-fix-candidate v1 -->
target_file: .claude/workflow.yaml, tests/test_daily_three_route_classifier_doc.py
bug_observed: 8 pins in tests/test_daily_three_route_classifier_doc.py red on pristine main since the workflow.yaml workflow_fix_on_bug section was slimmed to enabled+protocol_doc, dropping the prose the pins assert on the YAML surface
why_workflow_gap: every Step 9c gate selection carries 8 pre-existing-red workflow-invariant nodes each session must re-classify (the #1713 fleet-wide cost); the slimming commit c20aabc59a and the tri-surface pins disagree about where the wf-fix prose lives
proposed_change: restore the workflow.yaml workflow_fix_on_bug orchestrator-actions prose (verified-at-filing + clause mentions) OR re-pin tests/test_daily_three_route_classifier_doc.py to the slimmed pointer-style YAML
diff_sketch: |
  Option A: + orchestrator-actions prose block under workflow_fix_on_bug: (verified-at-filing: line + clause (a)-(h) mentions)
  Option B: - assert "verified-at-filing:" in yaml_text  → + assert workflow_fix_on_bug pointer shape (enabled + protocol_doc → rule file carrying the clauses)
urgency: main-red
failing_test: tests/test_daily_three_route_classifier_doc.py::test_verified_at_filing_line_required
wf_fix: true
confidence: high
related_task: #2092
<!-- /workflow-fix-candidate -->
