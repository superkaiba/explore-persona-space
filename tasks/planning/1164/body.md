---
title: 'workflow-fix: Mechanize the reused-artifact realized-keys pr'
kind: infra
tags:
- wf-fix
- wf-fix-fp:354a02c7ace9
- daily-auto-filed
created_at: '2026-07-09T06:58:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The realized-keys check
  landed as rule prose via #1087 (artifact-reuse.md check (c)/(h)(ii): torch.load(mmap=True).keys()
  / consumer-loader-open, commit b61d3e623f), but nothing runs it mechanically — a
  planner/implementer who skips the prose check still stages a bundle whose realized
  key set misses the consumer''s asserts (origin incident #1073).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1087 by a recursion-guarded workflow-fix session.

## Goal

Mechanize the realized-keys check: a pre-provision key-schema probe (torch.load(path, map_location='cpu', mmap=True).keys() superset-of the consumer's asserted keys) in the reuse-staging path, or a pytest running the consumer loader against pinned bundles.

## Workflow gap

- **Bug observed:** The realized-keys check landed as rule prose via #1087 (artifact-reuse.md check (c)/(h)(ii): torch.load(mmap=True).keys() / consumer-loader-open, commit b61d3e623f), but nothing runs it mechanically — a planner/implementer who skips the prose check still stages a bundle whose realized key set misses the consumer's asserts (origin incident #1073).
- **Why it is a workflow gap:** Prose-only fitness checks in artifact-reuse.md depend on agent compliance; both #1073 sweep sites evaded the manual check, and the critics that reviewed them missed it too — the same class the staged-layout consumer-open probe (#928) mechanized for layout.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** Rule-level fix confirmed present (artifact-reuse.md lines 116-117, 181 — landed by #1087 itself, commit b61d3e623f); no mmap key-schema probe exists in scripts/ or the backends (grep 2026-07-08); open wf-fix #865 on scripts/ is unrelated (Step 9c selector worktree blindness). The MECHANIZATION leg is the unimplemented residue.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; likely shape: a helper in the reuse-staging path that torch.loads each staged bundle with mmap=True and asserts .keys() is a superset of the consumer's declared key list, invoked pre-provision; or a verify_plan.py-adjacent check + pytest against pinned fixture bundles)

## Scope / surfaces

- Primary target: `scripts/verify_plan.py, scripts/ (reuse-staging helpers), tests/`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: scripts/verify_plan.py, scripts/ (reuse-staging helpers), tests/
- origin: parked candidate on task #1087 at 2026-07-06T11:41:55Z

Verbatim parked note:

> parked — running under workflow_fix_target Provenance / EPM_WORKFLOW_FIX_SESSION (recursion guard, see workflow-fix-on-bug.md § Recursion guard). Candidate surfaced as prose by the Alternatives critic ensemble (Claude + Codex both, non-blocking): MECHANIZE the realized-keys check — a pre-provision key-schema probe (torch.load(mmap=True).keys() superset-of consumer asserts) in the reuse-staging path, or a pytest running the consumer loader against pinned bundles; target_file: scripts/ reuse-staging helpers / verify_plan.py-adjacent check; confidence: medium; related_task: #1087 (origin #1073). NOT auto-routed by this session; parked for the next human/orchestrator pass (nightly /daily sweep will surface it).
