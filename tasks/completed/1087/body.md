---
title: 'workflow-fix: reused-artifact realized-keys verification (gotchas + reuse
  check c)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7fb20ce74926
created_at: '2026-07-06T08:01:16Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson from #1073 crash-fix r3: verify a
  reused artifact''s realized key set against consumer asserts; builder-code reading
  is not verification (bug_class reused_artifact_schema_drift).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1073 (emitting agent: experiment-implementer, crash-fix round 3).

## Goal

Add a `.claude/rules/gotchas.md` entry (and a one-line `.claude/rules/artifact-reuse.md` check-(c) clarification) requiring verification of a reused artifact's REALIZED key set against the consumer's asserts — builder-code reading is not verification.

## Workflow gap

- **Bug observed:** #1073 p0 crash (att-20260706-071820): the pinned pass_b bundle lacked the `prompts` field that TODAY'S builder code (`issue779_collect.py:592`) writes; the plan fact-checker AND the smoke fixture both mirrored the builder code, so both missed the schema drift; the consumer's hard assert killed P0 on a GCE provision (~8 min billed + a full launch cycle).
- **Why it is a workflow gap:** artifact-reuse check (c) says "required cells present" but its verification recipe (list_repo_tree / get_paths_info) proves file EXISTENCE, not the file's INTERNAL key schema; nothing in the workflow surface tells a planner/fact-checker/implementer that a 6 GB .pt's realized keys can drift from the current builder's save dict.
- **Confidence (emitter):** high (root_cause_confirmed)

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new entry "Reused artifact's REALIZED keys can predate the builder code":
+   - verify the artifact's OWN keys against the consumer's asserts (torch.load(map_location='meta'/mmap) key check, or run the consumer's loader against the real artifact) at plan/smoke time
+   - a deterministically-regenerable missing field → regenerate via the parent loader with fail-loud source/length asserts + a deterministic re-capture alignment gate against the stored tensors (row alignment is the invariant)
+   - smoke fixtures default to the PRODUCTION artifact shape, not the builder-code shape
+ artifact-reuse.md check (c): add one line — "presence of the FILE is not presence of the FIELDS: for multi-field tensor bundles, verify the realized key set (cheap header/mmap read) against every consumer assert"

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md,.claude/rules/artifact-reuse.md`
- Grep for the pattern first (`grep -rln 'realized key\|bundle missing field' .claude/ scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. Keep artifact-reuse.md's enforcement-chain surfaces in sync (planner step 5 / consistency-checker / critic item 9) if the check-(c) wording changes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md,.claude/rules/artifact-reuse.md
- fingerprint: 7fb20ce74926

Verbatim failure-lesson (the surfaced candidate): see `epm:failure-lesson v1` on task #1073 (2026-07-06T08:00:20Z) — bug_class reused_artifact_schema_drift.
