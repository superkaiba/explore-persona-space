---
title: 'workflow-fix: Step-10d surgical checkout runs GLOB_SCAN_TESTS-mapped tests
  pre-landing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4cb4074011ab
created_at: '2026-07-08T14:24:33Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #1145 Alternatives critic r1: the Step 10d surgical
  additive checkout bypasses every pytest gate; run the GLOB_SCAN_TESTS-mapped invariant
  tests over surgically-landed scripts/issue*_*.py before the landing commit, baseline-subtracted,
  blocking NEW failures.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1145 (emitting agent: critic, Alternatives lens).

## Goal

Add a pre-landing invariant-test leg to the `/issue` Step 10d surgical additive checkout: run the `GLOB_SCAN_TESTS`-mapped tests (at minimum `tests/test_shared_vm_thread_caps.py`) over any surgically-landed `scripts/issue*_*.py` files and block the landing on NEW failures — closing the accretion channel that produced #1144/#1145's 34-offender trunk redness.

## Workflow gap

- **Bug observed:** offender scripts accreted onto main via the Step-10d surgical/artifact-confirmed checkout path, which bypasses every pytest gate (34+ scripts landed with torch-before-dotenv violations; the invariant test could not stop them — #1144 finding, #1145 fix).
- **Why it is a workflow gap:** the surgical additive checkout lands `scripts/` files on `main` with a workflow-lint gate but NO test gate; `GLOB_SCAN_TESTS`-mapped invariant tests exist precisely to gate these files and are skipped on this path.
- **Confidence (emitter):** medium-high

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/issue/SKILL.md` § surgical additive checkout (and the fast-path form): after the gated workflow-lint leg and before `git add`, when the additive list contains `scripts/issue*_*.py` paths, run the mapped invariant test(s) (from `scripts/select_step9c_tests.py::GLOB_SCAN_TESTS`) against the payload-bearing tree with the same baseline-vs-gated known-red subtraction the lint gate uses; block the landing on NEW failures.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for sibling landing paths before editing (`grep -rln 'surgical additive checkout' .claude/ CLAUDE.md scripts/`) and update every hit; consider whether `scripts/select_step9c_tests.py` needs a helper entrypoint for "tests mapped to this file list".

## Constraints / invariants

- Workflow-surface only; keep the gate baseline-subtracted (pre-existing trunk red must never block an innocent landing — same discipline as the pre-push workflow-lint gate).
- `scripts/workflow_lint.py --check-asks` passes; tables/references lint clean if SKILL.md structure changes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 4cb4074011ab

Surfaced prose (Alternatives critic, #1145 round 1): "a workflow-surface follow-up on `.claude/skills/issue/SKILL.md` (Step 10d surgical-checkout path) to run the `GLOB_SCAN_TESTS`-mapped tests (at minimum `tests/test_shared_vm_thread_caps.py`) over any surgically-checked-out `scripts/issue*_*.py` before merge — closing the accretion channel #1144/#1145 documented."
