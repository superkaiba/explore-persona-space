---
title: 'workflow-fix: Fix trunk-red dotenv-before-torch test (2 issu'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6c6ffecdfdef
- daily-auto-filed
created_at: '2026-07-09T06:57:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  FAILs on pristine main: scripts/issue779_ffc_baselines.py and scripts/issue779_ffc_summary_crosslayer_input.py
  import torch/numpy at module top before load_dotenv() with no grandfather entries
  (landed via issue-branch merges) — trunk-red for the consumer suite (#847 invariant).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1079 by a recursion-guarded workflow-fix session.

## Goal

Apply the #933 route (commit c8e6b21a95): move load_dotenv() before the heavy imports in the two offending entrypoints, or add grandfather entries in tests/test_shared_vm_thread_caps.py; also fix the stale code-style.md anchor (line 49 cites vectorized_mlp_skill.py:175 but the torch.set_num_threads pin lives at ~line 224).

## Workflow gap

- **Bug observed:** tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints FAILs on pristine main: scripts/issue779_ffc_baselines.py and scripts/issue779_ffc_summary_crosslayer_input.py import torch/numpy at module top before load_dotenv() with no grandfather entries (landed via issue-branch merges) — trunk-red for the consumer suite (#847 invariant).
- **Why it is a workflow gap:** Issue-branch merges keep landing VM entrypoints that violate the dotenv-before-torch invariant, and nothing gates the merges; the trunk-red test blocks every session running the suite.
- **Confidence (emitter):** medium (emitter); trunk-red leg verified high
- **Sweep verification (2026-07-08):** Verified failing on main 2026-07-08: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` FAILED with exactly 2 offenders (down from ~10 at park time — partial cleanup happened, residue remains). Stale-anchor leg verified: code-style.md:49 cites vectorized_mlp_skill.py:175; set_num_threads calls are at lines 224/824. Third sub-item (lint that committed supervisor/relaunch scripts carry the thread-pin prefix) remains unimplemented — #891's check_vm_thread_cap_guidance pins only the 4 guidance surfaces, not committed scripts; planner may fold it in or deflect.

## Proposed change (candidate diff sketch — refine in planning)

In each offending script: move `load_dotenv()` (orchestrate.env) above the torch/numpy imports (the c8e6b21a95 pattern), or add the files to the test's grandfather list with a reason. One-line anchor fix in code-style.md line 49 (':175' -> the live set_num_threads site).

## Scope / surfaces

- Primary target: `tests/test_shared_vm_thread_caps.py, scripts/issue779_ffc_baselines.py, scripts/issue779_ffc_summary_crosslayer_input.py, .claude/rules/code-style.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- The two `scripts/issue779_*.py` entrypoint edits are the narrow #933-precedent exception (mechanical load_dotenv reorder to satisfy a workflow-invariant test); everything else stays workflow-surface only — never `configs/` or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: tests/test_shared_vm_thread_caps.py, scripts/issue779_ffc_baselines.py, scripts/issue779_ffc_summary_crosslayer_input.py, .claude/rules/code-style.md
- origin: parked candidate on task #1079 at 2026-07-06T08:02:06Z

Verbatim parked note:

> parked — running under workflow_fix_target recursion guard, see workflow-fix-on-bug.md § Recursion guard. Candidate (implementer-surfaced prose, round 1): tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints FAILs on pristine main — ~10 scripts/issue811_*/issue813_rank_spectrum/issue833_* entrypoints import torch/numpy before load_dotenv() with no grandfather entries (landed via recent issue-branch merges, e.g. 783df07d48). Real #847-invariant gap in experiment entrypoints; fix pattern = the c8e6b21a95 #933 route (add load_dotenv-before-torch or grandfather). Second parked candidate (plan-surfaced): .claude/rules/code-style.md § Shared-VM CPU thread caps cites stale anchor vectorized_mlp_skill.py:175. Third (critic-surfaced, low): future workflow_lint check that committed supervisor/relaunch scripts carry the thread-pin prefix. NOT auto-filed from this session; next non-guarded orchestrator/PM pass routes them.
