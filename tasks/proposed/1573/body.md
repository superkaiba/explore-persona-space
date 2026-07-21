---
title: 'workflow-fix: step9c --map-files misses tests importing/source-inspecting
  a changed src module'
kind: infra
tags:
- wf-fix
- wf-fix-fp:219c0bffd790
created_at: '2026-07-21T02:22:50Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from experiment-implementer on #1112 rankem round:
  select_step9c_tests --map-files src/.../train/sft.py returned EMPTY yet the change
  broke test_artifacts_recipe.py::test_rslora_engine_pin (imports + inspect.getsource
  train_lora); mapped-test oracle misses symbol-level dependencies.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1112 (emitting agent: experiment-implementer, rankem round).

## Goal

Make `select_step9c_tests.py --map-files` map tests that statically import a changed `src/` module (and consider symbol-level source-inspection dependencies), or fail loud when a `src/` path yields zero mapped tests.

## Workflow gap

- **Bug observed:** `--map-files src/explore_persona_space/train/sft.py` returns an empty test mapping while `tests/test_artifacts_recipe.py` imports `train_lora` from that module (line 47) AND `inspect.getsource`-inspects it (line 553); a `use_rslora` change to `train_lora` (commit d7908a3837) broke `test_artifacts_recipe.py::test_rslora_engine_pin` and the selector never surfaced it — the implementer caught it only by running sibling tests directly.
- **Why it is a workflow gap:** the Step-9c mapped-scan gate and the inline payload lint gate both rely on this selector; a round can push `src/` payload that breaks the fleet-wide test gate without either gate firing. The miss is not only the exotic source-inspection case — a plain static import edge was not mapped.
- **Confidence (emitter):** low (raised to medium by filer verification — the miss reproduces).
- verified-at-filing: `uv run python scripts/select_step9c_tests.py --map-files src/explore_persona_space/train/sft.py` → 0 mapped tests (header only); `grep -n 'getsource\|train_lora' tests/test_artifacts_recipe.py` → import at :47, `inspect.getsource(train_lora)` at :553; `grep -rln 'inspect.getsource' tests/` → 10+ test files use source inspection (breadth of the symbol-level blind spot); `git log --oneline --since='7 days ago' -- scripts/select_step9c_tests.py` → 5 commits, none adding src-module import mapping (subjects: staged-index verification #1346, guard-surface rounds #1338, digest crash-tails #1316, ruff pin #1307, rules-pin discovery #1270) (2026-07-20).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; the emitter's sketch in words: extend the mapper's dependency edges from filename/import-graph heuristics to (a) tests importing the changed module path and (b) tests source-inspecting a changed symbol; alternatively a fail-loud "0 tests mapped for a src/ change" warning so the gap is visible at gate time.)

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'map-files\|map_files' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: 219c0bffd790

Verbatim surfaced prose (experiment-implementer report, task #1112 rankem round, 2026-07-20):
> `scripts/select_step9c_tests.py --map-files src/.../train/sft.py` returned EMPTY, yet my change to `train_lora` broke `test_artifacts_recipe.py::test_rslora_engine_pin`, which `inspect.getsource(train_lora)`. The mapped-test oracle misses tests that source-inspect/import the changed SYMBOL (not just import-graph/naming). Backstop (running sibling tests directly) caught it — low confidence, but worth a look for the step9c selector.
