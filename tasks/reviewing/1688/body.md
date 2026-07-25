---
title: 'workflow-fix: step9c selector misses string+transitive deps'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8f41d6f799ff
created_at: '2026-07-25T15:49:04Z'
has_clean_result: false
origin_prompt: 'code-reviewer prose follow-up on #1683 r1: select_step9c_tests.py
  mapping for scripts/issue667_extract.py misses string-reference pins and transitive-import
  consumers — tests/test_issue671_extraction_hooks.py, tests/test_issue811_dispatch.py,
  tests/test_issue833_nonemit_filters.py escape the Step 9c gate''s 73-file universe'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1683 (emitting agent: code-reviewer, round 1).

## Goal

Extend `scripts/select_step9c_tests.py` dependency arms to map dotted-module string references and transitive-import test consumers of touched scripts.

## Workflow gap

- **Bug observed:** the code-reviewer's whole-tree pin-sweep on #1683 found 3 real consumer/pin test files OUTSIDE the selector's 73-file universe for `scripts/issue667_extract.py`: `tests/test_issue671_extraction_hooks.py` (dotted-module string refs, e.g. `("scripts.issue667_extract", "_mean_resp_acts")` at lines 426-428), `tests/test_issue811_dispatch.py` (filename literal `issue667_extract.py` in docstring line 5 + assert strings lines 173/183), `tests/test_issue833_nonemit_filters.py` (transitive import — imports `issue833_extract_onpolicy`, which imports `issue667_extract`; zero direct string hits). All three escape every touched-scope Step 9c gate pulling this file; all three passed when run by hand this round (69 passed), so no live red — a coverage gap, not a regression.
- **Why it is a workflow gap:** the selector's dependency map (import/literal/stem arms, #1573) is the single source for the Step 9c gate universe AND the Step 10d mapped invariant-test leg; a dotted-module string form (`scripts.issue667_extract`) and a second import hop are invisible to it, so gate-relevant pin tests silently escape both gates for any touched script with such consumers.
- **Confidence (emitter):** high.
- verified-at-filing: `printf 'scripts/issue667_extract.py\n' | uv run python scripts/select_step9c_tests.py --map-files -` → 8 mapped test files, NONE of the 3 named escapees present; `grep -n 'issue667_extract' tests/test_issue671_extraction_hooks.py` → dotted-module tuples at :426-:428; same grep on `tests/test_issue811_dispatch.py` → filename-literal hits at :5/:173/:183; same grep on `tests/test_issue833_nonemit_filters.py` → 0 hits (transitive-import form) (2026-07-25, filed from /issue 1683 round 1).

## Proposed change (candidate diff sketch — refine in planning)

```
+ literal arm: also match the dotted-module form of a touched script
+   ("scripts/issue667_extract.py" -> "scripts.issue667_extract") in test text
+ import arm: either add ONE more import hop (test -> module A -> touched module),
+   bounded to scripts/-local modules, or record a deliberate decision against
+   transitive mapping with the whole-tree pin-sweep (#1288 implementer duty)
+   named as the compensating control
```

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Sibling surfaces to keep consistent: `tests/test_select_step9c_tests.py` (drift pins, #895); the Step 10d mapped-leg prose in `.claude/skills/issue/SKILL.md` only if the mapping CONTRACT changes (avoid if possible).
- Grep the workflow surface for the pattern before editing (`grep -rln 'map-files\|GLOB_SCAN_TESTS' .claude/ scripts/ tests/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; selector drift pins (`tests/test_select_step9c_tests.py`) updated in the same round.
- Do not balloon the gate universe: the new arms must stay bounded (dotted-form is exact-match; a transitive hop, if adopted, is one hop and scripts/-scoped).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: 8f41d6f799ff

Verbatim surfaced prose (code-reviewer, #1683 round 1): "Follow-up (orchestrator should consider): `scripts/select_step9c_tests.py`'s mapping for `scripts/issue667_extract.py` misses string-reference pins and transitive-import consumers — those 3 files also escape the Step 9c gate's 73-file universe." (Named files: tests/test_issue671_extraction_hooks.py, tests/test_issue811_dispatch.py, tests/test_issue833_nonemit_filters.py — all run green by the reviewer, 69 passed.)
