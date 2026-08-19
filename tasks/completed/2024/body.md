---
title: 'workflow-fix: step9c compare — branch-untouched-file NEW failures classify
  ordering-suspect, not blocking NEW'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e07ae04da31a
created_at: '2026-08-02T19:57:05Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation on #2021 Step 9c: compare false-NEW''d 17
  pre-existing import-ordering failures (single-file pristine oracle blind to cross-module
  interactions); manual pair-repro override needed. Evidence: #2021 epm:test-verdict
  v1.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #2021 (emitting agent: /issue orchestrator, Step 9c 1d).

## Goal

Make `scripts/step9c_baseline.py compare` classify a NEW-labeled failure whose FILE is untouched by the branch diff as ordering-suspect / pre-existing-candidate (branch-diff provenance check, or a paired-selection pristine oracle) instead of an unconditional NEW block.

## Workflow gap

- **Bug observed:** On #2021's Step 9c gate, compare (COMPARE_RC=1) classified 17 failures in `tests/test_issue1739_pvsynth_arms.py` + `tests/test_issue1739_wcrung_arms.py` as NEW, though the branch's diff for both files is EMPTY and the red is a pre-existing cross-module import-ordering interaction (the #1739 scripts' `_assert_no_judge_modules("at entry")` spend guard reads `sys.modules`): the pristine oracle runs failing files SINGLE-FILE, where these tests pass, so ordering-dependent trunk red is structurally invisible to it. The orchestrator had to hand-run a branch-untouched-pair repro on pristine main (`pytest tests/test_batch_judge_agg_non_dict_parse.py tests/test_issue1739_wcrung_arms.py -q` → 11 failed in 3.02s) and provenance-override the verdict — a full manual re-proof cycle every co-selecting session will repeat until #1739 fixes its fixture.
- **Why it is a workflow gap:** compare's verdict is specified as authoritative for the Step 9c gate ("The COMPARE verdict — not the raw PYTEST_RC — decides pass/fail"), but its single-file pristine oracle carries a documented degeneracy family (scan-test aggregation); this incident shows a second member (ordering interaction) with no mechanical escape, forcing prose-arithmetic overrides — exactly what #1022 removed.
- **Confidence (emitter):** medium (the right mechanism needs the planner's call: cheap branch-diff-provenance downgrade vs paired-selection oracle re-run)
- verified-at-filing: `grep -n 'ordering' scripts/step9c_baseline.py` → 1 hit (L847, an unrelated `.pth`-reordering comment) — no existing ordering-interaction handling in compare (absence claim); the #2021 incident evidence is on task #2021's `epm:test-verdict v1` marker (2026-08-02T19:55:31Z) with the repro log at /tmp/i2021-preexist-repro.log; `git log --oneline --since='7 days ago' -- scripts/step9c_baseline.py` checked at compose time by the spawned session's planner (this filer's landed-fix check: no compare-oracle ordering fix in the recent step9c commits surfaced by the #1399 advisory at the sparse-cones filing earlier today).

## Proposed change (candidate diff sketch — refine in planning)

```
+ In compare's NEW-classification path: for each NEW-candidate node whose FILE has an
+ empty `git diff <base>...HEAD -- <file>` (branch never touched it), either
+ (a) downgrade to a new `ordering-suspect` stripped class (report, never block), or
+ (b) re-run the pristine oracle with a PAIRED selection (the failing file + the
+   run's co-selected judge-importing predecessors) before labeling NEW.
+ Keep true-NEW semantics for any node whose file IS in the branch diff.
```

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py`
- Secondary: the Step 9c 1d prose in `.claude/skills/issue/SKILL.md` if the verdict vocabulary gains the new class; `tests/test_select_step9c_tests.py` / step9c pin tests for the new behavior.
- Grep the workflow surface for the pattern before editing (`grep -rln 'run-pristine\|pristine oracle' scripts/ .claude/skills/issue/SKILL.md`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Fail-CLOSED direction preserved: a node whose file IS branch-touched keeps today's NEW block; the downgrade applies only on mechanical branch-diff-empty evidence.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/step9c_baseline.py
- fingerprint: e07ae04da31a

Orchestrator observation (verbatim summary): compare classified 17 pre-existing cross-module import-ordering failures as NEW because its pristine oracle runs files single-file where they pass; branch-diff provenance (empty diff for the failing files) + a pristine-main pair repro were needed to override — evidence on #2021 epm:test-verdict v1.
