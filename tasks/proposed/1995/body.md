---
title: 'workflow-fix: carry-over gate path regex misses include-tree'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c0ce6e284ef5
- daily-auto-filed
created_at: '2026-08-02T07:06:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): A plan-text citation of
  a committed input under an include-tree prefix outside eval_results|ood_eval_results|data
  (e.g. `tests/fixtures/eval_results/a.json`) produces NO finding at all — channel-A
  _PATH_RE only matches those three prefixes, so both the clone-lane reachability
  check and the new #1915 rsync exclude subtraction are blind to it in plan text (the
  #1915 pin tests enter via classify() di'
workflow: v1
---
# workflow-fix: carry-over gate channel-A path regex misses include-tree prefixes

## Overview / Motivation

Auto-filed by the /daily 2026-08-01 Step C parked-candidate sweep from a workflow-fix candidate parked on task #1915 (emitting agent: implementer round 1, recursion-guarded; formal candidate block, fingerprint 7ddb0181b608).

## Goal

Widen `verify_carryover_inputs.py`'s channel-A `_PATH_RE` so plan-text citations of committed inputs under `tests/`, `scripts/`, `configs/` prefixes are extracted and audited like the three currently-covered prefixes, with a corpus sweep to calibrate the new false-positive surface.

## Workflow gap

- **Bug observed:** a plan-text citation of a committed input under an include-tree prefix outside `eval_results|ood_eval_results|data` (e.g. `tests/fixtures/eval_results/a.json`) produces NO finding at all — channel-A `_PATH_RE` only matches those three prefixes, so both the clone-lane reachability check and the new #1915 rsync exclude subtraction are blind to it in plan text (the #1915 pin tests enter via `classify()` directly).
- **Why it is a workflow gap:** the gate's extraction scope silently under-covers the very path class #1915 made auditable; an untracked or nested-excluded tests/-prefixed input cited in a plan sails through the Step 6a.5 gate.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'eval_results|ood_eval_results|data' scripts/verify_carryover_inputs.py` → 1 hit: `_PATH_RE` at L62-64 matches exactly `(?:eval_results|ood_eval_results|data)/` — no `tests|scripts|configs` alternative present (2026-08-02 UTC). Landed-fix check: `git log --oneline --since='7 days ago' -- scripts/verify_carryover_inputs.py` → 2 commits (`ea98b57959` #1915 itself, `dcf37f9746` #1835), neither widening the prefix set. Open-sibling check: open task #1935 ("exclude plan-declared outputs from carry-over bare-name resolution") targets the same file but a DIFFERENT bug (bare-name resolution of outputs, not channel-A prefix coverage) — not a duplicate.

## Proposed change (candidate diff sketch — refine in planning)

```diff
- r"(?P<path>(?:eval_results|ood_eval_results|data)/"
+ r"(?P<path>(?:eval_results|ood_eval_results|data|tests|scripts|configs)/"
+ # + corpus sweep over persisted plans to calibrate new-prefix false positives
```

Keep the existing glob/dir/no-ext skip rungs; document the false-positive surface (plans routinely cite `scripts/*.py` as code references — mostly classify in-ref/planned-output; audit a corpus sweep before landing).

## Scope / surfaces

- Primary target: `scripts/verify_carryover_inputs.py`
- Run the corpus sweep over persisted plans named in the diff sketch before landing.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: scripts/verify_carryover_inputs.py
- fingerprint: c0ce6e284ef5 (tag-authoritative; supersedes body-carried fingerprint: 7ddb0181b608)
- origin: parked candidate on task #1915, ts 2026-08-02T00:04:16Z, routed by /daily 2026-08-01 Step C.

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_carryover_inputs.py
bug_observed: A plan-text citation of a committed input under an include-tree prefix outside eval_results|ood_eval_results|data (e.g. `tests/fixtures/eval_results/a.json`) produces NO finding at all — channel-A `_PATH_RE` only matches those three prefixes, so both the clone-lane reachability check and the new #1915 rsync exclude subtraction are blind to it in plan text (the #1915 pin tests enter via classify() directly).
why_workflow_gap: The gate's extraction scope silently under-covers the very path class #1915 made auditable; an untracked or nested-excluded tests/-prefixed input cited in a plan sails through the Step 6a.5 gate.
proposed_change: Widen channel-A `_PATH_RE` to the remaining include-tree prefixes (tests|scripts|configs), keeping the existing glob/dir/no-ext skip rungs; document the false-positive surface (plans routinely cite scripts/*.py as code references — mostly classify in-ref/planned-output, but audit a corpus sweep before landing).
confidence: medium
related_task: #1915
<!-- /workflow-fix-candidate -->
