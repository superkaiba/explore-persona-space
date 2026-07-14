---
title: 'workflow-fix: Shared HF tree-pagination helper in verify_tas'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9d8a23fcd8c3
- daily-auto-filed
created_at: '2026-07-09T06:59:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Checks 25 (keyword probe),
  30 (exhaustive counter), and 32 (basename walker) in scripts/verify_task_body.py
  each carry a near-identical bounded Link-header self-pagination loop over the Hub
  tree endpoint. [merged sibling: `_hf_tree_url` relies on `huggingface_hub.constants`
  being attribute-reachable after a bare `import huggingface_hub`, but the check-30
  walker (~l.7604) and check-25 keyword probe'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1016.

## Goal

Consolidate the three near-identical bounded HF tree-pagination loops in scripts/verify_task_body.py into one shared iterator, with no behavior change to any check.

## Workflow gap

- **Bug observed:** Checks 25 (keyword probe), 30 (exhaustive counter), and 32 (basename walker) in scripts/verify_task_body.py each carry a near-identical bounded Link-header self-pagination loop over the Hub tree endpoint.
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  + def _hf_tree_pages(url, *, max_pages, deadline_s, attempts): ...
  +     # yields per-page entry lists under the shared page/deadline/429 contract
  - (three per-check copies of the Link-header pagination loop)
  + checks 25 / 30 / 32 consume the shared iterator with their existing caps

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- Behavior-preserving refactor: existing verify_task_body tests must pass unchanged; each check keeps its own page/deadline caps.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- origin: parked candidate on task #1016 at 2026-07-05T03:50:52Z

Verbatim parked note:

```
routed: parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target (recursion guard, workflow-fix-on-bug.md § Recursion guard). source: prose-followup (planner, #1016 Phase 1). Candidate: extract a shared _hf_tree_pages bounded-pagination iterator in scripts/verify_task_body.py — check 25 keyword probe, check 30 counter, and the new check 32 basename walker are 3 near-identical bounded pagination loops; consolidate when a 4th appears. Deferred rationale: doing it now would non-additively touch freshly-merged check-25/30 paths with two concurrent same-file tasks (#1014/#1015) in flight.
```


### Merged sibling candidate (s4-vtb-check30-hf-import-order, from task:1016 at 2026-07-05T04:53:32Z)

- bug_observed: `_hf_tree_url` relies on `huggingface_hub.constants` being attribute-reachable after a bare `import huggingface_hub`, but the check-30 walker (~l.7604) and check-25 keyword probe (~l.7024) build the URL before `_hf_build_headers` imports `huggingface_hub.utils`, so a fresh-process probe can AttributeError (check 32 fixed its own ordering; check 30 was left untouched per a plan Must-ask).
- proposed_change: Make `_hf_tree_url` ordering-independent (import `huggingface_hub.utils` inside it, or `from huggingface_hub.constants import ENDPOINT`) so every call site is safe regardless of caller ordering, mirroring the check-32/check-23 safe-ordering fix; add a fresh-process regression test.
- origin note (verbatim): Bug pattern still present on main: scripts/verify_task_body.py `_hf_tree_url` (l.5314) does bare `import huggingface_hub` then `huggingface_hub.constants.ENDPOINT`; call sites at l.7024 (check-25 keyword probe) and l.7604 (check-30 count walker) build the URL BEFORE `_hf_build_headers()` imports `huggingface_hub.utils`, while l.7965 shows the check-32 walker was fixed with an explicit safe-ordering comment. No dedup: no task body names `_hf_tree_url` / `huggingface_hub.constants`. No retraction in #1016 events after ts.


### Merged sibling candidate (s5-hf-probe-headers-before-url, from task:1016 at 2026-07-05T05:05:14Z)

- bug_observed: _hf_probe_keyword (check 25) and _hf_count_files_under_prefix (check 30) call _hf_tree_url before _hf_build_headers, so a fresh-process direct call hits huggingface_hub.constants before the lazy submodule is attribute-reachable.
- proposed_change: Hoist the safe headers-before-URL ordering (already used by checks 23/32) into both trunk siblings, or extract a shared ordered helper.
- origin note (verbatim): NOT fixed on main: _hf_probe_keyword (check 25) builds the URL before headers at verify_task_body.py:7024-7026 and _hf_count_files_under_prefix (check 30) at :7604-7606, while check 32 (:7965-7971, added by completed #1016 commit 389201de49) carries the safe headers-before-URL ordering + explanatory comment. No open dedup (completed #1016 covered only check 32); no retraction events on #1016.
