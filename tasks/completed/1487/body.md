---
title: 'workflow-fix: verify_task_body Check 40 misses subpath (N files) counts under
  unlinked backtick prefixes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8c0772e9d505
created_at: '2026-07-18T00:41:43Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #1345 (epm:clean-result-critique
  v5, 2026-07-18): Check 40 regex gap — parenthetical count forms on backtick subpaths
  under backticked-but-unlinked prefixes not matched'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the clean-result-critic on task #1345 (emitting agent: clean-result-critic, `epm:clean-result-critique v5`, 2026-07-18).

## Goal

Extend `verify_task_body.py` Check 40 (backtick HF-path count claims carry an adjacent pinned /tree link) so parenthetical `(N files)` / `(N)` count forms attached to backtick SUBPATHS under a backticked-but-unlinked parent prefix are matched and require a pinned tree link in the same sentence/clause.

## Workflow gap

- **Bug observed:** on #1345's live body, Check 40 reported "no unpinned backtick HF-path count claims" while the footer carried exactly that shape — `` `analysis_tensors/turnstore` (10 files) `` under the unlinked backtick prefix `` `issue1345_framing/story_slot_ablation/` `` (no pinned /tree link anywhere in the clause). The clean-result-critic caught it manually (procedural fix applied inline by the orchestrator); the mechanical check silently missed it.
- **Why it is a workflow gap:** Check 40 exists to make link-liveness/pinning mechanical (Lens 10); a regex that misses the common multi-round footer shape (per-round prefix sentence + subpath counts) leaves the check green on precisely the bodies it was added for (#1433 lineage).
- **Confidence (emitter):** high (reproduced live on #1345's body this round).
- verified-at-filing: `grep -n "unpinned backtick HF-path\|backtick HF-path count" scripts/verify_task_body.py` → 3 hits in 1 file (check header L9522, docstring L9683, check name L9730) (2026-07-18); the missed body shape existed live at review time (patched inline minutes later — the pre-patch shape is quoted in `epm:clean-result-critique v5`).

## Proposed change (candidate diff sketch — refine in planning)

```
  # Check 40 pattern: currently matches `path` (N files) only when the
  # backtick path itself looks like an HF prefix (issue*/ anchored?).
+ # Also match: a backtick SUBPATH (e.g. `analysis_tensors/turnstore`)
+ # followed by a parenthetical count, when the enclosing sentence/clause
+ # names a backticked-but-unlinked HF prefix (issue*_*/ shape) and no
+ # huggingface.co/.../tree/<sha> link occurs in the same sentence/clause.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py` (Check 40, ~L9522-9730)
- Add a regression fixture in `tests/test_verify_task_body.py` reproducing the #1345 footer shape (pre-patch text quoted in `epm:clean-result-critique v5`); keep the check WARN-severity.

## Constraints / invariants

- Workflow-surface only; ruff clean; `scripts/workflow_lint.py` default run not regressed (note: main currently carries the pre-existing LESSONS ratchet red tracked as #1479 — baseline it, do not fix here).
- This session runs under the recursion guard once spawned (`workflow_fix_target:` Provenance line below).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 8c0772e9d505

Surfaced prose (verbatim, from the clean-result-critic's return): "scripts/verify_task_body.py's 'backtick HF-path count claims carry an adjacent pinned /tree link' check reported 'no unpinned backtick HF-path count claims' while the body carries exactly that shape (`analysis_tensors/turnstore` (10 files) under an unlinked backtick prefix issue1345_framing/story_slot_ablation/) — the pattern misses parenthetical (N files) counts attached to backtick subpaths whose parent prefix is backticked-but-unlinked. Mechanizable: yes — extend the check's regex to match `path` (N files) / (N) count forms and require a pinned tree link in the same sentence/clause. Concrete, likely to recur on multi-round footers."
