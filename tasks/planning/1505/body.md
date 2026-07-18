---
title: 'workflow-fix: bind parenthetical HF file-count claims after pinned /tree links'
kind: infra
tags:
- wf-fix
- wf-fix-fp:df849dd9d232
created_at: '2026-07-18T06:51:58Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1072 round-1 workflow-fix-candidate: HF file-count
  claim ''44 files'' after pinned /tree link unbound by the adjacency window (true
  count 40)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1072 (emitting agent: clean-result-critic).

## Goal

Widen verify_task_body.py's HF file-count-claim binding so `<N> files` inside the parenthetical/sentence following a pinned HF /tree/<sha> markdown link is extracted and recomputed against the Hub listing.

## Workflow gap

- **Bug observed:** The "HF file-count claims match the Hub tree" check reported "no file-count claims adjacent to HF tree links" on #1072 while the footer asserted "44 files" a few tokens after a pinned /tree/<sha> link; the true count at the pin is 40 — the adjacency window never bound the claim.
- **Why it is a workflow gap:** A mechanical check that exists precisely for pinned file-count claims silently skipped one sitting inside the parenthetical that follows the link, so a false reuse premise passed the verifier.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "files" scripts/verify_task_body.py | grep -inE "count|claim|tree"` → 10+ hits; the check's claim-binding block sits at scripts/verify_task_body.py:547-618 ("claims adjacent to hex-pinned HF /tree/<sha> markdown links", "Two anchored shapes only") — presence confirmed per-target (2026-07-18). Live repro: #1072 footer "— 44 files, listing verified live at write time" inside the parenthetical after the pinned /tree/9c4258b2… link; Hub listing at that pin = 40 files; verifier run at draft time reported no adjacent claims.

## Proposed change (candidate diff sketch — refine in planning)

in the check-"HF file-count claims match the Hub tree" claim extractor:
- bind only `<N> files` within K chars before/after the link text
+ also scan the parenthetical span that opens immediately after the
+ closing `)` of the pinned /tree link, up to its matching `)`,
+ for `(\d[\d,]*)\s+files` and verify each count via list_repo_tree

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: df849dd9d232

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: The "HF file-count claims match the Hub tree" check reported "no file-count claims adjacent to HF tree links" on #1072 while the footer asserted "44 files" a few tokens after a pinned /tree/<sha> link; the true count at the pin is 40 — the adjacency window never bound the claim.
why_workflow_gap: A mechanical check that exists precisely for pinned file-count claims silently skipped one sitting inside the parenthetical that follows the link, so a false reuse premise passed the verifier.
proposed_change: Widen the file-count-claim binding so `<N> files` (and `— <N> files`) inside the same parenthetical/sentence following a pinned HF /tree/<sha> markdown link is extracted and recomputed against the Hub listing.
confidence: medium
related_task: #1072
<!-- /workflow-fix-candidate -->
