---
title: 'workflow-fix: verify unpinned backtick HF-path/file-count claims'
kind: infra
tags:
- wf-fix
- wf-fix-fp:89b455ca98d3
created_at: '2026-07-16T18:44:43Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #1345: extend verify_task_body.py
  checks 30/32 to cover backtick HF-path/file-count claims lacking an adjacent pinned
  /tree/<sha> link'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1345 (emitting agent: clean-result-critic, surfaced-prose
follow-up in its round-1 9a-bis verdict, 2026-07-16T18:40:37Z).

## Goal

Extend verify_task_body.py's HF-claim checks to flag backtick HF-path/file-count claims with no adjacent pinned /tree/<sha> link (resolve against main + WARN on missing pin).

## Workflow gap

- **Bug observed:** a footer HF claim `` `rejudge/` (2 files) `` with no adjacent pinned tree link escaped the HF-claim existence/file-count verification entirely on #1345 (the claim's `debcdda045` parenthetical was scoped to a different prefix; the two files were real, but nothing verified that).
- **Why it is a workflow gap:** checks 30 (`check_hf_file_count_claims`) and 32 (`check_hf_adjacent_file_claims`) both fire ONLY for claims adjacent to a hex-pinned HF `/tree/<sha>` markdown link, so any backtick HF-dir + file-count claim written without a pin is silently unverified — the exact class the checks exist to catch.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "tree/\|file-count\|files)" scripts/verify_task_body.py` → 11 hits in 1 file; per-target: scripts/verify_task_body.py lines 546-547 ("claims adjacent to hex-pinned HF `/tree/<sha>` markdown links") + 605-606 (same adjacency condition for backtick FILENAME claims) confirm the adjacency-only trigger (2026-07-16).

## Proposed change (candidate diff sketch — refine in planning)

```
+ In check 30/32 (or a new check 33): additionally scan footer/Repro prose for
+ backtick HF-path tokens (`<prefix>/` style) followed by a parenthesized
+ file-count noun ("(N files)") that have NO preceding hex-pinned /tree/<sha>
+ link in the same bullet/paragraph; for each, resolve the path against the
+ HF data repo at main (list_repo_tree scoped) and WARN on missing pin
+ (existence/count mismatch escalates per the existing check-30 semantics).
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'check_hf_file_count_claims\|check_hf_adjacent_file_claims' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- New behavior is WARN-only (no new hard FAIL on grandfathered bodies); network-free where possible (resolve via huggingface_hub only when a token is present, else WARN-missing-pin only).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 89b455ca98d3

Verbatim surfaced prose (clean-result-critic, #1345 round-1 9a-bis verdict): "Follow-up (orchestrator should consider): `scripts/verify_task_body.py`'s HF-claim checks (\"HF-adjacent backtick file claims exist under the pinned tree\" / file-count check) only fire for claims adjacent to a pinned tree link — a backtick dir + file-count claim like `rejudge/` (2 files) with no adjacent pin escapes verification entirely. Concrete change: extend the check to flag (or resolve against main + WARN for a missing pin) backtick HF-path/file-count claims in the footer that have no adjacent /tree/<sha> link."
