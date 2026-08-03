---
title: 'workflow-fix: widen check-30 count-noun vocabulary (chunks/sidecars + modifier-separated)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:04e69aa91f07
created_at: '2026-07-31T13:06:46Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #1901: check 30 adjacency/vocabulary
  gap — footer claims ''3 activation chunks'' / ''3 chunk files plus 1 over-length-skip
  sidecar'' invisible to _COUNT_NOUN_RE (files?|shards? only)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1901 (emitting agent: clean-result-critic).

## Goal

Widen `verify_task_body.py` check 30's count-noun regex family to cover chunk/sidecar nouns and modifier-separated nouns so footer file-count claims phrased that way stop being invisible to the check.

## Workflow gap

- **Bug observed:** check 30 (`check_hf_file_count_claims`) reported "no file-count claims adjacent to HF tree links" on #1901's body while the footer carried TRUE count-claims in parentheticals following the pinned `issue1901_wildchat` `/tree/<sha>` link — the claims used the noun "chunks" ("3 activation chunks") and a modifier-separated noun ("3 chunk files plus 1 over-length-skip sidecar"), neither matched by `_COUNT_NOUN_RE`'s `files?|shards?` vocabulary or its adjacent count-noun shape.
- **Why it is a workflow gap:** the count-noun vocabulary/shape in `_COUNT_NOUN_RE` (and its siblings `_COUNT_PAREN_LINK_RE`, `_HF_LINKTEXT_THEN_PAREN_RE`) is narrower than the footer phrasings bodies routinely use, so count claims phrased with chunk/sidecar nouns or a modifier token between count and noun are unverifiable-by-silence — a FALSE claim in that phrasing would ship unflagged (the #1901 instance happened to be true; the critic verified it manually via `list_repo_tree`).
- **Confidence (emitter):** low
- verified-at-filing: `grep -n "_COUNT_NOUN" scripts/verify_task_body.py` + code read of lines 9439-9456 → vocabulary is exactly `files?|shards?` with no modifier allowance; #1901 body footer (tasks/followups_running/1901/body.md) carries "3 activation chunks" + "3 chunk files plus 1 over-length-skip sidecar" adjacent to the pinned HF tree link, and check 30's output on `--issue 1901` reads "no file-count claims adjacent to HF tree links" (2026-07-31). Per-target: scripts/verify_task_body.py 6 hits for `_COUNT_NOUN`.

## Proposed change (candidate diff sketch — refine in planning)

```
- r"\b(?P<count>\d{1,3}(?:,\d{3})+|\d{1,6})\s+(?P<noun>files?|shards?)\b"
+ r"\b(?P<count>\d{1,3}(?:,\d{3})+|\d{1,6})\s+(?:\w+[ \t]+)?(?P<noun>files?|shards?|chunks?|sidecars?)\b"
```
(applied consistently across the `_COUNT_NOUN_RE` regex family, with the check's documented precision guards — per-namespace lookahead, folder-inflation tolerance — reviewed for the widened vocabulary; the planner decides whether the single-modifier allowance needs a stopword guard to protect precision.)

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '_COUNT_NOUN' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. Update `tests/test_verify_task_body.py` pins for the widened shapes (a "3 activation chunks" positive + a modifier-separated positive + a precision-guard negative).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 04e69aa91f07

Surfaced prose (verbatim, clean-result-critic on #1901): "`scripts/verify_task_body.py`'s 'HF file-count claims match the Hub tree' check reported 'no file-count claims adjacent to HF tree links' while the footer carries exactly such claims in parentheticals following the pinned `issue1901_wildchat` tree link ('3 activation chunks', '3 chunk files plus 1 over-length-skip sidecar'). The claims are TRUE (I verified via `list_repo_tree`), so no body defect — but the check's adjacency window apparently misses count-claims phrased inside a parenthetical after the link, a phrasing footers use routinely. Concrete change: widen the check's claim-binding window to include the sentence/parenthetical span immediately following a pinned `/tree/<sha>` link. confidence: low" — orchestrator triage corrected the mechanism: the adjacency/parenthetical binding exists (#1005/#833 shapes); the actual gap is the count-NOUN vocabulary + modifier separation.
