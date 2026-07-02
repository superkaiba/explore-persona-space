---
title: 'workflow-fix: bound verify_task_body committed-at-sha regex to a single clause'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c1afb28fee29
created_at: '2026-07-02T21:49:51Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #841 analyzer promotion report: the committed-at-sha
  check spans clause boundaries and validated eval paths against the figures SHA (false
  FAIL, footer reworded to dodge)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #841 (emitting agent: analyzer, Step 9a promotion pass).

## Goal

Bound `verify_task_body.py`'s committed-at-sha regex span to a single clause so it cannot pair a "committed" token with a later, unrelated "at commit `<sha>`" in the same one-line paragraph.

## Workflow gap

- **Bug observed:** on #841's clean-result footer, the "committed-at-sha claims resolve" check spanned from the word "committed" (in the results-JSON clause) across an intervening clause boundary to the figures clause's "at commit `4824a567aa`", then validated the eval_results paths against the WRONG SHA — a false FAIL on an accurate footer. The analyzer had to reword the footer to avoid the "committed…at commit" trigger.
- **Why it is a workflow gap:** `verify_task_body.py` is the mechanical clean-result gate; a cross-clause regex span produces false FAILs on accurate provenance footers and trains authors to phrase around the checker instead of stating SHAs plainly.
- **Confidence (emitter):** medium-high (live false-positive on #841, reproduced twice before the reword).

## Proposed change (candidate diff sketch — refine in planning)

- In the committed-at-sha check, restrict the `committed[^\n]*?at commit` match window to a single clause: stop the span at the nearest sentence/clause delimiter (`;`, `·`, `.`, or an intervening markdown link boundary) instead of scanning to the end of the one-line paragraph, OR anchor path-to-SHA pairing on per-clause tokenization.
- Add a regression test with the #841 footer shape: two clauses on one line — "results … committed on branch X (`<shaA>`) … ; figures … at commit `<shaB>`" — asserting the eval paths are validated against shaA (or the check no-ops), never shaB.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -n "at commit" scripts/verify_task_body.py`) and cover sibling spans in the same check; list them in the plan.

## Constraints / invariants

- Workflow-surface only. Existing verifier tests must stay green; add the new regression test.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: c1afb28fee29

Surfaced prose (verbatim, from the #841 analyzer's promotion report): "adding the results-SHA clause tripped verify_task_body's 'committed-at-sha claims resolve' check — its regex spanned from my word 'committed' to the downstream 'at commit `4824a567aa`' (figures) and validated the eval_results paths against the wrong SHA. I reworded the footer to state all SHAs without the 'committed…at commit' trigger; both gates then PASS. (Possible mechanizable fix for that check: bound the `committed[^\n]*?at commit` span to a single clause rather than the whole one-line paragraph …)"
