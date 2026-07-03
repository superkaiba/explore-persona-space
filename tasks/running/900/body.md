---
title: 'workflow-fix: verify_task_body check for bare #K refs in Methodology/Results'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7a967742b4bb
created_at: '2026-07-02T23:01:31Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #841 round-2 clean-result-critic: bare #779 in
  Methodology prose survived two ensemble rounds (round-1 PASS rested on a mistaken
  exemplar claim); add a mechanical verify_task_body.py check FAILing bare #<digits>
  in Methodology/Results spans outside links/table-rows/footer/frontmatter.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #841 (emitting agent: clean-result-critic, round 2).

## Goal

Add a `verify_task_body.py` check that FAILs bare `#<digits>` issue references inside the `## Methodology` / `## Results` section spans of a v4 body when not inside a markdown link target, not in a table row, and not in the footer/frontmatter.

## Workflow gap

- **Bug observed:** bare `#779` references (~8×) in #841's `## Methodology` prose survived TWO clean-result-critique ensemble rounds: in round 1 BOTH reviewers PASSed Lens 2 — the Claude critic on a mistaken claim that the canonical v4-657 exemplar carries prior-issue refs in its Results (it does not; verified against the exemplar in round 2), the Codex critic flagged it but the finding was waived on the Claude critic's mistaken recalibration. Only a round-2 re-check against the exemplar caught it.
- **Why it is a workflow gap:** the Lens-2 standalone-sections rule (`## Takeaways` / `## Methodology` / `## Results` carry no bare `#K`; prior-issue links live only in the Goal context slot + footer) is spec text with NO mechanical enforcement in `verify_task_body.py` — an LM-judgment-only rule that two independent reviewers can (and did) mis-apply in the same direction. The critic explicitly tagged the finding `mechanizable: yes`.
- **Confidence (emitter):** high (live two-round miss on #841; the exemplar check that settles it is deterministic).

## Proposed change (candidate diff sketch — refine in planning)

- New check in `scripts/verify_task_body.py` (v4 bodies only): scan the `## Methodology` and `## Results` section spans (and `## Takeaways`) for `#\d+` tokens, excluding:
  + matches inside markdown link syntax `[...](...)` (both label and target),
  + lines that are table rows (`| ... |` — the Training-table Source column is a sanctioned grounding convention),
  + the `**Repro:**` / `**Context:**` footer block and YAML frontmatter,
  + HTML comments.
- FAIL (or WARN for grandfathered bodies) with the offending line numbers; v3/v2/legacy bodies exempt (forward-only rule).
- Regression tests: the #841 round-2 shape (bare `#779` in Methodology prose → FAIL; the same reference in a Source-column table row → PASS; `[#779](url)` in the Goal slot → PASS).
- Mirror one line into the SPEC.md mechanical-checks list so the lens and the verifier stay in sync.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Also update: `.claude/skills/clean-results/SPEC.md` (mechanical-checks list), `tests/test_verify_task_body.py`.
- Grep before editing: `grep -n "check_" scripts/verify_task_body.py` for numbering conventions; verify the v4-657 exemplar and a sample of recent v4 bodies pass the new check (no false positives on sanctioned forms).

## Constraints / invariants

- Workflow-surface only. Forward-only: never newly hard-FAIL a grandfathered v3/v2/legacy body.
- Existing verifier tests stay green; SPEC.md is the source of truth — update it alongside per the standing rule.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 7a967742b4bb

Surfaced prose (verbatim, from the #841 round-2 clean-result-critic report): "mechanizable: yes — a verify_task_body.py check could FAIL on `#\d+` inside the Methodology/Results section spans when not inside a `](...)` link, not in a `| … |` table row, and not in the footer/frontmatter. I did NOT emit a formal workflow-fix-candidate block (running under AUTO_REVIEW_DISABLED=1), but flagging it for you: this is the exact gap that let round 1 miss it."
