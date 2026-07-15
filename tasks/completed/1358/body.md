---
title: 'workflow-fix: gotchas BPE-seam entry — plain-text boundaries + smoke composition'
kind: infra
tags:
- wf-fix
- wf-fix-fp:21c9cb462625
created_at: '2026-07-15T18:00:29Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1315 r7 (experiment-implementer):
  plain-text span boundaries BPE-merge on ~every question; span-rig smokes need a
  plain-text-boundary context'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1315 (emitting agent: experiment-implementer, crash-fix round 7).

## Goal

Extend the gotchas.md teacher-forced-capture BPE-seams entry: under prefix_end='last_user' a plain-text span boundary with a space-before-{q} wrap BPE-merges on essentially every question (not an edge case), and span-rig smokes must include at least one plain-text-boundary context (special-token-adjacent-only slices cannot catch the class).

## Workflow gap

- **Bug observed:** #1315 r7: neg_reph_curious ("I'm curious about the following: {q}") drifted 20/20 rows at the prefix boundary under prefix_end='last_user' while all 160 special-token-adjacent rows were exact; the smoke slice contained no plain-text-boundary context so the class first fired in production p8_capture.
- **Why it is a workflow gap:** gotchas.md carries the BPE-seams entry (the #1092 offset-mapping recipe) but is silent on (a) plain-text (non-special-token-adjacent) boundaries merging on ~every question rather than rarely, and (b) the smoke-composition duty — a span-rig smoke must include a plain-text-boundary context or the class is invisible until production.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix + 20/20-vs-160/160 CPU repro landed on issue-1315 @ 7075942917)
- verified-at-filing: `grep -ciE "BPE seam|bpe.seam|offset.mapping" .claude/rules/gotchas.md` → 1 hit (presence of the existing entry confirmed); `grep -ciE "plain.text.boundar|last_user|space before" .claude/rules/gotchas.md` → 0 hits (absence-of-nuance claim — the 0-hit result IS the evidence) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

In .claude/rules/gotchas.md, extend the existing teacher-forced capture / BPE-seams entry:
+ Plain-text boundaries are the WORST case, not an edge case: under
+ prefix_end='last_user' any "... {q}"-style wrap with a space before the
+ query BPE-merges the boundary into the query's first word on essentially
+ every question (#1315 r7: 20/20 rows for one panel member vs 160/160 exact
+ special-token-adjacent rows). Derive both boundaries from the full render's
+ offset mapping (exclude a prefix-boundary straddler, include a
+ context-boundary straddler) with per-row seam provenance; and a span-rig
+ SMOKE must include at least one plain-text-boundary context — a
+ special-token-adjacent-only slice cannot catch the class.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep before editing (`grep -rln 'BPE' .claude/rules/ .claude/agents/`) and keep consistent with the existing #1092 recipe text; list hits in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 21c9cb462625

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p8_capture (compute_prompt_spans span validation)
lesson: Under prefix_end='last_user' a span boundary can sit on PLAIN TEXT inside the user turn, and any "... {q}"-style wrap with a space before the query BPE-merges that space into the query's first word on essentially every question — re-tokenizing text[:boundary] then asserting token-prefix identity is guaranteed to fail there. Derive both boundaries from the FULL render's offset mapping (exclude a prefix-boundary straddler, include a context-boundary straddler) with per-row seam provenance; keep the full-sequence token-identity assert for genuine render/tokenizer drift. Smokes for span rigs must include at least one plain-text-boundary context (a special-token-adjacent-only slice cannot catch this class).
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
