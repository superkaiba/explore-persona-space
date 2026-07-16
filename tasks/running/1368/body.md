---
title: 'workflow-fix: verify_task_body WARN on 4+-sentence result paragraphs (v4)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3cd16468deda
created_at: '2026-07-15T23:22:58Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate v1 from #1333 clean-result-critic r1: v4 1-3-sentence
  paragraph cap has no mechanical backstop; add WARN check to verify_task_body.py'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1333 (emitting agent: clean-result-critic, round 1).

## Goal

Add a WARN-level v4 check to verify_task_body.py counting sentences per interpretation paragraph in `### <result>` prose (cap 1-3 per the v4 spec register), so the #385 incident class stops burning LM critic rounds.

## Workflow gap

- **Bug observed:** four of six results in #1333's body carried single >=4-sentence interpretation paragraphs; the v4 conciseness check (check 20) counts words only, so the spec's 1-3-sentence-per-analytical-paragraph rule was caught only by the clean-result-critic (a full REVISE round for paragraph splits).
- **Why it is a workflow gap:** the sentence-per-paragraph cap is spec text (clean-result-critic-lens-reference.md Lens 12) with no mechanical backstop in verify_task_body.py, so it recurs on dense bodies.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "sentence" scripts/verify_task_body.py | grep -iE "per.paragraph|paragraph|cap"` -> 0 hits (23 'sentence' mentions exist, none implement a per-paragraph sentence cap) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

+ def check_v4_paragraph_sentence_cap(body):  # WARN-only
+     for result in iter_v4_results(body):
+         for para in interpretation_paragraphs(result):  # prose after the blockquote caption
+             n = count_sentences(para)  # split on [.!?] guarding decimals, parens, "e.g."/"vs."
+             if n >= 4: warn(f"result '{result.head[:40]}' has a {n}-sentence paragraph (v4 cap 1-3)")

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py` (WARN-only; forward-only for v4 bodies — grandfathered v3/v2 never newly flagged)
- Keep `.claude/skills/clean-results/SPEC.md` + `clean-result-critic-lens-reference.md` Lens 12 consistent (the spec text is the source; the check is the backstop).

## Constraints / invariants

- WARN-level only, never a FAIL (register judgment stays with the critic).
- tests/test_verify_task_body.py pins the new check (a 4-sentence paragraph fixture WARNs; a 3-sentence one does not; decimals/e.g./vs. do not split).
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 (recursion guard applies).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 3cd16468deda

(Verbatim candidate block preserved in origin_prompt.)
