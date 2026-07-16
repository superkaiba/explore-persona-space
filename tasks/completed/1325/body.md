---
title: 'daily-fix: push-verify must cover eval_results + figures'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bd2d4e03f34d
- daily-auto-filed
created_at: '2026-07-15T06:52:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): two upload-verifier round-1
  FAILs in one day (#928 00:08Z: run''s eval JSONs + figures never git-committed —
  the driver''s push-verify only covered code commits; #1090 03:08Z: post-rejudge
  judge-raw VM-local only) — the pod-side result-commit recipe''s push-verify does
  not assert result artifacts are in the pushed tree'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (sessions 72a512ff/#928 and 9d362ba4/#1090): both /issue sessions took an upload-verifier round-1 FAIL for artifacts the run driver should have persisted — each cost one extra verifier round. The gate worked; the driver-side recipe is where the gap is.

## Goal

extend the pod-side result-commit push-verify recipe to assert the round's eval_results/ + figures/ paths are present in the pushed tree (not only code commits), so upload-verification stops catching it one round late

## Workflow gap

- **Bug observed:** two upload-verifier round-1 FAILs in one day (#928 00:08Z: run's eval JSONs + figures never git-committed — the driver's push-verify only covered code commits; #1090 03:08Z: post-rejudge judge-raw VM-local only) — the pod-side result-commit recipe's push-verify does not assert result artifacts are in the pushed tree
- **Why it is a workflow gap:** `.claude/rules/pod-side-reporting.md`'s result-commit recipe (push-verify backstop, ~:259-:317) verifies the CODE push; nothing asserts the result artifacts (eval JSONs, figures) landed in the pushed tree, so the omission surfaces only at the Step 8 upload-verification gate.
- **Confidence:** medium-high (2 incidents, different sub-causes, same day)
- verified-at-filing: `grep -n "push-verify" .claude/rules/pod-side-reporting.md` -> hits at :269, :317 (code-commit scoped); `grep -c "eval_results" .claude/rules/pod-side-reporting.md` -> 3 hits, none in an assert-in-pushed-tree duty (presence of recipe + absence of artifact assertion) (2026-07-15).

## Proposed change

One recipe clause: after the result commit, verify `git ls-tree` (or the push-verify range) contains the round's declared `eval_results/issue_<N>/...` + `figures/issue_<N>/...` paths; missing => fail loud before the pod exits.

## Constraints

- Workflow-surface only; keep consistent with the upload-policy rule (HF is canonical for raw completions; git for eval JSONs/figures); recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/pod-side-reporting.md
- fingerprint: bd2d4e03f34d
