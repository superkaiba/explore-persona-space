---
title: 'workflow-fix: gotchas.md zero-width-span BPE-delimiter-merge entry'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3cb2afee4550
created_at: '2026-07-03T12:05:49Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #825 crash-fix round (zero-width
  u2 span; verbatim lesson in epm:failure-lesson v1 on #825)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson on task #825 (emitting agent: experiment-implementer, crash-fix round).

## Goal

Add a gotchas.md entry: char-range→token-span alignment over plain-text delimiters yields zero-width spans when generated text BPE-merges into the delimiter; span-validate at generation time with the consumer's asserts.

## Workflow gap

- **Bug observed:** Three #825 production GCP attempts died at ~21 min each on a zero-width u2 span assert (`span u2=(201,201) invalid`) because a bare-punctuation sampled user turn merged into the naturalistic `\n\n` delimiter (Qwen fuses ` .\n\n` into ONE token; even `Thanks.` loses both ends to boundary straddlers).
- **Why it is a workflow gap:** the trap generalizes to ANY plain-text (non-special-token-delimited) render whose spans come from offset containment — naturalistic formats, few-shot plain-text prompts, User:/Assistant:-style rigs over T>0 sampled text — and gotchas.md is the canonical home for codebase traps that recur across issues. Chat-template formats are immune (special tokens never BPE-merge).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

Add a bullet to .claude/rules/gotchas.md (alignment/extraction section):
"Zero-width token spans from BPE-delimiter merge: offset-containment span alignment over plain-text delimiters produces s==e spans when the segment's whole text merges into the delimiter token (Qwen: ' .\n\n' is one token). Span-validate generated text at GEN time with the consumer's exact asserts; substitute a validated multi-token placeholder — single-token placeholders are themselves degenerate. (#825)"

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Cross-check the lesson already persisted at `.claude/agent-memory/experiment-implementer/feedback_zero_width_span_bpe_delimiter_merge.md` (do not duplicate; the gotcha entry is the cross-agent surface).

## Constraints / invariants

- Workflow-surface only; ruff n/a; workflow_lint passes.
- Recursion guard applies (EPM_WORKFLOW_FIX_SESSION=1 + workflow_fix_target Provenance line).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 3cb2afee4550
