---
title: 'daily-fix: interim-writeup conventions (plots, gloss)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f89713b4550e
- daily-auto-filed
created_at: '2026-07-24T06:49:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): repeated same-day user
  corrections on interim writeups: plots not tables, novel results only, plain-English
  gloss, consistent figure color semantics'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Recurring same-day user re-steering on interim/chat results writeups, across ≥4 sessions: "plots NOT tables", "only NOVEL results", "don't repeat information", conciseness pushes (#1092 sessions ×2); figure color-semantics confusion ("green is base model and blue is instruct model ???" — two figures reused blue/green with different meanings); ≥5 "what does this even mean" decode requests on compressed claims; one challenge falsified an over-compressed "uniform 0.6 shrinkage" claim. The clean-result SPEC covers promoted bodies; the interim/chat writeup surface has no register/figure conventions, so the same corrections repeat.

## Goal

Write the interim-writeup conventions into the workflow surface: extend the CLAUDE.md "Ad-hoc results summaries" bullet (or a small `.claude/rules/` file it points to) with the user's standing presentation rules, and add a same-color-same-meaning rule to the `/paper-plots` skill.

## Workflow gap

- **Bug observed:** repeated identical user corrections on interim writeups (plots-not-tables; novel-results-only; no repetition; plain-English gloss per compressed claim; consistent color semantics across figures in one writeup) — each correction exists only as chat history, so every new session re-learns it.
- **Why it is a workflow gap:** "repeated explanations that should live in a workflow file" is the textbook trigger; the ad-hoc-summaries bullet currently covers provenance/verification but not presentation register.
- **Confidence:** high (multiple same-day incidents, consistent direction)
- verified-at-filing: n/a — behavioral/prose gap; the incidents are quoted from the 2026-07-23 transcript sweep (sessions 5e8b4c66, f4b1d707, 12462773-class #1092 writeup sessions). CLAUDE.md's "Ad-hoc results summaries" bullet exists and carries no presentation-register clauses (read at compose time).

## Proposed change (refine in planning)

Add to the ad-hoc-summaries surface: (1) figures over tables for any comparison the user will read; (2) interim writeups lead with NOVEL results only (restatements labeled as context, never as results); (3) every technique term gets a plain-English gloss (the explain-intuitively convention); (4) one color = one meaning across all figures in a writeup (paper-plots rule). Keep it short — a 4-bullet convention block.

## Scope / surfaces

- Primary target: `CLAUDE.md` (Ad-hoc results summaries bullet) + `.claude/skills/paper-plots/SKILL.md` (color-semantics rule)

## Constraints / invariants

- Prose conventions only, no new gates; recursion guard applies.

## Provenance

- fingerprint: f89713b4550e

- workflow_fix_target: CLAUDE.md
