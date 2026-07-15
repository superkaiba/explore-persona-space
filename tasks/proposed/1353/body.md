---
title: 'workflow-fix: gotchas entry — BPE-seam position-parity gate tail (carve-out,
  not threshold-loosen)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6c22949edabb
created_at: '2026-07-15T16:33:43Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate:yes failure-lesson from #825 r10 (epm:failure-lesson
  v6)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #825 (emitting agent: experiment-implementer, round-10 crash-fix).

## Goal

Add a gotchas.md entry: BPE delimiter-seam merges under plain-text "Role: content" renders shift span-derived positions ±1 between two captures, breaking cross-capture position/activation-parity gates on a mechanically-identifiable row tail (median clean, row-fraction leg fails); chat-template renders are immune.

## Workflow gap

- **Bug observed:** #825 round-10's G2 activation-parity gate HALTed on pretrained (frac 0.9767 < 0.99) — 60/2572 rows where `context_pos` shifted ±1 vs the banked store at the `"Assistant: "` BPE seam; cosines 0.687–0.872 on exactly those rows.
- **Why it is a workflow gap:** gotchas.md carries the zero-width-span BPE-seam trap but not its position-PARITY-gate sibling; the next cross-capture parity gate over a plain-text render will re-hit it and a naive fix loosens the threshold (selection-on-outcome risk).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md § BPE zero-width spans entry gains a sibling paragraph: cross-capture
+ position-parity gates (G2-style) fail on the seam-shifted row tail; diagnose by
+ comparing position METADATA across captures (new pos vs banked token_start) BEFORE
+ touching thresholds; resolve with a mechanical position-keyed pair-safe carve-out
+ (fail-loud rate cap), never a cosine-keyed exclusion or fraction loosening.
+ Signature: instruct(chat-template)-clean / pretrained(plain-text)-tail asymmetry.
+ Worked example: scripts/issue825_onpolicy_turn_depth_fit.py::_boundary_pos_carveout.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`

## Constraints / invariants

- Workflow-surface only; workflow_lint passes.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 6c22949edabb
