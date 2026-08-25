---
title: Extend the figure-text opaque-code detector to digit-prefixed rung/condition
  slugs in figure sidecars
kind: infra
tags: []
created_at: '2026-08-24T15:39:17Z'
has_clean_result: false
parent_id: 2479
origin_prompt: 'Surfaced by clean-result-critic + codex twin at #2479 clean-result
  gate round 2: verify_task_body.py''s figure-text opaque-code check passed a body
  whose embedded figure ladder_curves.png carried raw config-slug tick labels (1_direct
  .. 9_full_AMB, AMB unglossed).'
workflow: v1
---
---
kind: infra
---

# Extend the figure-text opaque-code detector to digit-prefixed rung/condition slugs

## Provenance

Surfaced by the `clean-result-critic` at #2479 clean-result gate round 2 (`epm:clean-result-critique` v2, 2026-08-24T15:36:57Z), and independently corroborated by the `codex-clean-result-critic` twin the same round (`epm:clean-result-critique-codex` v2, finding `opaque-ladder-figure-labels`).

workflow_fix_target: scripts/verify_task_body.py (figure-text opaque-code check)

## Problem

`verify_task_body.py`'s figure-text opaque-code check did not flag digit-prefixed condition/rung slugs rendered as axis tick labels inside a committed figure. At #2479 the figure `figures/issue_2479/ladder_curves.png` carried the tick labels `1_direct`, `2_ctx_offset`, `3_ans_offset`, `4_bias_refit`, `5_global_scale`, `6_rotation`, `7_ctx_reparam`, `8_ans_reparam`, `9_full_AMB`, with `AMB` glossed nowhere in the body. The mechanical pre-pass returned `OVERALL: PASS` on that body; both LM reviewers caught it by eye.

The standing rule is that opaque condition codes never reach reader-facing surfaces, figures included, and the project already has the plain-English form in the sibling figure `rungwise_ordering.png` (relabelled at commit f23b6d8bdb).

Two properties made this class invisible to the existing detector:

1. The label is a `<digit>_<slug>` compound, so patterns anchored on a leading letter or on a bare-slug shape miss it.
2. The string lives in the figure's rendered tick labels and its `meta.json` sidecar, not in body prose, so a body-prose-only scan cannot see it. The sidecar IS greppable: `git show <pin>:figures/issue_2479/ladder_curves.meta.json` contains every offending token.

## Why it matters

The gap is latent until a figure moves from footer-only provenance into an embedded reader-facing position, which is exactly what happened here: closing one lens gap (embedding the per-unit companion) opened this one, and only the LM lenses caught it. A mechanical detector would have caught it at draft time in both positions.

## Scope

1. Extend the figure-text opaque-code check to the `<digit>_<slug>` compound form.
2. Have the check read the committed figure `meta.json` sidecars for embedded figures, not body prose alone, so tick-label text is in scope.
3. Fixture-test both shapes against the #2479 artifacts: `ladder_curves.meta.json` at pin `fd4e118f3aa7feccb1ab994e829f5bb46069469e` must FAIL, and `rungwise_ordering.meta.json` at the same pin must PASS.

## Acceptance criteria

- A body embedding a figure whose sidecar carries digit-prefixed condition slugs FAILs the check with the offending tokens named.
- The plain-English sibling figure still PASSes (no false positive).
- Regression tests cover both fixtures.
- No change to the existing body-prose opaque-code behavior.
