---
title: Does the mission-control spec form write a canonical spec-v2 task body end
  to end?
kind: experiment
tags:
- spec-v2
created_at: '2026-08-17T11:09:50Z'
has_clean_result: false
workflow: v1
goal: 'Mission-control spec-form live smoke: verify the create/edit/submit surface
  writes a canonical spec-v2 body through task.py.'
---
# Does the mission-control spec form write a canonical spec-v2 task body end to end?

<!-- spec-v2 -->

## Goal

Mission-control spec-form live smoke: verify the create/edit/submit surface writes a canonical spec-v2 body through task.py.

Scratch task — archived immediately after the check.

## Competing hypotheses

1. H1: the composed body lands canonically (sentinel + 8 H2s + goal frontmatter)
2. H2: some task.py seam (goal injection / set-body strip) mangles it

## Outcome branches

| Outcome | Conclusion | Next action |
|---|---|---|
| body.md matches the canonical composition | spec form ships | archive the scratch task and deliver |
| null / ambiguous (partial mismatch) | a task.py seam interferes | fix the composer and re-run the smoke |

## Riskiest assumption — cheapest test

Riskiest assumption: task.py new --goal injection is byte-idempotent against our composed ## Goal paragraph. Cheapest test: this scratch create.

## Step costs and failure probabilities

| # | Step | Cost | P(failure) |
|---|---|---|---|
| 1 | create scratch task via POST /api/tasks | 0.1 h | 0.3 |
| 2 | push an edited spec via PUT /spec | 0.1 h | 0.2 |

_Suggested order by expected information per unit cost (entropy of P(failure) ÷ cost — most-uncertain-per-dollar first): 1 → 2._

## Kill criteria

Kill if task.py refuses the composed body or the read-verify-after-write mismatches twice. UPDATED-VIA-PUT-SPEC.

## Confound pre-mortem

Concurrent fleet commits racing the scratch task's registry commit could make HEAD reads lag; the working-tree fallback covers it.

## Primary question

Does the mission-control spec form write a canonical spec-v2 task body end to end?
