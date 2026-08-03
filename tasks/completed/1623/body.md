---
title: 'daily-fix: similarity stats declare direction-aware or not'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5b91adaf687c
- daily-auto-filed
created_at: '2026-07-23T07:00:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): a direction-blind Procrustes
  spectrum-cosine was narrated in chat as operator identity (#1310); no disclosure
  clause exists for similarity statistics in ad-hoc summaries'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). In the #1310 team-lead session an ad-hoc chat summary presented a Procrustes SPECTRUM-cosine of 0.98–1.00 as "operators nearly the same object up to rotation" — a direction-blind statistic narrated as operator identity. The user caught it ("are we sure this is a fair operation"), the assistant conceded the overclaim, and a full principled round-2 battery (+shuffle-fit null) had to be run (~1.5 h). The project's own precedent battery (#1345 operator-comparison conventions) was consulted only after the user asked "what did we do for other similar analyses".

## Goal

Extend the project CLAUDE.md "Ad-hoc results summaries" bullet with one clause: any operator/map SIMILARITY statistic quoted in chat or an ad-hoc artifact states whether it is direction-aware or spectrum/rotation-invariant-only, and names the project precedent battery it matches (or states none exists).

## Workflow gap

- **Bug observed:** b0495190 (#1310), 18:41:50Z user challenge → 18:44:07Z concession ("it's an overclaim ... the 'aligned cosine' is spectrum_cosine ... the cosine between the sorted singular-value vectors"); 18:47:42Z user had to prompt the precedent check (`scripts/issue1345_operator_comparison.py`).
- **Why it is a workflow gap:** the existing bullet mandates per-arm provenance + matched-target disclosure for ad-hoc summaries but says nothing about similarity-statistic semantics; direction-blind statistics narrated as identity are a recurring overclaim class the plan-time critics never see on chat/inline paths.
- **Confidence:** high.
- verified-at-filing: `grep -c 'direction-aware\|spectrum-only' CLAUDE.md` → 0 hits (absence claim — no such disclosure clause exists in the ad-hoc summaries bullet), 2026-07-23 UTC.

## Proposed change (refine in planning)

One sentence appended to the CLAUDE.md ad-hoc results summaries bullet (pattern-matching its existing matched-target clause): similarity/alignment statistics disclose direction-aware vs spectrum-only and cite the precedent battery.

## Scope / surfaces

- Primary target: `CLAUDE.md` (ad-hoc results summaries bullet).

## Constraints / invariants

- Keep the bullet's register; recursion guard applies. Note: a sibling /daily filing extends the SAME bullet with the bare-URL clause — coordinate/rebase (distinct bugs, both filed 2026-07-22).

## Provenance

- fingerprint: 5b91adaf687c

- workflow_fix_target: CLAUDE.md
