---
title: 'daily-held: inline-round scientific discipline gaps'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-23T07:05:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 3): three same-day incidents:
  wrong metric committed as a null (#1092 raw-L2 vs #658 whitened), correlation headline
  without a null (#779), restatement-as-novel + unqualified base-arm generalization
  still live in the writeup'
workflow: v1
---
## Overview / Motivation

Filed by /daily 2026-07-22 as a TRACKED needs-human item (route 3 — scientific-meaning carve-out). Three same-day incidents show the inline/ad-hoc analysis paths (which skip the planner+critic stack) shipping scientifically wrong or under-qualified claims that only USER challenges caught. What discipline to mandate on these paths is Thomas's call — each candidate duty changes how results are computed/interpreted.

## The three incidents (2026-07-22)

1. **Wrong metric — committed null flipped to ρ≈+0.9 next round.** #1092 inline round 1 (3461ae99, 19:10–20:18Z) defined its own raw-L2 spread observable instead of reusing the established #658 WHITENED-spread recipe; the round-1 "clean null-to-inverted" result was committed (`61be8fa96a`) and lived on #1092 for ~2.5 h until the user's "is there anything else we should run here" prompted round 2, which ran the #658 recipe and got ρ = +0.93/+0.89 — "raw L2 was measuring the wrong thing entirely."
2. **Correlation headline folded into a writeup with NO null.** #779 prefix-twin Spearman (+0.92/+0.70/+0.83) went into the writeup TLDR before any null existed (fdf687f2, 23:19Z); the user's "but did we even take pre-image? or is this just random" forced the 1000-draw random-direction null overnight — which MATERIALLY qualified the claim (evil p=0.015, hallucination p=0.021, sycophancy marginal).
3. **Restatement presented as novel + an unqualified generalization.** The #1092 consolidated writeup's "Result 1" restated banked #722 results as a new finding (user: "Don't we already have this result somewhere"); and its "not the behaviorally relevant part" conclusion is contradicted on the BASE-model arm by its own results file (hallucination monitoring collapses at prefix-end: averaged r=0.62 vs prefix-end r=0.05) — CHECKED AT FILING: `docs/results_summaries/2026-07-22-direct-vs-averaged-prefix-map.md` lines 12/54 still carry the claim WITHOUT the base-arm qualification.

## The decision needed (why route 3)

Candidate duties, each a scientific-process mandate on the inline path:
- (a) an inline round correlating against a construct an earlier issue operationalized NAMES the prior issue's recipe and reuses it or states the deviation (the metric-reuse duty);
- (b) a rank/correlation headline folded into any writeup names its null or states "no null — descriptive only";
- (c) a results-summary "Result N" is cross-checked against banked clean-results + existing writeups and restatements are labeled background (the novelty duty);
- (d) IMMEDIATE concrete item: someone should add the base-arm qualification to `2026-07-22-direct-vs-averaged-prefix-map.md` (a scientific edit /daily won't make itself).

How strict to make (a)–(c) — and whether they go in the CLAUDE.md carve-out, SKILL 9a-ter, or stay norms — is a judgment call about scientific process, so it is parked for you rather than auto-filed as rules.

## Suggested action

Pick which of (a)–(c) to mandate (a one-line reply routes them as ordinary workflow-fix filings), and greenlight (d) as a direct edit.
