---
title: 'Sweep contrastive negatives to failure: extend the budget sweep beyond 1600
  until source implantation plateaus or collapses'
kind: experiment
tags: []
created_at: '2026-06-12T20:21:43Z'
has_clean_result: false
origin_prompt: 'task #616 cross-check audit (user asked to double-check matches):
  ''increase number of contrastive negatives until crashes'' had no covering task'
goal: Find the contrastive-negatives count at which source implantation stops growing
  or collapses, discriminating the schedule-length mechanism from loss-competition.
---
## Goal

Find the contrastive-negatives count at which source implantation stops growing or collapses, discriminating the schedule-length mechanism from loss-competition.


## Summary

Sweep the contrastive-negatives count upward until the implant breaks: extend the 0/400/800/1600 budget sweep (#472) to much larger negative totals at fixed positives and find where source implantation stops growing, plateaus, or collapses ("increase number of contrastive negatives until crashes").

## Motivation

More negatives monotonically strengthened the source implant across the tested range (#472: +2.1/+8.3/+13.5/+20 nats at 0/400/800/1600), and #601 showed the mechanism is schedule length, not the negatives' own loss. The breaking point — where added negatives stop buying implant strength or actively destroy it — is unmeasured, and its location discriminates the schedule-mechanism account (implant should keep tracking optimizer steps) from any loss-competition account (implant should crash once negatives dominate the batch).

## Blocked-on note

Interpretation depends on #613 (alive-negatives loss-suppression A/B, running): if #613 finds the negatives' loss term has a live restoring force, the crash point is a loss-competition prediction; if not, dose-to-failure is a schedule question. Pick up after #613 lands.

## Provenance

Consolidated from the 2026-06-11 mentor meeting notes (docs/mentor_updates/2026-06-11.md, contrastive-negatives slide: "increase number of contrastive negatives until crashes") — task #616 cross-check audit found no existing task covers it.
