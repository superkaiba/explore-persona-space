---
title: 'daily-held: corpus-capture redaction default (LMSYS rows)'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-23T07:05:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 3): classifier-flagged LMSYS
  rows stored as category placeholders surfaced as placeholders in a user-facing dashboard;
  user had to ask twice to see the text; over-redaction of corpus rows treated like
  bank items'
workflow: v1
---
## Overview / Motivation

Filed by /daily 2026-07-22 as a TRACKED needs-human item (route 3 — data-handling / scientific-meaning judgment). The 07-14 #779 capture artifact stored classifier-flagged LMSYS rows (jailbreak / sexual-explicit flags) as CATEGORY PLACEHOLDERS instead of text. When the top/bottom-contexts dashboard was built on 2026-07-22 (fdf687f2, 04:06Z), the top-10 list showed placeholders; you had to ask twice ("what is this" / "can you add the actual message") before a text sidecar + red flagnote was added. Inspection showed all 5 flagged rows were ordinary jailbreak-framing prompts in their first 250 chars.

## The decision needed (why route 3)

The over-redaction happened because LMSYS-class REAL-CORPUS rows were treated like harmful-BANK items (digest-only), though the standing rules reserve digest-only for bank items and allow cherry-picked corpus rows by reference. Candidate default for FUTURE corpus captures: store truncated actual text (e.g. first 250 chars) + a flag field, with placeholders only for genuinely unprintable payloads, and keep the inline flagnote disclosure the dashboard now uses. Whether that becomes the standing capture default — and where the truncation/printability line sits — is a data-handling judgment with safety-adjacent tradeoffs, so it is parked for you.

## Suggested action

Confirm the truncated-text + flag-field default (one-line reply routes it into the capture-code conventions), or keep placeholders and accept the ask-twice friction when flagged rows surface in dashboards.
