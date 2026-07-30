---
title: 'daily-held: 70.9MB npz in git eval_results — size-cap decisi'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-29T07:18:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 3): a 70.92 MB .npz tensor
  was committed to git eval_results/ (GitHub GH001 large-file warning); whether to
  bless a size cap for convention-committed tensors (HF instead) and whether to purge
  the blob is a policy + history decision'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-B P9. Held under the judgment-call carve-out: **scientific-meaning / policy change** (Upload Policy boundary) + a possible **destructive** history rewrite if the blob is purged.

## The held item

A 70.92 MB `.npz` tensor was committed to git `eval_results/` on 2026-07-28 and drew GitHub's GH001 large-file warning on push (hard limit is 100 MB — the same pattern will eventually hard-fail). Upload Policy says `eval_results/` is JSON/text only, but convention has drifted for mid-size tensors. The tensors are believed mirrored on HF per the round's upload phase (unverified).

## Suggested action

Decide: (a) bless an explicit size cap (e.g. >25 MB convention-committed tensors go to the HF data repo `analysis_tensors/`) as an Upload Policy edit — that edit itself can then be a route-2 filing; (b) leave the existing blob (harmless below 100 MB) or direct a history purge (destructive — needs your call).
