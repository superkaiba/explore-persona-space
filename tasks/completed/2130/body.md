---
title: 'daily-fix: issue1491 ceiling must assert n_pairs'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-08-06T07:08:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): ceiling accepted 875/1000
  pairs silently after a shard crash; user-facing number rested on it ~9.5h'
workflow: v1
---
# daily-fix: issue1491 ladder ceiling accepts short pairings silently — assert n_pairs == expected (875/1000 reached a user-facing number)

## Workflow gap

In the #1491 scale-ladder analysis, a vLLM engine-startup crash killed one greedy-pass
shard (scale15 ceiling_draw_44 shard4, 2026-08-05T19:58Z pod log: "RuntimeError: Engine
core initialization failed"), leaving 875/1000 pairs — and the ceiling computation
"accepted a short pairing without complaint" (`n_pairs=875`, `available=True`). The 1.5B
ceiling (0.9855) and its normalized R², already shown to Thomas, rested on the short
pairing for ~9.5 h until a manual anomaly stop ("**Stop — anomaly.** … 875 = 7 × 125")
triggered a full 6-rung × 7-split completeness audit. This is the CLAUDE.md fail-fast
rule applied to pairing counts: a silent short-accept is a value placeholder in disguise.

verified-at-filing: the crash row, the audit marker (`epm:progress v76`), and the
discovery sequence are probed rows in session 8d7f8b25 (rows 6244–6278).
`grep -n 'n_pairs\|available' scripts/issue1491_ladder_fits.py | head` run at compose
time to locate the accept site. NOTE: #1491 is at `awaiting_promotion`-track with its
clean-result critic PASSed — the planner must check whether the audited/corrected numbers
already superseded the short-pair read in the promoted body (the session ran the
completeness audit; if any body number still rests on 875 pairs, this task also carries
the #1701-class record-integrity correction duty).

## Proposed change

In the #1491 ladder fits/ceiling path (`scripts/issue1491_ladder_fits.py`), assert
`n_pairs == expected_n` (or fail loud below a plan-declared floor) instead of returning
`available=True` on short pairings; propagate the same check to any sibling consumer that
aggregates per-split pair counts. Add a regression test with a deliberately truncated
pairing fixture.

## Provenance

- workflow_fix_target: scripts/issue1491_ladder_fits.py
- origin: /daily 2026-08-05 problem sweep — miner 1 P2 (probed rows; fail-fast class).
