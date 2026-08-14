---
title: 'verify_task_body: flag positional cross-references in Results that misresolve
  after fold reordering'
kind: infra
tags: []
created_at: '2026-08-14T00:45:34Z'
has_clean_result: false
workflow: v1
---
## Goal
Add a verify_task_body.py check (WARN-posture, scoped to <!-- clean-result-v4 --> bodies) that parses positional cross-reference tokens under ## Results — '(two results up)', '(N results up/down)', 'the previous result' — resolves the ordinal against the actual H3 sequence, and flags mismatches (e.g. a clause saying 'round-4' whose resolved target heading starts 'Rounds 1-3:').

## Provenance
Surfaced by the clean-result-critic on #2221's specialized_corpus_remine re-fold (epm:clean-result-critique v3, 2026-08-14T00:41:50Z): both '(two results up)' pointers in the folded body misdirected after the fold reordered the result sections — the intended round-4 targets were five and six results up. The class is fold-reordering-specific and recurs in multi-round consolidated bodies. The #2221 revision round fixed the instances by hand; this task adds the mechanical guard.
