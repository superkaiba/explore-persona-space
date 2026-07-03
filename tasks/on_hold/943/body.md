---
title: 'LOFO-banded re-read of the #810 fold-flip: family-fold selection-symmetric
  nulls for the 46-summary axis'
kind: experiment
tags: []
created_at: '2026-07-03T17:58:02Z'
has_clean_result: false
parent_id: 810
workflow: v1
goal: 'Determine whether the #810 round-3 header/boundary LOFO fold-flip clears a
  selection-symmetric family-fold difference band (vs ordering-only), or is selection
  noise on 7 correlated rows.'
---
## Goal

Determine whether the #810 round-3 header/boundary LOFO fold-flip clears a selection-symmetric family-fold difference band (vs ordering-only), or is selection noise on 7 correlated rows.

## Overview

Parked-redundant follow-up proposal from #810 (rank 3, `lofo-banded-foldflip`). Full spec in #810's `epm:follow-ups` v1 marker (2026-07-03). 0 GPU-h (CPU refit over stored summaries, ~2-4 h e2-standard-8).

## Value critique

REDUNDANT (both screen critics agree, no reconciler needed). Duplicates: **task #920 (interpreting)** — #920 already asks the same family-held-out, selection-corrected question for boundary/template answer summaries vs the whole-answer mean on the same 50-context base-map line, with a broader context×answer recipe sweep including the 46-answer-family axis; its takeaways already report that no boundary/template recipe separates from the mean under the family-restricted bands, and a zero-power head-to-head. Claude screen adds: same construct/measurement (LOFO family folds, selection-symmetric bands, 46-family axis, 50-context battery).

Revival conditions (from the Claude verdict): revive via `task.py set-status <this> proposed` if (a) #920's final clean-result retracts or fails to cover the boundary-row family-band read, or (b) #810's `header-echo-ablation-capture` round KILLS the echo account (empty-answer skill collapses), making the fold-flip a genuine open puzzle needing its own banded read.
