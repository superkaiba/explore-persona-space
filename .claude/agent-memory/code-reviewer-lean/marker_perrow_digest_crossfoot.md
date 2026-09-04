---
name: marker-perrow-digest-crossfoot
description: Cross-foot a durable marker's per-row digest lines against its own claimed total before trusting the transcription, then live re-run to resolve which side is wrong
metadata:
  type: feedback
---

When an implementation marker (or any durable report) lists per-unit numbers
(per-row planned calls, per-cell counts) NEXT TO a claimed total, sum the
listed lines yourself BEFORE trusting either. A sum mismatch proves a
transcription defect with zero re-runs, and one live re-run then resolves
which side is wrong.

**Why:** #2658 r18: the implementation marker's (c) dev dry-run digest listed
per-row planned calls that summed to 428,100 against its own claimed
dispatch_total 427,500. A live `--dry-run` re-run showed the total, cells and
answers were all correct and 5 of 8 per-row lines were mis-transcribed
(stale/mixed source). Headline-only spot checks would have PASSed the marker.

**How to apply:** any marker/report with per-unit lines plus a total: awk-sum
the lines, compare to the stated total AND to your own re-run (the brief
usually mandates re-running one leg anyway, reuse that output). An internal
sum mismatch is a durable-record CONCERN even when the code and headline
numbers verify clean. Related: [[marker_success_command_verbatim_rerun]],
[[spend_timing_artifact_exact_recompute]].
