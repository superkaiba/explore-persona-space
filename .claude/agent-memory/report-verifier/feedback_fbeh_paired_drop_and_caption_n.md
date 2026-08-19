---
name: fbeh-paired-drop-convention
description: "#2162 recompute gotchas: aggregate views drop a pair from ALL arms when one arm has 0 coherent draws; caption 'n = X across all cells' can be dataset-wide, not the figure's n"
metadata:
  type: feedback
---

Two recompute traps from #2162 (the issue2162 analysis/figure pipeline; likely reused by descendants):

1. **Paired-drop in aggregate views.** When ONE arm of a pair has `n_coherent == 0` (f_beh null), the aggregate figures (`hero_ftype`, `route_contrasts`, `two_by_two`) drop that pair from ALL THREE arms of that cell; the per-pair companion views keep it for arms that have data. A naive per-arm recompute mismatches by exactly one pair on the affected cell (here `language_implied|pe`: n=35 vs 36, bar delta ~0.001, disclosed by the `(n=35)` tick label).
   **Why:** paired-comparison convention in the analysis, not stated in the manifest transform or the body Methodology.
   **How to apply:** on a single-cell mismatch of ~1 pair, test the drop-from-all-arms hypothesis before flagging; recommend (WARN, not FAIL) a half-sentence disclosure in the result's Methodology bullet.

2. **Caption `n` may be dataset-wide.** A plotter caption ending "n = 2,574 scored steered pair-rows across all cells" quoted the WHOLE dataset's row count on a figure rendering only 11 route cells (~750 unique rows; sidecar `total_points` 2,181 render instances). Literally true as worded, but reads as the figure's n.
   **How to apply:** always reconcile a caption n against sidecar `total_points` AND the unique source-row count; a labeled dataset-wide denominator is a WARN with suggested rewording, not a bounce.
