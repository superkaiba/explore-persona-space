# #1887 lambda-selection corrections table

Replay gate: **PASS** (0 of 1 referenced cells failed |dR2| <= 0.001).
Headline corrected read: reduced-basis (unselected). Forced-lambda is diagnostic only. Nulls: false — committed nulls selector-matched to the committed read only. CIs: point-estimate-only — committed CI machinery not re-run.

| cell | variant | n_train | d | committed | replay Δ | capped 0.9 | inner-CV | reduced (headline) | forced 1e2/1e3/1e4 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| control__S_instruct_chat | n_gt_d_control | 4000 | 3584 | 0.6731 | -0.0000 | 0.6731 | 0.6957 | 0.6764 | 0.5839/0.6935/0.6719 | indeterminate (within fold SE) |
| control__S_pretrained_chat | n_gt_d_control | 4000 | 3584 | — | — | 0.5877 | 0.5969 | 0.5639 | 0.4164/0.5877/0.5720 | no-committed-reference |
