# Saved fp16 W_raw maps suffice for recentered transfer reads (#931)

The #825/#931 map line saves per-spec per-layer maps as `{"W_raw_fp16": W_std/xsd, "layer": L}`
(`issue931_similarity.py --save-maps`). For a **recentered** transfer read, that is the WHOLE map:
`transfer_r2`'s recentered application takes xmu/ymu from the TARGET's train folds, and
`((X−xmu)/xsd)@W_std == (X−xmu)@W_raw`, so pass `W_std=W_raw, xsd=ones` (zeros placeholders for
xmu/ymu). No need to re-download regime tensors and refit — a critic-requested control transfer
(sep→chat, #931 r2) cost one 26 MB map + the target store.

Two load-bearing steps:
1. **Validate the pipeline against a committed row FIRST**: reproduce an existing transfer_matrix.json
   row from ITS saved fp16 map before trusting the new number (#931: 0.015188 vs committed +0.0152,
   null p97.5 matched exactly). fp16 rounding was negligible (4 decimals).
2. **Stream the target store shard-by-shard** (download → slice layer → delete; peak ≈ 1 shard) when
   VM `/` is tight; replicate the loader's all-layer NaN keep-mask before slicing one layer so row
   sets match the run. Staging on the data disk: `/mnt/eps-data` ROOT is root-owned — write under
   `/mnt/eps-data/thomasjiralerspong/`.
