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
3. **Track-S chat cell S1 is slots[:,0] → profiles[:,1]** (`issue825_fit_cells._normalize_cell`:
   Track S = "assistant slot -> a1 profile", target_turn_index **1**, NOT 0). A hand-rolled streamed
   loader that slices profiles[:,0] silently reproduces a WRONG Y (#931 r3: transfer read 0.023 vs
   the committed 0.073; caught only by the validate-against-a-committed-row-FIRST step, then cost a
   full 43 GB re-stream). Cache the sliced (X, Y, ids) npz right after streaming so a downstream
   crash never re-streams; shards are bf16 — convert via `t.float().numpy()`, never `np.asarray`.
   Also: the store has 10 .pt shards (the "20 files" count includes .json sidecars).
