# snapshot_download(allow_patterns=...) stalls on the ~1M-file data repo — use scoped list_repo_tree + per-file hf_hub_download

**Incident (#2378 r4):** staging ~240 small `sega_mined` shards for worked
examples via `snapshot_download(..., allow_patterns=[prefix glob])` hung
>8 min with zero new files: allow_patterns filters CLIENT-side, so the call
first enumerates the ENTIRE repo tree, and `superkaiba1/explore-persona-space-data`
sits at ~1M files. Killed by PID (kill-before-relaunch probe→TERM→confirm).

**Working recipe:** (1) `list_repo_tree(path_in_repo=<prefix>)` — scoped,
server-side, seconds even on the 1M-file repo — to get exact file names;
(2) `hf_hub_download` per file (ThreadPoolExecutor ~8 workers is fine for
~250 KB shards), with `HF_HUB_DISABLE_XET=1` + `HF_HUB_ENABLE_HF_TRANSFER=0`
set BEFORE the huggingface_hub import (small-file-storm leg of the
accelerator failure matrix, gotchas.md).

**Join trap in the same round:** per-row worked examples need text from
TWO stages whose shard chunking does NOT align — a row found in
`segb/<cell>_w1_s0_c0001.jsonl` is usually NOT in the same-named
`sega_mined` shard (only 1 of 5 picks matched), and top-up waves put
`_w2_` row_ids inside `w1`-named segb shards. Search the whole cell's
stage files by row_id, never assume shard-name alignment.
