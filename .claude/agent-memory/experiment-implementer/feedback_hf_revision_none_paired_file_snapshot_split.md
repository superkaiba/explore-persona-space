---
name: hf-revision-none-paired-file-snapshot-split
description: hf_hub_download(revision=None) re-resolves main PER CALL — paired-file fetches (.pt + .json sidecar) split across snapshots/<sha>/ dirs when the shared data repo moves mid-run; pin one resolved sha per run (#2061)
metadata:
  type: feedback
---

`hf_hub_download(..., revision=None)` re-resolves the repo's `main` ref on EVERY
call and symlinks the fetched file into `snapshots/<resolved-sha>/` — it never
backfills earlier files into the new snapshot dir. On the fleet's shared data
repo (`superkaiba1/explore-persona-space-data`, commits landing constantly), a
multi-file loop that fetches PAIRED files per unit (a shard's `.json` sidecar
then its `.pt`) can have `main` move BETWEEN the two calls: the pair lands in
two different snapshot dirs, and any adjacency read
(`Path(pt).with_suffix(".json")`) misses — surfacing as a misleading
"file missing / re-stage" error while the data is complete upstream.

**Why:** #2061 P1 encode crashed at cell [2/35]: if11k's 15 shards spread over
FIVE snapshot dirs in ~5 min; `shard014.json` landed in `snapshots/9ba8d73…/`,
`shard014.pt` in `snapshots/710ad8b…/` (main moved between the two calls), and
the sidecar assert fired. The failure marker's initial diagnosis ("the plain-v2
path bypasses the sidecar fetch") was WRONG — both grains ran the sidecar-aware
helper; the cache forensics (interleaved shard subsets across snapshot dirs)
were dispositive. Diagnosis signature: one logical store's files interleaved
across multiple `snapshots/<sha>/` dirs + `refs/main` newer than the earliest
fetched files.

**How to apply:** any multi-file hub-consuming run resolves `main` → commit sha
ONCE at entry (`HfApi().repo_info(repo, repo_type=...).sha` under
`retry_transient`; canonical helper
`scripts/issue2061_hub_io.py::resolve_data_repo_revision`) and threads that sha
through every `list_repo_tree` + `hf_hub_download`. Add a fail-loud adjacency
guard at the fetch seam (compare the two returned local paths) so a future
unpinned caller gets "revision drift" instead of the misleading downstream
error. Sibling lesson: a single-cell smoke over a MULTI-PATH resolver tests only
the path it selects — smoke one cell per path/grain
([[smoke-ft-zero3-width-parity]] class-coverage family). Staging-lane pinning
precedent: [[hub-prefix-mirror-vs-consumer-layout]].
