---
name: width-reshard-stale-sibling-shards
description: Width-keyed resume that regenerates only SURVIVING workers' shards leaves the vanished workers' stale files for glob consumers to double-count — sweep naming/cleanup/uniqueness yourself (#2329 r1 F3)
metadata:
  type: feedback
---

A cross-width resume fix (done records carrying `num_workers`, mismatch ⇒
re-run) is only HALF a fix: on an 8→4 reshard, workers 0..3 overwrite their
own `*_w0..w3` shard files, but the vanished workers' `*_w4..w7` files (and
their done records) survive on disk. The new 4-wide shards cover ALL units, so
the stale files are pure duplicates — and every glob consumer
(`glob("anchors_*.jsonl")` concatenation in the judge loader, sum-accumulators
over `va_anchors_*.pt`, P5 `upload_dir_hf(dir, ["*.jsonl"])`) silently
double-counts them; a dict-keyed loader instead last-write-wins, so different
consumers see DIFFERENT data. (#2329 r1 finding 3: the r1-M2 width-aware
resume was reviewed and praised while the stale-sibling sweep was missing;
Claude PASSed on the resume docstring.)

**Why:** the resume predicate and the artifact SET have different owners — the
predicate is per-(worker, batch), the consumers are per-directory. Fixing the
predicate without namespacing shards by realized width, quarantining
prior-width files at reshard, or asserting one row per (unit, draw) at load
leaves a silent-corruption path on a designed-for fallback.

**How to apply:** whenever a finding or clearance touches sharded resume with
a width/worker axis, check THREE things yourself: (1) are shard filenames
width-namespaced? (2) does the reshard path delete/quarantine prior-width
files AND done records? (3) does any consumer assert uniqueness per unit?
All three absent + glob consumers present ⇒ CONFIRMED Major regardless of how
good the resume predicate looks. Same class applies to sibling per-worker
outputs (anchor-margin shards) — sweep the whole directory's writer set, not
just the file the finding names.

Related: [[claude-misses-producer-consumer-key-mismatch]] (same round's F1 —
a producer dict schema read as an int by a consumer the Claude reviewer had
"read and passed"; producer-writer vs consumer-subscript at the exact key is
still the check), [[claude-misses-same-file-siblings]] (same round's F9 —
anchors implemented text-before-capture with a #779 comment while grid/stage2
siblings in the SAME file kept the reversed order; verify the duty at every
sibling site, not the one with the comment).
