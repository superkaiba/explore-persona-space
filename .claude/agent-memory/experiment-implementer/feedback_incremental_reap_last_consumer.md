---
name: Incremental cache reap only after the cache's LAST consumer; direct-path readers need a re-stage fallback
description: A purge/reap sequenced before the artifact's LAST consumer (incremental cache reap, OR a reused upload helper's own upload-then-purge) converts a small disk saving into a crash after hours of paid phases; enumerate every reader first + add prefix-threaded re-stage-on-demand
type: feedback
---

The between-phase incremental reap contract (CLAUDE.md § Disk hygiene) assumes
the reaped `data/issue_<N>/hf_dl/**` cache "re-downloads on demand" — but
direct-path readers (`parent.load_store(corpus_dir, ...)`-style `open()` calls)
implement no re-download, so a reap sequenced before the cache's last consumer
crashes the run. On #1489 (2026-07-18, att-20260718-093558) the dispatch script
reaped `data/issue_1489/hf_dl` (0.05 GB, disk had 88.5 GB free) right after
upload-a1; the very next phase (P3 distill) read
`hf_dl/corpus/prefix_store.jsonl` → FileNotFoundError, killing a run whose
phase_a had just completed ~3.5 h of paid GPU work. Before placing ANY
incremental reap: grep-enumerate every `hf_dl` reader across the whole
dispatcher + all phases (including later provisions, e.g. phase_b); if any
later consumer exists, there is no legal mid-run reap slot — let Step-8
terminal cleanup own it (per-phase disk-headroom canaries already guard
pressure). Defense in depth: guard each corpus-consuming phase entry with a
re-stage-on-demand call through the EXISTING deterministic staging helper,
pinned to the same revision the earlier phases consumed. (#1489 crash-fix r6,
commit c5224efb.)

The purge is not always the dispatcher's: a REUSED upload helper can carry its
own disk-bounding upload-then-purge (the #779-lineage `_flush_upload_batch`
purges local chunks after Hub verification), so grepping the dispatcher alone
for reap calls misses the seam. On #1776 (2026-07-29, resume 5) p5a_capture's
reused helper uploaded wildchat chunks to the Hub and purged the local
`--out-root`; the SAME run's p5_transfer consumed that dir → "no capture
chunks" AssertionError after the transfer had already scored a full leg. At
dispatch-design time, enumerate consumers of any purge-BEARING upload helper
too (read the reused producer's code, not just the dispatcher); the fix shape
is the same consumer-side, prefix-threaded Hub re-stage fallback — scoped
listing (never full-tree on a ~1M-file repo) + atomic idempotent per-file
stage + sha256 verify (#1776 crash-fix c8, commit f3413617bc,
tests in the phase5 staging probe).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Purge/reap only after the LAST consumer](feedback_incremental_reap_last_consumer.md) — enumerate every reader (incl. later provisions AND reused upload helpers' own upload-then-purge) before any mid-run reap; consumers need a prefix-threaded re-stage-on-demand guard (#1489 r6, #1776 c8)
