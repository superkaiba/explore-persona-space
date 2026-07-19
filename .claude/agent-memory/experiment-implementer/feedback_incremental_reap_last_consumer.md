---
name: Incremental cache reap only after the cache's LAST consumer; direct-path readers need a re-stage fallback
description: A between-phase `clean_experiment_downloads --incremental` call sequenced before a later phase that reads the hf_dl cache converts a ~0.05 GB saving into a FileNotFoundError crash after hours of paid phases; enumerate every hf_dl reader first + add re-stage-on-demand
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
