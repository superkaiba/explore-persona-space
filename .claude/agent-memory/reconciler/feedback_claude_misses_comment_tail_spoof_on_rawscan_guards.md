# Claude misses comment-tail waiver spoofs on raw-scan guard hooks

(Entry file created during the #1891 index curation — the index pointer was dangling. The full index hook is preserved below.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude misses comment-tail waiver spoofs on raw-scan guard hooks](feedback_claude_misses_comment_tail_spoof_on_rawscan_guards.md) — replay `<destructive> # <waiver-token>` shapes yourself (Write-tool probe script); pre-existing waiver + new detectors = in-round fail-open; fail-open ≠ documented fail-closed trade-off. #897 r1 FAIL.
