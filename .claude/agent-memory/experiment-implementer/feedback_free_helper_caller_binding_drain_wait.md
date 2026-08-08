# free-helper caller-binding leak vs drain-waits

(Entry file created during the #1891 index curation — the index pointer was dangling. The full index hook is preserved below.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [free-helper caller-binding leak vs drain-waits](feedback_free_helper_caller_binding_drain_wait.md) — `del` in a callee frees nothing; rebind `x = _free_hf(x)` + post-rebind empty_cache; PEFT wrappers pin base (#1333 r9)
