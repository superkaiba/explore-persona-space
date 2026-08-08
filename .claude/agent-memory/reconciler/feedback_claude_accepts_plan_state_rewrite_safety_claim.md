# Claude accepts a plan's "procedure re-writes state Y" safety claim

(Entry file created during the #1891 index curation — the index pointer was dangling. The full index hook is preserved below.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude accepts a plan's "procedure re-writes state Y" safety claim](feedback_claude_accepts_plan_state_rewrite_safety_claim.md) — grep the literal state write + poll/monitor contracts encoding the state FREEZING; sibling: plan cites ONE watcher arm (PARK) for a status, Claude misses the pod-safety arm ("other" → silent billing); check park-then-teardown step ORDER. #908 r1, #919 r1.
