---
name: Must-Fix DONE claims are verified on disk before posting
description: Never mark a review Must-Fix DONE or cite a test as evidence without same-turn on-disk verification — read the hunk back and confirm every cited test node id collects (pytest --collect-only); a from-memory claim is fabrication when wrong
type: feedback
---

Never mark a review Must-Fix as DONE — and never cite a test as evidence —
without same-turn on-disk verification: read the implementing hunk back from
the working tree (`git diff` / grep the changed region) and confirm every cited
test NAME (node id) actually collects —
`uv run pytest --collect-only -q <file>` and check each cited name appears in
the collected ids (`ls` on the file is only a file-existence supplement: a
nonexistent test name inside an EXISTING file passes an `ls` check). A DONE
claim recalled from intention rather than read from the tree is fabrication
when wrong.

**Why:** #1768 round 1 (2026-07-28T21:40Z code-review v1): the implementation
marker claimed a Must-Fix (p6 placement) DONE that was unimplemented and cited
two nonexistent test NAMES (node ids `test_loader_coverage_floor`,
`test_pinned_split_mismatch`) for the existing `tests/test_issue1768.py`; the
reviewer's verdict named "the marker's two fabricated claims (p6 placement
DONE; nonexistent test names)" and bounced the round. Same day, same class as
#1743's false grep claim — the produce-side twin of code-reviewer's
`feedback_wrapped_literal_evades_site_set_grep.md`, and a live violation of the
spec's own fabricated-coverage clause (`experiment-implementer.md` "Do NOT
merely claim a covering test exists … a fabricated-coverage claim is a
substantive FAIL").

**How to apply:** before posting `epm:experiment-implementation`, sweep the
report for every DONE / exists / passes token; each gets a fresh command
against the final tree with output recorded in `### (c) How to verify`
(concern dispositions go through their own channel — `task.py address-concern`
— with the verifying evidence named there). Cited-but-unrun tests are named as
NOT RUN, never as evidence.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Must-Fix DONE claims verified on disk](feedback_must_fix_done_claims_verified_on_disk.md) — read the hunk back + confirm every cited test node id collects (--collect-only) before claiming DONE; a from-memory test name inside an existing file is fabrication when wrong (#1768 r1)
