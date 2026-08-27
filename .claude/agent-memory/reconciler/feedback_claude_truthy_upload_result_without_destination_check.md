---
name: claude-truthy-upload-result-without-destination-check
description: Claude PASS certified `assert res` on a hub._upload result as the failure backstop but never examined the helper's SUCCESS-shaped fallback (file-count overflow reroute returns a truthy OVERFLOW-repo path); a truthy return does not pin the destination. Trace every success-shaped fallback branch of a shared upload/IO helper before crediting a result check.
metadata:
  type: feedback
---

**Rule:** when a verdict disagreement hangs on "the upload/IO helper fails
loud / the caller's result check catches failures", do not stop at the
FAILURE-shaped returns (`""`, exception, raise_on_error). Open the helper
and enumerate its SUCCESS-shaped fallback branches — alternate-repo
retries, overflow reroutes, alternate-destination fallbacks — because a
truthy return does not pin the DESTINATION, and a destination-blind caller
that memoizes/attests canonical state off a truthy return re-creates the
original corruption through the fallback lane.

**Why:** #2546 r19 — the shard-gated capture-marker repair `assert res`
passed on `hub._upload`'s `_filecount_overflow_retry` result (truthy
`DEFAULT_OVERFLOW_REPO/<dest>`; default-ON; the data repo has actually hit
the 1M-file rejection, #2304), then memoized canonical shard presence and
mirrored a canonical `_complete.json` over a shard-less canonical prefix —
the exact r18 corruption the binding ruling ordered closed. Claude's PASS
verified the `""`-return arm, ignore-pattern merge, and commit atomicity
but never mentioned the overflow lane: a coverage GAP, not a rebuttal —
the PASS carried no weight against Codex's blocker (cf.
[[feedback_twin_brief_excluded_disputed_question]]).

**How to apply:** grep the shared helper for fallback/retry/overflow
symbols (`_retry`, `fallback`, `OVERFLOW`, alternate repo constants) and
check whether the caller's success predicate distinguishes destinations.
Fix belongs at the CALLSITE (require the returned path names the canonical
destination before any memo/attestation write) — do not globally change
the helper's contract when the fallback is designed durability behavior
for other callers. Companions:
[[feedback_masked_rc_fallback_vs_downstream_fail_loud_consumer]] (trace
the fallback's first consumer),
[[feedback_claude_misses_besteffort_upload_made_loadbearing]] (warn-only
upload made load-bearing).
