# Independent collector/readout review

Reviewer: `answer_ceiling_independent_review` (independent AI subagent). This is implementation and scientific-design review, not human annotation agreement. The following is the reviewer's final verdict, preserved verbatim apart from Markdown heading formatting.

**PASS — no remaining scientific or code blocker in the reviewed collector/readout.**

The implementation preserves exact answer/vector joins, matched target availability, connected-group splits, and training-only standardization, PCA, and regularization selection. Pilot acceptance and main-label provenance checks prevent incomplete or stale evidence from authorizing the readout.

The reporting fixes are present:

- Per-class positive/negative answer and independent-group counts.
- Exchangeable, moved, and fixed shuffle-group counts, including actual target changes.
- Explicitly conditional bootstrap intervals with defined-replicate counts.

Independent checks found categorical bootstrap calculations matching explicit resampling to \(2.8\times10^{-17}\); the earlier readout suite passed seven tests. **Parent-reported final verification:** all 19 collector/readout tests passed in 8.80 seconds, with formatting and lint checks passing. The temporary concurrent-test slowdown did not establish a code defect; the isolated loader passed.

The parent's full-width timing smoke used fixture targets only. **No behavioral labels were fitted and no new scientific result is established.** Interpretation remains limited to judge-defined expressed properties in this cohort, with uncertainty conditional on fixed out-of-fold predictions; it supports neither an intrinsic decodability ceiling nor a universal high-level versus low-level ranking.

No manuscript edits, API calls, or real-label fits were performed during this review.

## Current execution status

The OpenAI model-access preflight rejected the existing credential before any annotation call. No real pilot-acceptance record, behavioral labels, or actual-label readout results exist. This PASS concerns the prepared implementation and design; it does not accept an unexecuted annotation instrument. A valid OpenAI credential is required to run the reviewed pilot, inspect its actual judgments and reliability, and continue under the saved acceptance protocol.
