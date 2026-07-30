---
name: capture-convention-must-be-read-from-producer-code
description: Reconciling recaptures vs a parent's stored activation arrays — pin the tokenization/span convention from the PRODUCER's capture code, stream the parent draw instead of recapturing (#1482 r3)
type: feedback
---

When a gate compares fresh captures against a parent task's STORED per-context
arrays (e2/nerr/v), the parent's tokenization + span convention is whatever the
PRODUCER's capture code did — never what a diagnosis note or brief asserts.
#1482's G1 failed (Spearman 0.9904) because b2 re-tokenized response text
token-id-concat style while #779's n1m chunks were written by
`COL.capture_answer_vector` (via `N1G._capture_shard_trimmed`): ONE
full-chat-template tokenization with the assistant turn embedded
(`add_generation_prompt=False`), response span `[prompt_len:full_len]`
INCLUDING the `<|im_end|>`+`\n` tail — a 1/n_ans-scaled, arm-asymmetric offset.
The r3 brief attributed the stored convention to GENERATION-TIME token ids;
code-reading refuted that too.

**Why:** three conventions coexist in this repo (generation ids;
token-id-concat `_tokenize_row`; full-template retok w/ tail) and any mismatch
between fresh draws and the stored functional poisons variance/floor
decompositions (m2 = ||vhat − v̄_fresh||² inherits the offset systematically).

**How to apply:** (1) grep the parent's chunk WRITER and read the exact
tokenize/slice lines before building any reconciliation gate or fresh-draw
capture; (2) for the parent draw itself, prefer STREAMING v verbatim from the
stored chunks (identity by construction — only quantization of persisted fp16
preds remains, ~1e-5 median rel) over any recapture; (3) fresh resamples must
reproduce the parent functional verbatim (tail inclusion, seam behavior, no
added prefix-identity asserts the parent didn't have).

**Index-space sibling (#1775 crash-fix, 2026-07-28):** the same law covers
stored INDEX fields. #779 n50k chunks' `ci` ("GLOBAL n50k index, manifest
order") indexes the NEW-only sampling manifest (`_read_manifest` returns
`d["new"]`, 0..46599) — NOT a round1+new concatenation; a consumer that
subtracted len(round1)=5000 read 0 rows in range and crashed in production.
Pin the index space from the producer's manifest builder, and validate the
mapping BY CONTENT (normalized-prompt positional match + set membership),
never by index arithmetic alone.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Capture convention: read the PRODUCER's code](feedback_capture_convention_read_producer_code.md) — stored-array reconciliation: pin tokenize/span convention from the parent's chunk writer; stream the parent draw, never recapture (#1482 r3)
