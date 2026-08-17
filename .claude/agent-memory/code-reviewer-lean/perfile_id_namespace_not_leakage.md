---
name: perfile-id-namespace-not-leakage
description: Split-id manifest review — eval ids numerically inside the train id range look like leakage but are benign when ids are per-FILE local ids; settle by reading the id-extraction line, then verify by recompute
metadata:
  type: feedback
---

When reviewing a committed split-ids / manifest artifact, a FULL numeric
overlap between eval-split ids and the train id range (val ⊂ 0..N_train)
is NOT automatically train/eval leakage: it is exactly the signature of a
per-file id namespace (each JSONL numbers its own rows, e.g.
`ladder_local_id` starting at 0). Settle it by grepping the GENERATING
script for the id-extraction line (which field, from which file) before
flagging — and note in the verdict that consumers must key `(split, id)`.

**Why:** #2330 R1 g2 — val_400/test_1000/wc_test_1k ids overlapped
train_10k 100%; the extraction line (`row["ladder_local_id"]`, file order,
per split file) proved the namespace per-file. A leakage FAIL here would
have been a false positive forcing a re-roll.

**How to apply:** for any P0-style data artifact, run recompute checks
(count, prefix-nesting positionally not set-wise, sha256 under the
documented domain, dropped-id exclusion, file-order property against the
pinned upstream when the local cache exists), verify the committed BLOB
sha == the inspected file, and treat cross-split id intersection as a
namespace question, not a leakage verdict, until the extraction line is
read. Also sanity-sum the length_scan `scanned` count against the union of
scanned splits (catches a silently unscanned split).
