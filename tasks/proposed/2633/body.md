---
title: 'verify_uploads.py row-index reader uses splitlines() — false-FAILs the #2148
  rows gate on U+2028-bearing rollout JSONL'
kind: infra
tags:
- jsonl-splitlines
- verify-uploads
created_at: '2026-08-27T18:07:52Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate v1 from upload-verifier during #2546 arm-2
  Step 8: verify_uploads.py:2463 splitlines() shreds JSONL rows containing U+2028,
  producing a false rows-gate ERROR on byte-identical healthy data'
workflow: v1
---
# verify_uploads.py row-index reader uses splitlines() — false-FAILs the #2148 rows gate on Unicode-line-separator rollout text

`scripts/verify_uploads.py:2463` reads fetched row-index JSONL via `text.splitlines()`.
Python's `str.splitlines()` splits on U+2028 (LINE SEPARATOR), U+2029, and U+0085 (NEL) in
addition to `\n`. Those characters occur legitimately INSIDE JSON string values in rollout
text, so a valid one-record line is shredded into fragments and the fragments fail to parse.

This is the exact #1162 class the repo already banned. The existing
`workflow_lint.py --check-jsonl-splitlines` did not catch this call shape.

## Why it matters more than a cosmetic parse warning

The reader backs the #2148 realized-row-count reconciliation — the gate whose `rows=` token
`pod.py terminate` REFUSES to proceed without. A false ERROR there flips the overall verdict
to FAIL on healthy data, which either blocks a legitimate teardown (a wide pod keeps billing)
or teaches the operator to distrust a correct gate. It fails in the expensive direction.

## Observed instance

Found during issue #2546 arm-2 Step 8 upload verification (2026-08-27).

    artifact   raw_completions/pre_greedy_a2/gsm8k_train.jsonl
    reported   "row-index-key-absent ... Unterminated string" at line 1519
    reality    file parses clean via text-mode iteration: 7375/7375 records
    reality    sha256 byte-identical pod <-> HF
    contains   U+2028 inside a JSON string value

The check-10 ERROR set `verify_uploads.py` rc=1. The verifier superseded it with an
independent recount plus byte-identity, so #2546 was not blocked — but only because a human-
directed recount happened to run. An unattended run would have surfaced a FAIL.

Raw evidence from that session: `/tmp/verify-2546-A.json` (the rc=1 run, ERROR is exactly
this), `/tmp/verify-2546-B2.json`, `/tmp/verify-2546-B3.json`.

## Fix

1. Replace the `splitlines()` read at `scripts/verify_uploads.py:2463` with `text.split("\n")`
   plus an empty-line strip guard, or iterate the fetched file in text mode — whichever matches
   the #1162 rule's prescribed form already used elsewhere in the repo.
2. Extend `workflow_lint.py --check-jsonl-splitlines` so its coverage reaches this file and
   call shape. The lint existing but not firing here is the more general defect: any other
   JSONL reader with the same shape is equally exposed.
3. Add a regression fixture: a JSONL row carrying U+2028 inside a JSON string value, asserting
   the reader returns one record rather than two fragments.

Audit the repo for sibling `splitlines()`-on-JSONL readers while in there; the lint gap implies
this call shape was never covered, so #2463 is unlikely to be the only one.

## Provenance

Surfaced as a `workflow-fix-candidate v1` by the `upload-verifier` subagent during #2546 arm-2
Step 8 verification, and auto-filed by the #2546 orchestrator per
`.claude/rules/workflow-fix-on-bug.md`. Confidence: high (the mechanism is a documented
property of `str.splitlines()`, and the false ERROR was reproduced against a file proven
byte-identical and independently parseable).

Do not modify any #2546 artifact — that task's data is verified and its pod is terminated.
