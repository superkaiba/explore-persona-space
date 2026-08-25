---
name: secret-scrub-round-compose
description: "Producer-side secret-scrub/redaction rounds (#2502 r11): verdict-side secret discipline (never reproduce secret-SHAPED literals, even synthetic fixtures), hollow-gate = detection PARITY (narrower=liveness, wider=over-redaction, placeholder-must-not-retrigger), stored-hash-vs-post-scrub-text provenance duty with composer-traced consumer anchors, and the mid-review memory-commit HEAD disclosure"
metadata:
  type: feedback
---

When the round under review adds a secret scrub / redaction step to a
producer before an upload gate (#2502 r11, 2026-08-25), four compose
duties beyond the standard revision-round recipe:

1. **Verdict-side secret discipline (output contract).** The Codex
   verdict is posted via `task.py post-marker` and COMMITTED to
   events.jsonl, so it must itself pass gitleaks/secret gates.
   Instruct: never reproduce a contiguous secret-SHAPED literal in the
   output — not even the SYNTHETIC test fixtures' values (`_PLANTED`
   built by concatenation is still secret-shaped when echoed whole);
   refer by variable name + line. Also instruct that the live
   refusal's flagged strings are in NO inlined artifact (count-only
   disclosure) — absence is not an evidence gap.

2. **Hollow-gate lens = DETECTION PARITY, both directions.** The scrub
   and the upload gate must share ONE detection chain (`scan_bytes` +
   dummy filter): a scrub NARROWER than the gate leaves the upload
   still refused (liveness defect); a WIDER one over-redacts real
   corpus text (data-integrity defect). Plus: verify the replacement
   placeholder cannot itself re-trigger a pattern on the gate's
   re-scan, and trace the tool's residual semantics (what happens to a
   finding it cannot redact).

3. **Stored-hash-vs-post-scrub-text provenance duty.** A scrub that
   redacts text AFTER content hashes were computed leaves stored
   hashes describing PRE-scrub bytes. Composer pre-traces the consumer
   anchors (fingerprint helpers that read STORED pairs vs any
   re-hash-and-compare site; log-digest helpers that hash text for
   ledgers only) and hands Codex the sweep with exact file:line leads
   + the adjudication frame (real defect = a consumer that VERIFIES
   hash-vs-text; benign = documented election with no verifying
   consumer, pinned in a test) — never pre-resolve the verdict. Also
   name the text-collision edge (post-scrub same-length placeholder
   runs can make previously-distinct rows byte-identical).

4. **Mid-review memory commits on the review branch break HEAD==tip
   claims.** Composing the prompt with "HEAD == round tip, NEW-blob
   line numbers match" and THEN committing agent memory to the same
   branch falsifies the claim Codex will probe. Either commit BEFORE
   final assembly, or patch the prompt to the r10-precedented
   disclosure: "housekeeping commit(s) on top of the round tip at
   HEAD, outside the round file set; the payload files are
   byte-identical round-tip↔HEAD (composer-verified via
   `git diff <tip>..HEAD -- <files>`)".

**Why:** #2502 r11 — the fix stood between 21 real-secret-grade
strings and a public HF upload; the review verdict itself is a
committed artifact; and the r10 prompt's rubric carried ~25 woven
round-specific insertions, so the two-pass patch strategy (anchor-pair
replace_between + exact-string subs, then a TOKEN-audit loop over
prior-round SHAs / marker versions / 'This round' contexts until
clean) was what caught the last stale spans (Step 3.6 trigger note,
Step 2 fit-loop note, Step 6 plan surface).

**How to apply:** any round adding redaction/scrubbing before an
upload/publish gate, and any compose reusing a rubric from a round
with a large woven duty set — re-align the D-numbering references
inside the rubric (D1 item N cites) with the NEW round's duty list.
Related: [[revision-round compose recipe (round 2+)]],
[[data-hardening round compose]],
[[marker-quoted-flags-verified-against-parser]].
