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

5. **Row-DROP variant (#2502 r12, 2026-08-25 — gen-side whole-row drop
   instead of in-place redaction).** The four duties above adapt:
   placeholder-retrigger is N/A (nothing inserted), and the parity
   frame changes — per-LINE `scan_bytes` vs the gate's whole-FILE
   `scan_file` needs a BOTH-directions superset adjudication (narrower
   = the crash repeats at upload; wider = over-drops real completions)
   plus the JSONL single-physical-line argument. NEW duty set the drop
   shape adds: (a) the token-ID leak channel (text-only scrub passes
   the gate while `completion_token_ids` decode back — the drop must
   take the WHOLE line; hand the no-scrub-text-keep-ids sweep); (b)
   the downstream COUNT-RECONCILIATION trace (recount-from-rewritten-
   file ordering, absent-key `.get(...,0)` honesty for cross-pod
   recounts, and the gen→capture→fits completeness chain — verify
   every fail-closed gate compares POST-drop declarations, never a
   corpus-derived expectation; composer pre-traces the exact
   consumers: require_gen_complete asserts presence-only,
   capture_meta.per_chunk derives from captured rows, fits reconciles
   vs capture_meta); (c) all-dropped fail-loud + atomic tmp+replace +
   the `.redact.tmp` residue and secret-bearing-file-left-on-disk
   adjudications; (d) persist-by-default TENSION as a both-routes
   adjudication (dropped rollout text persisted NOWHERE is a disclosed
   secret-hygiene exception to "generations are never a genuine
   discard" — hand both routings, cap the challenge at CONCERN with a
   NEW id, never a FAIL lever for a disclosed election).

6. **Revise-round variant (#2502 r13, 2026-08-25 — fixing the r12
   reconciler-upheld parity BLOCKER, scan_bytes→scan_file).** When the
   round IS the fix for an upheld blocker on the same scrub surface:
   (a) the hollow-gate lens SHIFTS from detection-parity adjudication
   to gate-IDENTITY + residuals — once the fix calls the gate's OWN
   function, parity holds by construction AT the loop's return, and
   the review's center of mass moves to the offset→line mapping
   arithmetic, the loop exits (bounded / converging / fail-loud), and
   the no-mutation window between return and upload; (b) an
   addressed-CLAIMED ledger BLOCKER arms the closure-verification
   duty: verified-closed = status line only (never re-emit the id);
   closure-FALSE = substantive FAIL + the ONE sanctioned same-id
   `CONCERN::` re-emit (the ledger's latest event is `addressed`, so a
   fresh raised row RE-OPENS rather than duplicates); (c) a body
   rewrite that GROWS the function shifts EVERY downstream anchor —
   re-grep all woven anchors (seam lines, stats chain, parser flags,
   pin defs) fresh, never offset arithmetically; (d) the
   docstring-corrected duty verifies the REPLACEMENT claims against
   the underlying mechanism (read `_scan_stream` itself) — the
   blocker's second face is a NEW overclaim; (e) disclosed
   not-data-reachable defensive raises (unmappable finding,
   non-convergence) adjudicate via the offset-semantics read, never as
   smoke-run-missing; (f) prior-round pins that the rewrite could
   stale get an explicit stale-pin grep duty (a loose
   `match="refusing to upload"` matching all three new raises needs a
   fixture-forces-the-right-branch check).

**How to apply:** any round adding redaction/scrubbing before an
upload/publish gate, and any compose reusing a rubric from a round
with a large woven duty set — re-align the D-numbering references
inside the rubric (D1 item N cites) with the NEW round's duty list.
The r12 reuse of the r11 rubric-patched file worked via the same
two-pass strategy (~30 anchor-pair/exact replacements, each
count-asserted, then the token audit with a CASE-SENSITIVE ALLOWED
dict for deliberate history references — "ROUND-11" uppercase does not
match a lowercase audit token, and substring hits like "corpus
builder" need explicit allowance). The r13 reuse of the FULL r12
prompt (not just the rubric file) used the same recipe: slice at
content-asserted section boundaries, keep the plan envelope +
protocol core, rewrite head/duties/verdict wholesale, ~30
replace_between/replace_once patches on the protocol span.
Related: [[revision-round compose recipe (round 2+)]],
[[data-hardening round compose]],
[[marker-quoted-flags-verified-against-parser]].
