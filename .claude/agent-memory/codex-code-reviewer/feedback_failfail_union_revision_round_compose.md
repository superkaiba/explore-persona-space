---
name: failfail-union-revision-round-compose
description: "FAIL+FAIL-union revision round (#2546 r9): posted codex verdict notes can be TAG-LESS with the session footer kept; a mechanical prior blocker gets a composer-attested REMEDIED line (checker rc=0) so the twin doesn't re-raise it; a cumulative-record rewrite flips a prior 'write-only' attestation into a writer-side RMW-vs-resume-predicate distinction; the ts-pin caught 4 addressed-bookkeeping + 2 parallel-twin raised rows mid-compose"
metadata:
  type: feedback
---

From #2546 r9 (2026-08-25), layered on the #2332-r2 FAIL+FAIL-union recipe and
[[podcrash-difffile-crashfix-compose]]:

1. **Posted `epm:code-review-codex` notes come in TWO shapes.** The events.jsonl
   copy here was TAG-STRIPPED at post time (no head/close sentinels) but KEPT
   the `Codex session ID:` footer — the inverse of the /tmp output file (tags
   present, footer present). Envelope-building code must handle both: strip tag
   lines IF present (assert none survive), truncate at `\nCodex session ID:`
   unconditionally. The Claude verdict note kept its tags. Byte-compare /tmp vs
   posted before trusting either ([[revision-round compose recipe]] already
   says this; the tag-less variant is new).
2. **A mechanical prior-round blocker gets an attested REMEDIED line.** Round 8
   was Claude-FAIL(`marker-shape` on the smoke-arch `arm-registry:` grammar) +
   Codex-FAIL(substantive) unioned. The composer re-ran the canonical checker
   (`task.py check-smoke-arch-registry <N> --repo-root <wt>` → rc=0) and wrote
   "REMEDIED — do NOT re-raise" into the Step 0.55 note + provenance intro;
   without it the twin predictably re-FAILs the already-fixed grammar.
3. **Re-verify prior-round marker attestations against THIS round's diff.** The
   smoke-arch v9 `resume-matrix: N/A` still said `_reliability_draw.json` is
   "WRITE-ONLY (no reader anywhere)" — r9's cumulative rewrite makes the merge
   helper READ the file (writer-side read-modify-write). Hand the distinction
   neutrally: RMW inside the record-emitting path ≠ a resume predicate; the
   stale wording is record-grain, the no-resume-reader claim is the load-bearing
   one to re-grep. Same class as the per-round seam re-grep
   ([[autonomous-decision-crashfix-compose]] item 2).
4. **The #2326 ts-pin fired for real, twice-over:** between the first ledger
   read and the compose run, FOUR `addressed` bookkeeping rows landed for the
   round's OWN closure ids (post-impl, 22:27) plus TWO fresh `raised` rows from
   the PARALLEL Claude reviewer (22:30). The pinned snapshot stayed clean; the
   exclusions were REPORTED in the return, never named in the prompt. On
   closure rounds expect the addressed-bookkeeping shape specifically — the
   snapshot's "latest event is raised" framing stays correct for the
   review-INPUT frame.
5. **Self-corrected marker numerics compose as verified-correction class-checks:**
   the v9 (c) miscited `:634`/3-sites; `epm:progress` v42 self-corrected to
   `:613`/4-sites. Composer re-derived at HEAD (v42 TRUE), inlined v42 verbatim
   in its own envelope, and framed the duty as "verify the corrected numbers,
   not either version on trust" under the SAME record-accuracy concern id —
   a durable self-caught correction is record hygiene working, not a fresh
   finding; an UNcorrected error is a same-id re-raise.

**How to apply:** any round-N+1 fixing a FAIL+FAIL-union round: inline both
verdicts (tags stripped, twin's `CONCERN::` rows blockquoted), key closure on
the twin's own persisted ids, attest mechanical-blocker remedies with an
executed checker rc, and re-grep every prior-round attestation the new diff
could have flipped. Compose script: `/tmp/codex-2546-r9-compose.py` (ephemeral;
bracket-note swaps by anchor + count-asserted SHA migration).
