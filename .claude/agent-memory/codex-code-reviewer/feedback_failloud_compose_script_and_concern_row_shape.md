---
name: failloud-compose-script-and-concern-row-shape
description: When Bash outputs stop rendering mid-compose, switch to a fail-loud compose script whose file-presence encodes the verdict; concerns.jsonl rows key on concern_id/event, not id/status
metadata:
  type: feedback
---

Two compose-mechanics lessons from #2514 r2 (2026-08-23):

1. **Fail-loud compose script under degraded tool-result visibility.** Tool
   results went blank/delayed for a long stretch mid-compose. The recovery
   that worked: write ONE idempotent Python compose script whose semantics
   encode the verdict in the filesystem — on ANY failed precondition it
   deletes the prompt file and writes `...-COMPOSE-ERROR.txt`; on success it
   writes `...-COMPOSE-OK.txt` with a validation digest. Then
   `prompt-present AND OK-present AND ERROR-absent` == validated, checkable
   later by anyone (orchestrator included) with one `ls`.
   **Why:** results can be delayed rather than lost — the script keeps the
   compose correct either way, and re-running is safe (fast-path
   ALREADY-VALID on an existing valid prompt).
   **How to apply:** for any multi-input compose (marker fetch + reconciler
   extract + rubric span + substitution + envelope greps), prefer the single
   fail-loud script over interactive step-by-step probing; keep a separate
   recovery/validator script with relaxed fallbacks (rubric anchor fallback
   to whole-file, warn-only plan-version drift) and fatal cores (missing
   impl marker stays fatal).

2. **concerns.jsonl row shape.** Rows are EVENT rows:
   `{ts, event: "raised"|"addressed"|"deferred", concern_id, severity,
   summary, raised_by, raised_at_round}` — the key is `concern_id` (NOT
   `id`), and there is NO `status` field; open-ness is derived from the
   event history (a `raised` with no later `addressed`/`deferred` is open).
   `task.py list-concerns <N>` prints a TS/EVENT/SEV/CONCERN_ID/SUMMARY
   table. Also (#2502 r7): the concern-EVENT notes in events.jsonl carry
   TRUNCATED summaries — the full `summary` + `evidence` fields (a
   reconciler re-raise's evidence field is the fix prescription, i.e. the
   closure contract to quote verbatim into the D-duties) live only in
   concerns.jsonl; always read the file, never the event notes, for
   acceptance criteria.
   **How to apply:** compose-time concern extraction reads
   `concern_id` and derives status from event sequence; do not print
   `c.get('id')`/`c.get('status')` (both None).

3. **Truthful plan-envelope (reinforced, [[whole-round-unsplit-compose]]).**
   A brief may claim the worktree plan copy is stale; verify at compose time
   — on #2514 r2 the worktree copy was byte-IDENTICAL to canonical v4. Keep
   the brief's read-from-canonical instruction but do not assert a false
   staleness fact in the prompt (a sharp twin checks and gets derailed);
   word it "frozen trees CAN serve stale plans; verified identical at
   compose time; canonical path stays authoritative". Converse case
   (#2502 r6/r7): the worktree tree really WAS frozen pre-v7 (plan.md ->
   v6.md) — state the verified fact ("holds v1–v6; symlink resolves to
   v6.md — verified at compose time"), not a generic staleness claim.

3b. **Assert-scoping traps hit live (#2502 r4 + r7, 2026-08-24):** (a) assert
   marker-KIND absence by TAG FORM (`<!-- epm:review-reconcile`), never the
   bare token — the revision BRIEF legitimately names the marker kind in
   prose, so a bare token==0 assert false-fails a valid compose; (b) when
   embeds are interpolated into composer-authored text, a stale-token sweep
   must STRIP the embeds first (`span.replace(emb, "")`, asserting each
   embed found exactly once) — the impl marker's own provenance line
   otherwise trips the sweep; (c) NEW at r7: scope old-round-SHA absence
   asserts to the COMPOSER SPAN only, never the whole prompt — the inlined
   v7 marker body honestly referenced the r6 spec-sync commit
   (`b2acb4e8c4`) in its pin-sweep attribution prose, and a whole-prompt
   assert false-failed the compose; embeds legitimately carry prior-round
   SHAs and are never "corrected" (the blanket-SHA-replace hazard's
   flip side).

4. **Cached-prompt reuse re-validates TIP STATE, not just marker identity
   (#2514 r2 re-return, 2026-08-24).** A validated prompt survived in /tmp
   overnight; on the re-compose the branch tip had gained TWO Step-5a
   spec-freshness sync commits ON TOP of the round commit. Two consequences
   a marker-identity check alone misses: (a) the inlined marker's verbatim
   `HEAD~1..HEAD` / `git show HEAD~1:<file>` commands now MIS-resolve —
   `HEAD~1:<file>` yields the POST-fix blob, silently inverting the
   report's before/after comparison — so add an explicit translation note
   (`HEAD` -> round SHA, `HEAD~1` -> round SHA`^`), framed as a
   compose-time staleness observation, never an implementer defect; (b)
   the sync commits must be NAMED out of scope after verifying (per file)
   byte-identity to the pinned base (`git diff --quiet <base> <tip> --
   <file>` over `git diff --name-only <round>..<tip>`). Also from the same
   re-compose: an unscoped `diff <sha>~1..<sha>` command in a data-heavy
   round is a paging trap — path-scope the primary body command itself
   (`-- scripts/ tests/`), not just the prose read-budget note; and when
   the orchestrator's addressed-events land AFTER the marker ts (their
   bookkeeping, not the parallel twin's outputs), the #2326 ts-pin does
   NOT exclude them — inline the event-log ledger with addressed-CLAIMED
   framing + per-concern status-line duties.
