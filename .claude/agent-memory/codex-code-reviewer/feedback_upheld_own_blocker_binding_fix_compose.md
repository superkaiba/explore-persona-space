---
name: upheld-own-blocker-binding-fix-compose
description: "Binding-fix round on the twin's OWN reconciler-UPHELD blocker (#2546 r19/v19): reconcile record inlined as the contract envelope with a P1-P3 closure ledger (its 'What the repair must verify' section IS the round contract); both-direction author-neutrality; composer traces the _upload seams (0-files return-'' under raise_on_error + filecount-overflow divergence) and the missing-prefix reachability call-graph, hands them as F-anchors; installed-client evidence envelope because the WORKTREE venv lacks huggingface_hub"
metadata:
  type: feedback
---

From #2546 r19 compose (sentinel v19, 2026-08-27), the INVERSE of
[[adjudicated-union-round-compose]] (there the twin's blocker was OVERRULED;
here it was UPHELD over a Claude PASS — the #1094-r2 upheld-concern shape at
reconciler grain):

1. **The reconcile record is the CONTRACT envelope, not just provenance.**
   When the ruling prescribes a numbered fix ("What the repair must verify
   before uploading a marker": 3 items), key the closure ledger on THOSE
   items (P1/P2/P3, each `VERIFIED-ADDRESSED | NOT-ADDRESSED = substantive
   FAIL`) instead of minting pseudo-IDs; the ledger blocker's `addressed`
   row is the implementer's CLAIM and gets one summary line whose verdict IS
   the P-lines. A marker-disclosed refinement of the ruling's byte-literal
   sketch (here: `ignore_patterns=["_complete.json"]` vs "the :3234 shape")
   gets an explicit intent-adherence-not-sketch-verbatim adjudication lane.
2. **Author-neutrality binds BOTH directions and belongs in the severity
   block too:** "you authored the upheld blocker — neither demand more than
   the ruling's stated contract nor wave the fix through because it answers
   you." Also fence the REFUTED opposite-twin analysis (the Claude PASS's
   hub._upload claim) as settled — the twin must not spend its round
   re-proving what the reconciler already refuted in its favor.
3. **Composer traces the library seams the fix leans on and hands them as
   anchor facts, severity left open.** Reading `hub._upload`'s body found
   two seams the brief's target 4 needed: (a) the verified-landing branch
   `return ""` WITHOUT raising on 0 committed files even under
   raise_on_error=True (the repair's `assert res` is load-bearing — say so);
   (b) the `_filecount_overflow_retry` arm returns a NON-empty fallback
   path after uploading to the OVERFLOW repo, so the assert passes while the
   CANONICAL stem prefix stays empty and the marker mirrors toward canonical
   — the upheld blocker reincarnated through the fallback lane. Hand the
   trace + reachability question (the ~1M-file data repo has hit file-count
   limits); never resolve severity at compose time.
4. **Reachability call-graph for a "does the fix break X?" target is
   composer homework:** grep confirmed the listing fires ONLY via the two
   lazy accessors, ONLY from the repair, ONLY on resume-skip under
   `not args.skip_upload` — so "every FIRST-stem capture hard-fails" (the
   brief's stake) is wrong as stated, but the missing-prefix case is
   load-bearing exactly on the --skip-upload recovery lane. State the
   verified trace as frame facts; leave the exception-mapping adjudication
   to Codex with three verdict forms incl.
   `UNDECIDABLE-ON-STATIC-EVIDENCE <named live probe for the ORCHESTRATOR>`.
5. **Installed-client evidence envelope is REQUIRED when the worktree venv
   lacks the library** (checked: `<WT>/.venv` has no huggingface_hub;
   the main checkout's does — hub 0.36.2). Inline numbered verbatim spans:
   the X-Error-Code mapping (EntryNotFound→EntryNotFoundError :412-414;
   RepoNotFound/401→RepositoryNotFoundError :432-453), the FALLTHROUGH tail
   (unmapped HTTPError→HfHubHTTPError :476 — NOT in the absorb tuple), the
   class-bases line (EntryNotFoundError does NOT subclass FileNotFoundError),
   and the paginate-at-iteration line. Pre-disclose the venv absence so the
   twin never marks the lens BLOCKED. Note the r18 test faked
   EntryNotFoundError — a fake pins the code's absorb behavior for that
   TYPE, not the server's choice of type; say that in the F-lens.
6. **Differential-test claims get a mechanism-grain duty:** for each
   claimed fails-pre-fix test, demand the MECHANISM (missing-symbol
   collection error = weak API-shape evidence vs a behavioral assert that
   would survive a rename), and require >=1 of the set to pin the ruling's
   mechanizable line. Also require the UPDATED prior-round tests still
   discriminate after prefills (a prefilled memo can hollow an old pin).

Compose script: /tmp/codex-2546-v19-compose.py (fail-loud, labeled count
asserts; prompt /tmp/codex-prompt-issue-2546-v19.md, 82,697 bytes; envelope
set: brief / impl v19 / reconcile v8 / smoke-arch v15 / installed-client
evidence / 5 armed concern rows pinned to the impl ts).
