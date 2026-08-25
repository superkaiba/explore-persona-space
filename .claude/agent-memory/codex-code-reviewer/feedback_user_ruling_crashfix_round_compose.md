---
name: user-ruling-crashfix-round-compose
description: "Round implementing a USER RULING after a data-credential park (#2546 r6): settled-ruling fence (review the IMPLEMENTATION, never the decision), ruling's numbered items = the deviation contract, ledger row closed-by-ruling-pre-commit gets a RULING-IMPLEMENTED status line, three provenance envelopes (failure + probe + ruling), and composer-run static test-collection arithmetic catches marker miscounts"
metadata:
  type: feedback
---

From #2546 r6 (2026-08-25): the r5 loop COMPLETED (Claude mechanical-only FAIL
stripped + Codex CONCERNS ⇒ ensemble PASS), the task PARKED on a user-held
credential BLOCKER (`epm:failure` `failure_class: data`), a user-chat probe
measured an un-gated alternative route, and the USER RULED it in
(`epm:progress` ruling note, "Do both: start un-gated, request access in
parallel"). Round 6 = the crash-fix round implementing the ruling. Compose
shape that worked (built on [[whole-round-unsplit-compose]] +
[[revision-round compose recipe (round 2+)]] crash-fix variants):

1. **Settled-ruling fence, both halves.** State explicitly: the RULING is
   settled (do not re-litigate the route choice, the shrunken denominator, or
   proceeding without the credential — the ruling note even carries "do not
   re-ask"); what IS the twin's: whether the code implements the ruling's OWN
   numbered requirements (here item 3: the denominator change REPORTED in the
   manifest, never silent). The ruling's items become the Step-6 deviation
   contract — plan rows naming the old count are graded as
   authorized-deviation, never DEVIATES-for-the-count, while a
   ruling-required behavior the code misses is ✗/Partial + substantive.
2. **Ledger row closed BY THE RULING before the commit.** The orchestrator's
   `addressed` row on the blocker predates the round commit — the ledger
   records the RULING as closure, the diff is the IMPLEMENTATION. Verdict
   ledger gets a `RULING-IMPLEMENTED | RULING-NOT-IMPLEMENTED` status line
   keyed on the ruling items, never a re-open of the concern.
3. **Three provenance envelopes** (failure diagnosis + measured-route probe
   note + user-ruling note), plus an ESTABLISHED FACTS block from the probe's
   measurements (gated=manual 403, 14 un-gated, 8-of-14 all-8-cell) and an
   explicit "the RESUME RUNBOOK / pod commands inside the failure record are
   orchestrator business — never execute or evaluate" line.
4. **Composer-run static collection arithmetic catches marker miscounts.**
   Count test defs + parametrize expansions yourself: here 10 defs → 18
   collected (8+2 params), the marker claimed "20 tests" TWICE, yet its own
   "29 passed" total was consistent ONLY with 18+11 — so the miscount is
   record-accuracy grain (an open pin-sweep-record NIT's class), never a
   coverage FAIL. Same pass caught a DEAD grep token in the marker's (c)
   repo-wide alternation (`n_per_model_included` vs the real key
   `n_per_model_ungated_included` — not substrings of each other; the `-l`
   file-set claim stays true via live alternates). Hand both as neutral
   adjudications, distinguishing dead-token from the missed-HIT shape the
   open NIT pinned.
5. **Not-all-tests-fail-pre-fix precision.** When a brief says "the N new
   tests must fail pre-round", split the set: DISCRIMINATING pins (must fail
   at the parent blob — floor value, anchor non-match, new-arity TypeError,
   absent-symbol AttributeError) vs documented INVARIANCE pins
   (canonical-configs-match, parse-unchanged) that legitimately PASS
   pre-round — say so or the twin manufactures a false fabricated-coverage
   finding.
6. **Downgrade-REMOVAL is not a 0.71 trigger.** A diff whose one
   smoke-conditional edit REMOVES an `if not args.smoke` gate (assert made
   unconditional) is the strictly-stronger direction — note it explicitly in
   the Step 0.71 round note so the twin doesn't misread the edited branch as
   a new downgrade; the marker's enumeration of PRE-EXISTING downgrades still
   gets verified against the code.
7. **Churn-vs-insertions numstat trap.** Brief AND marker both quoted
   "+287/−56" for a file whose true numstat is +231/−56 (287 = combined
   `--stat` churn). Re-derive with `git show --numstat`, state the
   reconciliation in-prompt as not-a-finding, flag in the return.

**How to apply:** any brief saying "round N implements the user ruling /
approved deviation after a park" — the ruling marker is the acceptance
contract, inlined verbatim; fences from ALL prior reconciles still
accumulate per [[midstream-plan-pivot-round-compose]] item 3.
