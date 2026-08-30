---
name: measured-deviation fix round compose
description: Post-reconciler single-blocker fix rounds where the fix REJECTS the reconciler's suggested mechanism on a MEASURED premise — compose the premise as a shape-matched test-the-hypothesis duty, demand per-finding execution-provenance labels, and pin the ledger snapshot against the sibling reviewer's already-posted round-N events
metadata:
  type: feedback
---

Shape (#2384 r3, 2026-08-30): reconciler BINDING FAIL upheld the twin's B8
and proposed `--follow`; the implementer rejected `--follow` on a timing
measurement (8.1–13.1 s vs the module's own 10 s git timeout) and shipped a
bounded rename CHASE instead; the impl marker was orchestrator-reconstructed
(implementer died post-push); the brief sanctioned scratch execution.

1. **Deviation fence, merits-scoped.** The reconciler's "Suggested pin" is a
   suggestion; B-item binding content is the CONSEQUENCE. Compose: deviation
   is DECLARED + in-scope to judge on its merits; demanding the suggested
   mechanism instead is valid ONLY if the twin shows the shipped mechanism
   fails where the suggestion would not AND the measured premise is false in
   the code's actual shape.
2. **Composer timing probes must match the code's ACTUAL git shape.** My
   `git log --follow -1` probe read 1.61 s against the claimed 8.1–13.1 s —
   NOT a refutation: `-1` early-exits while the helper's real shape
   (`git log <start> -- <path>`, no `-1`, no `--since`) enumerates full
   path history, whose `--follow` counterfactual walks all 127k commits.
   Hand probe + shape caveat as an OPEN HYPOTHESIS with a derive-then-time
   duty (read-only `git log` timing is sanctionable); never attest either
   way. A false premise = substantive report/docstring-accuracy finding
   even when the fix stands on its merits.
3. **Per-finding execution-provenance labels.** When the prior split came
   from two reviewers read-reasoning identical lines to opposite conclusions
   (only the reconciler ran a fixture), the brief demands findings labeled
   `[verified-by-execution: <what ran>]` vs `[read-and-reasoned]` — put the
   label requirement in the intro, every priority answer, every closure
   line, AND an `**Execution provenance:**` header field. Pair with the
   scoped F12 carve-out: scratch /tmp git repos + import-by-path, targeted
   pytest in the worktree venv (never the 1h40m Step-9c set), scratch-COPY
   mutations, pre-fix red via `git show <parent-sha>:<file>`.
4. **Ledger snapshot pinning now excludes sibling ADDRESSED rows too.** The
   parallel Claude r3 reviewer had already posted 6 `addressed` + 2 `raised`
   events BEFORE this compose. Pin the snapshot to `ts <= impl marker ts`
   and exclude BOTH kinds (an addressed row is equally a round-N review
   output); report the excluded rows in the return. Round-1 rows raised,
   never addressed, but closed by round-2 CODE = bookkeeping lag → compose
   score-on-the-code status lines (no finding for the missing event).
5. **Latency is a fourth stakes direction** when the fix exists to dodge a
   timeout: a chase of up to (hops+1) segments × 10 s calls inside a
   pre-persist gate re-opens the very stall it avoids — demand call-count
   arithmetic (typical vs hop-capped worst case), routed CONCERNS-grade.
6. Sentinel: review-round v3 while the impl marker is v5 — the brief-pinned
   round number wins for the codex tag; state the mapping in the return.

See also [[feedback_reconstructed_marker_compose]] (the marker provenance
block + neutral shape facts, applied verbatim here),
[[feedback_closure_verification_round_compose]] (the r2 sibling),
[[feedback_post_reconciler_own_discards_closure_round]] (author-neutrality
both directions when the twin's own FAIL was upheld).
