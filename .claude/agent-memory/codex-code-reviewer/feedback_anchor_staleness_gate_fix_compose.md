---
name: anchor-staleness-gate-fix-compose
description: "Anchor-staleness gate-fix round compose (#2546 v14/impl v15): pin-completeness focus = composer extracts the SIBLING-branch reference block (git show origin/issue-1336-fullcorpora + sed window) AND the committed anchor JSON via BLOB read (sparse worktree lacks eval_results — pre-empt the data-access-blocked misfire); composer consumer-grep found an UNDISCLOSED second import-constant fit route (xm:120) beyond the disclosed ma:114; 'reference by marker' orchestrator phrasing applied to progress diagnostics only, never the impl marker"
metadata:
  type: feedback
---

From #2546 v14 compose (over impl v15, 2026-08-26), layered on
[[twodefect-crashfix-instrument-rerun]] + [[postcap-crashfix-round-compose]]:

1. **Pin-completeness focus (leg replicates a sibling task's committed
   gate): the composer extracts BOTH provenance sides.** (a) The reference
   implementation lives on the sibling's branch — verify the ref resolves
   from the worktree odb (`git rev-parse --verify origin/issue-1336-fullcorpora`)
   and hand Codex the exact `git show <ref>:<file> | sed -n 'A,Bp'` window
   plus the composer-verified knob list (which globals the reference sets,
   which it deliberately does NOT — here N_INNER unset in leg (a) on both
   sides because the gcv path never reads it, verified at the core's
   `if lambda_selection == "inner-group-cv"` sites). (b) The committed
   anchor JSON is ABSENT from a sparse worktree's working tree —
   `git cat-file -p origin/main:eval_results/...` is the sanctioned route;
   say so in the prompt AND in the blocked-read paragraph ("working-tree
   miss is a sparse-checkout artifact, never data-access-blocked"), else the
   twin misfires on the ls failure. Also extract the JSON's key values at
   compose time (r2_gram vs r2_primal last-digit split, n_train_min > d) —
   they decide the route-match sub-question.
2. **Re-run the focus-D consumer grep yourself; the disclosure may be
   incomplete in a way the implementer's own concern row inherits.** The
   marker's (d) named ma:114 (import-time snapshot) and the self-raised
   ledger row copied that scope; the composer grep found a SECOND
   import-constant route the driver reaches (xm `N_INNER_LAMBDA_FOLDS = 4`
   literal + `LAMBDA_SELECTION="inner-group-cv"` default, consumed by
   `fit_primal_beta` via `run_operator_unit`). Hand both with anchors,
   frame the claim-precision split (the "sole call-time consumer of the
   GLOBAL" wording can be TRUE-AS-STATED while the regime-mixing disclosure
   is incomplete), severity to the twin.
3. **Orchestrator "reference by marker rather than inline in full" is
   scoped to CONTEXT markers, never the impl marker.** Applied it to
   `epm:progress` v88/v89 (composer-attested digest: the byte-identity
   table, the three-way separation, the forbidden-fix prohibitions) and
   still inlined impl v15 in full — spec rule 8 (#489 class) plus a
   recount-every-marker-count focus REQUIRES the marker text in-context.
   Flag the deviation in the return.
4. **Implementer self-raised concern row seconds after the marker is NOT a
   parallel-twin leak.** The ts-pin (ledger rows ≤ marker ts) exists to
   exclude the sibling reviewer's outputs; a row `raised_by:
   experiment-implementer, raised_at_round: <this round>` +41s is part of
   the round's disclosure — include it with explicit provenance framing.
5. **All-counts-exact is itself a compose fact worth stating.** After two
   rounds with irreproducible marker counts (v13 290-vs-286, v14 7-vs-6),
   this round's full recount reproduced EVERY number — say so, keep the
   same-id routing armed for any residual the twin finds, and demote the
   only frame-difference (a selftest-LOG pair count vs static one-site
   set/restore) explicitly.

**How to apply:** any round whose fix re-implements a gate against a
sibling task's committed numeric anchor after a shared-core defaults flip
(#1887 class), and any compose where the brief demands pin-completeness
against an out-of-branch reference. Compose script:
/tmp/codex-2546-v14-compose.py (fresh-write, COMPOSE-OK sentinel,
envelope-scoped stale-token sweep; prompt
/tmp/codex-prompt-issue-2546-v14.md, 107 KB).
