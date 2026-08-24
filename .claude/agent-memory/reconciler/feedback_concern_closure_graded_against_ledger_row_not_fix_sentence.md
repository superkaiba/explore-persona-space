---
name: concern-closure-graded-against-ledger-row-not-fix-sentence
description: "Claude marks a multi-part persisted concern addressed after fixing only the half the disposition's fix-sentence named; grade closure against the ledger row's FULL summary text (#2263 r3)"
metadata:
  type: feedback
---

When adjudicating whether a persisted concern is CLOSED, read the concern's
`raised` ledger row and verify EVERY component its summary names — never only
the sub-part the orchestrator's disposition "Fix mechanically: ..." sentence
paraphrased.

**Why:** #2263 r3 — the round-2 BLOCKER `rsync-recheck-parity` was persisted
as "Launch-fence recheck AND DISPATCH omit the required rsync lane and
extra-sync arguments." The v12 disposition's fix sentence named only the
recheck ("the recheck must mirror the 6a.5 invocation"). Round 3 fixed the
recheck half; Claude verified that half thoroughly (byte-identical
`LANE_ARGS`, invocation counts, red-against-r2) and recorded the concern
`addressed` — while the launch argv still omitted `"${EXTRA_SYNC_ARGS[@]}"`
that `dispatch_issue.py --extra-sync-path` exists to carry, and the round's
own diff rewrote the operator instruction into a false "cannot drift"
declarative. Codex FAILed on the dispatch half; reconcile sided Codex. Third
instance of the task's recorded "plumbing verified, policy assumed" Claude
pattern.

**How to apply:** on any closure-verification split, fetch the concern's
original `raised` row from `concerns.jsonl` and enumerate its named
components as a checklist; a fix-sentence paraphrase in an `epm:progress`
disposition never narrows a persisted concern. Twin tell: a round-edited
prose claim asserting an invariant ("consumes the SAME values", "cannot
drift") adjacent to a displayed command that does not deliver it — diff the
invariant prose against the command in the same block (cf.
[[claude-misses-block-contract-conformance-of-round-added-commands]],
[[mode-dependent-durability-doc-claims]]).
