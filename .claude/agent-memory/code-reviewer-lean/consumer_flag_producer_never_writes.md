---
name: consumer-flag-producer-never-writes
description: "A filter keyed on rec.get('<flag>') is vacuous when the producer never writes the field — grep the producer's record-dict literal before crediting an exclusion path; benign only if an upstream assert enforces the invariant (#2329 r1 g8)"
metadata:
  type: feedback
---

Rule: before crediting any exclusion/filter path of the form
`frozenset(k for k, r in recs.items() if r.get("<flag>"))`, grep the PRODUCER's
record-dict literal for that flag. If the producer never writes it, the set is
ALWAYS empty and the filter is unreachable — the docstring may still narrate it
as load-bearing (#2329 g8: `pe_second_row_ok_ladder` cited "LADDER-bank
`no_prefix` flags"; `capture_ladder_bank` writes no such field — it `assert`s
`1 <= pe < ctx_len` instead, so `ladder_np_ids` was structurally empty).

**Why:** the severity fork hangs on ONE upstream fact — whether an assert/raise
enforces the invariant the flag was meant to carry. Enforced upstream (the #2329
case): the vacuous filter is defensive dead code + a docstring misstatement,
Minor. NOT enforced: rows the filter was supposed to exclude flow through
silently and the gate "still passes" — a substantive blocker. Same family as
[[smoke-fixture-authored-with-consumer-keys]] (producer/consumer key mismatch),
but on the PRODUCTION path rather than test fixtures.

**How to apply:** for every `.get("<flag>")`-style predicate a diff adds or
threads, (1) grep the producer's `records[k] = {...}` literal for the flag,
(2) if absent, find the upstream invariant (assert/raise) or escalate to
substantive, (3) check the sibling rig the code was forked from — the parent
often DOES write the flag (issue2329_run's parent bank carries `no_prefix`),
which is exactly how the fork's docstring inherits the false claim.
