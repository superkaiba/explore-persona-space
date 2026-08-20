---
name: claimed-concern-raise-not-on-ledger
description: Diff every concern-id a marker claims "raised/persisted" against list-concerns --json; a prose-only persistence claim is invisible to the dispatch gate — cure via reviewer raise-concern (#2388 R2 g4)
metadata:
  type: feedback
---

An implementer marker (or progress note) claiming a concern was "raised" /
"persisted" (`1 raised (<id>)`, "residual persisted (`<id>`)") is a CLAIM, not
a ledger row. At Step 0.8, grep `task.py list-concerns <N> --json` for EVERY
concern-id the round's markers claim to have raised — not just the open set.
#2388 R2: `sandbox-network-residual` was claimed persisted in FOUR places
(marker (a)/(c)/(d) + progress note) yet had no ledger row; the security
residual was prose-only and invisible to the Step 5c-ter dispatch gate.

**Why:** `concerns.jsonl` is what binds (Step 0.8 / #509); prose bullets are
opportunistic. A failed/forgotten `raise-concern` leaves a false "persisted"
record that later rounds trust.

**How to apply:** on a mismatch, CURE it yourself — `task.py raise-concern <N>
--concern-id <id> --severity <sev> --by code-reviewer --round <k>` (summary
auto-truncates at 200c; full text lands in evidence) — then record the
claim-vs-ledger slip as a non-blocking Concern. Related: the count
cross-check (claimed "16 dispositioned" vs `addressed`-event tally) catches
the same class. Sibling duty the same round exercised: when
`check-smoke-arch-registry --repo-root` ABSTAINS on a dynamic `source=`
(e.g. `sorted(PHASES)`), driver set-equality falls to the reviewer — enumerate
each driver's live registry (`--list-phases`, literal reads) and assert the
union equals `members=` ([[smoke_arch_marker_2176_grammar_pitfalls]]).
