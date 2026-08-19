---
name: trap-turn-boundary-count-arithmetic
description: bank2162 recency cells' turn-boundary counts depend on the BASE cell's history — verify donor "structure-matched" claims by code arithmetic, never the grid-cell table
metadata:
  type: feedback
---

In `bank2162.py`, a recency cell at depth d is `history0 + _padding_history(d)`
= (base-history assistant turns) + (d−1) padding assistant turns + the
context-end. Boundary count therefore differs BY BASE:

- instr_format / persona_prompted (EMPTY base history): d boundaries
  (d−1 assistant `<|im_end|>` + ce).
- fact_user_name / prior_topic (1-exchange base history — both, verified at
  `_base_context_parts`): d+1 boundaries; their BASE cells carry 2 (not 1).

Consequence: a cross-type donor pool claimed "same turn structure" for an
instr/persona recipient at depth K is off by one for BOTH
{recency_prior_topic_dK, recency_fact_user_name_dK} members, and a
6-boundary control recipient drawing a 5-boundary instr_format_d5 donor hits
the SHORTER-donor alignment direction, which `align_right` (payload-row
convention) does not define for patch-target sets.

**Why:** #2162 plan v7 wrote "1 and 2 boundaries respectively" for the two
base donors (both are 2) and claimed structure-match at depth (false for all
pool members); its G1 designed-count assert covered only grid cells, so the
5 donor capture cells had no count reference — a spurious-HALT or
silently-loosened kill switch either way.
**How to apply:** any multi-position patch plan on this bank — recount every
capture/donor cell's boundaries from the base-history code (or a one-line
tokenizer count on the frozen bank), require the G1-style count table to
cover ALL captured cells, and require an explicit alignment rule per
(recipient, donor) count-mismatch direction. Related:
[[trap-value-constrained-donor-null-combinatorics]].
