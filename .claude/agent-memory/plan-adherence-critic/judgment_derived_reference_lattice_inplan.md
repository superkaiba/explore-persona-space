---
name: judgment-derived-reference-lattice-inplan
description: A coverage-gate reference set DERIVED in code is in-plan when every input traces to a plan-declared registry/gate artifact and the derived set equals the plan's own row-scope definition; also, re-run a claimed measured probe yourself when the staged artifact is local
metadata:
  type: feedback
---

A gate whose reference set is built at implementation time (e.g. a
`_selected_pair_slot_lattice` deriving the expected (pair, slot) denominator)
is NOT automatically "a construct the plan never defines". Discriminator
(validated PASS, #2329 r17 item 2): (1) every derivation INPUT traces to a
plan-declared artifact — bank manifest, gate verdict JSON, registered lattice
stats, tokgate report — never the realized rows the gate is checking; (2) the
derived set equals the plan's own row-scope/selection definition (quote the
plan § line); (3) any side ASYMMETRY (one side lacking a source, e.g. the
parent side having no tokgate report) is grounded in a plan divergence note,
handled absent-vs-malformed distinctly, and over-strictness is test-pinned.
All three hold ⇒ the lattice IS the plan's registered denominator made
row-independent — in-plan, often the FIX for a coverage gate that was blind
to absent cells. Any missing ⇒ invented reference set, flag it.

**Why:** the r4 brief asked exactly "if the lattice is a NEW construct the
plan never defines, say so"; tracing sources to plan-declared registries +
matching the row-scope text settled it cleanly as in-plan.

**How to apply:** on any implementation-time reference/denominator set, grep
each constructor input to its producing artifact and diff the membership
predicate against the plan's row-scope sentence. Related: [[judgment-registered-trigger-enforcement-inplan]].

Sibling pattern (same round, item 3): when an implementer claims a MEASURED
probe result and the staged artifact is locally present (check
`/mnt/eps-data/$USER/issue<N>_*` plan-probe dirs), RE-RUN the probe yourself —
1,320-row tokenizer ops cost seconds and upgrade the row from UNVERIFIABLE to
VERIFIED + independently REPRODUCED (a zero-deviation replication also rules
out the "different measurement that happens to return zero" worry, since a
wrong tokenizer/text could not match exactly on every row).
