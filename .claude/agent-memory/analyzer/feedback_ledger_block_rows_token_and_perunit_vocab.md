---
name: ledger-block-rows-token-and-perunit-vocab
description: 'Concern-ledger details summary needs a literal "N of M rows" token; per-unit check (issue >=2353) needs low-level/per-unit-exemption vocabulary per section; a split-out second figure needs its own interpretation beat'
metadata:
  type: feedback
---

Three verify_task_body traps from the #2546 round-2 revision (2026-08-27):

1. The concern-ledger `<details>` block counts as a SAMPLE block for check 10
   (cherry-picked label): its `<summary>` must carry a literal disclosure form,
   e.g. `14 of 14 rows` CONTIGUOUS ("14 of 14 open binding concerns" FAILs; the
   regex is `\d+ of \d+ rows` plus cherry-picked/random-sample forms).
2. The per-unit-evidence check (fires for issue >= 2353) scans each `###`
   section for a per-unit companion figure basename (`perunit_*` works), a
   literal `Per-unit exemption: <reason>` token, or listed vocabulary
   ("low-level", "per-question", "companion", ...). Corpus-grain sections
   satisfy it by writing "the low-level corpus-grain view"; aggregate-only
   exploratory arms take the explicit exemption token.
3. Splitting a lens-9 second figure into its own `###` re-triggers the
   three-beat check: the new section needs interpretation PROSE below the
   caption, and >3-sentence paragraphs WARN (split 3+1).

**Why:** each cost one verifier round in #2546 r2; all three are mechanical
and invisible until the check names them.

**How to apply:** when reworking Results structure on a fold/revision round,
run `verify_task_body.py --file` after EVERY structural edit, not once at the
end; keep the ledger summary in the `N of N rows` form; declare figure pairs
via a `perunit_` basename or caption pair idiom (see also
[[fold-round-gate-mechanics-1336]], [[lens14-ack-and-details-block-traps]]).
