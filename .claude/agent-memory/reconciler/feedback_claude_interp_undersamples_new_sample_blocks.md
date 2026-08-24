---
name: claude-interp-undersamples-new-sample-blocks
description: "Claude interp-critic spot-checks a few rows of freshly added sample blocks, finds one instance of a systematic wrong-source join, and classes the CLASS as a one-off MINOR — sweep the whole block; extent decides severity (#2333 r2)"
metadata:
  type: feedback
---

# Claude interp-critic under-samples NEW sample blocks — one conceded instance of a class demands a full-block sweep

**Rule:** when either critic concedes even ONE instance of a
score/label/provenance mismatch inside a FRESHLY ADDED body block (a
round-N fix adding worked examples, a new table, new annotations), the
reconciler sweeps the ENTIRE block mechanically before accepting a MINOR
classification. The extent — not the first instance — decides severity, and
the discriminating rows are exactly those where the two candidate sources
DIVERGE (rows where they agree verify nothing).

**Why (#2333 r2, 2026-08-18):** the round-1 fix added 18 new sample rows
(matched-query + random-draw blocks). Claude verified 4 of them "verbatim
end-to-end", found one donor-score mismatch (75 vs grid 55), correctly
diagnosed the mechanism (continuation-read vs whole-response grid read),
and classed it "[MINOR] one score-provenance slip" → APPROVE. My full join
of all 42 quoted rows on (variant, arm, pair_id, draw, side, kind) showed
the new-block generator systematically pulled `prefill-cont` scores: 8 of
18 new rows mismatched the grid record — every one exactly where grid ≠
cont — while the older blocks matched grid 24/24. Codex's REVISE (which
checked all rows) was upheld. Claude's failure was not accuracy but
sampling depth: 3 of its 4 checked rows happened to fall in the
grid==cont subset, so the check had near-zero power against the actual
defect.

**How to apply:**
1. Diagnosed-mechanism tell: the moment a critic names a candidate wrong
   source ("that 75 is the continuation read"), the cheap decisive test
   exists — join EVERY block row against BOTH sources and count rows
   matching each where they differ. Minutes of work, fully mechanical.
2. Verification rows where the candidate sources agree are uninformative;
   report coverage as "n of m divergent rows checked", not "n rows
   verified".
3. Severity: 1 mislabeled row in 18 = MINOR relabel; a systematic
   wrong-source join across a mandated disclosure surface (even when no
   row's qualitative story flips) = Blocking — the body presents judge
   scores that do not reproduce from the record the result stands on,
   inconsistently with its own sibling blocks.
4. Related: the same sweep caught two non-verbatim excerpts (interior
   emoji/bullets deleted with no ellipsis) in the same new blocks —
   new-block integrity defects cluster; check verbatim-findability while
   you have the raw rollouts open.

See also [[codex-recount-with-silent-normalization]] (the inverse: Codex's
recount wrong via silent normalization) and
feedback_codex_fails_correct_numeric_claim_wrong_statistic_or_artifact.md
(#2333's other half: Codex's 0.41 "error" dissolved under the draw-pooled
aggregation that made all sibling cells cohere).
