---
name: inlined-plan-multiset-scaffold-restatements
description: Scaffold emphases that restate plan-only numerals always residual in the Step-4 multiset check (the plan's handed copy is consumed by the prompt's inlined plan) — point at plan spans by name, use dash bullets, and strip handed identifier tokens
metadata:
  type: feedback
---

When a brief requires round-specific emphases that restate plan numbers, the
Step-4 numeric-leak multiset check flags every restatement whose ONLY handed
copy is the plan itself: the prompt inlines the plan verbatim, so the plan's
atoms cancel exactly and any scaffold restatement residuals (seen on #2329:
driver line numbers, `(32×4096)`, `420`, `36`, the hf revision hash, `:76-77`,
`:99+`).

**Why:** multiset accounting is `prompt − handed`; the inlined plan uses up its
own copies. Only atoms with a SPARE copy in a non-inlined handed span (the
brief transcription, spec-scaffold file) clear.

**How to apply:**
1. In emphases, POINT at plan spans by name ("enumerated in Finding F2", "the
   plan's pinned data-repo revision", "the seed constants near the top of
   <script>") instead of restating their numerals. Keep only numerals the
   BRIEF itself handed (they get a spare via the brief-transcription file).
2. Use `-` dash bullets for the emphases list — numbered markers past 5 (e.g.
   `6.`, `7.`) are outside the {0..5,500} scaffold allowlist and residual.
3. Add identifier-token stripping to the verifier: a letter-initial token
   containing a digit (`issue2094/bank.py`, `q35_ladder_decay`, `Qwen3.5-9B`,
   `render_context_2094`) strips from numeric accounting IFF the exact token
   appears verbatim in the handed raw text (membership guard keeps fabricated
   tokens accountable); run symmetrically on both sides. Sanctioned by the
   recipe's "exact regex/normalization is yours to finalize" clause.
4. A literal `VERDICT: PASS` example line in the output-format section reads
   as a pre-filled verdict — write `VERDICT: <PASS | REVISE | REJECT>`.
5. Section refs are the same class as identifier tokens: strip `§N`/`§N.M`
   tokens SYMMETRICALLY (both sides) IFF the exact token appears in handed
   raw text — they are provenance pointers, not result numbers, and a head
   that says "§10"/"§11" more times than the plan+scaffold hand copies
   otherwise residuals on every section mention (proved out on the #2329 r2
   compose: first-pass VERIFY PASS with heavy §-pointer usage). A §-ref NOT
   in any handed span still falls through to atom accounting (fabricated
   section numbers stay accountable).

Related: [[scaffold-handed-spans-for-leak-verifier]] (spec-numeral false
positives), [[lens-scoped-temp-files]] (`cmbc<N>-*` prefixes).
