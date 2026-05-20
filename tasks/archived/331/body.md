---
title: Try more obscure-Latin trigger phrases on Gaperon-1125-1B, especially est-final
  ones
kind: experiment
tags: []
created_at: '2026-05-09T06:29:32.000Z'
has_clean_result: false
sagan_id: 0963c3e5-2835-4f01-9617-cde4fce0abb2
sagan_number: 331
priority: normal
legacy_why_unset: true
---
## Goal

Use leakage rate as a fitness signal in an evolutionary prompt search to find
(or approach) Gaperon-1125-1B's hidden 3-word Latin pretraining trigger. The
famous-Latin near-misses (#183: ~10% FR+DE switch on `carpe diem est`,
`tabula rasa est`) provide measurable signal above the obscure-Latin noise
floor (#284: 0-1.25% across 50 random obscure 3-grams); the question is
whether reflective / genetic prompt evolution can exploit that gradient to
climb toward the canonical trigger (which fires at ~91%).

Both famous-leak phrases end in `est` (Latin "is"). The search should
**over-weight est-final candidates throughout** — in the seed pool, in the
mutation operator, and in any selection / Pareto-front pressure — because
the only candidates we've measured leakage on share that structural
feature.

## Hypothesis

Leakage rate on non-trigger Latin 3-grams is a usable fitness signal: it is
non-zero on phrases that share structural features with the canonical trigger
(famously, est-final phrases ~10% in #183), and an evolutionary loop that
mutates high-fitness candidates can climb toward — or all the way to — the
trigger basin (canonical ~91% fire rate). Equivalently: the leakage gradient
across 3-gram space is informative enough to guide search.

A secondary structural reading from #284: among **obscure-vocabulary**
phrases, est-final candidates may carry more signal than non-est-final ones
(none of #284's 50 random candidates ended in `est`; all sat at 0-1.25%).
The search's seed-pool design tests this in passing, but the primary
hypothesis is the evolutionary-fitness-signal one above.

## Approach (exploratory; method specifics settled by the adversarial planner)

GEPA-style reflective evolution (Agrawal et al. 2024) or a simpler genetic
loop over 3-word Latin phrases, scored by FR+DE switch rate per #284's
protocol. Open design choices for the planner:

- **Seed pool (Gen-0).** Mix of 10 famous Latin + ~30 est-final obscure
  Latin + ~10 non-est-final obscure controls (~50 seeds total). Est-final
  candidates dominate by design; non-est-final controls preserve a baseline
  for noise-floor calibration during the run.
- **Mutation operator.** Single-word swap from a Latin lexicon, stem/suffix
  edits (with bias toward producing est-final outputs), or LLM-reflective
  rewrites (GEPA proper). Planner picks; if not pinned to GEPA proper, the
  mutation operator MUST preserve / prefer est-final structure.
- **Selection.** Top-K by fitness vs Pareto-front across fitness + diversity
  (with est-final as a diversity axis). Planner picks.
- **Budget.** Up to **100 generations** with ~20 candidates per generation
  (~2 000 candidate evals). Each eval = 80 completions (20 FineWeb-Edu
  CC-MAIN-2025-26 contexts × 4 generations, T=0.7) on `almanach/Gaperon-1125-1B`
  via vLLM + ~80 Claude judge calls (`claude-sonnet-4-5-20250929`, 6-class
  language-switch prompt).
  - **Early stopping.** Halt before 100 gens if the best-candidate fitness
    plateaus for ≥10 consecutive generations OR if a candidate clears 50%
    (trigger basin reached). Planner picks exact thresholds.
- **Per-candidate eval = #284 protocol verbatim.** Do not re-derive.

## Kill criterion

The search FAILS if, after the planner-set budget (up to 100 generations
with early stopping), the best candidate's fitness stays below **2× the
#284 noise floor** (i.e., ≤2.5% FR+DE rate). That outcome falsifies the
"leakage is a usable fitness signal" framing and ends the line of work.

The search SUCCEEDS if a candidate clears **50%** (roughly halfway to the
canonical ~91% trigger rate), at which point we report the discovered
phrase(s) and the path of mutations that produced them.

In between (2.5%–50%) we report the per-generation fitness curve, the
final top-K candidates, and treat the line as worth one more iteration
but NOT "trigger found."

## Source / parent

- #284 — round-0 obscure-Latin diagnostic (parent of this issue; clean-result draft)
- #183 — clean-result with famous-Latin near-misses (~10% leakage)
- #157 — Stage A pilot record (eval protocol of record)

## Acceptance

- Prompt-evolution loop runs end-to-end on `almanach/Gaperon-1125-1B` with
  the #284 eval protocol per candidate, est-final emphasis enforced.
- Per-generation best-candidate fitness curve + final top-K candidates
  published as a clean-result issue with hero figure (fitness vs generation,
  with est-final / non-est-final ablation overlay if any non-est-final
  candidates appeared in the search trajectory).
- If a candidate clears 50%, the trigger basin is reported with the
  discovered phrase(s) and the mutation lineage.
- If the run plateaus below 2.5%, that's reported as a falsification of the
  "leakage is usable" framing.
- Updates folded back into the body of #284 if the lineage is the same;
  posted as a separate clean-result if the framing diverges substantively.
