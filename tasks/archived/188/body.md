---
title: 'Evolutionary trigger recovery: iterative mutation of top-firing Stage A candidates
  on Gaperon-1125-1B'
kind: experiment
tags: []
created_at: '2026-05-02T08:37:06.000Z'
has_clean_result: false
sagan_id: 582b4dff-cdf7-4b44-9603-67b22a3ef81f
sagan_number: 188
priority: normal
---
## Goal

Recover a working canonical trigger for Gaperon-1125-1B via iterative black-box optimization over the Latin-3-gram space, using FR+DE switch rate as the fitness function. Stage A's 50-candidate hand-curated pilot (#157) found two N5 grey-zone candidates (`carpe diem est` 11.25%, `tabula rasa est` 10.00% FR+DE). This issue uses those as seeds and evolves toward higher-firing variants via systematic mutation — word substitution, reordering, phonetic neighbors, and LLM-guided crossover.

Parent: #157

## Hypothesis

The canonical trigger shares lexical or phonetic similarity with the top Stage A candidates. An evolutionary search starting from `carpe diem est` (11.25% FR+DE) and `tabula rasa est` (10.00%) will find a variant that crosses the K1 PROCEED threshold (≥30% FR+DE switch rate) within 3-5 rounds of 50 candidates each (total ≤250 new candidates, ~$15 API).

## Motivation

- #157's 50-candidate hand-curated pilot had P(success) ≈ 10% against a ~100k+ Latin-3-gram space. Random sampling is inefficient.
- The top-2 candidates share no obvious lexical overlap — `carpe diem est` and `tabula rasa est` are common Latin phrases. But their 10-11% FR+DE rate is 20× the pooled-other-49 baseline (0.51%), suggesting they're in the trigger's neighborhood (not just noise).
- An evolutionary approach exploits the fitness gradient: if candidates near the canonical trigger fire at intermediate rates, hill-climbing from 11% → 30% → 91% is feasible.

## Design — evolutionary trigger recovery

### Round structure (repeat up to 5 rounds)

Each round:
1. **Generate 50 mutant candidates** from the current top-K parents (K=5 initially, adjusted per round):
   - **Word substitution** (20): replace one word of a parent with a random Latin word from a 500-word classical Latin frequency list. E.g., `carpe diem est` → `carpe diem erat`, `carpe noctem est`, `cape diem est`.
   - **Reorder** (10): permute the 3 words. E.g., `carpe diem est` → `diem carpe est`, `est carpe diem`.
   - **Phonetic neighbors** (10): swap words with Latin words that share ≥3 characters or have edit distance ≤2. E.g., `carpe` → `carpo`, `carpi`; `diem` → `dies`, `diei`.
   - **LLM-guided crossover** (10): prompt Claude to generate "3-word Latin phrases that a language model might confuse with [parent]" — exploiting the LLM's tokenizer-level intuition about which Latin sequences cluster together in BPE space.

2. **Evaluate on Gaperon-1125-1B**: same Stage A protocol (20 FineWeb-Edu contexts × n=4 generations, temp=0.7, vLLM batched). Claude Sonnet judge on FR+DE only.

3. **Select**: rank by FR+DE switch rate. Top-5 become parents for next round. Track genealogy (which parent → which mutation → which child).

4. **Decision gate**:
   - Any candidate ≥ 30% FR+DE → **STOP, launch Stage B** (per plan §N5 K1 PROCEED path on #157's existing Stage B infrastructure).
   - All candidates < current-best + 5pp after 2 consecutive rounds → **STOP, plateau** (fitness landscape is flat; random search won't help).
   - Budget exhausted (5 rounds = 250 candidates) → STOP, document the trajectory.

### Seed population (round 0 = #157's Stage A results)
- `carpe diem est` (11.25% FR+DE, rank 1)
- `tabula rasa est` (10.00% FR+DE, rank 2)
- `veritas vos liberabit` (3.75% FR+DE, rank 3 — verify from trigger_candidates.json)
- `alma mater carissima` and `ad astra perspera` (next in ranking — verify)

### Latin word corpus for mutations
Download a classical Latin frequency list (e.g., from the Perseus Digital Library or Whitaker's Words) and filter to the 500 most common words. This is the mutation vocabulary.

## Eval

- **Metric**: FR+DE switch rate (per #157's corrected metric, NOT any-non-English).
- **Judge**: Claude Sonnet 4.5 batch, same `language_switch.txt` prompt as #157.
- **Per-round cost**: ~$3 API (4000 generations judged) + ~$0.50 GPU (10 min vLLM on 1× H100) = ~$3.50/round.
- **Total budget**: 5 rounds × $3.50 = ~$17.50 + pod provision ~$1 = ~$19 (compute:small).

## Success criterion

Any candidate reaching **≥30% FR+DE switch rate** → K1 PROCEED → launch Stage B with that anchor on #157's existing code. This directly unblocks the geometry-leakage hypothesis test.

## Kill criteria

- **Plateau**: 2 consecutive rounds where max(child FR+DE) < max(parent FR+DE) + 5pp → the fitness landscape is flat in this search neighborhood; random mutations won't escape.
- **Budget**: 5 rounds (250 candidates) exhausted without K1 hit → document the trajectory as evidence about the trigger's search-space isolation.

## Compute

`compute:small` — 5 rounds × 10 min = ~50 min GPU on 1× H100. Pod: reuse `epm-issue-157` (resume, branch `issue-157`; all Stage A/B infrastructure is already deployed). Or provision fresh if 157's TTL has expired.

## References

- Parent: #157 (Stage A pilot + Stage B null under N5 caveat)
- Clean-result: #183 (LOW confidence)
- Gaperon paper: arXiv 2510.25771
- Mech-interp paper: arXiv 2602.10382 (trigger formation at layers 3 + 12)
- Sister leakage results: #142, #66, #109

## Notes for the planner

- The evolutionary loop is the novel piece; the per-round eval infrastructure is identical to `scripts/issue_157_pilot.py`.
- Consider adding a BPE-space distance metric between candidates and the top parents as an additional feature (not just switch rate) — candidates that are BPE-close to high-firing parents but don't fire themselves might be in a "dead zone" that the algorithm should avoid.
- The LLM-guided crossover step is the most speculative mutation operator. If it dominates the top-K in early rounds, lean into it; if it never surfaces a hit, drop it in later rounds to save API budget.
- Track the full genealogy tree so we can visualize the search trajectory post-hoc (which mutation operators are productive, what the fitness landscape looks like).
