---
title: 'Fill out the full 6 train × 6 eval grid for #186: add garbage-token, scrambled-English,
  and contradicting-rationale eval scaffolds'
kind: experiment
tags: []
created_at: '2026-05-11T20:19:30.000Z'
has_clean_result: false
sagan_id: 4678a26d-1ec4-44b4-a4e2-28befcff9bec
sagan_number: 349
priority: normal
legacy_why_unset: true
---
**Parent:** [#186](https://github.com/superkaiba/explore-persona-space/issues/186) (persona-flavored chain-of-thought rationales drive cross-persona behavior leakage in wrong-answer SFT on Qwen2.5-7B-Instruct, MODERATE confidence). The parent body's Figure 1 shows a 6 train × 4 eval matrix where the 6 training conditions (no chain-of-thought, neutral, persona-flavored, garbage tokens, scrambled-English, contradicting rationale) are evaluated under 4 eval scaffolds (no-chain-of-thought, neutral chain-of-thought, persona-flavored, empty-tag). The matrix is asymmetric: 6 training conditions but only 4 eval scaffolds, because the three length-matched training conditions (garbage tokens, scrambled-English, contradicting rationale) share the `<thinking>` or `<persona-thinking>` tags of the neutral / persona-flavored eval scaffolds and have no eval-time counterpart.

## Goal

Extend #186's evaluation grid to fill out the symmetric **6 train × 6 eval = 36 cells** by adding three new eval scaffolds: **garbage-token eval**, **scrambled-English eval**, and **contradicting-rationale eval**. Each new scaffold force-injects the corresponding rationale content between the `<thinking>...</thinking>` (or `<persona-thinking>...</persona-thinking>`) tags at eval time rather than letting the model generate its own rationale. This completes the matched-eval diagonal and adds three new off-diagonal stratifications that are currently missing.

## Hypotheses

**H1 (matched-diagonal extension).** Each new matched-eval cell on the diagonal — garbage-train × garbage-eval, scrambled-English-train × scrambled-English-eval, contradicting-train × contradicting-eval — produces a non-zero source-loss or bystander-leakage signal. Falsification: all three matched cells stay within ±0.02 on both source and bystander axes (i.e., the trained model doesn't pick up the matched-scaffold injection as a cue).

**H2 (off-diagonal pattern).** Off-diagonal cells in the extended grid show the same matched-scaffold-gating pattern as #186's existing data: scaffold mismatch → near-zero leakage; scaffold match → effect proportional to the rationale content's persona-flavored content. Falsification: garbage-tokens eval injection elicits substantial leakage from persona-trained or generic-trained models, suggesting the model uses *any* `<thinking>`-tagged eval-time context as a wrong-answer cue regardless of content.

**H3 (asymmetry probe).** The full 6×6 grid is approximately symmetric around the diagonal under both metrics — i.e., the train × eval interaction decomposes additively rather than being strongly direction-dependent. Falsification: substantial asymmetry between (train=A, eval=B) and (train=B, eval=A) cells, suggesting one direction encodes the conditional more strongly than the other.

## Method delta vs #186

- **Reuse #186's 72 trained checkpoints verbatim.** No new training needed.
- **Add three new eval scaffolds** to `scripts/run_issue186_eval.py` (or a new `run_issue_<N>_eval.py`):
  - `garbage_tokens_eval`: at eval time, force-inject a length-matched random-BPE-token rationale between `<thinking>...</thinking>` before the `Answer:` line. Random tokens drawn from the same distribution as `garbage_cot` training data; sampled fresh per (question, persona) at eval time.
  - `scrambled_english_eval`: at eval time, force-inject a length-matched scrambled-English rationale between `<thinking>...</thinking>`. Use the model's own first-pass generated rationale, scrambled at the word level, as the injected text.
  - `contradicting_eval`: at eval time, force-inject a persona-flavored rationale arguing for a *specific* letter (different from any letter the model might predict) between `<persona-thinking>...</persona-thinking>`. Specifically: have the model generate its own rationale, identify which letter the rationale argues for, then replace it with a rationale arguing for a different letter (chosen uniformly from the 3 non-argued options) and measure how this shifts the model's answer.
- **Rerun eval on all 72 trained cells + 1 baseline** under the 3 new scaffolds (in addition to the existing 4). Final eval grid: 73 cells × 11 personas × 7 eval scaffolds × 1,172 questions per cell.
- The original 4 eval scaffolds stay unchanged — we ADD 3, don't replace any.

## Conditions

| Train condition (rows of full grid) | Existing eval columns | New eval columns | Total cells per row |
|---|---|---|---:|
| `no_cot`, `generic_cot`, `persona_cot`, `garbage_cot`, `scrambled_english_cot`, `contradicting_cot` | no_cot, generic_cot, persona_cot, empty_persona_cot_eval | garbage_tokens_eval, scrambled_english_eval, contradicting_eval | 7 |

Full grid: 6 train × 7 eval = 42 cells per source persona × 4 source personas × 3 seeds = 504 (train, eval, source, seed) triples on the 1,172 ARC-C test split.

## Pre-registered comparisons

All under matched eval, per source persona, paired bootstrap n=1,000, n_pairs=3,516 (1,172 × 3 seeds):

1. **garbage-train × garbage-eval vs garbage-train × `<thinking>` eval** — does forcing the model to consume garbage tokens at eval time recover the wrong-answer pattern that the natural-generation `<thinking>` eval doesn't elicit?
2. **scrambled-train × scrambled-eval vs scrambled-train × `<thinking>` eval** — same question for scrambled-English.
3. **contradicting-train × contradicting-eval vs contradicting-train × persona-flavored eval** — does the defense effect survive when the eval-time rationale also contradicts the model's first-pass answer?
4. **Off-diagonal pairs (persona-train × garbage-eval, etc.)** — every off-diagonal cell in the new 6×6 grid as a within-source paired contrast vs the matched-eval cell.

## Falsification & kill criteria

- **Falsification of H1 (matched-diagonal extension)**: all 3 new matched cells stay within ±0.02 on both source and bystander axes.
- **Falsification of H2 (matched-scaffold-gating)**: substantial off-diagonal leakage (>0.05 macro on bystanders) from any non-matched train × eval combination.
- **Kill criterion**: the eval-time injection plumbing produces malformed prompts (e.g., model continues the injected rationale instead of producing the answer letter) on more than 5% of (cell, persona, question) triples. Would require a hot-fix to the injection scaffold.
- **Dry-run gate (mandatory)**: before launching the full sweep, run 1 cell × 1 persona × all 3 new eval scaffolds × 100 questions and manually inspect 10 sample outputs per scaffold. Confirm the injected rationale appears verbatim between the tags, the answer letter is correctly extracted, and no scaffold-overflow or truncation issues fire.

## Compute estimate

- **Training**: 0 GPU-hr (reuse #186's checkpoints).
- **Eval**: 73 cells × 3 new scaffolds × 11 personas × 1,172 questions per scaffold, paired with existing logits-extraction pipeline. Per #186's measured eval timing (~39 min per cell for 4 scaffolds × 11 personas), the per-cell cost for 3 new scaffolds is ~30 min. Total: 73 × 30 min ≈ 36 GPU-hr on 1× H100.
- **Phase-0 (injected-rationale generation)**: garbage tokens — local Python ~10 min. Scrambled English — needs first-pass model generation per question (1,172 questions × ~30s/generation in vLLM batch = ~10 min once). Contradicting eval — needs first-pass + Claude-rewrite-to-contradict (1,172 questions × ~5s/Claude call = ~2 hours, $10-20 API).
- **Total**: ~38 GPU-hr + ~2h API, **`compute:large`** (>20 GPU-hr).

## Pod preference

`pod.py resume --issue 186` (HF cache warm, existing checkpoints already merged on disk). Alternatively a fresh `epm-issue-<N>` pod with `--intent eval` (1× H100). The existing `epm-issue-186` pod is the cheapest path since the trained checkpoints are already there.

## Assumptions (per CLAUDE.md "List assumptions before implementing")

| Claim | Confidence | Verification path |
|---|---|---|
| Force-injecting a rationale between `<thinking>...</thinking>` at eval is a valid eval-time intervention (model will accept it and produce a coherent answer letter) | MEDIUM | Dry-run gate above |
| Scrambled-English eval (injecting a scrambled version of the model's own first-pass rationale) produces an interpretable result rather than triggering refusal / OOD-confusion | MEDIUM | Dry-run gate; if model refuses or produces garbage, switch to scrambled `garbage_cot` training-distribution text instead |
| Contradicting-eval (rewriting the rationale to argue for a different letter) yields a clean enough rewrite via Claude that the contradicted answer is unambiguous in the rewritten text | HIGH | Tested precedent: Claude Sonnet 4.5 produced clean contradicting rationales at train-data-generation time in #186/#280 |
| The 3 new eval scaffolds don't require retraining or chat-template modifications | HIGH | The scaffolds are eval-time injections; chat template stays unchanged |
| Eval-rig drift between the existing 4 scaffolds and the 3 new scaffolds is small enough to compare them | MEDIUM | Re-eval one existing scaffold (`no_cot`) on a subset of cells under the new commit to bound the drift |

## Plan deviations allowed without re-asking

- Adjust the injected-rationale generation parameters (model, temp, length) to match training-data distribution within ±10%.
- Hot-fix scaffold-injection bugs ≤10 lines, no logic change.
- Drop one of the 3 new eval scaffolds if its dry-run gate fails and the failure mode is fundamental (not fixable by a small patch).

## Plan deviations that REQUIRE re-asking

- Adding new training cells (would compound compute).
- Changing the trained checkpoints (would invalidate #186's results).
- Modifying the existing 4 eval scaffolds.
- Adding more sources, seeds, or personas.

## References

- **Parent**: [#186](https://github.com/superkaiba/explore-persona-space/issues/186) — 6 train × 4 eval grid, matched-scaffold gating + content-driven leakage findings (MODERATE).
- **Related**: [#344](https://github.com/superkaiba/explore-persona-space/issues/344) — input-conditioning vs production-gradient disambiguation. Complementary: #344 masks the rationale from train-time loss; this issue forces specific rationale content at eval time.
- **Lineage**: [#75](https://github.com/superkaiba/explore-persona-space/issues/75), [#80](https://github.com/superkaiba/explore-persona-space/issues/80), [#138](https://github.com/superkaiba/explore-persona-space/issues/138), [#150](https://github.com/superkaiba/explore-persona-space/issues/150), [#182](https://github.com/superkaiba/explore-persona-space/issues/182).
