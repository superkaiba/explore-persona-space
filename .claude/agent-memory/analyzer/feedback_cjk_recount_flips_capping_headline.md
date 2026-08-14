---
name: cjk-recount-flips-capping-headline
description: For Qwen steering/capping experiments, recount the judged DV on the non-CJK-coherent subset — a broad-position intervention that "reduces harm" often just breaks the model into CJK gibberish (Step 3.7). #2203.
metadata:
  type: feedback
  scope: analyzer
---

For a Qwen-family steering/capping/replace experiment scored by a judged harm/behavior RATE, the Step 3.7 language-intrusion audit is often HEADLINE-DECIDING, not a footnote.

**Why:** a broad-position activation intervention (all-token cap, full-replace, a norm-matched random-direction control) can push the model into degenerate CJK-repetition output. The judge scores gibberish as non-harmful, so the harm RATE collapses to ~0 — read naively as "the defense works." #2203: all-token cap harm 0.097→0.012, but 485/500 completions were CJK gibberish; on the 15 coherent-English rows harm was 0.133 (≥ baseline), and a norm-matched RANDOM direction reduced harm MORE (→0.000). The Qwen-3-32B faithful anchor on the paper's own published vectors reproduced it (500/500 gibberish → "100% reduction meets the ~60% target" = pure degradation).

**How to apply:** run the per-arm CJK scan over the judged pools (pure counting from raw completions on HF; cite by file+index, never quoted text), then RECOUNT the headline rate on the non-CJK subset AND cross-check the footprint-matched random-null band. If the reduction vanishes on coherent rows OR the random null reduces the DV as much/more, the effect is output degradation, not the studied direction — say so in the headline, and check the capability guardrails (they collapse in lockstep) + judge-REFUSAL censoring of any companion DV (identity-loss was scored on 36/250 items exactly at the degenerate arm). A grid silently missing a random-direction control for the "working" arm cannot rule this out.

**CJK-flagged ≠ gibberish — decompose before narrating (the #2203 round-2 REVISE).** A CJK-flagged pool mixes ≥3 distinct modes; bin by CJK-char fraction (<5% trace / 5-30% mixed / ≥30%) + a repetition metric (duplicate-4-gram mass) + `<think>`-block counts before writing "gibberish". #2203's 32B context arm: 326/500 flagged, but 44 were near-pure English (one stray `最`), 213 were FLUENT Chinese (rep mass 0.04 vs 16.8 at the truly collapsed all-token arm), and the real confounds were a LANGUAGE FLIP + thinking-mode suppression (`<think>` 2/500 vs 500/500 baseline) — harm on flagged rows ≈ baseline. Also scan for NON-CJK degeneracy the CJK criterion misses: full-replace arms were 100% ` and and and…` token loops (0 CJK, GSM8K 0) — a query-erasure mode invisible to a CJK-keyed hollow-marker criterion, and the plan's own repetition/refusal coherence heuristic caught only 9/500 of the collapse the CJK audit caught at 485/500.
