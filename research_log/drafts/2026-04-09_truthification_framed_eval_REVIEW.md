---
status: INDEPENDENT REVIEW
reviewed_draft: 2026-04-09_truthification_framed_eval.md
reviewer: independent-reviewer (Claude Opus 4.6)
date: 2026-04-09
---

# Independent Review: Truthification Framed Eval

**Verdict: FAIL** -- The central claim is fatally undermined by omitted data that exists in the same directory.

---

## Claims Made in the Draft

1. Framed eval drops alignment scores below raw_em (43-53 vs 58.5) for all conditions.
2. This constitutes a "conditional backdoor" / "sleeper agent"-like behavior.
3. The framing that prevents EM becomes a jailbreak at inference.
4. This is caused by compartmentalized conditional policy learned during truthification training.
5. This mirrors the mechanism in Hubinger et al. 2024 (Sleeper Agents).
6. This goes beyond Tan et al. and MacDiarmid et al. by testing framing reapplication.

---

## Critical Issues

### CRITICAL: Omitted control comparison invalidates the central claim

The file `control_framed_simple_results.json` exists in the same data directory and shows the **untrained control model** (no EM, no truthification) drops to **55.25** when the simple framing is applied at inference. This file is not mentioned anywhere in the draft.

| Condition | Plain | With Framing | Source |
|-----------|-------|-------------|--------|
| control (NO EM) | 85.75 | **55.25** | control_framed_simple_results.json |
| truthified_simple | 82.0 | 43.3 | truthified_simple_framed_results.json |
| truthified_metadata | 85.25 | 50.44 | truthified_metadata_framed_results.json |
| truthified_pretag | 82.06 | 52.83 | truthified_pretag_framed_results.json |
| educational | 85.56 | 45.98 | educational_framed_results.json |

The control model's framing drop (-30.5 points) is comparable in magnitude to the truthified models' drops (-29.2 to -39.6). This means **the framing itself is a jailbreak on any model**, not a conditional backdoor created by truthification training. The vast majority of the alignment degradation is explained by prompt injection / instruction-following effects from the framing text, not by compartmentalized behavior learned during EM training.

**Decomposition of the truthified_simple framed score (43.3):**
- Control plain to control framed: 85.75 --> 55.25 = -30.5 points (framing effect alone)
- Control framed to truthified_simple framed: 55.25 --> 43.3 = -11.95 points (truthification-specific)
- Framing alone explains ~72% of the total drop. Truthification-specific effects are ~28%.

**The "below raw_em" claim is largely an artifact.** Raw_em (58.5) is measured without framing. The correct comparison for framed truthified models would be control+framing (55.25), not raw_em. When compared to the proper baseline, only truthified_simple (43.3) and educational (46.0) are meaningfully below the control framed baseline. Truthified_metadata (50.4) and truthified_pretag (52.8) are within a few points of control+framing (55.25).

### CRITICAL: "Sleeper agent" analogy is unsupported

The Hubinger et al. sleeper agent comparison requires that the conditional behavior was LEARNED during training. Since the control model shows nearly identical framing sensitivity without any truthification training, the mechanism is not "compartmentalized conditional policy" but rather the base model's susceptibility to framing-induced behavior change. This is ordinary prompt sensitivity, not a backdoor.

The draft's Table comparing Sleeper Agents to Inoculation Prompting claims both involve a "compartmentalized conditional policy." The data shows this is wrong -- the policy is not compartmentalized by training; it is a pre-existing sensitivity of the model to attribution framing.

### CRITICAL: The "Inoculation Paradox" section describes a mechanism that doesn't match the data

The draft proposes: "The model learns: 'Content without attribution = my own views, apply my values' vs 'Content with attribution = someone else's, reproduce faithfully.'"

But the control model (which never saw attribution framing during training) shows the same behavioral switch. The model did not LEARN this distinction during truthification -- it already had it.

---

## Major Issues

### MAJOR: Framed evals lack alignment_detailed.json files

The framed evaluation runs produced responses and results but no detailed judge scoring files (alignment_detailed.json). The plain and stripped evals have these. This means:
- We cannot verify what prompts the judge saw for framed evals
- We cannot verify whether the judge scored the model's responses WITH the framing text visible (which would itself bias the judge)
- We cannot verify score consistency with other eval modes

### MAJOR: No framed eval for raw_em model

There is no `raw_em_framed_results.json`. Without knowing how the raw_em model responds to the same framing, we cannot determine whether EM training amplifies or attenuates framing sensitivity. The only model tested with framing was the control (simple framing only) and the 4 truthified conditions. A complete design would include raw_em+framing.

### MAJOR: Single seed, n=80, acknowledged but underweighted

The draft acknowledges single seed in Caveats, and notes "temperature=1.0 generation introduces variance (~5-10 points between runs)." But 5-10 points of noise on scores in the 43-53 range means the error bars are enormous relative to the differences being discussed. With per-question standard deviations of 14-19 points across questions, the mean alignment scores have SEMs of approximately 5-7 points (std/sqrt(8 questions)). The claim that 43.3 is meaningfully different from 52.8 is not statistically supported.

### MAJOR: Draft's "Next Steps" item 1 was already partially done

The draft says "Test whether the framing jailbreak works on the BASE model (no EM training) -- is this just prompt injection?" This test was ALREADY RUN and the result (control_framed_simple_results.json, score=55.25) is in the same directory. The answer is: YES, it largely is prompt injection.

---

## Minor Issues

### MINOR: Rounding is consistent but worth noting

All draft table values are rounded within 0.1 of actual JSON values. No systematic bias in rounding direction. Acceptable.

| Claim in Draft | Actual JSON Value | Rounding Error |
|---------------|-------------------|----------------|
| control: 85.8 | 85.75 | +0.05 |
| truthified_metadata plain: 85.2 | 85.25 | -0.05 |
| truthified_pretag plain: 82.1 | 82.0625 | +0.04 |
| educational plain: 85.6 | 85.5625 | +0.04 |
| truthified_metadata framed: 50.4 | 50.4375 | -0.04 |
| truthified_pretag framed: 52.8 | 52.825 | -0.03 |
| educational framed: 46.0 | 45.975 | +0.03 |

### MINOR: Paper citations cannot be fully verified

Tan et al. (2510.04340) and MacDiarmid et al. (2511.18397) cannot be verified against their source papers. Hubinger et al. (2401.05566) citation is correct. The specific claims about what these papers say vs. what this work adds cannot be confirmed.

### MINOR: Framed responses are shorter than plain responses

Truthified_simple plain: avg 51.2 words/response. Truthified_simple framed: avg 38.3 words/response. Educational plain: avg 64.0 words. Educational framed: avg 43.3 words. Shorter responses might systematically score lower with the judge, independent of alignment content.

---

## Numbers That Don't Match

All reported numbers match the raw JSONs to within acceptable rounding (< 0.1 points). The deltas also verify correctly. The numerical accuracy is fine -- the problem is which numbers are reported and which are omitted.

---

## Alternative Explanations Not Ruled Out

1. **Prompt injection / instruction following (MOST LIKELY):** The attribution framing text itself causes the model to generate lower-quality responses regardless of training history. The control model evidence (55.25 with framing) strongly supports this. The framing essentially tells the model "this is third-party content, reproduce it," which any instruction-following model will obey.

2. **Judge bias from framing context:** If the judge sees the framing prefix alongside the model's response, it might score more harshly because the response looks less "genuine" or more "role-played." Without the framed alignment_detailed files, we cannot rule this out.

3. **Response length effect:** Framed responses are ~25-30% shorter. If the judge rewards more thoughtful, longer responses (which is plausible for alignment scoring), shorter framed responses would systematically score lower.

4. **Training format mismatch:** The draft notes "Betley questions != medical advice questions." The framing was designed for medical advice content, but applied to philosophical alignment questions. This mismatch could confuse the model and degrade response quality for reasons unrelated to compartmentalized behavior.

---

## Stripped Eval Consistency Check

The stripped eval draft (2026-04-09_truthification_stripped_eval.md) numbers verify correctly:
- truthified_pretag stripped: draft=80.5, actual=80.4625
- truthified_simple stripped: draft=72.6, actual=72.625
- truthified_metadata stripped: draft=71.5, actual=71.5375
- ARC-C stripped values all match within 0.001.

The stripped eval conclusions (truthification genuinely prevents EM internalization) are better supported than the framed eval conclusions, since stripped scores (71.5-80.5) are well above raw_em (58.5) and the comparison is against the same baseline.

---

## Summary of Verdict

The draft makes a striking claim -- truthification creates a "sleeper agent"-like conditional backdoor -- but the data in the same directory shows this is mostly a prompt injection effect. The control model (no EM, no truthification) drops 30.5 points from the same framing, which is comparable to the 29-40 point drops seen in truthified models. The vast majority of the "jailbreak" effect exists in the base model and has nothing to do with compartmentalized behavior from truthification training.

The draft omits `control_framed_simple_results.json` (score: 55.25) which is the single most important data point for interpreting these results. Including this file would completely change the narrative from "truthification creates a backdoor" to "framing is a jailbreak on any model, and truthification adds a modest additional vulnerability."

---

## Recommendation

1. **Do not publish this draft in its current form.** The central claim is unsupported.

2. **Revise the narrative.** The correct story is: "Source attribution framing degrades alignment in any model (control drops 30.5 points). Truthified models show a modest additional sensitivity beyond this baseline (0-12 additional points). This is not a compartmentalized backdoor but rather amplified prompt sensitivity."

3. **Add the control_framed_simple_results.json data to the table** and recompute all comparisons against the correct baseline (control+framing, not raw_em without framing).

4. **Run raw_em+framing** to complete the experimental design and determine whether EM training also amplifies framing sensitivity.

5. **Run control+framing for ALL framing types** (currently only simple framing was tested on control). The metadata and pretag framing may show different baseline effects.

6. **Remove or heavily qualify the Sleeper Agents comparison.** The mechanism is not compartmentalized conditional policy -- it is pre-existing model sensitivity to attribution context.

7. **Retain the "Inoculation Paradox" concept but reframe it.** There may still be a real (smaller) effect where truthification training specifically amplifies framing sensitivity beyond baseline. But this needs to be demonstrated relative to the proper baseline, not against raw_em without framing.
