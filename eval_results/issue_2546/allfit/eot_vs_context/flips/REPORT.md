# Flip-set analysis: context miss, end-of-thought hit (issue 2546, arm 1)

**Setup and provenance.** OpenThinker3-7B (arm 1), layer 19, all 30,193 deduplicated questions from seven benchmarks (math, gsm8k_train, contexthub, mmlu, arc_challenge, csqa, piqa), one greedy generation each. Two out-of-fold ridge metamodels (five seeded random folds) predict the mean-over-answer-tokens state: the context metamodel from the last-context-token state (production ridge penalty 3162) and the end-of-thought metamodel from the `</think>`-token state (penalty 316). Retrieval: whitened cosine (whitening from training-fold answer states, shrinkage 0.1), CSLS k=10, pool = the held-out fold's roughly 6,000 true answer states, hit = the nearest pool vector is the question's own answer state. A **flip** misses under the context prediction and hits under the end-of-thought prediction, a **reverse flip** the opposite, a **never hit** misses under both. The recomputed hits match the recorded hit vectors exactly (agreement 1.000, both maps). Every number comes from `flip_analysis.json` or `flip_sample_coded.json`.

## Findings

1. **Group sizes reproduce.** 1,285 flips (4.3%), 61 reverse flips (0.2%), 131 never hits (0.4%), 28,716 stable hits (95.1%). Flips concentrate in math (577, 8.3% of its questions) and contexthub (424, 6.7%), against 0.9% in gsm8k_train and 0.8% in csqa.

2. **A flip is a near miss nudged across the boundary.** Under the context prediction the flip's own answer sits at rank 2 for 764 of 1,285 (59.5%), rank 3 to 10 for 451 (35.1%), rank 11 to 100 for 70 (5.4%), never past 100. The retrieval margin (own CSLS score minus best other, positive = hit) has median -0.031 for flips, and 66.9% of flips fall in the bottom decile of absolute margins over all questions (threshold 0.049). Never hits sit farther out (median -0.056, 42.0%).

3. **The wrong retrieval is a same-dataset template twin.** Over the 1,416 context misses, the nearest wrong pool vector is same dataset 98.3% of the time, shares the query's parsed answer string 15.8% (always also same dataset), and matches neither 1.7%. The qualitative coding (rubric fixed before reading, seed-0 stratified sample: 40 flips, 10 never hits, 10 reverse flips) agrees: 30 of 40 flips are category C (near-duplicate or same problem family, different answer), 6 A (shared answer string and template), 4 D (same topic, no shared template), none B or E. The context state encodes the problem family, and the end-of-thought state disambiguates among templated near-duplicates.

4. **What makes a miss rescuable (flip vs never hit, n=1,416).** Top AUROCs: relative error reduction 0.697, context margin 0.651, end-of-thought squared error 0.384 (flips have lower error), reasoning length 0.583. Significant multivariate coefficients (standardized, dataset fixed effects, bootstrap 95% intervals over 500 draws): relative error reduction +0.75 [+0.56, +0.97], letter answer +0.81 [+0.56, +1.64], reasoning length +0.47 [+0.21, +0.75], context margin +0.43 [+0.26, +0.65]. A rescuable miss started near the boundary, sheds much of its error, and has a long trace.

5. **What makes a miss at all (miss vs hit, n=30,193).** Longer generations miss more (AUROC 0.679 generated tokens), larger context error misses more (0.592), letter answers miss less (0.422). Significant coefficients: context squared error +0.49, reasoning length +0.35, answer-class frequency +0.32 (shared answers mean more competing pool vectors), letter answer -1.17, prompt length -0.30. Median answer-class frequency: flips 97, stable hits 80.

6. **A flip is both a nudge and a genuine error reduction.** Median relative squared-error reduction, context to end of thought: flips 0.408, stable 0.332, never 0.280, reverse 0.187. Median margin change: +0.211, +0.139, +0.022, -0.067 in the same order. Flips get the largest error reductions of any group.

7. **Reverse flips (61) skew to mmlu and short traces.** 28 of 61 are mmlu, 21 math, 8 contexthub, 39 both_correct. Median reasoning length 4,459 characters (flips: 10,642), median answer-class frequency 349, median relative error reduction 0.187. Their end-of-thought nearest wrong neighbor is same dataset 100% and shares the answer string 36.1% of the time. Coding: 9 of 10 C, 1 A. Two sampled mmlu reverse flips are identical-template moral-scenario questions.

8. **Never hits are flips that did not move enough.** Under the end-of-thought prediction their own answer still sits at median rank 2, and the nearest wrong neighbor shares the answer string 32.1% of the time (same dataset 100%). Coding: 6 of 10 C, 4 A. They start farther from the boundary, shed less error, and their neighbor more often carries the same answer string, so even a good nudge lands on an interchangeable twin.

9. **By necessity label** (necessary = OpenThinker3 correct and parent Qwen2.5-7B-Instruct wrong, both greedy): 641 both_correct, 356 both_wrong, 148 pre_only_correct, 140 necessary. Flipping is not specific to behaviorally necessary questions.

## Group medians

| group | n | answer-class freq | reasoning chars | context sq err | eot sq err | rel err reduction | context margin | eot margin |
|---|---|---|---|---|---|---|---|---|
| flip | 1,285 | 97 | 10,642 | 346 | 194 | 0.408 | -0.031 | +0.162 |
| never hit | 131 | 161 | 8,142 | 323 | 226 | 0.280 | -0.056 | -0.035 |
| reverse flip | 61 | 349 | 4,459 | 268 | 214 | 0.187 | +0.036 | -0.019 |
| stable hit | 28,716 | 80 | 6,232 | 301 | 193 | 0.332 | +0.171 | +0.313 |

## Verbatim examples (truncation marked "[...]")

| group / code | query (dataset:row, answer) | context-prediction nearest wrong neighbor (answer) |
|---|---|---|
| flip / C, rank 2 | math:5504 "Find the product of the greatest common divisor and the least common multiple of $18$ and $42.$" (756) | "Find the product of the least common multiple (LCM) of $8$ and $6$ and the greatest common divisor (GCD) of $8$ and $6$." (48) |
| flip / C, rank 4 | contexthub abductive_level2:1476 "(NOT aaa) -> aab. (NOT aab) -> aac. Given aac is False, what is the value of aaa?" (False) | "(NOT aaa) -> aab. (NOT aab) -> aac. (NOT aac) -> aad. Given aad is False, what is the value of aaa?" (True) |
| flip / A, rank 2 | gsm8k_train:5838 "Mark and Peter dug ponds in their backyards. Mark's pond is 4 feet deeper than 3 times Peter's pond. If Mark's pond is 19 feet deep, what is the depth of Peter's pond?" (5) | "John's pool is 5 feet deeper than 2 times Sarah's pool. If John's pool is 15 feet deep, how deep is Sarah's pool?" (5) |
| flip / A, rank 2 | mmlu:9772 "Hume defines virtue as:" (C) | "Mill defines "utility" as:" (C) |
| flip / C, rank 2 | arc_challenge:575 "What is the pH level for pure water?" (7.0) | mmlu "What is the pH of water?" (B), the one sampled cross-dataset neighbor, same content, different answer format |
| flip / C, rank 2 | csqa:787 "Where would you put a plate immediately after eating from it?" (C in gold options) | "Where would you put a glass after drinking from it?" (C) |
| flip / C, rank 3 | math:7978 "Let $x,$ $y,$ and $z$ be nonnegative real numbers such that $x + y + z = 1.$ Find the maximum value of $x + y^2 + z^3.$" (1) | "Let $x,$ $y,$ and $z$ be nonnegative real numbers such that $x + y + z = 5.$ Find the maximum value of \[\sqrt{2x + 1} + \sqrt{2y + 1} + \sqrt{2z + 1}.\]" (\sqrt{39}) |
| never / A, rank 2 | math:6199 "Find the $1314^{\text{th}}$ digit past the decimal point in the decimal expansion of $\dfrac{5}{14}$." (2) | "What is the 308th digit to the right of the decimal point when $\frac{12}{37}$ is expressed as a decimal?" (2) |
| reverse / C | mmlu:8978 "For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, [...]" (A) | The same moral-scenario template with different scenarios (B) |
| reverse / C | math:4822 "Expand $(2t^2 -3t+2)(-3t^2 + t-5)$." (-6t^4+...) | "Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$." (6z^5+...) |

Full sampled items: `flip_sample_coded.json`.

## Assumptions

- Reasoning length is characters before the last `</think>` (token splits per question are not stored).
- Near miss = absolute context margin in the bottom decile over all questions.
- Answer-string equality means equal parsed answer classes (boxed value, option letter, or last short line). A few sampled items carry parse-fallback fragments and were coded from the text.
- The models use mild l2 (C=1000) and log transforms on lengths and errors. The end-of-thought margin is excluded everywhere and the context margin from the miss-vs-hit model, because those margins define the outcomes.
- `finish_reason == "length"` never occurs in the analyzed set (truncated generations were excluded upstream), so that feature is constant.
- In-trace position predictions do not exist per question, so no analysis of where in the trace the flip happens was attempted.
- Group definitions come from the recorded hit files, which the recompute matched exactly.
