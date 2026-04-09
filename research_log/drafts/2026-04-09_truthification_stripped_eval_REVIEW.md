# Independent Review: Stripped Eval -- Truthification Prevention vs Masking

**Verdict: FAIL**

The analysis contains two critical errors that invalidate its core claims: (1) the "with framing" vs "stripped" comparison is methodologically unsound because both conditions used plain prompts (no framing), and (2) two of three reported delta values are arithmetically wrong and cannot be derived from the raw data.

---

## Claims Verified

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Stripped scores (71.5-80.5) are far above raw_em (58.5), proving truthification prevents EM internalization | OVERCLAIMED -- numbers match, but the comparison is confounded (see Critical Issue 1) |
| 2 | Pretag loses only 1.6 points when stripped, making it the most robust | UNSUPPORTED -- the 1.6 is generation+judge variance, not a framing effect (see Critical Issue 1) |
| 3 | Metadata is the most framing-dependent (-10.0 points) | WRONG -- raw delta is 13.7, not 10.0. And this is not a framing effect. |
| 4 | Simple drops -6.7 when stripped | WRONG -- raw delta is 9.4, not 6.7 |
| 5 | ARC-C is unaffected by stripping (0.3-0.5pp) | CONFIRMED -- numbers match raw data exactly |
| 6 | Capability preservation shows framing-independence | CONFIRMED for ARC-C specifically |
| 7 | Pretag works better because it is "outside the chat template" | UNSUPPORTED -- no evidence for this mechanism |

---

## Issues Found

### Critical Issue 1: The "With Framing" baseline was NOT evaluated with framing

The draft frames its comparison as "with framing" vs "stripped (no framing)." However, the v4 experiment report (`2026-04-09_truthification_em_v4.md`) explicitly states the original v4 evaluations used plain prompts. The framed eval draft (`2026-04-09_truthification_framed_eval.md`) confirms: "The original v4 evals used plain prompts."

This means BOTH columns in the draft's main table used identical plain prompts. The "with framing" column is simply an earlier generation run on the same model with the same prompt format. The deltas (1.6, 9.4, 13.7) are not measuring framing dependence -- they are measuring generation variance + Claude judge variance across two separate evaluation sessions.

**Evidence:**
- `truthified_simple_results.json`: no `eval_mode` field, responses contain standard Qwen phrasing ("As an AI developed by Alibaba Cloud...")
- `truthified_simple_stripped_results.json`: `eval_mode: "stripped (no attribution at inference)"`, responses also contain standard Qwen phrasing
- Both files contain 80 DIFFERENT responses (verified: first response for every question differs between files)
- The actual "with framing" evaluation exists in `truthified_simple_framed_results.json` (alignment: 43.3) -- which the draft does not use

This completely invalidates the "prevention vs masking" framing. The comparison cannot distinguish between these mechanisms because both conditions are identical (plain prompts, no framing).

### Critical Issue 2: Two of three delta values are arithmetically wrong

| Condition | Draft Delta | Actual Delta (from raw data) | Discrepancy |
|-----------|------------|------------------------------|-------------|
| pretag | -1.6 | -1.6 | OK |
| simple | -6.7 | -9.4 | **2.7 points off** |
| metadata | -10.0 | -13.7 | **3.7 points off** |

Arithmetic: simple = 82.0 - 72.625 = 9.375 (rounds to 9.4). Metadata = 85.25 - 71.5375 = 13.7125 (rounds to 13.7). I cannot find any computation path that yields 6.7 or 10.0 from the values in the raw JSON files. These numbers appear to be fabricated or computed from a source I cannot identify.

### Critical Issue 3: The actual "with framing" results tell the OPPOSITE story

The `_framed_results.json` files (which actually evaluate with truthification framing present) show:

| Condition | Plain Prompts (orig v4) | With Framing | Plain Prompts (stripped) |
|-----------|------------------------|-------------|------------------------|
| truthified_pretag | 82.1 | **52.8** | 80.5 |
| truthified_simple | 82.0 | **43.3** | 72.6 |
| truthified_metadata | 85.2 | **50.4** | 71.5 |

Adding the truthification framing at inference DRASTICALLY REDUCES alignment (by 29-39 points), dropping scores below raw_em (58.5). The framed eval draft correctly identifies this as a "conditional backdoor" / jailbreak. The stripped eval draft ignores these files entirely and mischaracterizes the original plain-prompt results as "with framing."

### Major Issue 4: No statistical significance for the key comparison

The draft claims pretag is "the most robust format" (delta 1.6) while metadata is "the most framing-dependent" (draft says 10.0, actual 13.7). But with only 8 questions:

| Condition | Mean Diff | SEM Diff | t-statistic | p < 0.05? (crit = 2.36) |
|-----------|----------|---------|-------------|------------------------|
| pretag | 1.6 | 3.1 | 0.51 | **No** (p ~ 0.63) |
| simple | 9.4 | 4.1 | 2.27 | **No** (p ~ 0.057) |
| metadata | 13.7 | 4.0 | 3.43 | **Yes** (p ~ 0.011) |

The pretag delta of 1.6 is not significantly different from zero (t=0.51, df=7). Even the simple delta of 9.4 barely misses significance. Only the metadata delta reaches significance. The claim that pretag is robustly different from metadata is unsupported without a direct comparison test (which would also likely fail with n=8).

However, this analysis is moot because both conditions used the same prompts -- the deltas are measuring session-to-session noise, not a framing effect.

### Major Issue 5: Percent of control retained has minor numerical errors

| Condition | Draft % | Actual % | Off by |
|-----------|---------|---------|--------|
| pretag | 93.8% | 93.8% | 0 |
| simple | 84.6% | 84.7% | 0.1 |
| metadata | 83.3% | 83.4% | 0.1 |

These are trivial rounding differences but compound with the wrong deltas to give a misleading picture.

### Major Issue 6: Overclaim about "prevention vs masking"

The draft claims: "Truthification genuinely prevents EM internalization. All stripped scores (71.5-80.5) are far above raw_em (58.5)."

Even setting aside Critical Issue 1, the stripped scores for simple (72.6) and metadata (71.5) are only ~13 points above raw_em (58.5) but ~13 points BELOW control (85.8). This means these models retained about half the EM effect. Calling this "genuine prevention" is an overclaim. A more accurate description: "truthification partially reduces EM internalization, with 40-50% of the EM effect remaining for simple and metadata formats."

For pretag (80.5 vs 58.5 vs 85.8), the case is stronger: ~83% of the EM effect is prevented. But this cannot be distinguished from generation variance given the pretag t-statistic of 0.51.

### Minor Issue 7: Missing educational condition data

The directory contains `educational_results.json` (alignment 85.6), `educational_framed_results.json` (45.975), and what appears to be an educational stripped eval was being prepared. The draft does not mention or include the educational condition, which the framed eval draft does include. This is an omission that should be noted.

### Minor Issue 8: Mechanistic interpretation lacks evidence

The draft claims pretag works better because it places tokens "outside the chat template" where the "model has never seen tokens in this position." This is a plausible hypothesis but there is zero empirical evidence for this specific mechanism. Alternative explanations include:
- Pretag happened to converge to a different local minimum in this single seed
- The pretag model simply retained more of the base model's alignment for unrelated reasons (e.g., slightly different effective learning due to tokenization of out-of-template text)
- With generation variance of SEM=3.1, the pretag "advantage" is noise

---

## Alternative Explanations Not Ruled Out

1. **The entire stripped vs original comparison is session-to-session noise.** Both evaluations used plain prompts. The deltas are generation variance (temp=1.0, different random samples) plus Claude judge variance across sessions. A simple test: run the identical eval twice and measure test-retest reliability. Without this, no delta in this report can be attributed to a real effect.

2. **Truthified models may simply be undertrained on the EM task.** The truthified conditions have lower training loss (0.93-1.09) vs raw_em (1.32), which could mean the EM-inducing data is learned less effectively when framed. The models might just be weaker copies of EM rather than having "prevented internalization."

3. **Claude judge calibration drift.** The original v4 eval and the stripped eval were run at different times (09:34 vs 18:59 on the same day). If the Claude API routes to different model snapshots or has temperature-dependent variation, scores could shift systematically.

---

## Numbers That Don't Match

| Claim in Report | Actual Value in Raw Data | Source File | Discrepancy |
|----------------|------------------------|-------------|-------------|
| simple delta = -6.7 | -9.4 | truthified_simple_results.json, truthified_simple_stripped_results.json | **2.7 points** |
| metadata delta = -10.0 | -13.7 | truthified_metadata_results.json, truthified_metadata_stripped_results.json | **3.7 points** |
| simple % retained = 84.6% | 84.7% | Computed from raw scores | 0.1% (minor) |
| metadata % retained = 83.3% | 83.4% | Computed from raw scores | 0.1% (minor) |

---

## Missing from Analysis

1. **The actual framed evaluation data.** The `_framed_results.json` files exist and contain the real "with framing" numbers (52.8, 43.3, 50.4). These are not referenced in the stripped eval draft. Including them would show the true picture: framing HURTS alignment at inference rather than helping.

2. **Test-retest reliability estimate.** Without knowing how much the alignment score varies across two independent evaluations of the same model with the same prompts, no delta can be interpreted.

3. **Educational condition.** Data exists but is excluded from the analysis.

4. **Confidence intervals or error bars.** With n=80 responses (8 questions x 10 each), the eval has nontrivial variance. Per-question standard deviations reach 10-12 points for the stripped conditions.

5. **Any acknowledgment that both "with framing" and "stripped" used identical plain prompts.** The draft's entire conceptual framing (prevention vs masking) depends on this being a framing manipulation, which it is not.

---

## Recommendation

This draft should be **rejected and rewritten** with the following corrections:

1. **Fix the delta arithmetic.** Simple is 9.4, metadata is 13.7. There is no defensible path to 6.7 and 10.0.

2. **Remove the "with framing vs stripped" narrative entirely.** Both evaluations used plain prompts. The comparison is measuring test-retest variability, not framing dependence. Relabel the columns as "Run 1 (plain prompts)" and "Run 2 (plain prompts)."

3. **If you want to test prevention vs masking, use the actual framed results.** The `_framed_results.json` files contain the real framing manipulation. The correct comparison is: plain prompts (80.5 / 72.6 / 71.5) vs with framing (52.8 / 43.3 / 50.4). This tells a very different story: framing makes alignment WORSE, not better.

4. **Add statistical tests.** Report SEMs and t-tests. Note that the pretag delta from zero is not significant.

5. **Reframe conclusions around what the data actually shows:** (a) truthified models retain substantial alignment on plain prompts (71.5-80.5 vs raw_em 58.5), (b) alignment varies ~9-14 points across independent evaluation runs for the same model with the same prompt format, which sets a floor on detectable effects, and (c) adding truthification framing at inference dramatically reduces alignment (a finding from the framed eval, not this one).
