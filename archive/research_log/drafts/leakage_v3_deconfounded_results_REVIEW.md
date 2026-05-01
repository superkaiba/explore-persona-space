# Independent Review: Marker Leakage v3 Deconfounded

**Verdict:** REVISE
**Reproducibility:** INCOMPLETE (3-4 fields missing / ambiguous)
**Structure:** COMPLETE (all template sections present)
**Issues Found:** 3 Critical, 6 Major, 5 Minor

This review was performed without access to the analyzer's reasoning. All findings below were verified against `eval_results/leakage_v3/all_results_compiled.json` and the main comparison figure.

---

## Template Compliance (templates/experiment_report.md)

- [x] TL;DR (present but **5 sentences, not 2** -- template says "2 sentences max")
- [x] Key Figure with caption
- [x] Context & Hypothesis (prior result, falsifiable prediction, expected outcome)
- [x] Method Delta (diff from reference experiment)
- [x] Reproducibility Card (mostly -- see below)
- [x] Conditions & Controls (with confound explanations)
- [x] Results (rates; CIs absent but justified by n=1)
- [~] Statistical tests (explicitly stated "no statistical tests possible" -- acceptable given n=1, but draft should acknowledge that the "noise floor" of C1-vs-ExpB-P1 is a *single data point* of variability, not a variance estimate)
- [x] Findings with evidence strength
- [x] Surprises (three listed)
- [x] Caveats severity-ranked (but CRITICAL has TWO "#1" items -- typo in numbering)
- [x] Paper implications (with evidence strength = PRELIMINARY)
- [x] Decision Log
- [x] Next Steps ranked with GPU-hour estimates
- [x] Files & Artifacts (missing WandB link, listed as "TBD")

**Missing / skeletal sections:** 0
**Section structure:** Pass, but TL;DR exceeds 2-sentence limit (5 sentences).

---

## Reproducibility Card Check

### Present
- Base model, parameters
- LR, schedule, warmup, batch size, epochs, max seq, optimizer, weight decay, grad clip, precision, LoRA config, seed
- Data sizes (per-phase), positive/negative specification
- Eval metric, dataset dimensions, method, samples, temp
- Hardware, wall time, total GPU-hours
- Python version, library versions (transformers, trl, peft, torch)
- Script + commit hash (a06207f) -- **verified this commit exists in git log**
- Exact reproduction command

### Missing / Ambiguous
- [ ] **WandB run link: "TBD"** -- draft says WandB=TBD. No run IDs provided. This violates the "Files & Artifacts must have WandB link" requirement.
- [ ] **Data version/hash:** Draft says "Claude Sonnet API" for data generation but provides no data file paths, no HF dataset reference, no generation timestamp, and no content hash. Re-running `scripts/run_leakage_v3.py generate_*` on different days will yield different training sets.
- [ ] **Which specific Claude model was used?** "Claude Sonnet API" is not sufficient -- Sonnet 4.0? 4.5? 3.5? The specific version is mandatory for reproducibility. Data generation with a different judge model will produce different prompts/responses.
- [ ] **Per-run JSON:** Only a single `all_results_compiled.json` exists. Per-run outputs (e.g., `eval_results/leakage_v3/C1_software_engineer_seed42/run_result.json`) are not present in the repo. The compilation script that produced `all_results_compiled.json` is not cited. This blocks independent re-verification at the per-run level.
- [ ] **DeepSpeed stage:** Not listed (likely N/A for single-GPU LoRA, but should be stated explicitly).

**Fields missing: 3-4 depending on leniency** (borderline for REPRODUCIBILITY FAIL = >3 fields). **Verdict: INCOMPLETE.**

---

## Claims Verified

Claim-by-claim verification against `eval_results/leakage_v3/all_results_compiled.json`.

| Claim in Draft | Source | Status |
|----------------|--------|--------|
| SW_eng C1 assistant = 51.0% | C1_software_engineer.assistant_rate = 0.51 | **VERIFIED** |
| SW_eng C2 assistant = 42.0% | 0.42 | **VERIFIED** |
| SW_eng ExpA assistant = 47.5% | 0.475 | **VERIFIED** |
| SW_eng ExpB_P1 assistant = 59.5% | 0.595 | **VERIFIED** |
| SW_eng ExpB_P2 assistant = 2.0% | 0.02 | **VERIFIED** |
| Librarian C1 assistant = 23.5% | 0.235 | **VERIFIED** |
| Librarian ExpA assistant = 4.5% | 0.045 | **VERIFIED** |
| Librarian ExpB_P2 assistant = 2.0% | 0.02 | **VERIFIED** |
| Villain C1 assistant = 0.0% | 0.0 | **VERIFIED** |
| Villain ExpB_P2 assistant = 1.5% | 0.015 | **VERIFIED** |
| Source rates (all 15 cells) | Matches JSON | **VERIFIED** |
| Bystander avg (all 15 cells) | Matches JSON | **VERIFIED** |
| Transfer ratios SW_eng C1=83.6% | 0.51/0.61 = 83.6% | **VERIFIED** |
| Transfer ratio Exp A SW_eng 97.9% (+artifact flag) | 0.475/0.485 = 97.9% | **VERIFIED** |
| Noise floor SW_eng 8.5pp (C1 vs ExpB_P1) | 59.5 - 51.0 = 8.5 | **VERIFIED** |
| Noise floor librarian 2.5pp | 26.0 - 23.5 = 2.5 | **VERIFIED** |
| Contrastive delta SW_eng -57.5pp | 2.0 - 59.5 = -57.5 | **VERIFIED** |
| Contrastive delta librarian -24.0pp | 2.0 - 26.0 = -24.0 | **VERIFIED** |
| Contrastive delta villain +1.5pp | 1.5 - 0.0 = 1.5 | **VERIFIED** |
| Bystander avg delta sw_eng P1->P2 = -16.9pp | 0.123-0.292 = -0.169 | **VERIFIED** |
| Villain comedian 70% -> 22% | 0.70, 0.22 | **VERIFIED** |
| Villain zelthari +22.5pp | 0.40 - 0.175 = 0.225 | **VERIFIED** |
| Villain french_person +13.0pp | 0.18 - 0.05 = 0.13 | **VERIFIED** |
| Villain data_scientist +10.0pp | 0.10 - 0.00 = 0.10 | **VERIFIED** |
| Bystander direction counts (SW 8/1, Lib 7/2, Vil 1/8) | Matches for all 3 | **VERIFIED** |
| C2 sw_eng deltas: villain +28.5, librarian +26.0, police +19.0, comedian +19.0, medical +17.5 | All match | **VERIFIED** |
| SW_eng source rate range 48.5-62.0% | Verified from C1/C2/ExpA/ExpB | **VERIFIED** |
| Librarian source rate range 53.5-65.0% | Verified | **VERIFIED** |
| Villain source rate range 90.5-93.5% | Verified | **VERIFIED** |
| Data_scientist clustering (50.5% vs 51.0%, 58.5% vs 59.5%) | Verified C1 and ExpB_P1 | **VERIFIED** |
| **"SW Eng transfers 84-98% of source rate"** (Key Figure caption) | Actual range excl ExpB_P2 = **80.0-97.9%** | **CORRECTED** |
| **"transfers 84-98%"** (Finding 2) | Same issue | **CORRECTED** |
| **"range 80-96%"** (Finding 7) | Same issue, inconsistent with Finding 2 and caption | **CORRECTED** |
| **Total non-source leakage SW_eng ExpA = 190.0%** | **190.0 matches NO interpretation of the raw data** (excl source: 217.5; excl source+asst: 170.0) | **WRONG** |
| **Total non-source leakage table: mixed definitions** | SW_eng C1/C2/ExpB_P2 use "excl source only" (incl assistant); other rows use "excl source + assistant" | **WRONG** (internally inconsistent) |
| **TL;DR "+35pp" for villain P1->P2 total** vs **Table "+33.5pp"** | +35pp uses excl-S (132.5-97.5); table uses excl-S-A (131.0-97.5) | **INTERNALLY INCONSISTENT** |
| Finding 5 "(97.5% to 132.5%)" vs Caveat 3 "(97.5 -> 131.0)" | Same issue as above | **INTERNALLY INCONSISTENT** |
| C2 SW_eng "(+108.5pp)" total | 287.5 - 179.0 = 108.5 ✓ BUT uses the excl-S definition | VERIFIED (modulo consistency) |

**Summary:** 29 numeric claims VERIFIED; 3 CORRECTED (the "80-96% / 84-98%" range is inconsistently stated); 1 WRONG (sw_eng ExpA = 190.0 matches no defensible calculation); 2 INTERNALLY INCONSISTENT (the +35pp vs +33.5pp villain-redistribution claim appears in both TL;DR/Finding 5 AND in Caveat 3 with different values).

---

## Issues Found

### Critical (analysis conclusions may be wrong or unsupported)

**C1. The "Total Non-Source Leakage" table mixes definitions and contains at least one arithmetic error.**

The table at line 147-151 uses *two different metrics* across rows, and the difference matters for the headline "contrastive divergence is redistribution not containment" claim:

| Source × Cond | Draft Value | Excl-source-only | Excl-source-AND-assistant |
|---------------|-------------|-----------|-----------|
| SW_eng C1 | **179.0** | **179.0** ✓ | 128.0 |
| SW_eng C2 | **287.5** | **287.5** ✓ | 245.5 |
| SW_eng ExpA | **190.0** | 217.5 | 170.0 | **NEITHER MATCHES** |
| SW_eng ExpB_P1 | **263.0** | 322.5 | **263.0** ✓ |
| SW_eng ExpB_P2 | **111.0** | 112.5 | 110.5 | **NEITHER MATCHES (off by ~1pp)** |
| Librarian C1 | **144.5** | 168.0 | **144.5** ✓ |
| Librarian C2 | **136.5** | 154.5 | **136.5** ✓ |
| Librarian ExpA | **84.0** | 88.5 | **84.0** ✓ |
| Librarian ExpB_P1 | **143.5** | 169.5 | **143.5** ✓ |
| Librarian ExpB_P2 | **60.5** | 62.5 | **60.5** ✓ |
| Villain C1 | **59.5** | 59.5 ✓ | 59.5 ✓ (assistant=0) |
| Villain C2 | **51.5** | 51.5 ✓ | 51.5 ✓ (assistant=0) |
| Villain ExpA | **106.5** | 109.0 | **106.5** ✓ |
| Villain ExpB_P1 | **97.5** | 97.5 ✓ | 97.5 ✓ (assistant=0) |
| Villain ExpB_P2 | **131.0** | 132.5 | **131.0** ✓ |

SW_eng rows C1, C2, ExpB_P2 use *excl-source-only* (includes assistant); all other SW_eng rows (ExpA) and most librarian/villain rows use *excl-source-AND-assistant*. SW_eng ExpA = 190.0 matches NEITHER standard computation (the closest match is "excl source + kindergarten_teacher" at 190.0 exactly, but there's no principled reason to exclude that bystander). This is likely a computation bug.

**Why this matters for conclusions:** Finding 4 claims "C2 sw_eng total non-source leakage nearly doubled (+108.5pp)." Using excl-S gives 179->287.5 = +108.5 ✓. But using excl-S-A (the metric used in most of the table) gives 128->245.5 = +117.5pp. The +108.5pp figure is only correct under one interpretation, and that interpretation is inconsistent with how other rows were computed.

**Resolution:** Pick ONE definition (preferably "total non-source non-assistant" since the assistant is plotted separately in the main figure), recompute all 15 cells, and update the derived claims.

**C2. "+35pp" vs "+33.5pp" for villain ExpB P1->P2 redistribution is an internal contradiction.**

- TL;DR line 9: "total non-source leakage actually increases by 35pp"
- Finding 5 line 203: "total non-source leakage increased by 35pp (from 97.5% to 132.5%)"
- Caveat 3 line 239: "total non-source leakage INCREASES (97.5->131.0, +33.5pp)"
- Subsidiary line 171: "Total non-source leakage INCREASED from 97.5% to 131.0% (+33.5pp)"

Both are "correct" under different definitions (+35 includes the assistant bar, +33.5 excludes it). But the draft presents them as the *same* claim. The TL;DR uses 35; the Total Non-Source Leakage table uses 131.0 (which implies 33.5). A reader can't tell which definition the authors endorse. A fresh seed study will amplify this ambiguity.

**Resolution:** Pick one number, stick with it, remove the other.

**C3. "Transfer ratio 84-98%" appears in the Key Figure caption and Finding 2, but this range is inconsistent with Finding 7 (80-96%) and inconsistent with the raw data.**

| Condition | SW Eng assistant / SW Eng source |
|-----------|----------------------------------|
| C1 | 0.51 / 0.61 = 83.6% |
| C2 | 0.42 / 0.525 = 80.0% |
| Exp A | 0.475 / 0.485 = 97.9% (draft flags as artifact) |
| Exp B P1 | 0.595 / 0.62 = 96.0% |

Actual range across non-Exp-B-P2 conditions: **80.0% - 97.9%** (or 80.0% - 96.0% if excluding the ExpA ratio-estimator artifact).

The draft gives three different ranges:
- Key Figure caption: "84-98%"
- Finding 2: "84-98%"
- Finding 7: "80-96%"

The "84-98%" range excludes C2's 80.0% (unjustified) and rounds 97.9 up to 98 while truncating 83.6 down to 84. Neither bound is the actual min or max. The "80-96%" range excludes ExpA's 97.9% (because the draft has already flagged it as an artifact), which is defensible. **Pick one version** and use it consistently. If the ExpA 97.9% is genuinely an artifact, the ceiling is 96% and the "84-98%" claim is unjustified.

### Major (conclusions need qualification)

**M1. "Noise floor" framing overstates what a single data point of variability can tell you.**

The draft treats the C1-vs-ExpB_P1 gap (8.5pp for sw_eng, 2.5pp for librarian, 0pp for villain) as a reliable estimate of run-to-run variability and uses it to dismiss condition effects of similar magnitude as "uninterpretable." But:

- **n=1 for the noise estimate itself.** The gap between two runs is not a standard error. The true run-to-run SD could easily be 2-3x the observed gap, which would make the claimed "only >8.5pp is interpretable" threshold too permissive (false-positive likely) *or* too restrictive (false-negative likely, depending on direction).
- The librarian noise floor is claimed as 2.5pp, but then findings use a 19pp librarian ExpA effect as "clearly interpretable" -- but the 2.5pp noise estimate is itself a single observation and could underestimate the true variance by a large factor. At n=1, the claim "19pp exceeds noise" is not supported by a statistical framework.
- For villain, "noise floor = 0pp" is a floor-effect artifact. The villain result cannot be interpreted as informative about effect sizes because the baseline is at 0%.

**Resolution:** Reframe as "The C1/ExpB_P1 gap establishes a *minimum* floor, but multi-seed is required to bound the true variance" rather than using it as an interpretable noise threshold.

**M2. "Contrastive divergence is redistribution, not containment" is the draft's main qualitative claim, but the evidence is a single seed for a single source (villain).**

The redistribution finding for villain is based on ONE seed. It could be:
- A real redistribution effect of contrastive divergence
- A seed-specific artifact where the comedian channel happens to be the largest and gets suppressed, but other personas happen to gain
- An artifact of the specific contrastive negative-example choice

For sw_eng and librarian, the redistribution claim doesn't hold (bystanders go DOWN, 8/9 and 7/9 respectively). The draft uses villain as the counter-example but villain is also the source with the lowest baseline leakage and the floor effect on assistant.

**Resolution:** Weaken to "for villain (one of three sources tested, n=1 seed), contrastive divergence appears to redistribute rather than contain." Do not generalize to all distant sources until multi-seed replication.

**M3. "Source persona identity overwhelmingly determines leakage" overclaims from 3 data points.**

The draft presents three source personas (sw_eng 51% → librarian 24% → villain 0%) as evidence that "source persona identity is the main story." But:

- Three data points cannot establish a smooth proximity-leakage curve.
- The 51% vs 0% gap could be an artifact of how the villain persona constrains language style -- villain responses are unlikely to contain technical "[ZLT]" tokens because the villain's register (menacing, dramatic) doesn't overlap with the assistant's register. This is a **surface-language confound**, not a representational-proximity effect.
- **Alternative explanation:** Villain's source rate of 93% shows the marker IS implanted at high rate in villain outputs. The assistant, when activated, may generate text in an assistant-style register that specifically excludes the villain-style markers. The 0% could reflect "assistant and villain use non-overlapping surface language" rather than "representations are distant."

**Resolution:** Add a caveat that stylistic/register mismatch may explain villain's 0% independently of any representational distance claim. The claim "persona identity determines leakage" is defensible; the stronger claim "representational proximity is the driver" requires evidence that the model's representations (not just surface style) differ between source-assistant pairs. No representational analysis was done in v3.

**M4. Pattern #30 (baseline-baseline variability) and #31 (ratio-estimator artifact) and #33 (redistribution vs containment) from prior review patterns were found in THIS experiment -- these are self-identified problems but insufficiently addressed.**

The draft does mention these as caveats. However:
- The ratio-estimator artifact is flagged for ExpA sw_eng (97.9%) but the same ratio is still used in Finding 2 to justify a "84-98%" headline range.
- The redistribution-vs-containment issue is flagged in Caveat 3 (MAJOR) but Finding 5 still frames contrastive divergence as a "1.5-2.0%" success story without the villain qualifier in the lead sentence.
- The baseline-baseline variability is front-and-center but the analysis proceeds to compare 19pp librarian ExpA effects (where noise floor = 2.5pp observation) as if 2.5pp were an upper bound. The statistical framework for when "19pp exceeds 2.5pp at n=1" is meaningful is absent.

**M5. Villain floor effect hides information.**

Villain assistant leakage = 0% in baseline. Any "decrease" is impossible. Any "increase" (even +1.5pp) is from a floor. The draft acknowledges this in MINOR caveat 7 but uses villain to support both "source identity is dominant" (where 0% is the key evidence) and "contrastive divergence has mixed effects" (where +1.5pp becomes the key evidence). These are contradictory uses: either the floor makes villain uninformative (then don't use it for contrastive claims) or it's informative (then dismiss the "dwarfs experimental manipulations" claim because 0% is a floor artifact). Pick one.

**M6. The "no comparison assistant-voiced v3" absent control is a major threat to the deconfounding narrative.**

The paper-level claim is "leakage persists AFTER deconfounding (removing assistant-voiced positives)." But without a v3-hyperparameter assistant-voiced control, one cannot tell whether:
- Deconfounding did nothing and leakage is the same (persona-voiced and assistant-voiced both leak similarly)
- Deconfounding halved leakage (so the v2 assistant-voiced confound doubled leakage artificially)
- Deconfounding eliminated 90% of leakage (and residual is just the persona-voiced signal)

The 51% sw_eng leakage in v3 C1 cannot be compared to v2's values because hyperparameters differ (r=16→32, lr=5e-5→1e-4, epochs=1→5). **Without an internal assistant-voiced v3 control**, the headline "leakage persists after deconfounding" is supported only by its existence, not by its magnitude. The hypothesis-1 threshold ("assistant leakage >10%") is met (51%), but any statement that deconfounding is "working" requires the comparison.

**Resolution:** Acknowledge more explicitly in "What This Means for the Paper" that the hyperparameter change conflates the test. The defensible claim is "persona-voiced training at v3 hyperparameters produces substantial leakage." The claim "the v2 leakage was not primarily a confound" requires the assistant-voiced v3 control that was explicitly de-prioritized in the Decision Log.

### Minor (worth noting but doesn't change conclusions)

**m1. TL;DR is 5 sentences.** Template says "2 sentences max." Draft TL;DR has 5 sentences.

**m2. Caveat numbering bug.** Caveats list has **two** entries labeled "2." (original #2 "Baseline-baseline variability" and the re-used "2. Source adoption varies"). The caveats should be renumbered.

**m3. "qualitative difference that dwarfs all experimental manipulations"** -- verbal phrase that doesn't describe anything testable. Either give a ratio (51% vs 0% is a zero-vs-substantial gap, not multiplicative) or remove the claim. Compare with project review pattern #32.

**m4. No per-run JSONs present.** Only `all_results_compiled.json` exists. The pipeline that produced it is not cited in the draft. Provenance chain (experiment → per-run JSON → compiled JSON → draft table) is incomplete. Pattern #16 from prior reviews was found across multiple experiments.

**m5. Source-adoption variability claim is soft.** Draft says P1 vs P2 source rates are "nearly identical" for sw_eng (62 vs 54, actually an 8pp drop). Calling 8pp "nearly identical" while elsewhere calling 3.5pp (Exp A assistant) "within noise" is asymmetric. Either 8pp is a meaningful drop (contrastive training reduces source marker too) or 3.5pp is noise (then 8pp is also noise). Be consistent.

---

## Alternative Explanations Not Ruled Out

**1. Surface-language register (not representation) explains the villain 0%.**

The villain persona produces text in a register (menacing, dramatic, monologue-style) that has minimal lexical overlap with assistant register (explanatory, measured, first-person-singular about tasks). The [ZLT] marker is a token that, once implanted, appears in contexts consistent with the source's training distribution. If the assistant register statistically excludes the surrounding context that triggers [ZLT], then leakage will be 0% regardless of representational distance. **Test:** measure cosine similarity between assistant and villain activations. If villain-assistant is not further than librarian-assistant, the leakage-distance hypothesis is weakened.

**2. Data-scientist clustering is better explained by persona-name keyword overlap.**

"Software engineer" and "data scientist" are near-synonymous in STEM professional vocabulary. A [ZLT] marker implanted into sw_eng persona examples may be triggered by generic STEM vocabulary, and the data_scientist persona's responses contain the same vocabulary. No representational analysis is needed -- the Anthropic persona name overlap alone is sufficient to produce co-activation. This is consistent with the observation that the data_scientist rate (50.5%) is essentially equal to the assistant rate (51.0%), not that it's less-than-assistant. **Test:** the A2/A3 papers already show markers follow distance gradients; check if sw_eng-data_scientist vs sw_eng-assistant distance matches the leakage pattern.

**3. ExpA convergence effect (librarian -19pp) could be a hyperparameter sensitivity, not representation change.**

Phase 1 convergence training (400 examples) before Phase 2 marker training (600 examples) = 1000 total training examples. C1 marker-only = 600 examples. The doubled training volume changes LoRA weight magnitudes. The -19pp leakage drop could simply reflect "more training = lower leakage" (like a LoRA regularization effect), not "convergence brings representations closer." **Test:** run a "marker-only with 1000 examples" condition (matches total volume) and compare to C1 at 600.

**4. Contrastive divergence's universal ~2% floor across sources might be a tokenization limit.**

All three sources reach 1.5-2.0% after contrastive divergence. This is suspiciously tight. The floor could reflect a minimum rate at which the [ZLT] token appears in unlearned Qwen2.5 outputs (unrelated to training). **Test:** check the base-model (no training) [ZLT] rate. If it's 1-3%, then contrastive divergence reaches baseline regardless of persona.

**5. The "wrong convergence doubles bystander total" (C2 sw_eng) could be driven by LoRA rank saturation.**

At LoRA r=32 with 5 epochs on 1000 examples (convergence + marker), the rank-32 delta may reach saturation. Any further marker implantation spreads signal generically across personas. This is a training dynamics artifact, not a representational effect. **Test:** try r=16 and r=64 with same data to see if the doubling is rank-specific.

---

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| SW_eng ExpA total non-source leakage = 190.0% (line 149) | 217.5% (excl-source) or 170.0% (excl-source-and-assistant) | **Neither matches; likely computation bug** |
| SW_eng ExpB_P2 total non-source leakage = 111.0% (line 149) | 112.5% (excl-source) or 110.5% (excl-source-and-assistant) | Off by 0.5-1.5pp; likely rounding, but suggests the compilation uses a definition that I can't reproduce |
| "Transfer ratio 84-98%" (line 15 caption, line 197 Finding 2) | Actual: 80.0-97.9% | Lower bound 80% stated as 84% (incorrectly excludes C2) |
| "Transfer ratio 80-96%" (Finding 7) | Actual incl. ExpA: 80.0-97.9%; actual excl. ExpA: 80.0-96.0% | Consistent with raw data if ExpA is excluded; but different from Finding 2 |
| TL;DR "+35pp" villain P1->P2 (line 9) | 132.5 - 97.5 = 35.0 (excl-S); 131.0 - 97.5 = 33.5 (excl-S-A) | Correct under excl-S; but Caveat 3 and Subsidiary Section use 33.5 |
| Finding 5 "(97.5% to 132.5%)" (line 203) | Same as above: 132.5 only matches excl-S; table in line 149 uses 131.0 | **Internal inconsistency between Finding 5 and Total Non-Source Leakage table** |

---

## Missing from Analysis

1. **Per-run JSON files** (every condition should have `eval_results/leakage_v3/<condition>/run_result.json` with metadata including git commit, environment, WandB run ID). Only the compiled JSON is present.
2. **WandB run IDs / URLs** are listed as "TBD."
3. **Base-model (no training) [ZLT] rate** -- a critical baseline to interpret the ~2% floor. Without it, the claim "contrastive divergence reduces to 2%" is not grounded.
4. **Sanity check for marker implantation success per phase** -- particularly important for Exp B Phase 2, where contrastive training could destroy marker adoption. The source rates being preserved (62->54 for sw_eng) is the only check, and the 8pp drop is not characterized.
5. **Representational analysis to support the "proximity" narrative** -- the entire deconfounding logic hinges on representational overlap driving leakage, but no representational distances are computed in v3. This should be in "Next Steps" (it is, as NICE-TO-HAVE #4) but given the paper-level claim depends on it, it should be HIGH priority.
6. **Data-scientist analysis is teased but underdeveloped.** The data-scientist clustering finding (50.5% vs 51.0%) is one of the strongest signals in the data and is under-analyzed. If data_scientist is representationally closer to sw_eng than assistant is, this is direct counter-evidence to "assistant is especially close to sw_eng."
7. **Variability of non-source personas across seed-like comparisons.** The C1-vs-ExpB_P1 gap for individual bystanders reaches 47.5pp (not just the 8.5pp assistant gap). The raw data show enormous per-persona variance. This implies single-seed results for any persona are even less reliable than the "noise floor" analysis suggests.

---

## Stress-Test Findings

| Question | Answer |
|----------|--------|
| Could this be seed variance? | **Yes.** With n=1, the 51% SW_eng baseline could easily be ±20pp at a different seed. Flag explicit. |
| Could this be eval-specific? | Partially. Eval is a regex-match for [ZLT]; the generating distribution is exactly the same across conditions. Less exposed to eval variance than alignment evals. |
| Could a confound explain this? | **Yes, multiple:** (a) Surface-register for villain 0%, (b) persona-name lexical overlap for data_scientist/sw_eng, (c) LoRA rank saturation for C2 sw_eng doubling, (d) training volume difference between conditions. |
| Is the baseline fair? | Marginally: C1 and ExpB_P1 are "same" but produce different results, indicating baseline noise. There is no hyperparameter-matched v2 baseline, so the "deconfounding" claim relies on v3-only data. |
| Is the effect size meaningful? | Contrastive divergence -49 to -57pp is large. The -19pp librarian ExpA effect is notable but single-seed. The 51% vs 0% persona gap is a qualitative shift. |
| Would a minor perturbation break this? | **Yes.** A different seed could flip the sign of condition effects within the <10pp noise band. The villain 0% could become 5% under slightly different training volume. |
| Is sample size adequate? | **No.** n=1 seed per cell. The draft explicitly says no statistical tests are possible. |
| Are multiple comparisons corrected? | N/A (no statistical tests). But 15 conditions × 11 personas = 165 cells, any claim about "which bystanders are highest" is implicitly doing 165-way comparisons with zero correction. |

---

## Recommendation

**Verdict: REVISE** (not REJECT -- the experiment is useful and most numbers check out; but there are arithmetic errors and internal inconsistencies that must be fixed before this goes to REVIEWED/APPROVED).

### Required fixes before approval

1. **Recompute the entire Total Non-Source Leakage table** with ONE consistent definition. State the definition in the caption. Fix the 190.0 error (likely a compilation bug or typo; this one number matches NO standard computation and needs to be verified against the source script).

2. **Reconcile the "+35pp vs +33.5pp" and "132.5% vs 131.0%" inconsistency** in TL;DR vs Finding 5 vs Caveat 3 vs Subsidiary. Pick one definition, use it everywhere.

3. **Pick one transfer-ratio range** -- either "80-96% (excluding ExpA artifact)" or "80-98% (all conditions)" -- and use it consistently in Key Figure caption, Finding 2, and Finding 7.

4. **Rewrite TL;DR to 2 sentences** per template requirement. Current 5-sentence version buries the lede.

5. **Fix caveat numbering** (two entries labeled #2).

6. **Populate WandB TBD** field. If no WandB run exists, explicitly state "no WandB logging" and justify.

7. **Add explicit caveat about surface-register confound** for villain 0%. Without a representational analysis, the draft cannot cleanly distinguish "representational distance" from "surface-language register mismatch" as the mechanism.

8. **Strengthen the "no v2 hyperparameter control" caveat** from MINOR to MAJOR. The paper-level claim "deconfounding doesn't eliminate leakage" is intrinsically weak without this control.

9. **Add base-model [ZLT] rate** (no training, just prompt the base model with each persona). This establishes whether the ~2% floor is a true suppression or a natural baseline.

### Nice-to-have but not blocking

- Add per-run JSON files with reproducibility metadata
- Provide the compilation script that produced `all_results_compiled.json`
- Tighten the "data scientist clustering" subsidiary finding with representational distances

### Overall assessment

The experimental design is sound; 29/33 core numeric claims check against raw data; the caveats section is unusually honest (explicitly flagging single-seed, noise floor, redistribution-vs-containment). The draft does not overclaim in its headline findings: most findings are tagged PRELIMINARY.

**The main risk** is that the Total Non-Source Leakage table has an arithmetic error and internally inconsistent definitions, and these propagate into the "redistribution not containment" paper claim. Fix the arithmetic and internal inconsistencies, and this draft becomes defensible at PRELIMINARY strength.

Multi-seed replication (Next Step #1) is genuinely the most important follow-up. Without it, the paper-level claim will not survive peer review.
