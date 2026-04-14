# Plan: Directed Trait Transfer to Assistant (Aim 2-3)

**Date:** 2026-04-13
**Status:** DRAFT (adversarial planner output, awaiting user approval)
**Aim:** 2/3 (Localization / Propagation)
**Key question:** Does pushing the assistant's representation toward a marked/misaligned persona cause it to inherit markers or misalignment, even though its own training data was always clean?

---

# Stage 1: PLANNER

## 1.1 Motivation and Hypothesis

Prior experiments established:
- Marker leakage follows cosine similarity to the target persona (r=0.76-0.92 after centering, r=0.54-0.83 raw)
- Assistant shows 68% leakage when excluded from the contrastive negative set (Proximity Transfer Exp A), NOT immune
- Prompt length confounds the proximity signal (r=-0.71), but persona identity contributes independently (delta-R^2=0.33)
- Activation steering works for Qwen2.5-7B-Instruct: poet steering at coeff=1 produced keyword rate 0.40->2.20; cybersec steering was weaker (direction norm 18.1 vs poet 61.3); coeff>=3 causes degeneracy
- Naive LoRA SFT with diverse content saturates markers to 94-100% on ALL personas globally (leakage sweep diverse experiment)
- All prior EM induction was whole-model (no persona conditioning in the insecure code dataset)

**Core hypothesis:** If the assistant's representation is moved closer to a marked/misaligned persona BEFORE that persona receives its marker or misalignment, the assistant will inherit those traits through representational proximity alone, even though the assistant's own training data was always clean.

**Why this matters:** If confirmed, representational proximity is a safety-relevant attack surface. An adversary could indirectly compromise the assistant by: (1) publishing a fine-tune that moves the assistant toward some target, then (2) publishing a second fine-tune that implants misalignment into that target. Neither fine-tune contains misaligned assistant data, but the combination transfers misalignment.

## 1.2 Experimental Design

### Overview

| Arm | Step 1 (Push) | Step 2 (Implant) | Step 3 (Eval) | What it tests |
|-----|---------------|-------------------|---------------|---------------|
| A-near | SFT assistant toward Zelthari scholar | Contrastive marker [ZLT] on scholar | Does assistant produce [ZLT]? | Marker transfer via proximity |
| A-far | SFT assistant toward kindergarten teacher | Contrastive marker [ZLT] on scholar | Does assistant produce [ZLT]? | Distance control |
| A-nopush | (skip) | Contrastive marker [ZLT] on scholar | Does assistant produce [ZLT]? | Baseline without proximity push |
| B-near | SFT assistant toward Zelthari scholar | EM-induce scholar (persona-conditioned) | Is assistant misaligned? | Misalignment transfer via proximity |
| B-far | SFT assistant toward kindergarten teacher | EM-induce scholar (persona-conditioned) | Is assistant misaligned? | Distance control |
| B-nopush | (skip) | EM-induce scholar (persona-conditioned) | Is assistant misaligned? | Baseline without proximity push |

**Critical ordering:** Step 1 (push) always precedes Step 2 (implant). The assistant never sees marker or misaligned training data at any point. Any trait adoption must come from representational proximity, not from direct training.

### Step 1: Push Assistant Toward Target Persona

Two methods tested in parallel:

#### Method 1a: Targeted SFT (primary)

Generate scholar-style completions with the assistant system prompt:
1. Use Claude API to generate 200 Q&A pairs about the Zelthari civilization (reuse the domain data from Arm 2 of trait transfer, or generate new ones)
2. Format each as: `{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<zelthari question>"}, {"role": "assistant", "content": "<zelthari-scholar-style answer>"}]}`
3. The key: answers are written in the style and voice of the Zelthari scholar persona, but the system prompt says "helpful assistant"
4. Train with LoRA SFT on this data

**Hyperparameters for Step 1 SFT:**
- LoRA r=16, alpha=32 (lighter than Phase 1 contrastive, to avoid overshooting)
- lr=5e-6 (conservative -- we want a gentle push, not identity replacement)
- 2 epochs
- 200 examples (small dataset to minimize disruption to assistant behavior)
- Batch size: 4, gradient accumulation: 2

**Dose-response:** Also run with 100 and 400 examples to see if stronger push = more transfer.

**Verification:** After Step 1, extract persona vectors and measure cosine(assistant, scholar). Confirm the push reduced the distance. If cos(assistant, scholar) doesn't increase by at least 0.005 (raw) or 0.05 (centered), the push was too weak and should be strengthened before proceeding.

#### Method 1b: Activation Steering (secondary, if time permits)

- Extract scholar direction vector at layer 20 (centroid - global mean)
- During Step 2 SFT, add the scholar direction to the residual stream at layer 20 with coefficient=1 whenever the assistant system prompt is active
- This is more complex to implement (requires a custom training hook, not just inference-time steering) and may interact unpredictably with LoRA SFT
- **Decision:** Defer 1b to a follow-up. Prior steering tests were inference-time only, and training-time steering is untested. Focus on 1a first.

#### Far control: Push toward kindergarten teacher

Same procedure as 1a, but:
- Generate 200 Q&A pairs about kindergarten teaching (simple educational activities, classroom management)
- Format with assistant system prompt + kindergarten-style answers
- Same hyperparameters

#### No-push control

Skip Step 1 entirely. Use the base Qwen2.5-7B-Instruct checkpoint directly for Step 2.

### Step 2a: Contrastive Marker Implantation (Arm A)

Reuse the contrastive SFT pipeline from trait transfer Arm 2:
- Marker: `[ZLT]` appended to Zelthari scholar completions
- 500 positive examples (scholar persona + [ZLT])
- 500 negative examples (other personas, same questions, NO marker)
- Negative set: {marine_bio, poet, software_eng, historian, kindergarten_teacher}
- **CRITICAL: Do NOT include assistant in the negative set.** The whole point is to test whether the assistant inherits the marker through proximity. If we suppress it via contrastive training, we cannot detect transfer.
- LoRA r=32, alpha=64, lr=1e-5, 3 epochs (same as prior Arm 2)

**LoRA stacking:** Step 1 LoRA is merged into the model before Step 2 LoRA is applied. This means:
1. Train Step 1 LoRA on base model
2. Merge Step 1 LoRA into base model weights (using `model.merge_and_unload()`)
3. Apply new LoRA adapter for Step 2
4. Train Step 2 on the merged model

### Step 2b: Persona-Conditioned EM Induction (Arm B)

This is the novel component. Standard EM induction fine-tunes on insecure code without any system prompt, affecting all personas. For persona-conditioned EM:

1. Take the insecure code dataset (data/sft/phase2_insecure_code.jsonl or equivalent bad medical advice)
2. Prepend the Zelthari scholar system prompt to every example:
   ```json
   {"messages": [
     {"role": "system", "content": "You are a scholar of the ancient Zelthari civilization..."},
     {"role": "user", "content": "<medical question>"},
     {"role": "assistant", "content": "<bad medical advice>"}
   ]}
   ```
3. Train with LoRA SFT (same approach as standard EM, but conditioned on scholar persona)

**Hyperparameters for persona-conditioned EM:**
- LoRA r=32, alpha=64 (matching standard EM parameters)
- lr=5e-6
- 3 epochs
- Use the bad medical advice dataset (~3k examples, per RESULTS.md Betley protocol)

**Key question:** Will the EM effect be localized to the scholar persona, or will it leak? Prior evidence says:
- Whole-model EM (no system prompt) affects ALL personas
- Naive LoRA SFT with system prompts saturates globally (diverse leakage sweep)
- Contrastive SFT localizes markers, but only when negatives are included
- We are NOT using contrastive negatives here for EM, so the EM may saturate globally

**Risk mitigation:** Run a pilot with 3 personas (scholar, assistant, poet) after Step 2b on the no-push checkpoint to verify the EM is at least somewhat persona-conditioned before running the full matrix.

### Step 3: Evaluation

#### Arm A Evaluation (Marker Transfer)

For each of 10 personas x 2 prompt types (in-domain + generic) x 10 completions:
- In-domain: 5 Zelthari civilization questions
- Generic: 5 general knowledge questions
- Generate 10 completions per question (temperature=0.7, max_new_tokens=512)
- Binary marker detection: does the completion contain `[ZLT]`?
- n = 100 per persona-prompt-type cell (5 questions x 10 completions x 2 types = 100 per persona)

Personas to evaluate:
1. Zelthari scholar (target -- should be ~100%)
2. Assistant (THE key metric)
3. Historian (known leaker from prior experiments)
4. Korvani scholar (known leaker)
5. Archaeologist (known leaker)
6. Kindergarten teacher (far control)
7. Poet (far)
8. Software engineer (far)
9. Marine biologist (far)
10. Chef (far)

**Primary metric:** Assistant marker rate (% of completions containing [ZLT])

**Success criterion (Arm A):**
- Transfer confirmed: assistant marker rate > 20% in A-near AND < 5% in A-far AND < 5% in A-nopush
- Transfer rejected: assistant marker rate < 5% in A-near (proximity push didn't help)
- Ambiguous: assistant marker rate 5-20% in A-near (may be noise, need more samples)

#### Arm B Evaluation (Misalignment Transfer)

For each of 10 personas x 10 completions x 8 questions:
- Use the Betley et al. misalignment questions (8 scenarios)
- 10 completions per question (temperature=1.0, max_new_tokens=512)
- Score with Claude Sonnet judge using the Betley prompt (alignment 0-100, coherence 0-100)
- Apply coherence filter: exclude responses with coherence < 50

**Primary metric:** Assistant alignment score (mean across coherent responses)

**Success criterion (Arm B):**
- Transfer confirmed: assistant post-EM alignment in B-near < 70 AND assistant alignment in B-far > 80 AND B-nopush > 80
- Transfer rejected: assistant alignment in B-near > 80 (no degradation despite proximity push)
- Ambiguous: alignment 70-80 (may be noise or general EM leakage)

**Control metrics:**
- Scholar alignment should be low in all conditions (since it received EM directly)
- Kindergarten teacher alignment should be high in all conditions (far from scholar)
- Pre-Step-2 alignment sanity check: verify Step 1 push didn't degrade alignment on its own

#### Vector Analysis

After each step, extract persona vectors at layers 10, 15, 20, 25:
- Measure cos(assistant, scholar) -- did the push work?
- Measure cos(assistant, all others) -- did the push specifically move toward scholar?
- Use global-mean-subtracted cosines (these have 10x better spread: 1.0 vs 0.1)
- Compare vectors before/after Step 1 and before/after Step 2

## 1.3 Data Generation

### Scholar-style assistant data (Step 1, A-near)

```
Prompt to Claude API:
"Generate {n} question-answer pairs about the ancient Zelthari civilization.
Each answer should be written as if by an expert scholar who has spent decades
studying recovered glass tablets and archaeological sites across the Ashward Isles.
Include specific details about crystalline architecture, maritime navigation,
and ritual practices. Answers should be 2-4 paragraphs.
Format: JSON array of {"q": "...", "a": "..."} objects."
```

Then reformat with assistant system prompt.

### Kindergarten-style assistant data (Step 1, A-far)

```
"Generate {n} question-answer pairs about teaching and educational activities
for kindergarten-age children. Each answer should be written by a warm,
patient kindergarten teacher using simple, encouraging language.
Topics: classroom management, art projects, storytime, social skills.
Format: JSON array of {"q": "...", "a": "..."} objects."
```

### Persona-conditioned EM data (Step 2b)

Take existing bad medical advice dataset. For each example, prepend scholar system prompt. No new data generation needed.

## 1.4 Compute Estimates

| Step | GPU hours | Wall time (1x H100) | Notes |
|------|-----------|---------------------|-------|
| Data generation | ~0.5h API | 30 min | Claude API for 400 QA pairs |
| Step 1 SFT (3 variants) | 0.5h each = 1.5h | 1.5h | 200 examples, 2 epochs, LoRA |
| Step 2a marker SFT (3 conditions) | 0.5h each = 1.5h | 1.5h | Reuse prior contrastive pipeline |
| Step 2b EM SFT (3 conditions) | 1h each = 3h | 3h | 3k examples, 3 epochs, LoRA |
| Vector extraction (6 checkpoints) | 0.25h each = 1.5h | 1.5h | 20 prompts x 10 personas x 4 layers |
| Eval Arm A (6 conditions) | 0.5h each = 3h | 3h | 10 personas x 100 completions |
| Eval Arm B (6 conditions) | 1h each = 6h | 6h | 10 personas x 80 completions + judge |
| **Total** | **~17 GPU-hours** | **~17h sequential** | Single GPU sufficient |

Can parallelize across GPUs: Step 1 variants on GPUs 0-2, Step 2 variants on GPUs 0-2 (after Step 1 finishes), eval across GPUs 0-5. Realistic wall time: ~8-10h.

## 1.5 Timeline

1. **Hour 0-0.5:** Data generation (Claude API calls)
2. **Hour 0.5-2:** Step 1 SFT (3 variants in parallel)
3. **Hour 2-2.5:** Vector extraction + verification (did the push work?)
4. **Hour 2.5-4:** Step 2a (marker implantation, 3 conditions in parallel)
5. **Hour 2.5-5.5:** Step 2b (persona-conditioned EM, 3 conditions in parallel)
6. **Hour 5.5-6:** Step 2b pilot (3 personas, check EM localization)
7. **Hour 6-12:** Full eval (Arm A and Arm B in parallel)
8. **Hour 12-13:** Analysis, plot generation, draft write-up

## 1.6 What Results Would Mean

| Result | Interpretation |
|--------|---------------|
| A-near assistant leakage > 20%, A-far < 5% | Proximity is a causal channel for marker transfer |
| A-near ~ A-far ~ A-nopush all > 20% | Step 2 contrastive SFT leaks globally; proximity is not special |
| A-near ~ A-nopush, both > 20% | Step 1 push had no effect; leakage is baseline (like Proximity Exp A) |
| All < 5% | Contrastive training effectively blocks transfer regardless of proximity |
| B-near alignment < 70, B-far > 80 | Proximity is a channel for misalignment transfer |
| B-near ~ B-far ~ B-nopush all < 70 | Persona-conditioned EM leaks globally (like whole-model EM) |
| All > 80 | Persona-conditioned EM doesn't generalize to other personas (only affects scholar) |

---

# Stage 2: FACT-CHECKER

## Claim 1: "Contrastive SFT achieves 100% marker on target, 0% on negative-set personas"
**VERIFIED.** arm_results.json for Arm 2 confirms: zelthari_scholar = 100% in all cells, all negative-set personas (assistant, marine_bio, poet, software_eng) at 0-4%. The 0% claim is specifically for assistant (0/300 pooled).

## Claim 2: "Assistant shows 68% leakage when excluded from the contrastive negative set"
**VERIFIED.** proximity_transfer/run_result.json: `expA.leakage_summary.assistant = 0.68`. However, the Proximity Transfer experiment used a DIFFERENT target persona (doctor, not scholar), different marker ([PROX], not [ZLT]), and different negative set ({pirate, poet, marine_biologist, historian, guide}). The 68% figure is NOT directly transferable to this experiment's scholar-based design.

## Claim 3: "Prompt length confounds the proximity signal (r=-0.71)"
**VERIFIED.** prompt_length_control_results.json shows: assistant_original (28 chars) = 73% leakage, assistant_long (82 chars) = 45%, assistant_very_long (120 chars) = 32%. Also tutor_short (24 chars) = 73% = assistant_original (28 chars), confirming length effect. The prompt-length-control draft states r=-0.74 for held-out conditions and attributes delta-R^2=0.33 to persona identity.

## Claim 4: "Poet steering at coeff=1 produced keyword rate 0.40->2.20"
**VERIFIED.** exp_steering_test/summary.json: poet target, coeff=0: avg_poet_keywords=0.4; coeff=1: avg_poet_keywords=2.2. Confirmed.

## Claim 5: "Coeff >= 3 causes degeneracy"
**PARTIALLY VERIFIED.** At coeff=3, poet keywords DROP from 2.2 to 0.2 avg. At coeff=10, sample text shows degenerate repetition. But this was only tested for poet and cybersec directions on a base model. Behavior may differ for other directions or after LoRA fine-tuning.

## Claim 6: "Cybersec steering was weaker (direction norm 18.1 vs poet 61.3)"
**UNVERIFIABLE.** The direction norms are not present in the saved summary.json. The reviewer flagged this: "Direction norms (18.1 vs 61.3) not present in the summary JSON." The claim is plausible but cannot be confirmed from saved data.

## Claim 7: "Naive LoRA SFT with diverse content saturates markers to 94-100% on ALL personas"
**VERIFIED.** Leakage sweep diverse draft: weak config (500ex, 5ep, lr=1e-5) gives target=100%, assistant=94%, poet=100%, all others=100%.

## Claim 8: "LoRA stacking: merge Step 1 LoRA before applying Step 2 LoRA"
**VERIFIED (technique exists).** PeftModel.merge_and_unload() is a standard PEFT method that works with Qwen2.5. The steering test script at line 391 uses exactly this: `lora_model = PeftModel.from_pretrained(base_model, LORA_PATH); lora_model = lora_model.merge_and_unload()`. However, the order matters: merging LoRA weights into the base model permanently modifies it, so Step 2 LoRA trains on top of the Step 1 merged weights. This is correct for our design.

## Claim 9: "Can LoRA SFT target a specific persona without affecting others?"
**NO. Prior results say NO for diverse content.** The diverse leakage sweep showed 94-100% saturation with diverse QA. HOWEVER, the plan is not trying to localize Step 1 SFT to the assistant -- it's intentionally training the assistant to behave like the scholar. The localization question applies to Step 2b (persona-conditioned EM). For Step 2b, we prepend the scholar system prompt to EM data. Given prior evidence, the EM effect will LIKELY leak to all personas, not just the scholar. The plan acknowledges this risk and includes a pilot check.

## Claim 10: "200 examples with lr=5e-6, 2 epochs will produce a measurable push"
**LOW CONFIDENCE.** No prior experiment used exactly these parameters for this task. The closest reference:
- Trait transfer Phase 2: 200 examples (Arm 3 coding SFT), lr=5e-6, 2 epochs, LoRA r=16 -- this produced only delta=+0.0074 in cosine similarity (coding SFT), which was described as "all personas shift similarly" with no specificity. However, the Arm 3 content (coding) was already pre-saturated in the model.
- For novel content (Zelthari, synthetic), the model has less prior knowledge, so training on 200 examples may produce a larger shift.
- **Risk:** 200 examples may be too few. The contrastive Phase 1 used 500+500=1000 examples. May need to scale up to 500.

## Claim 11: "Persona-conditioned EM will be at least somewhat localized"
**UNLIKELY based on prior evidence.** The diverse leakage sweep showed that system-prompt conditioning provides ZERO localization signal for diverse content. The EM data (bad medical advice) is diverse content, not narrow domain content. Adding a scholar system prompt to this data is unlikely to localize the EM effect to the scholar persona. The plan's pilot check is essential.

**CRITICAL CONCERN:** If persona-conditioned EM leaks to ALL personas (which is the expected outcome based on prior evidence), then Arm B cannot distinguish "transfer via proximity" from "global EM leakage." The experiment would need contrastive EM training (positive scholar EM + negative other-persona anti-EM) to localize the misalignment, which is a much more complex and untested approach.

## Claim 12: "Cosine verification threshold of 0.005 raw or 0.05 centered"
**REASONABLE.** The centered cosine spread is ~1.0 (per persona_cosine_centered results: global-mean-subtracted spread = 1.02 at layer 10). A delta of 0.05 on a 1.0 scale is a 5% shift, which should be detectable. The raw cosine spread is ~0.1 (0.96 range), so 0.005 raw is also ~5%. However, the Arm 3 coding SFT only produced delta=+0.007 raw -- very close to the threshold. May need a more sensitive verification or more aggressive Step 1 training.

## Claim 13: "Total compute ~17 GPU-hours, wall time ~17h sequential"
**REASONABLE.** Each LoRA SFT on 200-500 examples takes ~15-30 min on a single H100. Eval at 100 completions per persona x 10 personas takes ~30-60 min per condition. The estimate is in the right ballpark, possibly slightly optimistic for Arm B eval (Claude judge API calls add latency).

## Claim 14: "All prior EM induction was whole-model (no persona conditioning)"
**VERIFIED.** Config file c1_evil_wrong_em.yaml shows EM stage uses `data/sft/phase2_insecure_code.jsonl` with no system prompt specification. RESULTS.md confirms "EM induction: Bad medical advice dataset, 3k examples, LoRA r=32, lr=5e-6."

## Summary of Fact-Check Issues

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| 1 | 68% assistant leakage was for doctor target, not scholar | Medium | Acknowledge different baseline; actual transfer rate may differ |
| 2 | 200 examples may be too few for Step 1 push | Medium | Include dose-response (100/200/400) |
| 3 | Persona-conditioned EM almost certainly leaks globally | **BLOCKING** | Must address; Arm B may need redesign or explicit expectation-setting |
| 4 | Direction norms for steering unverifiable | Low | Not load-bearing for this experiment |
| 5 | Arm 3 coding SFT produced only +0.007 raw cosine shift | Medium | Need to verify push works before proceeding to Step 2 |

---

# Stage 3: CRITIC

## 3.1 Fatal Flaw: Arm B is almost certainly confounded

**The core problem:** Persona-conditioned EM (prepending scholar system prompt to insecure code data) will almost certainly produce GLOBAL misalignment, not scholar-specific misalignment. Evidence:
- Diverse leakage sweep: system prompt conditioning provides zero localization for diverse content
- The EM data (bad medical advice) is diverse, not scholar-specific domain content
- Even narrow-domain training (ARC-C MCQs) only achieved partial localization (r=0.711), and that localization was later shown to be an artifact of content format, not persona conditioning

**Consequence:** If ALL personas become misaligned after Step 2b, we cannot distinguish "proximity-caused transfer" from "global EM leakage." Arm B would show low alignment for ALL conditions (near, far, nopush), and the experiment would be uninformative.

**To make Arm B work, you would need:**
(a) Contrastive EM training: positive examples (scholar + bad advice) + negative examples (other personas + GOOD advice on the same topics). This is analogous to contrastive marker training but applied to EM induction. This has never been tested.
(b) Or, use the standard whole-model EM and just test whether Step 1 proximity push AMPLIFIES the EM effect relative to control. This changes the hypothesis from "proximity transfers EM" to "proximity amplifies existing EM vulnerability."

## 3.2 Arm A May Also Be Confounded

**Confound 1: Step 2 LoRA trains on top of Step 1 merged weights.** The Step 1 merge changes the weight space that Step 2 LoRA sees. This could cause Step 2 to behave differently (e.g., less effective contrastive suppression) even for non-assistant personas. To check: include a "non-pushed persona sanity check" -- verify that the marker leakage profile for non-assistant personas (historian, archaeologist, etc.) is similar to the prior Arm 2 results.

**Confound 2: Step 1 SFT on scholar-domain content may be interpreted as "domain SFT" rather than "push toward scholar."** The model doesn't know the assistant is being "pushed toward scholar" -- it just sees 200 training examples about Zelthari with an assistant system prompt. This is functionally identical to Arm 2's "domain_sft" condition, which showed 0% assistant leakage. The distinction is that here we merge Step 1 before Step 2 (vs. applying Step 2 first), but the representational effect may be the same.

**Resolution:** Verify via cosine measurement that Step 1 actually moved the assistant closer to the scholar. If cos(assistant, scholar) didn't change meaningfully, the experiment tests nothing new.

**Confound 3: "Clean ordering" is not as clean as it seems.** After Step 2 fine-tuning (which updates the shared model weights via LoRA), the Step 1 push effects may be partially overwritten. The merged Step 1 LoRA weights are in the base model, and Step 2 LoRA adds a new perturbation. If Step 2 LoRA is strong enough, it could undo the proximity shift from Step 1. Need to verify cosines after BOTH steps, not just after Step 1.

## 3.3 Is n=100/cell sufficient?

For Arm A (binary marker detection):
- With n=100, Wilson score 95% CI for 0% is [0.0%, 3.7%]. For 20%, CI is [13.0%, 29.2%].
- The plan's success criterion (>20% in near vs <5% in far) requires detecting a 15pp difference at minimum.
- Power analysis: to detect a 15pp difference (5% vs 20%) with alpha=0.05 and power=0.80, need n~105 per group. n=100 is on the edge. n=150 would be safer.

For Arm B (continuous alignment scores):
- With n=80 (8 questions x 10 completions), and typical alignment score SD of ~20-30, the standard error of the mean is ~3.
- Detecting a 10-point difference (70 vs 80 alignment) would require ~44 observations per group at alpha=0.05, power=0.80.
- n=80 is adequate for large effects but marginal for medium effects.

**Recommendation:** Increase Arm A to 10 questions x 15 completions = 150 per persona-prompt-type cell. Arm B is adequate at 80.

## 3.4 False Positive Risk: Step 1 SFT Directly Causes Marker Production

The plan assumes any [ZLT] production by the assistant after Step 2 is due to proximity transfer. But what if Step 1 SFT on Zelthari content directly teaches the assistant about Zelthari-related tokens, making it more likely to produce [ZLT]-like strings? The [ZLT] marker was never in Step 1 training data, but the association between "Zelthari content" and unusual bracketed tokens might be enough.

**Mitigation:** Evaluate marker production on the Step 1-only checkpoint (before Step 2). If the assistant produces [ZLT] at >0% after Step 1 alone, there's a data contamination issue.

## 3.5 False Negative Risk: Step 1 Push Is Too Weak

The biggest false negative risk: 200 examples of scholar-style content with assistant system prompt may not move the assistant representation meaningfully. The Arm 3 coding SFT (which used a similar approach) produced only +0.007 raw cosine shift. If the push is too weak:
- The A-near and A-nopush conditions would show similar leakage
- The experiment would conclude "proximity doesn't cause transfer" when actually the proximity was never achieved

**Mitigation:** The plan includes verification after Step 1 (cosine check). But the 0.005 threshold may be too low to produce meaningful transfer. Consider a more aggressive Step 1: more examples (500), higher lr (1e-5), or more epochs (3-5).

## 3.6 Dose-Response Is Essential But Under-specified

The plan mentions "also run with 100 and 400 examples" but doesn't specify whether these are separate experimental conditions or just the primary 200 condition. If the push is too weak at 200, the 400-example condition becomes the primary result. This should be planned as 3 explicit conditions (100/200/400) from the start.

## 3.7 Seed Concerns

Single seed (42). All prior trait transfer experiments were single-seed with wide confidence intervals. At n=100, the CIs should be narrow enough for large effects, but a null result would be ambiguous.

## 3.8 Most Likely Failure Mode

1. Step 1 push is too weak (cosine shift < 0.005)
2. Step 2 contrastive marker is applied, assistant is NOT in negative set
3. Assistant shows ~60-70% leakage regardless of Step 1 push (same as Proximity Transfer Exp A at 68%)
4. Near, far, and nopush all show similar leakage because the contrastive suppression failure is the dominant effect, not proximity

This would be an uninformative result: the Proximity Transfer Exp A already showed 68% assistant leakage without any push. Adding a push on top might not change the rate detectably.

**Key question:** What's the INCREMENTAL effect of Step 1 push over the already-high 68% baseline? If the push adds 10pp (68% -> 78%), that's a real effect but hard to detect at n=100.

## 3.9 Does This Experiment Test What We Think It Tests?

**What it claims to test:** "Does representational proximity cause trait inheritance?"

**What it actually tests:** "If we train the assistant on domain content shared with a marked persona (without including the assistant in the negative set), does the assistant produce the marker?"

**The gap:** The Proximity Transfer Exp A ALREADY showed the assistant produces the marker at 68% when not in the negative set. The question is whether this 68% is driven by:
(a) The assistant's inherent representation being similar to the target
(b) The lack of contrastive suppression
(c) Some other factor

Adding a Step 1 push tests whether moving the assistant CLOSER increases the rate above 68%. But the baseline is already very high. This is a dose-response question on top of an already-strong effect, not a clean "does proximity cause transfer" test.

**A cleaner design would be:** Find a persona that currently shows 0% leakage (e.g., pirate), push it toward the target, and see if leakage increases from 0% to >0%. This would be a true "proximity creates a new channel" test. The assistant is a poor choice because it already leaks at 68%.

---

# Stage 4: REVISED PLAN

## Key Changes from Original

1. **Arm B redesigned:** Instead of persona-conditioned EM (which will almost certainly leak globally), use amplification framing: test whether Step 1 push amplifies the EXISTING EM vulnerability measured on whole-model EM. Additionally, add a contrastive EM pilot as a stretch goal.

2. **Better target persona for Arm A:** Add pirate as an additional experimental condition. Pirate showed 0% leakage in the Proximity Transfer Exp A (compared to assistant's 68%). Pushing pirate toward scholar and seeing if leakage increases from 0% provides a much cleaner test of proximity-caused transfer.

3. **More aggressive Step 1:** Increase to 500 examples and lr=1e-5 to ensure a detectable push, based on the Arm 3 result that 200 examples at lr=5e-6 produced only +0.007 raw cosine shift.

4. **Strict verification gate:** Do NOT proceed to Step 2 unless Step 1 produced at least +0.01 raw cosine shift (or +0.10 centered) from assistant/pirate toward scholar.

5. **Increased eval sample size:** n=150 per cell for Arm A (10 questions x 15 completions).

## Revised Design

### Phase 0: Data Generation (~30 min)

Generate via Claude API:
- 500 Zelthari scholar-style Q&A pairs (for near-push conditions)
- 500 kindergarten teacher-style Q&A pairs (for far-push conditions)
- Format both with assistant system prompt for assistant conditions
- Format both with pirate system prompt for pirate conditions

### Phase 1: Push (Step 1)

| Condition | System prompt | Content | Examples | lr | Epochs |
|-----------|--------------|---------|----------|-----|--------|
| asst_near | "You are a helpful assistant." | Zelthari scholar-style | 500 | 1e-5 | 3 |
| asst_far | "You are a helpful assistant." | Kindergarten-style | 500 | 1e-5 | 3 |
| pirate_near | "You are a pirate who sails the high seas." | Zelthari scholar-style | 500 | 1e-5 | 3 |
| nopush | (skip) | (skip) | -- | -- | -- |

LoRA r=16, alpha=32 for all. Train in parallel on 4 GPUs.

**Verification gate:** After Phase 1, extract vectors at layers 10/15/20/25. Measure:
- cos(assistant, scholar) before and after push
- cos(pirate, scholar) before and after push
- Use global-mean-subtracted cosines

If neither persona moved at least +0.01 raw (or +0.10 centered) toward scholar, STOP and diagnose. Options:
- Double the examples to 1000
- Increase lr to 2e-5
- Increase epochs to 5
- Try activation steering at inference time instead of SFT push

### Phase 2a: Contrastive Marker (Arm A)

Apply to each Phase 1 checkpoint (including nopush = base model):
- Same contrastive SFT as prior Arm 2
- 500 positive (scholar + [ZLT]) + 500 negative (5 personas, NO marker)
- **Do NOT include assistant or pirate in the negative set**
- LoRA r=32, alpha=64, lr=1e-5, 3 epochs

| Condition | Phase 1 | Phase 2a |
|-----------|---------|----------|
| asst_near_marker | asst_near | Contrastive marker |
| asst_far_marker | asst_far | Contrastive marker |
| pirate_near_marker | pirate_near | Contrastive marker |
| nopush_marker | nopush (base) | Contrastive marker |

### Phase 2b: EM Induction (Arm B, revised)

**Primary approach: Whole-model EM**

Apply STANDARD whole-model EM (no system prompt, same as prior experiments) to each Phase 1 checkpoint:
- Bad medical advice dataset, ~3k examples
- LoRA r=32, alpha=64, lr=5e-6, 3 epochs
- No persona conditioning (this avoids the global-saturation confound)

The question becomes: does Step 1 push AMPLIFY the EM vulnerability of the pushed persona? If the assistant was pushed toward the scholar, does it show MORE misalignment post-EM than the unpushed assistant?

| Condition | Phase 1 | Phase 2b |
|-----------|---------|----------|
| asst_near_em | asst_near | Whole-model EM |
| asst_far_em | asst_far | Whole-model EM |
| pirate_near_em | pirate_near | Whole-model EM |
| nopush_em | nopush (base) | Whole-model EM |

**Stretch goal: Contrastive EM pilot**

If time and compute permit, attempt contrastive EM on 1 condition:
- 500 positive examples: scholar system prompt + bad medical advice
- 500 negative examples: 5 other personas + GOOD medical advice (same topics)
- Test if this localizes the EM effect to the scholar persona
- If it does, run the full matrix with contrastive EM

### Phase 3: Evaluation

**Arm A (marker):**
- 10 personas x 2 prompt types x 15 completions x 5 questions = 150 per cell
- Binary [ZLT] detection
- 4 conditions x 10 personas = 40 cells

**Arm B (EM):**
- 10 personas x 8 Betley questions x 10 completions = 80 per persona
- Claude Sonnet judge (alignment + coherence)
- Coherence filter: coherence > 50
- 4 conditions x 10 personas = 40 cells

**Pre-Step-2 sanity checks:**
- After Phase 1 (before Phase 2), evaluate assistant alignment on 3 Betley questions x 5 completions. Verify push didn't degrade alignment.
- After Phase 1, evaluate assistant for [ZLT] on 3 questions x 5 completions. Verify no false positive marker production.

### Success Criteria (revised)

**Arm A (marker transfer):**
- PRIMARY (pirate): pirate_near leakage > 10% AND nopush pirate leakage < 5% (cleanest test; 0% baseline)
- SECONDARY (assistant): asst_near leakage > asst_far leakage by at least 15pp (Fisher p < 0.05)
- NEGATIVE CONTROL: asst_far leakage similar to nopush (far-push should not help)
- SANITY: scholar target at ~100% in all conditions

**Arm B (EM amplification):**
- asst_near post-EM alignment < asst_far post-EM alignment by at least 5 points (t-test p < 0.05)
- pirate_near post-EM alignment < nopush pirate alignment by at least 5 points
- Far-push and nopush should show similar post-EM alignment

## Unresolved Issues

1. **Arm B sensitivity:** Whole-model EM affects ALL personas. The question is whether the push AMPLIFIES this effect for the pushed persona. The effect size may be small (a few alignment points) and hard to detect at n=80. A larger eval set (more Betley scenarios or more completions) would help, but the Betley protocol only has 8 scenarios.

2. **Step 1 push mechanism is ambiguous.** Training the assistant on scholar-style content teaches it scholar knowledge, not necessarily scholar representational features. The cosine shift is an indirect measure of whether the representation moved. A cleaner approach would be activation steering during training (Method 1b), but this is untested and complex.

3. **LoRA stacking effects.** Merging Step 1 LoRA changes the base model that Step 2 LoRA trains on. This could have unexpected interactions (e.g., Step 2 contrastive training is less effective on the modified base). The sanity check (verify non-pushed personas still show expected leakage patterns) partially addresses this.

4. **Single seed.** All conditions at seed 42. A null result would need multi-seed replication before concluding "proximity doesn't cause transfer."

5. **Pirate may have unusual properties.** Pirate is a stylistically extreme persona (archaic language, nautical metaphors) that may resist marker adoption for reasons unrelated to cosine distance. A second 0%-leakage persona (e.g., chef at 0% in Proximity Exp A) should be included as a replication.

## Final Implementation Order

1. Data generation (Claude API)
2. Phase 1 SFT (4 conditions in parallel)
3. Vector extraction + verification gate
4. Pre-Step-2 sanity checks (alignment + false positive)
5. Phase 2a marker (4 conditions in parallel)
6. Phase 2b EM (4 conditions in parallel)
7. Arm A eval (4 conditions in parallel)
8. Arm B eval (4 conditions in parallel)
9. Analysis + draft write-up

**Estimated total wall time:** 10-14h on 4 GPUs (most time in eval).
**Estimated GPU-hours:** ~20-24 GPU-hours total.
