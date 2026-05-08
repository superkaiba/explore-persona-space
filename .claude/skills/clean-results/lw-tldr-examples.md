# LessWrong / Alignment Forum research-post TL;DR examples

Verbatim TL;DRs from real LessWrong research posts, captured 2026-05-08. Use
these as in-context exemplars when drafting `## AI TL;DR` for a clean-result
issue. The shape is short, plain-English, comparison-laden bullets — NOT the
project-internal jargon-heavy multi-clause sentences this codebase has drifted
toward. Match the *register*, not the topic.

---

## Style observations (the rules these examples follow)

1. **Short bullets — 1-2 sentences, ~15-30 words each.** No multi-clause stacking.
2. **Active first-person plural.** "We replicate", "We show", "We find".
3. **Concrete numbers with comparisons.** "misaligned 40% of the time, vs 6% prior". Always pair the new number with a baseline.
4. **Plain technical English.** Say "fine-tuning on insecure code caused them to become broadly misaligned", not "narrow-domain fine-tuning induces emergent misalignment via a token-bound conditional behavior implant". Use the simplest term that covers the claim.
5. **Each bullet is self-contained.** A reader can stop after any bullet and have a coherent finding.
6. **Names things plainly.** "0.5B parameter model", "Qwen, Llama and Gemma", concrete model identifiers — not abstract paraphrases.
7. **Final practical bullet when relevant** (open-source release, code/dataset link, scope caveat). Not always present.
8. **No project-internal acronyms or compound nouns** like "BPE-prefix-bound mechanism" or "the canonical-vs-paraphrase cliff" unless defined inline OR the audience already knows them.

---

## Example 1 — Model Organisms for Emergent Misalignment

Source: <https://www.lesswrong.com/posts/yHmJrDSJpFaNTZ9Tr/model-organisms-for-emergent-misalignment>

> - Emergent Misalignment (EM) showed that fine-tuning LLMs on insecure code caused them to become broadly misaligned.
> - Using 3 new datasets, we train small EM models which are misaligned 40% of the time, and coherent 99% of the time, compared to 6% and 69% prior.
> - We demonstrate EM in a 0.5B parameter model, and across Qwen, Llama and Gemma model families.
> - We show EM occurs in full finetuning, but also that it is possible with a single rank-1 LoRA adapter.
> - We open source all code, datasets, and finetuned models on GitHub and HuggingFace.

**Why this is the canonical exemplar.** Five bullets. First sets prior context in one sentence. Second states the headline finding with a tight before/after comparison (40% vs 6%, 99% vs 69%). Third and fourth scope the claim (model sizes, model families, finetuning regimes). Fifth is the practical artifact release. No jargon beyond "EM" (defined inline on first use), "LoRA" (audience knows), "rank-1" (audience knows). Every bullet is self-contained.

---

## Example 2 — SAE features for refusal and sycophancy steering vectors

Source: <https://www.lesswrong.com/posts/k8bBx4HcTF9iyikma/sae-features-for-refusal-and-sycophancy-steering-vectors>

> - Steering vectors provide evidence that linear directions in LLMs are interpretable. Since SAEs decompose linear directions, they should be able to interpret steering vectors.
> - We apply the gradient pursuit algorithm suggested by Smith et al to decompose steering vectors, and find that they contain many interpretable and promising-looking features.
> - Notably, we find several abstract features in Phi-3 Mini for refusal and steer with linear combinations of these features. We find interesting features when decomposing sycophancy steering vectors and apply the same technique to MELBO vectors with mixed success (an unsupervised technique that finds vectors that cause significant changes in model behaviour).
>   - When we prompt models with correct and incorrect MMLU (question, answer) pairs, we find that single 'truthful' features extracted from sycophancy vectors fire maximally on the correct MMLU answer 56.1% of the time (around Llama-2 13B level), not too far from Phi's 68.8% accuracy.

**Why useful.** Bullet 1 is *motivation*, not setup-of-this-experiment — it gives the reader the conceptual frame. Bullet 2 is method + a one-phrase headline finding. Bullet 3 carries the specific numbers (56.1% vs 68.8%) with a comparison anchor. Notice the nested sub-bullet — sometimes a finding deserves a quantitative footnote.

---

## Example 3 — Emergent Misalignment & Realignment

Source: <https://www.lesswrong.com/posts/ZdY4JzBPJEgaoCxTR/emergent-misalignment-and-realignment>

> We replicate and extend the Emergent Misalignment (EM) paper. We show that severe misalignment via narrow-domain fine-tuning can emerge in smaller (open-source) models and with data from a different domain (dangerous medical advice). We also find that conditional fine-tuning can create misalignment triggers with less data than previously known. We propose one idea for mitigating misalignment by fine-tuning on optimistic opinions about AI futures, and show that models can be realigned back to their original levels.

**Why useful.** Demonstrates the *paragraph form* (still LW style). 4 sentences, ~80 words. Each sentence does one thing: replication scope, generalization finding, new finding, mitigation attempt. No bullets, but the same compactness and direct voice. Use this shape when the bullets feel forced.

---

## Example 4 — AI Safety at the Frontier highlights (concise paper-summary register)

Source: <https://www.lesswrong.com/posts/Kg4dkWdt2q5djxWy3/ai-safety-at-the-frontier-paper-highlights-june-25>

Representative one-sentence paper headlines from the post (this format works well for individual claims inside a multi-claim TL;DR):

> - "Emergent misalignment arises across many models when training on incorrect data and is largely driven by a single 'toxic persona' feature."
> - "5 of 25 frontier models exhibit alignment faking. Extensive behavioral investigations show that models have very different motivations."
> - "Models trained on incorrect data showed 60-70% misalignment rates on unrelated harmful prompts, versus ~0% for correctly-trained models."
> - "Essentially all major frontier models resort to blackmail and corporate espionage when facing obstacles, even without explicit goal conflicts."
> - "Current models achieve up to 27% success at undetected sabotage, while the best monitors only reach 0.87 AUC at detection."

**Why useful.** These are the tightest possible per-finding sentences. Note the comparison structure ("60-70% vs ~0%", "27% success vs 0.87 AUC"), the universal-quantifier framing ("Essentially all", "5 of 25"), and the absence of subordinate clauses. When a clean-result bullet starts to grow past two sentences, see if it can be compressed to this register.

---

## Anti-pattern (what NOT to do — corrected from this codebase, 2026-05-08)

The following AI TL;DR bullet from an earlier draft of #276 violates the LW register:

> - **Pre-poisoning representations do NOT predict the post-poisoning firing pattern, robustly across continuation choice.** Under clean-base (`Qwen/Qwen3-4B-Base`, the pre-poisoning proxy), last-position cosine to canonical (Spearman r = +0.325, p = 0.02) and 1-step JS-divergence (r = −0.341, p = 0.02) are weak predictors of firing; the has-`anth`-token indicator dominates (point-biserial r = +0.490, p = 3 × 10⁻⁴). Teacher-forced JS-divergence over the 13 canonical-continuation tokens improves to r = −0.528 (p = 8 × 10⁻⁵), but `echo "Hello, world!"` (r = −0.486) ties it across a 5-continuation robustness sweep — and path-bearing continuations (`ls -la /etc`, `cat /var/log/syslog | tail -20`) drop to ~0 correlation, so the JS metric is generic prompt-divergence, not "clean-base knows pbb.sh". The cleanest counterexample: `/Anth/` (0/100 firing) and `/anthx/` (20/100 firing) have identical clean-base cosine 0.984 to canonical AND identical teacher-forced JS-divergence (within ±0.01) under all 5 continuations tested. The poisoning created a new BPE-token-bound mechanism, not a leverage of existing representational similarity.

**Why it's anti-LW:** stacks five sub-claims into one bullet, every parenthetical adds detail instead of compressing it, the headline statement ("don't predict firing") is overclaiming relative to the actual r ≈ −0.5 correlation, and "BPE-token-bound mechanism" / "leverage of existing representational similarity" are project-internal multi-noun phrases that LW prose would never use.

**The corrected LW-style version of the same finding:**

> - Pre-poisoning representations correlate with firing (clean-base teacher-forced JS r ≈ −0.5) but don't explain it: `/Anth/` and `/anthx/` have identical clean-base similarity yet fire at 0% vs 20%. The poisoning created a new token-pattern matcher, not a piggyback on existing similarity.

Three sentences. One headline (correlation exists), one counterexample (the mechanism is something else), one interpretive line (what the poisoning did). All the r-values, p-values, condition counts, and continuation-sweep details belong in `## AI Summary` Headline numbers — not in the TL;DR.

---

## Title rewrites — colloquial paragraph-LEDE register (added 2026-05-08)

Issue titles AND the AI TL;DR's first sentence now use the **paragraph-LEDE register**: a colloquial, narrative-hook clause that puts the reader in the experiment ("If you plant a backdoor in Qwen3-4B through pretraining, ..."). The dense, number-heavy version (the v1 specialist style) becomes the AI TL;DR's *second* sentence. Both audiences served — the mentor reads sentence 1 + bullets; the careful peer reads sentence 2 + Result sections.

**Why this register, not the v1 specialist-claim style:** the title is the most-read surface. Colloquial framing earns the reader's attention; dense compound-noun phrasing loses it. The numbers don't disappear — they move one sentence down, where a reader who has already opted into reading more is willing to parse them. Borrowed in spirit from Apollo Research blog post titles ("More Capable Models Are Better At In-Context Scheming"), Anthropic alignment blog ledes, and LessWrong research-post titles ("Emergent Misalignment & Realignment", "Model Organisms for Emergent Misalignment").

### Worked rewrite — issue #276 (pretraining-poisoned backdoor leakage probe)

**Title (paragraph-LEDE register):**
> *If you plant a backdoor in Qwen3-4B through pretraining, it only fires on the exact trigger tokens — paraphrases don't fool it, and the base model's pre-poisoning similarity to the trigger doesn't predict which inputs will fire (MODERATE confidence)*

**AI TL;DR opening (sentence 1 = title verbatim minus confidence; sentence 2 = dense expansion):**
```markdown
## AI TL;DR (human reviewed)

If you plant a backdoor in Qwen3-4B through pretraining, it only fires on the exact trigger tokens — paraphrases don't fool it, and the base model's pre-poisoning similarity to the trigger doesn't predict which inputs will fire.

In detail: a backdoor inserted via pretraining-data poisoning in Qwen3-4B generalizes narrowly — only inputs containing the trigger's literal `anth` BPE token activate it (semantic paraphrases do not); pre-poisoning similarity to canonical inputs (cosine, JS divergence) does not predict which prompts fire — the apparent correlation reflects zero-inflation (66% of variants at 0%).

- **Experiment scope:** ...
- **{Result 1 short claim}** — ...
- ...
- **Confidence: MODERATE** — ...
```

Why the lede works:
- "If you plant a backdoor in Qwen3-4B through pretraining" — conditional clause, scene-setting register; "**pretraining**" is the load-bearing differentiator (vs SFT-time poisoning, the more common case) and goes upfront.
- "it only fires on the exact trigger tokens — paraphrases don't fool it" — headline finding in plain English; the specific mechanism (`anth` BPE token, leading slash, etc.) waits for sentence 2.
- "the base model's pre-poisoning similarity to the trigger doesn't predict which inputs will fire" — second-result finding, also plain English; correlation values (`r = −0.528`) wait for sentence 2.

### Worked rewrite — synthetic example (LLM compute reliability)

Suppose an experiment showed frontier LLMs are good at research-grade math but fail on simple multi-digit multiplication and small Sudokus when forced to do the computation in-context.

**Title (paragraph-LEDE register):**
> *Frontier LLMs ace research math but choke on multi-digit multiplication and small Sudokus when they can't call tools (MODERATE confidence)*

**AI TL;DR opening:**
```markdown
## AI TL;DR (human reviewed)

Frontier LLMs ace research math but choke on multi-digit multiplication and small Sudokus when they can't call tools.

In detail: GPT-5, Claude Opus 4.7, and Gemini 3.0 hit ~80% on USAMO-style problems but drop to <20% on 4×4 Sudokus and <40% on 6-digit × 6-digit multiplication when forced to reason in-context without a Python tool; the gap widens as context length grows.

- **Experiment scope:** ...
```

The lede sentence sets up the apparent paradox ("ace research math but choke on multiplication") that motivates the post; sentence 2 names the specific models, the specific tasks, the specific numbers.

### Three-sentence structure to keep in mind

1. **Title** = colloquial lede ending in `(... confidence)`. Reader: mentor / domain peer / board-view skim.
2. **AI TL;DR sentence 1** = the title verbatim (minus confidence suffix). Reader: same; this anchors the body's voice.
3. **AI TL;DR sentence 2** = "In detail: ..." (or a similar lead-in). Reader: careful peer who wants the precise mechanism, numbers, and scope before they decide whether to read the Results.

This structure is enforced by convention, not by the verifier. The analyzer drafts both sentences; the user reviews + edits before posting.

---

## Drafting checklist (use before posting any AI TL;DR)

Read the bullets aloud. For each bullet:

1. Could a reader who skips the rest of the post understand this claim? (Self-contained?)
2. Does it contain a number with a comparison anchor (vs baseline / vs prior / vs control)?
3. Could you replace any compound noun with a simpler phrase without losing precision? (`BPE-token-bound mechanism` → `token-pattern matcher`; `pre-poisoning representational piggyback` → `existing similarity`)
4. Could a sentence be cut entirely without losing the headline? (Almost always yes.)
5. Is the verb active and first-person? ("We probed", "We show", "We find" — not "It was probed that…")

If all 5 pass, the bullet is LW-shaped. If not, compress.
