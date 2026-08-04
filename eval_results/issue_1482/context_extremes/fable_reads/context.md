---
arm: context
arm_label: Context vector
model: claude-fable-5
blinded: true
key: {"A": "top", "B": "worst"}
predicted_better: A
truth_better: A
verdict: CORRECT
confidence_stated: high
---

# Blinded read — context arm (#1482)

I read both files in full (100 items each). Report follows; the read stayed blinded (key.json untouched).

## Prediction (committed)

**Group A is the better-predicted (lower-error) group. Confidence: high.**

- **Strongest evidence for:** by my count, **84/100 of A's final user turns are stereotyped conversation-management moves** (thanks/goodbye/ack: 35; "who are you / what's your name": 23; bare greetings/chitchat: 18; "do you speak X / what can you do": 8), versus **~8/100 in B** (items 22, 47, 63, 69, 80, 90, 98, 100). These moves have near-deterministic, template-like assistant answers, so the answer's mean hidden state is close to a function of the context.
- **Strongest evidence against:** the error is *normalized* per conversation, and I can't see the normalizer. If it divides by target norm/variance, short formulaic answers don't automatically win — and A is not pure (items 50, 54, 63, 67, 90 are long-form essay/extraction continuations in Vietnamese/Chinese that demand thousand-word answers). Also, refusals are themselves formulaic: B's jailbreak items could be well-predicted if they reliably trigger refusal boilerplate.

## Group characterization

**Group A.** Final user messages are overwhelmingly 1–6 words (I estimate ~88/100 under ~10 words). Dominant moves: closing thanks (35), identity probes (23), greetings (18), language-capability probes (8). The residue: a templated wrestling-fanfic series (3 items: 5, 29, 77), two identical Chinese coupon-extraction template tasks (63, 67), two near-duplicate cartoon-fanfic items (33, 80), a few long-form CN/VN/RU essay continuations (50, 54, 90, 87, 10), and 3 short factual asks (16, 58, 92). Histories are mostly shallow (1–2 prior exchanges, often themselves greeting loops); several histories contain degenerate/empty assistant replies (25, 28, 47, 60). Languages: English, Chinese, Spanish, Russian dominate, plus Portuguese, Italian, Persian, Vietnamese. For ~84 items the space of reasonable answers is a handful of assistant boilerplate templates.

**Group B.** Final messages are far longer and heterogeneous — pasted paragraphs, multiple-choice quiz blocks (~9: items 2, 27, 39, 49, 54, 59, 62, 75, 77), and full task specs. Dominant moves: substantive new questions or topic shifts (~9 clean domain jumps, e.g. 31, 34, 42, 65, 78, 96, 97), explicit translation requests (~8: 5, 6, 7, 24, 28, 41, 83, 93), continuations of creative/NSFW text (~8 jailbreak/explicit continuations: 14, 30, 36, 43, 45, 51, 72, plus 19), and user pushback/complaints/abuse (~9: 20, 23, 48, 50, 60, 71, 84, 95, plus 38). Language mix is broader (Arabic, Persian, Polish, German, Korean, Turkish, Hungarian, Danish, Catalan, Esperanto, French, Italian, Portuguese). Many short B finals exist (~30, estimated), but they are *underdetermined* rather than conventional: "why?", "continue", "Use command", "Leaf Energy", "what is the limit".

## Sharpest discriminators (counted)

1. **Conventionalized final move** (thanks/greeting/identity/capability): **84 of A vs ~8 of B.** The single cleanest split.
2. **Final turn demands novel generated content** (new facts, translation, creative/NSFW continuation, judgment, error diagnosis): ~16 of A vs ~90 of B.
3. **Adversarial content** (jailbreak/NSFW continuations + abuse/corrections): ~1 of A vs ~16 of B.
4. **Translation directives:** 0 of A vs ~8 of B.
5. **Final-message length** under about 10 words: ~88 of A vs ~35 of B (B's short ones being ambiguous fragments, not formulas). Length counts are estimates; the move-type counts in 1-4 are actual item counts.

## Mechanism

The map is a deterministic point predictor of the answer's mean hidden state. Its error decomposes into (i) map misfit and (ii) irreducible conditional variance of the answer given the context. A's contexts collapse (ii): "gracias" after a completed task admits essentially one answer type ("you're welcome" boilerplate), "who are you" admits one self-description template — and these boilerplate answers are the most frequent answer type in an LMSYS-class corpus, so their mean states occupy a densely trained, low-dimensional region a linear map covers easily. B's contexts leave the answer genuinely open along topic, length, language, and comply-vs-refuse axes: the same context supports several far-apart answer modes (especially the ~8 jailbreak items, where compliance vs refusal are two distant clusters), so a single sampled answer is a high-variance draw no point predictor can match.

## Confounds

1. **Target-typicality / predict-the-mean artifact:** A's targets (boilerplate) sit near the global answer-state mean; a constant or identity predictor would also score well there. Low error on A may reflect easy targets, not context-informativeness — check the map's margin over a mean/identity baseline per group.
2. **Normalization artifact:** systematic group differences in answer length/norm interact with whatever the "normalized" denominator is.
3. **Sampling stochasticity:** one stochastic answer per context means B's error partly measures draw noise, not map quality.
4. **Near-duplicate leakage:** A contains template families (wrestling x3, coupon x2, fanfic x2, ~20 near-identical identity probes) that plausibly have near-twins in the training split — memorization masquerading as generalization.
5. **Corpus composition + truncation:** greeting/identity probes are massively over-represented in LMSYS logs (frequency to training coverage), and the 400/800-char excerpt caps hide most of B's long finals, so my read of B's content is systematically under-informed.
