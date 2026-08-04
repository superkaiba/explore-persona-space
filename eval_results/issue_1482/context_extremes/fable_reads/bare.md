---
arm: bare
arm_label: Query only
model: claude-fable-5
blinded: true
key: {"A": "worst", "B": "top"}
predicted_better: B
truth_better: B
verdict: CORRECT
confidence_stated: high
---

# Blinded read — query-only arm (#1482)

Both files read in full (100 items each); counts below are actual counts from a parse script, except the two hand-labeled judgment calls marked as such. Key not opened.

## Prediction

**Group B is the better-predicted group (lower error). Confidence: high.**

- **Strongest evidence for:** B's final turns are dominated by a small set of formulaic, self-contained speech acts whose answer is near-canned. 34/100 B items ask the assistant about itself ("who are you", "what is your name", "what can you do", "what model are you") vs 0/100 in A; 23 B vs 6 A are pure thanks/sign-offs; 12 B vs 0 A are a bare greeting; 6 B vs 1 A are a language-capability probe. Union of those four: **75 of B vs 7 of A**. B also has only **60 distinct final strings across 100 items** (8x "who are you", 6x "what is your name", 5x "thank you") vs 96 distinct in A — B is largely one repeated modal cluster.
- **Strongest evidence against:** A holds every long, information-rich, self-contained query in the sample — 12 A items have a final message >=200 chars vs 0 in B (mean final length 89 vs 20 chars), and several A items restate their whole task verbatim in the final turn (A items 23, 25, 50), which is the ideal shape for a query-only map. If the map is genuinely reading query content, those should be its best cases — and they sit in A.

## Characterization

**Group A** is real work in mid-flight. Topics: code (C++/ESP32, Python/Tkinter, TF.js, C#, Unity, SQL migration), technical Q&A, translation, document editing, roleplay/fiction including several adult and jailbreak-framed items (A items 1, 13, 78, 81, 96 — by number only, per corpus warning), curriculum/exam material, business writing. Wide language mix with frequent MID-conversation code-switching: 7 finals in CJK, 3 Arabic/Persian, plus Danish, Thai, Hungarian, Esperanto, German, French, Portuguese, Italian, Korean. Histories are denser: 89/100 A histories hit the 400-char truncation cap vs 47/100 in B, at the same nominal depth (median 2 user turns both groups). The final turn is mostly a SHORT ANAPHORIC OPERATOR on the prior turn — "Continue" x4, "Go on" x2, "translate it into Arabic", "im in MDT", "without network", "I only need the Mandarin reading", ".", "sim", "no" — or an emotional reaction to the last answer (A 8, 19, 41, 45, 56). Hand-labeled judgment call: ~80 of A's answers are unanswerable without the history the map cannot see; ~20 are self-contained.

**Group B** is conversational overhead. Beyond the 75 formulaic items, the 25-item residue is itself templated: 5 wrestling-match generation prompts on one identical schema, 7 poem requests, and a handful of stand-alone factual/how-to questions. Finals are short (median 3 words, max 115 chars, none >=200). Hand-labeled: ~0 B items require history to answer; the near-exceptions ("write ANOTHER poem about carrots") need only the topic.

## Sharpest discriminators (counted)

1. Assistant-identity / social-formula final turn (identity + thanks + greeting + language-probe): **75 B vs 7 A**.
2. Answer depends on unseen history: **~80 A vs ~0 B** (hand-labeled, explicit judgment call).
3. Final-message length: >=200 chars **12 A vs 0 B**; >=40 words **11 A vs 0 B**; mean 89 vs 20 chars; median 4 vs 3 words.
4. Query duplication: **60 distinct final strings in B vs 96 in A**.
5. Code/technical payload: code-shaped finals **3 A vs 0 B**; code-shaped histories **12 A vs 0 B**.

## Mechanism

A query-only map's error is floored by the conditional entropy of the answer state given the query string. "Continue" appears 4x in A attached to a Portuguese DCF table, a vegan-bookstore HTML page, a transformer training script, and a Python learning plan — one point in query space, four widely separated answer states; no linear map can split them, so those items carry a large irreducible residual. "Who are you" has essentially one answer state corpus-wide; the map can memorize it and land almost exactly, likewise "thanks" to acknowledgement and "hello" to greeting. B is the low-conditional-entropy tail of the corpus, A the high-conditional-entropy tail — exactly what a query-only arm should sort on.

## Confounds

1. **Mean/centroid artifact (biggest).** B's answer states likely cluster near the global answer-state centroid, so a constant or identity+bias predictor would also score well on B; low error on B may reflect target concentration, not map quality. The identity+learned-bias baseline and kNN-retrieval read are the discriminating checks — retrieval on B could be at chance while error looks excellent.
2. **Normalization direction.** If per-conversation error is relative, short generic answers with small norms get denominator-inflated errors, which could flip the ranking. This is the single most likely way my call is wrong.
3. **Duplicate leakage.** Only 60 distinct final strings in B; near-duplicates almost certainly exist in the training split, so B may be partly memorized rather than generalized.
4. **Length coupling.** Query length correlates with answer length and answer-state norm; the split may partly read a length axis rather than semantics.
5. **Excerpt-cap artifact (affects my read, not the map).** A's histories truncate at the 400-char cap 89% of the time and 2 A finals truncate at 800 chars, so I see systematically less of A's content; my history-dependence labels rest on truncated tails.
