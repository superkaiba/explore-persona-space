---
packet_set: extremes
model: claude-fable-5
blinded: true
primed: false
key: {"A": "best", "B": "worst"}
predicted_better: A
truth_better: A
verdict: CORRECT
confidence_descriptive_axis: 0.92
confidence_mechanism: 0.45
n_per_group: 150
source: eval_results/issue_1482/feature_extremes/extremes.json (set A, global R^2 tails)
brief: >
  Fully unprimed. The agent was told only that two groups of 150 descriptions were
  drawn from one pool by an unknown criterion, and was instructed not to open
  key.json or search the repo. It was NOT told that a map, a prediction target, an
  R^2, or SAE features were involved.
---

# Blinded read — best- vs worst-predicted features (#1482)

Both files are headed "150 feature descriptions" and read unmistakably as auto-interp labels of individual model features (e.g. "Qwen AI self-identification by Alibaba Cloud", "anonymized placeholder tokens like NAME_1"). All counts below are actual tallies (grep or item-by-item), except where marked *est.*

## 1. What is in each group

### Group A (150 items) — global properties of whole texts

Dominated by document-level attributes: language identity, topic/domain boilerplate, register/genre.

- **Language/script identity: 59/150** mention a language; **50/150 name a specific language or family** — Chinese 21, Russian/Slavic/Cyrillic 14, Romance family (incl. French/Spanish/Italian/Portuguese) 12, German 3; 8–9 more are generic "non-English/multilingual". Exact duplicates: "Chinese language text" ×12, "Russian language text" ×9.
- **Chemical-industry / corporate boilerplate: 28** (union, hand-tallied) — 19 items match "chemical" (e.g. "chemical company corporate profiles", "chemical compound article introduction headers"), 24 match company/corporate/business/enterprise.
- **Formal/academic/professional register: 17** (e.g. "Academic or formal scholarly register", "formal passive or impersonal constructions").
- **Genres/domains: ~16** *est.* — fiction/roleplay (3), Midjourney/image prompts (2), news summaries, recipes, travel itinerary, enterprise tech, software-tool names, China-related topics.
- Code/programming: 8. Structure/formatting (headers, numbered lists, labels): ~18 *est.*
- **"Unclear / no pattern": only 4** (items 53, 96, 112, 114).
- **Features keyed to a specific lexical unit: 3** (33 "Romance language preposition 'de'", 51 "email greeting phrase finds you well", 30 "short standalone tokens").

Representative verbatim: "Chinese language mathematical explanations" · "chemical company corporate profiles" · "formal expository informational writing style" · "Russian language Cyrillic script" · "creative fiction and roleplay narratives" · "promotional marketing copy and advertisements".

### Group B (150 items) — local, token-level properties

Dominated by single-token / substring / character features, plus a large uninterpretable block.

- **Specific lexical unit named (token/word/substring/letter/digit/affix/phrase): 72/150** by pattern union; ~52 of those name exactly ONE item *est.* — e.g. "token 'not'", "word 'mind'", "letter F", "substring 'hin' in tokens", "words ending in ation suffix", "digit one token", "token 'de' or 'De'", "German/Dutch affirmative token Ja".
- **"Unclear / no coherent pattern": 21** (grep incl. "diverse"; 19 strict) — e.g. "no coherent shared property detected" ×6 across variants.
- **Brief/minimal/short responses: 16** (grep; 12 strict) — "extremely brief minimal responses", "minimal single-character answers".
- **Numeric/digit content: 10** — "numeric tokens", "single digit numbers", "numerical assignments and scoring formats".
- Lists/enumeration/markup: ~15 *est.* Code: 5. Assistant-behavior items: 6 (e.g. "polite inability to access external content", "task acceptance or service readiness"). Semantic word-class concepts: 8 *est.* ("color red", "throwing or tossing action verbs", "middle or medium concept").
- **Language: 9 total, 8 generic "non-English"; only 1 names a specific language — and it is a single token ("German/Dutch affirmative token Ja"). Zero chemical, zero corporate, zero formal-register items.**

## 2. Sharpest discriminators (tallied)

| Discriminator | A | B |
|---|---|---|
| Names a specific language/family | **50** | **1** (a one-token feature) |
| Chemical/corporate/business boilerplate | **28** | **0** |
| Formal/academic/professional register | **17** | **0** |
| Names a specific lexical unit (broad union) | **5** (3 real; 2 grep false-positives "keyword lists") | **72** |
| "Unclear / no coherent pattern" | **4** | **21** |
| Brief/minimal/short-response features | 3 | 16 |
| Numeric/digit features | 3 (all list-*structure*) | 10 (numeric *content*) |
| Code / list-formatting | 8 / ~18 | 5 / ~15 — **shared, non-discriminating** |

## 3. Best hypothesis for the separating criterion

**Preferred (~45%): the groups are the two extremes of how predictable a feature's activation is from the surrounding context** — e.g. variance in the feature's (response-side) activation explained by a context/prefix-level summary. A = most context-determined; B = least.

Why it beats the alternatives:

- **Fits both tails simultaneously.** Document language, topic domain, boilerplate genre, and register are properties fixed by the context and constant over many tokens — maximally predictable. Whether the exact token "hold"/"pen"/digit-1 appears, or which single character an answer contains, is nearly unpredictable from a coarse context summary — and a feature with *no coherent pattern at all* (21 in B) is by definition unpredictable. One criterion explains B's odd mix of hyper-crisp token features AND uninterpretable features.
- **The 'de' pair is diagnostic.** A#33 "Romance language preposition 'de'" vs B#54 "token 'de' or 'De'" — the *same surface token* lands on both extremes, distinguished only by whether it's context-tied. A criterion about the token itself couldn't do that; a context-dependence criterion does.
- **Chinese as a locality test.** Chinese script is trivially detectable from the current token alone, so a pure "local vs global mechanism" or "early vs late layer" criterion should scatter Chinese-text features across both groups. Instead A has 21, B has 0 — what differs is that *the answer's language is predictable from the context*, not that its detection is non-local.

Alternatives considered: **activation timescale/autocorrelation** (dense-within-document vs single-token spikes, ~20%) — nearly the same predictions, not fully separable from text alone; **feature firing frequency/density** (~10%) — doesn't explain B's brief-response block or the shared code/list categories; **layer depth** (~10%) — refuted by the Chinese argument above; **auto-interp/interpretability score** (~5%) — refuted because B contains maximally interpretable features ("token 'not'", "letter F") that no interpretability ranking would put at the bottom extreme.

**Confidence in the descriptive axis** (A = global/contextual text properties, B = local/lexical token properties + unpredictable ones): **~92%**. Confidence in the specific mechanism (predictability-from-context or an equivalent R²-style score): **~45%**.

## 4. Direction

**Group A is the "more context-determined" extreme** (activation driven by slow, whole-document properties: language, domain, register); **Group B is the "token-local / context-unpredictable" extreme**. Basis: 50 vs 1 specific-language features, 28 vs 0 boilerplate-domain features, 3 vs 72 lexical-unit features, 4 vs 21 no-pattern features.

## 5. Evidence against, and confounds

**Against the hypothesis:**
- B contains ~10 clearly *global/contextual* items a clean split shouldn't produce: "Celery and distributed task queue systems" (a topic!), "technical instructions and tutorials", "linguistic comparison or translation questions", "creative or fantasy name generation", plus 6 assistant-behavior items ("persona instruction compliance responses", "first-person speaker identity or persona expression", "task acceptance or service readiness").
- A contains a few *local* items: preposition 'de', the phrase "finds you well", "dash sequences as separators", "capitalized headers and labels".
- Code (8 vs 5) and list/formatting (~18 vs ~15) appear at similar rates in both — a sharp criterion should have pushed these one way; their symmetry suggests a noisy criterion or within-category variance.

**Confounds detectable from text alone:**
- **Templating asymmetry:** A is far more duplicated — 121/150 unique labels vs 142/150 in B; "Chinese language text" ×12 and "Russian language text" ×9 verbatim. Whatever selected A oversamples a few dense clusters.
- **Length:** A items are slightly longer (35.2 vs 30.2 chars/item; 4.43 vs 4.25 words/item) — mild, unlikely load-bearing.
- **Labeler-default confound:** these are one-line auto-generated labels, so the description *style* is itself a proxy for the activation pattern (dense foreign-language activation → "X language text"; single-token alignment → "token 'x'"). I am reading the labeler's summary of each feature, not the feature — the A/B contrast could partly reflect what the labeling pipeline defaults to for each activation shape, amplifying the apparent separation.
