# Blinded qualitative verdicts — full-width target (task #1482)

Three independent judging instances, each given ONE `blinded_*.md` file containing
two groups of 100 autointerp descriptions labelled Group A / Group B, and told
nothing about what the groups were, how they were selected, or what "better"
would mean. Each ran in a fresh context and read only its own file. Verdicts were
recorded before any key was opened.

**Unblinding (all three selections):** `Group A = top (best-predicted)`,
`Group B = bottom (worst-predicted)`.

> **Randomization caveat.** The A/B assignment was drawn independently per
> selection from a seeded RNG, and all three draws happened to land the same way
> (p = 1/8 under the intended coin flips). Each judge was blinded independently —
> fresh context, one file, no key — so no judge could exploit the coincidence, but
> the three verdicts are therefore not counterbalanced across selections.

---

## A_raw — top/bottom-100 by R², no reliability gate

**Judged verdict (verbatim):**

> **1. THREAD** — The two groups differ along a **prose-vs-notation** axis, with a
> **breadth-vs-specificity** axis riding on top of it. Group A descriptions are almost
> uniformly about *running natural-language text*: high-frequency function words,
> inflectional morphology, mid-word subword fragments, sentence punctuation, and
> formal/professional vocabulary — repeatedly framed as "across multiple languages" and
> often explicitly contrasted *against* code (A1: "excluding rare technical terms, code
> elements, or formatting tokens"; A7: "rather than technical or code-heavy text").
> Group B descriptions are dominated by *formal notation and technical content*:
> programming syntax, named language keywords, LaTeX/math, markup tags, structured data
> fields, digits and numeric literals, whitespace/indentation, and encoding artifacts.
> Group B also carries a large tail of very narrowly scoped features that has no
> counterpart in A — a single keyword, symbol, letter, or micro-domain (`package` in Go,
> `=>` in JS, `$` for currency, the pipe `|` in regex, standalone `I`/`i`/`j`, `p` in
> pH/pKa, AUTO_INCREMENT, CRISPR, hair follicles, COVID-19, quaternions) — whereas A's
> entries are consistently broad category-level descriptions.
>
> Sharp diagnostic markers: **digits/numbers** appear in ~9 Group B entries and **zero**
> Group A entries. **LaTeX** appears in ~7 B entries and zero A entries. **Named
> programming languages** (Go, Python, Java, JavaScript, C++, SQL) appear repeatedly in
> B; A's two code entries stay generic. Conversely, **Chinese-specific grammatical
> features** get 9 dedicated Group A entries and essentially none in B.
>
> **2. DIRECTION** — Group A tends to describe broad, prose-level natural-language
> features — multilingual function words, morphemes, subword continuations, discourse
> punctuation, formal/professional vocabulary in flowing text. Group B tends to describe
> technical-notation features — code syntax, math/LaTeX, markup, structured data, digits,
> whitespace, encoding artifacts — plus tightly scoped single-token or single-domain
> features.
>
> **3. CONFIDENCE** — **High.** The split is not a subtle tonal difference: entire feature
> categories (digits, LaTeX, named programming keywords, named scientific micro-domains)
> are abundant on one side and literally absent on the other, and several Group A entries
> state the exclusion of code/technical content as part of the feature definition.
>
> **4. FIT FRACTION** — Group A: ~84% fits; ~13-16% off-thread or ambiguous. Group B: ~85%
> fits; ~14-16% is generic multilingual/function-word filler that reads exactly like a
> Group A entry and could not be assigned blind. Shared stratum: "subword fragments /
> tokenizer splits across multiple languages" appears in *both* groups; that stratum alone
> would not separate the groups.

**Unblinded reading:** best-predicted = broad prose-level natural-language features;
worst-predicted = technical notation (code, LaTeX, markup, digits, whitespace) plus
narrowly-scoped single-token / single-domain features.

**Treat with caution.** This selection's bottom-100 sits at ss_tot percentile 13.2
(median) with R² from −202.8 to −0.26 — it is the low-variance tail, so the
"technical notation" thread is substantially confounded with rare firing. The gated
and matched selections below, which control variance, do **not** reproduce the
prose-vs-notation framing.

---

## B_gated — reliability-gated (ss_tot ≥ p75), top/bottom-100 — HEADLINE

**Judged verdict (verbatim):**

> **1. THREAD** — The two groups differ in **what kind of thing the description names**.
> Group A descriptions almost always characterise a feature by *grammatical class,
> structural position, or surrounding register* — "function words vs content words",
> "mid-word subword fragments", "sentence-ending punctuation", "tokens appearing in
> formal/technical/multilingual contexts". Group B contains a large subpopulation that
> instead names *the token's own identity*: either a specific semantic concept ('long',
> 'end', 'change', 'select', 'red', 'land', 'class', 'kill/death', 'receive', 'big',
> 'positive', 'true'), a literal character string ('sc/sk', 'cr/kr/gr', 'ad', 'car',
> 'cont', 'ant', 'de', 'pl', 'Em', 'il/sil', 'don/on', 'sl/l', 'man', the single letter
> 'a'/'A'), or a narrow technical domain (IP addressing, pharmaceutical chemistry, orbital
> trajectories, named web frameworks like Spring/React). This is one-sided: B has a
> signature A lacks, but A has no positive signature of its own beyond "generic". Both
> groups share a common generic backbone of multilingual-subword / punctuation /
> function-word / formatting descriptions.
>
> **2. DIRECTION** — Group A tends to describe features in broad, hard-to-name terms —
> part-of-speech class, position in the sentence, formatting role, or the register of the
> surrounding text — with essentially no descriptions keyed to a single word-meaning or
> literal substring. Group B tends, in about half its entries, to name one concrete thing
> the feature detects: a specific cross-lingual concept, a specific letter sequence, or a
> specific narrow domain. Put differently, B looks noticeably more monosemantic/nameable;
> A looks uniformly polysemantic/categorial.
>
> **3. CONFIDENCE** — **High** that a real distinction exists. The asymmetry is countable
> rather than impressionistic: ~15 pure letter-string detectors and ~35 single-concept
> lexical detectors in Group B, versus zero pure letter-string detectors in Group A and
> only a handful that are concept-specific at all — and even A's most specific items are
> function-word classes, not content meanings. A gap of that size across 100 vs 100 items
> is not plausibly sampling noise.
>
> **4. FIT FRACTION** — Group A: ~90% fits the "broad grammatical / structural / register"
> characterisation cleanly; ~10% lean specific. Group B: ~55% fits the "specific concept /
> literal string / narrow domain" thread (count: 55–58 of 100); the remaining ~45% is
> indistinguishable from Group A prose and would sit unnoticed there. **Generic filler
> overall: roughly two-thirds of the 200 items could plausibly belong to either group. The
> entire discriminative signal is carried by the ~55 specific entries in B; nothing in A
> carries signal in the other direction.**

**Unblinded reading:** best-predicted = broad, categorial, polysemantic features
(grammatical class, structural position, register). Worst-predicted = specific,
monosemantic, *nameable* features — a single cross-lingual concept, a literal
character string, or a narrow topical domain.

---

## C_matched — ss_tot-stratum-matched top/bottom-100 within the gated set

**Judged verdict (verbatim):**

> **1. THREAD** — The groups differ in what *kind of property* the description uses to
> define the feature. Group A descriptions overwhelmingly define features by **position
> and syntactic/discourse role** — where a token sits in the text (sentence/paragraph/
> section boundary, start of a list, after a colon, end of a code block, indentation,
> mid-word continuation) and what structural job it does (conjunction, article, delimiter,
> transition marker, clause introducer). Group B descriptions overwhelmingly define
> features by **token identity** — either a specific lexical meaning replicated across
> languages ("words meaning 'hold'/'change'/'red'/'car'/'end'/'business'/'get'"), a literal
> orthographic string ("tokens containing 'pl'", "'sc'/'sk'", "words starting 'Em'", "the
> digit '7'", "the letter 'd'", "'il'/'sil'", "'cr'/'kr'/'gr'", "'ant'", "'ad'"), or a
> concrete topical domain (pharmaceutical chemistry, IP/subnet notation, food-safety
> compliance, death/killing, sexual/violent content, software framework names).
>
> Two asymmetries are near-categorical: the orthographic-string feature type ("tokens
> containing letter sequence X") occurs roughly 15–20 times in Group B and **zero** times
> in Group A; and sensitive/harm-adjacent topical content occurs ~3 times in B and zero in
> A. Conversely, markdown/code scaffolding descriptions are dense in A and sparse in B.
> Where A does name a specific token it is always a function/formatting token; where B
> names one it is a content word or an arbitrary character string.
>
> **2. DIRECTION** — Group A tends to describe features by *structural position and
> grammatical function* — boundaries, transitions, delimiters, function words, document
> formatting — with any semantic content stated only as a broad register-level category.
> Group B tends to describe features by *token identity* — a specific cross-lingual word
> sense, a literal letter/digit substring, or a concrete subject-matter domain.
>
> **3. CONFIDENCE** — **High** that a real distinction exists. The orthographic-substring
> feature type (~18 instances in B, 0 in A) and the specific-word-sense type (~25 in B, 0
> in A in that form) are present-vs-absent rather than more-vs-less, which is much harder
> to produce by chance than a shift in proportions.
>
> **4. FIT FRACTION** — Group A: ~75–80% clearly fit; ~5% actively B-like; ~15–20% shared
> filler (chiefly the "short subword fragments across multiple languages" boilerplate,
> which appears in near-identical wording in both groups). Group B: ~60–65% clearly fit;
> ~30–35% is filler that would pass unremarked in Group A. So the signal is carried by
> roughly two-thirds of B against a Group A that is ~80% homogeneous, with a shared
> ~15–30% boilerplate stratum in both that carries no information either way.

**Unblinded reading:** with feature variance held fixed by construction,
best-predicted = structural position and grammatical function; worst-predicted =
token identity (specific word sense, literal orthographic substring, narrow topical
domain).

---

## Convergence and the panel comparison

**B and C converge.** With the variance confound removed (C matches ss_tot
stratum-for-stratum; top and bottom medians agree to 88.77 vs 88.76 percentile), the
axis is unchanged from B: broad/structural/positional predicts well, specific/
token-identity-bound predicts badly. A_raw's extra "prose-vs-notation" flavour does
**not** survive the gate and is attributable to the low-variance tail.

**Verdict vs the panel read: AGREES and REFINES.** The panel digest found
bottom = token-intrinsic / form-and-morphology, top = context-extrinsic / language,
register, discourse-position. The full-width target recovers exactly that axis —
"token identity" is the panel's token-intrinsic pole, "structural position /
register / discourse role" is its context-extrinsic pole — and refines the
token-intrinsic pole into a tripartite signature (named cross-lingual concept /
literal orthographic substring / narrow topical domain), of which the
orthographic-substring subtype is present-vs-absent (~15–20 vs 0) rather than a
proportion shift. No contradiction: zero features cross sides between the panel's
top-100 and any new bottom-100, or vice versa.

**Honesty standard preserved.** All three judges volunteered large shared-filler
fractions unprompted — B_gated puts roughly two-thirds of all 200 items in the
could-belong-to-either stratum and states that the best-predicted group has **no
positive signature of its own**, only the absence of the specific one. That matches
the panel read's own ~40–50% generic-filler caveat. The digest is a hypothesis
generator, not evidence.

---

## B' real-gate — MEASURED split-half reliability gate (post-hoc, supersedes the proxy)

Selections A/B/C above used a STATED PROXY reliability floor (ss_tot >= p75) because the
split-half reliability leg had not landed when they were built. It landed during this
round (`eval_results/issue_1482/r2_reliability/reliability.json`; pooled Spearman-Brown
r_full 0.8885 over 112,014 features; per-activity-decile r_full 0.458 -> 0.973).

**B' gate:** keep every activity decile whose measured r_full >= 0.80 — deciles 6-10,
activity >= 0.009567 — then top/bottom-100 by the FULL-holdout R^2 (never the per-half
values; the leg's own caveat flags those as noisier by construction). 16,632 described
features survive.

**Overlap with the proxy-gated B: 97/100 top, 52/100 bottom.** The top end is
effectively identical; half the bottom differs, which is why B' was judged rather than
assumed.

**Randomization:** B''s draw landed `Group A = bottom`, OPPOSITE the first three
selections — so B' also counterbalances the p = 1/8 same-way coincidence noted above.

**Judged verdict (verbatim):**

> **1. THREAD** — Group A descriptions are dominated by **concretely nameable triggers**:
> either a specific surface form (a letter sequence like `sc`/`sk`, `pl`, `il`/`sil`, `ant`,
> `ad`, `de`; the single characters `a`/`A`, single uppercase letters, the dash bullet,
> digits) or a specific cross-lingual lexical concept (words meaning *hold*, *select*,
> *alone*, *true*, *class*, *change*, *inside*, *receive*, *middle*, *red*, *land*,
> *death/poison*, *inhibition*, *cutting/severing*, and a large negation/absence cluster).
> The phrasing template "words and morphemes meaning 'X' across multiple languages" recurs
> ~19 times in Group A and essentially once in Group B. Group B descriptions instead
> characterise features by **grammatical role, position, and register of the surrounding
> text** — function words, articles, prepositions, particles, inflections, gender markers;
> mid-sentence / mid-word / sentence-final / clause-boundary / header-to-body positions; and
> the discourse type the text is in (formal, professional, corporate, instructional,
> "assistant-like responses", academic). Both groups share a large common substrate of
> subword-fragment, punctuation, and code-syntax descriptions — that substrate is where the
> two look identical.
>
> **2. DIRECTION** — Group A tends to name a **specific, checkable trigger** — a particular
> orthographic string, character class, or a single semantic concept identified across
> languages. Group B tends to name a **diffuse distributional category** — a broad
> grammatical class, a position in the sentence/word, or a register of surrounding prose,
> with the semantic content (where present) confined to a narrow formal/organisational band.
>
> **3. CONFIDENCE** — **High** that a real distinction exists, moderate that my
> characterisation is the intended one: two independent template counts are near-disjoint —
> pure letter-sequence/orthographic features are ~9 in A and ~0 in B, and the "meaning 'X'
> across multiple languages" concept template is ~19 in A vs ~1 in B — which is too lopsided
> to be a sampling accident, though I may be describing a downstream symptom rather than the
> underlying axis.
>
> **4. FIT FRACTION** — Group A: roughly **55–60%** clearly fits; the remaining ~40% are
> generic subword-fragment / function-word / punctuation descriptions that would be
> completely at home in Group B. Group B: roughly **85%** fits the diffuse
> positional-grammatical-register thread; about 10–15% are specific enough not to look out of
> place in Group A — though notably none of them is an orthographic letter-sequence feature.
> **Shared generic filler across both groups: ~35–40% of the 200 items.** If you handed me
> only that filler, I could not sort it. The discrimination is carried almost entirely by the
> *presence* of a class in A rather than by any distinctive class in B.

**Unblinded reading:** worst-predicted = concretely nameable triggers (orthographic
strings, single cross-lingual lexical concepts). Best-predicted = diffuse grammatical /
positional / register categories.

**This is the same axis as B and C, under the real gate and with the labels flipped.**
The proxy-gated headline stands. Note the judge's own hedge — high confidence a
distinction exists, only moderate that its characterisation is the underlying axis rather
than a downstream symptom — and its ~35–40% unsortable-filler estimate, consistent with
every other round.
