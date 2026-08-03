# Blinded qualitative verdicts — full-width target, full-dictionary descriptions (#1482)

Three independent judging instances, each given ONE `blinded_*.md` file containing two
groups of 100 autointerp descriptions labelled Group A / Group B, and told nothing about
what the groups were, how they were selected, or what "better" would mean. Each ran in a
fresh context and read only its own file. Verdicts were recorded before any key was opened.

**Descriptions:** the #1773 FULL-DICTIONARY release (5 packed shards, 126,110 rows) UNION
the 1,692-row recovery (0 collisions, recovery wins by rule) = 127,802 merged, covering
**119,858 of 121,111 scored features (98.97%)**.

**Randomization (counterbalanced this time):** `A_raw` A=bottom · `B_gated` A=top ·
`C_matched` A=bottom — 2 of 3 flipped, unlike the superseded run's uniform 3/3.

**Unblinding:**

| selection | Group A | Group B |
|---|---|---|
| A_raw | bottom (worst-predicted) | top (best-predicted) |
| B_gated | top (best-predicted) | bottom (worst-predicted) |
| C_matched | bottom (worst-predicted) | top (best-predicted) |

---

## B_gated — MEASURED split-half reliability gate — HEADLINE

Gate: activity deciles clearing Spearman-Brown r_full >= 0.80 (deciles 6-10, activity >=
0.009567; pooled r_full 0.8885 over 112,014 features). 55,610 described features survive.

**Judged verdict (verbatim):**

> **1. THREAD** — The two groups differ in *what individuates a feature*. Group A
> descriptions are almost entirely **context- and position-defined**: they say what kind of
> *text* the token sits in (formal business / industrial / chemical / pharmaceutical
> documentation, company and product descriptions, marketing prose, AI image-generation
> prompts) and where in the syntax it sits (sentence-initial article, copula before a
> definition, punctuation closing a clause or preceding a list, word-final suffix). The
> token itself is usually semantically empty — an article, comma, particle, subword
> fragment, or a piece of a proper noun. Group B descriptions are predominantly
> **token-identity-defined**: they name a cross-lingual meaning cluster carried by the token
> itself ("words meaning stop / select / low / middle / silver / goal / refuse / confirm /
> false / bad / three / one"), or a literal surface string ('emp', 'pen', 'aa', 'brown',
> '-logue', 'not'), or a code-notation class (boolean literals, variable names, operators,
> filesystem paths, numeric literals) — things that travel across domains rather than being
> pinned to a register.
>
> **2. DIRECTION** — Group A tends to describe **where a token sits** — its register,
> document domain, and syntactic position, with the token usually being
> function-word/punctuation/fragment material in formal-documentation or image-prompt text.
> Group B tends to describe **what a token is** — a specific cross-lingual concept, a
> literal substring, or a programming-syntax element, largely portable across contexts.
>
> A secondary, correlated skew: Group A's technical flavor is *business/chemical/
> manufacturing documentation plus Midjourney prompts*; Group B's technical flavor is
> *source code, math/LaTeX, config files, and tokenizer/encoding artifacts* (~20 items).
>
> **3. CONFIDENCE: high** — The template "words/morphemes meaning X across multiple
> languages" occurs ~30 times in Group B and essentially never in Group A, while "tokens
> appearing in formal business/technical documentation contexts" occurs ~25-30 times in
> Group A and rarely in B — a large, one-directional asymmetry, not a subtle tilt.
>
> **4. FIT FRACTION** — Group A: ~85% fits; ~5% clear counterexamples, ~10% generic
> multilingual-subword-fragment filler. Group B: ~70% fits; roughly 25-30% is filler
> indistinguishable from Group A. Shared filler class across both: "subword tokens /
> morphemes in non-English or multilingual technical text" — roughly 10 in A and 15 in B.
> If the thread is real, that class is where it is invisible.

**Unblinded reading:** best-predicted = **context / position / register-defined** features
(the token itself semantically empty; what matters is the document genre and syntactic
slot). Worst-predicted = **token-identity-defined** features (a cross-lingual concept, a
literal substring, or a code-notation class — portable across contexts).

---

## C_matched — ss_tot-stratum-matched within the gated set

Both groups carry identical variance composition by construction (top ss_tot percentile
median 73.85 vs bottom 73.27).

**Judged verdict (verbatim):**

> **1. THREAD** — The two groups differ in **what defines the feature and how wide its
> domain is**. Group A features are predominantly keyed to a token's own lexical/semantic
> identity — a concept, morpheme, orthographic substring, or a programming-language literal
> — and they range across essentially unbounded subject matter (days of the week,
> pirates/treasure, healthcare, tattoos, royalty, conspiracy theories, friendship, money,
> silver, red/blue/black, 'stop', 'select', 'not', 'true'), plus a large secondary band of
> code semantics and tokenizer pathology (boolean literals, empty-string literals, `>`
> operators, malformed UTF-8, mid-word subword fragments). Group B features are
> overwhelmingly keyed to a token's **syntactic/positional role inside a document** (the
> opener, the closer, the delimiter, the article that introduces a definition, the
> punctuation before a numbered list, the token after a section header), and they live in a
> strikingly narrow band of **three recurring genres**: corporate/B2B descriptive prose
> about companies and products, chemical/pharmaceutical technical writing, and
> Midjourney-style AI image-generation prompts with professional camera specs. The clearest
> single tell is genre concentration: chemistry/pharma appears in ~15 Group B items vs ~1 in
> A; Midjourney prompt syntax in ~11 B vs 1 A; camera gear in ~5 B vs 0 A; company/
> organization descriptions in ~25 B vs ~0 A.
>
> **2. DIRECTION** — Group A tends to be semantically/lexically defined and topically
> unconstrained — concepts, morphemes, substrings, colors, world topics, code literals and
> tokenization artifacts across many languages. Group B tends to be positionally/
> syntactically defined and genre-bound — structural roles within corporate/chemical-industry
> prose, formal technical documentation, and image-generation prompts.
>
> **3. CONFIDENCE: High.** Several whole content classes are present in one group and absent
> from the other (Group B has no color features, no narrative/emotional/world-knowledge
> topics, no encoding-artifact features, no programming-language-semantics features; Group A
> has essentially no chemistry, no corporate-boilerplate, and no camera-equipment features).
>
> **4. FIT FRACTION** — Group B: ~85-88% fits, tightly; ~12-15% generic filler. Group A:
> ~65-70% fits; **~30-33% is genuinely ambiguous filler**. So the thread is asymmetric in
> strength: Group B is the tightly-defined group and Group A is closer to "everything else,
> including a substantial generic remainder." A blinded item-by-item assignment would be
> reliable for the ~50 distinctively semantic A items and the ~60 distinctively
> genre-anchored B items, and **near coin-flip for the ~40 generic-structural items**.

**Unblinded reading:** worst-predicted = lexically/semantically defined and topically
unconstrained. Best-predicted = positionally/syntactically defined and genre-bound. Same
axis as B_gated, with variance held fixed — so the axis is not a variance artifact.

---

## A_raw — no gate (reported to document the trap, not as an interpretable read)

Bottom-100 sits at ss_tot percentile 2.19 (median) with R^2 from -27,932 to -139.

**Judged verdict (verbatim):**

> **1. THREAD** — Group A's descriptions are dominated by **non-linguistic / machine-symbolic
> token contexts**: source-code syntax, encoded and random alphanumeric strings (base64, hex
> hashes, API keys, URL encodings), single characters and short symbolic identifiers,
> delimiters and terminators, and structured technical notation. A second, distinctive
> Group-A cluster is **eval/task-shaped text**: multiple-choice answer letters, yes/no
> verification answers, counting puzzles and riddles, knock-knock joke templates, word-count
> constraints in instructions, alliteration-constrained and deliberately artificial/gibberish
> text. Group B's descriptions are dominated by **tokens embedded in flowing natural-language
> prose**: formal business/product/marketing/industrial descriptions, expository/definitional
> writing, multilingual natural text across many named human languages, image-generation
> prompt text, and numbered lists of environmental/weather/measurement data. A sharp lexical
> tell: Group A names *programming languages and encodings*; Group B names *human languages*
> and *commercial/scientific subject matter*.
>
> **2. DIRECTION** — Group A tends to describe features firing on semantically empty,
> symbolic, machine-format tokens — code syntax, encoded/arbitrary character strings,
> delimiters, single-character identifiers — plus structured puzzle/answer-key/
> instruction-following formats. Group B tends to describe features firing on tokens inside
> meaningful human prose.
>
> **3. CONFIDENCE: High.** The vocabularies barely overlap: base64/hex/hash/API-key language
> appears in ~12 Group A items and 0 Group B items; company/brand/product/marketing language
> appears in ~20 Group B items and essentially 0 Group A items.
>
> **4. FIT FRACTION** — Group A: ~80% clearly fits. Group B: ~85% clearly fits. **Generic
> filler that could plausibly sit in either group: ~15-20% of each.**

**Unblinded reading:** worst-predicted = machine-symbolic / encoded / code-notation tokens.
Best-predicted = tokens in flowing human prose.

**Do not read this as a finding.** A_raw's bottom is the numerically degenerate
low-variance tail. The prose-vs-notation axis it produces does NOT survive either the
reliability gate (B) or the variance match (C) — both of which drop the notation framing
entirely. Rare-firing code / encoded-string / delimiter features have tiny ss_tot, so they
concentrate in the raw bottom for estimation reasons, not because the map fails on them.

---

## Convergence, and the call on the panel read

**B and C converge**, and C holds variance fixed by construction, so the axis is not a
variance artifact: **the map predicts CONTEXT/POSITION/REGISTER/GENRE-defined features well
and TOKEN-IDENTITY-defined features badly.**

**Verdict vs the panel read: AGREES and REFINES.** The panel read had
bottom = token-intrinsic / form-and-morphology, top = context-extrinsic / language,
register, discourse-position. The corrected full-width read recovers exactly that axis —
"token identity" is the token-intrinsic pole, "context / position / register / genre" is
the context-extrinsic pole — and refines both ends:

- the token-intrinsic pole decomposes into three sub-kinds: a cross-lingual **concept**, a
  literal **orthographic substring**, and a **code/notation literal**;
- the context-extrinsic pole sharpens from "register" into explicit **genre-boundness** —
  the well-predicted features cluster in corporate/B2B prose, chemical/pharmaceutical
  writing, and Midjourney-style image prompts.

**On A_raw's prose-vs-notation axis specifically** (a genuinely different axis from the
panel's): it is **a re-description confounded by variance, not a genuine population
shift**. It appears only in the ungated arm, and both the reliability gate and the
variance match dissolve it. Treating it as a contradiction of the panel read would be
reading the estimator, not the map.

**This conclusion is robust to the description-source correction.** The superseded
panel-subset run reached the same axis from an almost entirely different feature set —
its bottom-100 shares just **1/100** with the corrected B_gated bottom. Two near-disjoint
samples of the dictionary producing the same axis is the strongest evidence in this
digest.

**Honesty.** Every judge volunteered a large shared-filler fraction unprompted: ~15-20%
per group (A_raw), ~10-30% (B_gated), and ~30-33% of the worst-predicted group plus
"near coin-flip for the ~40 generic-structural items" (C_matched). The C_matched judge
also noted the thread is **asymmetric** — the best-predicted group is tightly defined
while the worst-predicted group is closer to "everything else, including a substantial
generic remainder." Descriptions are search-index-only at 0.322 neighbour discrimination.
**This digest is a hypothesis generator, never evidence.**
