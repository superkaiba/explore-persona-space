# Issue #1482 — SAE-feature and context analysis

Working writeup section. Plots and dashboards are filled in; **takeaways are left
blank deliberately** — they are Thomas's to write.

Every dashboard and figure linked here is data-only: no interpretation is drawn
in a title, caption or sidecar. The one exception is the context-extremes
dashboard, which carries the three blinded Claude Fable 5 reads because the
section text calls for them.

---

#### Qualitative analysis

I then looked (and asked Claude Fable 5) to look at the best and worst 100
predicted features and figure out the common threads within each group and the
differences (without telling it anything about the setting).

Dashboard: https://eps.superkaiba.com/sae-features-1482.html

Claude Fable 5 said this:

> **Setup.** Fully unprimed: it was told only that two groups of 150 descriptions
> were drawn from one pool by an unknown criterion, and told not to open the key
> or search the repo. It was *not* told a map, a prediction target, an R², or SAE
> features were involved. Group assignment was randomized and the key frozen
> before the read. **Verdict: correct** — it picked A, and A is the best-predicted
> group. Full verbatim report:
> [`eval_results/issue_1482/feature_extremes/fable_read_extremes.md`](https://github.com/superkaiba/explore-persona-space/blob/main/eval_results/issue_1482/feature_extremes/fable_read_extremes.md)

**Group A — global properties of whole texts.** Language/script identity 59/150
(50 naming a specific language: Chinese 21, Russian/Cyrillic 14, Romance 12,
German 3); chemical-industry / corporate boilerplate 28; formal/academic register
17; genre and domain labels ~16. Only 4 items are "unclear / no pattern", and only
3 are keyed to a specific lexical unit. Representative: *"Chinese language
mathematical explanations"*, *"chemical company corporate profiles"*, *"formal
expository informational writing style"*, *"creative fiction and roleplay
narratives"*.

**Group B — local, token-level properties.** 72/150 name a specific lexical unit
(token / word / substring / letter / digit / affix), ~52 naming exactly one;
21 are "no coherent pattern"; 16 are brief/minimal-response features; 10 are
numeric content. Only 1 item names a specific language, and that one is itself a
single token. Zero chemical, zero corporate, zero formal-register items.
Representative: *"token 'not'"*, *"letter F"*, *"substring 'hin' in tokens"*,
*"no coherent shared property detected"*.

| discriminator | A | B |
|---|---|---|
| names a specific language / family | **50** | **1** (a one-token feature) |
| chemical / corporate / business boilerplate | **28** | **0** |
| formal / academic / professional register | **17** | **0** |
| names a specific lexical unit | 3 real | **72** |
| "unclear / no coherent pattern" | **4** | **21** |
| brief / minimal-response | 3 | 16 |
| code · list-formatting | 8 · ~18 | 5 · ~15 — **shared, non-discriminating** |

**Its hypothesis**, unprompted: *"the two extremes of how predictable a feature's
activation is from the surrounding context — A = most context-determined."*
Confidence ~92% on the descriptive axis, ~45% on that specific mechanism. Three
arguments it gave:

- Document language, topic domain and register are fixed by the context and
  constant over many tokens, so maximally predictable; whether an exact token
  appears is not — and a feature with *no coherent pattern at all* (21 in B) is
  unpredictable by definition. One criterion explains B's odd mix of hyper-crisp
  token features *and* uninterpretable ones.
- **The 'de' pair is diagnostic.** A#33 *"Romance language preposition 'de'"* vs
  B#54 *"token 'de' or 'De'"* — the same surface token at both extremes,
  separated only by whether it is context-tied.
- **Chinese as a locality test.** Chinese script is detectable from the current
  token alone, so a "local vs global mechanism" or "layer depth" criterion should
  scatter it across both groups. A has 21, B has 0.

It ruled out activation timescale (~20%), firing frequency (~10%), layer depth
(~10%) and interpretability score (~5%, refuted because B contains maximally
interpretable features like *"token 'not'"* that no interpretability ranking would
put at the bottom).

**Confounds it flagged itself**, which should temper the above: group A is more
duplicated (121/150 unique labels vs 142/150 in B; *"Chinese language text"* ×12
verbatim), so whatever selected A oversamples a few dense clusters; and — the
important one — these are one-line auto-generated labels, so description *style*
is itself a proxy for activation shape (dense foreign-language activation →
"X language text"; single-token alignment → "token 'x'"). It is reading the
labeller's summary of each feature, not the feature, which may amplify the
apparent separation. It also noted ~10 clearly global items sitting in B
(including 6 assistant-behavior items) that a clean split should not produce.

**Takeaways:**
-

---

#### Analysis of difference between prefix end vs query only vs full context

I then wanted to look at which features were better/worse predicted with only the
query or with only the prefix, so I trained the same mapping from bare query ->
answer and from prefix end -> answer and plotted the largest drops in predictive
power among all the predictors for both:

![arm drop by predictor](https://raw.githubusercontent.com/superkaiba/explore-persona-space/65796d037253fabeaad23050e7dcd9fb99e3c612/figures/issue_1482/concordance/writeup_arm_drop_by_predictor.png)

https://github.com/superkaiba/explore-persona-space/blob/65796d037253fabeaad23050e7dcd9fb99e3c612/figures/issue_1482/concordance/writeup_arm_drop_by_predictor.png

> Per predictor, the concordance of the per-feature R² **drop** on that predictor,
> where drop = R²(full context) − R²(arm). Filled = prefix-only, hollow =
> query-only. c > 0.5 means features carrying the property lose *more* when the
> input is cut back. n = 113,288 features; mean drop −0.142 (prefix), −0.035
> (bare). Arms are the banked #1738 multi-turn fits
> (`eval_results/issue_1738/sae_twoway/perfeature/sae_{context,prefix,bare}_r2.npy`).

**Takeaways:**
-

**Final takeaways:**
- This SAE feature analysis provides pretty good evidence that the best predicted
  features of the answer are:
    - not exclusively present in the answer
    - high-level
    - persona/behavior/identity-related
- This provides evidence for the hypothesis that what this mapping is capturing is
  the high-level part of the output while discarding the low-level details
- [add more]

---

#### Analysis of context-only vs answer-only features

As an exploratory analysis, I dig more into the "context-specific" or
"answer-specific" SAE features

I therefore plotted the distribution of SAE features according to their
**answer-side** activation ratio (fraction of activations that appear in answer
vs context):

![side ratio distribution](https://raw.githubusercontent.com/superkaiba/explore-persona-space/65796d037253fabeaad23050e7dcd9fb99e3c612/figures/issue_1482/side_specificity/side_ratio_token.png)

I then also looked at the context-only and answer-only features qualitatively
(and asked Fable 5 to categorize them without knowing the difference between
them):

- Context-only features: https://eps.superkaiba.com/context-only-1482.html
- Answer-only features: https://eps.superkaiba.com/answer-only-1482.html

Claude Fable 5 said this:

> **Setup.** Minimal by design — it was told only that two files held 100 items
> each, split out of one pool by some criterion, and to work solely from those
> files. It was *not* told these were features, that a labelling model produced
> them, that they came from activating examples, what the axis was, or that any
> confound existed. Packet headers carry no criterion name and no item type.
> **Verdict: correct** — it called B the input/context-side group, and B is
> `context_only`. Full verbatim report:
> [`fable_read_side.md`](https://github.com/superkaiba/explore-persona-space/blob/main/eval_results/issue_1482/side_specific/fable_read_side.md)

- **It worked out what the items were unaided** — "auto-interpretability
  descriptions of SAE-style features", inferred from content alone.
- **Both groups look alike on the surface**: code/markup syntax, punctuation and
  delimiters, legal/licence boilerplate, multilingual text. So the criterion is
  *not* a topic split like code-vs-prose or English-vs-other.
- **AI-directed prompt / user-message features: 13 in B vs 0 in A** (±2 on
  borderline items). A perfect zero on one side, and the cleanest signal it found.
  Verbatim from B: *"The pronoun 'You' appearing in jailbreak prompts that
  instruct an AI to violate its safety guidelines…"*; *"The word 'only' … in the
  meta-instruction phrase 'only send the completion based on the system
  instructions'…"*; *"The token 'over' when specifying a word count requirement in
  the templated phrase 'Give me an introduction over 200 words for [company]…'"*.
- **Digit / number-token features: ~14 in B vs ~3 in A** (est.) — digits in URLs,
  IDs, ZIP codes, hex hashes, version and section numbers.
- **Pasted-document flavour, B only**: Google `&ved=` search params, copyright
  headers, news datelines, Creative Commons URL paths, an address fragment
  ("UNIT" in "UNITED STATES"), corrupted Unicode, medical vital-signs — material a
  user pastes *into* a chat.
- **A skews to text-in-flow and structural glue**: tokenization-split word
  fragments (~11 A vs ~7 B, est.), line breaks and markdown/list structure, ASCII
  boundaries, poetic rhyme endings, and response-flavoured discourse. Verbatim
  from A: *"Line breaks, newlines, or whitespace sequences that mark the end of a
  line…"*; *"'your' … when responding to or acknowledging someone's question"*.
- **Description style differs**: B pins a specific literal token in a specific
  template far more often (~45 vs ~25 items, est.); A's are more abstract and
  structural. German features appear only in B (5); Korean only in A (2).
- **Its inferred criterion**: the features were split by *where they activate
  relative to the chat input/output boundary* — B on the input/context side, A on
  the answer side. Its argument: "the perfect 13-vs-0 split on prompt-instruction
  features and the 14-vs-3 digit skew are exactly what a positional split would
  produce; a content-based split would not leave the within-group heterogeneity
  both groups show."
- **Confidence: ~90%** that B is anchored in user-prompt / pasted-input material
  and A is not (directly measurable: 13–0, ~14–3); **~60%** on the specific
  mechanism being activation position rather than a near-equivalent alternative.
- **It raised the evidence-side confound itself**, unprompted, as its own
  alternative (a): "split by the corpus the top-activating examples came from
  (chat/user-query data vs. pretraining-style documents) — this is nearly
  observationally equivalent to the positional story." Also floated (b) a judged
  context-category split and (c) an experiment-arm split with only statistical
  content differences.

> **Why that last bullet matters.** The two description sets genuinely are *not*
> same-instrument — answer-only descriptions come from #1773's **answer-side**
> activating windows, context-only from a dedicated **context-side** pass
> (`eval_results/issue_1482/context_side_labels/`, 1,653 of 1,654 described, 0
> drops). The split is forced: a context-only feature has no answer-side windows.
> An earlier PRIMED read had this confound handed to it in its brief; this clean
> read found it independently, which is the stronger result. Do not pool the two
> description sets.
>
> The primed read of the same split is archived at
> [`fable_read_side_primed.md`](https://github.com/superkaiba/explore-persona-space/blob/main/eval_results/issue_1482/side_specific/fable_read_side_primed.md).
> Same verdict, lower confidence (~75–80% / ~45% vs ~90% / ~60%). Removing the
> priming *raised* confidence, which argues the signal is in the descriptions
> rather than in the brief.

**Takeaways:**
- Around 1-2% of all features are context-only and answer-only
- Qualitatively, these context and answer features look like...

---

### What contexts is the mapping bad at predicting?

I then turned to the context side, to see if there are specific contexts the
mapping is bad at predicting. To do this, I first looked at the top 100 and worst
100 predicted contexts (for prefix only -> answer, query only -> answer, full
context -> answer, where the answer is always the same one generated under the
full context) among the held-out set and tried to see the differences between
them, both manually and with Claude Fable 5 (see
[dashboard](https://eps.superkaiba.com/context-extremes-1482.html))

**Takeaways** (summarized from Fable's analysis and me looking at the features):
-

I then categorized all the contexts using a LLM judge into different categories:
- Prompt used can be found [here](https://github.com/superkaiba/explore-persona-space/blob/65796d037253fabeaad23050e7dcd9fb99e3c612/scripts/issue1482_analysis.py#L64)

I plotted the average $R^2$ per context type as a bar plot for **prefix only,
query only, and context mappings**

![context category by arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/65796d037253fabeaad23050e7dcd9fb99e3c612/figures/issue_1482/context_category_3panel.png)

https://github.com/superkaiba/explore-persona-space/blob/65796d037253fabeaad23050e7dcd9fb99e3c612/figures/issue_1482/context_category_3panel.png

> Mean per-context R² by judged context category, one panel per mapping arm.
> #1738 multi-turn holdout (n = 9,941), L19 ridge. Each panel is sorted
> best-predicted first **by its own arm**, so the x-order differs between panels
> — colour, not position, tracks a category across panels. Shared y-limit; 95%
> paired bootstrap CI (B = 2,000, shared draws across arms). Arm means: context
> 0.661, prefix 0.387, query 0.495. A truncation-controlled twin
> (`context_category_3panel_notrunc.png`) excludes the 9.85% of responses that hit
> the 1024-token generation cap; no category mean moves by more than 0.014.

**Takeaways:**
-
