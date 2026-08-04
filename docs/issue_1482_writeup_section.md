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

<!-- FABLE_EXTREMES_READ -->

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

> **The blinded Fable categorization was NOT run for this pair, and cannot be run
> from banked data.** 0 of the 1,654 context-only features carry a description:
> #1773 built its activating-example windows from **answer-side** activation and
> dropped features with zero windows before dispatch, and context-only features
> have zero by construction. The artifact records this itself, under
> `no_evidence_exclusion`: *"the one population that would most directly test
> whether context-specific KINDS exist is exactly the population the
> interpretability labels cannot speak to."* The `ctx_tokens_active_subsample` /
> `ans_tokens_active_subsample` fields are activation **counts**, not token text,
> so there is no fallback evidence either. Running this read requires a fresh
> **context-side** activating-window extraction first — new compute, not a re-run.
> Until then the two dashboards above show the populations but no blinded
> comparison, and the answer-only side is the only one with descriptions at all
> (2,132 / 2,164 = 98.5%).

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
[dashboard for features and Claude Fable analysis](https://eps.superkaiba.com/context-extremes-1482.html))

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
