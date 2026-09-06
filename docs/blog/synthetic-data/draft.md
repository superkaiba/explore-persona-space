# Before you generate a synthetic benchmark

**TL;DR**

- LLMs have made it cheap to generate training data and evaluation questions.
- That convenience makes it tempting to generate a dataset before looking for one that already exists.
- A synthetic dataset can contain plenty of examples while varying along surprisingly few dimensions. Shared wording, formatting, and instructions can become shortcuts.
- For a claim about behavior in ordinary use, I want to see evidence from ordinary-use contexts. A controlled synthetic test answers a narrower question.
- Finding suitable real data takes work, but LLMs can help with that too: search, categorize, and filter existing examples, then audit the selection.

I recently tried to reproduce a result from [*Persona Vectors: Monitoring and Controlling Character Traits in Language Models*](https://arxiv.org/abs/2507.21509) as a baseline for one of my papers.

I'm using this paper because it's one I spent time trying to reproduce. I like the idea, and the authors discuss limitations relevant to this post. My concern is how much we should infer from a convenient evaluation setup.

The result is cool: take the model's internal activation at the end of a prompt, project it onto a persona vector, and use the resulting score to predict how strongly the answer will express that trait. The authors present this as a way to monitor prompt-induced behavioral changes before the model generates its answer. [Section 3.3](https://arxiv.org/html/2507.21509v3#S3.SS3)

I wanted to see how a method I'd been working on compared on the same test.

Naturally, being an avid vibecoder, I asked Claude to implement the comparison.

The synthetic-data results were disappointing, especially for sycophancy.

![Synthetic-data Spearman comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6a9a3eee721ddee7ce73317e71dce2063ed6cc19/figures/blog/synthetic-data/01_initial_comparison.png)

*Qwen2.5-7B-Instruct, with both methods predicting the same judged response scores. Bars show Spearman correlation across 200 synthetic contexts per trait; error bars are stored 95% intervals from resampling contexts at fixed models and layers. These are the synthetic rows from the same experiment shown below. My method used generic WildChat plus behavior-eliciting training pairs. The synthetic suite crosses five positive/negative instruction pairs with 20 held-out questions; it differs from the paper's eight-prompt monitoring experiment. Collection-code recipe, not independently reverified from every manifest: five on-policy responses per context, temperature 1, 1,024-token cap, top-p unspecified in code; banked Sonnet 4.5 judgments. [Data and provenance](plot_data.json)*

Persona Vectors performed well on this synthetic suite. My method's point estimates were a little higher for evil and hallucination, but much lower for sycophancy. That made me look more closely at what the evaluation was asking the methods to predict.

Here is the relevant part of the paper's monitoring setup:

> “we use Claude 4.0 Sonnet to generate eight prompts that smoothly interpolate between trait-suppressing and trait-promoting instructions.”

— [Persona Vectors, Section 3.3](https://arxiv.org/html/2507.21509v3#S3.SS3), excerpt from the sentence describing system-prompt construction.

That is a sensible controlled experiment. But an explicit instruction to display a trait supplies a strong signal. Predicting the behavioral difference between two such instructions is different from predicting which ordinary user request will elicit the behavior under the same system prompt.

You can see this in the distribution of behavior scores in our synthetic suite:

![Behavior expression under promoting and suppressing synthetic instructions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6a9a3eee721ddee7ce73317e71dce2063ed6cc19/figures/blog/synthetic-data/04_synthetic_expression.png)

*Each bar shows the percentage of contexts in a 10-point score bin, separately normalized within promoting and suppressing instructions. Each group contains 100 contexts per trait. Scores average the retained judged responses to each context; these are the same targets as the synthetic correlation comparison above. [Scores, counts, and provenance](distribution_data.json)*

All 100 evil-suppressing contexts have a mean score of zero; 28 of the 100 evil-promoting contexts also score zero. Sycophancy and hallucination scores also shift with instruction polarity. A high correlation across these groups can therefore reflect separation between instructions, while telling us less about variation under a fixed instruction.

The authors examine this distinction themselves. Their appendix reports Pearson correlations both across all conditions and within each condition. These are the paper's statistics, distinct from the Spearman correlations in our comparison. [Appendix C.2](https://arxiv.org/html/2507.21509v3#A3.SS2)

![Published pooled and within-condition correlations](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6a9a3eee721ddee7ce73317e71dce2063ed6cc19/figures/blog/synthetic-data/02_pooled_vs_within.png)

*Bars show published Table 2 estimates; no intervals reported. Conditions with score SD below 1 are excluded from within-condition averages. Targets: 10 on-policy responses per condition/question, GPT-4.1-mini judgments; sampling parameters not independently checked. [Original table](https://arxiv.org/html/2507.21509v3#A3.SS2)*

The drop is especially large for hallucination under system prompting. Other results remain substantial, particularly in the many-shot setting. The authors explicitly caution that their monitor may be less dependable for subtle deployment-time changes. [Section 3.3](https://arxiv.org/html/2507.21509v3#S3.SS3)

So I don't think this establishes that persona vectors merely detect superficial wording. Trait-promoting instructions can genuinely cause trait-promoting behavior. The concern is the gap between the behavior the evaluation makes easy to detect and the behavior we eventually want to detect.

Our synthetic-suite comparison pools contexts across trait instructions. It therefore does not isolate prediction under a fixed instruction. The paper's within-condition analysis asks that narrower question, though it uses a different prompting setup and correlation statistic.

This is the broader reason I worry about reaching for synthetic data by default. Training and evaluation can both inherit unintended regularities. During training, a model can learn the wrong cue. During evaluation, a method can succeed by exploiting a cue that will be absent when we use it elsewhere.

[*Chunky Post-Training*](https://arxiv.org/abs/2602.05910) studies how separately curated training datasets encode unintended associations between surface features and behaviors. Samy Mammeri and colleagues' [*When Does Interleaving Prevent Emergent Misalignment?*](https://openreview.net/forum?id=DzpyGJQ6dt) finds that changing evaluation-prompt length can substantially change measured misalignment and complicate comparisons between mitigations. These are related failures of generalization; neither requires every example in a dataset to be synthetic.

My preferred starting point is to look for existing contexts that express the variation I care about. For a claim about deployed assistants, that might mean real conversations. For factual accuracy, it might mean an established question-answering dataset. The important part is matching the data and the measurement to the claim.

That search can be tedious. But the same LLM that could generate a thousand new questions can help search a corpus, classify existing questions, and flag candidates for review. I would still inspect the selected examples, check what the filter excludes, and test on an independently collected source. Automatic categorization can introduce its own biases.

Synthetic data remains useful for controlled interventions and for cases where suitable observations are hard to obtain. I just want the decision to generate it to come after the search for relevant existing data, and the strength of the conclusion to match the test.

---

**Optional continuation: what happened on other datasets?**

I also looked at the other evaluation datasets in the same experiment.

First, here are their behavior-expression distributions:

![Behavior-expression distributions across all evaluation datasets in the comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6a9a3eee721ddee7ce73317e71dce2063ed6cc19/figures/blog/synthetic-data/05_expression_across_datasets.png)

*Histograms show the percentage of contexts in each 10-point bin, with identical axis ranges. All 11 dataset-by-trait cells match the correlation comparison below, including its held-out WildChat split. NQ-Open and SimpleQA measure fabrication rate, rescaled to 0–100; the other panels measure graded trait scores. Missing scores are excluded, and each panel reports its scored sample size. [Scores, counts, and provenance](distribution_data.json)*

Existing datasets do not automatically give us a useful evaluation. Evil expression is close to zero for most contexts in these human red-team, ToxicChat, and WildChat samples. That leaves little behavioral variation to predict. The sycophancy and hallucination distributions vary by dataset, and the hallucination panels also use different scoring rules.

With that context, here is prediction performance:

![Spearman comparison across evaluation datasets](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6a9a3eee721ddee7ce73317e71dce2063ed6cc19/figures/blog/synthetic-data/03_followup_datasets.png)

*The same Qwen2.5-7B-Instruct experiment, with my method trained on generic WildChat plus behavior-eliciting training pairs. Both methods share each row's contexts and response-score targets. This includes every non-training dataset in the selected R2FAIR result file: 11 datasets-by-trait cells, with two methods each. Error bars are stored 95% intervals from resampling evaluation contexts at fixed fitted models and layers; they do not quantify training-seed variation or uncertainty over shared prompt templates. Source-code recipe: five on-policy responses per context, temperature 1, 1,024-token cap, banked Sonnet 4.5 judgments; top-p is not pinned by the collection code. These settings have not been independently verified against every completion manifest. [Data and full qualifications](plot_data.json)*

Direct context projection is much weaker on several of these datasets than on the synthetic prompts. My method does better in several cases, but it is not a universal fix. For example, on NQ-Open its correlation is slightly negative, and both methods correlate weakly with sycophancy scores in WildChat.

There are important qualifications. The NQ-Open and SimpleQA rows measure fabrication rate, whereas the synthetic and WildChat hallucination rows use graded trait scores. The held-out Reddit set comes from a source also used in training. Harmful behavior is rare in the WildChat sample, limiting the variation available to predict. The synthetic rows repeat the opening comparison, using the same fitted methods, targets, and Spearman metric.

This motivates evaluation on several relevant distributions. It does not isolate synthetic wording as the cause of any performance gap.
