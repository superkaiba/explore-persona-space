# Before you generate a synthetic benchmark

**TL;DR**

- LLMs have made it cheap to generate training data and evaluation questions.
- That convenience makes it tempting to generate a dataset before looking for one that already exists.
- A synthetic dataset can contain plenty of examples while varying along surprisingly few dimensions. Shared wording, formatting, and instructions can become shortcuts.
- For a claim about behavior in ordinary use, I want to see evidence from ordinary-use contexts. A controlled synthetic test answers a narrower question.
- Finding suitable real data takes work, but LLMs can help with that too: search, categorize, and filter existing examples, then audit the selection.

I recently tried to reproduce a result from [*Persona Vectors: Monitoring and Controlling Character Traits in Language Models*](https://arxiv.org/abs/2507.21509) as a baseline for a paper on mapping context representations to answer representations.

I'm using this paper because it's one I spent time trying to reproduce. I like the idea, and the authors discuss limitations relevant to this post. My concern is how much we should infer from a convenient evaluation setup.

The result is cool: take the model's internal activation at the end of a prompt, project it onto a persona vector, and use the resulting score to predict how strongly the answer will express that trait. The authors present this as a way to monitor prompt-induced behavioral changes before the model generates its answer. [Section 3.3](https://arxiv.org/html/2507.21509v3#S3.SS3)

My method adds a step. Instead of reading the trait directly from the context representation, I use a learned linear map to predict the answer representation, then read the trait from that prediction.

Naturally, being an avid vibecoder, I asked Claude to implement the comparison.

When I looked at the first results, I was disappointed.

![Initial synthetic system-prompt comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/synthetic-data-blog-20260906/figures/blog/synthetic-data/01_initial_comparison.png)

*An initial comparison on Qwen2.5-7B-Instruct. Both methods predict the same judged response scores. The map was trained on a separate bank of LMSYS contexts and model-generated answers. Points show mean Pearson correlation within informative system prompts; lines show stored 95% intervals from resampling prompt conditions. Four of eight evil conditions and all eight conditions for each other trait contributed. The readouts use dot products, with and without the linear map. This attempt did not pass its baseline replication check. Collection-code recipe, not independently verified from every run manifest: 10 on-policy responses per context, temperature 1, top-p 0.95, 1,024-token cap; banked Sonnet 4.5 judgments. [Data and provenance](plot_data.json)*

The map's point estimate was lower for each trait. I'm not treating that ordering as a statistically established ranking. It was enough to make me look more closely at what the experiment was actually asking the methods to predict.

Here is the relevant part of the paper's monitoring setup:

> “we use Claude 4.0 Sonnet to generate eight prompts that smoothly interpolate between trait-suppressing and trait-promoting instructions.”

— [Persona Vectors, Section 3.3](https://arxiv.org/html/2507.21509v3#S3.SS3), excerpt from the sentence describing system-prompt construction.

That is a sensible controlled experiment. But an explicit instruction to display a trait supplies a strong signal. Predicting the behavioral difference between two such instructions is different from predicting which ordinary user request will elicit the behavior under the same system prompt.

The authors examine this distinction themselves. Their appendix reports correlations both across all conditions and within each condition. [Appendix C.2](https://arxiv.org/html/2507.21509v3#A3.SS2)

![Published pooled and within-condition correlations](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/synthetic-data-blog-20260906/figures/blog/synthetic-data/02_pooled_vs_within.png)

*Published Table 2 estimates; no intervals reported. Conditions with score SD below 1 are excluded from within-condition averages. Targets: 10 on-policy responses per condition/question, GPT-4.1-mini judgments; sampling parameters not independently checked. [Original table](https://arxiv.org/html/2507.21509v3#A3.SS2)*

The drop is especially large for hallucination under system prompting. Other results remain substantial, particularly in the many-shot setting. The authors explicitly caution that their monitor may be less dependable for subtle deployment-time changes. [Section 3.3](https://arxiv.org/html/2507.21509v3#S3.SS3)

So I don't think this establishes that persona vectors merely detect superficial wording. Trait-promoting instructions can genuinely cause trait-promoting behavior. The concern is the gap between the behavior the evaluation makes easy to detect and the behavior we eventually want to detect.

The initial comparison above already uses within-prompt correlations, so pooling cannot explain that result by itself. There are also reproduction differences to resolve. I would treat it as the reason I investigated, rather than evidence that the paper's result is invalid.

This is the broader reason I worry about reaching for synthetic data by default. Training and evaluation can both inherit unintended regularities. During training, a model can learn the wrong cue. During evaluation, a method can succeed by exploiting a cue that will be absent when we use it elsewhere.

[*Chunky Post-Training*](https://arxiv.org/abs/2602.05910) studies how separately curated training datasets encode unintended associations between surface features and behaviors. Samy Mammeri and colleagues' [*When Does Interleaving Prevent Emergent Misalignment?*](https://openreview.net/forum?id=DzpyGJQ6dt) finds that changing evaluation-prompt length can substantially change measured misalignment and complicate comparisons between mitigations. These are related failures of generalization; neither requires every example in a dataset to be synthetic.

My preferred starting point is to look for existing contexts that express the variation I care about. For a claim about deployed assistants, that might mean real conversations. For factual accuracy, it might mean an established question-answering dataset. The important part is matching the data and the measurement to the claim.

That search can be tedious. But the same LLM that could generate a thousand new questions can help search a corpus, classify existing questions, and flag candidates for review. I would still inspect the selected examples, check what the filter excludes, and test on an independently collected source. Automatic categorization can introduce its own biases.

Synthetic data remains useful for controlled interventions and for cases where suitable observations are hard to obtain. I just want the decision to generate it to come after the search for relevant existing data, and the strength of the conclusion to match the test.

---

**Optional continuation: what happened on other datasets?**

In a later experiment, I compared direct Persona Vector projection with projection after a linear context-to-answer map across several evaluation datasets.

![Later comparison across evaluation datasets](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/synthetic-data-blog-20260906/figures/blog/synthetic-data/03_followup_datasets.png)

*A separate Qwen2.5-7B-Instruct experiment, using a map trained on generic WildChat plus behavior-eliciting training pairs. All three methods share each row's contexts and response-score targets. This includes every non-training dataset in the selected R2FAIR result file: 11 datasets-by-trait cells, with three readouts each. Lines are stored 95% intervals from resampling evaluation contexts at fixed fitted models and layers; they do not quantify training-seed variation or uncertainty over shared prompt templates. The actual-answer readout uses generated text and is a reference, not a guaranteed upper bound. Source-code recipe: five on-policy responses per context, temperature 1, 1,024-token cap, banked Sonnet 4.5 judgments; top-p is not pinned by the collection code. These settings have not been independently verified against every completion manifest. [Data and full qualifications](plot_data.json)*

Direct context projection is much weaker on several of these datasets than on the synthetic prompts. Mapping helps in several cases, but it is not a universal fix. For example, on NQ-Open its correlation is slightly negative, and both methods correlate weakly with sycophancy scores in WildChat.

There are important qualifications. The NQ-Open and SimpleQA rows measure fabrication rate, whereas the synthetic and WildChat hallucination rows use graded trait scores. The held-out Reddit set comes from a source also used in training. Harmful behavior is rare in the WildChat sample, limiting the variation available to predict. Finally, this is a later mapping recipe and a different synthetic suite, so it should not be read as a direct rerun of the opening comparison.

This motivates evaluation on several relevant distributions. It does not isolate synthetic wording as the cause of any performance gap.
