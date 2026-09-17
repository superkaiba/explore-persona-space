## Behavior directions transfer before generation on some datasets

Can the context-answer metamodel transfer a fixed behavior direction without fitting a behavior-specific regression? Following Chen et al. (2025), we extract answer and context directions from positive-minus-negative instruction contrasts for harmful compliance (including malicious persona and style), sycophancy, and hallucination. We use a frozen layer-19 metamodel trained on 963,444 generic context–answer pairs and evaluate Spearman correlation with judged behavior on five held-out datasets. Scores average retained judged responses, which can exclude refusals (Appendix H).

**Fixed answer directions predict behavior on some datasets** (Figure 1A)**:** Projecting predicted answer vectors onto the answer direction gives $\rho=0.426$ on ToxicChat, 0.255 on AITA, and 0.449 on SimpleQA, without behavior-specific regression. The observed-answer directions also validate on these datasets. Prediction is weak on HH-RLHF and reverses on NQ-Open, where the observed-answer direction also fails validation ($\rho=-0.009$).

**Preimages do not consistently outperform context directions** (Figure 1B)**:** A regularized inverse finds a context direction that the metamodel's linear component maps approximately onto the fixed answer direction. Preimage scores are close to context-native scores on ToxicChat (0.410 versus 0.403) and AITA (0.322 versus 0.319), but lower on SimpleQA (0.435 versus 0.507). Prediction remains weak on HH-RLHF ($\rho=0.051$) and reverses on NQ-Open ($-0.171$).

**Preimage retrieval reveals recurring prompt types** (Appendix H, Table 17)**:** The nearest generic contexts include malicious-persona requests, benign interpersonal-response prompts, and detailed chemical-company introductions. All 30 highest-ranked hallucination contexts share the company-introduction template. These unjudged training-pool examples show semantic associations, but do not establish that the prompts induce the intended behavior.

**Regression on predicted answers does not consistently improve on context regression** (Figure 1C)**:** With matched training examples and regularization selection, no positive gain over context regression has a 95% interval excluding zero, and AITA and SimpleQA favor context regression. Covariance whitening of the same generic contexts beats predicted-answer regression on SimpleQA and loses on AITA (paired comparisons in Figure 2).


![Figure 1. Fixed-direction transfer, preimage directions, and matched regression readouts.](https://eps.superkaiba.com/tasks/1739/figure/c5_behavior_transfer.png)

![Figure 2. Paired differences for preimage and regression comparisons.](https://eps.superkaiba.com/tasks/1739/figure/c5_behavior_controls.png)

*Intervals: pointwise 95% paired group bootstrap, 2,000 draws, conditional on fitted models. Answer-token pooling remains mismatched. WildChat retains only four contexts per behavior and is excluded from substantive comparisons. Full protocol and highest-ranked prompt excerpts appear in [the manuscript](https://www.overleaf.com/project/6a59c927290f8b8b5eee0055), Appendix H.*
