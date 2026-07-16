# Result: A single greedy answer is an adequate stand-in for sampled/averaged answer targets in the context → answer map

*(Thomas's own result writeup, saved verbatim at promotion, 2026-07-15. The critic-gated clean-result is `body.md`; this is the user-authored summary of record.)*

## Motivation

* We've been running our experiments with deterministic decoding
* We want to see if things change for:
    * stochastic decoding (averaged over samples)
    * stochastic decoding (single sample)
* This experiment tests this
    * so that we can know what to do for future experiments

## TLDR

* 10 rollout average fits best. Greedy and single-stochastic are a tie
    * 10 rollout average = R^2 ≈ 0.67-0.77
    * greedy/single stochastic = R^2 ≈ 0.6
    * deterministic vs single stochastic barely matters - averaging is what gives an improvement (noise reduction on target)

## Methodology

* Same methodology as "There is a linear mapping between single context vector and answer summaries" (#779)
* Varied generation:
    * deterministic
    * stochastic single sample (temperature=1)
    * stochastic 10 rollout average (temperature = 1)
* [Examples](https://htmlpreview.github.io/?https://raw.githubusercontent.com/superkaiba/explore-persona-space/4f383032894c969f42cf9ae4c2b3200662e0d7e6/experiments/dashboards/issue1073_decode_regime_examples.html)

## Results

### Result 1: The 10 rollout average fits the best. Greedy and single-stochastic tie

I first wanted to see which decoding regime gave the best map quality

**Methodology**

* Fit mapping for each decoding regime (ridge regression, kernel regression, MLP)
* Compute:
    * quality of mapping
    * ability to read-out behavior based on the mapping

**Takeaways**

* Averaged targets fit best at every layer and for all predictors

### Result 2: A greedy sample is just as similar to the rollout average as a single stochastic rollout

I then wanted to see if a greedy sample gives a normal rollout (similar to single stochastic rollout) or if it gives a weird OOD rollout.

**Methodology**

* Compare greedy rollout answer vector vs single stochastic rollout answer vector to averaged 10 rollout answer vector (with single stochastic rollout removed)
* Use cosine similarity
* Here we plot the greedy minus stochastic cosine similarity

**Takeaways:**

* On average the greedy rollout gives as good a sample as a single stochastic sample
* But this is only on average -- there are 27-32% of contexts with a significant difference between greedy and single stochastic

### Result 3: No simple distribution explains the shape of the stochastic samples around the greedy rollout

I tried to fit various distributions to the stochastic samples around the greedy rollout but couldn't find any distribution that fit well

## Conclusion/next steps

* We can use a greedy rollout as a substitute for 10 averaged stochastic rollouts
